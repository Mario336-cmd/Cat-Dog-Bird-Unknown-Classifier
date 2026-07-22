import * as tf from '@tensorflow/tfjs'
import { CLASS_NAMES, type ModelStatus, type PredictionResult } from '../types'

const INPUT_SIZE = 224
const FEATURE_SHAPE: [number, number, number, number] = [1, 7, 7, 1280]
const BACKEND_ORDER = ['webgpu', 'webgl', 'wasm', 'cpu'] as const
const MODEL_URL = new URL(`${import.meta.env.BASE_URL}model/model.json`, window.location.href).href
const GRADCAM_URL = new URL(`${import.meta.env.BASE_URL}model/gradcam_weights.json`, window.location.href).href
const WASM_DIRECTORY_URL = new URL(`${import.meta.env.BASE_URL}wasm/`, window.location.href).href
const WASM_SCRIPT_URL = new URL(`${import.meta.env.BASE_URL}wasm/tf-backend-wasm.js`, window.location.href).href

type BackendName = (typeof BACKEND_ORDER)[number]
type ModelOutputs = tf.Tensor | tf.Tensor[] | tf.NamedTensorMap
type StatusListener = (status: ModelStatus) => void

interface PreparedModel {
  model: tf.GraphModel
  backend: BackendName
  modelLoadMs: number
}

interface GradcamPayload {
  kernel_shape: [number, number]
  kernel: number[]
  score: string
}

let preparedModelPromise: Promise<PreparedModel> | null = null
let wasmScriptPromise: Promise<void> | null = null
let gradcamPayloadPromise: Promise<GradcamPayload> | null = null

const backendLabel = (backend: BackendName): string => ({
  webgpu: 'WebGPU',
  webgl: 'WebGL',
  wasm: 'WASM',
  cpu: 'CPU',
})[backend]

const emit = (listener: StatusListener | undefined, status: ModelStatus): void => listener?.(status)

const loadScript = (src: string): Promise<void> => new Promise((resolve, reject) => {
  const existing = document.querySelector<HTMLScriptElement>(`script[data-classifier-src="${src}"]`)
  if (existing?.dataset.loaded === 'true') {
    resolve()
    return
  }
  if (existing) {
    existing.addEventListener('load', () => resolve(), { once: true })
    existing.addEventListener('error', () => reject(new Error('WASM backend script failed to load.')), { once: true })
    return
  }
  const script = document.createElement('script')
  script.src = src
  script.async = true
  script.dataset.classifierSrc = src
  script.addEventListener('load', () => {
    script.dataset.loaded = 'true'
    resolve()
  }, { once: true })
  script.addEventListener('error', () => reject(new Error('WASM backend script failed to load.')), { once: true })
  document.head.append(script)
})

const registerPatchedWasmBackend = async (): Promise<void> => {
  if (!wasmScriptPromise) {
    wasmScriptPromise = (async () => {
      const tfGlobal = { ...tf } as Record<string, unknown> & {
        wasm?: { setWasmPaths: (path: string) => void }
      }
      ;(globalThis as typeof globalThis & { tf?: typeof tfGlobal }).tf = tfGlobal
      await loadScript(WASM_SCRIPT_URL)
      if (!tfGlobal.wasm?.setWasmPaths) throw new Error('Patched WASM backend did not register.')
      tfGlobal.wasm.setWasmPaths(WASM_DIRECTORY_URL)
    })()
  }
  await wasmScriptPromise
}

const initializeBackend = async (backend: BackendName): Promise<void> => {
  if (backend === 'webgpu') await import('@tensorflow/tfjs-backend-webgpu')
  if (backend === 'wasm') await registerPatchedWasmBackend()
  const selected = await tf.setBackend(backend)
  await tf.ready()
  if (!selected || tf.getBackend() !== backend) throw new Error(`${backend} is unavailable.`)
}

const tensorList = (outputs: ModelOutputs): tf.Tensor[] => {
  if (Array.isArray(outputs)) return outputs
  if (outputs instanceof tf.Tensor) return [outputs]
  return Object.values(outputs)
}

const unpackOutputs = (outputs: ModelOutputs): { tensors: tf.Tensor[]; features: tf.Tensor4D; probabilities: tf.Tensor2D } => {
  const tensors = tensorList(outputs)
  const features = tensors.find((tensor) => tensor.rank === 4) as tf.Tensor4D | undefined
  const probabilities = tensors.find((tensor) => tensor.rank === 2 && tensor.shape[1] === 4) as tf.Tensor2D | undefined
  if (!features || !probabilities) {
    throw new Error(`Unexpected model outputs: ${tensors.map((tensor) => `[${tensor.shape.join(',')}]`).join(', ')}`)
  }
  return { tensors, features, probabilities }
}

const decodeImageBlob = (blob: Blob): Promise<HTMLImageElement> => new Promise((resolve, reject) => {
  const url = URL.createObjectURL(blob)
  const image = new Image()
  image.decoding = 'async'
  image.onload = () => {
    URL.revokeObjectURL(url)
    resolve(image)
  }
  image.onerror = () => {
    URL.revokeObjectURL(url)
    reject(new Error('Image decoding failed.'))
  }
  image.src = url
})

const createModelImageData = (image: HTMLImageElement): ImageData => {
  const width = image.naturalWidth
  const height = image.naturalHeight
  const scale = Math.min(INPUT_SIZE / width, INPUT_SIZE / height)
  const resizedWidth = Math.max(1, Math.round(width * scale))
  const resizedHeight = Math.max(1, Math.round(height * scale))
  const top = Math.floor((INPUT_SIZE - resizedHeight) / 2)
  const left = Math.floor((INPUT_SIZE - resizedWidth) / 2)

  const canvas = document.createElement('canvas')
  canvas.width = INPUT_SIZE
  canvas.height = INPUT_SIZE
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Canvas is unavailable.')
  context.fillStyle = '#000'
  context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(image, left, top, resizedWidth, resizedHeight)
  return context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
}

const createModelInput = (image: HTMLImageElement): tf.Tensor4D => tf.tidy(() => {
  // ImageData uses TensorFlow.js's CPU upload path. Keeping this input at one
  // fixed shape avoids WebGPU reusing a FromPixels shader compiled for a
  // previously classified image with different dimensions.
  return tf.browser.fromPixels(createModelImageData(image)).toFloat().expandDims(0)
})

const validateModel = async (model: tf.GraphModel): Promise<void> => {
  const input = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])
  let outputs: ModelOutputs | null = null
  try {
    outputs = await model.executeAsync(input)
    const unpacked = unpackOutputs(outputs)
    await Promise.all(unpacked.tensors.map((tensor) => tensor.data()))
  } finally {
    input.dispose()
    if (outputs) tf.dispose(outputs)
  }
}

const prepareAcrossBackends = async (listener?: StatusListener): Promise<PreparedModel> => {
  let finalError: unknown = null
  for (const backend of BACKEND_ORDER) {
    let model: tf.GraphModel | null = null
    try {
      emit(listener, {
        phase: 'starting',
        progress: null,
        backend: backendLabel(backend),
        message: `Preparing ${backendLabel(backend)} acceleration…`,
      })
      await initializeBackend(backend)
      const loadStarted = performance.now()
      model = await tf.loadGraphModel(MODEL_URL, {
        onProgress: (fraction) => emit(listener, {
          phase: 'downloading',
          progress: Math.round(fraction * 100),
          backend: backendLabel(backend),
          message: `Loading classifier — ${Math.round(fraction * 100)}%`,
        }),
      })
      const modelLoadMs = performance.now() - loadStarted
      emit(listener, {
        phase: 'optimizing',
        progress: null,
        backend: backendLabel(backend),
        message: 'Optimizing the classifier for this browser…',
      })
      await validateModel(model)
      const prepared = { model, backend, modelLoadMs }
      emit(listener, {
        phase: 'ready',
        progress: 100,
        backend: backendLabel(backend),
        message: `Classifier ready with ${backendLabel(backend)}.`,
      })
      return prepared
    } catch (error) {
      finalError = error
      model?.dispose()
    }
  }
  emit(listener, {
    phase: 'error',
    progress: null,
    backend: null,
    message: 'The classifier could not be prepared. Refresh the page and try again.',
  })
  throw finalError instanceof Error ? finalError : new Error('No TensorFlow.js backend was available.')
}

export const preloadClassifier = (listener?: StatusListener): Promise<PreparedModel> => {
  preparedModelPromise ??= prepareAcrossBackends(listener).catch((error) => {
    preparedModelPromise = null
    throw error
  })
  return preparedModelPromise
}

export const classifyImage = async (blob: Blob): Promise<PredictionResult> => {
  const prepared = await preloadClassifier()
  const image = await decodeImageBlob(blob)
  const input = createModelInput(image)
  let outputs: ModelOutputs | null = null
  try {
    const started = performance.now()
    outputs = await prepared.model.executeAsync(input)
    const unpacked = unpackOutputs(outputs)
    const probabilityData = await unpacked.probabilities.data()
    const predictionMs = performance.now() - started
    const probabilities = Array.from(probabilityData) as [number, number, number, number]
    let predictedIndex = 0
    for (let index = 1; index < probabilities.length; index += 1) {
      if (probabilities[index] > probabilities[predictedIndex]) predictedIndex = index
    }
    const featureData = new Float32Array(await unpacked.features.data())
    return {
      label: CLASS_NAMES[predictedIndex],
      predictedIndex,
      probabilities,
      predictionMs,
      timestamp: Date.now(),
      backend: backendLabel(prepared.backend),
      modelLoadMs: prepared.modelLoadMs,
      features: featureData,
    }
  } finally {
    input.dispose()
    if (outputs) tf.dispose(outputs)
  }
}

const loadGradcamPayload = async (): Promise<GradcamPayload> => {
  gradcamPayloadPromise ??= fetch(GRADCAM_URL).then(async (response) => {
    if (!response.ok) throw new Error('Grad-CAM weights could not be loaded.')
    return response.json() as Promise<GradcamPayload>
  })
  return gradcamPayloadPromise
}

const calculateHeatmap = async (result: PredictionResult): Promise<Float32Array> => {
  const payload = await loadGradcamPayload()
  if (payload.kernel_shape[0] !== 1280 || payload.kernel_shape[1] !== 4) {
    throw new Error('Unexpected Grad-CAM weight shape.')
  }
  const heatmap = tf.tidy(() => {
    const features = tf.tensor4d(result.features, FEATURE_SHAPE)
    const probabilityTensor = tf.tensor1d(result.probabilities, 'float32')
    const denseKernel = tf.tensor2d(payload.kernel, payload.kernel_shape, 'float32')
    const expectedKernel = denseKernel.matMul(probabilityTensor.reshape([4, 1])).squeeze()
    const classKernel = denseKernel.slice([0, result.predictedIndex], [1280, 1]).squeeze()
    const classProbability = probabilityTensor.gather(result.predictedIndex)
    const gradients = classKernel.sub(expectedKernel).mul(classProbability).div(49)
    const rawHeatmap = features.squeeze([0]).mul(gradients.reshape([1, 1, 1280])).sum(2).relu()
    const maximum = rawHeatmap.max()
    return rawHeatmap.div(maximum.add(tf.scalar(1e-12))) as tf.Tensor2D
  })
  try {
    return new Float32Array(await heatmap.data())
  } finally {
    heatmap.dispose()
  }
}

const jetChannel = (value: number, points: Array<[number, number]>): number => {
  const clamped = Math.min(1, Math.max(0, value))
  for (let index = 1; index < points.length; index += 1) {
    const [x1, y1] = points[index - 1]
    const [x2, y2] = points[index]
    if (clamped <= x2) {
      const amount = x2 === x1 ? 0 : (clamped - x1) / (x2 - x1)
      return y1 + (y2 - y1) * amount
    }
  }
  return points[points.length - 1][1]
}

const jetColour = (value: number): [number, number, number] => [
  Math.round(255 * jetChannel(value, [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]])),
  Math.round(255 * jetChannel(value, [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]])),
  Math.round(255 * jetChannel(value, [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]])),
]

const canvasToBlob = (canvas: HTMLCanvasElement, type = 'image/png'): Promise<Blob> => new Promise((resolve, reject) => {
  canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('Image export failed.')), type)
})

export const createGradcamOverlay = async (imageBlob: Blob, result: PredictionResult): Promise<Blob> => {
  const [image, heatmap] = await Promise.all([decodeImageBlob(imageBlob), calculateHeatmap(result)])
  const maxDimension = 1400
  const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight))
  const width = Math.max(1, Math.round(image.naturalWidth * scale))
  const height = Math.max(1, Math.round(image.naturalHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Canvas is unavailable.')
  context.drawImage(image, 0, 0, width, height)
  const original = context.getImageData(0, 0, width, height)

  const source = document.createElement('canvas')
  source.width = 7
  source.height = 7
  const sourceContext = source.getContext('2d')
  if (!sourceContext) throw new Error('Canvas is unavailable.')
  const sourcePixels = sourceContext.createImageData(7, 7)
  heatmap.forEach((value, index) => {
    const level = Math.round(Math.min(1, Math.max(0, value)) * 255)
    const offset = index * 4
    sourcePixels.data[offset] = level
    sourcePixels.data[offset + 1] = level
    sourcePixels.data[offset + 2] = level
    sourcePixels.data[offset + 3] = 255
  })
  sourceContext.putImageData(sourcePixels, 0, 0)

  const resized = document.createElement('canvas')
  resized.width = width
  resized.height = height
  const resizedContext = resized.getContext('2d', { willReadFrequently: true })
  if (!resizedContext) throw new Error('Canvas is unavailable.')
  resizedContext.imageSmoothingEnabled = true
  resizedContext.imageSmoothingQuality = 'high'
  resizedContext.drawImage(source, 0, 0, width, height)
  const levels = resizedContext.getImageData(0, 0, width, height).data
  const alpha = 0.4
  for (let offset = 0; offset < original.data.length; offset += 4) {
    const [red, green, blue] = jetColour(levels[offset] / 255)
    original.data[offset] = Math.round((1 - alpha) * original.data[offset] + alpha * red)
    original.data[offset + 1] = Math.round((1 - alpha) * original.data[offset + 1] + alpha * green)
    original.data[offset + 2] = Math.round((1 - alpha) * original.data[offset + 2] + alpha * blue)
  }
  context.putImageData(original, 0, 0)
  return canvasToBlob(canvas)
}

export const createNormalizedPng = async (imageBlob: Blob): Promise<Blob> => {
  const image = await decodeImageBlob(imageBlob)
  const imageData = createModelImageData(image)
  const canvas = document.createElement('canvas')
  canvas.width = INPUT_SIZE
  canvas.height = INPUT_SIZE
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas is unavailable.')
  context.putImageData(imageData, 0, 0)
  return canvasToBlob(canvas)
}
