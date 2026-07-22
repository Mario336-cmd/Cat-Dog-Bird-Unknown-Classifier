import { useEffect, useMemo, useRef, useState } from 'react'
import AdvancedDetails from './components/AdvancedDetails'
import FeedbackPanel from './components/FeedbackPanel'
import Hero from './components/Hero'
import { createHeroSequence, preloadHeroImages } from './lib/heroImages'
import { ACCEPTED_FILE_TYPES, MAX_UPLOAD_MB, createValidationError, prepareImageUrl, validateFile } from './lib/validators'
import type { AppStatus, InputSource, ModelStatus, PredictionResult, ValidationResult } from './types'
import './styles/app.css'

const initialModelStatus: ModelStatus = {
  phase: 'starting',
  progress: null,
  backend: null,
  message: 'Loading…',
}

function App() {
  const heroImages = useMemo(createHeroSequence, [])
  const [source, setSource] = useState<InputSource>('upload')
  const [status, setStatus] = useState<AppStatus>('idle')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState('')
  const [previewUrl, setPreviewUrl] = useState('')
  const [previewReady, setPreviewReady] = useState(false)
  const [error, setError] = useState<ValidationResult | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [classifiedBlob, setClassifiedBlob] = useState<Blob | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelStatus>(initialModelStatus)
  const classifyLock = useRef(false)

  useEffect(() => preloadHeroImages(heroImages), [heroImages])

  useEffect(() => {
    let cancelled = false
    void import('./lib/classifier')
      .then(({ preloadClassifier }) => preloadClassifier((nextStatus) => {
        if (!cancelled) setModelStatus(nextStatus)
      }))
      .then((prepared) => {
        if (!cancelled) {
          const backend = prepared.backend === 'webgpu'
            ? 'WebGPU'
            : prepared.backend === 'webgl'
              ? 'WebGL'
              : prepared.backend.toUpperCase()
          setModelStatus({ phase: 'ready', progress: 100, backend, message: `Classifier ready with ${backend}.` })
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModelStatus({
            phase: 'error',
            progress: null,
            backend: null,
            message: 'The classifier could not be prepared. Refresh the page and try again.',
          })
        }
      })
    return () => { cancelled = true }
  }, [])

  useEffect(() => () => {
    if (previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const clearPrediction = () => {
    setResult(null)
    setClassifiedBlob(null)
  }

  const replaceImage = () => {
    setStatus('idle')
    setSelectedFile(null)
    setImageUrl('')
    setPreviewUrl('')
    setPreviewReady(false)
    setError(null)
    clearPrediction()
  }

  const chooseFile = (file: File | null) => {
    if (!file) return
    const validation = validateFile(file)
    if (!validation.ok) {
      setError(validation)
      setStatus('error')
      return
    }
    setSource('upload')
    setSelectedFile(file)
    setImageUrl('')
    setPreviewUrl(URL.createObjectURL(file))
    setPreviewReady(false)
    setError(null)
    setStatus('ready')
    clearPrediction()
  }

  const updateImageUrl = (value: string) => {
    setImageUrl(value)
    setSource('url')
    clearPrediction()
    if (!value.trim()) {
      setSelectedFile(null)
      setPreviewUrl('')
      setPreviewReady(false)
      setError(null)
      setStatus('idle')
      return
    }
    const prepared = prepareImageUrl(value)
    if (!prepared.validation.ok || !prepared.normalizedUrl) {
      setSelectedFile(null)
      setPreviewUrl('')
      setPreviewReady(false)
      setError(prepared.validation)
      setStatus('error')
      return
    }
    setSelectedFile(null)
    setPreviewUrl(prepared.normalizedUrl)
    setPreviewReady(false)
    setError(null)
    setStatus('ready')
  }

  const classify = async () => {
    if (classifyLock.current || !previewReady || modelStatus.phase !== 'ready') return
    classifyLock.current = true
    setStatus('classifying')
    setError(null)
    clearPrediction()
    try {
      let blob: Blob
      if (source === 'upload' && selectedFile) {
        blob = selectedFile
      } else if (source === 'url' && previewUrl) {
        try {
          const response = await fetch(previewUrl, { mode: 'cors', cache: 'default' })
          if (!response.ok) throw new Error('Image request failed.')
          blob = await response.blob()
        } catch {
          setError(createValidationError('cors_blocked'))
          setStatus('error')
          return
        }
      } else {
        throw new Error('No image is selected.')
      }
      const { classifyImage } = await import('./lib/classifier')
      const nextResult = await classifyImage(blob)
      setClassifiedBlob(blob)
      setResult(nextResult)
      setStatus('success')
    } catch {
      setError(createValidationError('classification_failed'))
      setStatus('error')
    } finally {
      classifyLock.current = false
    }
  }

  const canClassify = previewReady && status !== 'classifying' && modelStatus.phase === 'ready'
  const confidence = result ? result.probabilities[result.predictedIndex] : 0
  const confidenceText = `${(confidence * 100).toFixed(2)}%`
  const displayLabel = result?.label === 'Unknown' ? 'Not a Cat, Dog or Bird' : result?.label
  const helperText = result?.label === 'Unknown' ? 'Please try an image of a cat, dog or bird.' : null
  const resultTone = result ? `result-${result.label.toLowerCase()}` : ''
  const classifyButtonLabel = status === 'classifying'
    ? 'Classifying...'
    : modelStatus.phase === 'error'
      ? 'Model Unavailable'
      : modelStatus.phase !== 'ready'
        ? 'Loading...'
        : previewUrl && !previewReady
          ? 'Loading Preview...'
        : result
          ? 'Classify Again'
          : 'Classify'

  return (
    <main className="page-shell">
      <Hero images={heroImages} />

      <section className="section-card how-it-works">
        <h2 className="section-heading">How It Works</h2>
        <div className="steps-grid">
          <article className="step-card">
            <p className="step-index">01</p>
            <h3>Add Your Image</h3>
            <p>Upload a photo from your device or paste an image URL.</p>
          </article>
          <article className="step-card">
            <p className="step-index">02</p>
            <h3>Check the Preview</h3>
            <p>If the image appears, it is ready to classify.</p>
          </article>
          <article className="step-card">
            <p className="step-index">03</p>
            <h3>Get the Result</h3>
            <p>Click Classify to see whether it is a cat, dog or bird.</p>
          </article>
        </div>
      </section>

      <section className="section-card workspace">
        <div className="controls-column">
          <h2 className="section-heading">Try Classification</h2>
          <p className="section-subtitle">Provide an image from your device or paste an image URL.</p>

          <div className="mode-switch" role="radiogroup" aria-label="Image source">
            <button type="button" role="radio" aria-checked={source === 'upload'} className={source === 'upload' ? 'mode-button is-active' : 'mode-button'} onClick={() => setSource('upload')} disabled={status === 'classifying'}>Upload Image</button>
            <button type="button" role="radio" aria-checked={source === 'url'} className={source === 'url' ? 'mode-button is-active' : 'mode-button'} onClick={() => setSource('url')} disabled={status === 'classifying'}>Image URL</button>
          </div>

          <div className="input-panel-stack">
            <div className={`input-panel input-panel-upload ${source === 'upload' ? 'is-active' : ''}`} aria-hidden={source !== 'upload'}>
              <div className="field-block">
                <label className="field-label" htmlFor="image-upload">Upload image</label>
                <div className="upload-row">
                  <input id="image-upload" type="file" className="visually-hidden" accept={ACCEPTED_FILE_TYPES} disabled={status === 'classifying' || source !== 'upload'} onChange={(event) => {
                    chooseFile(event.target.files?.[0] ?? null)
                    event.currentTarget.value = ''
                  }} />
                  <label className={status === 'classifying' || source !== 'upload' ? 'upload-trigger is-disabled' : 'upload-trigger'} htmlFor="image-upload">Choose File</label>
                  <p className="upload-file-name">{selectedFile?.name ?? 'No file selected'}</p>
                </div>
                <p className="field-help">Accepted: JPG, PNG, WEBP. Max size: {MAX_UPLOAD_MB}MB.</p>
              </div>
            </div>

            <div className={`input-panel input-panel-url ${source === 'url' ? 'is-active' : ''}`} aria-hidden={source !== 'url'}>
              <div className="field-block">
                <label className="field-label" htmlFor="image-url">Image URL</label>
                <input id="image-url" className="url-input" type="text" inputMode="url" placeholder="https://example.com/cat-dog-or-bird.jpg" value={imageUrl} onChange={(event) => updateImageUrl(event.target.value)} disabled={status === 'classifying' || source !== 'url'} />
                <p className="field-help">Paste direct image links, search-result wrapper links, or snippets (Markdown/HTML) that contain an image URL. Publicly accessible images work best.</p>
              </div>
            </div>
          </div>

          {error?.message ? <div className="error-banner" role="alert">{error.message}</div> : null}
          {!error?.message && modelStatus.phase === 'error' ? <div className="error-banner" role="alert">{modelStatus.message}</div> : null}

          <div className="action-row">
            <button type="button" className="primary-button" disabled={!canClassify} onClick={() => void classify()}>{classifyButtonLabel}</button>
            <button type="button" className="ghost-button" disabled={!previewUrl || status === 'classifying'} onClick={replaceImage}>Remove Image</button>
          </div>

          {!result ? status === 'classifying' ? (
            <section className="result-shell result-loading" aria-live="polite">
              <span className="loading-dot" aria-hidden="true" />
              <p>Running classification...</p>
            </section>
          ) : (
            <section className="result-shell result-empty" aria-live="polite">
              <h3>Prediction</h3>
              <p>Classification results will appear here.</p>
            </section>
          ) : (
            <section className={`result-shell ${resultTone}`} aria-live="polite">
              <p className="result-heading">Prediction Results</p>
              <p className="result-value">{displayLabel}</p>
              {helperText ? <p className="result-meta">{helperText}</p> : null}
              <div className="confidence-row">
                <div className="confidence-meta">
                  <span>Model Confidence</span>
                  <strong>{confidenceText}</strong>
                </div>
                <div className="confidence-track" aria-hidden="true">
                  <span className="confidence-fill" style={{ width: confidenceText }} />
                </div>
              </div>
              <p className="result-meta">Result generated at {new Date(result.timestamp).toLocaleTimeString([], { hour12: false })}</p>
            </section>
          )}
        </div>

        <div className="preview-column">
          {previewUrl ? (
            <figure className="preview-shell">
              <div className="preview-media">
                <img key={previewUrl} src={previewUrl} alt="Selected image for classification" className="preview-image" onLoad={() => setPreviewReady(true)} onError={() => {
                  setPreviewReady(false)
                  setStatus('error')
                  setError(createValidationError('image_load_failed'))
                }} />
              </div>
              <figcaption className="preview-caption">Preview ready</figcaption>
            </figure>
          ) : (
            <div className="preview-shell preview-empty">
              <p className="preview-empty-title">Image Preview</p>
              <p>Add an image to see its preview here.</p>
            </div>
          )}
        </div>
      </section>

      {result && classifiedBlob ? (
        <section className="section-card result-extras">
          <FeedbackPanel key={result.timestamp} result={result} imageBlob={classifiedBlob} />
          <AdvancedDetails key={`advanced-${result.timestamp}`} result={result} imageBlob={classifiedBlob} />
        </section>
      ) : null}
    </main>
  )
}

export default App
