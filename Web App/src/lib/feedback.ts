import { displayClassName, type ClassName, type FeedbackChoice, type PredictionResult } from '../types'

export const FEEDBACK_FORM_URL = import.meta.env.VITE_FEEDBACK_FORM_URL?.trim()
  || 'https://docs.google.com/forms/d/e/1FAIpQLSfSocfHZV35zSCbBWZR-m-iwIxM7sZ2x7kY8kxTf9bYUWMFEQ/viewform?usp=publish-editor'

const roundPercent = (value: number): number => Number((value * 100).toFixed(2))

export const buildFeedbackZip = async (
  imageBlob: Blob,
  result: PredictionResult,
  choice: FeedbackChoice,
  correctClass: ClassName,
): Promise<{ blob: Blob; filename: string }> => {
  const [{ default: JSZip }, { createNormalizedPng }] = await Promise.all([
    import('jszip'),
    import('./classifier'),
  ])
  const normalizedImage = await createNormalizedPng(imageBlob)
  const zip = new JSZip()
  const folder = zip.folder('feedback')
  if (!folder) throw new Error('Feedback ZIP could not be created.')
  folder.file('image.png', normalizedImage)
  folder.file('prediction.json', JSON.stringify({
    feedback: choice,
    predicted_class: displayClassName(result.label),
    correct_class: displayClassName(correctClass),
    confidence_scores_percent: {
      Cat: roundPercent(result.probabilities[0]),
      Dog: roundPercent(result.probabilities[1]),
      Bird: roundPercent(result.probabilities[2]),
      Other: roundPercent(result.probabilities[3]),
    },
    prediction_time_ms: Number(result.predictionMs.toFixed(2)),
    date_and_time: new Date(result.timestamp).toISOString(),
    input: {
      size: [224, 224, 3],
      colour_mode: 'RGB',
      resize: 'Aspect-ratio-preserving high-quality resize with padding',
      pixel_range: '[0,255]',
    },
    decision_rule: 'Highest probability',
  }, null, 2))
  const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } })
  const timestamp = new Date(result.timestamp).toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
  return {
    blob,
    filename: `classifier-feedback-${displayClassName(correctClass).toLowerCase()}-${timestamp}.zip`,
  }
}

export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
