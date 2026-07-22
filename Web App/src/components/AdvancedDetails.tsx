import { useEffect, useState } from 'react'
import { CLASS_NAMES, displayClassName, type PredictionResult } from '../types'

interface AdvancedDetailsProps {
  result: PredictionResult
  imageBlob: Blob
}

type OverlayStatus = 'idle' | 'loading' | 'ready' | 'error'

const percentage = (value: number): string => `${(value * 100).toFixed(2)}%`

function AdvancedDetails({ result, imageBlob }: AdvancedDetailsProps) {
  const [overlayStatus, setOverlayStatus] = useState<OverlayStatus>('idle')
  const [overlayUrl, setOverlayUrl] = useState('')

  useEffect(() => () => {
    if (overlayUrl) URL.revokeObjectURL(overlayUrl)
  }, [overlayUrl])

  const generateOverlay = async () => {
    if (overlayStatus !== 'idle') return
    setOverlayStatus('loading')
    try {
      const { createGradcamOverlay } = await import('../lib/classifier')
      const overlay = await createGradcamOverlay(imageBlob, result)
      setOverlayUrl(URL.createObjectURL(overlay))
      setOverlayStatus('ready')
    } catch {
      setOverlayStatus('error')
    }
  }

  return (
    <details className="advanced-details" onToggle={(event) => {
      if (event.currentTarget.open) void generateOverlay()
    }}>
      <summary>View Advanced Details</summary>
      <div className="advanced-content">
        <div className="advanced-overview-grid">
          <div className="advanced-info-stack">
            <section className="advanced-block probability-block">
              <h3>Class Probability Distribution</h3>
              <div className="probability-list">
                {CLASS_NAMES.map((name, index) => (
                  <div className="probability-item" key={name}>
                    <div className="probability-meta">
                      <span>{displayClassName(name)}</span>
                      <strong>{percentage(result.probabilities[index])}</strong>
                    </div>
                    <div className="probability-track" aria-hidden="true">
                      <span style={{ width: percentage(result.probabilities[index]) }} />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="advanced-block">
              <h3>Inference Performance</h3>
              <dl className="detail-list">
                <div><dt>Prediction time</dt><dd>{result.predictionMs.toFixed(2)} ms</dd></div>
                <div><dt>Model loading time</dt><dd>{result.modelLoadMs.toFixed(2)} ms</dd></div>
                <div><dt>TensorFlow.js backend</dt><dd>{result.backend}</dd></div>
              </dl>
            </section>

            <section className="advanced-block">
              <h3>Model Architecture</h3>
              <dl className="detail-list">
                <div><dt>Classifier</dt><dd>EfficientNetV2B0</dd></div>
                <div><dt>Input size</dt><dd>224 × 224 × 3</dd></div>
                <div><dt>Approximate size</dt><dd>22.7 MB</dd></div>
              </dl>
            </section>

            <section className="advanced-block">
              <h3>Input Preprocessing</h3>
              <p>The image is decoded locally, resized using high-quality browser resampling to fit within 224 × 224 while preserving its aspect ratio, centred with black padding, converted to RGB, and kept in the [0,255] pixel range. EfficientNetV2 preprocessing is included inside the model.</p>
            </section>

            <section className="advanced-block">
              <h3>Classification Decision Rule</h3>
              <p>The displayed prediction is the class with the highest probability. No confidence threshold is applied.</p>
            </section>
          </div>

          <section className="advanced-block heatmap-block">
            <h3>Grad-CAM Class Activation Map</h3>
            <p>Uses gradient-weighted feature activations to highlight the image regions that contributed most to the predicted class.</p>
            <div className="heatmap-frame" aria-live="polite">
              {overlayStatus === 'loading' ? (
                <div className="heatmap-status"><span className="spinner" />Generating Grad-CAM overlay…</div>
              ) : overlayStatus === 'error' ? (
                <div className="heatmap-status heatmap-error">The Grad-CAM overlay could not be generated.</div>
              ) : overlayUrl ? (
                <img src={overlayUrl} alt={`Grad-CAM overlay for the ${displayClassName(result.label)} prediction`} />
              ) : null}
            </div>
          </section>
        </div>
      </div>
    </details>
  )
}

export default AdvancedDetails
