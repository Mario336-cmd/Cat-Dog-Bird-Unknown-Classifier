import { useState } from 'react'
import { CLASS_NAMES, displayClassName, type ClassName, type FeedbackChoice, type PredictionResult } from '../types'
import { buildFeedbackZip, downloadBlob, FEEDBACK_FORM_URL } from '../lib/feedback'

interface FeedbackPanelProps {
  result: PredictionResult
  imageBlob: Blob
}

type ZipStatus = 'idle' | 'generating' | 'downloaded' | 'error'
type FeedbackStep = 'invite' | 'declined' | 'accuracy' | 'correction' | 'consent' | 'upload'

function FeedbackPanel({ result, imageBlob }: FeedbackPanelProps) {
  const [step, setStep] = useState<FeedbackStep>('invite')
  const [choice, setChoice] = useState<FeedbackChoice | null>(null)
  const [correctClass, setCorrectClass] = useState<ClassName | null>(null)
  const [consent, setConsent] = useState(false)
  const [zipStatus, setZipStatus] = useState<ZipStatus>('idle')

  const selectChoice = (nextChoice: FeedbackChoice) => {
    setChoice(nextChoice)
    setCorrectClass(nextChoice === 'correct' ? result.label : null)
    setConsent(false)
    setZipStatus('idle')
    setStep(nextChoice === 'correct' ? 'consent' : 'correction')
  }

  const goBack = () => {
    setZipStatus('idle')
    setConsent(false)
    if (step === 'declined' || step === 'accuracy') {
      setStep('invite')
      setChoice(null)
      setCorrectClass(null)
      return
    }
    if (step === 'correction') {
      setStep('accuracy')
      setChoice(null)
      setCorrectClass(null)
      return
    }
    if (step === 'consent') {
      setStep(choice === 'incorrect' ? 'correction' : 'accuracy')
      if (choice === 'correct') {
        setChoice(null)
        setCorrectClass(null)
      }
      return
    }
    if (step === 'upload') setStep('consent')
  }

  const canPrepare = Boolean(choice && correctClass && consent)

  const prepareZip = async () => {
    if (!canPrepare || !choice || !correctClass || zipStatus === 'generating') return
    setZipStatus('generating')
    try {
      const feedbackZip = await buildFeedbackZip(imageBlob, result, choice, correctClass)
      downloadBlob(feedbackZip.blob, feedbackZip.filename)
      setZipStatus('downloaded')
      setStep('upload')
    } catch {
      setZipStatus('error')
    }
  }

  return (
    <section className="feedback-panel">
      <h2>Help Improve the Classifier</h2>
      <div className="feedback-stage" aria-live="polite">
        {step === 'invite' ? (
          <div className="feedback-step" key="invite">
            <p className="feedback-question">Would you like to provide feedback?</p>
            <div className="choice-row" role="group" aria-label="Would you like to provide feedback?">
              <button type="button" className="choice-button" onClick={() => setStep('accuracy')}>Yes</button>
              <button type="button" className="choice-button" onClick={() => setStep('declined')}>No</button>
            </div>
          </div>
        ) : null}

        {step === 'declined' ? (
          <div className="feedback-step" key="declined">
            <button type="button" className="feedback-back" onClick={goBack}>Back</button>
            <p className="feedback-note">No problem. Nothing will be collected.</p>
          </div>
        ) : null}

        {step === 'accuracy' ? (
          <div className="feedback-step" key="accuracy">
            <button type="button" className="feedback-back" onClick={goBack}>Back</button>
            <p className="feedback-question">Was the prediction correct?</p>
            <div className="choice-row" role="group" aria-label="Was the prediction correct?">
              <button type="button" className="choice-button" onClick={() => selectChoice('correct')}>Correct</button>
              <button type="button" className="choice-button" onClick={() => selectChoice('incorrect')}>Incorrect</button>
            </div>
          </div>
        ) : null}

        {step === 'correction' ? (
          <div className="feedback-step correction-block" key="correction">
            <button type="button" className="feedback-back" onClick={goBack}>Back</button>
            <p className="feedback-question">What should the prediction have been?</p>
            <div className="class-choice-grid" role="radiogroup" aria-label="Correct class">
              {CLASS_NAMES.map((name) => (
                <button
                  type="button"
                  role="radio"
                  aria-checked={correctClass === name}
                  className={correctClass === name ? 'class-choice is-selected' : 'class-choice'}
                  disabled={name === result.label}
                  onClick={() => {
                    setCorrectClass(name)
                    setConsent(false)
                    setZipStatus('idle')
                    setStep('consent')
                  }}
                  key={name}
                >
                  {displayClassName(name)}
                  {name === result.label ? <small>Current prediction</small> : null}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {step === 'consent' ? (
          <div className="feedback-step consent-block" key="consent">
            <button type="button" className="feedback-back" onClick={goBack}>Back</button>
            <p>Review the consent statement below, then download the feedback ZIP. You&apos;ll upload the ZIP through the feedback form in the next step.</p>
            <label className="consent-label">
              <input type="checkbox" checked={consent} onChange={(event) => {
                setConsent(event.target.checked)
                setZipStatus('idle')
              }} />
              <span>I confirm I have the right to share this image and agree that it may be used to review and improve the classifier.</span>
            </label>
            <button type="button" className="primary-button feedback-download" disabled={!canPrepare || zipStatus === 'generating'} onClick={() => void prepareZip()}>
              {zipStatus === 'generating' ? 'Preparing ZIP…' : 'Download ZIP and Continue'}
            </button>
            {zipStatus === 'error' ? <p className="inline-error" role="alert">The feedback ZIP could not be prepared. Please try again.</p> : null}
          </div>
        ) : null}

        {step === 'upload' ? (
          <div className="feedback-step upload-feedback-block" key="upload">
            <button type="button" className="feedback-back" onClick={goBack}>Back</button>
            <h3>Upload Your Feedback</h3>
            <p>Your feedback has not been submitted yet. Click the button to open the feedback form, then upload the downloaded ZIP to complete your submission.</p>
            <a className="form-link" href={FEEDBACK_FORM_URL} target="_blank" rel="noreferrer">Open Feedback Form</a>
          </div>
        ) : null}
      </div>
    </section>
  )
}

export default FeedbackPanel
