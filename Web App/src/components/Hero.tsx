import { useEffect, useRef, useState } from 'react'

interface HeroProps {
  images: string[]
}

const ROTATION_MS = 6500
const CROSSFADE_MS = 620
const CROSSFADE_SETTLE_MS = 50

function Hero({ images }: HeroProps) {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [baseImageUrl, setBaseImageUrl] = useState(images[0] ?? '')
  const [overlayImageUrl, setOverlayImageUrl] = useState('')
  const [isOverlayVisible, setIsOverlayVisible] = useState(false)
  const overlayImageRef = useRef('')
  const crossfadeTimeoutRef = useRef<number | null>(null)

  const clearCrossfadeTimeout = () => {
    if (crossfadeTimeoutRef.current !== null) {
      window.clearTimeout(crossfadeTimeoutRef.current)
      crossfadeTimeoutRef.current = null
    }
  }

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const updatePreference = (event: MediaQueryListEvent) => setPrefersReducedMotion(event.matches)
    mediaQuery.addEventListener('change', updatePreference)
    return () => mediaQuery.removeEventListener('change', updatePreference)
  }, [])

  useEffect(() => {
    if (prefersReducedMotion || images.length < 2 || !baseImageUrl || overlayImageUrl) return

    const rotationTimeout = window.setTimeout(() => {
      const nextIndex = (currentIndex + 1) % images.length
      const nextImageUrl = images[nextIndex] ?? ''
      overlayImageRef.current = nextImageUrl
      setOverlayImageUrl(nextImageUrl)
      setIsOverlayVisible(false)
    }, ROTATION_MS)

    return () => window.clearTimeout(rotationTimeout)
  }, [baseImageUrl, currentIndex, images, overlayImageUrl, prefersReducedMotion])

  useEffect(() => () => clearCrossfadeTimeout(), [])

  const handleOverlayLoad = (loadedImageUrl: string) => {
    if (!loadedImageUrl || overlayImageRef.current !== loadedImageUrl) return

    window.requestAnimationFrame(() => {
      if (overlayImageRef.current === loadedImageUrl) setIsOverlayVisible(true)
    })

    clearCrossfadeTimeout()
    crossfadeTimeoutRef.current = window.setTimeout(() => {
      if (overlayImageRef.current !== loadedImageUrl) return

      const loadedIndex = images.indexOf(loadedImageUrl)
      overlayImageRef.current = ''
      setBaseImageUrl(loadedImageUrl)
      setOverlayImageUrl('')
      setIsOverlayVisible(false)
      if (loadedIndex >= 0) setCurrentIndex(loadedIndex)
      crossfadeTimeoutRef.current = null
    }, CROSSFADE_MS + CROSSFADE_SETTLE_MS)
  }

  const handleOverlayError = (failedImageUrl: string) => {
    if (overlayImageRef.current !== failedImageUrl) return

    clearCrossfadeTimeout()
    const failedIndex = images.indexOf(failedImageUrl)
    overlayImageRef.current = ''
    setOverlayImageUrl('')
    setIsOverlayVisible(false)
    if (failedIndex >= 0) setCurrentIndex(failedIndex)
  }

  const visibleImageUrl = overlayImageUrl || baseImageUrl

  return (
    <section className="section-card hero">
      <div className="hero-layout">
        <div className="hero-content">
          <div className="hero-meta">
            <p className="hero-kicker">Computer Vision</p>
            <span className="hero-meta-separator" aria-hidden="true">|</span>
            <a className="source-link" href="https://github.com/Mario336-cmd/Cat-Dog-Bird-Unknown-Classifier" target="_blank" rel="noreferrer">Source Code</a>
          </div>
          <h1>Cat, Dog and Bird Image Classifier</h1>
          <p className="hero-copy">
            This website is an interactive showcase of a convolutional neural network (CNN) model
            trained to classify images depicting <strong>domestic cats, domestic dogs, or birds</strong>, while
            identifying images outside those categories as <strong>Other</strong>. Users can
            upload an image from their device or provide a publicly accessible image link, and the model
            runs locally in their browser to return a prediction and class probability in real time, with
            an optional Grad-CAM visualization highlighting the image regions that contributed most to
            the prediction.
          </p>
          <p className="hero-copy">
            <strong>Note:</strong> Unlike the Cat and Dog categories, which focus on domesticated animals,
            the Bird category includes many species, including wild birds, with a wider range of appearances. For now, penguins
            and large flightless birds are excluded and fall outside the classifier&apos;s intended scope.
          </p>
        </div>

        {visibleImageUrl ? (
          <figure className="hero-image-shell" aria-hidden="true">
            {baseImageUrl ? (
              <img
                className="hero-image hero-image-base"
                src={baseImageUrl}
                alt=""
                loading="eager"
                decoding="async"
                fetchPriority="high"
              />
            ) : null}
            {overlayImageUrl ? (
              <img
                className={`hero-image hero-image-overlay ${isOverlayVisible ? 'is-visible' : ''}`}
                src={overlayImageUrl}
                alt=""
                loading="eager"
                decoding="async"
                fetchPriority="high"
                onLoad={() => handleOverlayLoad(overlayImageUrl)}
                onError={() => handleOverlayError(overlayImageUrl)}
              />
            ) : null}
          </figure>
        ) : null}
      </div>
    </section>
  )
}

export default Hero
