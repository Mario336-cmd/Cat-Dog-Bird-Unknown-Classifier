import { lazy, Suspense, useEffect, useState } from 'react'

const App = lazy(() => import('./App'))
const MOBILE_USER_AGENT = /android|iphone|ipad|ipod|mobile|iemobile|opera mini/i

const isMobileLikeDevice = (): boolean => {
  const coarsePointer = window.matchMedia('(pointer: coarse)').matches
  const smallViewport = window.innerWidth < 1024
  return MOBILE_USER_AGENT.test(navigator.userAgent.toLowerCase()) || (coarsePointer && smallViewport)
}

function DeviceGate() {
  const [blocked, setBlocked] = useState(isMobileLikeDevice)

  useEffect(() => {
    const update = () => setBlocked(isMobileLikeDevice())
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  if (blocked) {
    return (
      <main className="device-gate">
        <section>
          <p>Image Classifier</p>
          <h1>Desktop Only</h1>
          <span>This experience currently works on desktop and laptop browsers. Please open it on a computer.</span>
        </section>
      </main>
    )
  }

  return <Suspense fallback={<main className="app-loading"><span />Loading…</main>}><App /></Suspense>
}

export default DeviceGate
