import * as React from 'react'

export type CameraError = 'denied' | 'notfound' | 'unsupported' | 'unknown'

interface UseCameraResult {
  videoRef: React.RefObject<HTMLVideoElement | null>
  active: boolean
  error: CameraError | null
  start: () => Promise<void>
  stop: () => void
}

/** Manages a getUserMedia camera stream bound to a <video> element. */
export function useCamera(): UseCameraResult {
  const videoRef = React.useRef<HTMLVideoElement | null>(null)
  const streamRef = React.useRef<MediaStream | null>(null)
  const [active, setActive] = React.useState(false)
  const [error, setError] = React.useState<CameraError | null>(null)

  const stop = React.useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
  }, [])

  const start = React.useCallback(async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('unsupported')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => undefined)
      }
      setActive(true)
    } catch (e: unknown) {
      const name = e instanceof DOMException ? e.name : ''
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setError('denied')
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setError('notfound')
      } else {
        setError('unknown')
      }
    }
  }, [])

  React.useEffect(() => stop, [stop])

  return { videoRef, active, error, start, stop }
}
