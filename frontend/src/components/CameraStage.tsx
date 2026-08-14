import * as React from 'react'
import { computeCenterRoi } from '@/lib/roi'
import { drawRoiBox } from '@/lib/capture'

interface CameraStageProps {
  videoRef: React.RefObject<HTMLVideoElement | null>
  ratio: number
  active: boolean
  /** Optional label drawn above the ROI box (e.g. live prediction). */
  overlayText?: string
  overlayColor?: string
}

/** Live <video> with a center ROI box drawn on a synced canvas overlay. */
export function CameraStage({
  videoRef,
  ratio,
  active,
  overlayText,
  overlayColor = '#16a34a',
}: CameraStageProps) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const textRef = React.useRef<string | undefined>(overlayText)
  const colorRef = React.useRef(overlayColor)
  textRef.current = overlayText
  colorRef.current = overlayColor

  React.useEffect(() => {
    if (!active) return
    let raf = 0
    const tick = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (video && canvas && video.clientWidth > 0) {
        const w = video.clientWidth
        const h = video.clientHeight
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w
          canvas.height = h
        }
        const ctx = canvas.getContext('2d')!
        const roi = computeCenterRoi(h, w, ratio)
        drawRoiBox(ctx, roi, colorRef.current)
        if (textRef.current) {
          ctx.font = '600 16px system-ui, sans-serif'
          ctx.fillStyle = colorRef.current
          ctx.fillText(textRef.current, roi.x0, Math.max(16, roi.y0 - 8))
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [active, ratio, videoRef])

  return (
    <div className="relative overflow-hidden rounded-lg border bg-black">
      <video
        ref={videoRef}
        playsInline
        muted
        className="block max-h-[60vh] w-full object-contain"
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />
      {!active && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-white/70">
          Camera đang tắt
        </div>
      )}
    </div>
  )
}
