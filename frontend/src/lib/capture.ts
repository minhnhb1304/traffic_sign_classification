import { computeCenterRoi } from './roi'
import type { Roi } from './roi'

// Singleton offscreen canvas to prevent heavy Garbage Collection churn during 8fps realtime loops.
let _offscreenCanvas: HTMLCanvasElement | null = null

/**
 * Crop the center-square ROI of a video/canvas/image into a fresh canvas at
 * the ROI's native resolution. Returns the canvas plus the ROI used, so the
 * caller can also draw the box on a preview overlay.
 */
export function cropRoiToCanvas(
  source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
  width: number,
  height: number,
  ratio: number,
): { canvas: HTMLCanvasElement; roi: Roi } {
  const roi = computeCenterRoi(height, width, ratio)
  
  if (!_offscreenCanvas) {
    _offscreenCanvas = document.createElement('canvas')
  }
  const canvas = _offscreenCanvas
  
  if (canvas.width !== roi.side || canvas.height !== roi.side) {
    canvas.width = roi.side
    canvas.height = roi.side
  }
  
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(
    source,
    roi.x0,
    roi.y0,
    roi.side,
    roi.side,
    0,
    0,
    roi.side,
    roi.side,
  )
  return { canvas, roi }
}

/** Draw a green ROI rectangle on a 2D overlay context sized to the source. */
export function drawRoiBox(
  ctx: CanvasRenderingContext2D,
  roi: Roi,
  color = '#16a34a',
) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(2, Math.round(ctx.canvas.width / 240))
  ctx.strokeRect(roi.x0, roi.y0, roi.side, roi.side)
}
