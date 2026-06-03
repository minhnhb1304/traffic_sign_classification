/**
 * Center square ROI — mirrors Python `compute_center_roi` exactly:
 *   side = int(min(h, w) * ratio)   (truncation toward zero)
 *   x0   = (w - side) // 2          (integer floor division)
 *   y0   = (h - side) // 2
 */
export interface Roi {
  x0: number
  y0: number
  side: number
}

export function computeCenterRoi(h: number, w: number, ratio: number): Roi {
  const side = Math.trunc(Math.min(h, w) * ratio)
  const x0 = Math.floor((w - side) / 2)
  const y0 = Math.floor((h - side) / 2)
  return { x0, y0, side }
}
