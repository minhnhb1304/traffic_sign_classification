import * as React from 'react'
import ReactCrop, { type Crop, type PixelCrop } from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'
import type { LayersModel } from '@tensorflow/tfjs'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { ResultsPanel } from '@/components/ResultsPanel'
import { predictProba, topK } from '@/lib/preprocess'
import type { Pred, PixelSource } from '@/lib/preprocess'
import type { Label } from '@/lib/labels'

const MIN_CROP_PX = 16

interface CropPredictPanelProps {
  src: string | null
  model: LayersModel | null
  labels: Label[] | null
  emptyHint: string
}

/** Fit display size for the cropper — mirrors Streamlit `_fit_for_cropper`:
 * upscale small images so they are easy to crop, downscale large ones to fit.
 * Pure display scaling; the crop maps back to native pixels for inference. */
function fitDisplay(
  nw: number,
  nh: number,
  minSide = 260,
  maxSide = 400,
): { w: number; h: number } {
  const short = Math.min(nw, nh)
  const long = Math.max(nw, nh)
  let scale = 1
  if (short < minSide) scale = minSide / short
  if (long * scale > maxSide) scale = maxSide / long
  return { w: Math.round(nw * scale), h: Math.round(nh * scale) }
}

/** Centered crop in *display* pixels; square uses the shorter side. */
function centeredPixelCrop(w: number, h: number, square: boolean): PixelCrop {
  const cw = square ? Math.min(w, h) * 0.8 : w * 0.8
  const ch = square ? cw : h * 0.8
  return { unit: 'px', width: cw, height: ch, x: (w - cw) / 2, y: (h - ch) / 2 }
}

/** Map a display-space crop back to the image's native pixels via a canvas. */
function cropToCanvas(img: HTMLImageElement, crop: PixelCrop): HTMLCanvasElement {
  const sx = img.naturalWidth / img.width
  const sy = img.naturalHeight / img.height
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(crop.width * sx))
  canvas.height = Math.max(1, Math.round(crop.height * sy))
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(
    img, crop.x * sx, crop.y * sy, crop.width * sx, crop.height * sy,
    0, 0, canvas.width, canvas.height,
  )
  return canvas
}

/** Shared crop + predict panel — mirrors Streamlit `_render_crop_and_predict`. */
export function CropPredictPanel({ src, model, labels, emptyHint }: CropPredictPanelProps) {
  const imgRef = React.useRef<HTMLImageElement | null>(null)
  const [useCrop, setUseCrop] = React.useState(true)
  const [square, setSquare] = React.useState(true)
  const [crop, setCrop] = React.useState<Crop>()
  const [completedCrop, setCompletedCrop] = React.useState<PixelCrop>()
  const [preds, setPreds] = React.useState<Pred[]>([])
  const [busy, setBusy] = React.useState(false)
  const [tooSmall, setTooSmall] = React.useState(false)
  const [display, setDisplay] = React.useState<{ w: number; h: number } | null>(
    null,
  )

  React.useEffect(() => {
    setCrop(undefined)
    setCompletedCrop(undefined)
    setPreds([])
    setTooSmall(false)
    setDisplay(null)
  }, [src])

  const onImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const el = e.currentTarget
    const fit = fitDisplay(el.naturalWidth, el.naturalHeight)
    setDisplay(fit)
    const c = centeredPixelCrop(fit.w, fit.h, square)
    setCrop(c)
    setCompletedCrop(c)
  }

  React.useEffect(() => {
    if (!display || !useCrop) return
    const c = centeredPixelCrop(display.w, display.h, square)
    setCrop(c)
    setCompletedCrop(c)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [square])

  React.useEffect(() => {
    let cancelled = false
    const run = async () => {
      const img = imgRef.current
      if (!model || !src || !img || !img.complete || !img.naturalWidth) return
      let source: PixelSource = img
      if (useCrop) {
        if (!completedCrop?.width || !completedCrop?.height) return
        const canvas = cropToCanvas(img, completedCrop)
        if (Math.min(canvas.width, canvas.height) < MIN_CROP_PX) {
          if (!cancelled) {
            setTooSmall(true)
            setPreds([])
          }
          return
        }
        source = canvas
      }
      if (!cancelled) {
        setTooSmall(false)
        setBusy(true)
      }
      try {
        const probs = await predictProba(model, source)
        if (!cancelled) setPreds(topK(probs, 3))
      } finally {
        if (!cancelled) setBusy(false)
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [model, src, useCrop, completedCrop])

  if (!src) {
    return (
      <Card>
        <CardContent className="flex h-48 items-center justify-center text-center text-muted-foreground">
          {emptyHint}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="space-y-4 p-4">
          <div className="flex items-center justify-between">
            <label htmlFor="crop" className="text-sm font-medium">
              ✂️ Crop ROI thủ công
            </label>
            <Switch id="crop" checked={useCrop} onCheckedChange={setUseCrop} />
          </div>
          {useCrop && (
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={square ? 'default' : 'outline'}
                onClick={() => setSquare(true)}
              >
                Vuông 1:1
              </Button>
              <Button
                size="sm"
                variant={!square ? 'default' : 'outline'}
                onClick={() => setSquare(false)}
              >
                Tự do
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          {useCrop ? (
            <div className="w-full">
              <p className="mb-2 text-sm text-muted-foreground">
                🖱️ Kéo khung xanh để chọn vùng biển báo.
              </p>
              <div className="flex justify-center">
                <ReactCrop
                  crop={crop}
                  onChange={(c) => setCrop(c)}
                  onComplete={(c) => setCompletedCrop(c)}
                  aspect={square ? 1 : undefined}
                  keepSelection
                >
                  <img
                    ref={imgRef}
                    src={src}
                    alt="Ảnh đầu vào"
                    crossOrigin="anonymous"
                    onLoad={onImageLoad}
                    style={{
                      display: 'block',
                      width: display ? display.w : 'auto',
                      height: display ? display.h : 'auto',
                    }}
                  />
                </ReactCrop>
              </div>
            </div>
          ) : (
            <div className="mx-auto flex aspect-square w-full max-w-md items-center justify-center overflow-hidden rounded-md bg-muted">
              <img
                ref={imgRef}
                src={src}
                alt="Ảnh đầu vào"
                crossOrigin="anonymous"
                onLoad={onImageLoad}
                className="h-full w-full object-contain"
              />
            </div>
          )}
        </CardContent>
      </Card>

      {!useCrop && (
        <p className="text-sm text-amber-600 dark:text-amber-500">
          ⚠️ Chế độ full-size: ảnh không crop → độ chính xác giảm đáng kể.
        </p>
      )}
      {useCrop && tooSmall && (
        <p className="text-sm text-amber-600 dark:text-amber-500">
          Khung crop quá nhỏ (&lt;{MIN_CROP_PX}px). Kéo khung lớn hơn để dự đoán.
        </p>
      )}

      {busy && <Skeleton className="h-48 w-full" />}
      {!busy && !tooSmall && labels && preds.length > 0 && (
        <ResultsPanel preds={preds} labels={labels} />
      )}
    </div>
  )
}
