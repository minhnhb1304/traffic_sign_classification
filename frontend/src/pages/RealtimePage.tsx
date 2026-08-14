import * as React from 'react'
import { Video, VideoOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CameraStage } from '@/components/CameraStage'
import { CameraErrorAlert } from '@/components/CameraErrorAlert'
import { ControlsSidebar } from '@/components/ControlsSidebar'
import type { RealtimeControls } from '@/components/ControlsSidebar'
import { ResultsPanel } from '@/components/ResultsPanel'
import { ModelStatusBadge } from '@/components/ModelStatusBadge'
import { useCamera } from '@/hooks/useCamera'
import { useModel } from '@/hooks/useModel'
import { useLanguage } from '@/hooks/useLanguage'
import { activeBackend } from '@/lib/model'
import { predictProba, topK } from '@/lib/preprocess'
import type { Pred } from '@/lib/preprocess'
import { cropRoiToCanvas } from '@/lib/capture'
import { SmoothingBuffer } from '@/lib/smoothing'
import { displayName, NOT_DETECTED } from '@/lib/labels'

const INTERVAL_MS = 120

export function RealtimePage() {
  const { model, labels, loading, error } = useModel()
  const { videoRef, active, error: camError, start, stop } = useCamera()
  const { lang } = useLanguage()
  const [controls, setControls] = React.useState<RealtimeControls>({
    roiRatio: 0.5,
    threshold: 0.6,
    smoothing: 5,
    showTop3: false,
  })
  const [preds, setPreds] = React.useState<Pred[]>([])
  const [overlay, setOverlay] = React.useState('')
  const [color, setColor] = React.useState('#16a34a')

  const bufRef = React.useRef(new SmoothingBuffer(controls.smoothing))
  const ctrlRef = React.useRef(controls)
  ctrlRef.current = controls
  React.useEffect(() => {
    bufRef.current.setWindow(controls.smoothing)
  }, [controls.smoothing])

  React.useEffect(() => {
    if (!active || !model || !labels) return
    let timer = 0
    let running = true
    const loop = async () => {
      const video = videoRef.current
      if (running && video && video.videoWidth) {
        const { canvas } = cropRoiToCanvas(
          video,
          video.videoWidth,
          video.videoHeight,
          ctrlRef.current.roiRatio,
        )
        const probs = await predictProba(model, canvas)
        const { idx, conf } = bufRef.current.push(probs)
        const detected = conf >= ctrlRef.current.threshold
        const name = labels[idx] ? displayName(labels[idx], lang) : `#${idx}`
        setOverlay(
          detected ? `${name} ${(conf * 100).toFixed(0)}%` : NOT_DETECTED,
        )
        setColor(detected ? (conf >= 0.6 ? '#16a34a' : '#f59e0b') : '#dc2626')
        setPreds(ctrlRef.current.showTop3 ? topK(probs, 3) : [])
      }
      if (running) timer = window.setTimeout(loop, INTERVAL_MS)
    }
    loop()
    return () => {
      running = false
      window.clearTimeout(timer)
    }
  }, [active, model, labels, lang, videoRef])

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_300px]">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Nhận diện thời gian thực</h1>
          <ModelStatusBadge
            loading={loading}
            error={error}
            backend={activeBackend()}
          />
        </div>

        <CameraStage
          videoRef={videoRef}
          ratio={controls.roiRatio}
          active={active}
          overlayText={active ? overlay : undefined}
          overlayColor={color}
        />
        {camError && <CameraErrorAlert error={camError} />}

        <div className="flex flex-wrap gap-3">
          {!active ? (
            <Button onClick={start} disabled={!model}>
              <Video className="h-4 w-4" />
              Bật camera
            </Button>
          ) : (
            <Button variant="outline" onClick={stop}>
              <VideoOff className="h-4 w-4" />
              Dừng
            </Button>
          )}
        </div>

        {controls.showTop3 && labels && preds.length > 0 && (
          <ResultsPanel preds={preds} labels={labels} />
        )}
      </div>

      <ControlsSidebar value={controls} onChange={setControls} />
    </div>
  )
}
