import * as React from 'react'
import { Camera, CameraOff, Aperture } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CameraStage } from '@/components/CameraStage'
import { CameraErrorAlert } from '@/components/CameraErrorAlert'
import { CropPredictPanel } from '@/components/CropPredictPanel'
import { ModelStatusBadge } from '@/components/ModelStatusBadge'
import { useCamera } from '@/hooks/useCamera'
import { useModel } from '@/hooks/useModel'
import { activeBackend } from '@/lib/model'

const ROI_RATIO = 0.6

export function SnapshotPage() {
  const { model, labels, loading, error } = useModel()
  const { videoRef, active, error: camError, start, stop } = useCamera()
  const [shotUrl, setShotUrl] = React.useState<string | null>(null)

  const capture = React.useCallback(() => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')!.drawImage(video, 0, 0)
    setShotUrl(canvas.toDataURL('image/png'))
  }, [videoRef])

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Chụp ảnh từ camera</h1>
          <ModelStatusBadge
            loading={loading}
            error={error}
            backend={activeBackend()}
          />
        </div>

        <CameraStage videoRef={videoRef} ratio={ROI_RATIO} active={active} />
        {camError && <CameraErrorAlert error={camError} />}

        <div className="flex flex-wrap gap-3">
          {!active ? (
            <Button onClick={start}>
              <Camera className="h-4 w-4" />
              Bật camera
            </Button>
          ) : (
            <Button variant="outline" onClick={stop}>
              <CameraOff className="h-4 w-4" />
              Tắt camera
            </Button>
          )}
          <Button onClick={capture} disabled={!active}>
            <Aperture className="h-4 w-4" />
            Chụp ảnh
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          Bấm “Chụp ảnh” để bắt khung hình, rồi kéo khung xanh chọn vùng biển báo.
        </p>
      </div>

      <div>
        <CropPredictPanel
          src={shotUrl}
          model={model}
          labels={labels}
          emptyHint="Bật camera và chụp một khung hình để nhận diện."
        />
      </div>
    </div>
  )
}
