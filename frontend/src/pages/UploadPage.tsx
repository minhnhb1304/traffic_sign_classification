import * as React from 'react'
import { ImageDropzone } from '@/components/ImageDropzone'
import { SamplePicker } from '@/components/SamplePicker'
import { CropPredictPanel } from '@/components/CropPredictPanel'
import { ModelStatusBadge } from '@/components/ModelStatusBadge'
import { useModel } from '@/hooks/useModel'
import { activeBackend } from '@/lib/model'

export function UploadPage() {
  const { model, labels, loading, error } = useModel()
  const [imgUrl, setImgUrl] = React.useState<string | null>(null)
  const objectUrlRef = React.useRef<string | null>(null)

  const setObjectUrl = (file: File) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(file)
    objectUrlRef.current = url
    setImgUrl(url)
  }

  React.useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [])

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Nhận diện từ ảnh</h1>
          <ModelStatusBadge
            loading={loading}
            error={error}
            backend={activeBackend()}
          />
        </div>

        <ImageDropzone onFile={setObjectUrl} />
        <SamplePicker
          onPick={(url) => {
            if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
            objectUrlRef.current = null
            setImgUrl(url)
          }}
        />
      </div>

      <div>
        <CropPredictPanel
          src={imgUrl}
          model={model}
          labels={labels}
          emptyHint="Chọn hoặc kéo thả một ảnh để xem kết quả dự đoán."
        />
      </div>
    </div>
  )
}
