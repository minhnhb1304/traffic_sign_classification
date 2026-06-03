import * as React from 'react'
import { Upload } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ImageDropzoneProps {
  onFile: (file: File) => void
  className?: string
}

/** Click-or-drag image upload area. Accepts a single image file. */
export function ImageDropzone({ onFile, className }: ImageDropzoneProps) {
  const inputRef = React.useRef<HTMLInputElement | null>(null)
  const [dragging, setDragging] = React.useState(false)

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (file && file.type.startsWith('image/')) onFile(file)
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
      className={cn(
        'flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-10 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        dragging
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/60 hover:bg-accent/40',
        className,
      )}
    >
      <div className="rounded-full bg-primary/10 p-3 text-primary">
        <Upload className="h-6 w-6" />
      </div>
      <div>
        <p className="font-medium">Kéo thả ảnh vào đây hoặc bấm để chọn</p>
        <p className="text-sm text-muted-foreground">
          Hỗ trợ PNG, JPG · biển báo nằm giữa khung
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
