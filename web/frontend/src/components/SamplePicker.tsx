import { assetUrl } from '@/lib/utils'

/** Demo images copied from demo_images/tier1_gtsrb into public/samples. */
const SAMPLES = [
  '01_stop.png',
  '02_no_entry.png',
  '03_speed_30.png',
  '04_speed_50.png',
  '05_speed_60.png',
  '06_no_passing.png',
  '07_priority_road.png',
  '08_yield.png',
  '09_roundabout.png',
  '10_ahead_only.png',
]

interface SamplePickerProps {
  onPick: (url: string) => void
}

export function SamplePicker({ onPick }: SamplePickerProps) {
  return (
    <div>
      <p className="mb-2 text-sm font-medium text-muted-foreground">
        Hoặc chọn ảnh mẫu
      </p>
      <div className="grid grid-cols-5 gap-2 sm:grid-cols-10 md:grid-cols-5">
        {SAMPLES.map((name) => {
          const url = assetUrl(`samples/${name}`)
          return (
            <button
              key={name}
              type="button"
              onClick={() => onPick(url)}
              title={name}
              className="aspect-square overflow-hidden rounded-md border bg-muted transition-transform hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <img
                src={url}
                alt={name}
                className="h-full w-full object-cover"
                loading="lazy"
              />
            </button>
          )
        })}
      </div>
    </div>
  )
}
