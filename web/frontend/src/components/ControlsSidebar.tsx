import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export interface RealtimeControls {
  roiRatio: number
  threshold: number
  smoothing: number
  showTop3: boolean
}

interface ControlsSidebarProps {
  value: RealtimeControls
  onChange: (next: RealtimeControls) => void
  disabled?: boolean
}

/** Sidebar of realtime tuning controls — mirrors the Streamlit spec sliders. */
export function ControlsSidebar({
  value,
  onChange,
  disabled,
}: ControlsSidebarProps) {
  const set = <K extends keyof RealtimeControls>(
    key: K,
    v: RealtimeControls[K],
  ) => onChange({ ...value, [key]: v })

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Tùy chỉnh</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <label htmlFor="roi">Vùng ROI</label>
            <span className="tabular-nums text-muted-foreground">
              {value.roiRatio.toFixed(2)}
            </span>
          </div>
          <Slider
            id="roi"
            min={0.3}
            max={0.9}
            step={0.05}
            value={value.roiRatio}
            disabled={disabled}
            onValueChange={(v) => set('roiRatio', v)}
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <label htmlFor="thr">Ngưỡng tin cậy</label>
            <span className="tabular-nums text-muted-foreground">
              {value.threshold.toFixed(2)}
            </span>
          </div>
          <Slider
            id="thr"
            min={0.3}
            max={0.95}
            step={0.05}
            value={value.threshold}
            disabled={disabled}
            onValueChange={(v) => set('threshold', v)}
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <label htmlFor="smooth">Cửa sổ làm mượt</label>
            <span className="tabular-nums text-muted-foreground">
              {value.smoothing}
            </span>
          </div>
          <Slider
            id="smooth"
            min={1}
            max={10}
            step={1}
            value={value.smoothing}
            disabled={disabled}
            onValueChange={(v) => set('smoothing', v)}
          />
        </div>

        <div className="flex items-center justify-between">
          <label htmlFor="top3" className="text-sm">
            Hiển thị top-3
          </label>
          <Switch
            id="top3"
            checked={value.showTop3}
            disabled={disabled}
            onCheckedChange={(v) => set('showTop3', v)}
          />
        </div>
      </CardContent>
    </Card>
  )
}
