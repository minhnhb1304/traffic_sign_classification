import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

interface ConfidenceBarProps {
  label: string
  prob: number
  /** Larger styling for the top-1 result. */
  emphasis?: boolean
}

/** Color thresholds mirror the realtime detection palette. */
function indicatorClass(prob: number): string {
  if (prob >= 0.6) return 'bg-detect-ok'
  if (prob >= 0.4) return 'bg-detect-warn'
  return 'bg-detect-bad'
}

export function ConfidenceBar({ label, prob, emphasis }: ConfidenceBarProps) {
  const pct = prob * 100
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span
          className={cn(
            'truncate font-medium',
            emphasis ? 'text-lg' : 'text-sm text-foreground/90',
          )}
        >
          {label}
        </span>
        <span
          className={cn(
            'tabular-nums font-semibold',
            emphasis ? 'text-lg' : 'text-sm text-muted-foreground',
          )}
        >
          {pct.toFixed(1)}%
        </span>
      </div>
      <Progress
        value={pct}
        className={emphasis ? 'h-4' : 'h-2'}
        indicatorClassName={indicatorClass(prob)}
      />
    </div>
  )
}
