import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ConfidenceBar } from '@/components/ConfidenceBar'
import { displayName } from '@/lib/labels'
import type { Label } from '@/lib/labels'
import { useLanguage } from '@/hooks/useLanguage'
import type { Pred } from '@/lib/preprocess'

interface ResultsPanelProps {
  preds: Pred[]
  labels: Label[]
}

function confidenceBadge(prob: number) {
  if (prob >= 0.6) return { variant: 'success' as const, text: 'Tin cậy cao' }
  if (prob >= 0.4) return { variant: 'warning' as const, text: 'Tin cậy vừa' }
  return { variant: 'destructive' as const, text: 'Tin cậy thấp' }
}

export function ResultsPanel({ preds, labels }: ResultsPanelProps) {
  const { lang } = useLanguage()
  if (preds.length === 0) return null

  const [top, ...rest] = preds
  const topLabel = labels[top.idx]
  const badge = confidenceBadge(top.prob)

  return (
    <Card className="animate-fade-in">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle>Kết quả dự đoán</CardTitle>
          <Badge variant={badge.variant}>{badge.text}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <ConfidenceBar
          label={topLabel ? displayName(topLabel, lang) : `#${top.idx}`}
          prob={top.prob}
          emphasis
        />
        {rest.length > 0 && (
          <div className="space-y-3 border-t pt-4">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Các khả năng khác
            </p>
            {rest.map((p) => {
              const l = labels[p.idx]
              return (
                <ConfidenceBar
                  key={p.idx}
                  label={l ? displayName(l, lang) : `#${p.idx}`}
                  prob={p.prob}
                />
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
