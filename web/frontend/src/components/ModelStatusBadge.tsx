import { Loader2, CheckCircle2, AlertTriangle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface ModelStatusBadgeProps {
  loading: boolean
  error: string | null
  backend?: string
}

/** Compact status pill: loading / ready (+backend) / error. */
export function ModelStatusBadge({
  loading,
  error,
  backend,
}: ModelStatusBadgeProps) {
  if (loading) {
    return (
      <Badge variant="warning" className="gap-1.5">
        <Loader2 className="h-3 w-3 animate-spin" />
        Đang tải model…
      </Badge>
    )
  }
  if (error) {
    return (
      <Badge variant="destructive" className="gap-1.5">
        <AlertTriangle className="h-3 w-3" />
        Lỗi tải model
      </Badge>
    )
  }
  return (
    <Badge variant="success" className="gap-1.5">
      <CheckCircle2 className="h-3 w-3" />
      Sẵn sàng{backend ? ` · ${backend}` : ''}
    </Badge>
  )
}
