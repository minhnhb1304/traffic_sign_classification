import { AlertTriangle } from 'lucide-react'
import type { CameraError } from '@/hooks/useCamera'

const MESSAGES: Record<CameraError, string> = {
  denied: 'Bạn đã từ chối quyền truy cập camera. Hãy cấp quyền trong cài đặt trình duyệt rồi thử lại.',
  notfound: 'Không tìm thấy camera nào trên thiết bị này.',
  unsupported: 'Trình duyệt không hỗ trợ truy cập camera (getUserMedia).',
  unknown: 'Không mở được camera. Vui lòng thử lại.',
}

export function CameraErrorAlert({ error }: { error: CameraError }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <p>{MESSAGES[error]}</p>
    </div>
  )
}
