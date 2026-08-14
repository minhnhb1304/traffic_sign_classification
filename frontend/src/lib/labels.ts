import { assetUrl } from './utils'

export interface Label {
  idx?: number
  code?: string
  qcvn_code?: string
  name_en: string
  name_vi: string
  folder: string
}

export type Lang = 'vi' | 'en'

let _labels: Label[] | null = null
let _loading: Promise<Label[]> | null = null

/** Fetch and cache labels.json or labels_vn.json, sorted by folder/idx. */
export async function loadLabels(): Promise<Label[]> {
  if (_labels) return _labels
  if (!_loading) {
    _loading = (async () => {
      const modelType = localStorage.getItem('tsr-model-type') || 'gtsrb'
      const labelPath = modelType === 'vn' ? 'labels_vn.json' : 'labels.json'
      
      const res = await fetch(assetUrl(labelPath))
      if (!res.ok) throw new Error(`Không tải được ${labelPath} (${res.status})`)
      const data = (await res.json()) as Label[]
      
      // VN model might not have idx, so sort by folder
      data.sort((a, b) => a.folder.localeCompare(b.folder))
      _labels = data
      return data
    })()
  }
  return _loading
}

export const displayName = (l: Label, lang: Lang): string => {
  if (lang === 'vi') return l.name_vi || l.qcvn_code || l.folder
  return l.name_en || l.qcvn_code || l.folder
}

export function disposeLabels(): void {
  _labels = null
  _loading = null
}

/** Label shown when confidence is below the detection threshold. */
export const NOT_DETECTED = '— không phát hiện —'
