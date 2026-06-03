import { assetUrl } from './utils'

export interface Label {
  idx: number
  code: string
  name_en: string
  name_vi: string
  folder: string
}

export type Lang = 'vi' | 'en'

let _labels: Label[] | null = null
let _loading: Promise<Label[]> | null = null

/** Fetch and cache labels.json (43 GTSRB classes), sorted by class index. */
export async function loadLabels(): Promise<Label[]> {
  if (_labels) return _labels
  if (!_loading) {
    _loading = (async () => {
      const res = await fetch(assetUrl('labels.json'))
      if (!res.ok) throw new Error(`Không tải được labels.json (${res.status})`)
      const data = (await res.json()) as Label[]
      data.sort((a, b) => a.idx - b.idx)
      _labels = data
      return data
    })()
  }
  return _loading
}

export const displayName = (l: Label, lang: Lang): string =>
  lang === 'vi' ? l.name_vi : l.name_en

/** Label shown when confidence is below the detection threshold. */
export const NOT_DETECTED = '— không phát hiện —'
