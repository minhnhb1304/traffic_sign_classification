import * as React from 'react'
import * as tf from '@tensorflow/tfjs'
import { loadModel } from '@/lib/model'
import { loadLabels } from '@/lib/labels'
import type { Label } from '@/lib/labels'

interface UseModelResult {
  model: tf.LayersModel | null
  labels: Label[] | null
  loading: boolean
  error: string | null
  reload: () => void
}

/** Loads the TF.js model + labels once and exposes load status. */
export function useModel(): UseModelResult {
  const [model, setModel] = React.useState<tf.LayersModel | null>(null)
  const [labels, setLabels] = React.useState<Label[] | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [nonce, setNonce] = React.useState(0)

  React.useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.all([loadModel(), loadLabels()])
      .then(([m, l]) => {
        if (cancelled) return
        setModel(m)
        setLabels(l)
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : 'Không tải được model')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [nonce])

  const reload = React.useCallback(() => setNonce((n) => n + 1), [])

  return { model, labels, loading, error, reload }
}
