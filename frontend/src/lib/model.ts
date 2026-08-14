import * as tf from '@tensorflow/tfjs'
import { assetUrl } from './utils'

let _model: tf.LayersModel | null = null
let _loading: Promise<tf.LayersModel> | null = null

/**
 * Load the converted TF.js Layers model as a cached singleton. Prefers the
 * WebGL backend (fast) and falls back to CPU. Warms up the graph so the first
 * real inference is not slow.
 */
export async function loadModel(): Promise<tf.LayersModel> {
  if (_model) return _model
  if (!_loading) {
    _loading = (async () => {
      try {
        await tf.setBackend('webgl')
      } catch {
        await tf.setBackend('cpu')
      }
      await tf.ready()

      // Read modelType from localStorage (set by useModelType hook)
      const modelType = localStorage.getItem('tsr-model-type') || 'gtsrb'
      const modelPath = modelType === 'vn' ? 'model_vn/model.json' : 'model/model.json'

      const model = await tf.loadLayersModel(assetUrl(modelPath))
      // Warm-up pass with a zero tensor to compile shaders / allocate buffers.
      tf.tidy(() => {
        ;(model.predict(tf.zeros([1, 48, 48, 3])) as tf.Tensor).dataSync()
      })
      _model = model
      return model
    })()
  }
  return _loading
}

/** Current active TF.js backend name (e.g. 'webgl' | 'cpu'). */
export function activeBackend(): string {
  return tf.getBackend()
}

/** Dispose current model to force reload when switching model types. */
export function disposeModel(): void {
  if (_model) {
    _model.dispose()
    _model = null
  }
  _loading = null
}
