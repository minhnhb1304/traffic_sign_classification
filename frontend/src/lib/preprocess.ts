import * as tf from '@tensorflow/tfjs'

export type PixelSource =
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageData

/**
 * pixels -> model input tensor. Mirrors Python `preprocess_single_image`:
 *   fromPixels(RGB) -> resizeBilinear [48,48] -> float32 -> /255 -> [1,48,48,3]
 */
export function toModelTensor(src: PixelSource): tf.Tensor4D {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(src, 3) // RGB, uint8
    const resized = tf.image.resizeBilinear(pixels, [48, 48]) // bilinear
    const norm = resized.toFloat().div(255) // [0,1]
    return norm.expandDims(0) as tf.Tensor4D // [1,48,48,3]
  })
}

/** Async predict: returns the length-43 softmax vector. */
export async function predictProba(
  model: tf.LayersModel,
  src: PixelSource,
): Promise<Float32Array> {
  const input = toModelTensor(src)
  const out = model.predict(input) as tf.Tensor // [1,43]
  try {
    const data = (await out.data()) as Float32Array
    return data
  } finally {
    input.dispose()
    out.dispose()
  }
}

export interface Pred {
  idx: number
  prob: number
}

/** Top-K predictions sorted by descending probability. */
export function topK(probs: Float32Array, k = 3): Pred[] {
  return Array.from(probs)
    .map((prob, idx) => ({ idx, prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k)
}
