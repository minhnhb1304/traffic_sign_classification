/**
 * Rolling-mean smoothing buffer — mirrors Python `deque(maxlen=N)` of softmax
 * vectors with element-wise mean, then argmax. Keeps the last `window` frames.
 */
export class SmoothingBuffer {
  private buf: Float32Array[] = []
  private window: number
  private numClasses: number

  constructor(window = 5, numClasses = 43) {
    this.window = window
    this.numClasses = numClasses
  }

  setWindow(n: number) {
    this.window = Math.max(1, Math.floor(n))
    while (this.buf.length > this.window) this.buf.shift()
  }

  reset() {
    this.buf = []
  }

  /** Push a softmax vector; return the averaged {idx, conf}. */
  push(probs: Float32Array): { idx: number; conf: number } {
    this.buf.push(probs)
    if (this.buf.length > this.window) this.buf.shift()

    const avg = new Float32Array(this.numClasses)
    for (const v of this.buf) {
      for (let i = 0; i < this.numClasses; i++) avg[i] += v[i]
    }
    const n = this.buf.length
    let idx = 0
    let conf = -1
    for (let i = 0; i < this.numClasses; i++) {
      avg[i] /= n
      if (avg[i] > conf) {
        conf = avg[i]
        idx = i
      }
    }
    return { idx, conf }
  }
}
