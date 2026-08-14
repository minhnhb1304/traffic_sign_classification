# Browser demo — traffic sign classifier in TensorFlow.js

The trained CNN converted to TensorFlow.js and served as a static React app. Inference runs
entirely in the browser: no backend, no upload, no Python. The model is ~1.3 MB and is fetched
once, then cached.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
```

```bash
npm run build      # → dist/, deployable to any static host
npm run preview    # serve the production build locally
```

The router is a **hash router**, so `dist/` works on GitHub Pages or any host without rewrite
rules configured.

## What's in it

Three routes, all sharing one cached model instance:

| Route | Page | What it does |
|---|---|---|
| `/` | `UploadPage` | Drop or pick an image, crop the sign, see top-k predictions with confidence bars |
| `/#/snapshot` | `SnapshotPage` | Take a still from the webcam, then classify it |
| `/#/realtime` | `RealtimePage` | Continuous classification of a centre ROI from the live camera feed |

Ten pre-verified sample signs ship in `public/samples/` — use the sample picker if you don't
have a test image to hand.

The UI is bilingual (English / Tiếng Việt) with a light-dark theme toggle.

## How inference works

`src/lib/model.ts` loads the converted Layers model as a cached singleton. It tries the **WebGL**
backend and falls back to **CPU**, then runs one warm-up pass on a zero tensor so the first real
prediction isn't slowed by shader compilation.

Each prediction resizes the cropped region to **48×48 RGB** — the same preprocessing as training —
and returns a softmax over 43 GTSRB classes. Labels come from `public/labels.json`.

> The model is a **classifier, not a detector.** It assumes the input is already cropped to a
> single sign. That's why every page has a crop step or a fixed ROI box — feeding it a whole
> street scene gives meaningless output.

## Layout

```
public/
  model/            converted TF.js model (model.json + weight shard)
  samples/          10 verified sample signs
  labels.json       43 class names
  metrics.json      accuracy figures shown in the UI
src/
  pages/            UploadPage, SnapshotPage, RealtimePage
  components/       AppShell, CropPredictPanel, CameraStage, ResultsPanel, ui/ primitives
  hooks/            useModel, useCamera, useLanguage
  lib/              model.ts (load + predict), capture.ts, labels.ts
```

## Regenerating the model

The bundled model is committed, so you only need this after retraining:

```bash
# from the project root, after python -m src.train
pip install tensorflowjs
tensorflowjs_converter --input_format=keras \
  models/custom_cnn_v1.keras web/frontend/public/model
```

Keep `public/labels.json` in sync with `models/labels.json` if the class list changes.

## Stack

React 19 · TypeScript · Vite 6 · Tailwind CSS 3 · `@tensorflow/tfjs` 4 · `react-image-crop`

## Related

- [Project README](../../README.md) — training pipeline, results, dataset
- [Streamlit app](../../app/streamlit_app.py) — the Python-side demo
