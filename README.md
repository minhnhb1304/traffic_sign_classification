# Traffic sign classification with a from-scratch CNN

A convolutional network for traffic sign classification, trained from scratch — no pretrained
backbone — on **GTSRB**. Ships as a reproducible training pipeline, three exported model formats,
and two working demos including one that runs entirely in the browser.

| | |
|---|---|
| **Model** | 3-block CNN, 48×48 RGB input, 43 GTSRB classes, ~334K params |
| **Accuracy** | **97.09%** top-1 · 99.34% top-3 on the 12,630-image GTSRB test set |
| **Exports** | Keras `.keras` · TFLite fp16 · TensorFlow.js |
| **Demos** | React + TF.js (fully client-side, no server) · Streamlit (upload + webcam) |

## Results

Evaluated on the official GTSRB test split (12,630 images, unseen during training).

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.9709 |
| Top-3 accuracy | 0.9934 |
| Test loss | 0.1108 |
| Macro F1 | 0.9617 |
| Weighted F1 | 0.9711 |

Per-class precision/recall: [`reports/classification_report.txt`](reports/classification_report.txt).
Confusion matrix and learning curves: [`reports/figures/`](reports/figures/).

The weakest classes are the visually near-identical speed limits (60 km/h recall 0.83, confused
with 80 km/h) and `End of no-passing for vehicles over 3.5t` (F1 0.81) — both low-support classes
where GTSRB itself is imbalanced.

## Quickstart

```bash
pip install -r requirements.txt
```

**Browser demo** — inference runs client-side via TensorFlow.js, no Python needed:

```bash
cd web/frontend && npm install && npm run dev
```

**Streamlit app** — image upload with interactive ROI cropping, plus a live webcam mode:

```bash
streamlit run app/streamlit_app.py     # http://localhost:8501
```

**Single image from the CLI:**

```bash
python -m src.predict --image path/to/sign.jpg
```

## Reproduce the training run

GTSRB ships in Kaggle format (`Train/`, `Test/`, `Train.csv`, `Test.csv`). Step 1 crops each sign
to its CSV bounding box and rebuilds the data as a classification set.

```bash
python -m src.prepare_data --raw data/gtsrb_raw   # → data/processed/{train,val,test}/
python -m src.train                               # → models/custom_cnn_v1.keras
python -m src.evaluate                            # → reports/
```

No GPU? [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) runs the whole pipeline
end-to-end on a Colab T4 and downloads the artifacts.

## Model

Three convolutional blocks (32 → 64 → 128 filters). Each block:
`Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool → Dropout`.
Head: `GlobalAvgPool → Dense(256) → BN → Dropout(0.5) → Dense(43, softmax)`.

| Hyperparameter | Value |
|---|---|
| Input | 48×48 RGB |
| Batch size | 128 |
| Epochs | 60, EarlyStopping patience 12 |
| Optimizer | Adam, lr 1e-3, ReduceLROnPlateau patience 3 |
| Loss | Sparse categorical cross-entropy |
| Augmentation | rotation ±12°, zoom ±10%, translate ±6%, brightness/contrast/saturation |

Defined in [`src/model.py`](src/model.py); tunable in [`src/config.py`](src/config.py).

## Vietnamese signs — in progress

GTSRB covers German signage. Adapting the model to Vietnamese signs uses a **third-party
dataset** (credited below), converted to 5,400 crops across 52 classes — bounding box plus 5%
padding, matching the GTSRB pipeline. The crops are not committed; rebuild them or supply your own.

**No Vietnamese model is trained yet.** [`notebooks/colab_finetune_vn.ipynb`](notebooks/colab_finetune_vn.ipynb)
implements two-stage transfer from `custom_cnn_v1` and is ready to run:

| Stage | Trainable | LR | Purpose |
|---|---|---|---|
| 1 | New Dense head only | 1e-3 | Fit the 52-class head without disturbing learned features |
| 2 | `block3` + head | 1e-4 | Adapt high-level features to Vietnamese signage |

The result worth reporting is the ablation in the notebook: GTSRB-only vs fine-tuned on the same
Vietnamese test set, which measures what the German-to-Vietnamese domain shift actually costs.
Numbers go here once the run has happened.

## Repository layout

```
src/          config, data prep, training, evaluation, inference
app/          Streamlit app — upload tab + realtime/ webcam mode (streamlit-webrtc)
web/frontend/ React 19 + Vite + TF.js browser demo
models/       custom_cnn_v1.keras, custom_cnn_v1_fp16.tflite, labels.json
data/         class name lists (en/vi), processed/ (GTSRB), processed_vn/ (VN crops, untracked)
notebooks/    Colab training and VN fine-tuning
scripts/      TFLite conversion, dataset builders, diagnostics
reports/      metrics, classification report, figures
demo_images/  sample images in three tiers, from in-distribution to known failures
```

## Status

- [x] GTSRB pipeline: prepare → train → evaluate, 97.09% test accuracy
- [x] Exports: Keras, TFLite fp16, TF.js
- [x] Streamlit demo with ROI cropping and webcam inference
- [x] React + TF.js in-browser demo
- [ ] Fine-tune on Vietnamese signs and publish the GTSRB-vs-fine-tuned comparison
- [ ] Transfer-learning baselines (VGG16 / ResNet50) for comparison
- [ ] Expand test coverage beyond `tests/test_overlay.py`

## License

Repository **code** is MIT — see [`LICENSE`](LICENSE).

Vietnamese sign data is **not** covered by that licence:

- The source dataset is **CC BY-SA 4.0**, a copyleft licence — derivatives carry the same terms.
- The cropped set in `data/processed_vn/` is a derivative. It is not committed here; if you
  redistribute it, it must be CC BY-SA 4.0 with attribution to the authors below.
- Whether trained weights count as a derivative of training data is legally unsettled. Any
  Vietnamese model published here will be released CC BY-SA 4.0 as the conservative reading.

## Acknowledgements

- **GTSRB** — Institut für Neuroinformatik, Ruhr-Universität Bochum
- **[Vietnam Traffic Signs](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs)** —
  Mai Tam, Y Cao Lâm, DangKhoi and Thanh Hiep Vo, CC BY-SA 4.0. The basis for the Vietnamese
  fine-tuning; thanks to the authors for publishing it openly.
