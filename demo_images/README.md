# Sample images

Hand-picked images for trying the model and for probing where it breaks. Three tiers, ordered by
how far each moves away from the GTSRB training distribution.

| Tier | Folder | Purpose | Status |
|---|---|---|---|
| 1 | `tier1_gtsrb/` | In-distribution sanity check | 10 images, all verified |
| 2 | `tier2_vn_real/` | Real Vietnamese signs the model *should* transfer to | empty — contributions welcome |
| 3 | `tier3_failure/` | Known out-of-distribution failures | empty — contributions welcome |

Try any of them:

```bash
python -m src.predict --image demo_images/tier1_gtsrb/01_stop.png
```

## Tier 1 — in-distribution

Ten pre-cropped GTSRB signs, verified top-1 correct. Mean confidence ~97%. The same ten ship in
the browser demo at `web/frontend/public/samples/`.

| File | Predicted class | Confidence |
|---|---|---|
| `01_stop.png` | Stop | 100.00% |
| `02_no_entry.png` | No entry | 100.00% |
| `03_speed_30.png` | Speed limit 30 km/h | 94.68% |
| `04_speed_50.png` | Speed limit 50 km/h | 99.44% |
| `05_speed_60.png` | Speed limit 60 km/h | 99.74% |
| `06_no_passing.png` | No passing | 99.99% |
| `07_priority_road.png` | Priority road | 100.00% |
| `08_yield.png` | Yield | 99.75% |
| `09_roundabout.png` | Roundabout mandatory | 80.35% |
| `10_ahead_only.png` | Ahead only | 100.00% |

All ten have a Vietnamese equivalent under QCVN 41:2019, which inherits the Vienna Convention
(1968) sign geometry — that's why they transfer at all.

## Tier 2 — real Vietnamese signs

Photographs of signs on Vietnamese roads whose shape and colour match a GTSRB class. These test
whether training on German data survives a change of country.

This folder is empty. Images sourced from a web search generally carry unclear reuse rights, so
nothing is committed here. Use photographs you took yourself, or images under a licence that
permits redistribution. If you add to this tier:

1. The sign should fill >60% of the frame, in focus, evenly lit.
2. Crop close to the sign border, leaving ~10% padding on each side.
3. Save as PNG, 128–256 px on the long edge.
4. Name it `vn_<nn>_<class>.png`, e.g. `vn_01_stop.png`.
5. Verify before committing — accept only top-1 correct at confidence ≥60%.

Classes that transfer reliably: Stop, No entry (P.102), No passing (P.125), Speed limit 50,
Roundabout (R.303).

## Tier 3 — known failures

Deliberate out-of-distribution cases. These are as informative as the successes: they mark the
boundary of what a GTSRB-trained classifier can be trusted with.

This folder is empty. Three cases worth adding, each failing for a different reason:

| Suggested file | Sign | Why it fails |
|---|---|---|
| `fail_01_warning_yellow.png` | Children crossing (W.225) | Vietnamese warning signs are **yellow**; GTSRB triangles are white |
| `fail_02_expressway.png` | Blue rectangular guide sign | GTSRB has no blue rectangular guide class — out of distribution entirely |
| `fail_03_speed_40.png` | Speed limit 40 km/h | No 40 km/h class exists in GTSRB; predicts 30 or 50 with mid confidence |

All three motivate the same fix: fine-tuning on Vietnamese data. See the Vietnamese signs
section of the [project README](../README.md) and `notebooks/colab_finetune_vn.ipynb`.

## Related

- [Project README](../README.md) — results and training pipeline
