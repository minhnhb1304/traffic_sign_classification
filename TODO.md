# Technical Debt & Backlog

This document tracks items identified during the codebase review that were deferred for later discussion and implementation.

## 1. Full UI Internationalization (i18n)
- **Current State:** The React frontend currently hardcodes Vietnamese strings (e.g., "Tải ảnh", "Thời gian thực") in UI components. The `LanguageProvider` is primarily used to swap dataset labels.
- **Problem:** Not fully ready for an international open-source audience who might want to use the UI in English.
- **Proposed Solution:** Introduce a robust i18n library (like `react-i18next`) and extract all hardcoded Vietnamese text into translation dictionaries (JSON files).
- **Status:** Deferred to prioritize core ML and architecture stability.

## 2. Configuration Management via Argparse / YAML
- **Current State:** On Google Colab, hyperparameters (like `IMG_SIZE`, `BATCH_SIZE`) are dynamically changed by running a hacky string-replacement script (`colab_patch_v2.py`) that overwrites `src/config.py`.
- **Problem:** Modifying source files directly via Python string manipulation is an anti-pattern. It makes the system fragile and hard to test.
- **Proposed Solution:** Refactor `src/config.py` to support dynamic overrides via command-line arguments using `argparse` in the entry points (`src/train.py`, `src/prepare_data.py`). Alternatively, adopt a configuration framework like Hydra or OmegaConf.
- **Status:** Deferred because changing this would immediately break all existing Google Colab notebooks that currently rely on the patching mechanism. Needs to be carefully planned alongside a notebook migration.
