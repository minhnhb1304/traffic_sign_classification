# 🚦 Traffic Sign Classification (Full-Stack ML App)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)

An end-to-end, open-source application for Traffic Sign Classification. Built from scratch using a custom Convolutional Neural Network (CNN) without relying on pre-trained models. 

This project natively supports:
- 🇩🇪 **GTSRB Dataset (German Traffic Sign Recognition Benchmark)** - 43 classes
- 🇻🇳 **Vietnam Traffic Signs** - 56 classes (via fine-tuning)

## 🌟 Key Features

- **Custom CNN Architecture**: Designed and trained from scratch on TensorFlow/Keras.
- **Dual Model Support**: Dynamically switch between the baseline GTSRB model and the fine-tuned Vietnam traffic sign model.
- **Robust Preprocessing**: Real-time augmentation, aspect-ratio preserving crops, and ROI (Region of Interest) extraction.
- **Modern Full-Stack Architecture**: 
  - **Backend**: FastAPI RESTful API for high-performance inference.
  - **Frontend**: React (Vite + TypeScript) with TailwindCSS for a sleek, responsive UI.
  - **Legacy UI**: Includes the original Streamlit application for quick prototyping.

## 🏗️ Project Architecture

```
traffic_sign_classification/
├── api/                  # FastAPI backend
├── frontend/             # React (Vite + TS) frontend
├── legacy_streamlit/     # Original Streamlit dashboard (deprecated)
├── src/                  # Core ML pipeline (data prep, model, training, evaluation)
├── models/               # Saved `.keras` weights and label JSONs
├── data/                 # Datasets (raw and processed)
├── reports/              # Metrics, confusion matrices, logs
└── demo_images/          # Sample images for testing
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Node.js 18+ (for the React frontend)
- (Optional) CUDA-enabled GPU for faster training

### 2. Backend & ML Setup (FastAPI + TensorFlow)

```bash
# Clone the repository
git clone https://github.com/your-username/traffic_sign_classification.git
cd traffic_sign_classification

# Create and activate a virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server (coming soon)
# uvicorn api.main:app --reload
```

### 3. Frontend Setup (React)

```bash
cd frontend
npm install
npm run dev
```

### 4. Running the Legacy Streamlit App

If you prefer the original Python-only dashboard:
```bash
streamlit run legacy_streamlit/streamlit_app.py
```

## 🧠 Machine Learning Pipeline

### Data Preparation
To prepare the dataset from the raw Kaggle GTSRB download:
```bash
python -m src.prepare_data --raw data/gtsrb_raw
```

### Training
Train the baseline GTSRB model locally:
```bash
python -m src.train
```

### Evaluation
Evaluate the model to generate classification reports and confusion matrices:
```bash
python -m src.evaluate
```

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get involved. Whether it's adding new traffic sign datasets (e.g., US, UK), improving the CNN architecture, or enhancing the UI, your PRs are highly appreciated.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
