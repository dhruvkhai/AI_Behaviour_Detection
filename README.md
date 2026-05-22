# Cow Behavior Detection System 🐄

A comprehensive AI-powered system for monitoring cow health and behavior using multi-modal sensor fusion (IMU, CBT, Pressure, UWB,2 THI).

## 🚀 Overview

This project implements a multi-stage behavior detection pipeline that leverages early sensor fusion and hardware-accelerated machine learning:
1.  **Sensor Fusion Pipeline**: Aligns asynchronous multi-sensor streams at 1Hz, dynamically computes rolling statistics, and gracefully degrades when sensors go offline.
2.  **Master Decision Tree (XGBoost)**: Fast, memory-efficient sensor fusion model handling missing data natively.
3.  **Deep Learning Sequence Model (1D CNN + BiLSTM)**: Accurately classifies temporal behavior sequences using 45-second sliding windows.
4.  **Anomaly Detection (Isolation Forest)**: Identifies unusual or potentially unhealthy behavior patterns without requiring labels.

## 🛠 Model Architectures

### 1. 1D CNN + BiLSTM (PyTorch)
- **Architecture**: A deep network that groups temporal inputs into 45-second overlapping sequence tensors.
- **CNN**: Extracts spatial features and frequency patterns from the fused multi-sensor data matrix.
- **BiLSTM**: Captures long-term temporal dependencies in complex behavior sequences (e.g., grazing to resting transitions).
- **Scale**: Hardware-accelerated training on CUDA (tuned for **NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM** — auto batch size 16 + mixed precision).

### 2. Master Sensor Fusion Model (XGBoost)
- **Features**: Trained on comprehensive statistical features (Mean, Std, Min, Max, Skewness, Kurtosis) extracted synchronously.
- **Memory Efficiency**: Utilizes stratified date-cow sampling for large sparse datasets (~10GB) and histogram-based tree methods (`tree_method='hist'`).
- **Robustness**: Inherently resilient to missing sensor streams (e.g., UWB disconnects) via XGBoost's optimal missing-value splitting.

## 📦 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.12.9+ (for local development)
- NVIDIA CUDA Toolkit & cuDNN (optional, for GPU training on CUDA-capable cards, e.g. RTX 3050 Ti 4 GB)

### Running with Docker (API)
This Docker configuration deploys the lightweight FastAPI backend for inference:
1.  **Build and Start**:
    ```bash
    docker-compose up --build
    ```
2.  The API will be available at `http://localhost:8000`.
3.  Access interactive API documentation at `http://localhost:8000/docs`.

### Training Models Locally

1. Place your extracted sensor CSV/Excel files into your data root mapped by `app.ml.sensor_configs`.
2. Ensure you have the required dependencies (`pip install -r requirements.txt`).

**Train the PyTorch Deep Learning Model:**
```bash
python train_dl_model.py
# Optional: more data / custom batch size for 4 GB VRAM
python train_dl_model.py --max-sessions 50 --batch-size 16
```
*Slices raw matrices into PyTorch sequence tensors and trains on your CUDA GPU (batch size auto-scales to VRAM; default 16 on 4 GB).*

**Train the XGBoost Master Fusion Model:**
```bash
python train_master_model.py --max-sessions 0 --top-features 50
```
*Uses all sessions, merges rare classes, class weights, top-50 features, macro F1 report + confusion matrix.*

**Beginner guide:** see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

**Optional — deep model + ensemble:**
```bash
python train_dl_model.py --max-sessions 50 --batch-size 16 --epochs 30
python evaluate_ensemble.py --max-sessions 0
```

## 📂 Project Structure
- `app/ml/`: Core machine learning logic (`fusion_pipeline.py`, classifiers, configs).
- `app/api/`: FastAPI endpoint routers.
- `models/`: Exported model binaries (`.joblib`, `.pth`), encoders, and performance reports.
- `train_dl_model.py`: Deep learning training pipeline.
- `train_master_model.py`: Master sensor fusion training script.
- `Dockerfile` & `docker-compose.yml`: API Containerization stack.

## 🛡 License
MIT License
