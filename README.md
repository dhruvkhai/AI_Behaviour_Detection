# Cow Behavior Detection System 🐄

A comprehensive AI-powered system for monitoring cow health and behavior using wearable sensor data (IMU/Acceleration).

## 🚀 Overview

This project implements a multi-stage behavior detection pipeline:
1.  **Anomaly Detection (Isolation Forest)**: Identifies unusual or potentially unhealthy behavior patterns (e.g., illness, stress) without requiring labels.
2.  **Behavior Classification (CNN-BiLSTM & XGBoost)**: Classifies specific behaviors like Eating, Ruminating, Walking, and Lethargy.
3.  **Rule Engine**: Converts ML predictions into actionable alerts in the system.
4.  **Post-Processing**: Uses majority voting and temporal smoothing to ensure stable and reliable output.

## 🛠 Model Architectures

### 1. 1D CNN + BiLSTM
- **CNN**: Extracts spatial features and frequency patterns from raw 3-axis acceleration data.
- **BiLSTM**: Captures long-term temporal dependencies in behavior sequences.
- **Best for**: Raw time-series data where high accuracy is needed.

### 2. XGBoost (Classical Baseline)
- Trained on statistical features (Mean, Std, Skewness, Kurtosis, RMS) extracted from sliding windows.
- **Best for**: Interpretable results and fast inference on lower-power devices.

### 3. Isolation Forest
- Used for unsupervised anomaly detection.
- Analyzes statistical distributions to flag data points that deviate from 'normal' behavior.

## 📦 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)

### Running with Docker
1.  **Build and Start**:
    ```bash
    docker-compose up --build
    ```
2.  The API will be available at `http://localhost:8000`.
3.  Access interactive documentation at `http://localhost:8000/docs`.

### Training Models
1.  Extract your dataset into a folder (e.g., `sensor_data_extracted`).
2.  Run the training script:
    ```bash
    python train_models.py
    ```
    *Note: The script includes feature extraction and windowing logic ready for your CSV data.*

## 📂 Project Structure
- `app/ml/`: Core ML logic (Anomaly detection, Classifiers, Pipeline).
- `app/models/`: Database and Pydantic schemas.
- `train_models.py`: Model training and feature extraction entry point.
- `Dockerfile` & `docker-compose.yml`: Containerization logic.

## 🛡 License
MIT License
