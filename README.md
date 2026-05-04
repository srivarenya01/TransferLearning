# 🌌 Thanos: Cross-Crop Yield Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)

Thanos is an advanced machine learning project designed to solve the challenge of **crop yield prediction in data-scarce environments**. By leveraging **Transfer Learning**, the project adapts knowledge gained from data-rich crops (e.g., Soybeans) to accurately predict yields for crops with limited historical data (e.g., Rice), integrated with high-resolution NASA POWER meteorological data.

---

## 🛠️ Repository Architecture

This repository is optimized for clarity and reproducibility, focusing on the core computational pipeline and validated results.

- [**`Codes/`**](file:///c:/Users/Sri%20Varenya/Desktop/Codes/thanos/final_repo/Codes)
  - Contains the end-to-end processing pipeline.
  - Automated data extraction from NASA AWS S3 buckets.
  - Feature engineering and weather aggregation modules.
  - Neural Network and Gradient Boosting (XGBoost) model implementations with Optuna hyperparameter optimization.

- [**`Results/`**](file:///c:/Users/Sri%20Varenya/Desktop/Codes/thanos/final_repo/Results)
  - Comprehensive performance metrics (NRMSE, RMSE).
  - Comparative analysis of "Hot Start" (Transfer) vs. "Cold Start" (Direct) training.
  - Visualizations: Error distribution density plots, performance boxplots, and statistical validation.

---

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and initialize a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 2. Install Dependencies
The pipeline requires high-performance data processing and ML libraries:
```bash
pip install tensorflow xarray pandas scikit-learn optuna s3fs
```

### 3. Execution Flow
Navigate to the `Codes/` directory and execute the pipeline:
1. **Extraction**: `VM_DataExtractionNasa_v3.py`
2. **Aggregation**: `VM_WeatherAggregator.py`
3. **Training**: `VM_TransferLearning.py`

---

## 📊 Core Technologies

| Category | Tools |
| :--- | :--- |
| **Deep Learning** | TensorFlow, Keras |
| **Ensemble Learning** | XGBoost |
| **Hyperparameter Tuning** | Optuna |
| **Data Engineering** | Xarray, Pandas, S3FS |
| **Weather Data** | NASA POWER (Meteorology & Solar Energy) |

---
*Developed for research in Agricultural AI and Remote Sensing.*
