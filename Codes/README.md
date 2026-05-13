# 💻 Pipeline Source Code

This directory contains the production-grade scripts responsible for the full data lifecycle—from raw satellite data extraction to transfer learning model training.

## 🔄 Pipeline Overview

The project follows a linear sequence of operations to ensure data integrity and feature consistency:

1.  **Extraction**: Retrieve high-resolution meteorological data.
2.  **Aggregation**: Transform daily parameters into meaningful crop-cycle features.
3.  **Modeling**: Execute the transfer learning logic and evaluate performance.

---

## 📜 Script Documentation

### 🛰️ `VM_DataExtractionNasa_v3.py`
- **Purpose**: Automated interface for the NASA POWER API.
- **Key Features**:
  - Leverages `xarray` for multi-dimensional data handling.
  - Utilizes `s3fs` for direct streaming from NASA's AWS S3 storage.
  - Fetches parameters: T2M (Temp), PRECTOTCORR (Precipitation), ALLSKY_SFC_SW_DWN (Solar Radiation), etc.

### 🌡️ `VM_WeatherAggregator.py`
- **Purpose**: Feature engineering and temporal aggregation.
- **Key Features**:
  - Maps daily weather data to specific crop growing seasons.
  - Calculates cumulative rainfall, average temperatures, and extreme weather indices.
  - Outputs a structured dataset ready for machine learning.

### 🧠 `VM_TransferLearning.py`
- **Purpose**: Core machine learning engine.
- **Key Features**:
  - **Hybrid Modeling**: Support for Deep Neural Networks (DNN) and XGBoost.
  - **Transfer Logic**: Implements "Hot Start" configurations where weights from a source crop (Soybeans) are transferred and fine-tuned for the target crop (Rice).
  - **Optimization**: Integrated with **Optuna** for automated Bayesian hyperparameter search.
  - **LOYO Validation**: Runs leave-one-year-out validation after the random split benchmark to test each target-crop year against models trained on the remaining years.
  - **Statistics**: Generates error metrics (NRMSE/RMSE) and p-values for model comparison.

---
> [!NOTE]
> All scripts are configured for modular execution. Ensure paths in the scripts match your local environment before running.
