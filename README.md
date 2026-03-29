# Tennessee Eastman Process (TEP) Fault Detection

## 📌 Project Overview
This project features a machine learning-based fault detection system for the Tennessee Eastman Process (TEP). By analyzing sensor data, this model classifies operational states and detects anomalies to ensure chemical process safety and efficiency.

The core predictive model uses **XGBoost** to handle the complex, non-linear dynamics of the reactor system.

## 📊 Dataset
The models were trained and tested on the standard TEP simulation dataset. 
* Due to file size constraints, the raw `.RData` files are not hosted in this repository.
* **Download the raw data here:** https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset

## 🗂️ Project Files
* `ChemProject1.ipynb`: Jupyter Notebook detailing exploratory data analysis, data scaling, and XGBoost training.
* `dashboard.py`: Python script for interactive visualization of sensor data and predictions.
* `tep_xgboost_model.pkl` & `tep_scaler.pkl`: The serialized machine learning model and scaler.
* `test_sensors.csv`: A lightweight sample dataset for quick testing.

## 💻 How to Run Locally
1. Clone this repository.
2. Install the required Python libraries (`pandas`, `xgboost`, `scikit-learn`).
3. Run the dashboard locally: `python dashboard.py`
