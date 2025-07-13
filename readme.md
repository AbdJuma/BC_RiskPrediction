# Breast Cancer Risk Prediction Pipeline

**Course:** Machine Learning  
**Author:** Abdurrahman Juma  
**Institution:** Birzeit University  

---

## 🚀 Overview

	This project builds an end-to-end classical ML pipeline to predict breast cancer risk using three national health registries from the Palestinian Ministry of Health. It emphasizes interpretability, resource efficiency, and robust handling of class imbalance.

---

## 📂 Repository Structure

BC_RiskPrediction/
├── data/
│ └── dhis.csv # Merged registry extract
├── models/ # Serialized model artifacts
├── notebooks/
│ └── EDA_and_Results.ipynb # Exploratory analysis & dashboards
├── src/
│ ├── preprocess.py # Cleaning & feature engineering
│ ├── train.py # Model training & hyperparameter search
│ ├── predict.py # Inference & threshold tuning
│ └── utils.py # Shared helpers (metrics, plotting)
├── requirements.txt # Python dependencies
└── README.md # This document


---

## 🔧 Installation

```bash
git clone https://github.com/ajumaa/BC_RiskPrediction.git
cd BC_RiskPrediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# 🏗️ Usage
## 1. Preprocess Data

	python src/preprocess.py \
	 --input data/dhis.csv \
	 --output data/processed.pkl


## 2. Train Models

	python src/train.py \
	--data data/processed.pkl \
	 --models-dir models/


## 3. Calibrate & Predict

	python src/predict.py \
	--model models/best_model.pkl \
	 --data data/processed.pkl \
	 --output predictions.csv

## 📊 Dashboards & Visualizations

	Generated in notebooks/EDA_and_Results.ipynb:

	Execution Time per Model (Line Chart)

	Sensitivity of Truly Detected (Pie Chart)

	Precision of Positive Predictions (Bar Chart)


## 🔑 Key Takeaways

	Ensembles Win
	RF, XGBoost & LightGBM capture complex BMI/age/BI-RAD interactions better than linear models.

	Balance is Key
	Class weighting and threshold tuning are essential to raise recall on the ~8% positive class.

	Speed vs. Power
	LightGBM/XGBoost train in seconds with high recall; RF runs in minutes but delivers top F1.

	Explainable AI
	Feature importance and clear cutoff logic foster clinician trust.
