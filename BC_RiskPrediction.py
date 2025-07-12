import pandas as pd
import numpy as np
import time
from packaging import version
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ---------------------------------------------------
# Breast Cancer Prediction Pipeline
# Target: Has_Breast_Cancer
# Classical Machine Learning Approach
# ---------------------------------------------------
# Approach:
# 1) Related Work: We apply six classical ML algorithms (Logistic Regression, Linear SVC,
#    Random Forest, AdaBoost, XGBoost, LightGBM) as baseline methods.
# 2) Feature Engineering: Numeric features are imputed and scaled; categorical features are one-hot encoded.
# 3) Algorithms: We tune linear and non-linear methods to balance precision/recall on imbalanced data.
#
# Experiment Setup:
# - 80/20 stratified train/test split
# - Numeric median imputation + scaling; one-hot encoding for categories
# - 5-fold stratified CV optimizing F1
# - Imbalance handled via class_weight or scale_pos_weight
#
# Steps:
# 1) Load & clean data
# 2) Feature engineering
# 3) Risk scoring
# 4) Train/test split
# 5) Preprocessing pipelines
# 6) Hyperparameter tuning
# 7) Model evaluation & comparison
# 8) Probability calibration & threshold optimization


def main():
    # 1) Load Data
    df = pd.read_csv('dhis.csv')

    # 2) Cleaning & Feature Engineering
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['Person_Age'] = pd.to_numeric(df['Person_Age'], errors='coerce')
    df['Menarch_Age'] = pd.to_numeric(df['Menarch_Age'], errors='coerce')
    df['Menopause_Age'] = pd.to_numeric(df['Menopause_Age'], errors='coerce')
    for col in ['Used_Contraceptives', 'Chronic_Diseases']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'], errors='coerce')
    df['Month'] = df['Appointment_Date'].dt.month.fillna(0).astype(int)
    df['Weekday'] = df['Appointment_Date'].dt.weekday.fillna(0).astype(int)
    df.dropna(subset=['BMI', 'Person_Age', 'Menarch_Age'], inplace=True)

    # 3) Risk Score
    df['Risk_Score'] = df[['Used_Contraceptives', 'Chronic_Diseases']].sum(axis=1)

    # 4) Features & Target
    num_feats = ['BMI','Person_Age','Menarch_Age','Menopause_Age','Month','Weekday','Risk_Score']
    cat_feats = ['Medical Ccenter','MM_D_density','Breast_Cancer_History','Got_Pregnant','Marital_Status']
    X = df[num_feats + cat_feats]
    y = df['Has_Breast_Cancer']

    # 5) Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6) Preprocessing Pipeline
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=(version.parse(sklearn.__version__) >= version.parse('1.2'))
    )
    preproc = ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ])

    # 7) Compute imbalance ratio
    neg, pos = np.bincount(y_train)
    ratio = neg / pos

    # 8) Define models & hyperparameter grids
    model_defs = {
        'LogReg': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            {'model__C': [0.1, 1.0, 10.0]}
        ),
        'LinearSVC': (
            LinearSVC(class_weight='balanced', max_iter=5000, dual=False, random_state=42),
            {'model__C': [0.1, 1.0, 10.0]}
        ),
        'RandomForest': (
            RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
            {
                'model__n_estimators': [200, 500, 1000],
                'model__max_depth': [None, 10, 20, 30],
                'model__max_features': ['sqrt', 'log2'],
                'model__min_samples_split': [2, 5, 10]
            }
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {'model__n_estimators': [50, 100], 'model__learning_rate': [0.5, 1.0]}
        ),
        'XGBoost': (
            XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss', random_state=42),
            {'model__n_estimators': [100, 200], 'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1]}
        ),
        'LightGBM': (
            LGBMClassifier(class_weight='balanced', random_state=42),
            {'model__n_estimators': [100, 200], 'model__num_leaves': [31, 50]}
        )
    }

    # 9) Grid search & evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_f1, best_name, best_grid = -1, None, None

    for name, (estimator, params) in model_defs.items():
        pipe = Pipeline([('preprocessor', preproc), ('model', estimator)])
        grid = GridSearchCV(pipe, params, cv=cv, scoring='f1', n_jobs=-1)
        start = time.time()
        grid.fit(X_train, y_train)
        duration = time.time() - start

        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:,1])
        except Exception:
            auc = None

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC_AUC': auc,
            'Time(s)': duration,
            'Params': grid.best_params_
        })
        print(f"{name}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, AUC={auc if auc else 'NA'}, Time={duration:.1f}s")
        if f1 > best_f1:
            best_f1, best_name, best_grid = f1, name, grid

    # 10) Summary
    perf_df = pd.DataFrame(results).sort_values('F1', ascending=False)
    print("\nOverall Model Comparison:")
    print(perf_df[['Model','Accuracy','Precision','Recall','F1','ROC_AUC','Time(s)']].to_string(index=False))

    # 11) Step 2: Probability Calibration & Threshold Optimization
    best_pipe = best_grid.best_estimator_
    if not hasattr(best_pipe, 'predict_proba'):
        best_pipe = CalibratedClassifierCV(best_pipe, cv=3, method='sigmoid')
        best_pipe.fit(X_train, y_train)
    proba = best_pipe.predict_proba(X_test)[:, 1]
    thrs = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_test, (proba >= thr).astype(int)) for thr in thrs]
    thr_opt = thrs[np.argmax(f1s)]
    print(f"\nOptimal threshold for {best_name}: {thr_opt:.2f}, F1={max(f1s):.3f}")
    y_thr = (proba >= thr_opt).astype(int)
    print("Confusion Matrix at threshold:\n", confusion_matrix(y_test, y_thr))
    print("Classification Report:\n", classification_report(y_test, y_thr, zero_division=0))

if __name__ == '__main__':
    main()
