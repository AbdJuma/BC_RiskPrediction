# train.py

import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocess import load_and_clean, build_preprocessor
from models import get_model_defs

def run_training(data_path):
    # 1) Load & clean
    df = load_and_clean(data_path)

    # 2) Define feature lists
    num_feats = ['BMI','Person_Age','Menarch_Age','Menopause_Age','Month','Weekday','Risk_Score']
    cat_feats = ['Medical Ccenter','MM_D_density','Breast_Cancer_History','Got_Pregnant','Marital_Status']

    # 3) Split into X/y
    X = df[num_feats + cat_feats]
    y = df['Has_Breast_Cancer']

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5) Compute imbalance ratio
    neg, pos = np.bincount(y_train)
    ratio = neg / pos

    # 6) Get our models + grids
    model_defs = get_model_defs(ratio)

    # 7) Cross‐validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 8) Run grid‐search on each
    results = []
    for name, (estimator, params) in model_defs.items():
        pipe = Pipeline([
            ('pre', build_preprocessor(num_feats, cat_feats)),
            ('model', estimator)
        ])
        grid = GridSearchCV(pipe, params, cv=cv, scoring='f1', n_jobs=-1)

        start = time.time()
        grid.fit(X_train, y_train)
        duration = time.time() - start

        # 9) Evaluate on test set
        y_pred = grid.predict(X_test)
        try:
            proba = grid.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, proba)
        except AttributeError:
            auc = np.nan

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC_AUC': auc,
            'Time': duration
        })

    return results
