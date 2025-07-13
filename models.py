from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_model_defs(ratio):
    return {
        'LogReg': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            {'model__C': [0.1, 1, 10]}
        ),
        'LinearSVC': (
            LinearSVC(class_weight='balanced', dual=False, max_iter=5000, random_state=42),
            {'model__C': [0.1, 1, 10]}
        ),
        'RandomForest': (
            RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
            {'model__n_estimators': [200,500], 'model__max_depth':[None,10,20]}
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {'model__n_estimators':[50,100], 'model__learning_rate':[0.5,1.0]}
        ),
        'XGBoost': (
            XGBClassifier(scale_pos_weight=ratio, use_label_encoder=False, eval_metric='logloss'),
            {'model__n_estimators':[100,200], 'model__max_depth':[3,5], 'model__learning_rate':[0.05,0.1]}
        ),
        'LightGBM': (
            LGBMClassifier(class_weight='balanced', random_state=42),
            {'model__n_estimators':[100,200], 'model__num_leaves':[31,50]}
        )
    }
