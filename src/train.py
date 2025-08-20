
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import os
from joblib import dump


def save_best_model(model,output_dir,output_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump(model,os.path.join(output_dir,output_name,'.joblib'))

def train_RFC_classifier(X,y,output_dir='models/',seed=23):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2, random_state=seed, stratify=y)
    param_grid = {
    'n_estimators': [500,1000,2500],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5],
    }
    grid = GridSearchCV(
    RandomForestClassifier(random_state=23),
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
    )
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_
    save_best_model(model=best_model, output_dir=output_dir, output_name="random_forest")
    print("Mejores valores para RandomForest:")
    print(grid.best_params_)
    return best_model, X_test, y_test

def train_XGB_classifier(X,y,output_dir='models/',seed=23):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2, random_state=seed, stratify=y)
    param_grid = {
    'n_estimators': [500,1000,2500],
    'max_depth':[None,10,20],
    'learning_rate':[0.01,.1,.25],
    }
    grid = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss',random_state=23),
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
    )
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_
    save_best_model(model=best_model, output_dir=output_dir, output_name="xg_boost")
    print("Mejores valores para XGBoost:")
    print(grid.best_params_)
    return best_model, X_test, y_test
