import os
import mlflow
import pandas as pd
import numpy as np
import time
import psutil
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src import ingest, preprocess, train, evaluate, int_exp

def log_confusion_matrix(y_true, y_pred, labels, name):
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {name}')
    plt.tight_layout()
    fname = f'confusion_matrix_{name}.png'
    plt.savefig(fname)
    plt.close()
    mlflow.log_artifact(fname)
    os.remove(fname)

def main():
    # 1. Ingesta de datos
    data_path = 'data/raw/Operational_events.csv'  # Ruta al archivo de datos
    target_column = 'Event_Type'

    df = ingest.load_data(data_path)

    # 2. Preprocesamiento

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_features:
        categorical_features.remove(target_column)


    preprocessor = preprocess.preprocessing_pipeline(numeric_features, categorical_features)
    X = df.drop(columns=[target_column])
    X_processed = preprocessor.fit_transform(X)
    # Get correct feature names after transformation
    feature_names = preprocessor.get_feature_names_out()

    target_preprocessor = preprocess.preprocessing_target_pipeline([target_column])
    y = df[target_column].to_frame()
    y_processed = target_preprocessor.fit_transform(y)
    y_processed = y_processed.ravel()  # <-- Flatten to 1D

    # 3. Entrenamiento
    mlflow.set_experiment('ml-pipeline')
    with mlflow.start_run() as run:
        # System metrics before training
        cpu_start = psutil.cpu_percent(interval=None)
        ram_start = psutil.virtual_memory().percent
        time_start = time.time()

        rf_model, X_test, y_test = train.train_RFC_classifier(X_processed,y_processed)
        y_pred_rf = rf_model.predict(X_test)
        # Model metrics for RF
        acc_rf = accuracy_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
        mlflow.log_metric('rf_accuracy', acc_rf)
        mlflow.log_metric('rf_recall', recall_rf)
        mlflow.log_metric('rf_precision', precision_rf)
        mlflow.log_metric('rf_f1', f1_rf)
        # Log per-class metrics
        for label, scores in report_rf.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    mlflow.log_metric(f'rf_{label}_{metric}', value)
        mlflow.sklearn.log_model(rf_model,'RandomForest')
        log_confusion_matrix(y_test, y_pred_rf, labels=np.unique(y_test), name='RandomForest')

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/RandomForest",
            name="RandomForestClassifier"
        )

        xgb_model, X_test, y_test = train.train_XGB_classifier(X_processed,y_processed)
        y_pred_xgb = xgb_model.predict(X_test)
        # Model metrics for XGB
        # 4. Evaluación del modelo
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
        precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
        f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
        report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True, zero_division=0)
        mlflow.log_metric('xgb_accuracy', acc_xgb)
        mlflow.log_metric('xgb_recall', recall_xgb)
        mlflow.log_metric('xgb_precision', precision_xgb)
        mlflow.log_metric('xgb_f1', f1_xgb)
        for label, scores in report_xgb.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    mlflow.log_metric(f'xgb_{label}_{metric}', value)
        mlflow.sklearn.log_model(xgb_model,'XGBoost')
        log_confusion_matrix(y_test, y_pred_xgb, labels=np.unique(y_test), name='XGBoost')

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/XGBoost",
            name="XGBoostClassifier"
        )

        # Convert X_processed and X_test to DataFrame for SHAP/LIME compatibility
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        class_names = [str(c) for c in np.unique(y_processed)]

        # --- Interpretability for RandomForest ---
        shap_paths_rf, idx_rf, top_feature_rf = int_exp.explain_shap(
            rf_model, X_processed_df, X_test_df, feature_names, model_name="RandomForest"
        )
        for path in shap_paths_rf:
            mlflow.log_artifact(path)
        lime_path_rf = int_exp.explain_lime(
            rf_model, X_processed_df, X_test_df, feature_names, class_names, idx_rf, model_name="RandomForest"
        )
        mlflow.log_artifact(lime_path_rf)
        # For PDP, use top 2 features by SHAP importance
        importances_rf = np.abs(rf_model.feature_importances_)
        top2_rf = np.argsort(importances_rf)[-2:]
        pdp_path_rf = int_exp.explain_pdp(
            rf_model, X_test_df, feature_names, top2_rf, model_name="RandomForest"
        )
        mlflow.log_artifact(pdp_path_rf)

        # --- Interpretability for XGBoost ---
        shap_paths_xgb, idx_xgb, top_feature_xgb = int_exp.explain_shap(
            xgb_model, X_processed_df, X_test_df, feature_names, model_name="XGBoost"
        )
        for path in shap_paths_xgb:
            mlflow.log_artifact(path)
        lime_path_xgb = int_exp.explain_lime(
            xgb_model, X_processed_df, X_test_df, feature_names, class_names, idx_xgb, model_name="XGBoost"
        )
        mlflow.log_artifact(lime_path_xgb)
        # For PDP, use top 2 features by SHAP importance
        shap_values_xgb = shap.TreeExplainer(xgb_model).shap_values(X_test_df)
        importances_xgb = np.abs(shap_values_xgb).mean(axis=0)
        top2_xgb = np.argsort(importances_xgb)[-2:]
        pdp_path_xgb = int_exp.explain_pdp(
            xgb_model, X_test_df, feature_names, top2_xgb, model_name="XGBoost"
        )
        mlflow.log_artifact(pdp_path_xgb)

        # System metrics after training
        cpu_end = psutil.cpu_percent(interval=None)
        ram_end = psutil.virtual_memory().percent
        time_end = time.time()
        mlflow.log_metric('cpu_percent_start', cpu_start)
        mlflow.log_metric('cpu_percent_end', cpu_end)
        mlflow.log_metric('ram_percent_start', ram_start)
        mlflow.log_metric('ram_percent_end', ram_end)
        mlflow.log_metric('training_duration_seconds', time_end - time_start)

        mlflow.log_param('numeric_features',numeric_features)
        mlflow.log_param('categorical_features',categorical_features)

        # Log additional useful artifacts and environment
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")
        # Optionally log sample data or preprocessor
        # pd.DataFrame(X_processed).to_csv("X_processed_sample.csv", index=False)
        # mlflow.log_artifact("X_processed_sample.csv")

    # 5. Tracking y versionado
    # Implementar lógica específica o integrar con orquestador externo

if __name__ == "__main__":
    main()
    # Implementar lógica específica o integrar con orquestador externo