import os
import mlflow
import pandas as pd

from src import ingest, preprocess, train, evaluate

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

    target_preprocessor = preprocess.preprocessing_target_pipeline([target_column])
    y = df[target_column].to_frame()
    y_processed = target_preprocessor.fit_transform(y)

    # 3. Entrenamiento
    mlflow.set_experiment('ml-pipeline')
    with mlflow.start_run():
        rf_model, X_test, y_test = train.train_RFC_classifier(X_processed,y_processed)
        # 4. Evaluación RF
        mse_rf = evaluate.evaluate_model(rf_model,X_test,y_test)

        mlflow.sklearn.log_model(rf_model,'RandomForest')

        xgb_model, X_test, y_test = train.train_XGB_classifier(X_processed,y_processed)
        # 4. Evaluación RF
        mse_xgb = evaluate.evaluate_model(xgb_model,X_test,y_test)

        mlflow.sklearn.log_model(xgb_model,'XGBoost')

        mlflow.log_metric('mse_rf',mse_rf)
        mlflow.log_metric('mse_xgb',mse_xgb)

        mlflow.log_param('numeric_features',numeric_features)
        mlflow.log_param('categorical_features',categorical_features)

    # 5. Tracking y versionado
    # Implementar lógica específica o integrar con orquestador externo

if __name__ == "__main__":
    main()
