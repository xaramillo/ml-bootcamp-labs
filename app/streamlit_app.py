import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import mlflow

# Cargar modelos
MODEL_DIR = "../models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xg_boost.joblib")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

st.title("Predicción de eventos operativos")

model_type = st.selectbox("Selecciona el modelo", ["RandomForest", "XGBoost"])

if model_type == "RandomForest":
    model = load_model(RF_MODEL_PATH)
    model_loaded = model is not None
    model_name = "RandomForest"
else:
    model = load_model(XGB_MODEL_PATH)
    model_loaded = model is not None
    model_name = "XGBoost"

if not model_loaded:
    st.error(f"No se encontró el modelo entrenado '{model_name}'. Por favor, ejecuta el pipeline de entrenamiento y asegúrate de que el archivo exista en '{MODEL_DIR}'.")
    st.stop()

# Cargar pipeline de preprocesamiento
from src.preprocess import preprocessing_pipeline
import yaml

# Asume que tienes los nombres de features guardados o los defines aquí
numeric_features = st.session_state.get("numeric_features", [])
categorical_features = st.session_state.get("categorical_features", [])

if not numeric_features or not categorical_features:
    # Puedes cargar los features desde un archivo o definirlos manualmente
    # Ejemplo:
    numeric_features = st.text_input("Nombres de features numéricos (coma separada)", "feature1,feature2").split(",")
    categorical_features = st.text_input("Nombres de features categóricos (coma separada)", "feature3").split(",")

preprocessor = preprocessing_pipeline(numeric_features, categorical_features)

# Entrada de usuario
st.header("Introduce los valores de entrada")
input_data = {}
for col in numeric_features:
    input_data[col] = st.number_input(f"{col}", value=0.0)
for col in categorical_features:
    input_data[col] = st.text_input(f"{col}", value="")

input_df = pd.DataFrame([input_data])

# Preprocesar y predecir
if st.button("Predecir"):
    try:
        X_proc = preprocessor.fit_transform(input_df)
        pred = model.predict(X_proc)
        st.success(f"Predicción: {pred[0]}")
        # Log de predicción en MLflow
        with mlflow.start_run(run_name="streamlit_prediction", nested=True):
            mlflow.log_params(input_data)
            mlflow.log_metric("prediction", int(pred[0]))
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
