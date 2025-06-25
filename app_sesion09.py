import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import numpy as np

# El objetivo de esta aplicación es:
# 1. cargar un modelo generado previamente <- pickle
# 2. recuperar los datos operativos del usuario <- streamlit 
# 3. validar los datos (identificar se sean valores numéricos) <- built-in
# 4. hacer la predicción de estos valores <- model.predict
# 5. devolver al usuario, el estado operativo (Blockage, Leakage, Normal, Pump Failure <- remapeo


# --- Implementación --- #
# 1. Carga del modelo
with open('/workspaces/ml-bootcamp-labs/models/model_op.pkl', 'rb') as file:
    model = pickle.load(file)


# Función de predicción

def prediccion_datos(campos,valores):
    diccionario = dict(zip(campos,valores))
    data = pd.DataFrame(diccionario,index=[0])
    return model.predict(data)

# 2. Carga de datos

# En la lista de campos de entrada, se ponen los nombres de las columnas en el orden que las necesitamos
campos = ['Well_ID', 'Pressure', 'Temperature', 'Flow_Rate', 'Pump_Speed',
       'Gas_Oil_Ratio', 'Water_Cut', 'Vibration', 'Maintenance_Required',
       'Downtime'] 
# En la lista de valores, esta se deja vacía porque la aplicación se encarga de llenarlas
valores = []

st.markdown("Introduce los datos de operación para predecir:")

# Ciclo: Se encarga de desplegar un formulario en la aplicación
for c in campos:
    v = st.text_input(f"{c} (numérico)")
    valores.append(v)


# 4. Predicción
if st.button("Predecir!"):
    resultado = prediccion_datos(campos,valores)
    st.success(f"Status operativo: {resultado}")