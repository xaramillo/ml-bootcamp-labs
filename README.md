# ml-bootcamp-labs

Laboratorio de Machine Learning para Bootcamp Institute - Meridian

---

## Descripción General

Este repositorio contiene un pipeline completo y modular para el desarrollo, entrenamiento, evaluación, interpretación y despliegue de modelos de machine learning, con un enfoque en buenas prácticas de ingeniería y trazabilidad de experimentos usando MLflow. Incluye scripts para preprocesamiento, entrenamiento, evaluación, interpretabilidad (SHAP, LIME, PDP), así como una aplicación interactiva de predicción con Streamlit y soporte para contenerización con Docker.

---

## Estructura del Proyecto

```
ml-bootcamp-labs/
├── app/                        # Aplicaciones y scripts interactivos
│   ├── streamlit_app.py        # App Streamlit para predicción
│   └── script_sesion12.py      # Ejemplo de entrenamiento manual
├── data/                       # Datos (no incluidos en el repo)
│   └── raw/                    # Datos crudos
├── models/                     # Modelos entrenados (guardados como .joblib)
├── src/                        # Código fuente del pipeline
│   ├── ingest.py               # Ingesta de datos
│   ├── preprocess.py           # Preprocesamiento de datos
│   ├── train.py                # Entrenamiento de modelos
│   ├── evaluate.py             # Evaluación de modelos
│   └── int_exp.py              # Interpretabilidad y explicabilidad
├── main.py                     # Pipeline principal (end-to-end)
├── requirements.txt            # Dependencias de Python
├── Dockerfile                  # Imagen Docker para reproducibilidad/despliegue
├── run_streamlit.sh            # Script para lanzar la app Streamlit
└── README.md                   # Documentación del proyecto
```

---

## Instalación y Primeros Pasos

### 1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/ml-bootcamp-labs.git
cd ml-bootcamp-labs
```

### 2. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 3. Prepara los datos

Coloca tus datos crudos (por ejemplo, `Operational_events.csv`) en `data/raw/`.

---

## Ejecución del Pipeline

### Entrenamiento y Evaluación

```bash
python main.py
```

Esto realiza:
- Ingesta y preprocesamiento de datos
- Entrenamiento de Random Forest y XGBoost (con GridSearch)
- Evaluación y logging de métricas, parámetros y artefactos en MLflow
- Interpretabilidad automática (SHAP, LIME, PDP)
- Registro y versionado de modelos

### Seguimiento de Experimentos

El pipeline utiliza [MLflow](https://mlflow.org/) para tracking y versionado. Para lanzar la UI de MLflow:

```bash
mlflow ui
```
Luego visita [http://localhost:5000](http://localhost:5000).

---

## Interpretabilidad y Explicabilidad

El pipeline genera automáticamente:
- **SHAP**: summary plots (bar y dot), force plot local, dependence plot de la feature más importante.
- **LIME**: explicación local para una instancia.
- **PDP**: gráficos de dependencia parcial para las dos features más importantes.

Todos los artefactos se registran en MLflow y pueden ser consultados desde la UI.

---

## Despliegue y Predicción con Streamlit

### 1. Entrena y guarda los modelos (`main.py` los guarda en `models/`).

### 2. Lanza la app de Streamlit localmente

```bash
streamlit run app/streamlit_app.py
```

### 3. Uso en Docker

Construye la imagen:

```bash
docker build -t ml-bootcamp-app .
```

Lanza la app:

```bash
docker run -p 8501:8501 ml-bootcamp-app bash run_streamlit.sh
```

Accede a [http://localhost:8501](http://localhost:8501).

---

## Personalización

- **Features**: Modifica los scripts de preprocesamiento para adaptar los features a tu dataset.
- **Modelos**: Puedes añadir otros modelos en `src/train.py`.
- **Interpretabilidad**: Personaliza los métodos en `src/int_exp.py`.

---

## Buenas Prácticas

- **MLflow**: Todos los experimentos, métricas, parámetros y artefactos se registran automáticamente.
- **Docker**: El entorno es reproducible y portable.
- **Modularidad**: El código está organizado por etapas del pipeline.
- **Interpretabilidad**: Incluye explicabilidad automática para facilitar la toma de decisiones.

---

## Contribuciones

¡Las contribuciones son bienvenidas! Abre un issue o pull request para sugerencias, mejoras o dudas.

---

## Licencia

Este proyecto es para fines educativos y de entrenamiento.
