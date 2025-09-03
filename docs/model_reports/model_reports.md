# Reporte y Seguimiento de Modelos con MLflow

El seguimiento y reporte de modelos es una parte esencial en cualquier flujo de trabajo de machine learning profesional. Permite comparar experimentos, reproducir resultados, versionar modelos y facilitar la colaboración entre equipos. En este repositorio, utilizamos **MLflow** como herramienta principal para el tracking y gestión de modelos.

---

## ¿Qué es MLflow?

[MLflow](https://mlflow.org/) es una plataforma open source para gestionar el ciclo de vida completo de modelos de machine learning, incluyendo experimentación, reproducción, despliegue y registro de modelos.

---

## Opciones y Funcionalidades de MLflow

A continuación, se describen las principales funcionalidades de MLflow para el reporte y seguimiento de modelos, con énfasis en las que aplicamos durante el bootcamp:

### 1. **Tracking de Experimentos**

- **Registro de parámetros** (`mlflow.log_param`): Guarda los hiperparámetros utilizados en cada experimento.
- **Registro de métricas** (`mlflow.log_metric`): Guarda métricas de desempeño como accuracy, recall, f1, etc.
- **Registro de artefactos** (`mlflow.log_artifact`): Permite guardar archivos relevantes como gráficos, reportes, modelos serializados, etc.
- **Registro de modelos** (`mlflow.sklearn.log_model`, `mlflow.xgboost.log_model`): Guarda el modelo entrenado para su posterior reutilización o despliegue.

### 2. **UI de MLflow**

- **Visualización de experimentos**: Permite comparar runs, filtrar por parámetros o métricas, y visualizar artefactos.
- **Comparación de runs**: Puedes seleccionar varios experimentos y comparar sus resultados de manera visual.
- **Descarga de artefactos**: Descarga modelos, gráficos y otros archivos generados durante el entrenamiento.

### 3. **Model Registry**

- **Registro y versionado de modelos**: Permite almacenar, versionar y promover modelos a diferentes etapas (Staging, Production, Archived).
- **Despliegue directo**: MLflow permite servir modelos registrados como APIs REST.

### 4. **Integración con código**

- **Automatización**: Puedes integrar MLflow en cualquier script de Python para registrar automáticamente todos los experimentos.
- **Reproducibilidad**: Cada run guarda el código fuente, parámetros, métricas y artefactos, facilitando la reproducción de resultados.

---

## Énfasis: ¿Qué hicimos durante el bootcamp?

Durante el bootcamp, pusimos especial énfasis en:

- **Registro automático de parámetros y métricas**: Cada experimento de entrenamiento (RandomForest, XGBoost) registra hiperparámetros y métricas clave (accuracy, recall, precision, f1, confusion matrix).
- **Logging de artefactos**: Guardamos gráficos de interpretabilidad (SHAP, LIME, PDP), matrices de confusión y otros artefactos relevantes.
- **Registro y versionado de modelos**: Los modelos entrenados se registran en el Model Registry de MLflow, permitiendo su reutilización y despliegue.
- **Uso de la UI de MLflow**: Visualizamos y comparamos los resultados de diferentes experimentos directamente desde la interfaz web.
- **Reproducibilidad**: Cada run guarda el entorno de ejecución y los archivos de requerimientos, asegurando que los experimentos puedan ser replicados.

---

## Ejemplo de Código de Tracking con MLflow

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Entrenamiento del modelo
    model = RandomForestClassifier(...)
    model.fit(X_train, y_train)
    
    # Registro de parámetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Registro de métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    
    # Registro de artefactos (ejemplo: matriz de confusión)
    mlflow.log_artifact("confusion_matrix_rf.png")
    
    # Registro del modelo
    mlflow.sklearn.log_model(model, "RandomForest")
    
    # Registro en el Model Registry
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/RandomForest",
        name="RandomForestClassifier"
    )
```

---

## Buenas Prácticas

- **Registra todos los parámetros y métricas relevantes** para cada experimento.
- **Guarda artefactos visuales** (gráficos, reportes) para facilitar la interpretación.
- **Versiona tus modelos** y utiliza el Model Registry para gestionar el ciclo de vida.
- **Utiliza la UI de MLflow** para comparar y analizar resultados.
- **Incluye el archivo `requirements.txt` como artefacto** para asegurar la reproducibilidad.

---

## Recursos adicionales

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## Conclusión

El uso de MLflow facilita el seguimiento, comparación y despliegue de modelos de machine learning, asegurando trazabilidad y reproducibilidad en proyectos reales. ¡Aprovecha todas sus funcionalidades para profesionalizar tu flujo de trabajo!

---
