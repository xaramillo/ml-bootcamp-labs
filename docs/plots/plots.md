# Gráficos Útiles en Modelos de Machine Learning

La visualización es clave para entender el comportamiento de los modelos de machine learning, identificar problemas y comunicar resultados. A continuación se describen los principales tipos de gráficos utilizados en proyectos de ML, con énfasis en los que trabajamos durante el bootcamp.

---

## 1. Gráficos de Evaluación de Modelos de Clasificación

### a. **Matriz de Confusión**
- Permite visualizar el desempeño del modelo mostrando verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
- Útil para Random Forest, XGBoost y cualquier clasificador.
- **En el bootcamp:** Generamos y analizamos matrices de confusión para comparar modelos.

### b. **Curva ROC y AUC**
- Muestra la relación entre la tasa de verdaderos positivos y la tasa de falsos positivos.
- El área bajo la curva (AUC) es una métrica de desempeño global.
- **En el bootcamp:** Analizamos curvas ROC para modelos multiclase y binarios.

### c. **Curva de Precisión-Recall**
- Útil especialmente en datasets desbalanceados.
- Permite analizar el trade-off entre precisión y recall.

---

## 2. Gráficos de Importancia de Variables

### a. **Feature Importance (Importancia de Variables)**
- Muestra qué variables son más relevantes para el modelo.
- Random Forest y XGBoost permiten extraer y graficar la importancia de cada feature.
- **En el bootcamp:** Visualizamos la importancia de variables para interpretar los modelos y seleccionar features.

### b. **SHAP Summary Plot**
- Gráfico avanzado que muestra el impacto de cada variable en las predicciones usando valores SHAP.
- Incluye versiones tipo "bar" (importancia global) y "dot" (distribución de impactos).
- **En el bootcamp:** Generamos summary plots de SHAP para XGBoost y Random Forest.

### c. **LIME Explanation Plot**
- Explicación local de una predicción específica.
- Visualiza el aporte de cada feature a la predicción de una instancia.

---

## 3. Gráficos de Interpretabilidad

### a. **SHAP Force Plot**
- Explicación visual de cómo cada feature contribuye a una predicción individual.
- **En el bootcamp:** Analizamos force plots para entender decisiones de modelos complejos.

### b. **Partial Dependence Plot (PDP)**
- Muestra el efecto marginal de una o dos variables sobre la predicción.
- Útil para interpretar relaciones no lineales.
- **En el bootcamp:** Generamos PDP para las dos variables más importantes.

---

## 4. Gráficos para Series de Tiempo

### a. **Gráfico de Serie Temporal**
- Visualiza la evolución de una variable a lo largo del tiempo.
- **En el bootcamp:** Analizamos tendencias y estacionalidad en series temporales.

### b. **Descomposición de Series de Tiempo**
- Separa la serie en tendencia, estacionalidad y residuales.
- Permite entender mejor el comportamiento de la serie.

### c. **Autocorrelación y Parcial Autocorrelación**
- Gráficos ACF y PACF para identificar dependencias temporales.

---

## 5. Gráficos para Modelos de Regresión

### a. **Predicción vs Realidad (y_pred vs y_true)**
- Dispersión de valores predichos frente a valores reales.
- Permite identificar sesgos y errores sistemáticos.

### b. **Distribución de Errores (Residual Plot)**
- Visualiza los residuos del modelo para detectar patrones no capturados.

### c. **Learning Curve**
- Muestra el desempeño del modelo en función del tamaño del set de entrenamiento.

---

## Ejemplo de Código para Algunos Gráficos

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# Importancia de variables (Random Forest)
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=feature_names)
plt.title("Importancia de Variables")
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.show()
```

---

## Herramientas y Librerías

- **Matplotlib & Seaborn:** Para la mayoría de gráficos estándar.
- **SHAP:** Para interpretabilidad avanzada.
- **LIME:** Para explicaciones locales.
- **scikit-learn:** Para métricas y algunas visualizaciones.
- **PDPbox / sklearn.inspection:** Para gráficos de dependencia parcial.
- **Plotly:** Para visualizaciones interactivas.

---

## Conclusión

La visualización es esencial para validar, interpretar y comunicar los resultados de modelos de machine learning. Utiliza estos gráficos para mejorar la comprensión y la calidad de tus modelos.

---
