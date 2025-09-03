# Análisis Exploratorio de Datos (EDA)

El Análisis Exploratorio de Datos (EDA, por sus siglas en inglés) es una etapa fundamental en cualquier proyecto de ciencia de datos o machine learning. Permite comprender la estructura, patrones, anomalías y relaciones en los datos antes de aplicar modelos predictivos.

---

## Opciones y Técnicas de EDA

A continuación se describen las principales técnicas y herramientas para realizar EDA, con énfasis en las que trabajamos durante el bootcamp:

### 1. **Carga y revisión inicial de los datos**
- Visualización de las primeras y últimas filas (`head()`, `tail()`)
- Revisión de dimensiones y tipos de datos (`shape`, `info()`, `dtypes`)
- Identificación de valores nulos o faltantes (`isnull().sum()`)

### 2. **Estadísticas descriptivas**
- Estadísticas básicas: media, mediana, moda, desviación estándar, percentiles (`describe()`)
- Estadísticas por grupo (`groupby()`)

### 3. **Distribución de variables**
- Histogramas y distribuciones (`hist()`, `sns.histplot()`)
- Boxplots para detectar outliers (`boxplot()`, `sns.boxplot()`)
- Diagramas de violín (`sns.violinplot()`)

### 4. **Análisis de variables categóricas**
- Conteo de frecuencias (`value_counts()`)
- Gráficos de barras (`sns.countplot()`)

### 5. **Correlación y relaciones entre variables**
- Matriz de correlación (`corr()`, `sns.heatmap()`)
- Gráficos de dispersión (`scatterplot()`, `pairplot()`)

### 6. **Detección de valores atípicos (outliers)**
- Boxplots
- Z-score, IQR

### 7. **Análisis de valores faltantes**
- Visualización de nulos (`sns.heatmap(df.isnull())`)
- Estrategias de imputación

### 8. **Análisis temporal (si aplica)**
- Series de tiempo: tendencias, estacionalidad, autocorrelación

### 9. **Análisis multivariado**
- Gráficos de pares (`sns.pairplot()`)
- Análisis de componentes principales (PCA)

---

## Énfasis: Técnicas Realizadas en el Bootcamp

Durante el bootcamp, pusimos especial énfasis en:

- **Carga y revisión inicial de los datos**: Siempre comenzamos con `df.head()`, `df.info()` y `df.describe()` para entender la estructura y calidad de los datos.
- **Análisis de valores nulos**: Identificamos y tratamos valores faltantes usando imputación o eliminación de filas/columnas.
- **Distribución de variables**: Utilizamos histogramas y boxplots para visualizar la distribución y detectar outliers.
- **Análisis de variables categóricas**: Aplicamos `value_counts()` y gráficos de barras para entender la frecuencia de cada categoría.
- **Correlación entre variables**: Calculamos y visualizamos la matriz de correlación para identificar relaciones lineales y posibles multicolinealidades.
- **Visualización**: Usamos `matplotlib` y `seaborn` para crear gráficos claros y efectivos.
- **Documentación de hallazgos**: Anotamos observaciones clave sobre la calidad de los datos, posibles problemas y oportunidades de mejora.

---

## Ejemplo de Código EDA

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/Operational_events.csv')

# Revisión inicial
print(df.head())
print(df.info())
print(df.describe())

# Valores nulos
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Distribución de variables numéricas
df.hist(bins=30, figsize=(10,8))
plt.show()

# Boxplot para detectar outliers
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.show()

# Variables categóricas
for col in df.select_dtypes(include=['object']):
    print(df[col].value_counts())
    sns.countplot(x=col, data=df)
    plt.show()

# Correlación
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

---

## Herramientas recomendadas

- **Pandas Profiling / ydata-profiling**: Genera reportes automáticos de EDA.
- **Sweetviz**: Alternativa visual para análisis exploratorio.
- **Seaborn y Matplotlib**: Para visualizaciones personalizadas.

---

## Conclusión

El EDA es un paso esencial para asegurar la calidad y el éxito de cualquier proyecto de machine learning. Una buena exploración permite tomar mejores decisiones de preprocesamiento, modelado y validación.

---
