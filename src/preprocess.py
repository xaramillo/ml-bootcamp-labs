
### ---- ANTES ----

le = LabelEncoder()
df['Event_Type'] = le.fit_transform(df['Event_Type'])

# 3.3 Normalización (MinMax)

X_prescale = df.drop(columns=['Event_Type'])

scaler = MinMaxScaler()

X = scaler.fit_transform(X_prescale)
X = pd.DataFrame(X, columns=X_prescale.columns)

y = df['Event_Type']

# ----- ANTES ------

# Refactorización

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


"""

Pipeline (

              1. MinMaxScaler -> num
                                        -> 3. df_preprocesado
              2. LabelEncoder -> cat

)

"""

def preprocessing_pipeline(numeric_features,categorical_features):
    # Paso 1.
    numeric_preprocessing = Pipeline(
        [
            ('scaler', MinMaxScaler())
        ]
    )
    # Paso 2.
    categorical_preprocessing = Pipeline(
        [
            ('encode',OneHotEncoder())
        ]
    )
    # Paso 3.
    preprocessor = ColumnTransformer(
        [
            ('num',numeric_preprocessing,numeric_features),
            ('cat',categorical_preprocessing,categorical_features)
        ]
    )
    return preprocessor
