from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


"""

Pipeline (

              1. MinMaxScaler -> num
                                        -> 3. df_preprocesado
              2. LabelEncoder -> cat

)

"""

def preprocessing_target_pipeline(target_column):
    # Paso 1.
    target_preprocessing = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación de valores faltantes
            ('encode', OrdinalEncoder())
        ]
    )
    return target_preprocessing


def preprocessing_pipeline(numeric_features,categorical_features):
    # Paso 1.
    numeric_preprocessing = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),  # Imputación de valores faltantes
            ('scaler', MinMaxScaler())
        ]
    )
    # Paso 2.
    categorical_preprocessing = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación de valores faltantes
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
