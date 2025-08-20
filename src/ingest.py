import pandas as pd

def load_data(datos):
    """Carga datos desde una ubicación proporcionada"""
    return pd.read_csv(datos)