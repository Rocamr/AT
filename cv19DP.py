# Importar bibliotecas necesarias
import pandas as pd
import joblib
from fontTools.misc.cython import returns
from sklearn.preprocessing import StandardScaler

# Cargar el modelo KNN entrenado
knn_model = joblib.load('cv19_model.pkl')

# Cargar el escalador
scaler = joblib.load('scaler.pkl')

def cv19PR(new_data):
    # Imputar valores faltantes en los nuevos datos (esto es opcional si tus nuevos datos no tienen faltantes)
    new_data_imputed = new_data.fillna(new_data.mean())

    # Escalar las características usando el escalador guardado
    # Asegúrate de que new_data_imputed tenga el mismo número de columnas que el escalador espera
    new_data_scaled = scaler.transform(new_data_imputed)

    # Hacer la predicción
    predicted_recovered = knn_model.predict(new_data_scaled)

    return predicted_recovered[0]
