import joblib
import pandas as pd
from fontTools.misc.cython import returns

# Cargar el modelo y el escalador
knn_model = joblib.load('best_knn_model.joblib')
scaler = joblib.load('scaler.joblib')


def londonCP(new_data):
    # Escalar los nuevos datos
    new_data_scaled = scaler.transform(new_data)

    # Realizar la predicción
    predicted_value = knn_model.predict(new_data_scaled)
    return predicted_value[0]
