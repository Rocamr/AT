import pandas as pd
import joblib
from fontTools.misc.cython import returns

# Cargar el modelo y el escalador
knn_model = joblib.load('acciones_knn_model.pkl')  # Asegúrate de que el modelo esté guardado como 'knn_model.pkl'
scaler = joblib.load('acciones_scaler.pkl')

def accionPP(new_data):
    # Escalar los nuevos datos usando el escalador entrenado
    new_data_scaled = scaler.transform(new_data)

    # Hacer la predicción
    predicted_price = knn_model.predict(new_data_scaled)
    return predicted_price[0]
