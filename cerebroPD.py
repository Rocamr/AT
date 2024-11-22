import pandas as pd
import joblib
from fontTools.misc.cython import returns

# Cargar el modelo y el escalador guardados
rf_model = joblib.load('stroke_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
def cerebroVC(new_data):
    # Escalar las características usando el escalador que fue entrenado en los datos originales
    new_data_scaled = scaler.transform(new_data)

    # Hacer la predicción con el modelo cargado
    y_pred_prob_new = rf_model.predict_proba(new_data_scaled)[:, 1]  # Probabilidad de que tenga un derrame cerebral

    # Ajustar el umbral de clasificación según el que usaste en el entrenamiento (en este caso 0.4)
    threshold = 0.4
    y_pred_new = (y_pred_prob_new > threshold).astype(int)

    # Mostrar la predicción
    return y_pred_prob_new[0]
