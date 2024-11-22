import pandas as pd
import joblib
from fontTools.misc.cython import returns

# Cargar el modelo guardado
rf_model = joblib.load('masacorporal_model.pkl')


def masaCP(new_data):
    # Hacer predicciones con los nuevos datos
    predictions = rf_model.predict(new_data)
    return predictions[0]
