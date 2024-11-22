# Importar bibliotecas necesarias
import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('aguacate_model.pkl')

def aguacatepp(new_data):
    # Hacer predicciones
    prediction = model.predict(new_data)
    return prediction
