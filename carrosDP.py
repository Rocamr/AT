import pandas as pd
import joblib

# Cargar el modelo entrenado y los LabelEncoders
rf_model = joblib.load('carros_model.pkl')

# Cargar los LabelEncoders guardados
label_encoders = joblib.load('label_encoders.pkl')
def carrosPP(new_data) :
    # Convertir las columnas categóricas que están en el LabelEncoder y en los datos nuevos
    for column in label_encoders:
        if column in new_data.columns:
            new_data[column] = label_encoders[column].transform(new_data[column])

    # Hacer la predicción
    predicted_price = rf_model.predict(new_data)
    return predicted_price[0]

