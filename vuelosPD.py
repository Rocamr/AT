import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('lightgbm_flight_delay_model.pkl')

# Nuevos datos de ejemplo para predicción
new_data = pd.DataFrame({
    'Year': [2023],
    'Month': [6],
    'DayofMonth': [15],
    'DayOfWeek': [3],
    'DepTime': [900],
    'CRSDepTime': [830],
    'ArrTime': [1130],
    'CRSArrTime': [1100],
    'UniqueCarrier': [5],  # Asumiendo que has codificado esto
    'FlightNum': [123],
    'ActualElapsedTime': [120],
    'CRSElapsedTime': [110],
    'AirTime': [100],
    'DepDelay': [10],
    'Origin': [10],  # Asumiendo que has codificado esto
    'Dest': [20],    # Asumiendo que has codificado esto
    'Distance': [500],
    'TaxiIn': [5],
    'TaxiOut': [15],
    'Cancelled': [0],
    'Diverted': [0]
})

# Realizar la predicción
prediction = model.predict(new_data)

# Mostrar la predicción
print(f"Predicción de retraso: {prediction[0]:.2f} minutos")
