import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo guardado
model = joblib.load('vino_modelo.pkl')

# Cargar y preparar el codificador de etiquetas (Label Encoder)
le = LabelEncoder()
le.fit(['red', 'white'])  # Asegúrate de que los tipos de vino estén en el mismo orden que en el entrenamiento


def predict_wine_quality(data):
    # Crear DataFrame a partir de los datos de entrada
    new_data = pd.DataFrame(data)

    # Transformar la columna 'type'
    new_data['type'] = le.transform(new_data['type'])  # Transformar la columna 'type'

    # Realizar la predicción
    prediction = model.predict(new_data)
    return prediction.tolist()  # Convertir la predicción a lista para que sea JSON serializable
