import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Función para cargar el modelo y hacer predicciones
def predict_cirrhosis(new_data):
    # Cargar el modelo entrenado
    model = joblib.load('cirrosis_model.pkl')
    scaler = joblib.load('scaler.pkl')  # Asumimos que guardaste el escalador

    # Manejar datos faltantes en los datos nuevos
    numeric_cols = new_data.select_dtypes(include=['float64', 'int64']).columns
    imputer_mean = SimpleImputer(strategy='mean')
    new_data[numeric_cols] = imputer_mean.fit_transform(new_data[numeric_cols])

    # Escalar los datos de entrada
    new_data_scaled = scaler.transform(new_data)

    # Hacer predicciones
    predictions = model.predict(new_data_scaled)

    return predictions


# Ejemplo de cómo llamar a la función
if __name__ == '__main__':
    # Definir nuevos datos en el mismo formato que los datos de entrenamiento
    # Definir nuevos datos, asegurándote de incluir la columna 'N_Days'
    new_data = pd.DataFrame({
        'N_Days': [100],  # Ejemplo de valor para N_Days
        'Status': [1],  # Ejemplo de un estado (codificado)
        'Drug': [0],  # Droga utilizada (codificado)
        'Age': [45],  # Edad
        'Sex': [1],  # Sexo (0 = Femenino, 1 = Masculino)
        'Ascites': [0],  # Presencia de Ascitis (codificado)
        'Hepatomegaly': [1],  # Presencia de Hepatomegalia (codificado)
        'Spiders': [0],  # Presencia de Arañas vasculares (codificado)
        'Edema': [0],  # Edema (codificado)
        'Bilirubin': [1.2],  # Bilirrubina
        'Cholesterol': [200],  # Colesterol
        'Albumin': [3.5],  # Albúmina
        'Copper': [150],  # Cobre
        'Alk_Phos': [150],  # Fosfatasa alcalina
        'SGOT': [45],  # SGOT
        'Tryglicerides': [130],  # Triglicéridos
        'Platelets': [250],  # Plaquetas
        'Prothrombin': [12]  # Trombina
    })

    # Hacer predicción
    prediction = predict_cirrhosis(new_data)
    print(f"Predicción del modelo: {prediction}")
