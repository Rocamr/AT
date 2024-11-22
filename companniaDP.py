import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

model = joblib.load('cambioCT_model.pkl')
def companniaTC(new_data):
    label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        new_data[col] = le.fit_transform(new_data[col])  # Codificar las nuevas entradas de datos

    # Escalar las características numéricas usando el mismo scaler
    scaler = StandardScaler()
    new_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(new_data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Realizar la predicción usando el modelo cargado
    prediction = model.predict(new_data)

    # Mostrar la predicción
    if prediction[0] == 1:
        return 1
    else:
        return 1
