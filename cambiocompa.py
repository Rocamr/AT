import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Cargar los datos
data = pd.read_csv('compa_cambio.csv')

# Preprocesamiento de datos

# Eliminar la columna customerID porque es un identificador único y no aporta información útil para el modelo
data = data.drop('customerID', axis=1)

# Convertir variables categóricas a numéricas usando LabelEncoder
# Algunas columnas tienen valores 'Yes' y 'No' que deben ser codificados, otras tienen múltiples categorías
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
              'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

# Aplicar LabelEncoder a cada columna categórica
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Guardar los codificadores por si se necesita revertir los valores más adelante

# Rellenar datos faltantes en TotalCharges con la mediana
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Escalar las características numéricas (tenure, MonthlyCharges, TotalCharges)
scaler = StandardScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Definir las características (X) y la etiqueta (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Regresión Logística
log_reg = LogisticRegression(max_iter=1000)  # Ajustar el número máximo de iteraciones si es necesario
log_reg.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(log_reg, 'cambioCT_model.pkl')