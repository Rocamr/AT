# Importar bibliotecas necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar los datos
data = pd.read_csv('precio_carros.csv')

# Verificar la carga de datos
print(data.head())

# Separar las columnas numéricas y no numéricas
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Imputar valores nulos en columnas numéricas con la media
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Convertir las columnas categóricas a variables numéricas
label_encoders = {}
for column in categorical_cols:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Definir las características (X) y la etiqueta (y)
X = data.drop(['Car_Name', 'Selling_Price'], axis=1)  # Excluir el nombre del carro y el precio a predecir
y = data['Selling_Price']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Regresión de Bosque Aleatorio
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {mse}")
print(f"R^2 Score: {r2}")

# Mostrar algunas predicciones
predictions = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
print(predictions.head())

joblib.dump(rf_model, 'carros_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
