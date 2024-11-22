# Importar librerías necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos
data = pd.read_csv('all_stocks_5yr.csv')

# Convertir la columna 'date' a tipo fecha (aunque no se usará directamente para la predicción)
data['date'] = pd.to_datetime(data['date'])

# Eliminar las columnas 'date' y 'Name' ya que no se utilizarán para la predicción
data = data.drop(['date', 'Name'], axis=1)

# Separar las características (X) y la etiqueta (y)
X = data.drop('close', axis=1)  # Eliminar la columna 'close' de las características
y = data['close']  # La columna que queremos predecir es 'close'

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Escalar las características (esto es importante para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo KNN con 5 vecinos
knn_model = KNeighborsRegressor(n_neighbors=5)

# Entrenar el modelo
knn_model.fit(X_train_scaled, y_train)

# Hacer predicciones en los datos de prueba
y_pred = knn_model.predict(X_test_scaled)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Mostrar algunas predicciones
predictions = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
print(predictions.head())

# Guardar el modelo KNN entrenado
joblib.dump(knn_model, 'acciones_knn_model.pkl')

# Guardar el escalador usado en el preprocesamiento
joblib.dump(scaler, 'acciones_scaler.pkl')
