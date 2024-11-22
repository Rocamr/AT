# Importar librerías necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos (especifica la ruta correcta a tu archivo CSV)
data = pd.read_csv('covid_19_data.csv')

# Convertir la columna 'ObservationDate' a tipo fecha
data['ObservationDate'] = pd.to_datetime(data['ObservationDate'])

# Eliminar columnas no relevantes ('SNo', 'Province/State', 'Country/Region', 'Last Update', 'ObservationDate')
data = data.drop(['SNo', 'Province/State', 'Country/Region', 'Last Update', 'ObservationDate'], axis=1)

# Imputar valores faltantes solo en las columnas numéricas
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Crear el conjunto de características (X) y la etiqueta (y)
X = data_imputed[:, :-1]  # Todas las columnas excepto 'Recovered'
y = data_imputed[:, -1]   # Columna 'Recovered' es la etiqueta

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
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
joblib.dump(knn_model, 'cv19_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
