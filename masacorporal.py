# Importar bibliotecas necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Cargar los datos
data = pd.read_csv('bodyfat.csv')

# Verificar la carga de datos
print(data.head())

# Imputar valores nulos con la media de cada columna
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Definir las características (X) y la etiqueta (y)
X = data_imputed.drop('BodyFat', axis=1)  # Suponiendo que BodyFat es la columna objetivo
y = data_imputed['BodyFat']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Bosque Aleatorio
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Mostrar algunas predicciones
predictions = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
print(predictions.head())
joblib.dump(rf_model, 'masacorporal_model.pkl')