# Importar bibliotecas necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar los datos
data = pd.read_csv('avocado.csv')

# Verificar la carga de datos
print(data.head())

# Verificar si existe la columna 'Unnamed: 0' y eliminarla
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Convertir la columna 'type' a variables numéricas
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Convertir la columna 'region' a variables numéricas
le_region = LabelEncoder()
data['region'] = le_region.fit_transform(data['region'])

# Imputar valores nulos solo en columnas numéricas
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data_imputed = data.copy()  # Hacer una copia de los datos originales
data_imputed[numeric_cols] = imputer.fit_transform(data_imputed[numeric_cols])

# Definir las características (X) y la etiqueta (y)
X = data_imputed.drop(['AveragePrice', 'Date'], axis=1)  # Eliminar 'Date' y la etiqueta
y = data_imputed['AveragePrice']

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

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Guardar el modelo entrenado
joblib.dump(rf_model, 'aguacate_model.pkl')
