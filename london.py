import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Cargar los datos
df = pd.read_csv('london_crime_by_lsoa.csv')

# Convertir las variables categóricas a numéricas
label_encoder = LabelEncoder()
df['lsoa_code'] = label_encoder.fit_transform(df['lsoa_code'])
df['borough'] = label_encoder.fit_transform(df['borough'])
df['major_category'] = label_encoder.fit_transform(df['major_category'])
df['minor_category'] = label_encoder.fit_transform(df['minor_category'])

# Características y variable objetivo
X = df[['lsoa_code', 'borough', 'major_category', 'minor_category', 'year', 'month']]
y = df['value']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los parámetros que queremos probar
param_grid = {'n_neighbors': range(1, 20)}

# Crear el modelo KNN
knn = KNeighborsRegressor()

# Realizar la búsqueda en la cuadrícula con validación cruzada
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Mejor número de vecinos
print("Mejor número de vecinos:", grid_search.best_params_['n_neighbors'])

# Evaluar el modelo con el mejor número de vecinos
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print("Error cuadrático medio optimizado:", mse)

# Guardar el modelo entrenado y el escalador
joblib.dump(best_knn, 'best_knn_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
