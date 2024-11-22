# Importar bibliotecas necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar los datos
data = pd.read_csv('vino_calidad.csv')

# Verificar la carga de datos
print(data.head())

# Convertir la columna 'type' a variables numéricas
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Imputar valores nulos con la media de cada columna (si hay)
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Definir las características (X) y la etiqueta (y)
X = data_imputed.drop('quality', axis=1)  # Suponiendo que 'quality' es la columna objetivo
y = data_imputed['quality']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Bosque Aleatorio con ajuste para clases desbalanceadas
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))  # Añadir zero_division=1 para evitar advertencias

# Mostrar algunas predicciones
predictions = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
print(predictions.head())

# Guardar el modelo
joblib.dump(rf_model, 'vino_modelo.pkl')
