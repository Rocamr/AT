# Importar las bibliotecas necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Cargar los datos
data = pd.read_csv('cirrhosis.csv')
# Imputar datos nulos
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer_mean = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer_mean.fit_transform(data[numeric_cols])

imputer_median = SimpleImputer(strategy='median')
data[['Bilirubin', 'Cholesterol']] = imputer_median.fit_transform(data[['Bilirubin', 'Cholesterol']])

categorical_cols = data.select_dtypes(include=['object']).columns
imputer_mode = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_mode.fit_transform(data[categorical_cols])

# Asegurarse de que Stage sea un entero
data['Stage'] = data['Stage'].astype(int)

# Comprobar si hay valores nulos en Stage
if data['Stage'].isnull().sum() > 0:
    print("Hay valores nulos en la columna Stage. Por favor, maneja esos valores antes de continuar.")

# Convertir variables categóricas a numéricas
label_cols = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Definir las características (X) y la etiqueta (y)
X = data.drop(['ID', 'Stage'], axis=1)  # Eliminar 'ID' y 'Stage' de X
y = data['Stage']  # Etiqueta para clasificación

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos (es importante para SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear')  # Puedes cambiar el kernel si es necesario
svm_model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = svm_model.predict(X_test)

# Evaluación del modelo
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(svm_model, 'cirrosis_model.pkl')