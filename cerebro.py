import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import resample
import joblib

# Cargar datos
df = pd.read_csv('cerebro_vascular.csv')

# Manejar valores faltantes (por ejemplo, para BMI)
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# Convertir características categóricas a numéricas
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'].fillna('Unknown'))

# Características y variable objetivo
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Manejar el desequilibrio de clases en el conjunto de entrenamiento
X_train_resampled, y_train_resampled = resample(X_train_scaled, y_train,
                                                replace=True,
                                                n_samples=y_train.value_counts().max(),
                                                random_state=42)

# Modelo Random Forest con balance de clases
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Predicciones
y_pred_prob = rf.predict_proba(X_test_scaled)[:, 1]

# Ajustar el umbral de clasificación
threshold = 0.4
y_pred = (y_pred_prob > threshold).astype(int)

# Evaluación del modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'ROC AUC Score: {roc_auc}')

joblib.dump(rf, 'stroke_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
