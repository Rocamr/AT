import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib  # Para guardar el modelo en formato .pkl

# Cargar los datos
df = pd.read_csv('DelayedFlights.csv')

# Convertir las variables categóricas a numéricas usando LabelEncoder
label_encoder = LabelEncoder()
df['UniqueCarrier'] = label_encoder.fit_transform(df['UniqueCarrier'])
df['Origin'] = label_encoder.fit_transform(df['Origin'])
df['Dest'] = label_encoder.fit_transform(df['Dest'])

# Seleccionar las características y la variable objetivo
features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
            'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime',
            'CRSElapsedTime', 'AirTime', 'DepDelay', 'Origin', 'Dest', 'Distance',
            'TaxiIn', 'TaxiOut', 'Cancelled', 'Diverted']
target = 'ArrDelay'

X = df[features]
y = df[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Manejar NaN
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)
y_test.fillna(y_test.mean(), inplace=True)

# Crear el dataset LightGBM
train_data = lgb.Dataset(X_train, label=y_train)

# Definir los parámetros del modelo
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Validación cruzada con KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []

for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Usar el callback de early_stopping y log_evaluation para verbosidad
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = mse ** 0.5
    rmse_scores.append(rmse)

# Mostrar el mejor RMSE
best_rmse = min(rmse_scores)
print("Best RMSE from CV:", best_rmse)

# Entrenar el modelo final con todos los datos de entrenamiento
model = lgb.train(params, train_data, num_boost_round=1000)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Calcular el error cuadrático medio (RMSE) en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("Root Mean Squared Error (RMSE) on test set:", rmse)

# Guardar el modelo entrenado en formato .pkl
joblib.dump(model, 'lightgbm_flight_delay_model.pkl')
