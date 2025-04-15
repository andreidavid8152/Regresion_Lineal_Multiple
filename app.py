# 1. Importación de bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Cargar el dataset
dataset = pd.read_csv("data/app_usage_dataset.csv")

# 3. Variables numéricas elegidas por análisis previo
features = [
    "Total_Screen_Time",
    "Addiction_Level",
    "Notification_Frequency",
    "App_Type",
]

# Definir X (variables independientes) e y (variable dependiente)
X = dataset[features]
y = dataset["Daily_App_Usage_Minutes"].values

# Mostrar X antes de codificar
print("Datos originales (X antes de codificación):\n", dataset[features].head(), "\n")

# 4. Codificación de variable categórica 'App_Type'
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder solo sobre la columna 'App_Type'
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), ["App_Type"])], remainder="passthrough"
)

X = np.array(ct.fit_transform(X))
print("Primeras filas de X (tras codificación):\n", X[:5], "\n")

# 5. División del dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 6. Entrenamiento del modelo
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 7. Predicción
y_pred = regressor.predict(X_test)

# 8. Comparar predicciones con valores reales
np.set_printoptions(precision=2)
resultados = np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)
print("Predicciones vs Valores Reales:\n", resultados, "\n")

# 9. Visualización
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="Valores reales")
plt.scatter(range(len(y_pred)), y_pred, label="Predicciones")
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Índice de muestra")
plt.ylabel("Daily_App_Usage_Minutes")
plt.legend()
plt.grid()
plt.show()
