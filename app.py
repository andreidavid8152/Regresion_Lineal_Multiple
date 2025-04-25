# 1. Importación de bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

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

# 4. Codificación de variable categórica 'App_Type'
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder solo sobre la columna 'App_Type'
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), ["App_Type"])], remainder="passthrough"
)

X = np.array(ct.fit_transform(X))

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

# ----- ANALISIS -----

# Métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
media_uso_diario = np.mean(y_test)
mae_porcentaje = (mae / media_uso_diario) * 100

# Interpretación
print(f"""

      MODELO DE REGRESION MULTIPLE

      En este modelo se han considerado diversas variables relacionadas con el uso de aplicaciones móviles, tales como el tiempo total de pantalla, el nivel de adicción, la frecuencia de notificaciones y el tipo de aplicación, para entender su influencia sobre los minutos de uso diario (Daily App Usage Minutes).

      El coeficiente de determinación (r²) obtenido es de {r2:.4f}, lo que indica que el modelo explica aproximadamente el {r2 * 100:.1f}% de la variabilidad observada en el uso diario de aplicaciones. El error absoluto medio (MAE) registrado fue de {mae:.2f} minutos, reflejando una desviación promedio del {mae_porcentaje:.2f}% respecto a los valores reales de uso diario.

      Estos resultados sugieren que, si bien el modelo logra capturar una porción considerable de la variabilidad, existe aún margen de error atribuible a otros factores no incluidos en el análisis actual.
""")
