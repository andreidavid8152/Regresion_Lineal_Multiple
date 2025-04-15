# Análisis Técnico y Estadístico del Modelo de Regresión Lineal Multiple

### Interpretación:
En esta gráfica se compara la predicción realizada por el modelo de regresión lineal múltiple con los valores reales de uso diario de la aplicación móvil (en minutos), usando el conjunto de prueba.

- Los **puntos azules** representan los **valores reales** observados en el dataset.
- Los **puntos naranjas** muestran los **valores predichos** por el modelo.

- En los rangos medios (130–180 minutos), las predicciones suelen estar bastante cercanas a los valores reales.
- En los extremos (uso muy bajo o muy alto), las predicciones pueden desviarse más, indicando que el modelo no captura tan bien esos casos menos frecuentes.
- Usuarios con características muy distintas a los comunes del conjunto de entrenamiento (por ejemplo, jóvenes con alto nivel de adicción o apps inusuales) pueden tener mayor error en la predicción.

### Análisis:
- Se puede observar una **alineación general** entre los valores reales y predichos.
- En muchos casos, los puntos de ambos colores están muy cercanos, lo que indica **predicciones acertadas**.
- Sin embargo, hay **cierta dispersión**, especialmente en los valores más altos y más bajos, lo cual es común cuando los datos tienen relaciones complejas o no lineales.
- No se detectan patrones cíclicos o de error sistemático, lo que sugiere que el modelo **no está sesgado**.

### Conclusión:
El modelo de regresión lineal múltiple ha logrado **capturar de manera adecuada la tendencia general** en los datos, aunque no es perfecto, el modelo muestra un rendimiento confiable.