import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Primero se crean los datos (en mi caso, los datos de la base de datos)

# A continuación, como ejemmplo, se van a utilizar datos que hacen referencia a metros cuadrados de una casa
# y su precio en miles de dólares.
metros_cuadrados = np.array([50, 60, 80, 100, 120], [1, 7, 9, 10]).reshape(-1, 1)
precio = np.array([100, 120, 160, 200, 240])

# Se dividen los datos en entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(metros_cuadrados, precio, test_size=0.2)


modelo = LinearRegression()
# Este método entrena el modelo
modelo.fit(X_train, y_train)
# Realiza la predicción con los datos que ha obtenido
modelo.predict(X_test)


# Ejecuta con lo aprendido una nueva predicción con datos nuevos
nueva_casa = np.array([[90]])  
precio_predicho = modelo.predict(nueva_casa)
print(f"Precio estimado para una casa de 90m²: ${precio_predicho[0]}K")


# -----------------------------
# Datos necesarios:
# - Rating
# - Comments