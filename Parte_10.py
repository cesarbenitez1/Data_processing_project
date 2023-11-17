import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el DataFrame desde el CSV procesado
df = pd.read_csv("datos_procesados.csv")

# Eliminar las columnas DEATH_EVENT, age y categoria_edad
X = df.drop(columns=["DEATH_EVENT", "age", "categoria_edad"])

# Extraer la columna age como el vector y
y = df["age"]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y ajustar el modelo de regresión lineal
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_regresion.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

print(f"Error Cuadrático Medio: {mse}")