import pandas as pd
import numpy as np
from datasets import load_dataset

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

#edades de los pacientes
edades = data["age"]

#convertir la lista de edades en un arreglo de numpy
edades_np= np.array(edades)

#calculo para el promedio de las edades
promedio_edad=np.mean(edades_np)

#imprimir el promedio de las edades
print("El promedio de las edades es:", promedio_edad)

#------------------PARTE 2---------------------------------
# Convertir la estructura Dataset en un DataFrame de Pandas
df = pd.DataFrame(data)

# Separar el dataframe en dos, uno con personas que perecieron y otro con el complemento
df_perecieron = df[df["is_dead"] == 1]
df_no_perecieron = df[df["is_dead"] == 0]

# Calcular el promedio de edades para cada dataset
promedio_edad_perecieron = df_perecieron["age"].mean()
promedio_edad_no_perecieron = df_no_perecieron["age"].mean()

# Imprimir los resultados
print(f"Promedio de edad de personas que perecieron: {promedio_edad_perecieron} años")
print(f"Promedio de edad de personas que no perecieron: {promedio_edad_no_perecieron} años")
