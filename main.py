from datasets import load_dataset
import numpy as np

#Cargar dataset del fallo cardiaco
dataset=load_dataset("mstz/heart_failure")
data = dataset["train"]

#edades de los pacientes
edades = data["age"]

#convertir la lista de edades en un arreglo de numpy
edades_np= np.array(edades)

#calculo para el promedio de las edades
promedio_edad=np.mean(edades_np)

#imprimir el promedio de las edades
print("El promedio de las edades es:", promedio_edad)
