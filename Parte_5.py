import pandas as pd
import numpy as np
from datasets import load_dataset
import requests

#--------------------PARTE 1--------------------------

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

#------------------------PARTE 2--------------------------------------

# Convertir la estructura Dataset en un DataFrame de Pandas
df = pd.DataFrame(data)

# Separar el dataframe en dos, uno con personas que perecieron y otro con el complemento
df_perecieron = df[df["is_dead"] == 1]
df_no_perecieron = df[df["is_dead"] == 0]

# Calcular el promedio de edades para cada dataset
promedio_edad_perecieron = df_perecieron["age"].mean()
promedio_edad_no_perecieron = df_no_perecieron["age"].mean()

# Imprimir los resultados de los promedios de edades
print(f"Promedio de edad de personas que perecieron: {promedio_edad_perecieron} años")
print(f"Promedio de edad de personas que no perecieron: {promedio_edad_no_perecieron} años")

#----------------------------PARTE 3--------------------------------------------

# Verificar tipos de datos
tipos_de_datos = df.dtypes
print("\nTipos de datos en cada columna:")
print(tipos_de_datos)

# Agrupar por sexo y fumar
df_agrupado = df.groupby(["is_male", "is_smoker"])

# Contar el número de filas en cada grupo
cantidad = df_agrupado.size()

# Imprimir el resultado
print(cantidad)

#-------------------------------PARTE 4----------------------------------------


def descargar_datos(url, nombre_archivo):
    # Realizar el GET request
    respuesta = requests.get(url)
    
    # Verificar si la solicitud fue exitosa (código de estado 200)
    if respuesta.status_code == 200:
        # Guardar la respuesta en un archivo CSV
        with open(nombre_archivo, 'wb') as archivo:
            archivo.write(respuesta.content)
        print(f"Los datos han sido descargados y guardados en {nombre_archivo}")
    else:
        print(f"Error al descargar los datos. Código de estado: {respuesta.status_code}")

# URL de los datos
url_datos = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"

# Nombre del archivo de salida
nombre_archivo_salida = "datos_heart_failure.csv"

# Llamar a la función para descargar los datos
descargar_datos(url_datos, nombre_archivo_salida)

#--------------------------------PARTE 5---------------------------------------------------------------


def procesar_datos(dataframe):
    # Verificar valores faltantes
    if dataframe.isnull().any().any():
        print("Existen valores faltantes. Realice la imputación o eliminación según sea necesario.")
        dataframe = dataframe.dropna()  # Eliminar filas con valores faltantes

    # Verificar filas duplicadas
    if dataframe.duplicated().any():
        print("Existen filas duplicadas. Realice la eliminación de duplicados.")
        dataframe = dataframe.drop_duplicates()

    # Verificar y eliminar valores atípicos (puedes ajustar este paso según tu criterio)
    # Por ejemplo, eliminando las filas con valores atípicos en la columna 'edad'
    q1 = dataframe['age'].quantile(0.25)
    q3 = dataframe['age'].quantile(0.75)
    iqr = q3 - q1
    filtro_sin_atipicos = (dataframe['age'] >= q1 - 1.5 * iqr) & (dataframe['age'] <= q3 + 1.5 * iqr)
    dataframe = dataframe[filtro_sin_atipicos]

    # Crear columna de categorías por edades
    bins = [0, 12, 19, 39, 59, float('inf')]
    labels = ['Niño', 'Adolescente', 'Joven adulto', 'Adulto', 'Adulto mayor']
    dataframe['categoria_edad'] = pd.cut(dataframe['age'], bins=bins, labels=labels, right=False)

    # Guardar el resultado como CSV
    nombre_archivo_salida = "datos_procesados.csv"
    dataframe.to_csv(nombre_archivo_salida, index=False)

    print(f"Procesamiento completado. Resultados guardados en {nombre_archivo_salida}")

# Cargar el DataFrame desde el CSV descargado
datos = pd.read_csv("datos_heart_failure.csv")

# Llamar a la función para procesar los datos
procesar_datos(datos)
