# scripts/deteccion_plagio.py
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scripts.procesamiento_texto import procesar_texto
from scripts.vectorizacion import vectorizar_documentos, vectorizar_nuevo_texto
import pandas as pd
import numpy as np

def analizar_documentos_carpeta(carpeta):
    """
    Analiza los documentos de una carpeta y guarda sus vectores en un archivo .pkl.

    Args:
        carpeta (str): Ruta a la carpeta que contiene los documentos originales.

    Returns:
        tuple: Archivos analizados, vectores generados y el vectorizador usado.
    """
    textos = []
    archivos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".txt"):
            with open(os.path.join(carpeta, archivo), 'r', encoding='utf-8') as file:
                texto = file.read()
                textos.append(texto)
                archivos.append(archivo)
    
    textos_preprocesados = [" ".join(procesar_texto(texto)) for texto in textos]
    vectores, vectorizador = vectorizar_documentos(textos_preprocesados)

    with open("./data/vectores.pkl", "wb") as f:
        pickle.dump((archivos, vectores, vectorizador), f)

    print("Análisis finalizado, vectores guardados")    

    return archivos, vectores, vectorizador

def comparar_nuevo_texto(nuevo_texto, archivos, vectores, vectorizador):
    """
    Compara un nuevo texto contra los textos originales usando similitud de coseno.

    Args:
        nuevo_texto (str): Texto a comparar.
        archivos (list): Lista de nombres de archivos originales.
        vectores (list): Lista de vectores de los archivos originales.
        vectorizador (TfidfVectorizer): Vectorizador usado para generar los vectores.

    Returns:
        dict: Diccionario con archivos y sus respectivas similitudes.
    """
    nuevo_texto = " ".join(procesar_texto(nuevo_texto))
    nuevo_texto_vector = vectorizar_nuevo_texto(nuevo_texto, vectorizador).toarray()

    similitudes = {}
    for archivo, vector in zip(archivos, vectores):
        vector = vector.toarray()
        similitud = cosine_similarity(nuevo_texto_vector, vector)[0][0]
        similitudes[archivo] = similitud
    
    return similitudes

def generar_tabla(nuevo_texto, archivos, vectores, vectorizador, umbral=0.4, max_r=10):
    """
    Genera una tabla con los resultados de similitud de un nuevo texto contra los textos originales.

    Args:
        nuevo_texto (str): Texto a comparar.
        archivos (list): Lista de nombres de archivos originales.
        vectores (list): Lista de vectores de los archivos originales.
        vectorizador (TfidfVectorizer): Vectorizador usado para generar los vectores.
        umbral (float, optional): Umbral de similitud para considerar plagio. Default es 0.4.
        max_r (int, optional): Número máximo de resultados a mostrar. Default es 10.

    Returns:
        pd.DataFrame: DataFrame con los resultados de similitud.
    """
    similitudes = comparar_nuevo_texto(nuevo_texto, archivos, vectores, vectorizador)

    datos = {
        'Archivo': [],
        'Similitud': [],
        'Plagio detectado': []
    }

    for archivo, similitud in similitudes.items():
        datos['Archivo'].append(archivo)
        datos['Similitud'].append(similitud)
        datos['Plagio detectado'].append(similitud >= umbral)
    
    tabla_resultados = pd.DataFrame(datos)
    tabla_resultados = tabla_resultados.sort_values(by='Similitud', ascending=False).head(max_r)
    
    return tabla_resultados
