import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scripts.procesamiento_texto import limpiar_parrafo, lematizar
from scripts.vectorizacion import vectorizar_documentos, vectorizar_nuevo_texto
import pandas as pd
import numpy as np

def analizar_documentos_carpeta(carpeta):
    # Cargar Documentos
    textos = []
    archivos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".txt"):
            with open(os.path.join(carpeta, archivo), 'r', encoding='utf-8') as file:
                texto = file.read()
                textos.append(texto)
                archivos.append(archivo)
    
    # Preprocesar Textos
    textos_preprocesados = [" ".join(lematizar(limpiar_parrafo(texto))) for texto in textos]
    
    # Vectorizar los documentos
    vectores, vectorizador = vectorizar_documentos(textos_preprocesados)

    # Guardar vectorizador y vectores en vectores.pkl
    with open("./data/vectores.pkl", "wb") as f:
        pickle.dump((archivos, vectores, vectorizador), f)

    print("Análisis finalizado, vectores guardados")    

    return archivos, vectores, vectorizador

def comparar_nuevo_texto(nuevo_texto, archivos, vectores, vectorizador):
    nuevo_texto = " ".join(lematizar(limpiar_parrafo(nuevo_texto)))
    nuevo_texto_vector = vectorizar_nuevo_texto(nuevo_texto, vectorizador).toarray()

    similitudes = {}
    for archivo, vector in zip(archivos, vectores):
        vector = vector.toarray()  # Convertir a array denso si es necesario
        similitud = cosine_similarity(nuevo_texto_vector, vector)[0][0]
        similitudes[archivo] = similitud
    
    return similitudes

def generar_tabla(nuevo_texto, archivos, vectores, vectorizador, umbral=0.5, max_r=10):
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

    # Ordenar la tabla por 'Similitud' y seleccionar los 'max_r' registros
    tabla_resultados = tabla_resultados.sort_values(by='Similitud', ascending=False).head(max_r)
    
    return tabla_resultados
