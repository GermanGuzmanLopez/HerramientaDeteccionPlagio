"""
Proyecto: Herramienta de detección de plagio
Equipo: Equipo 2

Integrantes:
Marco Barbosa Maruri   		A01746163
Germán Guzmán López   		A01752165
Isabel Vieyra Enríquez    	A01745860

deteccion_plagio.py
Contiene Word2Vec. Es una version preeliminar. No utilizada pero se mantienen por motivos de documentación.
"""

import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scripts.procesamiento_texto import limpiar_parrafo, lematizar
from scripts.vectorizacion import entrenar_word2vec, vectorizar
import pandas as pd


def analizar_textos_en_carpeta(carpeta):
    textos = []
    corpus = []

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".txt"):
            with open(os.path.join(carpeta, archivo), 'r', encoding='utf-8') as file:
                texto = file.read()
                texto_limpio = limpiar_parrafo(texto)
                lemas = lematizar(texto_limpio)
                textos.append((archivo, lemas))
                corpus.append(lemas)

    model = entrenar_word2vec(corpus)

    vectores = {}
    for archivo, lemas in textos:
        vectores[archivo] = vectorizar(lemas, model)

    # Crear el directorio 'data' si no existe
    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'data'))

    vectores_path = os.path.join(data_dir, "vectores.pkl")
    with open(vectores_path, "wb") as f:
        pickle.dump((model, vectores), f)

    print("Análisis completado y vectores almacenados.")
    return model, vectores


def comparar_nuevo_texto(nuevo_texto, model, vectores):
    nuevo_texto_limpio = limpiar_parrafo(nuevo_texto)
    nuevo_texto_lemas = lematizar(nuevo_texto_limpio)
    nuevo_texto_vector = vectorizar(nuevo_texto_lemas, model)

    similitudes = {}

    for archivo, vector in vectores.items():
        similitud = cosine_similarity([nuevo_texto_vector], [vector])[0][0]
        similitudes[archivo] = similitud

    return similitudes


def generar_tabla_resultados(nuevo_texto, model, vectores, umbral=0.5, top_n=10):
    similitudes = comparar_nuevo_texto(nuevo_texto, model, vectores)

    datos = {
        'Archivo': [],
        'Similitud Vectorial': [],
        'Plagio Detectado': []
    }

    for archivo in similitudes.keys():
        datos['Archivo'].append(archivo)
        datos['Similitud Vectorial'].append(similitudes[archivo])
        plagio_detectado = similitudes[archivo] >= umbral
        datos['Plagio Detectado'].append(plagio_detectado)

    tabla_resultados = pd.DataFrame(datos)

    # Ordenar la tabla por 'Similitud Vectorial' y seleccionar los 'top_n' registros
    tabla_resultados = tabla_resultados.sort_values(
        by='Similitud Vectorial', ascending=False).head(top_n)

    return tabla_resultados
