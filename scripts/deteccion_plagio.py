import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scripts.procesamiento_texto import limpiar_parrafo, lematizar
from scripts.vectorizacion import vectorizar_documentos, construir_vocabulario, vectorizar
import pandas as pd

def analizar_documentos_carpeta(carpeta, n=5):
    # Cargar Documentos
    textos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".txt"):
            with open(os.path.join(carpeta, archivo), 'r', encoding='utf-8') as file:
                texto = file.read()
                textos.append(texto)
    
    # Lematizar Textos
    textos_lematizados = []
    for texto in textos:
        texto_limpio = limpiar_parrafo(texto)
        lemas = lematizar(texto_limpio)
        textos_lematizados.append(lemas)
    
    # Construir vocabulario de n-gramas
    vocabulario = construir_vocabulario(textos_lematizados, n)
    
    # Vectorizar los documentos lematizados
    vectores = vectorizar_documentos(textos_lematizados, vocabulario, n)

    # Guardar vocabulario y vectores en vectores.pkl
    with open("./data/vectores.pkl", "wb") as f:
        pickle.dump((vocabulario, vectores), f)

    print("Análisis finalizado, vectores guardados")    

    return vocabulario, vectores

def comparar_nuevo_texto(nuevo_texto, vocabulario, vectores, n=5):
    nuevo_texto = limpiar_parrafo(nuevo_texto)
    nuevo_texto_lemas = lematizar(nuevo_texto)
    nuevo_texto_vector = vectorizar(" ".join(nuevo_texto_lemas), vocabulario, n)

    similitudes = {}

    for archivo, vector in vectores.items():
        similitud = cosine_similarity([nuevo_texto_vector], [vector])[0][0]
        similitudes[archivo] = similitud
    
    return similitudes

def generar_tabla(nuevo_texto, vocabulario, vectores, umbral=0.5, max_r=10, n=7):
    similitudes = comparar_nuevo_texto(nuevo_texto, vocabulario, vectores, n)

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
