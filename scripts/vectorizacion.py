import numpy as np
from collections import Counter

def n_gramas(texto, n):
    """Genera n-gramas a partir de un texto"""
    tokens = texto.split()
    n_grams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(n_gram) for n_gram in n_grams]

def construir_vocabulario(documentos, n):
    vocabulario = set()
    for doc in documentos:
        n_gramas_doc = n_gramas(" ".join(doc), n)
        vocabulario.update(n_gramas_doc)
    return list(vocabulario)

def vectorizar(parrafo, vocabulario, n=5):
    n_gramas_texto = n_gramas(parrafo, n)
    frecuencia = Counter(n_gramas_texto)
    
    vector = np.zeros(len(vocabulario))
    for i, n_grama in enumerate(vocabulario):
        vector[i] = frecuencia[n_grama]

    return vector

def vectorizar_documentos(documentos, vocabulario, n=7):
    vectores = {}
    for idx, lemas in enumerate(documentos):
        vector = vectorizar(" ".join(lemas), vocabulario, n)
        vectores[f"org-{idx + 1}"] = vector
    
    return vectores
