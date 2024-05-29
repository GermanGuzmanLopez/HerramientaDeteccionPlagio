# scripts/vectorizacion.py
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_documentos(textos):
    """
    Vectoriza una lista de textos usando TF-IDF.

    Args:
        textos (list): Lista de textos a vectorizar.

    Returns:
        tuple: Vectores generados y el vectorizador usado.
    """
    vectorizador = TfidfVectorizer()
    vectores = vectorizador.fit_transform(textos)
    return vectores, vectorizador

def vectorizar_nuevo_texto(nuevo_texto, vectorizador):
    """
    Vectoriza un nuevo texto usando un vectorizador existente.

    Args:
        nuevo_texto (str): Texto a vectorizar.
        vectorizador (TfidfVectorizer): Vectorizador ya entrenado.

    Returns:
        scipy.sparse.csr.csr_matrix: Vector del nuevo texto.
    """
    nuevo_texto_vector = vectorizador.transform([nuevo_texto])
    return nuevo_texto_vector
