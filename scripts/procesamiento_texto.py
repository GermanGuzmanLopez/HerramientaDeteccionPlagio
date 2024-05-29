# scripts/procesamiento_texto.py
import spacy
import string
import contractions

# Verificar si el modelo está instalado
import spacy.util
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

# Cargar el modelo inglés de spaCy
nlp = spacy.load("en_core_web_sm")

def expandir_contracciones(parrafo):
    """
    Expande contracciones comunes en inglés utilizando la biblioteca contractions.

    Args:
        parrafo (str): Párrafo a expandir.

    Returns:
        str: Párrafo con contracciones expandidas.
    """
    parrafo_expandido = contractions.fix(parrafo)
    return parrafo_expandido

def procesar_texto(parrafo):
    """
    Limpia y lematiza un párrafo eliminando stopwords y signos de puntuación.

    Args:
        parrafo (str): Párrafo a procesar.

    Returns:
        list: Lista de lemas del párrafo.
    """
    parrafo = expandir_contracciones(parrafo)
    parrafo = parrafo.lower().strip()
    signos_puntuacion = string.punctuation.replace('.', '')
    parrafo = parrafo.translate(str.maketrans('', '', signos_puntuacion))
    doc = nlp(parrafo)
    lemas = [token.lemma_ for token in doc if token.text not in spacy.lang.en.stop_words.STOP_WORDS and not token.is_punct and not token.is_stop]
    return lemas
