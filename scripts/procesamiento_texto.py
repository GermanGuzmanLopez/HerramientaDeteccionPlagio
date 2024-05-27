import spacy
import string

# Verificar si el modelo está instalado
import spacy.util
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

# Cargar el modelo inglés de spaCy
nlp = spacy.load("en_core_web_sm")

def procesar_texto(parrafo):
    """
    Limpia y lematiza un párrafo eliminando stopwords y signos de puntuación.

    Args:
        parrafo (str): Párrafo a procesar.

    Returns:
        list: Lista de lemas del párrafo.
    """
    parrafo = parrafo.lower().strip()
    doc = nlp(parrafo)
    lemas = [token.lemma_ for token in doc if token.text not in spacy.lang.en.stop_words.STOP_WORDS and not token.is_punct and not token.is_stop]
    lemas = [lemma.translate(str.maketrans('', '', string.punctuation.replace('.', ''))) for lemma in lemas]
    return lemas
