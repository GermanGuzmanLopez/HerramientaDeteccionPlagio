import spacy
import string

# Verificar si el modelo está instalado
import spacy.util
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

# Cargar el modelo inglés de spaCy
nlp = spacy.load("en_core_web_sm")

def limpiar_parrafo(parrafo):
    parrafo = parrafo.lower().strip()
    signos_puntuacion = string.punctuation.replace('.', '')
    parrafo = parrafo.translate(str.maketrans('', '', signos_puntuacion))
    return parrafo

def lematizar(parrafo):
    doc = nlp(parrafo)
    lemas = [token.lemma_ for token in doc if token.text not in spacy.lang.en.stop_words.STOP_WORDS and not token.is_punct and not token.is_stop]
    return lemas
