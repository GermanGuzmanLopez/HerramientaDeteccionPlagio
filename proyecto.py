import spacy
from gensim.models import Word2Vec
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string

# Verificar que el modelo está instalado
spacy.cli.download("es_core_news_md")

# Cargar el modelo español de spaCy
nlp = spacy.load("es_core_news_md")

def limpiar_parrafo(parrafo):
    """
    Obtiene un párrafo y elimina cualquier signo de puntuación o símbolo del texto, excepto los puntos.

    Regresa:
    Párrafo limpio
    """
    # Eliminación de símbolos especiales, excepto puntos
    parrafo = parrafo.lower()
    signos_puntuacion = string.punctuation.replace('.', '')
    parrafo = parrafo.translate(str.maketrans('', '', signos_puntuacion))
    return parrafo

def lematizar(parrafo):
    """
    Lematiza el párrafo usando spaCy.

    Regresa:
    Párrafo lematizado
    """
    doc = nlp(parrafo)
    lemas = [token.lemma_ for token in doc if token.text not in STOP_WORDS and not token.is_punct]
    return " ".join(lemas)

def entrenar_word2vec(corpus):
    """
    Entrena un modelo Word2Vec con el corpus dado.

    Regresa:
    Modelo Word2Vec entrenado
    """
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model

def vectorizar(parrafo, model):
    """
    Vectoriza un párrafo usando un modelo Word2Vec.

    Regresa:
    Vector del párrafo
    """
    tokens = [token for token in parrafo if token in model.wv]
    if not tokens:
        return np.zeros(model.vector_size)
    vectors = [model.wv[token] for token in tokens]
    return np.mean(vectors, axis=0)

print("Hola")
# Ejemplo de párrafos a comparar
parrafo1 = "Los videojuegos han avanzado enormemente, ofreciendo experiencias muy inmersivas. Hoy en día, los jugadores disfrutan de gráficos realistas y tramas complejas. Los deportes electrónicos han crecido con torneos mundiales. Las plataformas de streaming permiten compartir partidas en vivo con millones de espectadores. Sin duda, los videojuegos son una forma de entretenimiento masiva y accesible."
parrafo2 = "La industria de videojuegos ha crecido rápidamente, ofreciendo experiencias realistas e inmersivas. Los jugadores disfrutan de gráficos impresionantes y narrativas cinematográficas. Los deportes electrónicos se han vuelto globales con eventos millonarios. Las plataformas de streaming permiten transmisiones en vivo. Los videojuegos son ahora una forma de entretenimiento popular para todas las edades."

# Lematización de los párrafos
parrafo1_lematizado = lematizar(limpiar_parrafo(parrafo1))
parrafo2_lematizado = lematizar(limpiar_parrafo(parrafo2))

# Dividir en listas de palabras para Word2Vec
parrafo1_lematizado = parrafo1_lematizado.split()
parrafo2_lematizado = parrafo2_lematizado.split()

# Entrenar el modelo Word2Vec con los párrafos lematizados
corpus = [parrafo1_lematizado, parrafo2_lematizado]
model = entrenar_word2vec(corpus)

# Vectorizar los párrafos lematizados
vector1 = vectorizar(parrafo1_lematizado, model)
vector2 = vectorizar(parrafo2_lematizado, model)

# Cálculo de la similitud del coseno
matriz_similitud = cosine_similarity([vector1], [vector2])
similitud = matriz_similitud[0][0]

print("Párrafo 1 Lematizado:", parrafo1_lematizado)
print("Párrafo 2 Lematizado:", parrafo2_lematizado)
print("Similitud entre los párrafos:", similitud)
