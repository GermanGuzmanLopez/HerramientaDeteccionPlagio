from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_documentos(textos):
    vectorizador = TfidfVectorizer()
    vectores = vectorizador.fit_transform(textos)
    return vectores, vectorizador

def vectorizar_nuevo_texto(nuevo_texto, vectorizador):
    nuevo_texto_vector = vectorizador.transform([nuevo_texto])
    return nuevo_texto_vector
