from scripts.deteccion_plagio import analizar_documentos_carpeta, generar_tabla
import os
import pickle

def archivo_tiene_contenido(ruta):
    """Verifica si el archivo en la ruta especificada tiene contenido."""
    return os.path.exists(ruta) and os.path.getsize(ruta) > 0

# Ruta al archivo de vectores
ruta_vectores = "./data/vectores.pkl"

# Análisis inicial de textos en la carpeta
if not archivo_tiene_contenido(ruta_vectores):
    archivos, vectores, vectorizador = analizar_documentos_carpeta("./data/textos_origen")
else:
    with open(ruta_vectores, "rb") as f:
        archivos, vectores, vectorizador = pickle.load(f)

# Comparar un nuevo texto
nuevo_texto = "Artificial intelligence (AI) is developing and its application is spreading at an alarming rate, and AI has become part of our daily lives. As a matter of fact, AI has changed the way people learn. However, its adoption in the educational sector has been saddled with challenges and ethical issues. The purpose of this study is to analyze the opportunities, benefits, and challenges of AI in education. A review of available and relevant literature was done using the systematic review method to identify the current research focus and provide an in-depth understanding of AI technology in education for educators and future research directions. Findings showed that AI's adoption in education has advanced in the developed countries and most research became popular within the Industry 4.0 era. Other challenges, as well as recommendations, are discussed in the study."
tabla_resultados = generar_tabla(nuevo_texto, archivos, vectores, vectorizador, umbral=0.5, max_r=10)

# Mostrar los resultados
print(tabla_resultados)
