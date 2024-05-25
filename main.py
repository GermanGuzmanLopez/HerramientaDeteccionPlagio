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
    vocabulario, vectores = analizar_documentos_carpeta("./data/textos_origen", n=5)
else:
    with open(ruta_vectores, "rb") as f:
        vocabulario, vectores = pickle.load(f)

# Comparar un nuevo texto
nuevo_texto = "Recent developments in Artificial Intelligence (AI) have generated great expectations for the future impact of AI in education and learning (AIED)..."
tabla_resultados = generar_tabla(nuevo_texto, vocabulario, vectores, umbral=0.5, max_r=10, n=5)

# Mostrar los resultados
print(tabla_resultados)
