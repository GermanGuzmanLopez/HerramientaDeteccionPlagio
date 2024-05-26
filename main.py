from scripts.deteccion_plagio import analizar_documentos_carpeta, comparar_nuevo_texto
from scripts.calcular_umbral import evaluar_umbral
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def archivo_tiene_contenido(ruta):
    """Verifica si el archivo en la ruta especificada tiene contenido."""
    return os.path.exists(ruta) and os.path.getsize(ruta) > 0

# Ruta al archivo de vectores
ruta_vectores = "./data/vectores.pkl"

# Análisis inicial de textos en la carpeta de textos originales
if not archivo_tiene_contenido(ruta_vectores):
    archivos, vectores, vectorizador = analizar_documentos_carpeta("./data/textos_origen")
else:
    with open(ruta_vectores, "rb") as f:
        archivos, vectores, vectorizador = pickle.load(f)

# Leer archivo de plagios reales
plagios_reales = pd.read_csv("./data/plagios_reales.csv")

# Analizar documentos en data/test_data
carpeta_test_data = "./data/test_data"
resultados_similitudes = []

for archivo in os.listdir(carpeta_test_data):
    if archivo.endswith(".txt"):
        with open(os.path.join(carpeta_test_data, archivo), 'r', encoding='utf-8') as file:
            nuevo_texto = file.read()
            similitudes = comparar_nuevo_texto(nuevo_texto, archivos, vectores, vectorizador)
            resultado = {
                'Archivo': archivo,
                'Similitudes': similitudes
            }
            resultados_similitudes.append(resultado)

# Evaluar umbrales
metricas_df, umbral_optimo = evaluar_umbral(resultados_similitudes, plagios_reales)

# Mostrar resultados de métricas
# print(metricas_df)
# print(f"El umbral óptimo basado en la métrica F1 es {umbral_optimo['Umbral']}")
# print(f"Precision: {umbral_optimo['Precision']}, Recall: {umbral_optimo['Recall']}, F1: {umbral_optimo['F1']}")

# Generar tabla con porcentaje de plagio y fuentes solo para archivos en test_data
tabla_resultados = []

umbral_fijo = 0.40  # Puedes ajustar este umbral según sea necesario

for resultado in resultados_similitudes:
    archivo = resultado['Archivo']
    max_similitud = max(resultado['Similitudes'].values())
    
    # Encontrar las fuentes que tienen similitud mayor o igual al umbral fijo
    fuentes = []
    for k, v in resultado['Similitudes'].items():
        if v >= umbral_fijo:
            fuentes.append(f"{k}: %{100*round(v,2)}")
    
    # Calcular el porcentaje de plagio
    porcentaje_plagio = max_similitud * 100
    
    # Crear el diccionario de resultados para este archivo
    resultado_diccionario = {
        'Archivo': archivo,
        'Porcentaje de Plagio': porcentaje_plagio,
        'Fuentes': ', '.join(fuentes)
    }
    
    # Añadir el resultado a la lista de resultados
    tabla_resultados.append(resultado_diccionario)

# Crear DataFrame con todos los resultados
tabla_resultados_df = pd.DataFrame(tabla_resultados)

# Guardar la tabla completa en un archivo CSV
# tabla_resultados_df.to_csv('./data/tabla_resultados_completa.csv', index=False)

# Mostrar y guardar la tabla con los 20 registros con más porcentaje de plagio
tabla_resultados_top20 = tabla_resultados_df.sort_values(by='Porcentaje de Plagio', ascending=False)
print(tabla_resultados_top20)
# tabla_resultados_top20.to_csv('./data/tabla_resultados_top20.csv', index=False)

# Graficar métricas
# plt.figure(figsize=(10, 6))
# plt.plot(metricas_df['Umbral'], metricas_df['Precision'], label='Precision')
# plt.plot(metricas_df['Umbral'], metricas_df['Recall'], label='Recall')
# plt.plot(metricas_df['Umbral'], metricas_df['F1'], label='F1')
# plt.xlabel('Umbral')
# plt.ylabel('Métrica')
# plt.title('Evaluación de Umbrales para Detección de Plagio')
# plt.legend()
# plt.grid(True)
# plt.show()
