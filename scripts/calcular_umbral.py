# scripts/calcular_umbral.py
import numpy as np
import pandas as pd

def evaluar_umbral(resultados_similitudes, plagios_reales):
    umbrales = np.arange(0.0, 1.0, 0.05)
    metricas = []

    for umbral in umbrales:
        verdaderos_positivos = 0
        falsos_positivos = 0
        verdaderos_negativos = 0
        falsos_negativos = 0

        for resultado in resultados_similitudes:
            max_similitud = max(resultado['Similitudes'].values())
            archivo = resultado['Archivo']
            
            if not plagios_reales[plagios_reales['Archivo'] == archivo].empty:
                plagiado = plagios_reales.loc[plagios_reales['Archivo'] == archivo, 'Plagiado'].values[0]
                if plagiado == 1:  # El archivo es plagiado
                    if max_similitud >= umbral:
                        verdaderos_positivos += 1
                    else:
                        falsos_negativos += 1
                else:  # El archivo no es plagiado
                    if max_similitud >= umbral:
                        falsos_positivos += 1
                    else:
                        verdaderos_negativos += 1
            else:
                print(f"Advertencia: El archivo {archivo} no se encuentra en el archivo de plagios reales.")

        precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
        recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metricas.append((umbral, precision, recall, f1))

    metricas_df = pd.DataFrame(metricas, columns=['Umbral', 'Precision', 'Recall', 'F1'])

    # Determinar el umbral óptimo basado en F1
    umbral_optimo = metricas_df.loc[metricas_df['F1'].idxmax()]

    return metricas_df, umbral_optimo
