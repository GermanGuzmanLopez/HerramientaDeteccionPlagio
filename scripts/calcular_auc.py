import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

def calcular_auc(resultados_similitudes, plagios_reales):
    etiquetas_reales = []
    predicciones = []

    for resultado in resultados_similitudes:
        max_similitud = max(resultado['Similitudes'].values())
        archivo = resultado['Archivo']

        if not plagios_reales[plagios_reales['Archivo'] == archivo].empty:
            plagiado = plagios_reales.loc[plagios_reales['Archivo'] == archivo, 'Plagiado'].values[0]
            
            #etiquetas reales y predicciones
            etiquetas_reales.append(plagiado)
            predicciones.append(max_similitud)

    fpr, tpr, _ = roc_curve(etiquetas_reales, predicciones)
    roc_auc = auc(fpr, tpr)

    tabla_roc = pd.DataFrame({
        'Falsos Positivos (FPR)': fpr,
        'Verdaderos Positivos (TPR)': tpr
    })

    print (tabla_roc)
    # Graficar la curva
    plt.figure()
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsos positivos')
    plt.ylabel('Verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc
