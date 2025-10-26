import numpy as np
import matplotlib.pyplot as plt


def graficar_matriz_confusion(matriz, nombres_clases):
    """
    Grafica la matriz de confusión con matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matriz, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Ejes
    ax.set(xticks=np.arange(matriz.shape[1]),
           yticks=np.arange(matriz.shape[0]),
           xticklabels=nombres_clases,
           yticklabels=nombres_clases,
           xlabel='Predicción',
           ylabel='Etiqueta Real',
           title='Matriz de Confusión')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Los valores en cada celda
    thresh = matriz.max() / 2.
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, format(matriz[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if matriz[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()

def graficar_perdida(perdidas, titulo="Pérdida durante el entrenamiento"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perdidas) + 1), perdidas, marker='o', linestyle='-', color='blue')
    plt.title(titulo)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.show()