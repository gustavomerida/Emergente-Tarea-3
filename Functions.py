import torch
import torch.nn as nn   
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from NeuralNetwork import RedRectangular, RedConvolucional
import numpy as np
from Graphics import graficar_matriz_confusion


def mapear_clases_agrupadas(labels):
    """
    Mapea las 10 clases originales a 4 clases agrupadas:
    Top (0): T-shirt, Pullover, Coat, Shirt (originales: 0, 2, 4, 6)
    Bottom (1): Trouser, Dress (originales: 1, 3)
    Footwear (2): Sandal, Sneaker, Ankle boot (originales: 5, 7, 9)
    Bag (3): Bag (original: 8)
    """
    mapeo = {
        0: 0,  # T-shirt -> Top
        2: 0,  # Pullover -> Top
        4: 0,  # Coat -> Top
        6: 0,  # Shirt -> Top
        1: 1,  # Trouser -> Bottom
        3: 1,  # Dress -> Bottom
        5: 2,  # Sandal -> Footwear
        7: 2,  # Sneaker -> Footwear
        9: 2,  # Ankle boot -> Footwear
        8: 3   # Bag -> Bag
    }
    
    return torch.tensor([mapeo[label.item()] for label in labels])

def train(model, dataloader, loss_fn, optimizer, device, epochs=5):
    model.to(device)
    model.train()
    
    perdidas_por_epoca = [] 
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)

            salida_model = None
            if hasattr(model, '_config') and isinstance(model._config, dict):
                salida_model = model._config.get('salida')

            if salida_model == 4:
                labels = mapear_clases_agrupadas(labels).to(device)
            else:
                labels = labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        perdidas_por_epoca.append(avg_loss)
        print(f"\t\t\t🔥 Época {epoch+1}/{epochs} - Pérdida: {avg_loss:.4f}")
    
    return perdidas_por_epoca

def calcular_metricas(matriz_confusion):
    """
    Calcula accuracy, precisión y recall desde la matriz de confusión
    """
    # Accuracy
    accuracy = np.trace(matriz_confusion) / np.sum(matriz_confusion)
    
    # Precisión y recall por clase
    num_clases = matriz_confusion.shape[0]
    precisiones = []
    recalls = []
    
    for i in range(num_clases):
        tp = matriz_confusion[i, i]
        fp = np.sum(matriz_confusion[:, i]) - tp
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        precisiones.append(precision)
        
        fn = np.sum(matriz_confusion[i, :]) - tp
        
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        recalls.append(recall)
    
    total_muestras = np.sum(matriz_confusion)
    muestras_por_clase = np.sum(matriz_confusion, axis=1)
    
    precision_promedio = np.sum([p * n for p, n in zip(precisiones, muestras_por_clase)]) / total_muestras
    recall_promedio = np.sum([r * n for r, n in zip(recalls, muestras_por_clase)]) / total_muestras
    
    return accuracy, precision_promedio, recall_promedio

def calcular_matriz_confusion(predicciones, etiquetas_reales, num_clases):
    """
    Calcula la matriz de confusión manualmente
    predicciones: tensor o lista con las predicciones
    etiquetas_reales: tensor o lista con las etiquetas que en realidad eran
    num_clases: número de clases (10 o 4)
    """
    matriz = torch.zeros(num_clases, num_clases, dtype=torch.int64)
    
    for pred, real in zip(predicciones, etiquetas_reales):
        matriz[real, pred] += 1
    
    return matriz.numpy()

def probar_con_metricas(model, dataloader, device, nombres_clases):
    """Prueba el modelo y calcula métricas como accuracy, precision, recall y matriz de confusión"""
    model.to(device)
    model.eval()
    
    todas_predicciones = []
    todas_etiquetas = []
    
    print("\n\t\t\t🧪 Evaluando modelo en conjunto de prueba...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            salida_model = None
            if hasattr(model, '_config') and isinstance(model._config, dict):
                salida_model = model._config.get('salida')

            if salida_model == 4:
                etiquetas_mapeadas = mapear_clases_agrupadas(labels.cpu()).numpy()
                todas_etiquetas.extend(etiquetas_mapeadas)
            else:
                todas_etiquetas.extend(labels.cpu().numpy())

            todas_predicciones.extend(predicted.cpu().numpy())
    
    # Calcular matriz de confusión, métrics y mostart resultados
    num_clases = len(nombres_clases)
    matriz = calcular_matriz_confusion(todas_predicciones, todas_etiquetas, num_clases)

    accuracy, precision, recall = calcular_metricas(matriz)
    
    print(f"\n\t\t\t📊 Métricas del modelo:")
    print(f"\t\t\t   • Accuracy: {accuracy * 100:.2f}%")
    print(f"\t\t\t   • Precisión (promedio): {precision * 100:.2f}%")
    print(f"\t\t\t   • Recall (promedio): {recall * 100:.2f}%")
    
    # Graficar
    graficar_matriz_confusion(matriz, nombres_clases)
    
    return accuracy

def guardar_modelo(model, ruta):
    """Guarda el modelo en formato .pth"""
    # Intentar guardar metadata (arquitectura y configuración) si el modelo la contiene
    arch = getattr(model, '_arch', None)
    config = getattr(model, '_config', None)

    if arch is not None and config is not None:
        checkpoint = {
            'arch': arch,
            'config': config,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, ruta)
        print(f"\t\t\t💾 Modelo y metadata guardados en: {ruta}")
    else:
        # Fallback: guardar solo state_dict
        torch.save({'state_dict': model.state_dict()}, ruta)
        print(f"\t\t\t💾 Solo state_dict guardado en: {ruta} (falta metadata de arquitectura)")

def cargar_modelo(ruta, device=None, tipo_red=None, **kwargs):
    """Carga un modelo desde archivo .pth.

    Comportamientos:
    - Si el archivo es un checkpoint con keys 'arch' y 'config', reconstruye el modelo automáticamente.
    - Si el archivo solo contiene 'state_dict', requiere `tipo_red` y kwargs para reconstruir.
    - `device` puede ser una torch.device o una cadena aceptada por torch.load map_location.
    """
    map_loc = device if device is not None else 'cpu'
    checkpoint = torch.load(ruta, map_location=map_loc)

    # Caso: checkpoint con metadata
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']

        if 'arch' in checkpoint and 'config' in checkpoint:
            arch = checkpoint['arch']
            config = checkpoint['config']

            if arch == 'rectangular':
                model = RedRectangular(**config)
            elif arch == 'convolucional':
                model = RedConvolucional(**config)
            else:
                raise ValueError(f"Arquitectura desconocida en checkpoint: {arch}")

            model.load_state_dict(state)
            model._arch = arch
            model._config = config
            model.to(device or 'cpu')
            model.eval()
            print(f"\t\t\t📂 Modelo '{arch}' cargado desde: {ruta}")
            return model

        # Caso: solo state_dict (sin metadata)
        else:
            if tipo_red is None:
                raise ValueError("El archivo no contiene metadata. Provee 'tipo_red' y kwargs para reconstruir la arquitectura.")

            if tipo_red == 'rectangular':
                model = RedRectangular(**kwargs)
            elif tipo_red == 'convolucional':
                model = RedConvolucional(**kwargs)
            else:
                raise ValueError("Tipo de red debe ser 'rectangular' o 'convolucional'")

            model.load_state_dict(state)
            model._arch = tipo_red
            model._config = kwargs
            model.to(device or 'cpu')
            model.eval()
            print(f"\t\t\t📂 Modelo '{tipo_red}' cargado desde: {ruta} (sin metadata en archivo)")
            return model

    else:
        raise ValueError("Formato de archivo no reconocido: se esperaba un checkpoint dict con 'state_dict'.")


