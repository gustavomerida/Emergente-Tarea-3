import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from NeuralNetwork import RedRectangular, RedConvolucional
from Functions import train, probar_con_metricas, guardar_modelo, cargar_modelo, graficar_matriz_confusion, graficar_perdida

def main():
    print("\n" + "="*60)
    print("\033[1m\033[34m🧠 Clasificador Fashion MNIST con PyTorch 🤖\033[0m")
    print("="*60)
    print("\033[1m\t👤 Por: Gustavo Mérida C.I: 29948276\n\033[0m")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\t\t\t💻 Usando dispositivo: {device}\n")
    
    # Cargar dataset
    train_loader, test_loader = cargar_dataset()
    
    model = None
    nombres_clases = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    while True:
        print("\n" + "-"*60)
        print("\033[1m📋 MENÚ PRINCIPAL\033[0m")
        print("-"*60)
        print("1. 🆕 Crear nueva red neuronal")
        print("2. 📂 Cargar red desde archivo .pth")
        print("3. 🔥 Entrenar la red actual")
        print("4. 🧪 Probar la red actual")
        print("5. 💾 Guardar la red actual")
        print("6. 🚪 Salir")
        print("-"*60)
        
        opcion = input("Selecciona una opción (1-6): ")
        
        if opcion == '1':
            model = crear_red_interactivo(device)
        
        elif opcion == '2':
            if model is not None:
                print("\t\t⚠️  Ya tienes un modelo cargado. ¿Deseas reemplazarlo? (s/n)")
                if input().lower() != 's':
                    continue
            model = cargar_modelo_interactivo(device)
        
        elif opcion == '3':
            if model is None:
                print("\t\t\t❌ Error: Primero debes crear o cargar un modelo")
                continue
            
            epochs = int(input("\t\t\t¿Cuántas épocas deseas entrenar? (default: 10): ") or "10")
            
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            print("\n\t\t\t🚀 Iniciando entrenamiento...\n")
            perdidas = train(model, train_loader, loss_fn, optimizer, device, epochs)
            
            print("\n\t\t\t✅ Entrenamiento completado!")
            graficar_perdida(perdidas)
        
        elif opcion == '4':
            if model is None:
                print("\t\t\t❌ Error: Primero debes crear o cargar un modelo")
                continue
            
            print("\n\t\t\t🧪 Evaluando modelo...\n")
            accuracy = probar_con_metricas(model, test_loader, device, nombres_clases)
        
        elif opcion == '5':
            if model is None:
                print("\t\t\t❌ Error: No hay modelo para guardar")
                continue
            
            nombre = input("\t\t\tNombre del archivo (sin extensión): ")
            guardar_modelo(model, f"{nombre}.pth")
        
        elif opcion == '6':
            print("\n" + "="*60)
            print("\033[1m\033[34m👋 ¡Gracias por usar el clasificador! ¡Chaito!\033[0m")
            print("="*60)
            break
        
        else:
            print("\t\t\t❌ Opción inválida. Intenta de nuevo.")

def crear_red_interactivo(device):
    print("\n\t\t\t🔧 Configuración de nueva red")
    
    tipo = input("\t\t\tTipo de red (1: Rectangular, 2: Convolucional): ")
    while tipo not in ['1', '2']:
        tipo = input("\t\t\t❌ Opción inválida. Escoge 1 o 2: ")
    
    num_clases = input("\t\t\tNúmero de clases (1: 10 clases, 2: 4 clases agrupadas) [default: 1]: ") or "1"
    salida = 10 if num_clases == '1' else 4
    
    capas = int(input("\t\t\tNúmero de capas ocultas [default: 2]: ") or "2")
    
    if tipo == '1':  # Rectangular
        neuronas_str = input(f"\t\t\tNeuronas por capa separadas por comas (Ej: 512,512) [default: 512 en todas]: ")
        if neuronas_str:
            neuronas = [int(n.strip()) for n in neuronas_str.split(',')]
        else:
            neuronas = None
        
        model = RedRectangular(salida=salida, capas_ocultas=capas, neuronas_por_capa=neuronas)
    
    else:  # Convolucional
        model = RedConvolucional(salida=salida, capas_ocultas=capas)
    
    # Guardar metadata en la instancia para permitir guardado/carga automáticos
    arch = 'rectangular' if tipo == '1' else 'convolucional'
    if arch == 'rectangular':
        config = { 'entrada': 784, 'salida': salida, 'capas_ocultas': capas, 'neuronas_por_capa': neuronas }
    else:
        # Para convolucional guardamos los parámetros que el constructor usa
        config = { 'salida': salida, 'capas_ocultas': capas }

    model._arch = arch
    model._config = config

    model.to(device)
    print(f"\n\t\t\t✅ Red creada exitosamente!")
    print(f"\t\t\tTipo: {'Rectangular' if tipo == '1' else 'Convolucional'}")
    print(f"\t\t\tClases de salida: {salida}")
    
    return model

def cargar_modelo_interactivo(device):
    """
    Carga un modelo desde archivo .pth de forma interactiva
    """
    print("\n\t\t📂 Cargando modelo desde archivo")
    
    ruta = input("\t\tNombre del archivo .pth (Ej: mi_modelo.pth): ")

    # Verificar que el archivo existe
    import os
    if not os.path.exists(ruta):
        print(f"\t\t❌ Error: El archivo '{ruta}' no existe")
        return None

    # Intentar carga automática usando metadata guardada en el checkpoint
    try:
        model = cargar_modelo(ruta, device=device)
        # Si la carga automática funciona, el modelo ya incluye _arch/_config
        model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"\t\t\t⚠️  Carga automática falló: {e}")
        print("\t\t\tSe intentará la carga manual pidiéndote la configuración del modelo (fallback)")

    # Fallback: pedir configuración manualmente (compatibilidad hacia atrás)
    tipo = input("\t\t\tTipo de red del modelo (1: Rectangular, 2: Convolucional): ")
    while tipo not in ['1', '2']:
        tipo = input("\t\t\t❌ Opción inválida. Escoge 1 o 2: ")

    # Preguntar configuración (debe coincidir con la del modelo guardado)
    num_clases = input("\t\t\tNúmero de clases del modelo (1: 10 clases, 2: 4 clases) [default: 1]: ") or "1"
    salida = 10 if num_clases == '1' else 4

    capas = int(input("\t\t\tNúmero de capas ocultas del modelo [default: 2]: ") or "2")

    try:
        if tipo == '1':  # Rectangular
            neuronas_str = input(f"\t\t\tNeuronas por capa del modelo (Ej: 512,512) [default: 512 en todas]: ")
            if neuronas_str:
                neuronas = [int(n.strip()) for n in neuronas_str.split(',')]
            else:
                neuronas = None

            model = RedRectangular(salida=salida, capas_ocultas=capas, neuronas_por_capa=neuronas)
            arch = 'rectangular'
            config = { 'entrada': 784, 'salida': salida, 'capas_ocultas': capas, 'neuronas_por_capa': neuronas }

        else:  # Convolucional
            model = RedConvolucional(salida=salida, capas_ocultas=capas)
            arch = 'convolucional'
            config = { 'salida': salida, 'capas_ocultas': capas }

        # Cargar los pesos (archivo puede contener checkpoint con key 'state_dict' o directamente state_dict)
        ckpt = torch.load(ruta, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        model.load_state_dict(state)
        model._arch = arch
        model._config = config
        model.to(device)
        model.eval()  # Modo evaluación

        print(f"\n\t\t\t✅ Modelo cargado exitosamente desde: {ruta} (modo fallback)")
        print(f"\t\t\tTipo: {'Rectangular' if tipo == '1' else 'Convolucional'}")
        print(f"\t\t\tClases de salida: {salida}")

        return model

    except Exception as e:
        print(f"\t\t\t❌ Error al cargar el modelo: {e}")
        print("\t\t\tAsegúrate de que la configuración coincida con la del modelo guardado")
        return None

def cargar_dataset(clases_agrupadas=False, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


main()
