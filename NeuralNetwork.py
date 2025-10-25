import torch.nn as nn

class RedRectangular(nn.Module):
    """Red neuronal completamente conectada (fully connected)"""
    def __init__(self, entrada=784, salida=10, capas_ocultas=2, neuronas_por_capa=None):
        super(RedRectangular, self).__init__()
        
        # Valores por defecto: todas las capas con 512 neuronas
        if neuronas_por_capa is None:
            neuronas_por_capa = [512] * capas_ocultas
        
        # Validación
        if len(neuronas_por_capa) != capas_ocultas:
            raise ValueError(f"❌ Error: {capas_ocultas} capas especificadas pero {len(neuronas_por_capa)} valores de neuronas")
        
        layers = []
        # Primera capa oculta
        layers.append(nn.Linear(entrada, neuronas_por_capa[0]))
        layers.append(nn.ReLU())
        
        # Capas ocultas intermedias
        for i in range(1, capas_ocultas):
            layers.append(nn.Linear(neuronas_por_capa[i-1], neuronas_por_capa[i]))
            layers.append(nn.ReLU())
        
        # Capa de salida
        layers.append(nn.Linear(neuronas_por_capa[-1], salida))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Aplanar
        return self.model(x)


class RedConvolucional(nn.Module):
    """Red neuronal convolucional para Fashion MNIST"""
    def __init__(self, salida=10, capas_ocultas=2):
        super(RedConvolucional, self).__init__()
        
        # Calcular neuronas por capa con interpolación lineal
        entrada = 784
        neuronas = self._interpolacion_lineal(entrada, salida, capas_ocultas)
        
        # Capas convolucionales (arquitectura típica para 28x28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Después de 2 max pools: 28x28 -> 14x14 -> 7x7
        # 64 canales * 7 * 7 = 3136
        
        # Capas fully connected usando interpolación
        fc_layers = []
        fc_layers.append(nn.Linear(64 * 7 * 7, neuronas[0]))
        fc_layers.append(nn.ReLU())
        
        for i in range(1, len(neuronas)):
            fc_layers.append(nn.Linear(neuronas[i-1], neuronas[i]))
            fc_layers.append(nn.ReLU())
        
        fc_layers.append(nn.Linear(neuronas[-1], salida))
        
        self.fc = nn.Sequential(*fc_layers)
    
    def _interpolacion_lineal(self, entrada, salida, capas):
        """Calcula neuronas por capa usando interpolación lineal"""
        if capas == 0:
            return []
        
        # Calcular el paso entre capas
        paso = (entrada - salida) / (capas + 1)
        neuronas = []
        
        for i in range(1, capas + 1):
            n = int(entrada - paso * i)
            neuronas.append(n)
        
        return neuronas
    
    def forward(self, x):
        # Capas convolucionales
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Aplanar
        x = x.view(-1, 64 * 7 * 7)
        
        # Capas fully connected
        x = self.fc(x)
        return x
