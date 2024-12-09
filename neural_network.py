import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Cargando el conjunto de datos MNIST
train = datasets.MNIST('', train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor()]))  # Convierte las imágenes a tensores
test = datasets.MNIST('', train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor()]))

# Creando cargadores de datos para dividir el conjunto en lotes
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)  # Datos de entrenamiento en lotes de tamaño 10
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)   # Datos de prueba en lotes de tamaño 10

# Definiendo la red neuronal
class Net(nn.Module):
    def __init__(self):
        super().__init__()  # Inicializa la clase base nn.Module
        self.layer1 = nn.Linear(28*28, 64)  # Capa densa que toma entradas de tamaño 28x28 (784) y las reduce a 64
        self.layer2 = nn.Linear(64, 64)     # Segunda capa oculta de tamaño 64
        self.layer3 = nn.Linear(64, 64)     # Tercera capa oculta de tamaño 64
        self.layer4 = nn.Linear(64, 10)     # Capa de salida con 10 neuronas (una por cada clase del dígito)

    def forward(self, x):
        x = F.relu(self.layer1(x))  # Pasa la entrada por la primera capa y aplica la activación ReLU
        x = F.relu(self.layer2(x))  # Segunda capa con activación ReLU
        x = F.relu(self.layer3(x))  # Tercera capa con activación ReLU
        x = self.layer4(x)          # Capa de salida sin activación ReLU
        return F.log_softmax(x, dim=1)  # Aplica log_softmax para obtener probabilidades logarítmicas

net = Net()  # Instancia de la red neuronal
print(net)   # Muestra la arquitectura de la red

# Definiendo la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()  # Calcula la pérdida basada en la entropía cruzada
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Usa el optimizador Adam con una tasa de aprendizaje de 0.001

# Entrenamiento del modelo
for epoch in range(3):  # Realiza 3 épocas de entrenamiento
    for data in trainset:  # Itera sobre los lotes de datos
        X, y = data  # Divide cada lote en datos de entrada (X) y etiquetas (y)
        net.zero_grad()  # Reinicia los gradientes acumulados
        output = net(X.view(-1, 784))  # Reformatea X a un vector de tamaño 784 (entrada plana)
        loss = F.nll_loss(output, y)  # Calcula la pérdida negativa log-likelihood
        loss.backward()  # Calcula los gradientes de los parámetros
        optimizer.step()  # Actualiza los pesos según los gradientes
    print(loss)  # Imprime la pérdida al final de cada época

# Evaluación del modelo
correct = 0  # Contador de predicciones correctas
total = 0    # Contador total de predicciones

with torch.no_grad():  # Desactiva el cálculo de gradientes para evaluación
    for data in testset:  # Itera sobre los lotes de datos de prueba
        X, y = data
        output = net(X.view(-1, 784))  # Pasa los datos por la red
        for idx, i in enumerate(output):  # Itera sobre las salidas
            if torch.argmax(i) == y[idx]:  # Compara la predicción con la etiqueta real
                correct += 1  # Incrementa si es correcta
            total += 1  # Incrementa el total de predicciones

print("Exactitud: ", round(correct/total, 3))  # Imprime la precisión del modelo

# Visualización de una imagen y su predicción
plt.imshow(X[0].view(28,28))  # Muestra la primera imagen del último lote en formato 28x28
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))  # Imprime la predicción de la red para la imagen mostrada
