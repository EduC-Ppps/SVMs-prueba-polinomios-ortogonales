"""
Created on Fri Nov  7 12:17:38 2025

@author: Eduardo Contreras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from Kernel_Adaptado_NN import ChristoffelActivation


#Introducir datos
datos=np.genfromtxt("Datos/spiraldata.txt", delimiter=';')

#Separar columnas
x1 = datos[:, 0]
x2 = datos[:, 1]
Clases = datos[:, 2]

X = np.column_stack((x1, x2))
y = Clases

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Pasar a tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#Crear figura
plt.figure(figsize=(7,6))

#Graficar
plt.scatter(x1, x2, c=Clases, cmap='coolwarm', s=40, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Datos no lineales')
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################################
#Crear la red neuronal
ModeloNN= nn.Sequential(
    nn.Linear(2,120), #Red pequeña con 2 entradas y 20 salidas
    ChristoffelActivation(degree=3, poly_type="chebyshev"),
    nn.Linear(120,1),
    nn.Tanh() #Para que la salida esté entre -1 y 1
    )

#Entrenar la red
criterion = nn.MSELoss()
optimizer = optim.Adam(ModeloNN.parameters(), lr=0.01)

for epoch in range(60000):
    optimizer.zero_grad()
    output = ModeloNN(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
with torch.no_grad():
    y_pred_test = ModeloNN(X_test_tensor)
    test_loss = criterion(y_pred_test, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")       
    
#Convertir a clases (-1 o 1) según el signo
y_pred_classes = torch.sign(y_pred_test).numpy().flatten()
accuracy_val = accuracy_score(y_test, y_pred_classes)
print(f"\nTest Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {accuracy_val * 100:.2f}%")

# ==============================================
#  Visualización de la frontera de decisión
# ==============================================
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# Crear malla
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = ModeloNN(grid).numpy().ravel()
Z = Z.reshape(xx.shape)

# ==============================================
#  Gráfico
# ==============================================
plt.figure(figsize=(7, 6))

# Línea negra de frontera (donde salida ≈ 0)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# Todos los puntos (train + test) juntos y del mismo estilo
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=40)

plt.title(f"Frontera de decisión con activación Christoffel–Chebyshev\nAccuracy: {accuracy_val*100:.2f}%")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.tight_layout()
plt.show()