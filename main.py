"""
Created on Sun Nov  2 12:54:18 2025

@author: Eduardo Contreras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Kernel_CD import get_cd_kernel

#Introducir datos
datos=np.genfromtxt("Datos/spiraldata.txt", delimiter=';')

#Separar columnas
x1 = datos[:, 0]
x2 = datos[:, 1]
Clases = datos[:, 2]

X = np.column_stack((x1, x2))
y = Clases

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
#Elegir el kernel y el tipo de polinomio
#Polinomios a elegir: legendre, chebyshev, hermite

kernel = get_cd_kernel( degree=4, poly_type="legendre", combine="product", normalize=False)

#Realizamos la clasificación con el k-fold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
models = [] #Con estos guardamos tanto los scores como los modelos de cada k-fold
fold=1

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx] 
    y_train, y_test = y[train_idx], y[test_idx]

    SVM = SVC(kernel=kernel, C=1)
    SVM.fit(X_train, y_train)
   
    y_pred = SVM.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    models.append((SVM, X_train, y_train))
    print(f"Fold {fold}: accuracy = {acc:.3f}")
    fold += 1

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
print(f"\nMedia de accuracy (5-fold): {mean_acc:.3f} ± {std_acc:.3f}")

#Para graficar, nos quedamos con el modelo del k-fold con una mayor precisión
best_idx = np.argmax(scores)
clf_best, X_train_best, y_train_best = models[best_idx]

print(f"\nUsando modelo del Fold {best_idx+1} para visualización (accuracy={scores[best_idx]:.3f})")
# Para obtener las coordenadas de los vectores soporte usando support_:
sv_indices = clf_best.support_               # índices relativos a X_train_best
support_coords = X_train_best[sv_indices]    # (n_sv, 2)

#Graficamos el modelo
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.2, X[:,0].max()+0.2, 400),
    np.linspace(X[:,1].min()-0.2, X[:,1].max()+0.2, 400)
)

Z = clf_best.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7,6))
# Frontera de decisión
plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
# Líneas de margen
plt.contour(xx, yy, Z, levels=[-1, 1], colors='k', linestyles='--', linewidths=1)

# Puntos de cada clase (colores)
plt.scatter(X[y==1,0], X[y==1,1], color='orangered', label='Clase +1', s=40, edgecolor='k')
plt.scatter(X[y==-1,0], X[y==-1,1], color='royalblue', label='Clase -1', s=40, edgecolor='k')

# Vectores soporte
plt.scatter(support_coords[:,0], 
            support_coords[:,1],
            s=120, facecolors='none', edgecolors='k', linewidths=1.5,
            label='Vectores soporte')

plt.title(f"SVM con Kernel de Christoffel–Darboux (Legendre)\nModelo del Fold {best_idx+1}")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.7)
plt.axis("equal")
plt.tight_layout()
plt.show()