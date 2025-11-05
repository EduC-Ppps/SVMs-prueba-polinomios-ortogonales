"""
Created on Sun Nov  2 12:54:13 2025

@author: Eduardo Contreras
"""
import numpy as np
import math
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.hermite import hermval

# ======================================================
#   FUNCIONES AUXILIARES
# ======================================================

def _get_poly_func(poly_type):
    """Devuelve la función evaluadora correspondiente al tipo de polinomio."""
    if poly_type == "legendre":
        return legval
    elif poly_type == "chebyshev":
        return chebval
    elif poly_type == "hermite":
        return hermval
    else:
        raise ValueError("Tipo de polinomio no válido. Usa 'legendre', 'chebyshev' o 'hermite'.")

def _cd_diagonal(x, degree, poly_type):
    """
    Evalúa la suma de Christoffel-Darboux cuando x = y (caso diagonal).
    """
    poly_func = _get_poly_func(poly_type)
    n = degree
    k = np.arange(n + 1)
    
    # Evaluamos todos los P_k(x)
    P = np.array([poly_func(x, [0]*ki + [1]) for ki in k])
    
    # Pesos 1/h_k según ortogonalidad estándar
    if poly_type == "legendre":
        inv_hk = (2*k + 1)/2.0
    elif poly_type == "chebyshev":
        inv_hk = np.pi * np.ones_like(k) / np.where(k == 0, 2, 1)  # forma clásica
    elif poly_type == "hermite":
        inv_hk = 1.0 / (np.sqrt(np.pi) * (2**k) * np.array([math.factorial(i) for i in k]))
    
    # Suma ponderada
    Sxx = np.sum(inv_hk[:, None] * (P**2), axis=0)
    return Sxx


# ======================================================
#   KERNEL CD 1D
# ======================================================

def cd_kernel_1d(x, y, degree=4, poly_type="legendre", tol=1e-10):
    """
    Kernel Christoffel–Darboux unidimensional basado en el tipo de polinomio.
    """
    poly_func = _get_poly_func(poly_type)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = degree

    # Evaluamos P_n y P_{n+1}
    Pn_x   = poly_func(x, [0]*n + [1])
    Pnp1_x = poly_func(x, [0]*(n+1) + [1])
    Pn_y   = poly_func(y, [0]*n + [1])
    Pnp1_y = poly_func(y, [0]*(n+1) + [1])

    # Fórmula de Christoffel–Darboux general
    num = np.outer(Pnp1_x, Pn_y) - np.outer(Pn_x, Pnp1_y)
    den = x[:, None] - y[None, :]

    K = np.empty_like(den)
    mask = np.abs(den) < tol

    # Parte no diagonal
    K[~mask] = num[~mask] / den[~mask]
    # Parte diagonal (corregido)
    K[mask] = _cd_diagonal(x[mask.nonzero()[0]], degree, poly_type)

    return K


# ======================================================
#   KERNEL CD MULTIDIMENSIONAL (OPCIONAL)
# ======================================================

def cd_kernel(X, Y, degree=4, poly_type="legendre", combine="product", normalize=True):
    """
    Extiende el kernel CD 1D a datos multidimensionales (por dimensión).
    combine: 'sum' o 'product'
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    n, d = X.shape
    m = Y.shape[0]

    K = np.ones((n, m)) if combine == "product" else np.zeros((n, m))

    for j in range(d):
        Kj = cd_kernel_1d(X[:, j], Y[:, j], degree=degree, poly_type=poly_type)
        if combine == "product":
            K *= Kj
        else:
            K += Kj
    if combine == "sum":
        K /= d

    # Normalización tipo coseno
    if normalize:
        diag_X = np.sqrt(np.diag(cd_kernel(X, X, degree, poly_type, combine, False)) + 1e-12)
        diag_Y = np.sqrt(np.diag(cd_kernel(Y, Y, degree, poly_type, combine, False)) + 1e-12)
        K = K / (diag_X[:, None] * diag_Y[None, :] + 1e-12)

    return K


# ======================================================
#   INTERFAZ DE USO PARA SVM
# ======================================================

def get_cd_kernel(degree=4, poly_type="legendre", combine="product", normalize=True):
    """
    Devuelve una función kernel lista para usar con sklearn.SVC(kernel=...).
    """
    return lambda X, Y: cd_kernel(X, Y, degree=degree, poly_type=poly_type,
                                  combine=combine, normalize=normalize)