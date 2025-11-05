"""
Created on Sun Nov  2 12:54:13 2025

@author: Eduardo Contreras
"""
import numpy as np
from numpy.polynomial.legendre import legval as legendre_val
from numpy.polynomial.chebyshev import chebval as chebyshev_val
from numpy.polynomial.hermite import hermval as hermite_val

# --------------------------------------------------------
#  Selección del tipo de polinomio
# --------------------------------------------------------
def get_poly_eval(poly_type):
    """
    Devuelve la función de evaluación de polinomios correspondiente.
    """
    poly_type = poly_type.lower()
    if poly_type == "legendre":
        return legendre_val
    elif poly_type == "chebyshev":
        return chebyshev_val
    elif poly_type == "hermite":
        return hermite_val
    else:
        raise ValueError("Tipo de polinomio no válido. Usa: 'legendre', 'chebyshev' o 'hermite'.")


# --------------------------------------------------------
#  Christoffel–Darboux 1D
# --------------------------------------------------------
def cd_kernel_1d(x, y, degree=4, poly_type="legendre", tol=1e-10):
    """
    Calcula el kernel de Christoffel–Darboux 1D para el tipo de polinomio indicado.
    """
    poly_eval = get_poly_eval(poly_type)
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    n = degree

    # Evaluar P_n y P_{n+1}
    Pn_x = poly_eval(x, [0]*n + [1])
    Pn1_x = poly_eval(x, [0]*(n+1) + [1])
    Pn_y = poly_eval(y, [0]*n + [1])
    Pn1_y = poly_eval(y, [0]*(n+1) + [1])

    num = np.outer(Pn1_x, Pn_y) - np.outer(Pn_x, Pn1_y)
    den = x[:, None] - y[None, :]

    # Calcular el límite cuando x ≈ y
    mask = np.abs(den) < tol
    K = np.empty_like(den)
    K[~mask] = num[~mask] / den[~mask]
    K[mask] = np.diag(_cd_diagonal(x[mask.nonzero()[0]], degree, poly_type))

    return K


def _cd_diagonal(x, degree, poly_type):
    """
    Evalúa el límite del kernel CD cuando x → y.
    """
    poly_eval = get_poly_eval(poly_type)
    k = np.arange(degree + 1)
    P = np.array([poly_eval(x, [0]*i + [1]) for i in k])
    weights = (2*k + 1)/2  # Normalización genérica (válida para Legendre)
    return np.sum(weights[:, None] * P**2, axis=0)


# --------------------------------------------------------
#  Christoffel–Darboux multidimensional
# --------------------------------------------------------
def cd_kernel(X, Y=None, degree=4, poly_type="legendre", combine="product", normalize=True):
    """
    Kernel CD multidimensional (tensorial o suma promediada).
    combine = 'product' → producto tensorial
    combine = 'sum'     → promedio por dimensión
    """
    X = np.atleast_2d(X)
    Y = X if Y is None else np.atleast_2d(Y)

    n, d = X.shape[0], X.shape[1]
    m = Y.shape[0]

    # Inicializar matriz kernel
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
        Kxx = np.diag(cd_kernel(X, degree=degree, poly_type=poly_type, combine=combine, normalize=False))
        Kyy = np.diag(cd_kernel(Y, degree=degree, poly_type=poly_type, combine=combine, normalize=False))
        K /= np.sqrt(np.outer(Kxx, Kyy)) + 1e-12

    # Simetrizar si es autocorrelación
    if Y is X:
        K = 0.5 * (K + K.T)

    return K


# --------------------------------------------------------
#  Función
# --------------------------------------------------------
def get_cd_kernel(degree=4, poly_type="legendre", combine="product", normalize=True):
    """
    Devuelve una función kernel lista para usar en SVM de sklearn.
    Ejemplo:
        from sklearn.svm import SVC
        kernel = get_cd_kernel(degree=5, poly_type='hermite')
        clf = SVC(kernel=kernel)
    """
    return lambda X, Y: cd_kernel(X, Y, degree=degree, poly_type=poly_type,
                                  combine=combine, normalize=normalize)