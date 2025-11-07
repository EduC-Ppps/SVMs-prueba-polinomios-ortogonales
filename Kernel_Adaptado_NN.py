"""
Created on Fri Nov  7 12:26:36 2025

@author: Eduardo Contreras
"""

import numpy as np
import math
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.hermite import hermval
import torch
import torch.nn as nn

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


def christoffel_activation_np(x, degree=4, poly_type="legendre"):
    """
    Versión numpy: usa los polinomios ortogonales para generar una activación.
    Inspirada en la diagonal del kernel Christoffel–Darboux.
    """
    poly_func = _get_poly_func(poly_type)
    Pn   = poly_func(x, [0]*degree + [1])
    Pnp1 = poly_func(x, [0]*(degree+1) + [1])
    return Pn**2 + Pnp1**2

# ======================================================
#   ACTIVACIÓN EN PyTorch
# ======================================================

class ChristoffelActivation(nn.Module):
    """
    Activación personalizada inspirada en el kernel Christoffel–Darboux.
    Puede usarse dentro de redes neuronales de PyTorch.
    """
    def __init__(self, degree=4, poly_type="legendre"):
        super().__init__()
        self.degree = degree
        self.poly_type = poly_type

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        y_np = christoffel_activation_np(x_np, self.degree, self.poly_type)
        return torch.tensor(y_np, device=x.device, dtype=x.dtype)