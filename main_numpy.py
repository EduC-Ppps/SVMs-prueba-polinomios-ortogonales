# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:19:42 2025

@author: Eduardo Contreras
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from tkinter import (
    Tk, Label, filedialog, Text, END, Scale,
    HORIZONTAL, Entry, Frame, StringVar, OptionMenu
)
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.polynomial.chebyshev import chebval, chebder
from sklearn.preprocessing import LabelEncoder

# =========================================================
#   CHEBYSHEV ESTABILIZADO
# =========================================================

def cheb_stable_forward(z, degree):
    """
    Evaluación estable de la activación tipo Christoffel-Chebyshev.
    - Clampea la entrada con tanh para que esté en (-1,1)
    - Aplica polinomios de Chebyshev T_n y T_{n+1}
    - Aplica tanh a la salida para acotar
    - Calcula derivada total estable respecto a z
    """
    # Entrada estabilizada a (-1,1)
    z_safe = np.tanh(z)

    # Coeficientes para T_n y T_{n+1}
    coefs_n = np.zeros(degree + 1)
    coefs_n[-1] = 1.0
    coefs_np1 = np.zeros(degree + 2)
    coefs_np1[-1] = 1.0

    # Polinomios
    Tn = chebval(z_safe, coefs_n)
    Tnp1 = chebval(z_safe, coefs_np1)

    # Derivadas de los polinomios
    coefs_n_d = chebder(coefs_n)
    coefs_np1_d = chebder(coefs_np1)
    Tn_d = chebval(z_safe, coefs_n_d)
    Tnp1_d = chebval(z_safe, coefs_np1_d)

    # Kernel tipo Christoffel
    y = Tn**2 + Tnp1**2

    # Estabilizamos salida con tanh
    y_out = np.tanh(y)

    # dy/d(safe_z)
    dy_dsafe = 2*Tn*Tn_d + 2*Tnp1*Tnp1_d
    dy_dsafe = dy_dsafe * (1 - y_out**2)  # derivada de tanh en la salida

    # d(safe_z)/dz = 1 - tanh(z)^2
    dsafe_dz = 1 - z_safe**2

    dy_dz = dy_dsafe * dsafe_dz

    return y_out, dy_dz


# =========================================================
#   ACTIVACIONES GENERALES
# =========================================================

def activation_forward(z, name, degree=None):
    """
    Devuelve:
      a  = activación(z)
      da = derivada d(a)/d(z)
    """
    name = name.lower()

    if name == "cheb":
        return cheb_stable_forward(z, degree)

    # Clampeo suave para evitar overflow numérico
    z_safe = np.clip(z, -50, 50)

    if name == "tanh":
        a = np.tanh(z_safe)
        da = 1 - a**2
        return a, da

    if name == "relu":
        a = np.maximum(0, z_safe)
        da = (z_safe > 0).astype(float)
        return a, da

    if name == "silu":
        sig = 1/(1 + np.exp(-z_safe))
        a = z_safe*sig
        da = sig + z_safe*sig*(1-sig)
        return a, da

    if name == "elu":
        a = np.where(z_safe > 0, z_safe, np.exp(z_safe) - 1)
        da = np.where(z_safe > 0, 1, a + 1)
        return a, da

    if name in ["identity", "none"]:
        return z_safe, np.ones_like(z_safe)

    # Por defecto tanh
    a = np.tanh(z_safe)
    da = 1 - a**2
    return a, da


# =========================================================
#   RED NEURONAL DINÁMICA (NUMPY, ESTABLE)
# =========================================================

class ChebNetDynamicLiteStable:
    """
    Red neuronal totalmente en NumPy, estable numéricamente.

    Arquitectura:
      - Primera capa: Linear(input_dim -> max_neurons) + activación Cheb
      - Capas intermedias: reducen a la mitad con activación elegida
      - Última capa: Linear(... -> 1), sin activación (logit)
    """

    def __init__(self, input_dim, max_neurons, degree, activation_name):
        self.input_dim = input_dim
        self.max_neurons = max_neurons
        self.degree = degree
        self.activation_name = activation_name.lower()

        self.layers = []

        # Capa inicial (Chebyshev)
        W0 = np.random.randn(input_dim, max_neurons) * np.sqrt(2.0/input_dim)
        b0 = np.zeros(max_neurons)
        self.layers.append({
            "W": W0,
            "b": b0,
            "activation": "cheb",
            "degree": degree
        })

        current = max_neurons
        target_min = 15

        # Capas intermedias halving
        while current // 2 >= target_min:
            nxt = current // 2
            W = np.random.randn(current, nxt) * np.sqrt(2.0/current)
            b = np.zeros(nxt)
            self.layers.append({
                "W": W,
                "b": b,
                "activation": self.activation_name,
                "degree": None
            })
            current = nxt

        # Capa final → 1 neurona (logit)
        Wlast = np.random.randn(current, 1) * np.sqrt(2.0/current)
        blast = np.zeros(1)
        self.layers.append({
            "W": Wlast,
            "b": blast,
            "activation": "none",
            "degree": None
        })

    # ------------------------
    # Forward
    # ------------------------
    def forward(self, X):
        a = X
        caches = []

        for layer in self.layers:
            W, b = layer["W"], layer["b"]
            z = a @ W + b
            a_next, da_dz = activation_forward(z, layer["activation"], layer["degree"])
            caches.append({
                "a_prev": a,
                "z": z,
                "a": a_next,
                "da_dz": da_dz
            })
            a = a_next

        return a, caches

    # ------------------------
    # Predicción
    # ------------------------
    def predict_proba(self, X):
        logits, _ = self.forward(X)
        z = logits.reshape(-1)
        z = np.clip(z, -30, 30)
        probs = 1/(1 + np.exp(-z))
        return probs

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    # ------------------------
    # Entrenamiento
    # ------------------------
    def fit(self, X, y, lr=0.0003, epochs=2000, log_func=None):
        """
        Entrenamiento con Binary Cross Entropy y gradiente descendente
        estabilizado.
        """
        N = len(X)
        y = y.reshape(-1)

        for ep in range(epochs):
            # Forward
            logits, caches = self.forward(X)
            logits_flat = logits.reshape(-1)
            logits_flat = np.clip(logits_flat, -30, 30)
            probs = 1/(1 + np.exp(-logits_flat))

            # Pérdida BCE
            eps = 1e-8
            loss = -np.mean(y*np.log(probs+eps) + (1-y)*np.log(1-probs+eps))

            # dL/dlogit
            delta = (probs - y).reshape(-1,1)  # (N,1)

            grads_W = [None]*len(self.layers)
            grads_b = [None]*len(self.layers)

            # Backprop
            for l in reversed(range(len(self.layers))):
                cp = caches[l]
                da_dz = cp["da_dz"]
                a_prev = cp["a_prev"]

                delta_z = delta * da_dz
                # Clipping fuerte para evitar explosiones
                delta_z = np.clip(delta_z, -3, 3)

                dW = (a_prev.T @ delta_z)/N
                db = delta_z.mean(axis=0)

                dW = np.clip(dW, -1, 1)
                db = np.clip(db, -1, 1)

                grads_W[l] = dW
                grads_b[l] = db

                # Retropropagar
                delta = delta_z @ self.layers[l]["W"].T

            # Actualizar pesos
            for l in range(len(self.layers)):
                self.layers[l]["W"] -= lr*grads_W[l]
                self.layers[l]["b"] -= lr*grads_b[l]

            if log_func and ep % max(1, epochs//20) == 0:
                log_func(f"Epoch {ep}/{epochs} | Loss={loss:.5f}")


# =========================================================
#   APLICACIÓN TKINTER (LITE ESTABLE)
# =========================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal Dinámica Chebyshev (Lite, Estable)")
        self.root.configure(bg="#1e1e1e")

        # Estilo moderno
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Modern.TButton",
            font=("Segoe UI", 11),
            padding=8,
            borderwidth=0,
            foreground="white",
            background="#3a3a3a"
        )
        style.map(
            "Modern.TButton",
            background=[("active", "#555")],
            foreground=[("active", "white")]
        )

        # Título
        Label(
            root,
            text="Red Neuronal Dinámica basada en Chebyshev (Lite, NumPy Estable)",
            fg="white",
            bg="#1e1e1e",
            font=("Segoe UI", 16)
        ).pack(pady=12)

        # Marco principal
        main_frame = Frame(self.root, bg="#1e1e1e")
        main_frame.pack(pady=10)

        # Panel de parámetros (izquierda)
        params_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        params_frame.grid(row=0, column=0, padx=15, pady=5, sticky="n")

        Label(params_frame, text="Parámetros del Modelo", fg="white", bg="#2b2b2b",
              font=("Segoe UI", 14)).pack(pady=8)

        Label(params_frame, text="Neuronas capa grande:", fg="white", bg="#2b2b2b").pack()
        self.entry_max_neurons = Entry(params_frame, bg="#333", fg="white")
        self.entry_max_neurons.insert(0, "60")
        self.entry_max_neurons.pack(pady=4)

        Label(params_frame, text="Grado Chebyshev:", fg="white", bg="#2b2b2b").pack()
        self.slider_grado = Scale(params_frame, from_=1, to=10, orient=HORIZONTAL,
                                  bg="#444", fg="white", troughcolor="black", length=180)
        self.slider_grado.set(3)
        self.slider_grado.pack(pady=4)

        Label(params_frame, text="Función de activación:", fg="white", bg="#2b2b2b").pack()
        self.activation_var = StringVar(self.root)
        self.activation_var.set("tanh")
        activations = ["tanh", "relu", "silu", "elu", "identity"]
        OptionMenu(params_frame, self.activation_var, *activations).pack(pady=4)

        Label(params_frame, text="Learning Rate:", fg="white", bg="#2b2b2b").pack()
        self.entry_lr = Entry(params_frame, bg="#333", fg="white")
        self.entry_lr.insert(0, "0.0003")
        self.entry_lr.pack(pady=4)

        Label(params_frame, text="Epochs:", fg="white", bg="#2b2b2b").pack()
        self.entry_epochs = Entry(params_frame, bg="#333", fg="white")
        self.entry_epochs.insert(0, "2000")
        self.entry_epochs.pack(pady=4)

        Label(params_frame, text="Train Split (0-100%):", fg="white", bg="#2b2b2b").pack()
        self.entry_split = Entry(params_frame, bg="#333", fg="white")
        self.entry_split.insert(0, "80")
        self.entry_split.pack(pady=4)

        # Botones centrales
        btn_frame = Frame(main_frame, bg="#1e1e1e")
        btn_frame.grid(row=0, column=1, padx=15)

        ttk.Button(btn_frame, text="Cargar CSV", style="Modern.TButton",
                   command=self.cargar_csv).pack(pady=6, fill="x")
        ttk.Button(btn_frame, text="Entrenar Modelo", style="Modern.TButton",
                   command=self.entrenar_modelo).pack(pady=6, fill="x")
        ttk.Button(btn_frame, text="Mostrar Frontera", style="Modern.TButton",
                   command=self.mostrar_frontera).pack(pady=6, fill="x")
        ttk.Button(btn_frame, text="Guardar Figura PNG", style="Modern.TButton",
                   command=self.guardar_png).pack(pady=6, fill="x")

        # Panel gestión de modelo (derecha)
        model_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        model_frame.grid(row=0, column=2, padx=15, pady=5, sticky="n")

        Label(model_frame, text="Gestión del Modelo", fg="white", bg="#2b2b2b",
              font=("Segoe UI", 14)).pack(pady=8)

        ttk.Button(model_frame, text="Guardar Modelo", style="Modern.TButton",
                   command=self.guardar_modelo).pack(pady=6, fill="x")
        ttk.Button(model_frame, text="Cargar Modelo", style="Modern.TButton",
                   command=self.cargar_modelo).pack(pady=6, fill="x")

        self.inferencia_label = Label(
            model_frame,
            text="Modo inferencia: OFF",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 11)
        )
        self.inferencia_label.pack(pady=10)

        # Log
        Label(self.root, text="Registro:", fg="white", bg="#1e1e1e",
              font=("Segoe UI", 13)).pack()

        self.log = Text(self.root, height=12, width=100, bg="black", fg="lime",
                        font=("Consolas", 10))
        self.log.pack(pady=10)

        # Zona de gráfica
        self.canvas_frame = Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(pady=10)

        # Variables internas
        self.X_all_orig = None
        self.y_all = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_min = None
        self.X_max = None
        self.model = None
        self.loaded_model = False
        self.last_figure = None
        self.canvas = None

    # ======================================================
    # UTILIDADES
    # ======================================================

    def log_msg(self, msg):
        self.log.insert(END, msg + "\n")
        self.log.see(END)

    def normalize(self, X):
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1
        return 2*(X - self.X_min)/denom - 1

    # ======================================================
    # CARGA DE CSV
    # ======================================================

    def cargar_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file:
            return

        try:
            df = pd.read_csv(file)
        except Exception as e:
            self.log_msg(f"❌ Error cargando CSV: {e}")
            return

        if df.shape[1] != 3:
            self.log_msg("❌ El CSV debe tener exactamente 2 features + 1 etiqueta.")
            return

        X_orig = df.iloc[:, :-1].to_numpy(float)
        y_raw = df.iloc[:, -1].to_numpy()

        y_unique = np.unique(y_raw)
        if len(y_unique) != 2:
            self.log_msg("❌ El dataset debe tener exactamente 2 clases.")
            return

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Compatibilidad con modelo cargado
        if self.loaded_model and self.model is not None:
            expected_dim = self.model.input_dim
            if X_orig.shape[1] != expected_dim:
                self.log_msg(f"❌ INCOMPATIBLE: el modelo espera {expected_dim} features, "
                             f"pero el CSV tiene {X_orig.shape[1]}.")
                return

        # Normalización
        self.X_min = X_orig.min(axis=0)
        self.X_max = X_orig.max(axis=0)
        X_norm = self.normalize(X_orig)

        # Train/Test
        try:
            p = float(self.entry_split.get())/100.0
        except ValueError:
            p = 0.8
        N = len(X_norm)
        idx = np.arange(N)
        np.random.shuffle(idx)
        cut = int(N*p)

        self.X_train = X_norm[idx[:cut]]
        self.y_train = y[idx[:cut]]
        self.X_test = X_norm[idx[cut:]]
        self.y_test = y[idx[cut:]]
        self.X_all_orig = X_orig
        self.y_all = y

        self.log_msg(f"📥 CSV cargado: {file}")
        self.log_msg(f"Muestras: {N} | Train: {cut} | Test: {N-cut}")

    # ======================================================
    # ENTRENAMIENTO
    # ======================================================

    def entrenar_modelo(self):
        if self.loaded_model:
            self.log_msg("⚠ Modo inferencia activo: no se puede entrenar un modelo cargado.")
            return

        if self.X_train is None:
            self.log_msg("❌ Primero carga un CSV.")
            return

        try:
            max_neurons = int(self.entry_max_neurons.get())
            degree = int(self.slider_grado.get())
            act = self.activation_var.get()
            lr = float(self.entry_lr.get())
            epochs = int(self.entry_epochs.get())
        except ValueError:
            self.log_msg("❌ Parámetros numéricos inválidos.")
            return

        input_dim = self.X_train.shape[1]
        self.model = ChebNetDynamicLiteStable(input_dim, max_neurons, degree, act)
        self.inferencia_label.config(text="Modo inferencia: OFF", fg="white")
        self.log_msg("🧠 Modelo creado. Entrenando...")

        self.model.fit(self.X_train, self.y_train, lr=lr, epochs=epochs, log_func=self.log_msg)

        self.log_msg("🎯 Entrenamiento finalizado.")

    # ======================================================
    # MOSTRAR FRONTERA
    # ======================================================

    def mostrar_frontera(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo.")
            return
        if self.X_all_orig is None or self.y_all is None:
            self.log_msg("❌ No hay datos cargados.")
            return

        X = self.X_all_orig
        y = self.y_all

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_norm = self.normalize(grid)

        probs_grid = self.model.predict_proba(grid_norm)
        Z = probs_grid.reshape(xx.shape)

        # Accuracy global
        X_norm_all = self.normalize(X)
        preds_all = self.model.predict(X_norm_all)
        acc = (preds_all == y).mean()*100.0

        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

        for clase, color in zip([0, 1], ["blue", "red"]):
            mask = (y == clase)
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=40, edgecolors="black", label=f"Clase {clase}")

        ax.set_title(f"Frontera de decisión — Accuracy total: {acc:.2f}%")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True)
        ax.legend()

        self.last_figure = fig

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.log_msg(f"🖼️ Frontera mostrada (Accuracy {acc:.2f}%)")

    # ======================================================
    # GUARDAR FIGURA
    # ======================================================

    def guardar_png(self):
        if self.last_figure is None:
            self.log_msg("❌ No hay figura para guardar.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".png")
        if file:
            self.last_figure.savefig(file, dpi=300)
            self.log_msg(f"💾 Figura guardada en {file}")

    # ======================================================
    # GUARDAR / CARGAR MODELO
    # ======================================================

    def guardar_modelo(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo para guardar.")
            return

        file = filedialog.asksaveasfilename(defaultextension=".chebmodel")
        if not file:
            return

        try:
            with open(file, "wb") as f:
                pickle.dump(self.model, f)
            self.log_msg(f"💾 Modelo guardado en {file}")
        except Exception as e:
            self.log_msg(f"❌ Error guardando modelo: {e}")

    def cargar_modelo(self):
        file = filedialog.askopenfilename(filetypes=[("Modelo Chebyshev", "*.chebmodel"), ("Todos", "*.*")])
        if not file:
            return

        try:
            with open(file, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            self.log_msg(f"❌ Error cargando modelo: {e}")
            return

        self.loaded_model = True
        self.inferencia_label.config(text="Modo inferencia: ON", fg="lime")

        # Actualizar parámetros en la GUI desde el modelo cargado
        self.entry_max_neurons.delete(0, END)
        self.entry_max_neurons.insert(0, str(self.model.max_neurons))
        self.slider_grado.set(self.model.degree)
        self.activation_var.set(self.model.activation_name)

        self.log_msg(f"✔ Modelo cargado desde {file}. Modo inferencia activo.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
