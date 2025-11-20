# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:19:42 2025

@author: Eduardo Contreras
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tkinter import (
    Tk, Button, Label, filedialog, Text, END, Scale,
    HORIZONTAL, Entry, Frame, StringVar, OptionMenu
)
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.polynomial.chebyshev import chebval
from sklearn.preprocessing import LabelEncoder


# =========================================================
#   ACTIVACIÓN CHRISTOFFEL–CHEBYSHEV
# =========================================================

def christoffel_activation_np(x, degree=3):
    Pn = chebval(x, [0]*degree + [1])
    Pnp1 = chebval(x, [0]*(degree+1) + [1])
    return Pn**2 + Pnp1**2


class ChristoffelActivation(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        y_np = christoffel_activation_np(x_np, self.degree)
        return torch.tensor(y_np, device=x.device, dtype=x.dtype)


# =========================================================
#   FUNCIONES DE ACTIVACIÓN CONFIGURABLES
# =========================================================

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "identity":
        return nn.Identity()
    else:
        return nn.Tanh()


# =========================================================
#   RED NEURONAL DINÁMICA
# =========================================================

class ChebNetDynamic(nn.Module):
    def __init__(self, input_dim, max_neurons, degree, activation_name):
        super().__init__()

        layers = []
        act_layer = get_activation(activation_name)

        # Capa grande inicial
        layers.append(nn.Linear(input_dim, max_neurons))
        layers.append(ChristoffelActivation(degree=degree))

        # Reducción progresiva
        current = max_neurons
        target_min = 15

        while current // 2 >= target_min:
            next_size = current // 2
            layers.append(nn.Linear(current, next_size))
            layers.append(act_layer)
            current = next_size

        # Capa final → salida 1
        layers.append(nn.Linear(current, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# =========================================================
#   APLICACIÓN TKINTER MODERNA
# =========================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal Dinámica Chebyshev")
        self.root.configure(bg="#1e1e1e")

        # ===== Estilo Moderno =====
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
        Label(root, text="Red Neuronal Dinámica basada en Chebyshev",
              fg="white", bg="#1e1e1e", font=("Segoe UI", 18)).pack(pady=12)

        # Marco principal
        main_frame = Frame(self.root, bg="#1e1e1e")
        main_frame.pack(pady=10)

        # =======================================================
        #  IZQUIERDA — PARÁMETROS
        # =======================================================
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
        self.entry_lr.insert(0, "0.001")
        self.entry_lr.pack(pady=4)

        Label(params_frame, text="Epochs:", fg="white", bg="#2b2b2b").pack()
        self.entry_epochs = Entry(params_frame, bg="#333", fg="white")
        self.entry_epochs.insert(0, "2000")
        self.entry_epochs.pack(pady=4)

        Label(params_frame, text="Train Split (0-100%):", fg="white", bg="#2b2b2b").pack()
        self.entry_split = Entry(params_frame, bg="#333", fg="white")
        self.entry_split.insert(0, "80")
        self.entry_split.pack(pady=4)

        # =======================================================
        # CENTRO — BOTONES PRINCIPALES
        # =======================================================
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

        # =======================================================
        # DERECHA — GESTIÓN DE MODELO
        # =======================================================
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

        # =======================================================
        # LOG
        # =======================================================
        Label(self.root, text="Registro:", fg="white", bg="#1e1e1e",
              font=("Segoe UI", 13)).pack()

        self.log = Text(self.root, height=12, width=100, bg="black", fg="lime",
                        font=("Consolas", 10))
        self.log.pack(pady=10)

        # =======================================================
        # ZONA DE GRÁFICA
        # =======================================================
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
    # LOG / NORMALIZACIÓN
    # ======================================================

    def log_msg(self, msg):
        self.log.insert(END, msg + "\n")
        self.log.see(END)

    def normalize(self, X):
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1
        return 2 * (X - self.X_min) / denom - 1

    # ======================================================
    # CARGAR CSV + DETECCIÓN DE INCOMPATIBILIDAD
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
            self.log_msg("❌ El CSV debe tener 2 features + 1 etiqueta.")
            return

        X_orig = df.iloc[:, :-1].to_numpy(float)
        y_raw = df.iloc[:, -1].to_numpy()

        # Comprobar que es binario
        y_unique = np.unique(y_raw)
        if len(y_unique) != 2:
            self.log_msg("❌ El dataset debe tener 2 clases.")
            return

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Comprobar incompatibilidad con modelo cargado
        if self.loaded_model:
            expected_dim = self.model.net[0].in_features
            if X_orig.shape[1] != expected_dim:
                self.log_msg(f"❌ INCOMPATIBILIDAD: "
                             f"El modelo espera {expected_dim} features, "
                             f"pero el CSV tiene {X_orig.shape[1]}.")
                return

        # Normalizar
        self.X_min = X_orig.min(axis=0)
        self.X_max = X_orig.max(axis=0)
        X_norm = self.normalize(X_orig)

        # Split
        p = float(self.entry_split.get()) / 100
        N = len(X_norm)
        idx = np.arange(N)
        np.random.shuffle(idx)
        cut = int(N * p)

        self.X_train = X_norm[idx[:cut]]
        self.y_train = y[idx[:cut]]
        self.X_test = X_norm[idx[cut:]]
        self.y_test = y[idx[cut:]]
        self.X_all_orig = X_orig
        self.y_all = y

        self.log_msg("📥 CSV cargado correctamente.")

    # ======================================================
    # ENTRENAR MODELO
    # ======================================================

    def entrenar_modelo(self):
        if self.loaded_model:
            self.log_msg("⚠ No se puede entrenar un modelo cargado. Modo inferencia activo.")
            return

        if self.X_train is None:
            self.log_msg("❌ Primero carga un CSV.")
            return

        max_neurons = int(self.entry_max_neurons.get())
        degree = int(self.slider_grado.get())
        act = self.activation_var.get()
        lr = float(self.entry_lr.get())
        epochs = int(self.entry_epochs.get())
        input_dim = self.X_train.shape[1]

        self.model = ChebNetDynamic(input_dim, max_neurons, degree, act)
        self.inferencia_label.config(text="Modo inferencia: OFF", fg="white")
        self.log_msg("🧠 Modelo creado.")

        Xt = torch.tensor(self.X_train, dtype=torch.float32)
        yt = torch.tensor(self.y_train, dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for e in range(epochs):
            optimizer.zero_grad()
            pred = self.model(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            optimizer.step()

            if e % max(1, epochs // 20) == 0:
                self.log_msg(f"Epoch {e}/{epochs} | Loss={loss.item():.5f}")

        self.log_msg("🎯 Entrenamiento finalizado.")

    # ======================================================
    # MOSTRAR FRONTERA
    # ======================================================

    def mostrar_frontera(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo.")
            return

        X = self.X_all_orig
        y = self.y_all

        # Grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_norm = self.normalize(grid)

        grid_t = torch.tensor(grid_norm, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(grid_t).numpy()
        Z = (1 / (1 + np.exp(-logits))).reshape(xx.shape)

        # Accuracy
        X_norm_all = self.normalize(X)
        Xt = torch.tensor(X_norm_all, dtype=torch.float32)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(Xt)).numpy()
        pred_bin = (pred > 0.5).astype(int)
        acc = (pred_bin.flatten() == y).mean() * 100

        # Figura
        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

        for clase, color in zip([0, 1], ["blue", "red"]):
            mask = (y == clase)
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=40, edgecolors="black")

        ax.set_title(f"Frontera — Accuracy total: {acc:.2f}%")
        ax.grid(True)

        self.last_figure = fig

        if self.canvas:
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
            self.log_msg("❌ No hay figura.")
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

        file = filedialog.asksaveasfilename(defaultextension=".pt")
        if not file:
            return

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.X_train.shape[1],
            "max_neurons": int(self.entry_max_neurons.get()),
            "degree": int(self.slider_grado.get()),
            "activation": self.activation_var.get(),
        }

        torch.save(checkpoint, file)
        self.log_msg(f"💾 Modelo guardado en {file}")

    def cargar_modelo(self):
        file = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if not file:
            return

        checkpoint = torch.load(file, map_location="cpu")

        self.loaded_model = True
        self.inferencia_label.config(text="Modo inferencia: ON", fg="lime")
        self.log_msg("🔍 Modelo cargado — Modo inferencia activado.")

        # Restaurar hiperparámetros
        max_neurons = checkpoint["max_neurons"]
        degree = checkpoint["degree"]
        activation = checkpoint["activation"]
        input_dim = checkpoint["input_dim"]

        self.entry_max_neurons.delete(0, END)
        self.entry_max_neurons.insert(0, str(max_neurons))
        self.slider_grado.set(degree)
        self.activation_var.set(activation)

        # Reconstruir red
        self.model = ChebNetDynamic(input_dim, max_neurons, degree, activation)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.log_msg(f"✔ Modelo restaurado correctamente de {file}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()

