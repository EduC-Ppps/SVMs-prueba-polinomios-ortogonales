# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:19:42 2025

@author: Eduardo Contreras
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from tkinter import (
    Tk, Label, filedialog, Text, END, Scale,
    HORIZONTAL, Entry, Frame, StringVar, OptionMenu
)
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder

############################################################
# Polinomio de Chebyshev implementado en PyTorch
############################################################

def cheb_torch_Tn(z, n):
    """
    Polinomio de Chebyshev de primer tipo T_n(z) usando recurrencia.
    z: tensor
    n: grado (int)
    """
    if n == 0:
        return torch.ones_like(z)
    if n == 1:
        return z
    T0 = torch.ones_like(z)
    T1 = z
    for _ in range(2, n + 1):
        T2 = 2 * z * T1 - T0
        T0, T1 = T1, T2
    return T1


class ChebActivation(nn.Module):
    """
    Activación tipo Christoffel–Chebyshev estable:
      z_safe = tanh(z)
      y = T_n(z_safe)^2 + T_{n+1}(z_safe)^2
      salida = tanh(y)
    """
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        # estabilizar entrada
        z_safe = torch.tanh(x)
        n = self.degree

        Tn = cheb_torch_Tn(z_safe, n)
        Tnp1 = cheb_torch_Tn(z_safe, n + 1)

        y = Tn**2 + Tnp1**2
        out = torch.tanh(y)  # estabilizar salida

        return out


###########################################################
# Funciones de activación generales (PyTorch)
###########################################################

class Softsign(nn.Module):
    def forward(self, x):
        return x / (1 + torch.abs(x))


class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name in ["identity", "none"]:
        return nn.Identity()
    if name == "leaky_relu":
        return LeakyReLU(alpha=0.01)
    if name == "Sigmoid":
        return nn.Sigmoid()
    
    # por defecto, tanh
    return nn.Tanh()


############################################################
#   Red neuronal
############################################################

class ChebNetTorch(nn.Module):
    """
    Red neuronal:
      - Capa 1: Linear -> ChebActivation
      - Varias capas reduciendo a la mitad -> activación elegida
      - Última: Linear -> 1 neurona (logit)
    Guarda min/max del entrenamiento para normalizar datasets nuevos.
    """
    def __init__(self, input_dim, max_neurons, degree, activation_name):
        super().__init__()
        self.input_dim = input_dim
        self.max_neurons = max_neurons
        self.degree = degree
        self.activation_name = activation_name.lower()

        self.layers = nn.ModuleList()

# Primera capa: Linear + Chebyshev
        self.layers.append(nn.Linear(input_dim, max_neurons))
        self.layers.append(ChebActivation(degree=degree))

        current = max_neurons
        target_min = 15

# Capas intermedias reduciendo a la mitad
        while current // 2 >= target_min:
            nxt = current // 2
            self.layers.append(nn.Linear(current, nxt))
            self.layers.append(get_activation(self.activation_name))
            current = nxt

# Capa final: Linear -> 1
        self.layers.append(nn.Linear(current, 1))

# Escalado (se rellenan al entrenar)
        self.register_buffer("train_min_", torch.zeros(input_dim))
        self.register_buffer("train_max_", torch.ones(input_dim))
        self.scaler_initialized = False

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out  # logits

    def set_scaler(self, x_min: np.ndarray, x_max: np.ndarray):
        """
        Guardar min y max usados en el entrenamiento para normalizar luego.
        """
        x_min_t = torch.tensor(x_min, dtype=torch.float32)
        x_max_t = torch.tensor(x_max, dtype=torch.float32)
        self.train_min_.data = x_min_t
        self.train_max_.data = x_max_t
        self.scaler_initialized = True

    def normalize_with_model(self, X_np: np.ndarray) -> torch.Tensor:
        """
        Normalizar un nuevo dataset usando min/max del entrenamiento.
        """
        assert self.scaler_initialized, "Scaler del modelo no inicializado."
        x_min = self.train_min_.cpu().numpy()
        x_max = self.train_max_.cpu().numpy()
        denom = x_max - x_min
        denom[denom == 0] = 1
        X_norm = 2 * (X_np - x_min) / denom - 1
        return torch.from_numpy(X_norm.astype(np.float32))


############################################################
# Entrenamiento de la red y predicción
############################################################

def train_model(model: ChebNetTorch,
                X_train: np.ndarray,
                y_train: np.ndarray,
                lr: float,
                epochs: int,
                log_func=None):
    """
    X_train, y_train en NumPy. Entrenamos la red en PyTorch.
    """
    device = torch.device("cpu")
    model.to(device)
    model.train()

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t).squeeze()
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if log_func and ep % max(1, epochs // 20) == 0:
            log_func(f"[{ep}/{epochs}] Loss = {loss.item():.5f}")


def predict_proba(model: ChebNetTorch, X: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(X_t).squeeze()
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def predict_labels(model: ChebNetTorch, X: np.ndarray) -> np.ndarray:
    probs = predict_proba(model, X)
    return (probs > 0.5).astype(int)


############################################################
# GUI de la aplicación (no importante para el funcionamiento de la red)
############################################################

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal Chebyshev (PyTorch)")
        self.root.configure(bg="#1e1e1e")

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

        Label(
            self.root,
            text="Red Neuronal Dinámica Chebyshev (PyTorch)",
            fg="white",
            bg="#1e1e1e",
            font=("Segoe UI", 16)
        ).pack(pady=10)

        main_frame = Frame(self.root, bg="#1e1e1e")
        main_frame.pack(pady=10)

####################### Datos #######################
        data_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        data_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n")

        Label(
            data_frame,
            text="Datos (CSV / Train-Test)",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=6)

        ttk.Button(
            data_frame,
            text="Cargar CSV",
            style="Modern.TButton",
            command=self.cargar_csv
        ).pack(pady=6, fill="x")

        ttk.Button(
            data_frame,
            text="Eliminar CSV",
            style="Modern.TButton",
            command=self.resetear_csv
        ).pack(pady=6, fill="x")

        Label(
            data_frame,
            text="Porcentaje Train (0-100):",
            fg="white",
            bg="#2b2b2b"
        ).pack(pady=(10, 0))
        self.entry_split = Entry(data_frame, bg="#333", fg="white")
        self.entry_split.insert(0, "80")
        self.entry_split.pack(pady=4)

####################### Modelo de la red #######################
        model_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        model_frame.grid(row=0, column=1, padx=10, pady=5, sticky="n")

        Label(
            model_frame,
            text="Modelo (Entrenamiento)",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=6)

        Label(model_frame, text="Neuronas capa grande:", fg="white", bg="#2b2b2b").pack()
        self.entry_max_neurons = Entry(model_frame, bg="#333", fg="white")
        self.entry_max_neurons.insert(0, "60")
        self.entry_max_neurons.pack(pady=4)

        Label(model_frame, text="Grado Chebyshev:", fg="white", bg="#2b2b2b").pack()
        self.slider_grado = Scale(
            model_frame, from_=1, to=10, orient=HORIZONTAL,
            bg="#444", fg="white", troughcolor="black", length=180
        )
        self.slider_grado.set(3)
        self.slider_grado.pack(pady=4)

        Label(model_frame, text="Función de activación:", fg="white", bg="#2b2b2b").pack()
        self.activation_var = StringVar(self.root)
        self.activation_var.set("tanh")
        activations = [
            "tanh", "relu", "identity",
             "leaky_relu", "sigmoid"
        ]
        OptionMenu(model_frame, self.activation_var, *activations).pack(pady=4)

        Label(model_frame, text="Learning Rate:", fg="white", bg="#2b2b2b").pack()
        self.entry_lr = Entry(model_frame, bg="#333", fg="white")
        self.entry_lr.insert(0, "0.001")
        self.entry_lr.pack(pady=4)

        Label(model_frame, text="Epochs:", fg="white", bg="#2b2b2b").pack()
        self.entry_epochs = Entry(model_frame, bg="#333", fg="white")
        self.entry_epochs.insert(0, "2000")
        self.entry_epochs.pack(pady=4)

        ttk.Button(
            model_frame,
            text="Entrenar Modelo",
            style="Modern.TButton",
            command=self.entrenar_modelo
        ).pack(pady=8, fill="x")

        ttk.Button(
            model_frame,
            text="Guardar Modelo",
            style="Modern.TButton",
            command=self.guardar_modelo
        ).pack(pady=4, fill="x")

####################### Inferencia #######################
        infer_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        infer_frame.grid(row=0, column=2, padx=10, pady=5, sticky="n")

        Label(
            infer_frame,
            text="Inferencia / Modelo cargado",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=6)

        ttk.Button(
            infer_frame,
            text="Cargar Modelo",
            style="Modern.TButton",
            command=self.cargar_modelo
        ).pack(pady=6, fill="x")

        ttk.Button(
            infer_frame,
            text="Resetear Modelo",
            style="Modern.TButton",
            command=self.resetear_modelo
        ).pack(pady=6, fill="x")

        self.inferencia_label = Label(
            infer_frame,
            text="Modo: ENTRENAMIENTO",
            fg="orange",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        self.inferencia_label.pack(pady=10)

        ttk.Button(
            infer_frame,
            text="Mostrar Frontera",
            style="Modern.TButton",
            command=self.mostrar_frontera
        ).pack(pady=6, fill="x")

        ttk.Button(
            infer_frame,
            text="Guardar Figura PNG",
            style="Modern.TButton",
            command=self.guardar_png
        ).pack(pady=6, fill="x")

###################### Log #######################
        Label(
            self.root,
            text="Registro:",
            fg="white",
            bg="#1e1e1e",
            font=("Segoe UI", 13)
        ).pack()

        self.log = Text(
            self.root,
            height=10,
            width=110,
            bg="black",
            fg="lime",
            font=("Consolas", 10)
        )
        self.log.pack(pady=8)

####################### Gráfica de resultados #######################
        self.canvas_frame = Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(pady=8)

# Estado interno
        self.X_all_orig = None
        self.y_all = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_min = None
        self.X_max = None

        self.model: ChebNetTorch | None = None
        self.loaded_model = False
        self.scaler_from_model = False  # indica si el scaler viene del modelo guardado

        self.last_figure = None
        self.canvas = None

###########################################################
# Log
###########################################################
    def log_msg(self, msg):
        self.log.insert(END, msg + "\n")
        self.log.see(END)

###########################################################
# Eliminar CSV cargado
###########################################################
    def resetear_csv(self):
        self.X_all_orig = None
        self.y_all = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_min = None
        self.X_max = None

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.last_figure = None

        self.log_msg("🗑️ CSV eliminado. No hay datos cargados.")

###########################################################
# Cargar CSV
###########################################################
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
            self.log_msg("❌ El CSV debe tener 3 columnas: x1, x2, etiqueta.")
            return

        X_orig = df.iloc[:, :2].to_numpy(dtype=float)
        y_raw = df.iloc[:, 2].to_numpy()

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        self.X_all_orig = X_orig
        self.y_all = y

# Si hay modelo cargado (modo inferencia), normalizamos con el scaler del modelo
        if self.loaded_model and self.model is not None and self.model.scaler_initialized:
            self.log_msg("📌 CSV cargado en modo INFERENCIA (sin train/test).")
            X_norm_t = self.model.normalize_with_model(X_orig)
            X_norm = X_norm_t.numpy()
            self.X_train = X_norm
            self.y_train = y
            self.X_test = X_norm
            self.y_test = y
            return

# Modo entrenamiento: calcular min/max a partir del CSV
        try:
            split = float(self.entry_split.get()) / 100.0
        except ValueError:
            split = 0.8

        if split <= 0 or split >= 1:
            self.log_msg("⚠ Split inválido, usando 80% por defecto.")
            split = 0.8

        self.X_min = X_orig.min(axis=0)
        self.X_max = X_orig.max(axis=0)
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1
        X_norm = 2 * (X_orig - self.X_min) / denom - 1

        N = len(X_norm)
        idx = np.arange(N)
        np.random.shuffle(idx)
        cut = int(N * split)

        self.X_train = X_norm[idx[:cut]]
        self.y_train = y[idx[:cut]]
        self.X_test = X_norm[idx[cut:]]
        self.y_test = y[idx[cut:]]

        self.log_msg(f"✔ CSV cargado ({N} filas). Train={cut}, Test={N-cut}")

###########################################################
# Entrenamiento
###########################################################
    def entrenar_modelo(self):
        if self.X_train is None or self.y_train is None:
            self.log_msg("❌ Carga un CSV primero.")
            return

        try:
            max_neurons = int(self.entry_max_neurons.get())
            degree = int(self.slider_grado.get())
            activation = self.activation_var.get()
            lr = float(self.entry_lr.get())
            epochs = int(self.entry_epochs.get())
        except ValueError:
            self.log_msg("❌ Parámetros numéricos inválidos.")
            return

        self.model = ChebNetTorch(
            input_dim=2,
            max_neurons=max_neurons,
            degree=degree,
            activation_name=activation
        )

# Inicializar scaler en el modelo
        if self.X_min is not None and self.X_max is not None:
            self.model.set_scaler(self.X_min, self.X_max)

        self.loaded_model = False
        self.inferencia_label.config(text="Modo: ENTRENAMIENTO", fg="orange")

        self.log_msg("🧠 Entrenando modelo...")
        train_model(
            self.model,
            self.X_train,
            self.y_train,
            lr=lr,
            epochs=epochs,
            log_func=self.log_msg
        )

        if self.X_test is not None and len(self.X_test) > 0:
            preds_test = predict_labels(self.model, self.X_test)
            acc = (preds_test == self.y_test).mean() * 100
            self.log_msg(f"✔ ENTRENAMIENTO COMPLETADO — Accuracy test = {acc:.2f}%")
        else:
            self.log_msg("✔ ENTRENAMIENTO COMPLETADO (sin test).")

###########################################################
# Guardar modelo entrenado
###########################################################
    def guardar_modelo(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo para guardar.")
            return

        file = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("Modelo PyTorch", "*.pt")]
        )
        if not file:
            return

        data = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "max_neurons": self.model.max_neurons,
            "degree": self.model.degree,
            "activation_name": self.model.activation_name,
            "train_min_": self.model.train_min_.cpu(),
            "train_max_": self.model.train_max_.cpu()
        }

        torch.save(data, file)
        self.log_msg(f"💾 Modelo guardado en {file}")

###########################################################
# Cargar modelo desde un archivo
###########################################################
    def cargar_modelo(self):
        file = filedialog.askopenfilename(
            filetypes=[("Modelo PyTorch", "*.pt"), ("Todos", "*.*")]
        )
        if not file:
            return

        try:
            data = torch.load(file, map_location="cpu")
        except Exception as e:
            self.log_msg(f"❌ Error cargando modelo: {e}")
            return

        self.model = ChebNetTorch(
            input_dim=data["input_dim"],
            max_neurons=data["max_neurons"],
            degree=data["degree"],
            activation_name=data["activation_name"]
        )
        self.model.load_state_dict(data["state_dict"])
        self.model.set_scaler(data["train_min_"], data["train_max_"])

        self.loaded_model = True
        self.inferencia_label.config(text="Modo: INFERENCIA", fg="cyan")

# Actualizar interfaz con los hiperparámetros
        self.entry_max_neurons.delete(0, END)
        self.entry_max_neurons.insert(0, str(data["max_neurons"]))
        self.slider_grado.set(data["degree"])
        self.activation_var.set(data["activation_name"])

        self.log_msg(f"📌 Modelo cargado desde {file}")
        self.log_msg("Ahora puedes cargar un CSV nuevo y mostrar la frontera.")

###########################################################
# Eliminar modelo cargado
###########################################################
    def resetear_modelo(self):
        self.model = None
        self.loaded_model = False
        self.inferencia_label.config(text="Modo: ENTRENAMIENTO", fg="orange")
        self.log_msg("🔄 Modelo reseteado. Puedes entrenar uno nuevo o cargar otro modelo.")

###########################################################
# Mostrar la gráfica final
###########################################################
    def mostrar_frontera(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo entrenado ni cargado.")
            return
        if self.X_all_orig is None or self.y_all is None:
            self.log_msg("❌ Carga antes un CSV.")
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

# Normalización
        if self.loaded_model and self.model.scaler_initialized:
            grid_norm_t = self.model.normalize_with_model(grid)
            grid_norm = grid_norm_t.numpy()
            X_norm_t = self.model.normalize_with_model(X)
            X_norm = X_norm_t.numpy()
        else:
            if self.X_min is None or self.X_max is None:
                self.log_msg("❌ No hay min/max para normalizar.")
                return
            denom = self.X_max - self.X_min
            denom[denom == 0] = 1
            grid_norm = 2 * (grid - self.X_min) / denom - 1
            X_norm = 2 * (X - self.X_min) / denom - 1

        Z = predict_proba(self.model, grid_norm).reshape(xx.shape)
        preds = predict_labels(self.model, X_norm)
        acc = (preds == y).mean() * 100

        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

        colormap = {0: "blue", 1: "red"}
        for cls in np.unique(y):
            color = colormap.get(cls, "gray")
            ax.scatter(
                X[y == cls, 0],
                X[y == cls, 1],
                c=color,
                s=35,
                edgecolors="black",
                linewidths=1.0,
                label=f"Clase {cls}"
            )

        ax.set_title(f"Frontera de decisión — Accuracy: {acc:.2f}%")
        ax.grid(True)
        ax.legend()

        self.last_figure = fig
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.log_msg("🖼️ Frontera mostrada correctamente.")

###########################################################
# Guardar la gráfica
###########################################################
    def guardar_png(self):
        if self.last_figure is None:
            self.log_msg("❌ No hay figura para guardar.")
            return
        file = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png")]
        )
        if not file:
            return
        self.last_figure.savefig(file, dpi=150)
        self.log_msg(f"💾 Figura guardada en {file}")


###########################################################
# Main
###########################################################
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()


