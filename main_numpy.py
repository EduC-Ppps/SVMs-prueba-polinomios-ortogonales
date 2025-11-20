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
#   CHEBYSHEV ESTABLE
# =========================================================

def cheb_stable_forward(z, degree):
    z_safe = np.tanh(z)

    coefs_n = np.zeros(degree + 1)
    coefs_n[-1] = 1.0
    coefs_np1 = np.zeros(degree + 2)
    coefs_np1[-1] = 1.0

    Tn = chebval(z_safe, coefs_n)
    Tnp1 = chebval(z_safe, coefs_np1)

    # derivadas
    Tn_d = chebval(z_safe, chebder(coefs_n))
    Tnp1_d = chebval(z_safe, chebder(coefs_np1))

    y = Tn**2 + Tnp1**2
    y_out = np.tanh(y)

    dy_dsafe = 2*Tn*Tn_d + 2*Tnp1*Tnp1_d
    dy_dsafe *= (1 - y_out**2)

    dsafe_dz = 1 - z_safe**2

    return y_out, dy_dsafe * dsafe_dz



# =========================================================
#   ACTIVACIONES
# =========================================================

def activation_forward(z, name, degree=None):
    name = name.lower()

    # Proteger contra overflows numéricos
    z_safe = np.clip(z, -50, 50)

    if name == "cheb":
        return cheb_stable_forward(z, degree)

    if name == "tanh":
        a = np.tanh(z_safe)
        da_dz = 1 - a**2
        return a, da_dz

    if name == "relu":
        a = np.maximum(0, z_safe)
        da_dz = (z_safe > 0).astype(float)
        return a, da_dz

    if name in ["identity", "none"]:
        return z_safe, np.ones_like(z_safe)

    if name == "elu":
        alpha = 1.0
        a = np.where(z_safe > 0, z_safe, alpha * (np.exp(z_safe) - 1))
        da_dz = np.where(z_safe > 0, 1.0, a + alpha)
        return a, da_dz

    if name == "silu":
        sig = 1 / (1 + np.exp(-z_safe))
        a = z_safe * sig
        da_dz = sig + z_safe * sig * (1 - sig)
        return a, da_dz

    if name == "leaky_relu":
        alpha = 0.01
        a = np.where(z_safe > 0, z_safe, alpha * z_safe)
        da_dz = np.where(z_safe > 0, 1.0, alpha)
        return a, da_dz

    if name == "softsign":
        denom = 1 + np.abs(z_safe)
        a = z_safe / denom
        da_dz = 1 / (denom**2)
        return a, da_dz

    a = np.tanh(z_safe)
    return a, (1 - a**2)



# =========================================================
#   RED NEURONAL DINÁMICA (NUMPY, ESTABLE)
# =========================================================

class ChebNetDynamicLiteStable:
    def __init__(self, input_dim, max_neurons, degree, activation_name):
        self.input_dim = input_dim
        self.max_neurons = max_neurons
        self.degree = degree
        self.activation_name = activation_name.lower()

        # guardar min/max del ENTRENAMIENTO ORIGINAL
        self.train_min_ = None
        self.train_max_ = None

        self.layers = []

        # Capa inicial Chebyshev
        W0 = np.random.randn(input_dim, max_neurons) * np.sqrt(2.0 / input_dim)
        b0 = np.zeros(max_neurons)
        self.layers.append({
            "W": W0,
            "b": b0,
            "activation": "cheb",
            "degree": degree
        })

        current = max_neurons
        target_min = 15

        # Capas intermedias
        while current // 2 >= target_min:
            nxt = current // 2
            W = np.random.randn(current, nxt) * np.sqrt(2.0 / current)
            b = np.zeros(nxt)
            self.layers.append({
                "W": W,
                "b": b,
                "activation": activation_name,
                "degree": None
            })
            current = nxt

        # Capa final
        W_last = np.random.randn(current, 1) * np.sqrt(2.0 / current)
        b_last = np.zeros(1)
        self.layers.append({
            "W": W_last,
            "b": b_last,
            "activation": "none",
            "degree": None
        })


    # Normalización coherente con entrenamiento
    def normalize_with_model(self, X):
        denom = self.train_max_ - self.train_min_
        denom[denom == 0] = 1
        return 2*(X - self.train_min_) / denom - 1


    # Forward
    def forward(self, X):
        caches = []
        a = X

        for layer in self.layers:
            W, b = layer["W"], layer["b"]
            z = a @ W + b

            a_next, da_dz = activation_forward(
                z,
                layer["activation"],
                layer["degree"]
            )

            caches.append({
                "a_prev": a,
                "z": z,
                "a": a_next,
                "da_dz": da_dz
            })

            a = a_next

        return a, caches


    def predict_proba(self, X):
        logits, _ = self.forward(X)
        z = np.clip(logits.reshape(-1), -30, 30)
        return 1/(1 + np.exp(-z))


    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


    def fit(self, X, y, lr=0.0003, epochs=2000, log_func=None):

        # Guardar normalización del ENTRENAMIENTO ORIGINAL
        self.train_min_ = X.min(axis=0)
        self.train_max_ = X.max(axis=0)

        N = len(X)
        y = y.reshape(-1)

        for ep in range(epochs):
            logits, caches = self.forward(X)
            logits_flat = np.clip(logits.reshape(-1), -30, 30)
            probs = 1/(1 + np.exp(-logits_flat))

            loss = -np.mean(y*np.log(probs+1e-8) + (1-y)*np.log(1-probs+1e-8))

            delta = (probs - y).reshape(-1, 1)

            grads_W = []
            grads_b = []

            for l in reversed(range(len(self.layers))):
                cp = caches[l]
                da_dz = cp["da_dz"]
                a_prev = cp["a_prev"]

                delta_z = delta * da_dz
                delta_z = np.clip(delta_z, -3, 3)

                dW = (a_prev.T @ delta_z) / N
                db = delta_z.mean(axis=0)

                dW = np.clip(dW, -1, 1)
                db = np.clip(db, -1, 1)

                grads_W.insert(0, dW)
                grads_b.insert(0, db)

                delta = delta_z @ self.layers[l]["W"].T

            # Actualizar
            for l in range(len(self.layers)):
                self.layers[l]["W"] -= lr * grads_W[l]
                self.layers[l]["b"] -= lr * grads_b[l]

            if log_func and ep % max(1, epochs//20) == 0:
                log_func(f"[{ep}/{epochs}] Loss = {loss:.5f}")

# =========================================================
#   APLICACIÓN TKINTER (GUI)
#   Bloques: Datos | Modelo | Inferencia / Modelo cargado
# =========================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal Dinámica Chebyshev (Lite, Estable)")
        self.root.configure(bg="#1e1e1e")

        # ================= Estilo =================
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

        # ================= Título =================
        Label(
            self.root,
            text="Red Neuronal Dinámica basada en Chebyshev (Lite, NumPy Estable)",
            fg="white",
            bg="#1e1e1e",
            font=("Segoe UI", 16)
        ).pack(pady=10)

        # Marco principal
        main_frame = Frame(self.root, bg="#1e1e1e")
        main_frame.pack(pady=10)

        # =====================================================
        #   BLOQUE 1: TRATAMIENTO DE DATOS (IZQUIERDA)
        # =====================================================
        data_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        data_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n")

        Label(
            data_frame,
            text="Datos (CSV / Train-Test)",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=6)

        # Botón cargar CSV
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

        # Train/Test split
        Label(
            data_frame,
            text="Porcentaje Train (0-100):",
            fg="white",
            bg="#2b2b2b"
        ).pack(pady=(10, 0))

        self.entry_split = Entry(data_frame, bg="#333", fg="white")
        self.entry_split.insert(0, "80")  # 80% train, 20% test
        self.entry_split.pack(pady=4)

        Label(
            data_frame,
            text="* Solo se usa al ENTRENAR un modelo nuevo.\n"
                 "En modo inferencia, se usan todos los datos\n"
                 "como conjunto a evaluar.",
            fg="lightgray",
            bg="#2b2b2b",
            font=("Segoe UI", 8),
            justify="left"
        ).pack(pady=4)

        # =====================================================
        #   BLOQUE 2: CONFIGURACIÓN DE MODELO (CENTRO)
        # =====================================================
        model_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        model_frame.grid(row=0, column=1, padx=10, pady=5, sticky="n")

        Label(
            model_frame,
            text="Modelo (Entrenamiento)",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=6)

        # Neuronas capa grande
        Label(model_frame, text="Neuronas capa grande:", fg="white", bg="#2b2b2b").pack()
        self.entry_max_neurons = Entry(model_frame, bg="#333", fg="white")
        self.entry_max_neurons.insert(0, "60")
        self.entry_max_neurons.pack(pady=4)

        # Grado Chebyshev
        Label(model_frame, text="Grado Chebyshev:", fg="white", bg="#2b2b2b").pack()
        self.slider_grado = Scale(
            model_frame, from_=1, to=10, orient=HORIZONTAL,
            bg="#444", fg="white", troughcolor="black", length=180
        )
        self.slider_grado.set(3)
        self.slider_grado.pack(pady=4)

        # Activación intermedia
        Label(model_frame, text="Función de activación:", fg="white", bg="#2b2b2b").pack()
        self.activation_var = StringVar(self.root)
        self.activation_var.set("tanh")
        activations = [
        "tanh",
        "relu",
        "identity",
        "elu",
        "silu",
        "leaky_relu",
        "softsign"
        ]
        OptionMenu(model_frame, self.activation_var, *activations).pack(pady=4)

        # Learning rate
        Label(model_frame, text="Learning Rate:", fg="white", bg="#2b2b2b").pack()
        self.entry_lr = Entry(model_frame, bg="#333", fg="white")
        self.entry_lr.insert(0, "0.0003")
        self.entry_lr.pack(pady=4)

        # Epochs
        Label(model_frame, text="Epochs:", fg="white", bg="#2b2b2b").pack()
        self.entry_epochs = Entry(model_frame, bg="#333", fg="white")
        self.entry_epochs.insert(0, "2000")
        self.entry_epochs.pack(pady=4)

        # Botón entrenar
        ttk.Button(
            model_frame,
            text="Entrenar Modelo",
            style="Modern.TButton",
            command=self.entrenar_modelo
        ).pack(pady=8, fill="x")

        # Botón guardar modelo
        ttk.Button(
            model_frame,
            text="Guardar Modelo",
            style="Modern.TButton",
            command=self.guardar_modelo
        ).pack(pady=4, fill="x")

        # =====================================================
        #   BLOQUE 3: INFERENCIA / MODELO CARGADO (DERECHA)
        # =====================================================
        infer_frame = Frame(main_frame, bg="#2b2b2b", bd=2, relief="groove")
        infer_frame.grid(row=0, column=2, padx=10, pady=5, sticky="n")

        Label(
            infer_frame,
            text="Inferencia / Modelo Cargado",
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

        # Indicador modo
        self.inferencia_label = Label(
            infer_frame,
            text="Modo: ENTRENAMIENTO",
            fg="orange",
            bg="#2b2b2b",
            font=("Segoe UI", 11, "bold")
        )
        self.inferencia_label.pack(pady=10)

        # Botón mostrar frontera
        ttk.Button(
            infer_frame,
            text="Mostrar Frontera",
            style="Modern.TButton",
            command=self.mostrar_frontera
        ).pack(pady=6, fill="x")

        # Botón guardar PNG
        ttk.Button(
            infer_frame,
            text="Guardar Figura PNG",
            style="Modern.TButton",
            command=self.guardar_png
        ).pack(pady=6, fill="x")

        # =====================================================
        #   LOG
        # =====================================================
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

        # =====================================================
        #   ZONA DE GRÁFICA
        # =====================================================
        self.canvas_frame = Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(pady=8)

        # ================= Variables internas =================
        self.X_all_orig = None
        self.y_all = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_min = None
        self.X_max = None

        self.model = None           # modelo actual (entrenado o cargado)
        self.loaded_model = False   # True si viene de archivo

        self.last_figure = None
        self.canvas = None

    # ---------------------------------------------------------
    # Utilidad de log (la implementamos ya aquí)
    # ---------------------------------------------------------
    def log_msg(self, msg):
        self.log.insert(END, msg + "\n")
        self.log.see(END)

    # =========================================================
    #   CARGAR CSV (para entrenamiento o inferencia)
    # =========================================================
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
            self.log_msg("❌ El CSV debe tener EXACTAMENTE 3 columnas: x1, x2, etiqueta")
            return

        # X e y originales
        X_orig = df.iloc[:, :2].to_numpy(dtype=float)
        y_raw = df.iloc[:, 2].to_numpy()

        # Codificar etiquetas
        le = LabelEncoder()
        y = le.fit_transform(y_raw)  # 0 y 1

        self.X_all_orig = X_orig
        self.y_all = y

        # Si existe un modelo cargado (inferencia) → no hay train/test
        if self.loaded_model:
            self.log_msg("📌 CSV cargado en modo INFERENCIA (sin división train/test).")

            # Normalizar usando min/max del entrenamiento original
            self.X_train = self.model.normalize_with_model(X_orig)
            self.X_test = self.X_train
            self.y_train = y
            self.y_test = y

            return

        # ========= MODO ENTRENAMIENTO =========
        split = float(self.entry_split.get()) / 100.0
        if split <= 0 or split >= 1:
            self.log_msg("⚠ Split inválido, usando 80% por defecto.")
            split = 0.8

        # Normalizar con min/max del CSV actual
        self.X_min = X_orig.min(axis=0)
        self.X_max = X_orig.max(axis=0)
        denom = self.X_max - self.X_min
        denom[denom == 0] = 1
        X_norm = 2*(X_orig - self.X_min)/denom - 1

        # División train/test
        N = len(X_norm)
        idx = np.arange(N)
        np.random.shuffle(idx)
        cut = int(N * split)

        self.X_train = X_norm[idx[:cut]]
        self.y_train = y[idx[:cut]]

        self.X_test = X_norm[idx[cut:]]
        self.y_test = y[idx[cut:]]

        self.log_msg(f"✔ CSV cargado ({N} filas). Train = {cut}, Test = {N-cut}")


    # =========================================================
    #   ENTRENAR MODELO DESDE CERO
    # =========================================================
    def entrenar_modelo(self):
        if self.X_train is None:
            self.log_msg("❌ Carga primero un CSV.")
            return

        # Salimos de modo inferencia
        self.loaded_model = False
        self.inferencia_label.config(text="Modo: ENTRENAMIENTO", fg="orange")

        # Parámetros del modelo
        max_neurons = int(self.entry_max_neurons.get())
        degree = int(self.slider_grado.get())
        activation = self.activation_var.get()
        lr = float(self.entry_lr.get())
        epochs = int(self.entry_epochs.get())

        # Crear modelo nuevo
        self.model = ChebNetDynamicLiteStable(
            input_dim=2,
            max_neurons=max_neurons,
            degree=degree,
            activation_name=activation
        )

        self.log_msg("🧠 Entrenando modelo...")
        self.model.fit(
            self.X_train,
            self.y_train,
            lr=lr,
            epochs=epochs,
            log_func=self.log_msg
        )

        acc = (self.model.predict(self.X_test) == self.y_test).mean()*100
        self.log_msg(f"✔ ENTRENAMIENTO COMPLETADO — Accuracy test = {acc:.2f}%")


    # =========================================================
    #   GUARDAR MODELO
    # =========================================================
    def guardar_modelo(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo para guardar.")
            return

        file = filedialog.asksaveasfilename(
            defaultextension=".chebmodel",
            filetypes=[("Modelo Chebyshev", "*.chebmodel")]
        )
        if not file:
            return

        data = {
            "layers": self.model.layers,
            "train_min_": self.model.train_min_,
            "train_max_": self.model.train_max_
        }

        with open(file, "wb") as f:
            pickle.dump(data, f)

        self.log_msg(f"💾 Modelo guardado en {file}")


    # =========================================================
    #   CARGAR MODELO (MODO INFERENCIA)
    # =========================================================
    def cargar_modelo(self):
        file = filedialog.askopenfilename(
            filetypes=[("Modelo Chebyshev", "*.chebmodel")]
        )
        if not file:
            return

        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            self.log_msg(f"❌ Error cargando modelo: {e}")
            return

        # Crear un nuevo objeto modelo
        self.model = ChebNetDynamicLiteStable(
            input_dim=2,
            max_neurons=10,   # estos valores NO importan
            degree=3,         # porque se sobrescriben
            activation_name="tanh"
        )

        # Sobrescribir estructura real
        self.model.layers = data["layers"]
        self.model.train_min_ = data["train_min_"]
        self.model.train_max_ = data["train_max_"]

        self.loaded_model = True
        self.inferencia_label.config(text="Modo: INFERENCIA", fg="cyan")

        self.log_msg(f"📌 Modelo cargado correctamente: {file}")
        self.log_msg("Ahora puedes cargar un CSV nuevo y mostrar la frontera.")


    # =========================================================
    #   RESETEAR MODELO (salir de modo inferencia)
    # =========================================================
    def resetear_modelo(self):
        self.model = None
        self.loaded_model = False
        self.inferencia_label.config(text="Modo: ENTRENAMIENTO", fg="orange")

        self.log_msg("🔄 Modelo reseteado. Ahora puedes entrenar uno nuevo o cargar otro CSV.")

    # =========================================================
    #   RESETEAR CSV / ELIMINAR DATOS
    # =========================================================
    def resetear_csv(self):
        self.X_all_orig = None
        self.y_all = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_min = None
        self.X_max = None

        # Eliminar figura del canvas
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        self.last_figure = None

        self.log_msg("🗑️ CSV eliminado. El programa ha quedado sin datos.")
        
        
    # =========================================================
    #   MOSTRAR FRONTERA DE DECISIÓN
    # =========================================================
    def mostrar_frontera(self):
        if self.model is None:
            self.log_msg("❌ No hay modelo cargado ni entrenado.")
            return

        if self.X_all_orig is None:
            self.log_msg("❌ Carga antes un CSV.")
            return

        X = self.X_all_orig
        y = self.y_all

        # Crear grid en rango del NUEVO CSV
        x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
        y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Normalizar datos del grid dependiendo del modo
        if self.loaded_model:
            grid_norm = self.model.normalize_with_model(grid)
            X_norm = self.model.normalize_with_model(X)
        else:
            denom = self.X_max - self.X_min
            denom[denom == 0] = 1
            grid_norm = 2*(grid - self.X_min)/denom - 1
            X_norm = 2*(X - self.X_min)/denom - 1

        # Inferencia
        Z = self.model.predict_proba(grid_norm).reshape(xx.shape)
        preds = self.model.predict(X_norm)
        acc = (preds == y).mean() * 100

        # Dibujar
        fig = plt.Figure(figsize=(
6, 5))
        ax = fig.add_subplot(111)

        # Contorno (frontera)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

        # Puntos
        colormap = {
            0: "blue",
            1: "red"
            }

        for cls in np.unique(y):
            color = colormap.get(cls, "gray")  # si hubiera más clases, gris
            ax.scatter(
                X[y == cls, 0],
                X[y == cls, 1],
                c=color,
                s=30,
                edgecolors="black",
                linewidths=1.2,
                label=f"Clase {cls}"
            )   
        
        ax.set_title(f"Frontera de decisión — Accuracy: {acc:.2f}%")
        ax.grid(True)
        ax.legend()

        # Dibujar en Tkinter
        self.last_figure = fig
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.log_msg("🖼️ Frontera mostrada correctamente.")


    # =========================================================
    #   GUARDAR PNG
    # =========================================================
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
        self.log_msg(f"💾 Imagen guardada en {file}")

# =========================================================
#   MAIN — EJECUTAR APLICACIÓN
# =========================================================
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
