import numpy as np
import os

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    x = np.clip(x, -500, 500)  # evitar overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        return self.output

    def backward(self, X, y, lr):
        error = y - self.output
        d_output = error * sigmoid_derivative(self.z2)       # corregido
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.z1)  # corregido

        self.W2 += self.a1.T.dot(d_output) * lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
        self.W1 += X.T.dot(d_hidden) * lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr


    def train(self, X, y, epochs=10000, lr=0.1, label=""):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, lr)
            if epoch % 2000 == 0:
                loss = np.mean((y - self.output) ** 2)
                print(f"[{label}] Epoch {epoch}, Loss: {loss:.6f}")

def guardar_pesos(mlp, nombre):
    np.save(f"{nombre}_W1.npy", mlp.W1)
    np.save(f"{nombre}_b1.npy", mlp.b1)
    np.save(f"{nombre}_W2.npy", mlp.W2)
    np.save(f"{nombre}_b2.npy", mlp.b2)

def cargar_pesos(mlp, nombre):
    try:
        mlp.W1 = np.load(f"{nombre}_W1.npy")
        mlp.b1 = np.load(f"{nombre}_b1.npy")
        mlp.W2 = np.load(f"{nombre}_W2.npy")
        mlp.b2 = np.load(f"{nombre}_b2.npy")
        print(f"Pesos cargados para {nombre}.")
        return True
    except FileNotFoundError:
        print(f"No se encontraron pesos para {nombre}, entrenando desde cero.")
        return False

datos = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
    [2, 4],
    [4, 5],
    [5, 6],
    [6, 3],
    [10, 2],
    [9, 1]
])

# Normalizar entradas (cada columna se divide por su máximo)
datos_norm = datos / np.max(datos, axis=0)

def normalizar(y):
    return y / np.max(y)

def desnormalizar(y, max_val):
    return y * max_val

# ==========================
# Configuramos y entrenamos (o cargamos) redes
# ==========================

# SUMA
y_suma = np.array([[a + b] for a, b in datos])
mlp_suma = MLP(2, 6, 1)
if cargar_pesos(mlp_suma, "suma"):
    # Si hay pesos, entrenar más para mejorar incrementalmente
    mlp_suma.train(datos, normalizar(y_suma), epochs=5000, lr=0.1, label="Suma (entrenamiento incremental)")
else:
    # Si no hay pesos, entrenar desde cero
    mlp_suma.train(datos, normalizar(y_suma), epochs=20000, lr=0.1, label="Suma")
guardar_pesos(mlp_suma, "suma")

# RESTA
y_resta = np.array([[a - b] for a, b in datos])
mlp_resta = MLP(2, 6, 1)
if cargar_pesos(mlp_resta, "resta"):
    mlp_resta.train(datos, normalizar(y_resta + 20), epochs=5000, lr=0.1, label="Resta (incremental)")
else:
    mlp_resta.train(datos, normalizar(y_resta + 20), epochs=20000, lr=0.1, label="Resta")
guardar_pesos(mlp_resta, "resta")

# MULTIPLICACIÓN
y_mult = np.array([[a * b] for a, b in datos])
mlp_mult = MLP(2, 10, 1)
if cargar_pesos(mlp_mult, "multiplicacion"):
    mlp_mult.train(datos, normalizar(y_mult), epochs=5000, lr=0.1, label="Multiplicación (incremental)")
else:
    mlp_mult.train(datos, normalizar(y_mult), epochs=20000, lr=0.1, label="Multiplicación")
guardar_pesos(mlp_mult, "multiplicacion")

# DIVISIÓN
y_div = np.array([[a / b if b != 0 else 0] for a, b in datos])
mlp_div = MLP(2, 10, 1)
if cargar_pesos(mlp_div, "division"):
    mlp_div.train(datos, normalizar(y_div), epochs=5000, lr=0.1, label="División (incremental)")
else:
    mlp_div.train(datos, normalizar(y_div), epochs=20000, lr=0.1, label="División")
guardar_pesos(mlp_div, "division")

# ==========================
# PRUEBAS
# ==========================
def probar_operaciones(a, b):
    entrada = np.array([[a, b]])

    suma = mlp_suma.forward(entrada)
    resta = mlp_resta.forward(entrada)
    mult = mlp_mult.forward(entrada)
    div = mlp_div.forward(entrada)

    print(f"\nResultados para a = {a}, b = {b}")
    print(f"Suma:         ≈ {desnormalizar(suma, np.max(y_suma))[0][0]:.2f} (real: {a + b})")
    print(f"Resta:        ≈ {desnormalizar(resta, np.max(y_resta + 20))[0][0] - 20:.2f} (real: {a - b})")
    print(f"Multiplicación: ≈ {desnormalizar(mult, np.max(y_mult))[0][0]:.2f} (real: {a * b})")
    if b != 0:
        print(f"División:     ≈ {desnormalizar(div, np.max(y_div))[0][0]:.2f} (real: {a / b:.2f})")
    else:
        print("División:     No se puede dividir entre 0")

# Ejecutar pruebas
probar_operaciones(5, 6)
probar_operaciones(10, 2)
probar_operaciones(9, 1)
probar_operaciones(7, 0)
