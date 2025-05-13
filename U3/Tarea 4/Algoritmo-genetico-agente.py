import random
import numpy as np

# 1. Definir las ciudades y matriz de todas_las_distancias
ciudades = [
    "Bilbao", "Zaragoza", "Gerona", "Barcelona", "Valencia", "Murcia",
    "Albacete", "Granada", "Sevilla", "Jaen", "Madrid",
    "Valladolid", "Vigo", "Celta"
]

# Distancias directas (en km)
distancias = {
    ("Bilbao", "Zaragoza"): 324,
    ("Bilbao", "Celta"): 378,
    ("Zaragoza", "Gerona"): 289,
    ("Zaragoza", "Barcelona"): 296,
    ("Zaragoza", "Madrid"): 190,
    ("Zaragoza", "Valladolid"): 390,
    ("Zaragoza", "Albacete"): 215,
    ("Gerona", "Barcelona"): 100,
    ("Barcelona", "Valencia"): 349,
    ("Valencia", "Murcia"): 241,
    ("Valencia", "Albacete"): 191,
    ("Murcia", "Granada"): 257,
    ("Murcia", "Albacete"): 150,
    ("Albacete", "Madrid"): 251,
    ("Albacete", "Granada"): 244,
    ("Granada", "Jaen"): 207,
    ("Granada", "Sevilla"): 211,
    ("Sevilla", "Jaen"): 125,
    ("Jaen", "Madrid"): 193,
    ("Jaen", "Valladolid"): 411,
    ("Valladolid", "Vigo"): 356,
    ("Valladolid", "Celta"): 235,
    ("Vigo", "Celta"): 171,
    ("Vigo", "Sevilla"): 245
}

# 2. Crear matriz de todas_las_distancias
ciudades_index = {ciudad: i for i, ciudad in enumerate(ciudades)}
n = len(ciudades)
todas_las_distancias = [[float('inf')] * n for _ in range(n)]

# Llenar todas_las_distancias directas
for (c1, c2), dist in distancias.items():
    i, j = ciudades_index[c1], ciudades_index[c2]
    todas_las_distancias[i][j] = dist
    todas_las_distancias[j][i] = dist  # simétrica

# Distancia a sí mismo es 0
for i in range(n):
    todas_las_distancias[i][i] = 0

# Algoritmo para todas_las_distancias mínimas
for k in range(n):
    for i in range(n):
        for j in range(n):
            if todas_las_distancias[i][j] > todas_las_distancias[i][k] + todas_las_distancias[k][j]:
                todas_las_distancias[i][j] = todas_las_distancias[i][k] + todas_las_distancias[k][j]

# Función para obtener distancia entre dos ciudades
def obtener_distancia(ciudad1, ciudad2):
    i, j = ciudades_index[ciudad1], ciudades_index[ciudad2]
    return todas_las_distancias[i][j]

# 3. Función de fitness (inversa de la distancia total)
def fitness(ruta):
    distancia_total = 0
    for i in range(len(ruta)):
        distancia_total += obtener_distancia(ruta[i], ruta[(i + 1) % len(ruta)])
    return 1 / distancia_total

# 4. Operadores genéticos
def seleccion(poblacion, fitnesses):
    total = sum(fitnesses)
    elegir = random.uniform(0, total)
    actual = 0
    for i, ind in enumerate(poblacion):
        actual += fitnesses[i]
        if actual > elegir:
            return ind
    return poblacion[-1]

def cruce_ordenado(pariente1, pariente2):
    tamaño = len(pariente1)
    inicio, end = sorted([random.randint(0, tamaño - 1), random.randint(0, tamaño - 1)])
    hijo = [None] * tamaño
    hijo[inicio:end + 1] = pariente1[inicio:end + 1]
    puntero = 0
    for ciudad in pariente2:
        if ciudad not in hijo:
            while hijo[puntero] is not None:
                puntero += 1
            hijo[puntero] = ciudad
    return hijo

def mutar(ruta, radio_de_mutacion):
    if random.random() < radio_de_mutacion:
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

# 5. Algoritmo genético principal
def algoritmo_genetico(ciudades, generaciones=100, tamaño_poblacion=50, crossover_rate=0.8, radio_de_mutacion=0.3):
    poblacion = [random.sample(ciudades, len(ciudades)) for _ in range(tamaño_poblacion)]
    mejor_ruta, mejor_fitness = None, 0

    for gen in range(generaciones):
        fitnesses = [fitness(ruta) for ruta in poblacion]
        nueva_poblacion = []

        # Elitismo
        mejor_idx_actual = np.argmax(fitnesses)
        if fitnesses[mejor_idx_actual] > mejor_fitness:
            mejor_ruta = poblacion[mejor_idx_actual]
            mejor_fitness = fitnesses[mejor_idx_actual]

        while len(nueva_poblacion) < tamaño_poblacion:
            pariente1 = seleccion(poblacion, fitnesses)
            pariente2 = seleccion(poblacion, fitnesses)

            if random.random() < crossover_rate:
                hijo1 = cruce_ordenado(pariente1, pariente2)
                hijo2 = cruce_ordenado(pariente2, pariente1)
            else:
                hijo1, hijo2 = pariente1.copy(), pariente2.copy()

            hijo1 = mutar(hijo1, radio_de_mutacion)
            hijo2 = mutar(hijo2, radio_de_mutacion)
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion[:tamaño_poblacion]

        if gen % 10 == 0:
            print(f"Gen {gen}: Distancia = {1 / mejor_fitness:.2f} km")

    return mejor_ruta, 1 / mejor_fitness

# 6. Ejecución
mejor_ruta, distancia = algoritmo_genetico(ciudades, generaciones=200)
print(f"\nMejor ruta: {' → '.join(mejor_ruta)}")
print(f"Distancia total: {distancia:.2f} km")
