import random
import time
import math

def tablero(reinas):
    # Impresión del tablero con las reinas en la posición
    n = len(reinas)
    Linea_Horizontal = " " + "_" * (4 * n - 1)

    print(Linea_Horizontal)
    for i in range(n):
        fila = "|"
        for j in range(n):
            fila += " ♛ |" if reinas[i] == j else "   |"
        print(fila)
        print(Linea_Horizontal)

def Solucion_Inicial(n):
    # Crea una configuración aleatoria inicial
    return [random.randint(0, n - 1) for _ in range(n)]

def Colisiones(reinas):
    # Calcula la cantidad de colisiones en el tablero
    n = len(reinas)
    colisiones = 0
    for i in range(n):
        for j in range(i + 1, n):
            if reinas[i] == reinas[j] or abs(reinas[i] - reinas[j]) == abs(i - j):
                colisiones += 1
    return colisiones

def gen_vecino(Solucion):
    # Genera un vecino aleatorio cambiando la posición de una reina
    n = len(Solucion)
    vecino = Solucion.copy()
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    vecino[i] = j
    return vecino

def recocido_simulado(n, temp_inicial, temp_final, alpha, L):
    # Ejecuta el algoritmo de recocido simulado
    tiempo_inicio = time.time()
    
    solucion_actual = Solucion_Inicial(n)
    mejor_solucion = solucion_actual.copy()
    temperatura = temp_inicial
    num_movimientos = 0
    iteracion = 0
    
    while temperatura > temp_final:
        iteracion += 1
        
        # Realizar L iteraciones a esta temperatura
        # L = Nro. iteraciones p/temperatura
        for _ in range(L):
            if Colisiones(solucion_actual) == 0:
                break  # Solución óptima encontrada
            
            # Generar un vecino aleatorio
            vecino = gen_vecino(solucion_actual)
            delta = Colisiones(vecino) - Colisiones(solucion_actual)
            
            # Criterio de aceptación
            if delta < 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual = vecino
                num_movimientos += 1
                
                # Actualizar la mejor solución encontrada
                if Colisiones(solucion_actual) < Colisiones(mejor_solucion):
                    mejor_solucion = solucion_actual.copy()
        
        # Mostrar información de la iteración
        print(f"Iteration {iteracion}:")
        tablero(solucion_actual)
        print("Costo:", Colisiones(solucion_actual))
        print("Temperatura:", temperatura)
        print("-" * 40)
        
        # Enfriar el sistema
        temperatura *= alpha
    
    tiempo_finalizacion = time.time()
    tiempo_total = tiempo_finalizacion - tiempo_inicio
    
    return mejor_solucion, num_movimientos, tiempo_total

if __name__ == "__main__":
    n = 8
    temp_inicial = 120.0
    temp_final = 0.1
    alpha = 0.95  # Factor de enfriamiento
    # alpha = random.uniform(0.8,0.99)
    L = 100       # Longitud de la cadena de Markov (iter p/temp)

    Solucion, num_movimientos, tiempo_total = recocido_simulado(
        n, temp_inicial, temp_final, alpha, L)
    
    print("\nSolución Final:")
    tablero(Solucion)
    print(f"Colisiones: {Colisiones(Solucion)}")
    print(f"Movimientos totales: {num_movimientos}")
    print(f"Tiempo de ejecución: {tiempo_total:.4f} segundos")