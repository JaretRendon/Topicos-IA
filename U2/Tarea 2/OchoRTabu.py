import random
import time

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

def gen_vecinos(Solucion):
    # Genera los vecinos de la solución actual cambiando la posición de una reina
    vecinos = []
    n = len(Solucion)
    for i in range(n):
        for j in range(n):
            if Solucion[i] != j:
                vecino = Solucion.copy()
                vecino[i] = j
                vecinos.append(vecino)
    return vecinos

def busc_tabu(n, max_iteraciones, tam_tabu):
    # Ejecuta la búsqueda tabú
    tiempo_inicio = time.time()

    solucion_actual = Solucion_Inicial(n)
    mejor_solucion = solucion_actual.copy()
    lista_tabu = []
    num_movimientos = 0  

    for iteracion in range(1, max_iteraciones + 1):
        if Colisiones(solucion_actual) == 0:
            break  # Se encontró una solución óptima

        vecinos = gen_vecinos(solucion_actual)
        vecinos = sorted(vecinos, key=lambda x: Colisiones(x))

        encontrado = False
        for vecino in vecinos:
            if vecino not in lista_tabu and Colisiones(vecino) < Colisiones(solucion_actual):
                solucion_actual = vecino
                lista_tabu.append(solucion_actual)
                if len(lista_tabu) > tam_tabu:
                    lista_tabu.pop(0)
                num_movimientos += 1
                encontrado = True
                break

        print(f"Iteration {iteracion}:")
        tablero(solucion_actual)
        print("Costo:", Colisiones(solucion_actual))
        print("-" * 40)

        if Colisiones(solucion_actual) < Colisiones(mejor_solucion):
            mejor_solucion = solucion_actual.copy()

        if not encontrado:
            # Si todos los movimientos están en la lista Tabú, se elige el mejor posible
            solucion_actual = random.choice(vecinos)
            lista_tabu.append(solucion_actual)
            if len(lista_tabu) > tam_tabu:
                lista_tabu.pop(0)
            num_movimientos += 1

    tiempo_finalizacion = time.time()
    tiempo_total = tiempo_finalizacion - tiempo_inicio

    return mejor_solucion, num_movimientos, tiempo_total

if __name__ == "__main__":
    n = 8
    max_iteraciones = 50
    tam_tabu = 10

    Solucion, num_movimientos, tiempo_total = busc_tabu(n, max_iteraciones, tam_tabu)
    print(" ")
    print("\nSolución Final:")
    tablero(Solucion)
    print(f"Colisiones: {Colisiones(Solucion)}")
    print(f"Movimientos totales: {num_movimientos}")
    print(f"Tiempo de ejecución: {tiempo_total:.4f} segundos")
