import random

def crear_particula(n_variables, limites_inf=None, limites_sup=None, verbose=True):
    if limites_inf is None:
        limites_inf = [-10.0] * n_variables
    if limites_sup is None:
        limites_sup = [10.0] * n_variables

    # Crear una posición aleatoria
    posicion = [random.uniform(l_inf, l_sup) for l_inf, l_sup in zip(limites_inf, limites_sup)]

    # Velocidad inicial
    velocidad = [0.0] * n_variables

    # Mejor posición (igual a la inicial)
    mejor_pos = posicion[:]

    # Valor de función objetivo
    mejor_valor = None

    particula = {
        'posicion': posicion,
        'velocidad': velocidad,
        'mejor_pos': mejor_pos,
        'mejor_valor': mejor_valor
    }

    if verbose:
        print(f"Partícula creada: {particula}")

    return particula


def crear_enjambre(n_particulas, n_variables, limites_inf=None, limites_sup=None, verbose=True):
    # Límites por defecto
    if limites_inf is None:
        limites_inf = [-10.0] * n_variables
    if limites_sup is None:
        limites_sup = [10.0] * n_variables

    enjambre = []
    for i in range(n_particulas):
        particula = crear_particula(n_variables, limites_inf, limites_sup, verbose)
        enjambre.append(particula)

    if verbose:
        print(f"\nEnjambre creado con {n_particulas} partículas.\n")

    return enjambre


# Parámetros
n_particulas = 100
n_variables = 2
limites_inf = [-5] * n_variables
limites_sup = [5] * n_variables
verbose = True

# Crear el enjambre
enjambre = crear_enjambre(
    n_particulas=n_particulas,
    n_variables=n_variables,
    limites_inf=limites_inf,
    limites_sup=limites_sup,
    verbose=verbose
)

