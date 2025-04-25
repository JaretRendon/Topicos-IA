import random

class Particula:
    def __init__(self, n_variables, limites_inf=None, limites_sup=None, verbose=True):
        """Inicializa una partícula con posición y velocidad aleatorias."""
        # Establecer límites por defecto si no se proporcionan
        if limites_inf is None:
            limites_inf = [-10.0] * n_variables
        if limites_sup is None:
            limites_sup = [10.0] * n_variables
            
        self.n_variables = n_variables
        self.limites_inf = limites_inf
        self.limites_sup = limites_sup
        self.verbose = verbose
        
        # Inicializar posición aleatoria dentro de los límites
        self.posicion = [random.uniform(l_inf, l_sup) for l_inf, l_sup in zip(limites_inf, limites_sup)]
        
        # Inicializar velocidad en cero
        self.velocidad = [0.0] * n_variables
        
        # Mejor posición personal (inicialmente la posición actual)
        self.mejor_pos = self.posicion.copy()
        self.mejor_valor = None
        
        if self.verbose:
            print(f"Partícula creada: Posición: {self.posicion}, Velocidad: {self.velocidad}")

    def actualizar_velocidad(self, mejor_pos_global, w=0.5, c1=1.0, c2=1.0):
        """Actualiza la velocidad de la partícula según el algoritmo PSO."""
        for i in range(self.n_variables):
            # Componente cognitivo (atracción hacia mejor posición personal)
            r1 = random.random()
            cognitivo = c1 * r1 * (self.mejor_pos[i] - self.posicion[i])
            
            # Componente social (atracción hacia mejor posición global)
            r2 = random.random()
            social = c2 * r2 * (mejor_pos_global[i] - self.posicion[i])
            
            # Actualizar velocidad
            self.velocidad[i] = w * self.velocidad[i] + cognitivo + social

    def actualizar_posicion(self):
        """Actualiza la posición de la partícula según su velocidad."""
        for i in range(self.n_variables):
            self.posicion[i] += self.velocidad[i]
            
            # Asegurarse de que la posición está dentro de los límites
            self.posicion[i] = max(self.limites_inf[i], min(self.limites_sup[i], self.posicion[i]))

    def evaluar(self, funcion_objetivo):
        """Evalúa la partícula con una función objetivo y actualiza su mejor posición."""
        valor_actual = funcion_objetivo(self.posicion)
        
        # Si es la primera evaluación o encontramos un mejor valor
        if self.mejor_valor is None or valor_actual < self.mejor_valor:
            self.mejor_pos = self.posicion.copy()
            self.mejor_valor = valor_actual
            return True  # Indica que hubo una mejora
        
        return False  # No hubo mejora


class Enjambre:
    def __init__(self, n_particulas, n_variables, limites_inf=None, limites_sup=None, verbose=True):
        """Inicializa un enjambre de partículas."""
        self.n_particulas = n_particulas
        self.n_variables = n_variables
        self.limites_inf = [-10.0] * n_variables if limites_inf is None else limites_inf
        self.limites_sup = [10.0] * n_variables if limites_sup is None else limites_sup
        self.verbose = verbose
        
        # Crear las partículas del enjambre
        self.particulas = [Particula(n_variables, limites_inf, limites_sup, verbose) 
                           for _ in range(n_particulas)]
        
        # Mejor posición global y su valor
        self.mejor_pos_global = None
        self.mejor_valor_global = None
        
        if verbose:
            print(f"\nEnjambre creado con {n_particulas} partículas.\n")

    def optimizar(self, funcion_objetivo, iteraciones=100, w=0.5, c1=1.0, c2=1.0):
        """Ejecuta el algoritmo PSO para optimizar la función objetivo."""
        for iteracion in range(iteraciones):
            # Evaluar todas las partículas y actualizar mejores posiciones
            for particula in self.particulas:
                if particula.evaluar(funcion_objetivo):
                    # Si la partícula mejoró su mejor posición personal,
                    # verificar si también es la mejor global
                    if (self.mejor_valor_global is None or 
                        particula.mejor_valor < self.mejor_valor_global):
                        self.mejor_pos_global = particula.mejor_pos.copy()
                        self.mejor_valor_global = particula.mejor_valor
            
            # Actualizar velocidades y posiciones
            for particula in self.particulas:
                particula.actualizar_velocidad(self.mejor_pos_global, w, c1, c2)
                particula.actualizar_posicion()
            
            if self.verbose and (iteracion % 10 == 0 or iteracion == iteraciones-1):
                print(f"Iteración {iteracion}: Mejor valor = {self.mejor_valor_global}")
        
        return self.mejor_pos_global, self.mejor_valor_global


# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros
    n_particulas = 100
    n_variables = 2
    limites_inf = [-5] * n_variables
    limites_sup = [5] * n_variables
    
    # Función objetivo de ejemplo (esfera)
    def funcion_objetivo(x):
        return sum(xi**2 for xi in x)
    
    # Crear enjambre
    enjambre = Enjambre(
        n_particulas=n_particulas,
        n_variables=n_variables,
        limites_inf=limites_inf,
        limites_sup=limites_sup,
        verbose=True
    )
    
    # Ejecutar optimización
    mejor_pos, mejor_valor = enjambre.optimizar(funcion_objetivo, iteraciones=50)
    
    print(f"\nMejor solución encontrada: Posición: {mejor_pos}, Valor: {mejor_valor}")