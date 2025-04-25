import random
import matplotlib.pyplot as plt
import numpy as np
from enjambre2 import Particula, Enjambre  # Importamos las clases base

class ParticulaMejorada(Particula):
    def __init__(self, n_variables, limites_inf=None, limites_sup=None, verbose=True):
        super().__init__(n_variables, limites_inf, limites_sup, verbose)
        self.historial_posiciones = [self.posicion.copy()]
        self.historial_valores = []
        
    def actualizar_posicion(self):
        super().actualizar_posicion()
        self.historial_posiciones.append(self.posicion.copy())
        
    def evaluar(self, funcion_objetivo):
        resultado = super().evaluar(funcion_objetivo)
        self.historial_valores.append(self.mejor_valor)
        return resultado

class EnjambreMejorado(Enjambre):
    def __init__(self, n_particulas, n_variables, limites_inf=None, limites_sup=None, verbose=True):
        super().__init__(n_particulas, n_variables, limites_inf, limites_sup, verbose)
        # Reemplazamos las partículas básicas por las mejoradas
        self.particulas = [ParticulaMejorada(n_variables, limites_inf, limites_sup, verbose) 
                          for _ in range(n_particulas)]
        self.historial_global = []
        
    def optimizar(self, funcion_objetivo, iteraciones=100, w=0.5, c1=1.0, c2=1.0):
        for iteracion in range(iteraciones):
            for particula in self.particulas:
                if particula.evaluar(funcion_objetivo):
                    if (self.mejor_valor_global is None or 
                        particula.mejor_valor < self.mejor_valor_global):
                        self.mejor_pos_global = particula.mejor_pos.copy()
                        self.mejor_valor_global = particula.mejor_valor
            
            for particula in self.particulas:
                particula.actualizar_velocidad(self.mejor_pos_global, w, c1, c2)
                particula.actualizar_posicion()
            
            self.historial_global.append(self.mejor_valor_global)
            
            if self.verbose and (iteracion % 10 == 0 or iteracion == iteraciones-1):
                print(f"Iteración {iteracion}: Mejor valor = {self.mejor_valor_global}")
        
        return self.mejor_pos_global, self.mejor_valor_global
    
    def visualizar_convergencia(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.historial_global, label='Mejor Global')
        
        # Mostrar algunas partículas aleatorias como ejemplo
        for i in random.sample(range(len(self.particulas)), min(5, len(self.particulas))):
            plt.plot(self.particulas[i].historial_valores, alpha=0.3, linestyle='--')
        
        plt.title('Convergencia del Enjambre')
        plt.xlabel('Iteración')
        plt.ylabel('Valor de la Función Objetivo')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def visualizar_trayectorias(self, indice_dimension=0):
        plt.figure(figsize=(10, 6))
        
        for particula in self.particulas:
            trayectoria = [pos[indice_dimension] for pos in particula.historial_posiciones]
            plt.plot(trayectoria, alpha=0.4)
        
        plt.title(f'Trayectorias de las Partículas (Dimensión {indice_dimension})')
        plt.xlabel('Iteración')
        plt.ylabel(f'Posición en dimensión {indice_dimension}')
        plt.grid(True)
        plt.show()
    
    def generar_metricas(self):
        valores_finales = [p.mejor_valor for p in self.particulas]
        return {
            'mejor_global': self.mejor_valor_global,
            'promedio': np.mean(valores_finales),
            'desviacion': np.std(valores_finales),
            'mejor_particula': min(valores_finales),
            'peor_particula': max(valores_finales)
        }

class FuncionesBenchmark:
    @staticmethod
    def esfera(x):
        return sum(xi**2 for xi in x)
    
    @staticmethod
    def rastrigin(x):
        return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
    
    @staticmethod
    def rosenbrock(x):
        return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

# Ejemplo de uso mejorado
if __name__ == "__main__":
    # Configuración
    n_particulas = 30
    n_variables = 2
    limites_inf = [-5.0] * n_variables
    limites_sup = [5.0] * n_variables
    
    # Seleccionar función objetivo
    funcion_obj = FuncionesBenchmark.rastrigin
    
    # Crear enjambre mejorado
    enjambre = EnjambreMejorado(
        n_particulas=n_particulas,
        n_variables=n_variables,
        limites_inf=limites_inf,
        limites_sup=limites_sup,
        verbose=True
    )
    
    # Ejecutar optimización
    mejor_pos, mejor_valor = enjambre.optimizar(funcion_obj, iteraciones=50, w=0.7, c1=1.5, c2=1.5)
    
    # Resultados
    print("\n=== Resultados Finales ===")
    print(f"Mejor solución encontrada: {mejor_pos}")
    print(f"Valor óptimo: {mejor_valor}")
    
    # Métricas
    metricas = enjambre.generar_metricas()
    print("\n=== Métricas ===")
    for k, v in metricas.items():
        print(f"{k.replace('_', ' ').title()}: {v:.6f}")
    
    # Visualizaciones
    enjambre.visualizar_convergencia()
    enjambre.visualizar_trayectorias()