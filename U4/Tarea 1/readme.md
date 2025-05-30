# Tarea 1
Multi-layer Perceptron (MLP) que resuelve las operaciones aritmeticas 

## Participantes 
Rodrigo Alonso Páez Gastélum
Adrián Jaret Rendón Ríos 

## Objetivo de la actividad 

Implementar el algoritmo Multi-layer Perceptron (MLP), entrenar una red neuronal que resuelve las operaciones aritmeticas 

### Especificación del código

- la funcion relu activa la neurona 
- la funcion relu_derivative sirve para el backpropagation, para ajustar los pesos
- la funcion sigmoid es para las salidas binarias
- la funcion sigmoid_derivative calcula la derivada de la funcion para el backpropagation
- la clase MLP se subdivide en 
- __init__ que inicializa como objeto la clase
- forward es la propagacion hacia adelante para calcular la salida de la red dado un input
- backward es la propagacion hacia atras para ajustar los pesos 
- train es para entrenar la red de manera ciclica 
- la funcion guardar_pesos, guarda los pesos
- se cargan los pesos en la siguiente funcion
- se cargan unos datos de entrenamiento tipo numpy
- se normaliza las entradas para que aprenda mejor la red
- la suma, resta, etc funcionan de una manera parecida, siguen el mismo principio de entrenamiento 
- ejemplo y_suma = np.array([[a + b] for a, b in datos]) prepara los datos creando un array con los resultados
- mlp_suma crea la red neuronal 
- carga los pesos o crea los pesos desde 0
- guarda los pesos 
- el ultimo apartado de pruebas es para probar el funcionamiento de las redes neuronales 