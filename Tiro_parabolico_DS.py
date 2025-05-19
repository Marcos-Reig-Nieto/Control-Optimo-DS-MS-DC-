import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

tiempo_inicio = time.time()
# Parámetros conocidos
g = 9.81         # Gravedad (m/s^2)
v0 = 20.0        # Velocidad inicial (m/s)
theta0 = np.pi/4  # Ángulo inicial (radianes)

# Se utilizará Direct Shooting, donde la única variable de decisión es el tiempo final T.
# Se integrará la trayectoria (con N intervalos) mediante el método de Euler.

def simulate_trajectory(T, N=50):
    dt = T / N
    # Inicialización de la trayectoria
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    vx = v0 * np.cos(theta0)  # vx es constante
    vy = v0 * np.sin(theta0)  # velocidad inicial en y

    # Integración por el método de Euler
    for k in range(N):
        x[k+1] = x[k] + dt * vx
        y[k+1] = y[k] + dt * vy
        vy = vy - dt * g  # la aceleración en x es 0 y en y es -g
    return x, y

# Función objetivo: queremos maximizar la distancia horizontal alcanzada, es decir, x(T).
# Se minimiza el negativo de x(T).
def objective_decision(T):
    x, _ = simulate_trajectory(T[0])
    return -x[-1]

# Restricción: la condición terminal es que y(T)=0 (tocar el suelo)
def constraint_y(T):
    _, y = simulate_trajectory(T[0])
    return y[-1]

# Valor inicial de T (estimado)
T0 = np.array([2.5])

# Definir la restricción
cons = {'type': 'eq', 'fun': constraint_y}

# Resolver la optimización con SLSQP
solution = minimize(objective_decision, T0, constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 1000})

if solution.success:
    T_opt = solution.x[0]
    print(f"Tiempo óptimo T: {T_opt:.4f} s")
else:
    T_opt = T0[0]
    print("Optimización no convergió; se usa la estimación inicial T")

# Simulación de la trayectoria usando el tiempo óptimo (o la estimación en caso de fallo)
x_traj, y_traj = simulate_trajectory(T_opt)

plt.figure(figsize=(8, 5))
plt.plot(x_traj, y_traj, 'o-', label='Trayectoria')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Tiro Parabólico (Direct Shooting)')
plt.grid(True)
plt.legend()
plt.show()

print(f"Distancia máxima alcanzada: {x_traj[-1]:.2f} m")

tiempo_final = time.time()

tiempo_total = tiempo_final - tiempo_inicio

print(f"Tiempo del programa: {tiempo_total}")
