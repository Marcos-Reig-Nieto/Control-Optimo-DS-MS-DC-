import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parámetros del problema
g = 9.81         # Gravedad (m/s^2)
v0 = 20.0        # Velocidad inicial (m/s)
theta0 = np.pi / 4  # Ángulo inicial (radianes)
T = 2 * v0 * np.sin(theta0) / g           # Tiempo total de vuelo estimado (s)
N = 20           # Número de intervalos (colocación)
dt = T / N       # Paso de tiempo

# Función de restricción para Direct Collocation
def dynamics_constraints(z):
    x = z[:N+1]
    y = z[N+1:2*(N+1)]
    vx = z[2*(N+1):3*(N+1)]
    vy = z[3*(N+1):4*(N+1)]

    cons = []

    # Dinámica (Direct Collocation)
    for k in range(N):
        # Puntos de colocación intermedios
        x_mid = 0.5 * (x[k] + x[k+1]) - dt / 8 * (vx[k+1] - vx[k])
        y_mid = 0.5 * (y[k] + y[k+1]) - dt / 8 * (vy[k+1] - vy[k])
        vx_mid = 0.5 * (vx[k] + vx[k+1])
        vy_mid = 0.5 * (vy[k] + vy[k+1]) - dt / 2 * g

        # Restricciones de dinámica
        cons.append(x_mid - (x[k] + dt / 2 * vx[k]))
        cons.append(y_mid - (y[k] + dt / 2 * vy[k]))
        cons.append(vx_mid - vx[k])
        cons.append(vy_mid - (vy[k] - dt / 2 * g))

    # Condiciones iniciales
    cons.append(x[0])                                # x0 = 0
    cons.append(y[0])                                # y0 = 0
    cons.append(vx[0] - v0 * np.cos(theta0))          # vx0 = v0 * cos(theta0)
    cons.append(vy[0] - v0 * np.sin(theta0))          # vy0 = v0 * sin(theta0)

    # Restricción final (tocar el suelo)
                          # y(T) = 0

    return np.array(cons)

# Función objetivo
def objective(z):
    x = z[:N+1]
    return -x[-1]  # Maximizar x final

# Estimación inicial
def initial_guess():
    t = np.linspace(0, T, N+1)
    x_guess = v0 * np.cos(theta0) * t
    y_guess = v0 * np.sin(theta0) * t - 0.5 * g * t**2
    vx_guess = np.full(N+1, v0 * np.cos(theta0))
    vy_guess = v0 * np.sin(theta0) - g * t
    return np.concatenate([x_guess, y_guess, vx_guess, vy_guess])

# Configuración del optimizador
z0 = initial_guess()

constraints = {
    'type': 'eq',
    'fun': dynamics_constraints
}

# Se agregan bounds: x y y no deben ser negativos, velocidades sin límites estrictos
bounds = []
# Para x_guess: se supone x>=0
for _ in range(N+1):
    bounds.append((0, None))
# Para y_guess: puede ser cualquier valor
for _ in range(N+1):
    bounds.append((None, None))
# Para vx: sin restricciones  
for _ in range(N+1):
    bounds.append((None, None))
# Para vy: sin restricciones  
for _ in range(N+1):
    bounds.append((None, None))

options = {'disp': True, 'maxiter': 1000, 'ftol': 1e-9}

solution = minimize(objective, z0, bounds=bounds, constraints=constraints, method='SLSQP', options=options)

# Procesar y mostrar resultados
if solution.success:
    z_opt = solution.x
    x_opt = z_opt[:N+1]
    y_opt = z_opt[N+1:2*(N+1)]

    plt.figure(figsize=(8, 5))
    plt.plot(x_opt, y_opt, 'o-', label='Trayectoria óptima')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Tiro Parabólico (Direct Collocation)')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    print(f"Distancia máxima alcanzada: {x_opt[-1]:.2f} m")
else:
    print("Optimización no convergió")