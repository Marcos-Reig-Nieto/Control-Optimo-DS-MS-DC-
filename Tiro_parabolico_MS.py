import numpy as np
import matplotlib.pyplot as plt
import time

tiempo_inicio = time.time()
class TiroParabolico:
    def __init__(self, v0, angulo, h0=0, masa=1.0, cd=0.47, area=0.01, rho=1.225, g=9.81):
        self.v0 = v0
        self.angulo = angulo
        self.h0 = h0
        self.masa = masa
        self.cd = cd
        self.area = area
        self.rho = rho
        self.g = g

    def calcular_trayectoria(self, dt=0.01, max_tiempo=100):
        vx = self.v0 * np.cos(self.angulo)
        vy = self.v0 * np.sin(self.angulo)
        x = 0
        y = self.h0

        posiciones_x = [x]
        posiciones_y = [y]

        t = 0
        while y >= 0 and t <= max_tiempo:
            v = np.sqrt(vx**2 + vy**2)
            fuerza_drag = 0.5 * self.rho * self.cd * self.area * v**2

            fx_drag = -fuerza_drag * (vx / v)
            fy_drag = -fuerza_drag * (vy / v)

            ax = fx_drag / self.masa
            ay = (fy_drag / self.masa) - self.g

            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt

            posiciones_x.append(x)
            posiciones_y.append(y)

            t += dt

        return posiciones_x, posiciones_y


class MultipleShooting:
    def __init__(self, tiro, N, T):
        self.tiro = tiro
        self.N = N
        self.T = T
        self.dt = T / N

    def simular(self):
        posiciones_x = []
        posiciones_y = []

        for i in range(self.N):
            t0 = i * self.dt
            t1 = (i + 1) * self.dt
            x, y = self.simular_intervalo(t0, t1)
            posiciones_x.append(x)
            posiciones_y.append(y)

        return posiciones_x, posiciones_y

    def simular_intervalo(self, t0, t1):
        # Simular el intervalo usando la física del tiro parabólico
        vx = self.tiro.v0 * np.cos(self.tiro.angulo)
        vy = self.tiro.v0 * np.sin(self.tiro.angulo) - self.tiro.g * t1
        x = vx * t1
        y = self.tiro.h0 + vy * t1 - 0.5 * self.tiro.g * t1**2
        return x, y


def main():
    # Parámetros
    g = 9.81  # Gravedad
    v0 = 20  # Velocidad inicial en m/s
    theta0 = np.pi / 4  # Ángulo de lanzamiento (45 grados)
    N = 20  # Número de nodos (colocación)
    T = 2.5  # Tiempo total del vuelo (estimado)

    # Crear instancia de TiroParabolico
    tiro = TiroParabolico(v0, theta0, g=g)

    # Crear instancia de MultipleShooting
    multiple_shooting = MultipleShooting(tiro, N, T)

    # Ejecutar la simulación
    posiciones_x, posiciones_y = multiple_shooting.simular()

    # Graficar la trayectoria
    plt.figure(figsize=(10, 6))
    plt.plot(posiciones_x, posiciones_y, label='Trayectoria con múltiples disparos', color='darkblue')
    plt.title('Simulación de Tiro Parabólico con Múltiples Disparos')
    plt.xlabel('Distancia horizontal (m)')
    plt.ylabel('Altura (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main() 

tiempo_final = time.time()

tiempo_total = tiempo_final - tiempo_inicio

print(f"El programa ha tardado: {tiempo_total}")
