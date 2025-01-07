import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
Lx, Ly = 10e-3, 1e-3  # Tamaño del dominio (m)
Nx, Ny = 100, 10      # Número de nodos (y, x)
dx, dy = Lx / Nx, Ly / Ny  # Tamaño de celda
alpha = 1.14e-6       # Difusividad térmica (m^2/s)

# Parámetros temporales
dt = 0.5 * min(dx**2, dy**2) / (2 * alpha) * 0.5  # Condición de estabilidad
Nt = int(20 / dt)      # Número de pasos de tiempo

# Temperaturas características
T_inicial = -10.0      # Temperatura inicial (en toda la placa)
T_derecha_max = 75.0   # Temperatura máxima en el borde derecho

# Inicialización del campo de temperatura
def inicializar_temperatura(Nx, Ny, T_inicial):
    return np.full((Ny, Nx), T_inicial)

# Cálculo del Laplaciano con stencil
def calcular_laplaciano(T, dx, dy):
    Ny, Nx = T.shape
    laplaciano = np.zeros_like(T)
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            laplaciano[i, j] = (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dy**2 +  # Derivada segunda en y
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dx**2    # Derivada segunda en x
            )
    return laplaciano

# Aplicar condiciones de frontera
def aplicar_condiciones_frontera(T, n, dt, T_derecha_max):
    T[:, 0] = T[:, 1]   # Frontera izquierda: Neumann
    T[:, -1] = -10 + (85 + 10) * np.clip(n * dt / 10, 0, 1)  # Frontera derecha: rampa
    T[0, :] = T[1, :]   # Frontera inferior: Neumann
    T[-1, :] = T[-2, :] # Frontera superior: Neumann
    return T

# Actualizar el campo de temperatura
def actualizar_temperatura(T, alpha, dt, dx, dy):
    T_nueva = T.copy()
    laplaciano = calcular_laplaciano(T, dx, dy)
    for i in range(1, T.shape[0] - 1):
        for j in range(1, T.shape[1] - 1):
            T_nueva[i, j] += alpha * dt * laplaciano[i, j]
    return T_nueva

# Visualizar el campo de temperatura
def visualizar(T, ax, n, dt, im):
    im.set_data(T)
    ax.set_title(f'Temperatura (\u00b0C) - t = {n*dt:.1f} s')
    plt.pause(0.01)

# Función principal
def main():
    T = inicializar_temperatura(Nx, Ny, T_inicial)

    plt.ion()  # Modo interactivo
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(T, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                   origin='lower', cmap='coolwarm', vmin=-10, vmax=T_derecha_max)
    plt.colorbar(im, ax=ax, label='Temperatura (\u00b0C)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    for n in range(Nt):
        T = aplicar_condiciones_frontera(T, n, dt, T_derecha_max)
        T = actualizar_temperatura(T, alpha, dt, dx, dy)

        if n % int(5 / dt) == 0:  # Visualizar cada 5 segundos
            visualizar(T, ax, n, dt, im)

    plt.ioff()
    plt.show()

# Ejecutar la simulación
main()
