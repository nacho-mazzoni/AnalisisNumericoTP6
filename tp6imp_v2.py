import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parámetros físicos
Lx, Ly = 10e-3, 1e-3  # Tamaño del dominio (m)
Nx, Ny = 100, 10       # Número de nodos (y, x)
dx, dy = Lx / Nx, Ly / Ny  # Tamaño de celda
alpha = 1.14e-6        # Difusividad térmica (m^2/s)
L = 334000             # Calor latente de fusión (J/kg)
rho = 917             # Densidad (kg/m^3)
cp = 2100              # Calor específico (J/(kg·K))

# Parámetros temporales
dt = 0.5 * min(dx**2, dy**2) / (2 * alpha) * 0.5  # Condición de estabilidad
Nt = int(100 / dt)       # Número de pasos de tiempo

# Temperaturas características
T_inicial = -10.0       # Temperatura inicial (en toda la placa)
T_derecha_max = 85.0    # Temperatura máxima en el borde derecho
T_solido = 0.0          # Temperatura inicial de cambio de fase
T_liquido = 5.0         # Temperatura final de cambio de fase

# Inicialización del campo de temperatura y fracción de fase
def inicializar_campos(Nx, Ny, T_inicial):
    T = np.full((Ny, Nx), T_inicial)
    phi = np.zeros((Ny, Nx))  # Fracción de fase inicial: todo sólido
    return T, phi

# Cálculo del Laplaciano con esquema implícito
def construir_matriz_implícita(Nx, Ny, alpha, dt, dx, dy):
    N = Nx * Ny
    diagonal = (1 + 2 * alpha * dt * (1 / dx**2 + 1 / dy**2)) * np.ones(N)
    adyacente_x = (-alpha * dt / dx**2) * np.ones(N-1)
    adyacente_y = (-alpha * dt / dy**2) * np.ones(N-Nx)

    # Condiciones de frontera en las diagonales
    for i in range(1, Ny):
        adyacente_x[i * Nx - 1] = 0  # Evitar conexiones incorrectas en los bordes

    diagonales = [diagonal, adyacente_x, adyacente_x, adyacente_y, adyacente_y]
    offsets = [0, -1, 1, -Nx, Nx]
    matriz = diags(diagonales, offsets, format='csr')
    return matriz

# Resolver el sistema implícito
def resolver_temperatura_implicita(T, phi, matriz, alpha, dt, dx, dy, L, rho, cp):
    Ny, Nx = T.shape
    N = Nx * Ny
    b = T.flatten()

    # Añadir término de cambio de fase
    dphi_dt = (phi - np.clip(T - T_solido, 0, 1)) / dt
    b -= (L / (rho * cp)) * dphi_dt.flatten()

    T_nueva = spsolve(matriz, b)
    return T_nueva.reshape((Ny, Nx))

# Actualizar fracción de fase
def actualizar_fraccion_fase(T, phi, T_solido, T_liquido):
    phi_nueva = phi.copy()
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T_solido <= T[i, j] <= T_liquido:
                phi_nueva[i, j] = (T[i, j] - T_solido) / (T_liquido - T_solido)
            elif T[i, j] > T_liquido:
                phi_nueva[i, j] = 1.0
    return phi_nueva

# Aplicar condiciones de frontera
def aplicar_condiciones_frontera(T, n, dt, T_derecha_max):
    T[:, 0] = T[:, 1]   # Frontera izquierda: Neumann
    T[:, -1] = -10 + (85 + 10) * np.clip(n * dt / 10, 0, 1)  # Frontera derecha: rampa
    T[0, :] = T[1, :]   # Frontera inferior: Neumann
    T[-1, :] = T[-2, :] # Frontera superior: Neumann
    return T

# Visualizar el campo de temperatura y fracción de fase
def visualizar(T, phi, ax1, ax2, n, dt, im1, im2):
    im1.set_data(T)
    im2.set_data(phi)
    ax1.set_title(f'Temperatura (°C) - t = {n*dt:.1f} s')
    plt.pause(0.01)

# Función principal
def main():
    T, phi = inicializar_campos(Nx, Ny, T_inicial)
    matriz = construir_matriz_implícita(Nx, Ny, alpha, dt, dx, dy)

    # Configuración de la visualización
    plt.ion() # Modo interactivo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.tight_layout(h_pad=2)

    # Inicializar las visualizaciones
    im1 = ax1.imshow(T, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                     origin='lower', cmap='coolwarm', vmin=-10, vmax=85)
    
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Temperatura (°C)')

    im2 = ax2.imshow(phi, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                     origin='lower', cmap='viridis', vmin=0, vmax=1)
    
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Fracción de fase (φ)')
    ax2.set_xlabel('x (mm)')

    for n in range(Nt):
        T = aplicar_condiciones_frontera(T, n, dt, T_derecha_max)
        T = resolver_temperatura_implicita(T, phi, matriz, alpha, dt, dx, dy, L, rho, cp)
        phi = actualizar_fraccion_fase(T, phi, T_solido, T_liquido)

        if n % int(5 / dt) == 0:  # Visualizar cada 5 segundos
            visualizar(T, phi, ax1, ax2, n, dt, im1, im2)

    plt.ioff()
    plt.show()

# Ejecutar la simulación
main()
