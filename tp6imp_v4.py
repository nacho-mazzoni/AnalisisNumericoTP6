import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
Lx, Ly = 10e-3, 1e-3  # Tamaño del dominio (m)
Nx, Ny = 100, 10      # Número de nodos (y,x)
dx, dy = Lx/Nx, Ly/Ny # Tamaño de celda
alpha = 1.14e-6       # Difusividad térmica (m^2/s)
L = 334000            # Calor latente de fusión (J/kg)
rho = 917             # Densidad del hielo (kg/m³)
cp = 2100             # Calor específico del hielo (J/kg·K)

# Parámetros temporales
dt = 0.5 * min(dx**2, dy**2)/(2*alpha) * 0.5  # Condición de estabilidad
Nt = int(100/dt)     # Número de pasos de tiempo

# Temperaturas características
Tf = 0.0    # Temperatura de fusión (°C)
Tliq = 0.05 # Temperatura de liquidus (°C)

def inicializar_temperatura(Nx, Ny):
    """Inicializa el campo de temperatura con un gradiente suave."""
    T = -10 + 10 * np.tanh(np.linspace(-5, 5, Nx))  # Rango ajustable
    return T.reshape(1, Nx).repeat(Ny, axis=0)

from scipy.linalg import solve

def actualizar_temperatura_implicito(T, dt, dx, dy, alpha):
    """Actualiza el campo de temperatura usando un esquema implícito."""
    Ny, Nx = T.shape
    N = Ny * Nx  # Tamaño total de la matriz

    # Construir la matriz A
    h2 = dx**2  # Suponemos dx = dy para simplicidad
    A = np.eye(N) - (alpha * dt / h2) * construir_laplaciano(Nx, Ny)

    # Vectorizar T y resolver el sistema
    T_vec = T.flatten()  # Vector columna
    T_new_vec = solve(A, T_vec)

    # Reconvertir a la forma de matriz 2D
    return T_new_vec.reshape(Ny, Nx)

def construir_laplaciano(Nx, Ny):
    """Construye la matriz del operador Laplaciano en 2D."""
    N = Nx * Ny
    L = np.zeros((N, N))

    for i in range(N):
        L[i, i] = -4  # Término central
        if i % Nx != 0:  # No es borde izquierdo
            L[i, i - 1] = 1
        if (i + 1) % Nx != 0:  # No es borde derecho
            L[i, i + 1] = 1
        if i - Nx >= 0:  # No es borde superior
            L[i, i - Nx] = 1
        if i + Nx < N:  # No es borde inferior
            L[i, i + Nx] = 1

    return L


def actualizar_fraccion_fase(T, phi):
    """Actualiza la fracción de fase basándose en la temperatura."""
    return np.clip(np.where(T <= Tf, 0, 
                            np.where(T >= Tliq, 1, 
                                     (T - Tf)/(Tliq - Tf))), 0, 1)

def aplicar_condiciones_frontera(T, n, dt):
    """Aplica las condiciones de frontera a la temperatura."""
    T[:, 0] = T[:, 1]   # Izquierda
    T[:, -1] = -10 + (85 + 10) * np.clip(n * dt / 10, 0, 1)  # Derecha: rampa
    T[0, :] = T[1, :]   # Inferior
    T[-1, :] = T[-2, :] # Superior
    return T

def visualizar(T, phi, ax1, ax2, n, dt, im1, im2):
    """Actualiza la visualización en tiempo real."""
    im1.set_data(T)
    im2.set_data(phi)
    ax1.set_title(f'Temperatura (°C) - t = {n*dt:.1f} s')
    plt.pause(0.01)

def main():
    """Función principal para ejecutar la simulación."""
    # Inicialización de temperatura y fracción de fase
    T = inicializar_temperatura(Nx, Ny)
    phi = np.zeros((Ny, Nx))  # Fracción de fase (0:hielo, 1:agua)

    # Configuración de la visualización
    plt.ion()  # Modo interactivo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.tight_layout(h_pad=2)

    # Inicializar las visualizaciones
    im1 = ax1.imshow(T, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                     origin='lower', cmap='coolwarm')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Temperatura (°C)')

    im2 = ax2.imshow(phi, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                     origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Fracción de fase (φ)')
    ax2.set_xlabel('x (mm)')

    # Bucle temporal
    for n in range(Nt):
        T_old = T.copy()
        phi_old = phi.copy()

        # Aplicar condiciones de frontera
        T = aplicar_condiciones_frontera(T, n, dt)
        
        # Actualización de temperatura
        T = actualizar_temperatura_implicito(T, dt, dx, dy, alpha)
        
        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T, phi)
        
        # Visualización en tiempo real
        if n % 100 == 0:
            visualizar(T, phi, ax1, ax2, n, dt, im1, im2)

    # Desactivar modo interactivo y mostrar plot final
    plt.ioff()
    plt.show()

# Ejecutar la simulación
main()