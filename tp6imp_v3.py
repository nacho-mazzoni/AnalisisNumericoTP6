import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parámetros físicos
Lx, Ly = 10e-3, 1e-3  # Tamaño del dominio (m)
Ny, Nx = 10, 100      # Número de nodos (y,x)
dx, dy = Lx/Nx, Ly/Ny # Tamaño de celda
alpha = 1.14e-6       # Difusividad térmica (m^2/s)
L = 334000            # Calor latente de fusión (J/kg)
rho = 917             # Densidad del hielo (kg/m³)
cp = 2100             # Calor específico del hielo (J/kg·K)

# Parámetros temporales ajustados
dt = 0.01  # Paso de tiempo más pequeño
Nt = int(100/dt)    # Número de pasos de tiempo aumentado

# Temperaturas características
Tf = 0.0    # Temperatura de fusión (°C)
Tliq = 0.05 # Temperatura de liquidus (°C)

def inicializar_temperatura(Nx, Ny):
    """Inicializa el campo de temperatura con un gradiente más frío."""
    T = -20 + 20 * np.tanh(np.linspace(-5, 5, Nx))  # Temperatura inicial más fría
    return T.reshape(1, Nx).repeat(Ny, axis=0)

def construir_matriz(Nx, Ny, dx, dy, alpha, dt):
    """Construye la matriz del sistema lineal para el esquema implícito."""
    NxNy = Nx * Ny
    r = alpha * dt / dx**2
    s = alpha * dt / dy**2
    
    # Diagonales
    main_diag = (1 + 2 * r + 2 * s) * np.ones(NxNy)
    x_diag = -r * np.ones(NxNy - 1)
    y_diag = -s * np.ones(NxNy - Nx)

    # Ajuste para no cruzar fronteras en x
    for i in range(1, Ny):
        x_diag[i * Nx - 1] = 0

    # Matriz dispersa
    diagonals = [main_diag, x_diag, x_diag, y_diag, y_diag]
    offsets = [0, -1, 1, -Nx, Nx]
    A = diags(diagonals, offsets, shape=(NxNy, NxNy), format="csr")
    return A

def aplicar_condiciones_frontera(T, n, dt):
    """Aplica las condiciones de frontera a la temperatura."""
    T[:, 0] = T[:, 1]   # Izquierda
    # Rampa de temperatura más suave en el borde derecho
    T[:, -1] = -20 + (30) * np.clip(n * dt / 100, 0, 1)  # Temperatura máxima reducida
    T[0, :] = T[1, :]   # Inferior
    T[-1, :] = T[-2, :] # Superior
    return T

def resolver_implicito(A, T, Nx, Ny):
    """Resuelve el sistema implícito Ax = b."""
    b = T.flatten()
    T_new = spsolve(A, b)
    return T_new.reshape((Ny, Nx))

def actualizar_fraccion_fase(T, phi):
    """Actualiza la fracción de fase basándose en la temperatura."""
    return np.clip(np.where(T <= Tf, 0, 
                            np.where(T >= Tliq, 1, 
                                     (T - Tf)/(Tliq - Tf))), 0, 1)

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
    A = construir_matriz(Nx, Ny, dx, dy, alpha, dt)

    # Configuración de la visualización
    plt.ion()  # Modo interactivo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.tight_layout(h_pad=2)

    # Inicializar las visualizaciones con límites ajustados
    im1 = ax1.imshow(T, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                     origin='lower', cmap='coolwarm', vmin=-20, vmax=20)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Temperatura (°C)')

    im2 = ax2.imshow(phi, aspect='auto', extent=[0, Lx*1000, 0, Ly*1000], 
                     origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Fracción de fase (φ)')
    ax2.set_xlabel('x (mm)')

    # Bucle temporal
    for n in range(Nt):
        # Aplicar condiciones de frontera
        T = aplicar_condiciones_frontera(T, n, dt)

        # Resolver sistema implícito
        T = resolver_implicito(A, T, Nx, Ny)
        
        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T, phi)
        
        # Visualización en tiempo real
        if n % 100 == 0:
            visualizar(T, phi, ax1, ax2, n, dt, im1, im2)

    # Desactivar modo interactivo y mostrar plot final
    plt.ioff()
    plt.show()

# Ejecutar la simulación
if __name__ == "__main__":
    main()