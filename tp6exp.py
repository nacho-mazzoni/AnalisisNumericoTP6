import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Definimos parámetros
Lx, Ly = 10e-3, 1e-3  # Dominio (metros)
Nx, Ny = 100, 10       # Número de nodos
dx, dy = Lx/Nx, Ly/Ny  # Tamaño de celda
alpha = 1.14e-6        # Difusividad térmica (m^2/s)
dt = 0.001             # Paso de tiempo (s)
Nt = 2000              # Número de pasos

# Inicializamos las matrices
T = np.full((Ny, Nx), -10.0)  # Temperatura inicial (°C)
T[:, 0] = -10                 # Borde izquierdo
T[:, -1] = 75                # Borde derecho

# Construcción de la matriz del sistema lineal
r = alpha * dt / dx**2
s = alpha * dt / dy**2

main_diag = (1 + 2 * r + 2 * s) * np.ones(Nx * Ny)
off_diag_x = -r * np.ones(Nx * Ny - 1)
off_diag_y = -s * np.ones(Nx * Ny - Nx)

# Ajustamos diagonales para evitar conexiones incorrectas
off_diag_x[np.arange(1, Ny) * Nx - 1] = 0

# Construimos la matriz dispersa
diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
offsets = [0, -1, 1, -Nx, Nx]
A = diags(diagonals, offsets, shape=(Nx * Ny, Nx * Ny), format='csr')

# Simulación
def aplicar_condiciones_de_frontera(T):
    T[:, 0] = -10                # Borde izquierdo
    T[:, -1] = 75               # Borde derecho
    T[0, :] = T[1, :]           # Borde inferior
    T[-1, :] = T[-2, :]         # Borde superior
    return T

def visualizar(T, n):
    plt.clf()
    plt.imshow(T, cmap='coolwarm', extent=[0, Lx*1000, 0, Ly*1000], origin='lower')
    plt.colorbar(label='Temperatura (°C)')
    plt.title(f'Tiempo = {n * dt:.2f} s')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.pause(0.01)

for n in range(1, Nt + 1):
    # Vector b (estado actual)
    T = aplicar_condiciones_de_frontera(T)
    b = T.flatten()

    # Resolvemos el sistema lineal
    T_new = spsolve(A, b).reshape(Ny, Nx)

    # Actualizamos
    T = T_new

    # Visualización
    if n % 100 == 0 or n == 1:
        visualizar(T, n)

plt.show()