import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
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

def actualizar_temperatura_implicito(T, dt, dx, dy, alpha):
    Ny, Nx = T.shape
    N = Ny * Nx
    
    h2 = dx**2
    A = eye(N, format='csr') - (alpha * dt / h2) * construir_laplaciano(Nx, Ny)
    
    T_vec = T.flatten()
    T_new_vec = spsolve(A, T_vec)  # Usar solver disperso
    
    return T_new_vec.reshape(Ny, Nx)

def construir_laplaciano(Nx, Ny):
    N = Nx * Ny
    L = lil_matrix((N, N))  # Usar matriz dispersa
    
    for i in range(N):
        L[i, i] = -4
        if i % Nx != 0:
            L[i, i - 1] = 1
        if (i + 1) % Nx != 0:
            L[i, i + 1] = 1
        if i - Nx >= 0:
            L[i, i - Nx] = 1
        if i + Nx < N:
            L[i, i + Nx] = 1
            
    return L.tocsr()  # Convertir a formato CSR para operaciones eficientes


def actualizar_fraccion_fase(T, phi):
    """Actualiza la fracción de fase basándose en la temperatura."""
    return np.clip(np.where(T <= Tf, 0, 
                            np.where(T >= Tliq, 1, 
                                     (T - Tf)/(Tliq - Tf))), 0, 1)

def actualizar_temperatura_con_fase(T, phi, dt, dx, dy, alpha, L, cp):
    T_new = actualizar_temperatura_implicito(T, dt, dx, dy, alpha)
    
    # Fusión
    mask_fusion = (T_new > Tf) & (phi < 1)
    q_fusion = rho * cp * (T_new[mask_fusion] - Tf)
    delta_phi_fusion = np.minimum(q_fusion / (rho * L), 1 - phi[mask_fusion])
    phi[mask_fusion] += delta_phi_fusion
    T_new[mask_fusion] = Tf
    
    # Solidificación
    mask_solid = (T_new < Tf) & (phi > 0)
    q_solid = -rho * cp * (T_new[mask_solid] - Tf)
    delta_phi_solid = np.minimum(q_solid / (rho * L), phi[mask_solid])
    phi[mask_solid] -= delta_phi_solid
    T_new[mask_solid] = Tf
    
    return T_new, phi

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
                     origin='lower', cmap='coolwarm', vmin=-10, vmax=50)
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
        T, phi = actualizar_temperatura_con_fase(T, phi, dt, dx, dy, alpha, L, cp)
        
        """
        # Actualización de temperatura
        T = actualizar_temperatura_implicito(T, dt, dx, dy, alpha)
        
        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T, phi)
        """
        
        # Visualización en tiempo real
        if n % 100 == 0:
            visualizar(T, phi, ax1, ax2, n, dt, im1, im2)

    # Desactivar modo interactivo y mostrar plot final
    plt.ioff()
    plt.show()

# Ejecutar la simulación
main()