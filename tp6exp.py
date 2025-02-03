import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
Lx, Ly = 10e-3, 1e-3  # Tamaño del dominio (m)
Ny, Nx = 10, 100      # Número de nodos (y,x)
dx, dy = Lx/Nx, Ly/Ny # Tamaño de celda
alpha = 1.14e-6       # Difusividad térmica (m^2/s)
L = 334000            # Calor latente de fusión (J/kg)
rho = 917             # Densidad del hielo (kg/m³)
cp = 2100             # Calor específico del hielo (J/kg·K)

# Parámetros temporales
dt = 0.5 * min(dx**2, dy**2)/(2*alpha)   # Condición de estabilidad
Nt = int(100/dt)     # Número de pasos de tiempo

# Temperaturas características
Tf = 0.0    # Temperatura de fusión (°C)
Tliq = 0.05 # Temperatura de liquidus (°C)

def inicializar_temperatura(Nx, Ny):
    """Inicializa el campo de temperatura uniformemente a -10°C."""
    return -10 * np.ones((Ny, Nx))

def actualizar_temperatura(T, T_old, phi, phi_old, Nt, dt, dx, dy):
    """Actualiza el campo de temperatura y fracción de fase."""
    lap_T = (np.roll(T_old, -1, axis=0) - 2*T_old + np.roll(T_old, 1, axis=0))/dy**2 + \
            (np.roll(T_old, -1, axis=1) - 2*T_old + np.roll(T_old, 1, axis=1))/dx**2

    dphi_dt = np.where((Tf <= T_old) & (T_old <= Tliq), 
                        (T_old - Tf)/(Tliq - Tf) - phi_old, 
                        0)

    T[1:-1, 1:-1] += dt * (alpha * lap_T[1:-1, 1:-1] - (L/(cp*rho)) * dphi_dt[1:-1, 1:-1] / dt)

    return T

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
    tiempos_interesantes = [0, 10,25, 50,75, 100]  # Tiempos para graficar
    perfiles_temperatura = {}

    # Configuración de la visualización
    plt.ion()  # Modo interactivo
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
    n=0
    tiempo_actual = 0
    # Bucle temporal
    while tiempo_actual <= 100:
        T_old = T.copy()
        phi_old = phi.copy()

        # Aplicar condiciones de frontera
        T = aplicar_condiciones_frontera(T, n, dt)
        
        # Actualización de temperatura
        T = actualizar_temperatura(T, T_old, phi, phi_old, Nt, dt, dx, dy)
        
        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T, phi)

        tiempo_actual = n * dt
        n += 1
        if any(abs(tiempo_actual - t) < dt for t in tiempos_interesantes):
            perfiles_temperatura[round(tiempo_actual)] = T[Ny//2, :].copy()
        # Visualización en tiempo real
        if n % 100 == 0:
            visualizar(T, phi, ax1, ax2, n, dt, im1, im2)

    # Desactivar modo interactivo y mostrar plot final
    plt.ioff()
    plt.show()
    # Graficar perfiles de temperatura en la línea central
    plt.figure(figsize=(8, 6))
    for tiempo, perfil in perfiles_temperatura.items():
        plt.plot(np.linspace(0, Lx*1000, Nx), perfil, label=f't = {tiempo}s')
    
    plt.xlabel('Posición x (mm)')
    plt.ylabel('Temperatura (°C)')
    plt.title('Evolución de la temperatura en la línea central')
    plt.legend()
    plt.grid()
    plt.show()

# Ejecutar la simulación
main()