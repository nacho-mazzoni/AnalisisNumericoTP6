import numpy as np

# Parámetros del dominio
Lx = 21e-3  # Longitud total en el eje x (21 mm)
Ly = 11e-3  # Longitud total en el eje y (11 mm)
Nx = 210    # Número de nodos en x
Ny = 110    # Número de nodos en y
dx, dy = Lx / Nx, Ly / Ny  # Tamaño de celda

# Parámetros físicos ajustados
alpha = 1.14e-7       # Difusividad térmica reducida (m^2/s)
L = 334000           # Calor latente de fusión (J/kg)
rho = 917            # Densidad del hielo (kg/m³)
cp = 2100            # Calor específico del hielo (J/kg·K)
Tf = 0.0             # Temperatura de fusión (°C)
Tliq = 0.05          # Temperatura de liquidus (°C)

# Diferencial de tiempo estable
dt_estable =  min(dx**2 / (2 * alpha), dy**2 / (2 * alpha))

def definir_geometria(Ny, Nx):
    T = np.full((Ny, Nx), np.nan)

    # Convertir dimensiones físicas a índices
    ancho_hielo = int(1e-3 / dx)  # 1 mm
    altura_hielo = int(1e-3 / dy)  # 1 mm
    centro_x = Nx // 2
    altura_total = Ny

    # Hielo (parte central de la T)
    T[altura_total - altura_hielo:, centro_x - ancho_hielo//2:centro_x + ancho_hielo//2] = -10.0

    # Parte vertical del agua
    T[:altura_total - altura_hielo, centro_x - ancho_hielo//2:centro_x + ancho_hielo//2] = 20.0

    # Partes horizontales de agua
    T[altura_total - altura_hielo:, :centro_x - ancho_hielo//2] = 20.0
    T[altura_total - altura_hielo:, centro_x + ancho_hielo//2:] = 20.0

    return T

def inicializar_phi(T):
    phi = np.ones_like(T)
    phi[T <= Tf] = 0
    return phi

def actualizarFraccionFase(T):
    phi = np.ones_like(T)
    mascara_valida = ~np.isnan(T)
    
    phi[mascara_valida] = np.clip(
        (T[mascara_valida] - Tf) / (Tliq - Tf),
        0, 1
    )
    return phi

def actualizarTemperatura(T, phi, phi_old, dt):
    T_nuevo = T.copy()
    mascara_valida = ~np.isnan(T)
    
    lap_T = np.zeros_like(T)
    for i in range(1, T.shape[0] - 1):
        for j in range(1, T.shape[1] - 1):
            if mascara_valida[i, j]:
                lap_T[i, j] = (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dy**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dx**2
                )

    # Factor de ajuste para el calor latente
    factor_ajuste = 15.0
    
    # Calcular cambio de fase y calor latente
    delta_phi = phi - phi_old
    calor_latente = factor_ajuste * L * delta_phi / (cp * dt)
    
    # Actualizar temperatura
    T_nuevo[mascara_valida] += dt * (
        alpha * lap_T[mascara_valida] -
        calor_latente[mascara_valida] / factor_ajuste
    )
    
    return T_nuevo

def aplicarCondicionesFrontera(T):
    mascara_valida = ~np.isnan(T)
    
    # Bordes verticales
    T[:, 0][mascara_valida[:, 0]] = T[:, 1][mascara_valida[:, 0]]
    T[:, -1][mascara_valida[:, -1]] = T[:, -2][mascara_valida[:, -1]]
    
    # Bordes horizontales
    T[0, :][mascara_valida[0, :]] = T[1, :][mascara_valida[0, :]]
    T[-1, :][mascara_valida[-1, :]] = T[-2, :][mascara_valida[-1, :]]
    
    return T

def porcentajeRestanteHielo(phi, phi_inicial):
    hielo_inicial = np.sum(phi_inicial == 0)
    hielo_actual = np.sum(phi == 0)
    return (hielo_actual / hielo_inicial) * 100 if hielo_inicial > 0 else 0

def main():
    T = definir_geometria(Ny, Nx)
    phi = inicializar_phi(T)
    phi_inicial = phi.copy()
    tiempo_derretimiento = 0.0
    
    while True:
        T_old = T.copy()
        phi_old = phi.copy()
        
        T = actualizarTemperatura(T, phi, phi_old, dt_estable)
        T = aplicarCondicionesFrontera(T)
        phi = actualizarFraccionFase(T)
        
        tiempo_derretimiento += dt_estable
        porcentaje = porcentajeRestanteHielo(phi, phi_inicial)
        
        if tiempo_derretimiento % 0.01 < dt_estable:
            print(f"Tiempo: {tiempo_derretimiento:.2f}s, Porcentaje de hielo: {porcentaje:.2f}%")
        
        if porcentaje <= 50:
            print(f"Tiempo final: {tiempo_derretimiento:.2f}s")
            break
        
        if tiempo_derretimiento > 5:
            print("Tiempo máximo alcanzado")
            break

if __name__ == "__main__":
    main()