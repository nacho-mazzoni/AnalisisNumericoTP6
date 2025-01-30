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
    phi = np.full_like(T, np.nan)
    phi[T <= Tf] = 0.0
    phi[T > Tf] = 1.0
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
    mascara_valida = ~np.isnan(T)  # Filtramos las celdas válidas de T
    
    lap_T = np.zeros_like(T)  # Inicializamos la laplaciana de T con ceros
    for i in range(1, T.shape[0] - 1):
        for j in range(1, T.shape[1] - 1):
            if mascara_valida[i, j]:
                # Identificar los vecinos válidos de la celda (i, j)
                vecinos = []
                if mascara_valida[i+1, j]:  # Abajo
                    vecinos.append(T[i+1, j])
                if mascara_valida[i-1, j]:  # Arriba
                    vecinos.append(T[i-1, j])
                if mascara_valida[i, j+1]:  # Derecha
                    vecinos.append(T[i, j+1])
                if mascara_valida[i, j-1]:  # Izquierda
                    vecinos.append(T[i, j-1])
                
                if len(vecinos)==4:
                    lap_T[i, j] = ((T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dy**2 +
                        (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dx**2)
                elif len(vecinos) >= 2: # Si hay al menos 2 vecinos válidos, calculamos la laplaciana
                        lap_T[i, j] = (
                            (np.mean(vecinos) - T[i, j]) / dx**2  # Promediamos los vecinos válidos solamente porque dx == dy
                        )

    # Factor de ajuste para el calor latente
    factor_ajuste = 15.0
    
    # Calcular cambio de fase y calor latente
    delta_phi = phi - phi_old
    calor_latente = factor_ajuste * L * delta_phi / (cp * dt)
    
    # Actualizar temperatura solo en celdas válidas
    T_nuevo[mascara_valida] += dt * (
        alpha * lap_T[mascara_valida] - calor_latente[mascara_valida] / factor_ajuste
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
    if hielo_inicial <=0 : return 0
    else: return ((hielo_actual/hielo_inicial)*100)

def main():
    T = definir_geometria(Ny, Nx)
    phi = inicializar_phi(T)
    phi_inicial = phi.copy()
    tiempo_derretimiento = 0.0
    porcentaje = porcentajeRestanteHielo(phi, phi_inicial)
    while porcentaje >= 50:
        phi_old = phi.copy()
        T = actualizarTemperatura(T, phi, phi_old, dt_estable)
        T = aplicarCondicionesFrontera(T)
        phi = actualizarFraccionFase(T)

        print(f"Porcentaje restante de hielo: {porcentaje: .2f}%")
        porcentaje = porcentajeRestanteHielo(phi, phi_inicial)
        tiempo_derretimiento += dt_estable

    print(f"Tiempo total en derretirse: {tiempo_derretimiento: .2f} segundos")
        
#Ejecutar codigo
main()