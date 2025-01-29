import numpy as np

# Parámetros del dominio
Lx = 21e-3  # Longitud total en el eje x (21 mm)
Ly = 11e-3  # Longitud total en el eje y (11 mm)
Nx = 210    # Número de nodos en x
Ny = 110    # Número de nodos en y
dx, dy = Lx / Nx, Ly / Ny  # Tamaño de celda

# Parámetros del problema
alpha = 1.14e-6       # Difusividad térmica (m^2/s)
L = 334000            # Calor latente de fusión (J/kg)
rho = 917             # Densidad del hielo (kg/m³)
cp = 2100             # Calor específico del hielo (J/kg·K)
Tf = 0.0              # Temperatura de fusión (°C)
Tliq = 0.05           # Temperatura de liquidus (°C)

#diferencial de tiempo estable
dt_estable = min((dx**2/2*alpha), (dy**2/2*alpha))

# Inicialización del campo de temperatura
T = np.full((Ny, Nx), np.nan)  # Inicializar todo como nan

# Función para definir la geometría en forma de T
def definir_geometria(T):
    T[:, :] = np.nan  # Inicializar todo fuera del dominio como NaN

    # Convertir dimensiones físicas a índices
    ancho_hielo = int(1e-3 / dx)  # 1 mm
    altura_hielo = int(1e-3 / dy)  # 1 mm
    centro_x = Nx // 2
    altura_total = Ny

    # Hielo (parte central de la T)
    T[altura_total - altura_hielo:, centro_x - ancho_hielo//2:centro_x + ancho_hielo//2] = -10.0

    # Parte vertical del agua
    T[:altura_total - altura_hielo, centro_x - ancho_hielo//2:centro_x + ancho_hielo//2] = 20.0

    # Partes horizontales de agua (a la izquierda y derecha del hielo)
    T[altura_total - altura_hielo:, :centro_x - ancho_hielo//2] = 20.0
    T[altura_total - altura_hielo:, centro_x + ancho_hielo//2:] = 20.0

    return T

# Inicializar la fracción de fase (phi)
def inicializar_phi():
    # Convertir dimensiones físicas a índices
    ancho_hielo = int(1e-3 / dx)  # 1 mm
    altura_hielo = int(1e-3 / dy)  # 1 mm
    centro_x = Nx // 2
    altura_total = Ny

    # Inicializamos phi con 1 (agua) en todo el dominio
    phi = np.ones((Ny, Nx))

    # Establecemos el área de hielo (fracción de fase 0) en el centro
    phi[altura_total - altura_hielo:, centro_x - ancho_hielo//2:centro_x + ancho_hielo//2] = 0

    return phi

def actualizarFraccionFase(T, phi):
    mascara_valida = ~np.isnan(T)
    nueva_phi = np.copy(phi) 
    # Actualizar solo las celdas dentro de la máscara válida
    nueva_phi[mascara_valida] = np.clip(
        np.where(T[mascara_valida] <= Tf, 0,
                 np.where(T[mascara_valida] >= Tliq, 1,
                          (T[mascara_valida] - Tf) / (Tliq - Tf))), 0, 1
    )
    return nueva_phi

# Función para calcular la temperatura en cada punto
def actualizarTemperatura(T, T_old, phi, phi_old, dx, dy):
    mascara_valida = ~np.isnan(T)
    # Verificar dimensiones
    if phi_old.shape != T.shape or phi.shape != T.shape:
        raise ValueError(f"Las dimensiones de phi ({phi.shape}) o phi_old ({phi_old.shape}) no coinciden con T ({T.shape})")

    # Calcular el Laplaciano manualmente sin usar np.roll
    lap_T = np.zeros_like(T)

    # Para cada celda interna
    for i in range(1, T.shape[0] - 1):
        for j in range(1, T.shape[1] - 1):
            if mascara_valida[i, j]:
                lap_T[i, j] = (
                    (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dy**2 +
                    (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dx**2
                )

    # Actualizar la temperatura solo en las celdas válidas
    T_nuevo = T.copy()
    # Fusión
    mask_fusion = (T_nuevo > Tf) & (phi < 1)
    q_fusion = rho * cp * (T_nuevo[mask_fusion] - Tf)
    delta_phi_fusion = np.minimum(q_fusion / (rho * L), 1 - phi[mask_fusion])
    phi[mask_fusion] += delta_phi_fusion
    T_nuevo[mask_fusion] = Tf
    
    # Solidificación
    mask_solid = (T_nuevo < Tf) & (phi > 0)
    q_solid = -rho * cp * (T_nuevo[mask_solid] - Tf)
    delta_phi_solid = np.minimum(q_solid / (rho * L), phi[mask_solid])
    phi[mask_solid] -= delta_phi_solid
    T_nuevo[mask_solid] = Tf
    
    return T_nuevo

def aplicarCondicionesFrontera(T):
    #borde superior 21mm
    T[-1, :] = T[-2, :] # Superior

    #borde izquierdo parte de agua 1mm
    for n in range(110):
        if(n<=10): T[n, 0] = T[n, 1]
        else : T[n, 100] = T[n, 101] #borde izquierdo parte de agua 10mm

    #borde inferior parte de la T
    for n in range(10):
        T[0, 100+n] = T[1, 100+n]

    #borde derecho parte de agua 1mm
    for n in range(110):
        if(n<=10): T[n, -1] = T[n, -2]
        else: T[n, 110] = T[n, 109]

    return T

# phi inicial para calcular porcentaje de hielo
phi_inicial = inicializar_phi()

def porcentajeRestanteHielo(phi_actual):
    hielo_inicial = np.nansum(phi_inicial == 0)

    if(hielo_inicial == 0):
        return 0
    
    hielo_actual = np.nansum(phi_actual == 0)

    return (hielo_actual/hielo_inicial)*100

def main():
    T_main = definir_geometria(T)
    phi = phi_inicial
    phi_old = np.zeros_like(phi_inicial)
    tiempoDerretimiento = 0.0
    

    while(porcentajeRestanteHielo >= 50):
        T_old = T_main
        T_main = aplicarCondicionesFrontera(T_main)
        phi_old = phi
        phi = actualizarFraccionFase(T_main)
        T_main = actualizarTemperatura(T_main, T_old, phi, phi_old, dx, dy)
        porcentaje = porcentajeRestanteHielo(phi)
        tiempoDerretimiento += dt_estable

        print(f"Porcentaje de hielo restante: {porcentaje:.2f}%")
        tiempo_Derretimiento = tiempo_Derretimiento + dt_estable

    print(f"Tiempo en derretirse la mitad del hielo: {tiempo_Derretimiento}")

#ejecutar aplicacion
main()
