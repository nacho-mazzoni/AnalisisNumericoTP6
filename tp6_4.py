import numpy as np
import matplotlib.pyplot as plt

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

# Inicialización del campo de temperatura
T = np.full((Ny, Nx), 20.0)  # Inicializar todo como agua (20°C)

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

# Definir la geometría de T_inicial
T_inicial = definir_geometria(T)

# Calcular un paso de tiempo estable basado en la condición CFL
cfl = 0.5  # Reducir el número CFL para mayor estabilidad
dt_estable = 0.01

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

phi = inicializar_phi()
phi_old = np.zeros_like(phi)

# Función para calcular el porcentaje de hielo restante
def porcentaje_hielo_restante(phi):
    # Máscara de hielo (phi == 0)
    hielo_inicial = np.sum(phi == 0)  # Área inicial de hielo
    
    # Si no hay hielo inicial, devolver 0%
    if hielo_inicial == 0:
        return 0.0
    
    # Máscara de hielo actual
    hielo_actual = np.sum(phi == 0)  # Área actual de hielo
    
    # Calcular el porcentaje
    porcentaje_restante = (hielo_actual / hielo_inicial) * 100
    
    return porcentaje_restante

# Función para actualizar la temperatura
def actualizar_temperatura(T, T_old, phi, phi_old, dt, dx, dy):
    """
    Actualiza la temperatura usando diferencias finitas, considerando el cambio de fase
    y condiciones de borde de aislamiento térmico.
    """
    # Crear máscara para celdas válidas
    mascara_valida = ~np.isnan(T)

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

    # Condiciones de frontera de aislamiento (Neumann: derivada normal = 0)
    # Borde superior
    lap_T[-1, :] = lap_T[-2, :]
    # Borde inferior
    lap_T[0, :] = lap_T[1, :]
    # Borde izquierdo
    lap_T[:, 0] = lap_T[:, 1]
    # Borde derecho
    lap_T[:, -1] = lap_T[:, -2]

    # Actualizar la temperatura solo en las celdas válidas
    T_nuevo = T.copy()
    
    # Aquí ajustamos la contribución de calor latente y la fracción de fase
    delta_T = dt * (alpha * lap_T[mascara_valida])

    # Considerar el cambio de fase de hielo a agua
    delta_T -= (L / (cp * rho)) * (phi[mascara_valida] - phi_old[mascara_valida])

    T_nuevo[mascara_valida] += delta_T

    return T_nuevo

# Función para actualizar la fracción de fase
def actualizar_fraccion_fase(T):
    """
    Actualiza la fracción de fase basándose en la temperatura.
    Si T <= Tf: hielo (phi = 0)
    Si T >= Tliq: agua (phi = 1)
    Si Tf < T < Tliq: mezcla (phi entre 0 y 1)
    """
    # Crear matriz phi con la misma forma que T
    phi_actualizado = np.full_like(T, np.nan)  # Inicializar con NaN
    
    # Máscara de celdas válidas (no NaN)
    mascara_valida = ~np.isnan(T)
    
    # Aplicar cálculo solo en celdas válidas
    T_validas = T[mascara_valida]
    phi_actualizado[mascara_valida] = np.where(
        T_validas <= Tf, 
        0, 
        np.where(T_validas >= Tliq, 
                 1, 
                 (T_validas - Tf) / (Tliq - Tf))
    )
    
    # Asegurar límites [0, 1]
    phi_actualizado = np.clip(phi_actualizado, 0, 1)

    return phi_actualizado

# Función principal
def main():
    # Inicialización de temperatura y fracción de fase
    T_bucle = definir_geometria(T)
    phi = inicializar_phi()  # Fracción de fase (0:hielo, 1:agua)
    phi_old = np.zeros((Ny, Nx))  # Inicializar phi_old con las mismas dimensiones
    tiempo_Derretimiento = 0.0  # Tiempo acumulado
    
    # Bucle temporal
    while porcentaje_hielo_restante(phi) >= 50:  # Bucle por porcentaje de hielo
        T_old = T_bucle.copy()
        phi_old = phi.copy()

        # Actualización de temperatura
        T_bucle = actualizar_temperatura(T_bucle, T_old, phi, phi_old, dt_estable, dx, dy)

        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T_bucle)

        # Calcular el porcentaje de hielo restante
        porcentaje = porcentaje_hielo_restante(phi)
        print(f"Porcentaje de hielo restante: {porcentaje:.2f}%")
        tiempo_Derretimiento += dt_estable

    print(f"Tiempo en derretirse la mitad del hielo: {tiempo_Derretimiento}")

# Ejecutar la simulación
main()