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
cfl = 0.5  # Número CFL (típicamente entre 0.5 y 1.0)
dt_estable = cfl * min(dx**2 / (2 * alpha), dy**2 / (2 * alpha))

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

# Resto del código sin cambios...

# Definir la geometría de T_inicial para porcentaje_hielo_restante
T_inicial = definir_geometria(T)

#Temperaturas caracteristicas
Tf = 0.0 #Temperatura de fusion (C)
Tliq = 0.05 #Temperatura de liquidus (C)


def actualizar_temperatura(T, T_old, phi, phi_old, dt, dx, dy):
    print(f"T: {T.shape}, T_old: {T_old.shape}, phi: {phi.shape}, phi_old: {phi_old.shape}")
    # Verifica si phi_old tiene la forma correcta
    if phi_old.shape != T.shape:
        raise ValueError(f"Las dimensiones de phi_old {phi_old.shape} no coinciden con T {T.shape}")

    # Crear máscara para celdas válidas
    mascara_valida = ~np.isnan(T)

    # Calcular el Laplaciano solo para celdas válidas
    lap_T = np.zeros_like(T)
    lap_T[mascara_valida] = (
        (np.roll(T_old, -1, axis=0) - 2 * T_old + np.roll(T_old, 1, axis=0)) / dy**2 +
        (np.roll(T_old, -1, axis=1) - 2 * T_old + np.roll(T_old, 1, axis=1)) / dx**2
    )[mascara_valida]

    # Calcular el cambio de fracción de fase (dphi_dt) solo para celdas válidas
    dphi_dt = np.zeros_like(phi)
    dphi_dt[mascara_valida] = np.where(
        (Tf <= T_old[mascara_valida]) & (T_old[mascara_valida] <= Tliq),
        (T_old[mascara_valida] - Tf) / (Tliq - Tf) - phi_old[mascara_valida],
        0
    )

    # Actualizar la temperatura solo en las celdas válidas
    T_nuevo = T.copy()
    T_nuevo[mascara_valida] += dt * (
        alpha * lap_T[mascara_valida] -
        (L / (cp * rho)) * dphi_dt[mascara_valida] / dt
    )

    return T_nuevo



def actualizar_fraccion_fase(T, phi):
    """
    Actualiza la fracción de fase basándose en la temperatura.
    Si T <= Tf: hielo (phi = 0)
    Si T >= Tliq: agua (phi = 1)
    Si Tf < T < Tliq: mezcla (phi entre 0 y 1)
    """
    T_nuevo = T.copy()
    
    # Crear matriz phi con la misma forma que T
    phi_actualizado = np.zeros_like(T_nuevo)

    # Actualizar valores de phi basados en las condiciones
    phi_actualizado = np.where(T_nuevo <= Tf, 0, 
                       np.where(T_nuevo >= Tliq, 1, 
                                (T_nuevo - Tf) / (Tliq - Tf)))
    
    # Asegurar límites [0, 1]
    phi_actualizado = np.clip(phi_actualizado, 0, 1)

    return phi_actualizado

def aplicar_condiciones_frontera(T):
    """Aplica las condiciones de frontera a la simulación."""
    # Máscara de celdas válidas
    mascara_valida = ~np.isnan(T)

    # Coordenadas de la T
    Nx, Ny = T.shape
    ancho_hielo = np.sum(~np.isnan(T[-1, :]))  # Ancho del hielo
    altura_hielo = np.sum(~np.isnan(T[:, Nx // 2]))  # Altura del hielo
    centro_x = Nx // 2

    # 1. Borde superior (horizontal, hielo)
    T[-1, centro_x - ancho_hielo // 2 : centro_x + ancho_hielo // 2] = T[-2, centro_x - ancho_hielo // 2 : centro_x + ancho_hielo // 2]

    # 2. Borde inferior izquierdo (agua, horizontal)
    T[0, : centro_x - ancho_hielo // 2] = T[1, : centro_x - ancho_hielo // 2]

    # 3. Borde inferior derecho (agua, horizontal)
    T[0, centro_x + ancho_hielo // 2 :] = T[1, centro_x + ancho_hielo // 2 :]

    # 4. Borde izquierdo (agua, vertical)
    T[:, 0] = T[:, 1]

    # 5. Borde derecho (agua, vertical)
    T[:, -1] = T[:, -2]

    # 6. Borde interior (entre hielo y agua, vertical)
    T[:, centro_x - ancho_hielo // 2] = T[:, centro_x - ancho_hielo // 2 - 1]
    T[:, centro_x + ancho_hielo // 2 - 1] = T[:, centro_x + ancho_hielo // 2]

    return T


def porcentaje_hielo_restante(T):
    
    # Máscara de hielo (T < 0°C)
    hielo_inicial = np.sum(T_inicial < 0)  # Área inicial de hielo
    
    # Máscara de hielo actual
    hielo_actual = np.sum(T < 0)  # Área actual de hielo
    
    # Calcular el porcentaje
    porcentaje_restante = (hielo_actual / hielo_inicial) * 100
    
    return porcentaje_restante

def visualizar(T_bucle,img,step,porcentaje):
    """Actualiza la visualización en tiempo real."""
    img.set_data(T_bucle)
    
    plt.title(f"Paso: {step}, Hielo restante: {porcentaje:.2f}%")
    plt.pause(0.01)

def main():
    # Inicialización de temperatura y fracción de fase
    T_bucle = definir_geometria(T)
    phi = inicializar_phi()  # Fracción de fase (0:hielo, 1:agua)
    phi_old = np.zeros((Ny, Nx))  # Inicializar phi_old con las mismas dimensiones

    # Preparar la visualización
    plt.ion()  # Modo interactivo
    fig, ax = plt.subplots()
    img = ax.imshow(T_bucle, cmap='coolwarm', interpolation='nearest', animated=True)
    plt.colorbar(img)

    # Bucle temporal
    for step in range(100000):  # Bucle por pasos temporales
        T_old = T_bucle.copy()
        phi_old = phi.copy()

        # Aplicar condiciones de frontera
        T_bucle = aplicar_condiciones_frontera(T_bucle)

        # Actualización de temperatura
        T_bucle = actualizar_temperatura(T_bucle, T_old, phi, phi_old, dt_estable, dx, dy)

        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T_bucle, phi)

        # Calcular el porcentaje de hielo restante
        porcentaje = porcentaje_hielo_restante(T_bucle)
        print(f"Paso {step}, Porcentaje de hielo restante: {porcentaje:.2f}%")

        # Actualizar visualización cada 100 pasos
        if step % 100 == 0:
            visualizar(T_bucle,img,step,porcentaje)
            
            

        # Detener simulación si todo el hielo se ha derretido
        if porcentaje <= 0:
            print("Todo el hielo se ha derretido. Simulación completa.")
            break

    plt.ioff()  # Desactiva el modo interactivo
    plt.show()

# Ejecutar la simulación
main()
