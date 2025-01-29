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

# Definir la geometría de T_inicial para porcentaje_hielo_restante
T_inicial = definir_geometria(T)

#Temperaturas caracteristicas
Tf = 0.0 #Temperatura de fusion (C)
Tliq = 0.05 #Temperatura de liquidus (C)

def actualizar_temperatura(T, T_old, phi, phi_old, dt, dx, dy):
    """
    Actualiza la temperatura usando diferencias finitas, considerando el cambio de fase
    y condiciones de borde de aislamiento térmico.
    """
    # Verificar dimensiones
    if phi_old.shape != T.shape or phi.shape != T.shape:
        raise ValueError(f"Las dimensiones de phi ({phi.shape}) o phi_old ({phi_old.shape}) no coinciden con T ({T.shape})")

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

    # Condiciones de frontera de aislamiento para los bordes superior e inferior
    lap_T[-1, :] = lap_T[-2, :]  # Borde superior
    lap_T[0, :] = lap_T[1, :]    # Borde inferior

    # Actualizar la temperatura solo en las celdas válidas
    T_nuevo = T.copy()
    
    # Aquí ajustamos la contribución de calor latente y la fracción de fase
    delta_T = dt * (alpha * lap_T[mascara_valida])

    # Considerar el cambio de fase de hielo a agua
    # Esto hace que cuando la temperatura esté cerca de la fusión, se consuma más energía para derretir el hielo
    delta_T -= (L / (cp * rho)) * (phi[mascara_valida] - phi_old[mascara_valida])

    T_nuevo[mascara_valida] += delta_T

    # Ajuste adicional en la interfaz hielo-agua
    for i in range(1, T.shape[0] - 1):
        for j in range(1, T.shape[1] - 1):
            if phi[i, j] < 1 and phi[i, j] > 0:  # En la interfaz
                T[i, j] += alpha * dt * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dy**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dx**2
                )

    return T_nuevo



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

def aplicar_condiciones_frontera(T):
    """Aplica las condiciones de frontera a la simulación considerando solo celdas válidas."""
    # Máscara de celdas válidas
    mascara_valida = ~np.isnan(T)

    # Coordenadas de la T
    Nx, Ny = T.shape
    ancho_hielo = np.sum(~np.isnan(T[-1, :]))  # Ancho del hielo
    altura_hielo = np.sum(~np.isnan(T[:, Nx // 2]))  # Altura del hielo
    centro_x = Nx // 2

    # 1. Borde superior (horizontal, hielo)
    indices = (slice(-1, None), slice(centro_x - ancho_hielo // 2, centro_x + ancho_hielo // 2))
    if np.all(mascara_valida[indices]):
        T[indices] = T[-2, centro_x - ancho_hielo // 2 : centro_x + ancho_hielo // 2]

    # 2. Borde inferior izquierdo (agua, horizontal)
    indices = (0, slice(None, centro_x - ancho_hielo // 2))
    if np.all(mascara_valida[indices]):
        T[0, : centro_x - ancho_hielo // 2] = T[1, : centro_x - ancho_hielo // 2]

    # 3. Borde inferior derecho (agua, horizontal)
    indices = (0, slice(centro_x + ancho_hielo // 2, None))
    if np.all(mascara_valida[indices]):
        T[0, centro_x + ancho_hielo // 2 :] = T[1, centro_x + ancho_hielo // 2 :]

    # 4. Borde izquierdo (agua, vertical)
    if np.all(mascara_valida[:, 0]):
        T[:, 0] = T[:, 1]

    # 5. Borde derecho (agua, vertical)
    if np.all(mascara_valida[:, -1]):
        T[:, -1] = T[:, -2]

    # Interfaces entre hielo y agua
    # Borde interior izquierdo (vertical)
    T[:, centro_x - ancho_hielo//2] = 0.5 * (T[:, centro_x - ancho_hielo//2 - 1] + T[:, centro_x - ancho_hielo//2 + 1])
    # Borde interior derecho (vertical)
    T[:, centro_x + ancho_hielo//2] = 0.5 * (T[:, centro_x + ancho_hielo//2 - 1] + T[:, centro_x + ancho_hielo//2 + 1])

    return T

def porcentaje_hielo_restante(T):
    
    # Máscara de hielo (T < 0°C)
    hielo_inicial = np.nansum(T_inicial < 0)  # Área inicial de hielo
    
    # Máscara de hielo actual
    hielo_actual = np.nansum(T < 0)  # Área actual de hielo
    
    # Calcular el porcentaje
    porcentaje_restante = (hielo_actual / hielo_inicial) * 100
    
    return porcentaje_restante 

def registrar_temperatura_manual(T):
    """
    Imprime las temperaturas en puntos específicos del dominio.

    Parámetros:
        T (numpy.ndarray): Matriz de temperaturas actual.
    """
    # Convertir dimensiones físicas a índices
    ancho_hielo = int(1e-3 / dx)  # 1 mm
    altura_hielo = int(1e-3 / dy)  # 1 mm
    centro_x = Nx // 2
    altura_total = Ny
    # Definir manualmente las coordenadas de los puntos de interés
    puntos = {
        "Centro del hielo": (centro_x, (altura_hielo*0.5)),  # Aproximadamente en el centro del hielo
        "Intersección izquierda (hielo-agua)": (centro_x-(0.5*ancho_hielo), (altura_hielo*0.5)),  # En la frontera izquierda del hielo
        "Intersección derecha (hielo-agua)": (centro_x+(0.5*ancho_hielo), (altura_hielo*0.5)),  # En la frontera derecha del hielo
        "Centro del agua arriba (sobre el hielo)": (centro_x, (altura_total*0.5)),  # En el agua, sobre el hielo
    }

    print("Temperaturas en puntos de interés:")
    for nombre, (x, y) in puntos.items():
        try:
            temp = T[y, x]  # Acceder a la matriz con coordenadas (y, x)
            print(f"{nombre}: {temp:.2f}°C")
        except IndexError:
            print(f"{nombre}: Coordenadas ({x}, {y}) están fuera del rango del dominio.")



def main():
    # Inicialización de temperatura y fracción de fase
    T_bucle = definir_geometria(T)
    phi = inicializar_phi()  # Fracción de fase (0:hielo, 1:agua)
    phi_old = np.zeros((Ny, Nx))  # Inicializar phi_old con las mismas dimensiones
    #definir variable tiempo en segundos
    tiempo_Derretimiento = 0.0
    
    # Bucle temporal
    while(porcentaje_hielo_restante(T_bucle) >= 50):  # Bucle por porcentaje de hielo
        T_old = T_bucle.copy()
        phi_old = phi.copy()
        # Aplicar condiciones de frontera
        T_bucle = aplicar_condiciones_frontera(T_bucle)

        # Actualización de temperatura
        T_bucle = actualizar_temperatura(T_bucle, T_old, phi, phi_old, dt_estable, dx, dy)

        # Actualización de fracción de fase
        phi = actualizar_fraccion_fase(T_bucle)

        # Registrar y mostrar temperaturas con nombres
        registrar_temperatura_manual(T_bucle)
        
        # Calcular el porcentaje de hielo restante
        porcentaje = porcentaje_hielo_restante(T_bucle)
        print(f"Porcentaje de hielo restante: {porcentaje:.2f}%")
        tiempo_Derretimiento = tiempo_Derretimiento + dt_estable

    print(f"Tiempo en derretirse la mitad del hielo: {tiempo_Derretimiento}")


# Ejecutar la simulación
main()
