import numpy as np
import matplotlib.pyplot as plt

# Parámetros del dominio
Lx = 21e-3  # Longitud total en el eje x (21 mm)
Ly = 11e-3  # Longitud total en el eje y (11 mm)
Nx = 210    # Número de nodos en x
Ny = 110    # Número de nodos en y
dx, dy = Lx / Nx, Ly / Ny  # Tamaño de celda

# Inicialización del campo de temperatura
T = np.full((Ny, Nx), 20.0)  # Inicializar todo como agua (20°C)

# Función para definir la geometría en forma de T invertida
def definir_geometria(T):
    """
    Define la geometría en forma de T invertida en el dominio.
    """
    T[:, :] = np.nan  # Inicializar todo fuera del dominio como NaN

    # Convertir dimensiones físicas a índices
    ancho_hielo = int(1e-3 / dx)  # 1 mm
    altura_hielo = int(1e-3 / dy)  # 1 mm
    altura_vertical = int(10e-3 / dy)  # 10 mm
    ancho_horizontal = int(10e-3 / dx)  # 10 mm

    # Coordenadas de referencia
    centro_x = Nx // 2
    altura_total = Ny

    # Hielo (parte central de la T)
    T[altura_total - altura_hielo : , int(centro_x - ancho_hielo/2):  int(centro_x + ancho_hielo/2)] = -10.0

    # Parte vertical del agua
    T[ : altura_total - altura_hielo, int(centro_x - ancho_hielo/2):  int(centro_x + ancho_hielo/2)] = 20.0

    # Partes horizontales de agua (a la izquierda y derecha del hielo)
    T[altura_total - altura_hielo : ,  : int(centro_x - ancho_hielo/2)] = 20.0
    T[altura_total - altura_hielo : , int(centro_x + ancho_hielo/2) : ] = 20.0

    return T

# Definir la geometría
T = definir_geometria(T)

# Graficar la distribución inicial de temperatura
plt.figure(figsize=(8, 6))
plt.imshow(T, extent=[0, Lx * 1e3, 0, Ly * 1e3], origin="lower", cmap="coolwarm")
plt.colorbar(label="Temperatura (°C)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Distribución inicial de temperatura")
plt.grid()
plt.show()
