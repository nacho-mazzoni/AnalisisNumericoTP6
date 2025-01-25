import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 21e-3  # Total length in x-axis (21 mm)
Ly = 11e-3  # Total length in y-axis (11 mm)
Nx = 210    # Number of nodes in x
Ny = 110    # Number of nodes in y
dx, dy = Lx / Nx, Ly / Ny  # Cell size

# Physical parameters
alpha = 1.14e-6       # Thermal diffusivity (m^2/s)
L = 334000            # Latent heat of fusion (J/kg)
rho = 917             # Density of ice (kg/m³)
cp = 2100             # Specific heat of ice (J/kg·K)
Tf = 0.0              # Fusion temperature (°C)
Tliq = 0.05           # Liquidus temperature (°C)

# Define the T-shaped geometry
def define_geometry(T):
    T[:, :] = np.nan  # Initialize the entire domain as NaN

    # Convert physical dimensions to indices
    ice_width = int(1e-3 / dx)  # 1 mm
    ice_height = int(1e-3 / dy)  # 1 mm
    vertical_height = int(10e-3 / dy)  # 10 mm
    horizontal_width = int(10e-3 / dx)  # 10 mm

    # Reference coordinates
    center_x = Nx // 2
    total_height = Ny

    # Ice (central part of the T)
    T[total_height - ice_height:, center_x - ice_width//2:center_x + ice_width//2] = -10.0

    # Vertical part of water
    T[:total_height - ice_height, center_x - ice_width//2:center_x + ice_width//2] = 20.0

    # Horizontal parts of water (left and right of the ice)
    T[total_height - ice_height:, :center_x - ice_width//2] = 20.0
    T[total_height - ice_height:, center_x + ice_width//2:] = 20.0

    return T

# Initialize temperature and phase fraction
T_initial = define_geometry(np.full((Ny, Nx), 20.0))
phi = np.zeros_like(T_initial)
phi[T_initial < 0] = 0  # Ice
phi[T_initial >= 0] = 1  # Water

# Simulation loop
for t in range(100000):
    T_old = T_initial.copy()
    phi_old = phi.copy()

    # Update temperature
    T_initial = apply_boundary_conditions(T_initial)
    T_initial = update_temperature(T_initial, T_old, phi, phi_old, dx, dy)

    # Update phase fraction
    phi = update_phase_fraction(T_initial, phi)

    # Calculate the percentage of remaining ice
    ice_remaining = np.sum(T_initial < 0) / np.sum(T_initial < 0 for _initial in T_initial) * 100
    print(f"Time: {t}, Ice remaining: {ice_remaining:.2f}%")

    # Check if the ice has melted by 50%
    if ice_remaining <= 50:
        print(f"Time to reach 50% ice melt: {t} seconds")
        break

# Plot the final temperature field
plt.figure()
plt.imshow(T_initial, cmap='coolwarm')
plt.colorbar()
plt.title("Final Temperature Field")
plt.show()