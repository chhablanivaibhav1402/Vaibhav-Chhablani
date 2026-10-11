import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Grid
# ---------------------------------------------------------

Lx, Ly = 8.0, 4.0

nx = 160          # nodes along the length
ny_cold = 20      # nodes in cold brine channel
ny_hot = 20       # nodes in hot brine channel
ny_wall = 6       # nodes in copper wall

ny = ny_cold + ny_wall + ny_hot

# Materials
cold_fluid = "Brine"
hot_fluid = "Brine"
wall = "Copper"

if nx < 10 or min(ny_cold, ny_hot) < 3 or ny_wall < 2:
    raise ValueError(
        "Grid is too coarse. Use nx >= 10, "
        "ny_cold/ny_hot >= 3 and ny_wall >= 2."
    )

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

# 0 = cold brine, 1 = copper wall, 2 = hot brine
zone = np.zeros((ny, nx), dtype=int)
zone[ny_cold:ny_cold + ny_wall, :] = 1
zone[ny_cold + ny_wall:, :] = 2

print(f"Grid = {nx} x {ny} = {nx * ny} nodes")
print(f"dx = {dx:.4f}, dy = {dy:.4f}")
print(f"Cold fluid = {cold_fluid}")
print(f"Hot fluid = {hot_fluid}")
print(f"Wall = {wall}")


# ---------------------------------------------------------
# 2. Flow and heat transfer settings
# ---------------------------------------------------------

Re_hot = 700.0
Re_cold = 350.0

Pr = 7.0

# Copper conducts heat much better than brine
k_ratio = 20.0

# True means hot and cold brine flow in opposite directions
counter_flow = True

T_hot_in = 1.0
T_cold_in = -5.0

U_hot = 6.0
U_cold = 1.0

max_iter = 15000
tol = 1e-8
dt_factor = 0.4


# ---------------------------------------------------------
# 3. Velocity profile
# ---------------------------------------------------------

def velocity_profile(ny_cold, ny_wall, ny_hot,
                     nx, U_cold, U_hot, counter_flow):

    ny = ny_cold + ny_wall + ny_hot

    u = np.zeros((ny, nx))

    # Position across each channel
    eta_cold = np.linspace(0, 1, ny_cold)
    eta_hot = np.linspace(0, 1, ny_hot)

    # Simple parabolic velocity profile
    profile_cold = 6 * eta_cold * (1 - eta_cold)
    profile_hot = 6 * eta_hot * (1 - eta_hot)

    # Cold brine flows opposite to hot brine
    cold_sign = -1.0 if counter_flow else 1.0

    u[:ny_cold, :] = (
        cold_sign * U_cold * profile_cold[:, None]
    )

    # Copper wall has zero velocity
    u[ny_cold + ny_wall:, :] = (
        U_hot * profile_hot[:, None]
    )

    return u


u = velocity_profile(
    ny_cold,
    ny_wall,
    ny_hot,
    nx,
    U_cold,
    U_hot,
    counter_flow
)


# ---------------------------------------------------------
# 4. Thermal diffusivity
# ---------------------------------------------------------

D_cold = 1 / (Re_cold * Pr)
D_hot = 1 / (Re_hot * Pr)

# Give the copper wall higher thermal conductivity
D_wall = k_ratio * 0.5 * (D_hot + D_cold)

D = np.where(
    zone == 0,
    D_cold,
    np.where(zone == 1, D_wall, D_hot)
)

print(f"D_cold = {D_cold:.5f}")
print(f"D_hot  = {D_hot:.5f}")
print(f"D_wall = {D_wall:.5f}")


# ---------------------------------------------------------
# 5. Check velocity and geometry
# ---------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))

ax[0].plot(u[:, 0], y)
ax[0].axhline(y[ny_cold], color='k', lw=0.8)
ax[0].axhline(y[ny_cold + ny_wall], color='k', lw=0.8)

ax[0].set_xlabel("Velocity")
ax[0].set_ylabel("y")
ax[0].set_title("Velocity profile")
ax[0].grid(True)

ax[1].contourf(
    X, Y, zone,
    levels=[-0.5, 0.5, 1.5, 2.5],
    colors=['#5da5da', '#b0b0b0', '#f17cb0']
)

ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Cold brine / Copper / Hot brine")
ax[1].set_aspect("equal")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 6. Boundary conditions
# ---------------------------------------------------------

def apply_bc(T):

    # Hot brine enters from the left
    T[ny_cold + ny_wall:, 0] = T_hot_in

    # Hot brine outlet
    T[ny_cold + ny_wall:, -1] = \
        T[ny_cold + ny_wall:, -2]

    if counter_flow:

        # Cold brine enters from the right
        T[:ny_cold, -1] = T_cold_in

        # Cold brine leaves from the left
        T[:ny_cold, 0] = T[:ny_cold, 1]

    else:

        # Parallel flow
        T[:ny_cold, 0] = T_cold_in
        T[:ny_cold, -1] = T[:ny_cold, -2]

    # No heat loss from the ends of the copper wall
    T[ny_cold:ny_cold + ny_wall, 0] = \
        T[ny_cold:ny_cold + ny_wall, 1]

    T[ny_cold:ny_cold + ny_wall, -1] = \
        T[ny_cold:ny_cold + ny_wall, -2]

    # Outer surfaces are insulated
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]

    return T


# ---------------------------------------------------------
# 7. One FDM step
# ---------------------------------------------------------

def fdm_step(T, u, D, dx, dy, dt):

    Tc = T[1:-1, 1:-1]

    Te = T[1:-1, 2:]
    Tw = T[1:-1, :-2]

    Tn = T[2:, 1:-1]
    Ts = T[:-2, 1:-1]

    Dc = D[1:-1, 1:-1]

    # Harmonic mean at material boundaries
    De = 2 * Dc * D[1:-1, 2:] / (
        Dc + D[1:-1, 2:]
    )

    Dw = 2 * Dc * D[1:-1, :-2] / (
        Dc + D[1:-1, :-2]
    )

    Dn = 2 * Dc * D[2:, 1:-1] / (
        Dc + D[2:, 1:-1]
    )

    Ds = 2 * Dc * D[:-2, 1:-1] / (
        Dc + D[:-2, 1:-1]
    )

    # Heat diffusion
    diffusion_x = (
        De * (Te - Tc) -
        Dw * (Tc - Tw)
    ) / dx**2

    diffusion_y = (
        Dn * (Tn - Tc) -
        Ds * (Tc - Ts)
    ) / dy**2

    # Heat convection
    uc = u[1:-1, 1:-1]

    dTdx = np.where(
        uc >= 0,
        (Tc - Tw) / dx,
        (Te - Tc) / dx
    )

    T_new = T.copy()

    T_new[1:-1, 1:-1] = (
        Tc
        + dt * (
            diffusion_x
            + diffusion_y
            - uc * dTdx
        )
    )

    return T_new


# ---------------------------------------------------------
# 8. Stable time step
# ---------------------------------------------------------

def get_dt(D, u, dx, dy):

    # Diffusion limit
    dt_diff = 1 / (
        2 * D.max() *
        (1 / dx**2 + 1 / dy**2)
    )

    # CFL limit
    u_max = np.abs(u).max()

    if u_max > 0:
        dt_cfl = dx / u_max
    else:
        dt_cfl = np.inf

    return dt_factor * min(dt_diff, dt_cfl)


# ---------------------------------------------------------
# 9. Initial temperature
# ---------------------------------------------------------

T = np.full(
    (ny, nx),
    0.5 * (T_hot_in + T_cold_in)
)

# Cold brine starts cold
T[:ny_cold, :] = T_cold_in

# Hot brine starts hot
T[ny_cold + ny_wall:, :] = T_hot_in

# Initial temperature through copper wall
T[ny_cold:ny_cold + ny_wall, :] = np.linspace(
    T_cold_in,
    T_hot_in,
    ny_wall
)[:, None]

T = apply_bc(T)


# ---------------------------------------------------------
# 10. Solve
# ---------------------------------------------------------

dt = get_dt(D, u, dx, dy)

residuals = []

for it in range(1, max_iter + 1):

    T_old = T.copy()

    T = fdm_step(
        T, u, D, dx, dy, dt
    )

    T = apply_bc(T)

    residual = np.max(
        np.abs(T - T_old)
    )

    residuals.append(residual)

    if not np.all(np.isfinite(T)):
        raise FloatingPointError(
            f"Solution diverged at iteration {it}"
        )

    if it == 1 or it % 2000 == 0:
        print(
            f"iteration {it:6d}   "
            f"residual = {residual:.3e}"
        )

    if residual < tol and it > 50:
        break

print(
    f"\nStopped after {it} iterations"
)
print(
    f"dt = {dt:.5f}"
)
print(
    f"Final residual = {residual:.3e}"
)


# ---------------------------------------------------------
# 11. Bulk temperature
# ---------------------------------------------------------

def bulk_temp(T_slice, u_slice):

    # Velocity-weighted average temperature
    weights = np.abs(u_slice) + 1e-12

    return np.average(
        T_slice,
        axis=0,
        weights=weights
    )


hot_rows = slice(
    ny_cold + ny_wall,
    ny
)

cold_in_col, cold_out_col = (
    (-1, 0)
    if counter_flow
    else (0, -1)
)

hot_bulk = bulk_temp(
    T[hot_rows, :],
    u[hot_rows, :]
)

cold_bulk = bulk_temp(
    T[:ny_cold, :],
    u[:ny_cold, :]
)


# ---------------------------------------------------------
# 12. Temperature plots
# ---------------------------------------------------------

fig, ax = plt.subplots(
    2, 1,
    figsize=(10, 6),
    height_ratios=[2, 1.3]
)

cf = ax[0].contourf(
    X, Y, T,
    levels=30,
    cmap='inferno'
)

ax[0].contour(
    X, Y, zone,
    levels=[0.5, 1.5],
    colors='cyan',
    linewidths=0.8
)

plt.colorbar(
    cf,
    ax=ax[0],
    label='Temperature'
)

ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title(
    "Temperature field"
)

ax[0].set_aspect("equal")


ax[1].plot(
    x,
    hot_bulk,
    label="Hot brine"
)

ax[1].plot(
    x,
    cold_bulk,
    label="Cold brine"
)

ax[1].set_xlabel("x")
ax[1].set_ylabel("Temperature")

ax[1].set_title(
    "Bulk temperature along the heat exchanger"
)

ax[1].legend()
ax[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 13. Heat transfer calculation
# ---------------------------------------------------------

hot_in_T = float(
    bulk_temp(
        T[hot_rows, 0],
        u[hot_rows, 0]
    )
)

hot_out_T = float(
    bulk_temp(
        T[hot_rows, -1],
        u[hot_rows, -1]
    )
)

cold_in_T = float(
    bulk_temp(
        T[:ny_cold, cold_in_col],
        u[:ny_cold, cold_in_col]
    )
)

cold_out_T = float(
    bulk_temp(
        T[:ny_cold, cold_out_col],
        u[:ny_cold, cold_out_col]
    )
)


# Simple trapezoidal integration
def integrate(values, coordinates):

    values = np.asarray(values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)

    return float(
        np.sum(
            0.5 *
            (values[1:] + values[:-1]) *
            np.diff(coordinates)
        )
    )


mdot_hot = integrate(
    np.maximum(u[hot_rows, 0], 0),
    y[hot_rows]
)

mdot_cold = integrate(
    np.abs(u[:ny_cold, cold_in_col]),
    y[:ny_cold]
)


# Heat lost by hot brine
q_hot = mdot_hot * (
    hot_in_T - hot_out_T
)

# Heat gained by cold brine
q_cold = mdot_cold * (
    cold_out_T - cold_in_T
)

imbalance = abs(
    q_hot - q_cold
) / max(
    abs(q_hot),
    abs(q_cold),
    1e-30
)


# Maximum possible heat transfer
q_max = min(
    mdot_hot,
    mdot_cold
) * (
    T_hot_in - T_cold_in
)

effectiveness = (
    q_cold / max(q_max, 1e-30)
)


print("\n--- Heat exchanger results ---")

print(
    f"Hot brine   : {hot_in_T:.4f} -> {hot_out_T:.4f}"
)

print(
    f"Cold brine  : {cold_in_T:.4f} -> {cold_out_T:.4f}"
)

print(
    f"Heat lost by hot brine  = {q_hot:.5f}"
)

print(
    f"Heat gained by cold brine = {q_cold:.5f}"
)

print(
    f"Energy imbalance = {imbalance:.2e}"
)

print(
    f"Effectiveness = {effectiveness:.4f}"
)


# ---------------------------------------------------------
# 14. Save result for ParaView
# ---------------------------------------------------------

def write_vtk(path, x, y, T, zone):

    nx = len(x)
    ny = len(y)

with open(path, 'w') as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Brine-Copper heat exchanger\nASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")
    f.write(f"DIMENSIONS {nx} {ny} 1\n")
    f.write(f"POINTS {nx * ny} float\n")

    for j in range(ny):
        for i in range(nx):
            f.write(f"{x[i]:.6e} {y[j]:.6e} 0\n")

    f.write(f"POINT_DATA {nx * ny}\n")
    f.write("SCALARS temperature float 1\nLOOKUP_TABLE default\n")

    for j in range(ny):
        for i in range(nx):
            f.write(f"{T[j, i]:.6e}\n")

    f.write("SCALARS zone float 1\nLOOKUP_TABLE default\n")

    for j in range(ny):
        for i in range(nx):
            f.write(f"{zone[j, i]:.1f}\n")


write_vtk("double_pipe_cht.vtk", x, y, T, zone)
print("Saved: double_pipe_cht.vtk")
