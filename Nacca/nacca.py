import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Grid and flow settings
NX, NY = 120, 60
RE = 100
ALPHA_DEG = 10
ITERATIONS = 300
POISSON_ITERATIONS = 80

LX, LY = 5.0, 2.5
U_INF = 1.0
CHORD = 1.0
X_LE = 1.35

OUTPUT_DIR = Path("airfoil_result")


def naca0012_half_thickness(x):
    # NACA 0012 half-thickness
    xc = np.clip(x, 0.0, 1.0)

    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1036 * xc**4
    )

    return np.where((x >= 0.0) & (x <= 1.0), yt, -1.0)


def airfoil_outline(alpha_deg, n=401):
    # Create the airfoil outline
    beta = np.linspace(0.0, math.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = naca0012_half_thickness(xc)

    xb = np.concatenate((xc[::-1], xc[1:])) * CHORD
    yb = np.concatenate((-yt[::-1], yt[1:])) * CHORD

    angle = math.radians(alpha_deg)
    xq = X_LE + 0.25 * CHORD

    xr = xb - 0.25 * CHORD

    x = xq + math.cos(angle) * xr - math.sin(angle) * yb
    y = math.sin(angle) * xr + math.cos(angle) * yb

    return x, y


def airfoil_mask(x, y, alpha_deg):
    # True means inside the airfoil
    angle = math.radians(alpha_deg)
    xq = X_LE + 0.25 * CHORD

    xr = (
        math.cos(angle) * (x - xq)
        + math.sin(angle) * y
        + 0.25 * CHORD
    )

    yr = (
        -math.sin(angle) * (x - xq)
        + math.cos(angle) * y
    )

    yt = naca0012_half_thickness(xr / CHORD) * CHORD

    return (yt >= 0.0) & (np.abs(yr) <= yt)
def find_ghost_nodes(solid):
    # Ghost nodes are solid nodes next to the fluid
    ghost = np.zeros_like(solid, dtype=bool)

    ghost[1:, :] |= solid[1:, :] & ~solid[:-1, :]
    ghost[:-1, :] |= solid[:-1, :] & ~solid[1:, :]
    ghost[:, 1:] |= solid[:, 1:] & ~solid[:, :-1]
    ghost[:, :-1] |= solid[:, :-1] & ~solid[:, 1:]

    return ghost


def apply_ghost_values(field, solid, ghost, wall_value=0.0,
                       no_slip=False):
    # Use the closest fluid node for each ghost node
    old = field.copy()
    ny, nx = field.shape

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            if not ghost[j, i]:
                continue

            neighbors = [
                (j, i - 1),
                (j, i + 1),
                (j - 1, i),
                (j + 1, i)
            ]

            for jj, ii in neighbors:
                if not solid[jj, ii]:
                    if no_slip:
                        # Approximate a stationary wall halfway
                        field[j, i] = 2.0 * wall_value - old[jj, ii]
                    else:
                        # Zero-gradient condition
                        field[j, i] = old[jj, ii]
                    break

    # Keep deeper solid nodes at the wall value
    inside = solid & ~ghost
    field[inside] = wall_value

    return field


def ddx_central(f, dx):
    # Central difference in x
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (
        f[1:-1, 2:] - f[1:-1, :-2]
    ) / (2.0 * dx)

    return out


def ddy_central(f, dy):
    # Central difference in y
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (
        f[2:, 1:-1] - f[:-2, 1:-1]
    ) / (2.0 * dy)

    return out


def laplacian(f, dx, dy):
    # Second derivatives in x and y
    out = np.zeros_like(f)

    out[1:-1, 1:-1] = (
        (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1]
         + f[1:-1, :-2]) / dx**2
        +
        (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1]
         + f[:-2, 1:-1]) / dy**2
    )

    return out


def ddx_upwind(f, u, dx):
    # Upwind difference follows the flow direction
    out = np.zeros_like(f)

    out[1:-1, 1:-1] = np.where(
        u[1:-1, 1:-1] >= 0.0,
        (f[1:-1, 1:-1] - f[1:-1, :-2]) / dx,
        (f[1:-1, 2:] - f[1:-1, 1:-1]) / dx
    )

    return out


def ddy_upwind(f, v, dy):
    # Upwind difference follows the flow direction
    out = np.zeros_like(f)

    out[1:-1, 1:-1] = np.where(
        v[1:-1, 1:-1] >= 0.0,
        (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dy,
        (f[2:, 1:-1] - f[1:-1, 1:-1]) / dy
    )

    return out
def apply_bc(u, v, p, solid, ghost):
    # Inlet
    u[:, 0] = U_INF
    v[:, 0] = 0.0

    # Outlet: zero-gradient velocity
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]

    # Top and bottom: free-stream flow
    u[0, :] = U_INF
    u[-1, :] = U_INF
    v[0, :] = 0.0
    v[-1, :] = 0.0

    # Pressure boundaries
    p[:, 0] = p[:, 1]
    p[:, -1] = 0.0
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

    # Apply wall ghost values to velocity and pressure
    apply_ghost_values(u, solid, ghost, no_slip=True)
    apply_ghost_values(v, solid, ghost, no_slip=True)
    apply_ghost_values(p, solid, ghost, no_slip=False)


def poisson_jacobi(rhs, dx, dy, sweeps, solid, ghost):
    # Solve Laplacian(phi) = rhs
    phi = np.zeros_like(rhs)

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    denom = 2.0 * (inv_dx2 + inv_dy2)

    fluid = ~solid

    for _ in range(sweeps):
        old = phi.copy()

        new_phi = (
            (old[1:-1, 2:] + old[1:-1, :-2]) * inv_dx2
            + (old[2:, 1:-1] + old[:-2, 1:-1]) * inv_dy2
            - rhs[1:-1, 1:-1]
        ) / denom

        # Update only fluid nodes
        region = fluid[1:-1, 1:-1]
        phi[1:-1, 1:-1][region] = new_phi[region]

        # Apply pressure boundary conditions
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = 0.0
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]

        apply_ghost_values(phi, solid, ghost, no_slip=False)

    return phi
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build the grid
    x1 = np.linspace(0.0, LX, NX)
    y1 = np.linspace(-0.5 * LY, 0.5 * LY, NY)
    x, y = np.meshgrid(x1, y1)

    dx = x1[1] - x1[0]
    dy = y1[1] - y1[0]

    nu = U_INF * CHORD / RE

    # Airfoil and ghost nodes
    solid = airfoil_mask(x, y, ALPHA_DEG)
    ghost = find_ghost_nodes(solid)

    # Initial conditions
    u = np.full((NY, NX), U_INF, dtype=float)
    v = np.zeros_like(u)
    p = np.zeros_like(u)

    apply_bc(u, v, p, solid, ghost)

    # Stable time step
    dt = min(
        0.20 * min(dx, dy) / U_INF,
        0.15 * min(dx**2, dy**2) / nu
    )

    history = []

    print("-" * 60)
    print(f"Grid: {NX} x {NY}")
    print(f"Reynolds number: {RE}")
    print(f"Angle of attack: {ALPHA_DEG} degrees")
    print(f"Time step: {dt:.5e}")
    print("-" * 60)

    for step in range(1, ITERATIONS + 1):

        # Step 1: Predictor velocity
        u_old = u.copy()
        v_old = v.copy()

        du = (
            -u_old * ddx_upwind(u_old, u_old, dx)
            -v_old * ddy_upwind(u_old, v_old, dy)
            +nu * laplacian(u_old, dx, dy)
        )

        dv = (
            -u_old * ddx_upwind(v_old, u_old, dx)
            -v_old * ddy_upwind(v_old, v_old, dy)
            +nu * laplacian(v_old, dx, dy)
        )

        us = u_old + dt * du
        vs = v_old + dt * dv

        apply_bc(us, vs, p, solid, ghost)

        # Step 2: Pressure correction
        rhs = (
            ddx_central(us, dx)
            + ddy_central(vs, dy)
        ) / dt

        rhs[solid] = 0.0

        phi = poisson_jacobi(
            rhs, dx, dy, POISSON_ITERATIONS, solid, ghost
        )

        # Step 3: Correct velocity
        u = us - dt * ddx_central(phi, dx)
        v = vs - dt * ddy_central(phi, dy)

        apply_bc(u, v, p, solid, ghost)

        # Update pressure
        p += phi
        p[solid & ~ghost] = 0.0

        # Reapply boundary conditions
        apply_bc(u, v, p, solid, ghost)

        # Measure continuity in the fluid
        div = ddx_central(u, dx) + ddy_central(v, dy)
        fluid = ~solid

        continuity = float(
            np.sqrt(np.mean(div[fluid] ** 2))
        )

        speed = np.sqrt(u**2 + v**2)
        max_speed = float(np.max(speed[fluid]))

        history.append([step, step * dt, continuity, max_speed])

        # Stop if the solution becomes invalid
        if not (
            np.all(np.isfinite(u))
            and np.all(np.isfinite(v))
            and np.all(np.isfinite(p))
        ):
            raise FloatingPointError(
                f"Invalid value detected at iteration {step}"
            )

        if max_speed > 25.0:
            raise FloatingPointError(
                f"Solution diverged at iteration {step}"
            )

        if (
            step == 1
            or step == ITERATIONS
            or step % max(1, ITERATIONS // 5) == 0
        ):
            print(
                f"Iteration {step:4d} | "
                f"Continuity = {continuity:.4e} | "
                f"Max speed = {max_speed:.4f}"
            )

    # Final flow fields
    speed = np.sqrt(u**2 + v**2)
    vorticity = ddx_central(v, dx) - ddy_central(u, dy)

    # Save iteration history
    with open(
        OUTPUT_DIR / "history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "iteration", "pseudo_time", "continuity", "max_speed"
        ])
        writer.writerows(history)

    # Save numerical fields
    np.savez(
        OUTPUT_DIR / "airfoil_fields.npz",
        x=x, y=y, u=u, v=v, p=p,
        speed=speed, vorticity=vorticity,
        solid=solid, ghost=ghost
    )

    # Airfoil outline
    wall_x, wall_y = airfoil_outline(ALPHA_DEG)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].semilogy(
        np.array(history)[:, 0],
        np.maximum(np.array(history)[:, 2], 1e-14)
    )
    axes[0].set_title("Continuity")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("L2 divergence")
    axes[0].grid(True)

    plot = axes[1].contourf(
        x, y, speed, levels=30
    )
    axes[1].plot(wall_x, wall_y, "k-", linewidth=1.2)
    fig.colorbar(plot, ax=axes[1], label="Speed")
    axes[1].set_title("Velocity magnitude")

    plot = axes[2].contourf(
        x, y, vorticity, levels=30, cmap="RdBu_r"
    )
    axes[2].plot(wall_x, wall_y, "k-", linewidth=1.2)
    fig.colorbar(plot, ax=axes[2], label="Vorticity")
    axes[2].set_title("Vorticity")

    for ax in axes[1:]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.5, 4.0)
        ax.set_ylim(-1.0, 1.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "results.png", dpi=150)
    plt.show()

    print("-" * 60)
    print(f"Final continuity: {history[-1][2]:.4e}")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
