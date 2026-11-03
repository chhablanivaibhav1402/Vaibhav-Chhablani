from __future__ import annotations

import csv
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 7: 1D LINEARIZED SHALLOW-WATER / LAKE-WAVE MODEL
# Numerical methods: Upwind and Lax-Wendroff
# ============================================================

# ---------------------- USER PARAMETERS ----------------------
NX = 401                 # Number of grid points
LENGTH = 10.0            # Domain length (m)
DEPTH = 1.0              # Mean water depth (m)
AMPLITUDE = 0.02         # Initial wave amplitude (m)
SIGMA = 0.5              # Width of initial Gaussian pulse (m)
CENTER = LENGTH / 2.0    # Initial pulse centre (m)
END_TIME = 2.0           # Simulation end time (s)
CFL = 0.80               # Must be in the range 0 < CFL <= 1
OUTPUT_EVERY = 25        # Save a VTK snapshot every N steps
GRAVITY = 9.81           # Gravitational acceleration (m/s^2)
VERTICAL_SCALE = 180.0   # Visual scaling only; does not affect physics
NZ_VISUAL = 61           # Number of points in the VTK vertical direction

OUTPUT_DIR = Path("experiment_07_outputs")
MAKE_PLOTS = True
MAKE_ZIP = True


# ---------------------- VALIDATE INPUTS ----------------------
def validate_inputs():
    if NX < 41:
        raise ValueError("NX must be at least 41.")
    if LENGTH <= 0 or DEPTH <= 0 or SIGMA <= 0 or GRAVITY <= 0:
        raise ValueError("LENGTH, DEPTH, SIGMA and GRAVITY must be positive.")
    if END_TIME < 0:
        raise ValueError("END_TIME cannot be negative.")
    if not (0.0 < CFL <= 1.0):
        raise ValueError("CFL must be greater than 0 and no greater than 1.")
    if OUTPUT_EVERY < 1 or NZ_VISUAL < 3:
        raise ValueError("OUTPUT_EVERY must be >= 1 and NZ_VISUAL must be >= 3.")
    if not (0.0 < CENTER < LENGTH):
        raise ValueError("CENTER must lie inside the domain (0, LENGTH).")


# ---------------------- MATHEMATICAL FUNCTIONS ----------------------
def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def periodic_gaussian(x, center, sigma, length):
    """Gaussian pulse with periodic distance, consistent with the numerical grid."""
    distance = (x - center + 0.5 * length) % length - 0.5 * length
    return np.exp(-0.5 * (distance / sigma) ** 2)


def analytical_eta(x, time, speed, amplitude, center, sigma):
    """Analytical periodic solution: two pulses travelling in opposite directions."""
    right = periodic_gaussian(x, center + speed * time, sigma, LENGTH)
    left = periodic_gaussian(x, center - speed * time, sigma, LENGTH)
    return 0.5 * amplitude * (right + left)


def upwind(field, courant):
    """First-order upwind method for a constant-velocity advection wave."""
    if courant >= 0.0:
        return field - courant * (field - np.roll(field, 1))
    return field - courant * (np.roll(field, -1) - field)


def lax_wendroff(field, courant):
    """Second-order Lax-Wendroff method for a constant-velocity wave."""
    return (
        field
        - 0.5 * courant * (np.roll(field, -1) - np.roll(field, 1))
        + 0.5 * courant**2
        * (np.roll(field, -1) - 2.0 * field + np.roll(field, 1))
    )


def reconstruct(w_plus, w_minus, depth, wave_speed):
    velocity = 0.5 * (w_plus + w_minus)
    eta = depth * (w_plus - w_minus) / (2.0 * wave_speed)
    return eta, velocity


def relative_l2(numerical, reference):
    numerator = np.linalg.norm((numerical - reference).ravel())
    denominator = np.linalg.norm(reference.ravel())
    return float(numerator / max(float(denominator), 1.0e-30))


def periodic_mass(field, dx):
    # The grid excludes the duplicate end point, so use a periodic sum.
    return float(np.sum(field) * dx)


# ---------------------- VTK OUTPUT ----------------------
def write_structured_grid_vtk(path, x, y, scalars=None, title="Lake-wave output"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be equally shaped 2D arrays.")

    ny, nx = x.shape
    scalars = scalars or {}

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_GRID\n")
        handle.write(f"DIMENSIONS {nx} {ny} 1\n")
        handle.write(f"POINTS {nx * ny} float\n")

        for j in range(ny):
            for i in range(nx):
                handle.write(f"{x[j, i]:.12e} {y[j, i]:.12e} 0.0\n")

        handle.write(f"POINT_DATA {nx * ny}\n")
        for name, values in scalars.items():
            values = np.asarray(values, dtype=float)
            if values.shape != x.shape:
                raise ValueError(f"Scalar field {name} has an incorrect shape.")
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in values.ravel(order="C"):
                handle.write(f"{value:.12e}\n")


def write_visual_files(output_dir, step, x, eta, velocity, depth, scale, nz):
    vertical_fraction = np.linspace(0.0, 1.0, nz)
    xx = np.repeat(x[None, :], nz, axis=0)

    bottom = -depth * scale
    surface = eta * scale
    yy = bottom + vertical_fraction[:, None] * (surface[None, :] - bottom)

    eta_field = np.repeat(eta[None, :], nz, axis=0)
    velocity_field = np.repeat(velocity[None, :], nz, axis=0)
    fraction_field = np.repeat(vertical_fraction[:, None], x.size, axis=1)

    body_path = output_dir / f"lake_water_body_{step:06d}.vtk"
    write_structured_grid_vtk(
        body_path, xx, yy,
        {
            "surface_elevation": eta_field,
            "vertical_fraction": fraction_field,
            "velocity_magnitude": np.abs(velocity_field),
        },
        "Lake water-body visualization",
    )

    ribbon_y = np.vstack((eta * scale - 0.01 * scale,
                          eta * scale + 0.01 * scale))
    ribbon_x = np.repeat(x[None, :], 2, axis=0)
    ribbon_eta = np.repeat(eta[None, :], 2, axis=0)
    ribbon_path = output_dir / f"lake_surface_scaled_{step:06d}.vtk"
    write_structured_grid_vtk(
        ribbon_path, ribbon_x, ribbon_y,
        {"surface_elevation": ribbon_eta},
        "Lake free-surface ribbon",
    )


# ---------------------- MAIN SOLVER ----------------------
def run_simulation():
    validate_inputs()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Periodic grid: endpoint=False avoids duplicating the same periodic point.
    x = np.linspace(0.0, LENGTH, NX, endpoint=False)
    dx = LENGTH / NX

    wave_speed = math.sqrt(GRAVITY * DEPTH)
    dt_nominal = CFL * dx / wave_speed

    eta_initial = AMPLITUDE * gaussian(x, CENTER, SIGMA)

    # Riemann invariants for the two travelling wave components.
    w_plus_up = wave_speed * eta_initial / DEPTH
    w_minus_up = -wave_speed * eta_initial / DEPTH
    w_plus_lw = w_plus_up.copy()
    w_minus_lw = w_minus_up.copy()

    eta_up, velocity_up = reconstruct(
        w_plus_up, w_minus_up, DEPTH, wave_speed
    )
    eta_lw, velocity_lw = reconstruct(
        w_plus_lw, w_minus_lw, DEPTH, wave_speed
    )

    history = []
    snapshot_times = [0.0]
    snapshot_eta = [eta_lw.copy()]

    time = 0.0
    step = 0

    while time < END_TIME - 1.0e-14:
        dt = min(dt_nominal, END_TIME - time)
        courant = wave_speed * dt / dx

        w_plus_up = upwind(w_plus_up, courant)
        w_minus_up = upwind(w_minus_up, -courant)

        w_plus_lw = lax_wendroff(w_plus_lw, courant)
        w_minus_lw = lax_wendroff(w_minus_lw, -courant)

        time += dt
        step += 1

        eta_up, velocity_up = reconstruct(
            w_plus_up, w_minus_up, DEPTH, wave_speed
        )
        eta_lw, velocity_lw = reconstruct(
            w_plus_lw, w_minus_lw, DEPTH, wave_speed
        )

        reference = analytical_eta(
            x, time, wave_speed, AMPLITUDE, CENTER, SIGMA
        )

        error_up = relative_l2(eta_up, reference)
        error_lw = relative_l2(eta_lw, reference)

        mass_up = periodic_mass(eta_up, dx)
        mass_lw = periodic_mass(eta_lw, dx)

        history.append([
            step, time, dt, courant, error_up, error_lw, mass_up, mass_lw
        ])

        if step % OUTPUT_EVERY == 0 or time >= END_TIME - 1.0e-14:
            write_visual_files(
                OUTPUT_DIR, step, x, eta_lw, velocity_lw,
                DEPTH, VERTICAL_SCALE, NZ_VISUAL
            )
            snapshot_times.append(time)
            snapshot_eta.append(eta_lw.copy())

            print(
                f"Step={step:5d}  Time={time:.5f} s  "
                f"Upwind error={error_up:.6e}  "
                f"Lax-Wendroff error={error_lw:.6e}"
            )

    reference = analytical_eta(
        x, time, wave_speed, AMPLITUDE, CENTER, SIGMA
    )

    # Final VTK files with simple, easy-to-find names.
    write_visual_files(
        OUTPUT_DIR, step, x, eta_lw, velocity_lw,
        DEPTH, VERTICAL_SCALE, NZ_VISUAL
    )
    shutil.copy2(
        OUTPUT_DIR / f"lake_water_body_{step:06d}.vtk",
        OUTPUT_DIR / "OPEN_THIS_lake_visual_final.vtk"
    )
    shutil.copy2(
        OUTPUT_DIR / f"lake_surface_scaled_{step:06d}.vtk",
        OUTPUT_DIR / "OPEN_THIS_lake_surface_ribbon_final.vtk"
    )

    # Space-time VTK field.
    snapshot_array = np.asarray(snapshot_eta)
    time_array = np.asarray(snapshot_times)
    xx = np.repeat(x[None, :], time_array.size, axis=0)
    tt = np.repeat(time_array[:, None], NX, axis=1)
    write_structured_grid_vtk(
        OUTPUT_DIR / "lake_space_time.vtk",
        xx, tt,
        {"surface_elevation": snapshot_array},
        "Lake wave space-time field",
    )

    # CSV: time history.
    with open(OUTPUT_DIR / "lake_wave_history.csv",
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "step", "time", "dt", "CFL",
            "relative_l2_error_upwind",
            "relative_l2_error_lax_wendroff",
            "mass_upwind", "mass_lax_wendroff",
        ])
        writer.writerows(history)

    # CSV: final profiles.
    with open(OUTPUT_DIR / "lake_wave_profiles.csv",
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "x", "eta_upwind", "velocity_upwind",
            "eta_lax_wendroff", "velocity_lax_wendroff",
            "eta_analytical",
        ])
        writer.writerows(zip(
            x, eta_up, velocity_up, eta_lw, velocity_lw, reference
        ))

    # Text summary.
    with open(OUTPUT_DIR / "lake_wave_summary.txt",
              "w", encoding="utf-8") as handle:
        handle.write("Linearized one-dimensional shallow-water equations\n")
        handle.write(f"NX {NX}\n")
        handle.write(f"length_m {LENGTH:.8e}\n")
        handle.write(f"mean_depth_m {DEPTH:.8e}\n")
        handle.write(f"gravity_m_s2 {GRAVITY:.8e}\n")
        handle.write(f"wave_speed_m_s {wave_speed:.8e}\n")
        handle.write(f"initial_amplitude_m {AMPLITUDE:.8e}\n")
        handle.write(f"sigma_m {SIGMA:.8e}\n")
        handle.write(f"initial_center_m {CENTER:.8e}\n")
        handle.write(f"final_time_s {time:.8e}\n")
        handle.write(f"time_steps {step}\n")
        handle.write(f"CFL {CFL:.8e}\n")
        handle.write(f"final_mass_upwind {periodic_mass(eta_up, dx):.8e}\n")
        handle.write(f"final_mass_lax_wendroff {periodic_mass(eta_lw, dx):.8e}\n")
        handle.write(f"relative_L2_error_upwind {relative_l2(eta_up, reference):.8e}\n")
        handle.write(f"relative_L2_error_lax_wendroff {relative_l2(eta_lw, reference):.8e}\n")

    print("\nSimulation completed.")
    print((OUTPUT_DIR / "lake_wave_summary.txt").read_text(encoding="utf-8"))

    if MAKE_PLOTS:
        # Final wave profile.
        plt.figure(figsize=(9, 4.5))
        plt.plot(x, eta_up, label="Upwind")
        plt.plot(x, eta_lw, label="Lax-Wendroff")
        plt.plot(x, reference, "--", label="Analytical")
        plt.xlabel("Distance, x (m)")
        plt.ylabel("Surface displacement, eta (m)")
        plt.title("Final lake-wave profile")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "final_wave_profile.png", dpi=200)
        plt.show()

        # Error history.
        hist = np.asarray(history)
        if hist.size:
            plt.figure(figsize=(8, 4.5))
            plt.semilogy(hist[:, 1], hist[:, 4], label="Upwind")
            plt.semilogy(hist[:, 1], hist[:, 5], label="Lax-Wendroff")
            plt.xlabel("Time (s)")
            plt.ylabel("Relative L2 error")
            plt.title("Numerical error versus time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "error_evolution.png", dpi=200)
            plt.show()

        # Space-time contour plot.
        plt.figure(figsize=(9, 4.5))
        contour = plt.contourf(
            x, time_array, snapshot_array, levels=30
        )
        plt.colorbar(contour, label="Surface displacement, eta (m)")
        plt.xlabel("Distance, x (m)")
        plt.ylabel("Time (s)")
        plt.title("Lake-wave space-time development")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "space_time_contour.png", dpi=200)
        plt.show()

    if MAKE_ZIP:
        zip_path = OUTPUT_DIR.parent / "experiment_07_results.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for file_path in sorted(OUTPUT_DIR.rglob("*")):
                if file_path.is_file():
                    archive.write(
                        file_path, file_path.relative_to(OUTPUT_DIR)
                    )
        print(f"Results ZIP created: {zip_path.resolve()}")

    print(f"All output files are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_simulation()
