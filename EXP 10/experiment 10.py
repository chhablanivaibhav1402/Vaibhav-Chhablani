"""
Experiment 10: Pipe-flow turbulence-model comparison
----------------------------------------------------
Self-contained teaching-level radial eddy-viscosity solver.

Models:
  1. Laminar
  2. Mixing-length
  3. Van-Driest-damped mixing-length
  4. Equilibrium k-epsilon closure
  5. Equilibrium k-omega SST-like closure

Important:
This is a teaching model using algebraic/equilibrium closures. It is NOT
a full industrial k-epsilon or SST transport-equation RANS solver.

Run:
    python experiment_10_pipe_flow.py

Edit the USER SETTINGS section before running.
Requires: numpy, matplotlib
"""

from __future__ import annotations

import csv
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. USER SETTINGS — change these values
# =============================================================================

RUN_MODE = "custom"       # "quick", "lab", or "custom"

# Custom settings (used when RUN_MODE = "custom")
N_RADIAL = 88             # Number of radial control volumes; minimum 21
VTK_NX = 188              # Axial points in VTK field
VTK_NY = 48               # Vertical points in VTK field
RE_MIN = 8888.0           # Minimum Reynolds number
RE_MAX = 88888.0          # Maximum Reynolds number
N_RE = 4                  # Number of Reynolds-number cases
MAX_ITERATIONS = 1200
TOLERANCE = 1.0e-7

# Physical/geometry settings
PIPE_DIAMETER = 1.0       # m
BULK_VELOCITY = 1.0       # m/s
PIPE_LENGTH = 10.0        # m
DENSITY = 1.0             # kg/m^3; set actual density if required

# Output settings
OUTPUT_FOLDER = Path("experiment_10_outputs")
MAKE_ZIP = True
SHOW_PLOTS = True


# =============================================================================
# 2. Numerical utilities
# =============================================================================

def tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray,
         d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm."""
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()

    n = d.size
    if n < 1 or any(arr.size != n for arr in (a, b, c)):
        raise ValueError("TDMA arrays must have equal nonzero lengths.")

    for i in range(1, n):
        if abs(b[i - 1]) < 1e-30:
            raise FloatingPointError("Zero pivot encountered in TDMA.")
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    if abs(b[-1]) < 1e-30:
        raise FloatingPointError("Zero final pivot encountered in TDMA.")

    result = np.empty(n)
    result[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        if abs(b[i]) < 1e-30:
            raise FloatingPointError("Zero pivot encountered in back substitution.")
        result[i] = (d[i] - c[i] * result[i + 1]) / b[i]

    return result


def integrate_bulk_velocity(radius: np.ndarray, velocity: np.ndarray,
                            pipe_radius: float, dr: float) -> float:
    """Area-weighted bulk velocity for an axisymmetric pipe."""
    integral = np.sum(velocity * radius) * dr
    return 2.0 * float(integral) / (pipe_radius ** 2)


def write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_structured_vtk(path: Path, x: np.ndarray, y: np.ndarray,
                         scalars: dict[str, np.ndarray],
                         vectors: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Write a legacy ASCII VTK STRUCTURED_GRID file."""
    ny, nx = x.shape
    if x.shape != y.shape:
        raise ValueError("VTK x and y arrays must have the same shape.")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("Experiment 10 pipe-flow model comparison\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_GRID\n")
        handle.write(f"DIMENSIONS {nx} {ny} 1\n")
        handle.write(f"POINTS {nx * ny} float\n")

        for j in range(ny):
            for i in range(nx):
                handle.write(f"{x[j, i]:.10e} {y[j, i]:.10e} 0.0\n")

        handle.write(f"POINT_DATA {nx * ny}\n")

        for name, field in scalars.items():
            if field.shape != x.shape:
                raise ValueError(f"Scalar field {name} has an incorrect shape.")
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in field.ravel(order="C"):
                handle.write(f"{value:.10e}\n")

        for name, (u, v) in vectors.items():
            if u.shape != x.shape or v.shape != x.shape:
                raise ValueError(f"Vector field {name} has an incorrect shape.")
            handle.write(f"VECTORS {name} float\n")
            for ux, uy in zip(u.ravel(order="C"), v.ravel(order="C")):
                handle.write(f"{ux:.10e} {uy:.10e} 0.0\n")


# =============================================================================
# 3. Turbulence-model solver
# =============================================================================

@dataclass
class ModelResult:
    name: str
    reynolds: float
    nu: float
    friction_factor: float
    friction_velocity: float
    pressure_drop: float
    residual: float
    iterations: int
    radius: np.ndarray
    velocity: np.ndarray
    nut: np.ndarray
    k: np.ndarray
    epsilon: np.ndarray
    omega: np.ndarray


MODEL_LIST = [
    (0, "laminar"),
    (1, "mixing_length"),
    (2, "van_driest"),
    (3, "kepsilon_equilibrium"),
    (4, "komega_sst_equilibrium"),
]


def calculate_eddy_viscosity(model: int, radius: np.ndarray,
                             velocity: np.ndarray, u_tau: float,
                             nu: float, dr: float, pipe_radius: float
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate eddy viscosity and turbulence quantities from each closure."""
    derivative = np.gradient(velocity, dr, edge_order=2)
    strain = np.abs(derivative)

    wall_distance = np.maximum(pipe_radius - radius, 0.5 * dr)
    y_plus = wall_distance * u_tau / nu

    kappa = 0.41
    a_plus = 26.0
    c_mu = 0.09
    beta_star = 0.09
    a1 = 0.31

    damping = 1.0 - np.exp(-y_plus / a_plus)

    nut = np.zeros_like(radius)
    k = np.zeros_like(radius)
    epsilon = np.zeros_like(radius)
    omega = np.zeros_like(radius)

    if model == 0:
        pass

    elif model == 1:
        mixing_length = kappa * wall_distance
        nut = mixing_length**2 * strain

    elif model == 2:
        mixing_length = kappa * wall_distance * damping
        nut = mixing_length**2 * strain

    elif model == 3:
        k_damping = 1.0 - np.exp(-y_plus / 12.0)
        k = (u_tau**2 / math.sqrt(c_mu)) * k_damping**2

        y_use = np.maximum(wall_distance, 1.5 * nu / max(u_tau, 1e-12))
        epsilon = c_mu**0.75 * np.maximum(k, 1e-16)**1.5 / (kappa * y_use)

        nut = c_mu * k**2 / np.maximum(epsilon, 1e-18)
        nut *= damping**2

    elif model == 4:
        k_damping = 1.0 - np.exp(-y_plus / 10.0)
        k = (u_tau**2 / math.sqrt(beta_star)) * k_damping**2

        y_use = np.maximum(wall_distance, 1.5 * nu / max(u_tau, 1e-12))
        omega = u_tau / (math.sqrt(beta_star) * kappa * y_use)

        f2_argument = (
            2.0 * np.sqrt(np.maximum(k, 0.0))
            / np.maximum(beta_star * omega * y_use, 1e-18)
        )
        f2 = np.tanh(np.maximum(f2_argument, 0.0)**2)

        denominator = np.maximum(a1 * omega, strain * f2)
        nut = a1 * k / np.maximum(denominator, 1e-18)
        nut *= damping**2
        epsilon = beta_star * k * omega

    else:
        raise ValueError("Unknown turbulence model index.")

    # A numerical cap prevents extreme algebraic eddy-viscosity values.
    nut = np.clip(nut, 0.0, 300.0 * nu)
    return nut, k, epsilon, omega


def solve_model(model: int, name: str, n: int, reynolds: float,
                max_iterations: int, tolerance: float,
                diameter: float, bulk_velocity: float
                ) -> ModelResult:
    """Solve the steady radial momentum equation for one model."""
    pipe_radius = diameter / 2.0
    nu = bulk_velocity * diameter / reynolds
    dr = pipe_radius / n
    radius = (np.arange(n) + 0.5) * dr

    # Parabolic initial guess with the requested bulk velocity.
    velocity = 2.0 * bulk_velocity * (1.0 - (radius / pipe_radius)**2)

    nut = np.zeros(n)
    k = np.zeros(n)
    epsilon = np.zeros(n)
    omega = np.zeros(n)

    f_guess = 0.3164 * max(reynolds, 4000.0)**(-0.25)
    u_tau = bulk_velocity * math.sqrt(max(f_guess / 8.0, 1e-12))

    residual = 1.0
    completed = 0

    for iteration in range(1, max_iterations + 1):
        old_velocity = velocity.copy()

        nut, k, epsilon, omega = calculate_eddy_viscosity(
            model, radius, velocity, u_tau, nu, dr, pipe_radius
        )

        effective_nu = nu + nut

        # Face viscosities: arithmetic interpolation internally.
        face_nu = np.empty(n + 1)
        face_nu[0] = effective_nu[0]  # symmetry face; zero gradient is imposed
        face_nu[1:n] = 0.5 * (effective_nu[:-1] + effective_nu[1:])
        face_nu[n] = nu                # no-slip wall face: nut -> 0 at wall

        # Finite-volume equation:
        # d/dr [ r * (nu + nut) * dU/dr ] = -G*r
        # G is adjusted after solving to enforce the specified bulk velocity.
        lower = np.zeros(n)
        diagonal = np.zeros(n)
        upper = np.zeros(n)
        rhs = radius * dr

        for i in range(n):
            west_radius = i * dr
            east_radius = (i + 1) * dr

            west_coeff = west_radius * face_nu[i] / dr if i > 0 else 0.0
            east_distance = 0.5 * dr if i == n - 1 else dr
            east_coeff = east_radius * face_nu[i + 1] / east_distance

            if i > 0:
                lower[i] = -west_coeff
            upper[i] = -east_coeff
            diagonal[i] = west_coeff + east_coeff

        candidate = tdma(lower, diagonal, upper, rhs)

        current_bulk = integrate_bulk_velocity(
            radius, candidate, pipe_radius, dr
        )
        if not np.isfinite(current_bulk) or current_bulk <= 0.0:
            raise FloatingPointError(f"Invalid bulk velocity for {name}, Re={reynolds:g}")

        candidate *= bulk_velocity / current_bulk

        # Wall shear from the last cell centre to the no-slip wall.
        wall_gradient = candidate[-1] / (0.5 * dr)
        new_u_tau = math.sqrt(max(nu * abs(wall_gradient), 1e-18))

        relaxation = 1.0 if model == 0 else 0.45
        velocity = relaxation * candidate + (1.0 - relaxation) * velocity

        residual = float(
            np.linalg.norm(velocity - old_velocity)
            / max(np.linalg.norm(old_velocity), 1e-30)
        )
        u_tau = 0.35 * new_u_tau + 0.65 * u_tau
        completed = iteration

        if residual < tolerance and iteration >= 20:
            break

    # Recalculate turbulence quantities using the final velocity/u_tau.
    nut, k, epsilon, omega = calculate_eddy_viscosity(
        model, radius, velocity, u_tau, nu, dr, pipe_radius
    )

    # Darcy friction factor: f_D = 8*(u_tau/U_bulk)^2.
    friction_factor = 8.0 * (u_tau / bulk_velocity)**2
    pressure_drop = (
        friction_factor * (PIPE_LENGTH / diameter)
        * 0.5 * DENSITY * bulk_velocity**2
    )

    return ModelResult(
        name, reynolds, nu, friction_factor, u_tau, pressure_drop,
        residual, completed, radius, velocity, nut, k, epsilon, omega
    )


# =============================================================================
# 4. Reference correlations and output generation
# =============================================================================

def reference_friction_factors(reynolds: float) -> tuple[float, float, float]:
    """Return laminar, Blasius and smooth-Haaland Darcy factors."""
    laminar = 64.0 / reynolds
    blasius = 0.3164 * reynolds**(-0.25) if reynolds >= 4000 else float("nan")
    haaland = (
        1.0 / (-1.8 * math.log10(6.9 / reynolds))**2
        if reynolds > 0 else float("nan")
    )
    return laminar, blasius, haaland


def interpolate_profile(radius: np.ndarray, values: np.ndarray,
                        query_radius: np.ndarray, wall_value: float = 0.0) -> np.ndarray:
    pipe_radius = PIPE_DIAMETER / 2.0
    xp = np.concatenate(([0.0], radius, [pipe_radius]))
    fp = np.concatenate(([values[0]], values, [wall_value]))
    return np.interp(query_radius, xp, fp)


def generate_case_outputs(results: list[ModelResult], output_dir: Path,
                          vtk_nx: int, vtk_ny: int) -> None:
    reynolds = results[0].reynolds
    x_line = np.linspace(0.0, PIPE_LENGTH, vtk_nx)
    y_line = np.linspace(-PIPE_DIAMETER / 2.0, PIPE_DIAMETER / 2.0, vtk_ny)
    x, y = np.meshgrid(x_line, y_line)
    radial_position = np.abs(y)

    scalars: dict[str, np.ndarray] = {}
    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for result in results:
        u_field = interpolate_profile(
            result.radius, result.velocity, radial_position, 0.0
        )
        nut_field = interpolate_profile(
            result.radius, result.nut / result.nu, radial_position, 0.0
        )
        k_field = interpolate_profile(result.radius, result.k, radial_position, 0.0)
        eps_field = interpolate_profile(
            result.radius, result.epsilon, radial_position, 0.0
        )
        omega_field = interpolate_profile(
            result.radius, result.omega, radial_position, 0.0
        )
        pressure_field = result.pressure_drop * (1.0 - x / PIPE_LENGTH)

        scalars[f"U_{result.name}"] = u_field
        scalars[f"nut_over_nu_{result.name}"] = nut_field
        scalars[f"pressure_{result.name}"] = pressure_field
        scalars[f"k_{result.name}"] = k_field
        scalars[f"epsilon_{result.name}"] = eps_field
        scalars[f"omega_{result.name}"] = omega_field
        vectors[f"velocity_{result.name}"] = (u_field, np.zeros_like(u_field))

    tag = f"{reynolds:.0f}"
    vtk_path = output_dir / f"pipe2d_compare_Re{tag}.vtk"
    write_structured_vtk(vtk_path, x, y, scalars, vectors)

    header = ["r_over_R"]
    for result in results:
        header.extend([
            f"U_{result.name}", f"nut_over_nu_{result.name}",
            f"k_{result.name}", f"epsilon_{result.name}", f"omega_{result.name}"
        ])

    rows = []
    pipe_radius = PIPE_DIAMETER / 2.0
    for i in range(len(results[0].radius)):
        row = [results[0].radius[i] / pipe_radius]
        for result in results:
            row.extend([
                result.velocity[i], result.nut[i] / result.nu,
                result.k[i], result.epsilon[i], result.omega[i]
            ])
        rows.append(row)

    write_csv(output_dir / f"pipe2d_profiles_Re{tag}.csv", header, rows)


def write_summary(results_all: list[ModelResult], output_dir: Path) -> Path:
    rows = []
    for result in results_all:
        laminar, blasius, haaland = reference_friction_factors(result.reynolds)
        rows.append([
            result.reynolds, result.name, result.friction_factor,
            laminar, blasius, haaland, result.friction_velocity,
            result.pressure_drop, result.velocity[0],
            float(np.max(result.nut / result.nu)),
            result.iterations, result.residual
        ])

    header = [
        "Re", "model", "friction_factor_Darcy", "laminar_64_over_Re",
        "Blasius_reference", "Haaland_smooth_reference", "uTau",
        "pressure_drop_over_pipe_length", "U_center",
        "maximum_nut_over_nu", "iterations", "residual"
    ]
    csv_path = output_dir / "pipe2d_rans_summary.csv"
    write_csv(csv_path, header, rows)

    summary_path = output_dir / "pipe2d_rans_summary.txt"
    summary_path.write_text(
        "EXPERIMENT 10 — PIPE-FLOW MODEL COMPARISON\n"
        "==========================================\n"
        "Models: laminar, mixing length, Van-Driest damping,\n"
        "        equilibrium k-epsilon, equilibrium k-omega SST-like.\n\n"
        "The solver uses a radial finite-volume momentum equation and\n"
        "extrudes the fully developed velocity profile along the pipe.\n"
        "The k-epsilon and k-omega options are simplified equilibrium\n"
        "closures, not full transport-equation RANS models.\n\n"
        "Reference correlations:\n"
        "  Laminar Darcy factor: 64/Re\n"
        "  Blasius:              0.3164 Re^(-0.25), turbulent range only\n"
        "  Smooth Haaland:       1/[-1.8 log10(6.9/Re)]^2\n\n"
        "See pipe2d_rans_summary.csv for numerical results.\n",
        encoding="utf-8"
    )
    return csv_path


# =============================================================================
# 5. Plots
# =============================================================================

def make_plots(output_dir: Path) -> None:
    data = np.genfromtxt(
        output_dir / "pipe2d_rans_summary.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    data = np.atleast_1d(data)

    plt.figure(figsize=(9, 5))
    for model in np.unique(data["model"]):
        selected = data[data["model"] == model]
        plt.loglog(
            selected["Re"], selected["friction_factor_Darcy"],
            marker="o", label=str(model)
        )

    unique_re = np.unique(data["Re"])
    laminar_values = np.array([64.0 / re for re in unique_re])
    plt.loglog(unique_re, laminar_values, "k:", label="Laminar 64/Re")

    turbulent = unique_re >= 4000
    if np.any(turbulent):
        turbulent_re = unique_re[turbulent]
        plt.loglog(
            turbulent_re, 0.3164 * turbulent_re**(-0.25),
            "k--", label="Blasius"
        )
        plt.loglog(
            turbulent_re,
            1.0 / (-1.8 * np.log10(6.9 / turbulent_re))**2,
            "k-.", label="Smooth Haaland"
        )

    plt.xlabel("Reynolds number")
    plt.ylabel("Darcy friction factor")
    plt.title("Pipe-flow friction-factor comparison")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "friction_factor_comparison.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    profile_paths = sorted(output_dir.glob("pipe2d_profiles_Re*.csv"))
    if profile_paths:
        profile = np.genfromtxt(
            profile_paths[0], delimiter=",", names=True,
            dtype=None, encoding="utf-8"
        )
        profile = np.atleast_1d(profile)

        plt.figure(figsize=(8, 5))
        for field in profile.dtype.names:
            if field.startswith("U_"):
                plt.plot(profile[field], profile["r_over_R"], label=field[2:])
        plt.xlabel("Axial velocity (m/s)")
        plt.ylabel("r/R")
        plt.title(f"Velocity profiles — {profile_paths[0].stem}")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "velocity_profiles.png", dpi=180)
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()


def preview_vtk(output_dir: Path, reynolds: float) -> None:
    path = output_dir / f"pipe2d_compare_Re{reynolds:.0f}.vtk"
    if not path.exists():
        print(f"VTK preview skipped; file not found: {path.name}")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    dims_index = next(i for i, line in enumerate(lines) if line.startswith("DIMENSIONS "))
    _, nx, ny, nz = lines[dims_index].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    if nz != 1:
        print("VTK preview supports 2D structured grids only.")
        return

    points_index = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    npoints = int(lines[points_index].split()[1])
    points = np.array([
        [float(v) for v in lines[points_index + 1 + i].split()[:3]]
        for i in range(npoints)
    ])
    x = points[:, 0].reshape(ny, nx)
    y = points[:, 1].reshape(ny, nx)

    scalars: dict[str, np.ndarray] = {}
    i = next(j for j, line in enumerate(lines) if line.startswith("POINT_DATA ")) + 1
    while i < len(lines):
        if lines[i].startswith("SCALARS "):
            name = lines[i].split()[1]
            i += 2
            values = np.array([float(lines[i + k].split()[0]) for k in range(npoints)])
            scalars[name] = values.reshape(ny, nx)
            i += npoints
        elif lines[i].startswith("VECTORS "):
            i += npoints + 1
        else:
            i += 1

    field_name = "U_mixing_length"
    if field_name not in scalars:
        field_name = next(iter(scalars))

    plt.figure(figsize=(9, 4.8))
    contour = plt.contourf(x, y, scalars[field_name], levels=30)
    plt.colorbar(contour, label="Velocity (m/s)")
    plt.xlabel("Axial distance (m)")
    plt.ylabel("Radial coordinate (m)")
    plt.title(f"{field_name.replace('_', ' ')} — Re={reynolds:.0f}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "vtk_velocity_preview.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 6. Parameter selection, validation, main program
# =============================================================================

def select_parameters() -> tuple[int, int, int, float, float, int, int, float]:
    if RUN_MODE == "quick":
        return 31, 50, 31, 8000.0, 16000.0, 2, 250, 1e-5
    if RUN_MODE == "lab":
        return 121, 160, 61, 10000.0, 100000.0, 3, 1500, 1e-7
    if RUN_MODE == "custom":
        return (
            N_RADIAL, VTK_NX, VTK_NY, RE_MIN, RE_MAX,
            N_RE, MAX_ITERATIONS, TOLERANCE
        )
    raise ValueError("RUN_MODE must be 'quick', 'lab', or 'custom'.")


def validate_parameters(params: tuple[int, int, int, float, float, int, int, float]) -> None:
    n_radial, vtk_nx, vtk_ny, re_min, re_max, n_re, max_iter, tolerance = params

    if n_radial < 21:
        raise ValueError("N_RADIAL must be at least 21.")
    if vtk_nx < 10 or vtk_ny < 11:
        raise ValueError("VTK_NX must be >=10 and VTK_NY must be >=11.")
    if not (math.isfinite(re_min) and math.isfinite(re_max)):
        raise ValueError("Reynolds-number limits must be finite.")
    if re_min <= 0 or re_max < re_min:
        raise ValueError("Require 0 < RE_MIN <= RE_MAX.")
    if n_re < 1 or max_iter < 20:
        raise ValueError("N_RE must be >=1 and MAX_ITERATIONS must be >=20.")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("TOLERANCE must be positive and finite.")
    if PIPE_DIAMETER <= 0 or BULK_VELOCITY <= 0 or PIPE_LENGTH <= 0 or DENSITY <= 0:
        raise ValueError("Pipe dimensions, velocity and density must be positive.")


def make_zip(output_dir: Path) -> Path:
    archive = output_dir.parent / "experiment_10_results.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zip_handle.write(path, path.relative_to(output_dir))
    return archive


def main() -> None:
    params = select_parameters()
    validate_parameters(params)

    n_radial, vtk_nx, vtk_ny, re_min, re_max, n_re, max_iter, tolerance = params

    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    reynolds_values = (
        np.array([re_min]) if n_re == 1
        else np.geomspace(re_min, re_max, n_re)
    )

    print("EXPERIMENT 10 — PIPE-FLOW TURBULENCE MODEL COMPARISON")
    print(f"Run mode: {RUN_MODE}")
    print(f"Radial cells: {n_radial}; VTK grid: {vtk_nx} x {vtk_ny}")
    print(f"Reynolds numbers: {reynolds_values}")
    print(f"Maximum iterations: {max_iter}; tolerance: {tolerance:g}")
    print("No CFL number is used: this is a steady radial solver, not a transient solver.")
    print()

    all_results: list[ModelResult] = []

    for reynolds in reynolds_values:
        print(f"Solving Re = {reynolds:.0f}")
        case_results = []

        for model, model_name in MODEL_LIST:
            result = solve_model(
                model, model_name, n_radial, float(reynolds),
                max_iter, tolerance, PIPE_DIAMETER, BULK_VELOCITY
            )
            case_results.append(result)
            all_results.append(result)

            print(
                f"  {model_name:26s} "
                f"f_D={result.friction_factor:.6e} "
                f"U_center={result.velocity[0]:.5f} "
                f"iterations={result.iterations:4d} "
                f"residual={result.residual:.3e}"
            )

        generate_case_outputs(case_results, OUTPUT_FOLDER, vtk_nx, vtk_ny)
        print()

    write_summary(all_results, OUTPUT_FOLDER)
    make_plots(OUTPUT_FOLDER)
    preview_vtk(OUTPUT_FOLDER, float(reynolds_values[0]))

    print("Generated files:")
    for path in sorted(OUTPUT_FOLDER.iterdir()):
        if path.is_file():
            print(f"  {path.name:42s} {path.stat().st_size / 1024:8.1f} KB")

    if MAKE_ZIP:
        archive = make_zip(OUTPUT_FOLDER)
        print(f"\nResults ZIP: {archive.resolve()}")


if __name__ == "__main__":
    main()
