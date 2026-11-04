
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from IPython.display import Markdown, display

# ============================================================
# EXPERIMENT 11: CONVECTION-DIFFUSION HEAT TRANSFER
# ============================================================

IN_COLAB = False

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    colab_files = None


# ============================================================
# 1. CREATE WORKING DIRECTORIES
# ============================================================

EXPERIMENT = 11

ROOT = (
    Path("/content/cfd_lab_exp11")
    if IN_COLAB
    else Path.home() / "CFD_Runs" / "exp11"
)

CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "outputs"

CODE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. COMMON FUNCTIONS
# ============================================================

def run_solver(script_path, arguments, output_dir, timeout=1800):

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script_path),
        *[str(value) for value in arguments]
    ]

    print("Running:", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    (output_dir / "solver_stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )

    (output_dir / "solver_stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )

    if completed.stdout.strip():
        print(completed.stdout.strip())

    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"Solver exited with code {completed.returncode}"
        )

    if completed.stderr.strip():
        print(completed.stderr.strip())

    return output_dir


def read_csv(path):

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8"
    )

    return np.atleast_1d(data)


def show_summary(path):

    text = Path(path).read_text(
        encoding="utf-8"
    ).strip()

    display(
        Markdown(
            f"### Solver summary\n```text\n{text}\n```"
        )
    )

    return text


def list_outputs(output_dir):

    rows = []

    for path in sorted(Path(output_dir).iterdir()):

        if path.is_file():

            size = path.stat().st_size

            if size < 1024:
                size_text = f"{size} B"

            elif size < 1024**2:
                size_text = f"{size / 1024:.1f} KB"

            else:
                size_text = f"{size / 1024**2:.1f} MB"

            rows.append((path.name, size_text))

    table = "\n".join(
        f"| `{name}` | {size} |"
        for name, size in rows
    )

    display(
        Markdown(
            f"### Generated files\n| File | Size |\n|---|---:|\n{table}"
        )
    )

    return rows


# ============================================================
# 3. VTK READER AND VISUALIZATION
# ============================================================

def read_structured_vtk(path):

    lines = Path(path).read_text(
        encoding="utf-8"
    ).splitlines()

    dims_index = next(
        i for i, line in enumerate(lines)
        if line.startswith("DIMENSIONS ")
    )

    _, nx, ny, nz = lines[dims_index].split()

    nx, ny, nz = int(nx), int(ny), int(nz)

    if nz != 1:
        raise ValueError(
            "Only 2D structured VTK files are supported"
        )

    points_index = next(
        i for i, line in enumerate(lines)
        if line.startswith("POINTS ")
    )

    npoints = int(
        lines[points_index].split()[1]
    )

    points = np.array([
        [
            float(v)
            for v in lines[points_index + 1 + k].split()[:3]
        ]
        for k in range(npoints)
    ])

    x = points[:, 0].reshape(ny, nx)
    y = points[:, 1].reshape(ny, nx)

    data_index = next(
        i for i, line in enumerate(lines)
        if line.startswith("POINT_DATA ")
    ) + 1

    scalars = {}
    vectors = {}

    i = data_index

    while i < len(lines):

        if lines[i].startswith("SCALARS "):

            name = lines[i].split()[1]

            i += 2

            values = np.array([
                float(lines[i + k].split()[0])
                for k in range(npoints)
            ])

            scalars[name] = values.reshape(ny, nx)

            i += npoints

        elif lines[i].startswith("VECTORS "):

            name = lines[i].split()[1]

            i += 1

            values = np.array([
                [
                    float(v)
                    for v in lines[i + k].split()[:3]
                ]
                for k in range(npoints)
            ])

            vectors[name] = (
                values[:, 0].reshape(ny, nx),
                values[:, 1].reshape(ny, nx)
            )

            i += npoints

        else:
            i += 1

    return x, y, scalars, vectors


def plot_vtk_scalar(path, preferred_field=None):

    path = Path(path)

    if not path.is_file():
        print(
            f"VTK preview skipped: {path.name} was not generated"
        )
        return

    x, y, scalars, _ = read_structured_vtk(path)

    if not scalars:
        print("No scalar field found")
        return

    field = (
        preferred_field
        if preferred_field in scalars
        else next(iter(scalars))
    )

    plt.figure(figsize=(8, 4.8))

    image = plt.contourf(
        x,
        y,
        scalars[field],
        levels=30
    )

    plt.colorbar(
        image,
        label=field.replace("_", " ")
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"{field.replace('_', ' ').title()} - Field Preview"
    )

    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. COMMON NUMERICAL FUNCTIONS
# ============================================================

def laplacian(field, dx, dy):

    out = np.zeros_like(field)

    out[1:-1, 1:-1] = (

        (
            field[1:-1, 2:]
            - 2 * field[1:-1, 1:-1]
            + field[1:-1, :-2]
        ) / dx**2

        +

        (
            field[2:, 1:-1]
            - 2 * field[1:-1, 1:-1]
            + field[:-2, 1:-1]
        ) / dy**2

    )

    return out


def upwind_derivative_x(field, velocity, dx):

    out = np.zeros_like(field)

    backward = (
        field[1:-1, 1:-1]
        - field[1:-1, :-2]
    ) / dx

    forward = (
        field[1:-1, 2:]
        - field[1:-1, 1:-1]
    ) / dx

    out[1:-1, 1:-1] = np.where(
        velocity[1:-1, 1:-1] >= 0,
        backward,
        forward
    )

    return out


def upwind_derivative_y(field, velocity, dy):

    out = np.zeros_like(field)

    backward = (
        field[1:-1, 1:-1]
        - field[:-2, 1:-1]
    ) / dy

    forward = (
        field[2:, 1:-1]
        - field[1:-1, 1:-1]
    ) / dy

    out[1:-1, 1:-1] = np.where(
        velocity[1:-1, 1:-1] >= 0,
        backward,
        forward
    )

    return out


def require_finite(*arrays):

    for array in arrays:

        if not np.all(np.isfinite(array)):
            raise FloatingPointError(
                "Non-finite numerical field detected"
            )


# ============================================================
# 5. VTK WRITERS
# ============================================================

def write_structured_grid_vtk(
    path,
    x,
    y,
    scalars=None,
    vectors=None,
    title="CFD lab output"
):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Invalid grid")

    ny, nx = x.shape

    scalar_map = dict(scalars or {})
    vector_map = dict(vectors or {})

    for name, values in scalar_map.items():

        if np.asarray(values).shape != x.shape:
            raise ValueError(
                f"Invalid scalar shape: {name}"
            )

    for name, values in vector_map.items():

        if (
            len(values) != 2
            or np.asarray(values[0]).shape != x.shape
            or np.asarray(values[1]).shape != x.shape
        ):
            raise ValueError(
                f"Invalid vector shape: {name}"
            )

    with Path(path).open(
        "w", encoding="utf-8"
    ) as handle:

        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_GRID\n")
        handle.write(f"DIMENSIONS {nx} {ny} 1\n")
        handle.write(f"POINTS {nx * ny} float\n")

        for j in range(ny):

            for i in range(nx):

                handle.write(
                    f"{x[j,i]:.12e} {y[j,i]:.12e} 0\n"
                )

        handle.write(
            f"POINT_DATA {nx * ny}\n"
        )

        for name, values in scalar_map.items():

            handle.write(
                f"SCALARS {name} float 1\n"
            )

            handle.write(
                "LOOKUP_TABLE default\n"
            )

            for value in np.asarray(values).ravel():

                handle.write(
                    f"{value:.12e}\n"
                )

        for name, values in vector_map.items():

            handle.write(
                f"VECTORS {name} float\n"
            )

            vx = np.asarray(values[0]).ravel()
            vy = np.asarray(values[1]).ravel()

            for ax, ay in zip(vx, vy):

                handle.write(
                    f"{ax:.12e} {ay:.12e} 0\n"
                )


def write_multiline_vtk(path, lines, title="lines"):

    points = []
    connectivity = []

    for x, y in lines:

        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()

        if xa.size != ya.size or xa.size < 2:
            raise ValueError("Invalid polyline")

        ids = list(
            range(len(points), len(points) + xa.size)
        )

        points.extend(
            zip(xa.tolist(), ya.tolist())
        )

        connectivity.append(ids)

    size = sum(
        len(ids) + 1
        for ids in connectivity
    )

    with Path(path).open(
        "w", encoding="utf-8"
    ) as handle:

        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n")

        for xi, yi in points:

            handle.write(
                f"{xi:.12e} {yi:.12e} 0\n"
            )

        handle.write(
            f"LINES {len(connectivity)} {size}\n"
        )

        for ids in connectivity:

            handle.write(
                f"{len(ids)} {' '.join(map(str, ids))}\n"
            )


# ============================================================
# 6. CASE DATA
# ============================================================

class CaseData:

    def __init__(
        self,
        name,
        x,
        y,
        temperature,
        u,
        v,
        marker,
        pe,
        alpha,
        dx,
        dy
    ):

        self.name = name
        self.x = x
        self.y = y
        self.temperature = temperature
        self.u = u
        self.v = v
        self.marker = marker
        self.pe = pe
        self.alpha = alpha
        self.dx = dx
        self.dy = dy


# ============================================================
# 7. INITIALIZE FLOW CASES
# ============================================================

def initialize_case(mode, nx, ny, pe):

    if mode in [
        "forced_plate",
        "forced_pipe"
    ]:

        length, height = 5.0, 1.0

    elif mode == "natural_plate":

        length, height = 1.0, 1.0

    else:

        raise ValueError("Unknown case")

    x1 = np.linspace(
        0, length, nx
    )

    y1 = np.linspace(
        0, height, ny
    )

    x, y = np.meshgrid(
        x1, y1
    )

    dx = x1[1] - x1[0]
    dy = y1[1] - y1[0]

    temperature = np.zeros(
        (ny, nx)
    )

    marker = np.zeros_like(
        temperature
    )

    if mode == "forced_plate":

        delta = (
            0.035
            + 0.22 * np.sqrt(
                np.maximum(x, dx) / length
            )
        )

        u = np.tanh(
            y / np.maximum(delta, 1e-12)
        )

        v = np.zeros_like(u)

        marker[
            0,
            x1 >= 0.25 * length
        ] = 1

        name = "forced_flat_plate"

    elif mode == "forced_pipe":

        eta = y / height

        u = 6 * eta * (1 - eta)

        v = np.zeros_like(u)

        marker[0, :] = 1
        marker[-1, :] = 1

        name = "forced_pipe_channel"

    else:

        amplitude = 1 / math.pi

        u = (
            -amplitude * math.pi / height
            * np.sin(math.pi * x / length)
            * np.cos(math.pi * y / height)
        )

        v = (
            amplitude * math.pi / length
            * np.cos(math.pi * x / length)
            * np.sin(math.pi * y / height)
        )

        temperature = 1 - x / length

        marker[:, 0] = 1
        marker[:, -1] = 2

        name = "natural_vertical_plate_surrogate"

    return CaseData(
        name,
        x,
        y,
        temperature,
        u,
        v,
        marker,
        pe,
        1 / pe,
        dx,
        dy
    )


# ============================================================
# 8. BOUNDARY CONDITIONS
# ============================================================

def apply_bc(case):

    t = case.temperature

    if case.name == "forced_flat_plate":

        t[:, 0] = 0

        t[-1, :] = 0

        heated = (
            case.x[0, :]
            >= 0.25 * case.x[0, -1]
        )

        t[0, :] = np.where(
            heated, 1, 0
        )

        t[1:-1, -1] = t[1:-1, -2]

    elif case.name == "forced_pipe_channel":

        t[:, 0] = 0

        t[0, :] = 1

        t[-1, :] = 1

        t[1:-1, -1] = t[1:-1, -2]

    else:

        t[:, 0] = 1

        t[:, -1] = 0

        t[0, 1:-1] = t[1, 1:-1]

        t[-1, 1:-1] = t[-2, 1:-1]


# ============================================================
# 9. TEMPERATURE AND HEAT FLUX
# ============================================================

def outlet_or_mean(case):

    if case.name == "natural_vertical_plate_surrogate":

        return float(
            np.mean(
                case.temperature[1:-1, 1:-1]
            )
        )

    weights = np.maximum(
        case.u[1:-1, -1],
        0
    )

    if np.sum(weights) < 1e-14:

        return float(
            np.mean(
                case.temperature[1:-1, -1]
            )
        )

    return float(
        np.average(
            case.temperature[1:-1, -1],
            weights=weights
        )
    )


def average_wall_flux(case):

    t = case.temperature

    if case.name == "forced_flat_plate":

        heated = (
            case.x[0, 1:-1]
            >= 0.25 * case.x[0, -1]
        )

        flux = -(
            t[1, 1:-1] - t[0, 1:-1]
        ) / case.dy

        return float(
            np.mean(flux[heated])
        )

    if case.name == "forced_pipe_channel":

        bottom = np.abs(
            (
                t[0, 1:-1]
                - t[1, 1:-1]
            ) / case.dy
        )

        top = np.abs(
            (
                t[-1, 1:-1]
                - t[-2, 1:-1]
            ) / case.dy
        )

        return float(
            np.mean(
                np.concatenate((bottom, top))
            )
        )

    flux = -(
        t[1:-1, 1]
        - t[1:-1, 0]
    ) / case.dx

    return float(
        np.mean(flux)
    )


# ============================================================
# 10. MAIN NUMERICAL SOLVER
# ============================================================

def run_case(case, iterations, output_every):

    apply_bc(case)

    max_u = max(
        float(np.max(np.abs(case.u))),
        1e-12
    )

    max_v = max(
        float(np.max(np.abs(case.v))),
        1e-12
    )

    dt_convective = 0.35 * min(
        case.dx / max_u,
        case.dy / max_v
    )

    dt_diffusive = (
        0.20
        * min(case.dx**2, case.dy**2)
        / case.alpha
    )

    dt = min(
        dt_convective,
        dt_diffusive
    )

    history = []

    residual = float("inf")

    for iteration in range(
        1,
        iterations + 1
    ):

        old = case.temperature.copy()

        convection = (

            case.u * upwind_derivative_x(
                old,
                case.u,
                case.dx
            )

            +

            case.v * upwind_derivative_y(
                old,
                case.v,
                case.dy
            )
        )

        updated = old + dt * (

            -convection

            +

            case.alpha * laplacian(
                old,
                case.dx,
                case.dy
            )
        )

        case.temperature[1:-1, 1:-1] = np.clip(
            updated[1:-1, 1:-1],
            -0.05,
            1.05
        )

        apply_bc(case)

        residual = float(
            np.sqrt(
                np.mean(
                    (
                        case.temperature[1:-1, 1:-1]
                        - old[1:-1, 1:-1]
                    )**2
                )
            )
        )

        if (
            iteration == 1
            or iteration == iterations
            or iteration % output_every == 0
        ):

            outlet = outlet_or_mean(case)

            flux = average_wall_flux(case)

            history.append([
                iteration,
                residual,
                outlet,
                flux,
                dt
            ])

            print(
                f"Case: {case.name} | "
                f"Iteration: {iteration} | "
                f"Residual: {residual:.6e} | "
                f"Temperature: {outlet:.6e} | "
                f"Wall Flux: {flux:.6e}"
            )

        require_finite(
            case.temperature
        )

        if residual < 1e-9 and iteration > 100:
            break

    # Velocity magnitude

    speed = np.sqrt(
        case.u**2 + case.v**2
    )

    # Local Peclet number

    local_pe = speed / case.alpha

    # VTK field output

    write_structured_grid_vtk(
        f"{case.name}.vtk",
        case.x,
        case.y,
        {
            "temperature": case.temperature,
            "speed": speed,
            "wall_or_boundary_marker": case.marker,
            "local_peclet_speed_based": local_pe
        },
        {
            "velocity": (
                case.u,
                case.v
            )
        },
        case.name
    )

    length = case.x[0, -1]

    height = case.y[-1, 0]

    if case.name in [
        "forced_flat_plate",
        "forced_pipe_channel"
    ]:

        lines = [

            (
                np.array([0, length]),
                np.array([0, 0])
            ),

            (
                np.array([0, length]),
                np.array([height, height])
            )
        ]

    else:

        lines = [

            (
                np.array([0, 0]),
                np.array([0, height])
            ),

            (
                np.array([length, length]),
                np.array([0, height])
            )
        ]

    write_multiline_vtk(
        f"{case.name}_boundaries.vtk",
        lines,
        f"{case.name} boundaries"
    )

    # History CSV

    with open(
        f"{case.name}_history.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "iteration",
            "residual_l2",
            "outlet_or_mean_temperature",
            "average_wall_flux",
            "dt"
        ])

        writer.writerows(history)

    # Temperature profiles

    i_mid = case.temperature.shape[1] // 2

    j_mid = case.temperature.shape[0] // 2

    with open(
        f"{case.name}_profiles.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "x",
            "y",
            "temperature",
            "u",
            "v",
            "speed"
        ])

        for j in range(
            case.temperature.shape[0]
        ):

            writer.writerow([
                case.x[j, i_mid],
                case.y[j, i_mid],
                case.temperature[j, i_mid],
                case.u[j, i_mid],
                case.v[j, i_mid],
                speed[j, i_mid]
            ])

        for i in range(
            case.temperature.shape[1]
        ):

            writer.writerow([
                case.x[j_mid, i],
                case.y[j_mid, i],
                case.temperature[j_mid, i],
                case.u[j_mid, i],
                case.v[j_mid, i],
                speed[j_mid, i]
            ])

    return [
        case.name,
        case.temperature.shape[1],
        case.temperature.shape[0],
        case.pe,
        dt,
        residual,
        outlet_or_mean(case),
        average_wall_flux(case)
    ]


# ============================================================
# 11. SOLVER MAIN FUNCTION
# ============================================================

def main():

    mode = ARGS[0]

    nx = int(ARGS[1])
    ny = int(ARGS[2])

    pe = float(ARGS[3])

    iterations = int(ARGS[4])

    output_every = int(ARGS[5])

    if (
        nx < 21
        or ny < 21
        or pe < 1
        or iterations < 1
        or output_every < 1
    ):

        raise ValueError(
            "Invalid simulation parameters"
        )

    aliases = {

        "forced_plate": "forced_plate",
        "flat": "forced_plate",
        "forced_flat_plate": "forced_plate",

        "forced_pipe": "forced_pipe",
        "pipe": "forced_pipe",
        "forced_pipe_channel": "forced_pipe",

        "natural_plate": "natural_plate",
        "natural": "natural_plate",
        "natural_vertical_plate_surrogate": "natural_plate"
    }

    modes = (

        [
            "forced_plate",
            "forced_pipe",
            "natural_plate"
        ]

        if mode == "all"

        else [
            aliases.get(mode, "")
        ]
    )

    if not all(modes):

        raise ValueError(
            "Invalid simulation mode"
        )

    rows = []

    for selected in modes:

        case = initialize_case(
            selected,
            nx,
            ny,
            pe
        )

        rows.append(
            run_case(
                case,
                iterations,
                output_every
            )
        )

    # Summary CSV

    with open(
        "convective_transport_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "case",
            "nx",
            "ny",
            "Pe",
            "dt",
            "final_residual",
            "outlet_or_mean_temperature",
            "average_wall_flux"
        ])

        writer.writerows(rows)

    # Summary TXT

    with open(
        "convective_transport_summary.txt",
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            "Generic 2D convection-diffusion transport equation\n"
        )

        handle.write(
            "Forced flat plate: prescribed external flow.\n"
        )

        handle.write(
            "Forced pipe: prescribed parabolic internal flow.\n"
        )

        handle.write(
            "Natural plate: prescribed recirculating flow.\n"
        )

        handle.write(
            "Natural convection is not a coupled Navier-Stokes solution.\n"
        )

    print(
        "All output files generated successfully."
    )


# ============================================================
# 12. SELECT SIMULATION PARAMETERS
# ============================================================

RUN_MODE = "quick"

# Options: quick, lab, custom

mode = "all"

# Options: all, forced_plate, forced_pipe, natural_plate

nx = 121
ny = 61

Pe = 1000.0

iterations = 3000

output_every = 300


QUICK_ARGS = [
    "all", 25, 21, 100, 8, 8
]

LAB_ARGS = [
    "all", 121, 61, 200, 3000, 300
]

CUSTOM_ARGS = [
    mode,
    nx,
    ny,
    Pe,
    iterations,
    output_every
]


ARGS = (
    QUICK_ARGS
    if RUN_MODE == "quick"
    else LAB_ARGS
    if RUN_MODE == "lab"
    else CUSTOM_ARGS
)

print("Simulation mode:", RUN_MODE)
print("Simulation parameters:", ARGS)


# ============================================================
# 13. SAVE SOLVER SCRIPT AND EXECUTE
# ============================================================

# Run the numerical solver in the current process.
# Output files are created inside OUTPUT_DIR.

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

import os

previous_directory = Path.cwd()

try:

    os.chdir(OUTPUT_DIR)

    main()

finally:

    os.chdir(previous_directory)


OUT = OUTPUT_DIR


# ============================================================
# 14. READ SOLVER SUMMARY
# ============================================================

summary_text = show_summary(
    OUT / "convective_transport_summary.txt"
)


# ============================================================
# 15. PLOT CASE COMPARISON
# ============================================================

summary = read_csv(
    OUT / "convective_transport_summary.csv"
)

labels = [
    str(value)
    for value in summary["case"]
]

plt.figure(figsize=(8, 4))

plt.bar(
    labels,
    summary["outlet_or_mean_temperature"]
)

plt.ylabel(
    "Outlet or mean temperature"
)

plt.title(
    "Convection-Diffusion Case Comparison"
)

plt.xticks(rotation=20)

plt.grid(
    True,
    axis="y"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. PLOT TEMPERATURE PROFILES
# ============================================================

plt.figure(figsize=(8, 4))

for profile_file in sorted(
    OUT.glob("*_profiles.csv")
):

    profiles = read_csv(profile_file)

    vertical_mask = np.isclose(
        profiles["x"],
        profiles["x"][0]
    )

    plt.plot(
        profiles["temperature"][vertical_mask],
        profiles["y"][vertical_mask],
        label=profile_file.stem.replace(
            "_profiles", ""
        )
    )

plt.xlabel("Temperature")
plt.ylabel("Profile coordinate y")

plt.title(
    "Extracted Temperature Profiles"
)

plt.legend(fontsize=8)

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 17. DISPLAY TEMPERATURE CONTOUR
# ============================================================

plot_vtk_scalar(
    OUT / "forced_flat_plate.vtk",
    "temperature"
)


# ============================================================
# 18. LIST GENERATED FILES
# ============================================================

files = list_outputs(OUT)


# ============================================================
# 19. CREATE RESULTS ZIP
# ============================================================

def make_results_zip(
    output_dir,
    experiment_number
):

    output_dir = Path(output_dir)

    archive = (
        output_dir.parent
        / f"experiment_{experiment_number:02d}_results.zip"
    )

    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as handle:

        for path in sorted(
            output_dir.rglob("*")
        ):

            if path.is_file():

                handle.write(
                    path,
                    path.relative_to(output_dir)
                )

    print(
        f"Results archive: {archive}"
    )

    return archive


# ============================================================
# 20. DOWNLOAD RESULTS
# ============================================================

DOWNLOAD_RESULTS = True

RESULTS_ZIP = make_results_zip(
    OUT,
    EXPERIMENT
)

if DOWNLOAD_RESULTS:

    if IN_COLAB:

        colab_files.download(
            str(RESULTS_ZIP)
        )

    else:

        print(
            "Local notebook mode."
        )

        print(
            "Archive saved at:",
            RESULTS_ZIP
        )


print("\nEXPERIMENT 11 COMPLETED SUCCESSFULLY")
