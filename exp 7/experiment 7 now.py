#@title 1. Prepare the self-contained Colab runtime

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

IN_COLAB = False
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    colab_files = None


def run_solver(script_path, arguments, output_dir, timeout=1800):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script_path), *[str(value) for value in arguments]]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=output_dir, capture_output=True, text=True, timeout=timeout)
    (output_dir / "solver_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "solver_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"Solver exited with code {completed.returncode}")
    if completed.stderr.strip():
        print(completed.stderr.strip())
    return output_dir


def read_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(data)


def show_summary(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    display(Markdown(f"### Solver summary\n```text\n{text}\n```"))
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
    table = "\n".join(f"| `{name}` | {size} |" for name, size in rows)
    display(Markdown(f"### Generated files\n| File | Size |\n|---|---:|\n{table}"))
    return rows


def read_structured_vtk(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    dims_index = next(i for i, line in enumerate(lines) if line.startswith("DIMENSIONS "))
    _, nx, ny, nz = lines[dims_index].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    if nz != 1:
        raise ValueError("Only 2D structured VTK files are supported by this notebook preview")
    points_index = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    npoints = int(lines[points_index].split()[1])
    points = np.array([[float(v) for v in lines[points_index + 1 + k].split()[:3]] for k in range(npoints)])
    x = points[:, 0].reshape(ny, nx)
    y = points[:, 1].reshape(ny, nx)
    data_index = next(i for i, line in enumerate(lines) if line.startswith("POINT_DATA ")) + 1
    scalars = {}
    vectors = {}
    i = data_index
    while i < len(lines):
        if lines[i].startswith("SCALARS "):
            name = lines[i].split()[1]
            i += 2
            values = np.array([float(lines[i + k].split()[0]) for k in range(npoints)])
            scalars[name] = values.reshape(ny, nx)
            i += npoints
        elif lines[i].startswith("VECTORS "):
            name = lines[i].split()[1]
            i += 1
            values = np.array([[float(v) for v in lines[i + k].split()[:3]] for k in range(npoints)])
            vectors[name] = (values[:, 0].reshape(ny, nx), values[:, 1].reshape(ny, nx))
            i += npoints
        else:
            i += 1
    return x, y, scalars, vectors


def plot_vtk_scalar(path, preferred_field=None):
    path = Path(path)
    if not path.is_file():
        print(f"VTK preview skipped: {path.name} was not generated")
        return
    x, y, scalars, _ = read_structured_vtk(path)
    if not scalars:
        print("VTK preview skipped: no scalar field found")
        return
    field = preferred_field if preferred_field in scalars else next(iter(scalars))
    plt.figure(figsize=(8, 4.8))
    image = plt.contourf(x, y, scalars[field], levels=30)
    plt.colorbar(image, label=field.replace("_", " "))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{field.replace('_', ' ').title()} - inline field preview")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def make_results_zip(output_dir, experiment_number):
    output_dir = Path(output_dir)
    archive = output_dir.parent / f"experiment_{experiment_number:02d}_results.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir))
    print(f"Results archive: {archive} ({archive.stat().st_size / 1024:.1f} KB)")
    return archive


EXPERIMENT = 8
ROOT = Path("/content/cfd_lab_exp08") if IN_COLAB else Path.cwd() / "_colab_runtime_exp08"
CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "outputs"
CODE_DIR.mkdir(parents=True, exist_ok=True)
(CODE_DIR / "common.py").write_text('from __future__ import annotations\n\nimport csv\nimport math\nfrom pathlib import Path\nfrom typing import Iterable, Mapping\n\nimport numpy as np\n\n\ndef trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:\n    y = np.asarray(values, dtype=float).ravel()\n    x = np.asarray(coordinates, dtype=float).ravel()\n    if y.size != x.size or y.size < 2:\n        raise ValueError("trapezoidal integration requires equally sized arrays with at least two points")\n    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))\n\n\ndef write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:\n    with Path(path).open("w", newline="", encoding="utf-8") as handle:\n        writer = csv.writer(handle)\n        writer.writerow(list(header))\n        writer.writerows(rows)\n\n\ndef mesh_xy(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:\n    x = np.linspace(x0, x0 + lx, nx)\n    y = np.linspace(y0, y0 + ly, ny)\n    return np.meshgrid(x, y)\n\n\ndef write_structured_grid_vtk(\n    path: str | Path,\n    x: np.ndarray,\n    y: np.ndarray,\n    scalars: Mapping[str, np.ndarray] | None = None,\n    vectors: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,\n    title: str = "CFD lab output",\n) -> None:\n    x = np.asarray(x, dtype=float)\n    y = np.asarray(y, dtype=float)\n    if x.shape != y.shape or x.ndim != 2:\n        raise ValueError("x and y must be equally shaped 2D arrays")\n    ny, nx = x.shape\n    scalar_map = dict(scalars or {})\n    vector_map = dict(vectors or {})\n    for name, values in scalar_map.items():\n        if np.asarray(values).shape != x.shape:\n            raise ValueError(f"scalar {name} has shape {np.asarray(values).shape}, expected {x.shape}")\n    for name, values in vector_map.items():\n        if len(values) != 2 or np.asarray(values[0]).shape != x.shape or np.asarray(values[1]).shape != x.shape:\n            raise ValueError(f"vector {name} has incompatible shape")\n    with Path(path).open("w", encoding="utf-8") as handle:\n        handle.write("# vtk DataFile Version 3.0\\n")\n        handle.write(f"{title}\\n")\n        handle.write("ASCII\\n")\n        handle.write("DATASET STRUCTURED_GRID\\n")\n        handle.write(f"DIMENSIONS {nx} {ny} 1\\n")\n        handle.write(f"POINTS {nx * ny} float\\n")\n        for j in range(ny):\n            for i in range(nx):\n                handle.write(f"{x[j, i]:.12e} {y[j, i]:.12e} 0\\n")\n        handle.write(f"POINT_DATA {nx * ny}\\n")\n        for name, values in scalar_map.items():\n            data = np.asarray(values, dtype=float)\n            handle.write(f"SCALARS {name} float 1\\n")\n            handle.write("LOOKUP_TABLE default\\n")\n            for value in data.ravel(order="C"):\n                handle.write(f"{value:.12e}\\n")\n        for name, values in vector_map.items():\n            vx = np.asarray(values[0], dtype=float)\n            vy = np.asarray(values[1], dtype=float)\n            handle.write(f"VECTORS {name} float\\n")\n            for ax, ay in zip(vx.ravel(order="C"), vy.ravel(order="C")):\n                handle.write(f"{ax:.12e} {ay:.12e} 0\\n")\n\n\ndef write_polyline_vtk(path: str | Path, x: np.ndarray, y: np.ndarray, title: str = "polyline", close: bool = False) -> None:\n    x = np.asarray(x, dtype=float).ravel()\n    y = np.asarray(y, dtype=float).ravel()\n    if x.size != y.size or x.size < 2:\n        raise ValueError("polyline requires equally sized x and y arrays")\n    ids = list(range(x.size))\n    if close:\n        ids.append(0)\n    with Path(path).open("w", encoding="utf-8") as handle:\n        handle.write("# vtk DataFile Version 3.0\\n")\n        handle.write(f"{title}\\n")\n        handle.write("ASCII\\n")\n        handle.write("DATASET POLYDATA\\n")\n        handle.write(f"POINTS {x.size} float\\n")\n        for xi, yi in zip(x, y):\n            handle.write(f"{xi:.12e} {yi:.12e} 0\\n")\n        handle.write(f"LINES 1 {len(ids) + 1}\\n")\n        handle.write(f"{len(ids)} {\' \'.join(str(i) for i in ids)}\\n")\n\n\ndef write_multiline_vtk(path: str | Path, lines: list[tuple[np.ndarray, np.ndarray]], title: str = "lines") -> None:\n    points: list[tuple[float, float]] = []\n    connectivity: list[list[int]] = []\n    for x, y in lines:\n        xa = np.asarray(x, dtype=float).ravel()\n        ya = np.asarray(y, dtype=float).ravel()\n        if xa.size != ya.size or xa.size < 2:\n            raise ValueError("each line needs equally sized x and y arrays")\n        ids = list(range(len(points), len(points) + xa.size))\n        points.extend(zip(xa.tolist(), ya.tolist()))\n        connectivity.append(ids)\n    size = sum(len(ids) + 1 for ids in connectivity)\n    with Path(path).open("w", encoding="utf-8") as handle:\n        handle.write("# vtk DataFile Version 3.0\\n")\n        handle.write(f"{title}\\n")\n        handle.write("ASCII\\n")\n        handle.write("DATASET POLYDATA\\n")\n        handle.write(f"POINTS {len(points)} float\\n")\n        for xi, yi in points:\n            handle.write(f"{xi:.12e} {yi:.12e} 0\\n")\n        handle.write(f"LINES {len(connectivity)} {size}\\n")\n        for ids in connectivity:\n            handle.write(f"{len(ids)} {\' \'.join(str(i) for i in ids)}\\n")\n\n\ndef write_points_vtk(path: str | Path, x: np.ndarray, y: np.ndarray, scalars: Mapping[str, np.ndarray] | None = None, title: str = "points") -> None:\n    xa = np.asarray(x, dtype=float).ravel()\n    ya = np.asarray(y, dtype=float).ravel()\n    if xa.size != ya.size:\n        raise ValueError("point arrays must have equal size")\n    scalar_map = dict(scalars or {})\n    with Path(path).open("w", encoding="utf-8") as handle:\n        handle.write("# vtk DataFile Version 3.0\\n")\n        handle.write(f"{title}\\n")\n        handle.write("ASCII\\n")\n        handle.write("DATASET POLYDATA\\n")\n        handle.write(f"POINTS {xa.size} float\\n")\n        for xi, yi in zip(xa, ya):\n            handle.write(f"{xi:.12e} {yi:.12e} 0\\n")\n        handle.write(f"VERTICES {xa.size} {2 * xa.size}\\n")\n        for i in range(xa.size):\n            handle.write(f"1 {i}\\n")\n        if scalar_map:\n            handle.write(f"POINT_DATA {xa.size}\\n")\n            for name, values in scalar_map.items():\n                data = np.asarray(values, dtype=float).ravel()\n                if data.size != xa.size:\n                    raise ValueError(f"scalar {name} has wrong size")\n                handle.write(f"SCALARS {name} float 1\\n")\n                handle.write("LOOKUP_TABLE default\\n")\n                for value in data:\n                    handle.write(f"{value:.12e}\\n")\n\n\ndef gradient(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:\n    gy, gx = np.gradient(field, dy, dx, edge_order=2)\n    return gx, gy\n\n\ndef laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:\n    out = np.zeros_like(field)\n    out[1:-1, 1:-1] = (\n        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx * dx)\n        + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dy * dy)\n    )\n    return out\n\n\ndef upwind_derivative_x(field: np.ndarray, velocity: np.ndarray, dx: float) -> np.ndarray:\n    out = np.zeros_like(field)\n    backward = (field[1:-1, 1:-1] - field[1:-1, :-2]) / dx\n    forward = (field[1:-1, 2:] - field[1:-1, 1:-1]) / dx\n    out[1:-1, 1:-1] = np.where(velocity[1:-1, 1:-1] >= 0.0, backward, forward)\n    return out\n\n\ndef upwind_derivative_y(field: np.ndarray, velocity: np.ndarray, dy: float) -> np.ndarray:\n    out = np.zeros_like(field)\n    backward = (field[1:-1, 1:-1] - field[:-2, 1:-1]) / dy\n    forward = (field[2:, 1:-1] - field[1:-1, 1:-1]) / dy\n    out[1:-1, 1:-1] = np.where(velocity[1:-1, 1:-1] >= 0.0, backward, forward)\n    return out\n\n\ndef poisson_jacobi(phi: np.ndarray, rhs: np.ndarray, dx: float, dy: float, iterations: int, boundary: str = "neumann", mask: np.ndarray | None = None) -> np.ndarray:\n    dx2 = dx * dx\n    dy2 = dy * dy\n    denom = 2.0 * (dx2 + dy2)\n    for _ in range(iterations):\n        old = phi.copy()\n        phi[1:-1, 1:-1] = (\n            (old[1:-1, 2:] + old[1:-1, :-2]) * dy2\n            + (old[2:, 1:-1] + old[:-2, 1:-1]) * dx2\n            - rhs[1:-1, 1:-1] * dx2 * dy2\n        ) / denom\n        if mask is not None:\n            phi[mask > 0.5] = 0.0\n        if boundary == "neumann":\n            phi[:, 0] = phi[:, 1]\n            phi[:, -1] = 0.0\n            phi[0, :] = phi[1, :]\n            phi[-1, :] = phi[-2, :]\n        elif boundary == "closed":\n            phi[:, 0] = phi[:, 1]\n            phi[:, -1] = phi[:, -2]\n            phi[0, :] = phi[1, :]\n            phi[-1, :] = phi[-2, :]\n            phi -= np.mean(phi)\n        else:\n            raise ValueError("unsupported pressure boundary")\n    return phi\n\n\ndef tdma(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:\n    a = np.asarray(a, dtype=float).copy()\n    b = np.asarray(b, dtype=float).copy()\n    c = np.asarray(c, dtype=float).copy()\n    d = np.asarray(d, dtype=float).copy()\n    n = d.size\n    if n == 0 or a.size != n or b.size != n or c.size != n:\n        raise ValueError("TDMA arrays must have equal nonzero size")\n    for i in range(1, n):\n        if abs(b[i - 1]) < 1e-30:\n            raise FloatingPointError("zero TDMA pivot")\n        factor = a[i] / b[i - 1]\n        b[i] -= factor * c[i - 1]\n        d[i] -= factor * d[i - 1]\n    x = np.empty(n, dtype=float)\n    if abs(b[-1]) < 1e-30:\n        raise FloatingPointError("zero TDMA pivot")\n    x[-1] = d[-1] / b[-1]\n    for i in range(n - 2, -1, -1):\n        if abs(b[i]) < 1e-30:\n            raise FloatingPointError("zero TDMA pivot")\n        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]\n    return x\n\n\ndef relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:\n    a = np.asarray(numerical, dtype=float)\n    b = np.asarray(reference, dtype=float)\n    denominator = float(np.linalg.norm(b.ravel()))\n    numerator = float(np.linalg.norm((a - b).ravel()))\n    return numerator / max(denominator, 1e-30)\n\n\ndef require_finite(*arrays: np.ndarray) -> None:\n    for array in arrays:\n        if not np.all(np.isfinite(array)):\n            raise FloatingPointError("non-finite numerical field detected")\n\n\ndef positive_float(value: str, name: str) -> float:\n    result = float(value)\n    if not math.isfinite(result) or result <= 0.0:\n        raise ValueError(f"{name} must be positive and finite")\n    return result\n\n\ndef positive_int(value: str, name: str, minimum: int = 1) -> int:\n    result = int(value)\n    if result < minimum:\n        raise ValueError(f"{name} must be at least {minimum}")\n    return result\n', encoding="utf-8")
(CODE_DIR / 'exp08_rayleigh_taylor_vof.py').write_text('from __future__ import annotations\n\nimport csv\nimport math\nimport sys\n\nimport numpy as np\n\nfrom common import gradient, laplacian, poisson_jacobi, require_finite, upwind_derivative_x, upwind_derivative_y, write_multiline_vtk, write_points_vtk, write_structured_grid_vtk\n\n\ndef apply_velocity_bc(u: np.ndarray, v: np.ndarray) -> None:\n    u[:, 0] = 0.0\n    u[:, -1] = 0.0\n    u[0, :] = 0.0\n    u[-1, :] = 0.0\n    v[:, 0] = 0.0\n    v[:, -1] = 0.0\n    v[0, :] = 0.0\n    v[-1, :] = 0.0\n\n\ndef advect_conservative(volume_fraction: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float, dx: float, dy: float) -> np.ndarray:\n    ny, nx = volume_fraction.shape\n    flux_x = np.zeros((ny, nx + 1), dtype=float)\n    flux_y = np.zeros((ny + 1, nx), dtype=float)\n    u_face = 0.5 * (u[:, :-1] + u[:, 1:])\n    v_face = 0.5 * (v[:-1, :] + v[1:, :])\n    flux_x[:, 1:nx] = u_face * np.where(u_face >= 0.0, volume_fraction[:, :-1], volume_fraction[:, 1:])\n    flux_y[1:ny, :] = v_face * np.where(v_face >= 0.0, volume_fraction[:-1, :], volume_fraction[1:, :])\n    updated = volume_fraction - dt * ((flux_x[:, 1:] - flux_x[:, :-1]) / dx + (flux_y[1:, :] - flux_y[:-1, :]) / dy)\n    target_sum = float(np.sum(volume_fraction))\n    updated = np.clip(updated, 0.0, 1.0)\n    for _ in range(4):\n        difference = target_sum - float(np.sum(updated))\n        if abs(difference) < 1e-12:\n            break\n        weights = updated * (1.0 - updated) + 1e-6\n        updated += difference * weights / np.sum(weights)\n        updated = np.clip(updated, 0.0, 1.0)\n    return updated\n\n\ndef interface_coordinates(volume_fraction: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:\n    indices = np.argmin(np.abs(volume_fraction - 0.5), axis=0)\n    return x[0, :].copy(), y[indices, np.arange(volume_fraction.shape[1])]\n\n\ndef main() -> int:\n    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 96\n    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 192\n    atwood = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5\n    gravity = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0\n    nu = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0008\n    end_time = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0\n    max_steps = int(sys.argv[7]) if len(sys.argv) > 7 else 800\n    poisson_iterations = int(sys.argv[8]) if len(sys.argv) > 8 else 60\n    output_every = int(sys.argv[9]) if len(sys.argv) > 9 else 200\n    if nx < 24 or ny < 48 or not 0.0 < atwood < 0.95 or gravity <= 0.0 or nu <= 0.0 or end_time <= 0.0 or max_steps < 1 or poisson_iterations < 1 or output_every < 1:\n        raise ValueError("usage: python exp08_rayleigh_taylor_vof.py nx ny Atwood gravity nu tEnd maxSteps poissonIterations outputEvery")\n    length = 0.5\n    height = 1.0\n    x1 = np.linspace(0.0, length, nx)\n    y1 = np.linspace(0.0, height, ny)\n    x, y = np.meshgrid(x1, y1)\n    dx = x1[1] - x1[0]\n    dy = y1[1] - y1[0]\n    interface = 0.5 * height + 0.02 * height * np.cos(2.0 * math.pi * x / length)\n    smoothing = 1.5 * dy\n    volume_fraction = 0.5 * (1.0 + np.tanh((y - interface) / smoothing))\n    u = np.zeros_like(volume_fraction)\n    v = np.zeros_like(volume_fraction)\n    p = np.zeros_like(volume_fraction)\n    apply_velocity_bc(u, v)\n    rho_heavy = 1.0 + atwood\n    rho_light = 1.0 - atwood\n    initial_mass = float(np.sum(volume_fraction) * dx * dy)\n    history: list[list[float]] = []\n    time = 0.0\n    completed_steps = 0\n    for step in range(1, max_steps + 1):\n        speed_max = max(float(np.max(np.sqrt(u * u + v * v))), 1e-8)\n        dt_convective = 0.25 * min(dx, dy) / speed_max\n        dt_viscous = 0.15 * min(dx * dx, dy * dy) / nu\n        dt = min(dt_convective, dt_viscous, end_time - time)\n        if dt <= 1e-14:\n            break\n        du_dx = upwind_derivative_x(u, u, dx)\n        du_dy = upwind_derivative_y(u, v, dy)\n        dv_dx = upwind_derivative_x(v, u, dx)\n        dv_dy = upwind_derivative_y(v, v, dy)\n        buoyancy = -gravity * atwood * (2.0 * volume_fraction - 1.0)\n        us = u + dt * (-(u * du_dx + v * du_dy) + nu * laplacian(u, dx, dy))\n        vs = v + dt * (-(u * dv_dx + v * dv_dy) + nu * laplacian(v, dx, dy) + buoyancy)\n        apply_velocity_bc(us, vs)\n        rhs = np.zeros_like(p)\n        rhs[1:-1, 1:-1] = ((us[1:-1, 2:] - us[1:-1, :-2]) / (2.0 * dx) + (vs[2:, 1:-1] - vs[:-2, 1:-1]) / (2.0 * dy)) / dt\n        rhs -= np.mean(rhs)\n        phi = poisson_jacobi(np.zeros_like(p), rhs, dx, dy, poisson_iterations, boundary="closed")\n        gx, gy = gradient(phi, dx, dy)\n        u = us - dt * gx\n        v = vs - dt * gy\n        p += phi\n        apply_velocity_bc(u, v)\n        volume_fraction = advect_conservative(volume_fraction, u, v, dt, dx, dy)\n        time += dt\n        completed_steps = step\n        density = rho_light + (rho_heavy - rho_light) * volume_fraction\n        divergence = np.zeros_like(u)\n        divergence[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)\n        div_l2 = float(np.sqrt(np.mean(divergence[1:-1, 1:-1] ** 2)))\n        interface_x, interface_y = interface_coordinates(volume_fraction, x, y)\n        spike = float(np.min(interface_y))\n        bubble = float(np.max(interface_y))\n        mixing_width = bubble - spike\n        mass = float(np.sum(volume_fraction) * dx * dy)\n        mass_error = (mass - initial_mass) / max(initial_mass, 1e-30)\n        history.append([step, time, dt, div_l2, spike, bubble, mixing_width, mass, mass_error, float(np.min(volume_fraction)), float(np.max(volume_fraction))])\n        require_finite(u, v, p, volume_fraction)\n        if step % output_every == 0 or time >= end_time - 1e-14 or step == max_steps:\n            vorticity = np.zeros_like(u)\n            vorticity[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)\n            write_structured_grid_vtk(f"rti_{step:06d}.vtk", x, y, {"heavy_fluid_fraction": volume_fraction, "density": density, "pressure": p, "vorticity": vorticity, "speed": np.sqrt(u * u + v * v)}, {"velocity": (u, v)}, "Rayleigh-Taylor instability")\n            write_points_vtk(f"rti_interface_{step:06d}.vtk", interface_x, interface_y, title="RTI interface")\n            print(f"step={step} time={time:.6e} dt={dt:.6e} divergence_l2={div_l2:.6e} mixing_width={mixing_width:.6e} mass_error={mass_error:.6e}")\n        if time >= end_time - 1e-14:\n            break\n    density = rho_light + (rho_heavy - rho_light) * volume_fraction\n    vorticity = np.zeros_like(u)\n    vorticity[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)\n    write_structured_grid_vtk("rti_final.vtk", x, y, {"heavy_fluid_fraction": volume_fraction, "density": density, "pressure": p, "vorticity": vorticity, "speed": np.sqrt(u * u + v * v)}, {"velocity": (u, v)}, "Rayleigh-Taylor instability final")\n    interface_x, interface_y = interface_coordinates(volume_fraction, x, y)\n    write_points_vtk("rti_interface_final.vtk", interface_x, interface_y, title="RTI final interface")\n    write_multiline_vtk("rti_domain_boundary.vtk", [(np.array([0.0, length]), np.array([0.0, 0.0])), (np.array([length, length]), np.array([0.0, height])), (np.array([length, 0.0]), np.array([height, height])), (np.array([0.0, 0.0]), np.array([height, 0.0]))], "RTI domain boundary")\n    with open("rti_history.csv", "w", newline="", encoding="utf-8") as handle:\n        writer = csv.writer(handle)\n        writer.writerow(["step", "time", "dt", "divergence_l2", "spike_position", "bubble_position", "mixing_width", "heavy_fluid_mass", "relative_mass_error", "minimum_fraction", "maximum_fraction"])\n        writer.writerows(history)\n    final_mass = float(np.sum(volume_fraction) * dx * dy)\n    with open("rti_summary.txt", "w", encoding="utf-8") as handle:\n        handle.write("Teaching-level algebraic VOF-style Rayleigh-Taylor solver with Boussinesq projection flow\\n")\n        handle.write(f"nx {nx}\\nny {ny}\\nAtwood {atwood:.12e}\\ngravity {gravity:.12e}\\nnu {nu:.12e}\\n")\n        handle.write(f"rho_heavy {rho_heavy:.12e}\\nrho_light {rho_light:.12e}\\ncompleted_steps {completed_steps}\\nfinal_time {time:.12e}\\n")\n        handle.write(f"initial_heavy_fraction_integral {initial_mass:.12e}\\nfinal_heavy_fraction_integral {final_mass:.12e}\\nrelative_mass_error {(final_mass - initial_mass) / max(initial_mass, 1e-30):.12e}\\n")\n        handle.write(f"minimum_volume_fraction {float(np.min(volume_fraction)):.12e}\\nmaximum_volume_fraction {float(np.max(volume_fraction)):.12e}\\n")\n        handle.write("model_scope educational_boussinesq_vof_style_not_industrial_variable_density_vof\\n")\n    print("Wrote rti_final.vtk, rti_interface_final.vtk, rti_domain_boundary.vtk, rti_history.csv, rti_summary.txt")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n', encoding="utf-8")
SCRIPT = CODE_DIR / 'exp08_rayleigh_taylor_vof.py'
print("Runtime:", "Google Colab" if IN_COLAB else "local notebook test")
print("Working folder:", ROOT)
print("No package installation or external file upload is required.")
#@title 2. Select run mode and parameters
RUN_MODE = "quick" #@param ["quick", "lab", "custom"]

nx = 96 #@param {type:"integer"}
ny = 192 #@param {type:"integer"}

Atwood = 0.5 #@param {type:"number"}
gravity = 1.0 #@param {type:"number"}
nu = 0.0008 #@param {type:"number"}

t_end = 0.5 #@param {type:"number"}
max_steps = 500 #@param {type:"integer"}

poisson_iterations = 100 #@param {type:"integer"}
output_every = 20 #@param {type:"integer"}

QUICK_ARGS = [
    48, 96, 0.5, 1.0, 0.001,
    0.5, 500, 60, 20
]

LAB_ARGS = [
    96, 192, 0.5, 1.0, 0.0008,
    2.0, 3000, 150, 100
]

CUSTOM_ARGS = [
    nx, ny, Atwood, gravity, nu,
    t_end, max_steps, poisson_iterations, output_every
]

if RUN_MODE == "quick":
    ARGS = QUICK_ARGS
elif RUN_MODE == "lab":
    ARGS = LAB_ARGS
else:
    ARGS = CUSTOM_ARGS

print("Selected mode:", RUN_MODE)
print("Simulation parameters:", ARGS)
#@title 3. Run the numerical solver
OUT = run_solver(SCRIPT, ARGS, OUTPUT_DIR)

#@title 4. Read the solver summary
summary_text = show_summary(OUT / 'rti_summary.txt')
#@title 5. Plot the primary numerical results
history = read_csv(OUT / "rti_history.csv")
plt.figure(figsize=(7, 4))
plt.plot(history["time"], history["spike_position"], label="Spike")
plt.plot(history["time"], history["bubble_position"], label="Bubble")
plt.xlabel("Time")
plt.ylabel("Vertical position")
plt.title("Spike and bubble evolution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(history["time"], history["mixing_width"])
plt.xlabel("Time")
plt.ylabel("Mixing width")
plt.title("Mixing-layer growth")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.semilogy(history["time"], np.maximum(np.abs(history["relative_mass_error"]), 1e-18))
plt.xlabel("Time")
plt.ylabel("Absolute relative mass error")
plt.title("Volume-fraction conservation")
plt.grid(True)
plt.show()
#@title 6. Display the final field and generated files
plot_vtk_scalar(OUT / 'rti_final.vtk', 'heavy_fluid_fraction')
files = list_outputs(OUT)
#@title 7. Package and download the results
DOWNLOAD_RESULTS = False #@param {type:"boolean"}
RESULTS_ZIP = make_results_zip(OUT, 8)
if DOWNLOAD_RESULTS:
    if IN_COLAB:
        colab_files.download(str(RESULTS_ZIP))
    else:
        print("Local test mode. Archive retained at:", RESULTS_ZIP)
else:
    print("Set DOWNLOAD_RESULTS to True and run this cell when ready to download.")
