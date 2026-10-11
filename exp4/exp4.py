# ============================================================
# EXPERIMENT 4: FLOW PAST A CYLINDER USING FDM
# 2D incompressible Navier-Stokes equations
# Finite differences + pressure-projection method
# ============================================================

from pathlib import Path
import csv
import zipfile
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. INPUT PARAMETERS
# ------------------------------------------------------------
# Start with quick mode. Change to "lab" for a finer simulation.
RUN_MODE = "quick"  # Options: "quick", "lab", "custom"

if RUN_MODE == "quick":
    nx, ny = 121, 61
    steps = 900
    output_every = 5
    pressure_iterations = 35
elif RUN_MODE == "lab":
    nx, ny = 300, 81
    steps = 2400
    output_every = 4
    pressure_iterations = 45
else:
    nx, ny = 141, 71
    steps = 1400
    output_every = 5
    pressure_iterations = 40

Re = 100.0                 # Reynolds number based on cylinder diameter
U_in = 1.0                 # Uniform inlet speed
Lx, Ly = 25.0, 8.0         # Domain length and height, in cylinder diameters
D = 1.0
R = D / 2.0
cx, cy = 3.0, Ly / 2.0    # Cylinder center
nu = U_in * D / Re         # Kinematic viscosity

ROOT = Path.cwd() / "fdm_cylinder_exp04_results"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
# Clear old files so the archive only contains this run.
for old_file in OUT.iterdir():
    if old_file.is_file():
        old_file.unlink()

# ------------------------------------------------------------
# 2. BUILD THE GRID AND MARK THE CYLINDER
# ------------------------------------------------------------
# Arrays use [j, i] indexing: rows are y and columns are x.
x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y)
distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
solid = distance <= R
fluid = ~solid

if np.any(solid[:, 0]) or np.any(solid[:, -1]):
    raise ValueError("The cylinder must not touch the inlet or outlet.")

# Surface sampling points are placed just outside the solid mask.
theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
r_sample = R + 0.75 * max(dx, dy)
x_surface = cx + r_sample * np.cos(theta)
y_surface = cy + r_sample * np.sin(theta)

# ------------------------------------------------------------
# 3. SELECT A STABLE EXPLICIT TIME STEP
# ------------------------------------------------------------
# The time step is limited by both convection and diffusion.
dt_conv = 0.20 * min(dx, dy) / U_in
dt_diff = 0.20 / (nu * (1.0 / dx**2 + 1.0 / dy**2))
dt = min(dt_conv, dt_diff)
print("Experiment 4: cylinder flow using FDM")
print(f"Grid: {nx} x {ny}; Re = {Re:g}; nu = {nu:.5f}")
print(f"dx = {dx:.5f}; dy = {dy:.5f}; dt = {dt:.6f}")
print(f"Steps = {steps}; simulated time = {steps * dt:.3f}")

# ------------------------------------------------------------
# 4. INITIALIZE VELOCITY AND PRESSURE
# ------------------------------------------------------------
u = np.full((ny, nx), U_in, dtype=float)
v = np.zeros((ny, nx), dtype=float)
p = np.zeros((ny, nx), dtype=float)
u[solid] = 0.0
v[solid] = 0.0

# ------------------------------------------------------------
# 5. FINITE-DIFFERENCE OPERATORS
# ------------------------------------------------------------
def ddx(a):
    """First derivative in x: centered inside, one-sided at edges."""
    out = np.zeros_like(a)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
    out[:, 0] = (a[:, 1] - a[:, 0]) / dx
    out[:, -1] = (a[:, -1] - a[:, -2]) / dx
    return out

def ddy(a):
    """First derivative in y: centered inside, one-sided at edges."""
    out = np.zeros_like(a)
    out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dy)
    out[0, :] = (a[1, :] - a[0, :]) / dy
    out[-1, :] = (a[-1, :] - a[-2, :]) / dy
    return out

def laplacian(a):
    """Cartesian second derivative, used for viscous diffusion."""
    out = np.zeros_like(a)
    out[1:-1, 1:-1] = (
        (a[1:-1, 2:] - 2.0*a[1:-1, 1:-1] + a[1:-1, :-2]) / dx**2
        + (a[2:, 1:-1] - 2.0*a[1:-1, 1:-1] + a[:-2, 1:-1]) / dy**2
    )
    return out

def apply_velocity_boundaries(uu, vv):
    """Apply inlet/outlet, free-slip top/bottom, and no-slip cylinder."""
    # Inlet: fixed uniform flow entering from the left.
    uu[:, 0] = U_in
    vv[:, 0] = 0.0
    # Outlet: zero-gradient approximation lets the wake leave the domain.
    uu[:, -1] = uu[:, -2]
    vv[:, -1] = vv[:, -2]
    # Top and bottom: approximate free-slip boundaries.
    uu[0, :] = uu[1, :]
    uu[-1, :] = uu[-2, :]
    vv[0, :] = 0.0
    vv[-1, :] = 0.0
    # No-slip wall: both velocity components vanish in cylinder cells.
    uu[solid] = 0.0
    vv[solid] = 0.0
    return uu, vv

def solve_pressure(pp, rhs, iterations):
    """Jacobi solution of Laplacian(p) = rhs with approximate Neumann walls."""
    pp = pp.copy()
    for _ in range(iterations):
        old = pp.copy()
        center = old[1:-1, 1:-1]
        east = old[1:-1, 2:].copy()
        west = old[1:-1, :-2].copy()
        north = old[2:, 1:-1].copy()
        south = old[:-2, 1:-1].copy()
        # Where a neighboring point lies inside the cylinder, copy the
        # pressure from the fluid-side center as a simple wall treatment.
        east[solid[1:-1, 2:]] = center[solid[1:-1, 2:]]
        west[solid[1:-1, :-2]] = center[solid[1:-1, :-2]]
        north[solid[2:, 1:-1]] = center[solid[2:, 1:-1]]
        south[solid[:-2, 1:-1]] = center[solid[:-2, 1:-1]]
        updated = ((east + west) * dy**2 + (north + south) * dx**2
                   - rhs[1:-1, 1:-1] * dx**2 * dy**2) / (2.0*(dx**2 + dy**2))
        region = ~solid[1:-1, 1:-1]
        pp[1:-1, 1:-1][region] = updated[region]
        # Pressure boundary conditions: zero normal gradient except outlet reference.
        pp[:, 0] = pp[:, 1]
        pp[0, :] = pp[1, :]
        pp[-1, :] = pp[-2, :]
        pp[:, -1] = 0.0
        pp[solid] = old[solid]
    return pp

# Bilinear interpolation samples the computed pressure near the cylinder wall.
def sample_bilinear(field, xs, ys):
    """Bilinearly interpolate a grid field at physical coordinates."""
    fx = np.clip((xs - x[0]) / dx, 0.0, nx - 1.001)
    fy = np.clip((ys - y[0]) / dy, 0.0, ny - 1.001)
    i0, j0 = fx.astype(int), fy.astype(int)
    ax, ay = fx - i0, fy - j0
    return ((1-ax)*(1-ay)*field[j0, i0]
            + ax*(1-ay)*field[j0, i0+1]
            + (1-ax)*ay*field[j0+1, i0]
            + ax*ay*field[j0+1, i0+1])

# ------------------------------------------------------------
# 6. TIME LOOP: MOMENTUM PREDICTOR + PRESSURE CORRECTION
# ------------------------------------------------------------
history = []
latest_vort = np.zeros_like(u)
for step in range(1, steps + 1):
    # Enforce the boundary conditions before computing spatial derivatives.
    u, v = apply_velocity_boundaries(u, v)

    # Compute advection, pressure gradients, and viscous diffusion.
    dudx, dudy = ddx(u), ddy(u)
    dvdx, dvdy = ddx(v), ddy(v)
    lap_u, lap_v = laplacian(u), laplacian(v)

    # Predictor: advance momentum without the new pressure field.
    u_star = u + dt * (-u*dudx - v*dudy + nu*lap_u)
    v_star = v + dt * (-u*dvdx - v*dvdy + nu*lap_v)
    u_star, v_star = apply_velocity_boundaries(u_star, v_star)

    # Pressure is obtained from the Poisson equation so corrected velocity
    # is approximately divergence-free: Laplacian(p) = div(u_star) / dt.
    divergence_star = ddx(u_star) + ddy(v_star)
    rhs = divergence_star / dt
    rhs[solid] = 0.0
    p = solve_pressure(p, rhs, pressure_iterations)

    # Correct velocity using the pressure gradient.
    u = u_star - dt * ddx(p)
    v = v_star - dt * ddy(p)
    u, v = apply_velocity_boundaries(u, v)

    # Save data often enough to resolve the oscillation for the FFT.
    if step % output_every == 0 or step == 1 or step == steps:
        latest_vort = ddx(v) - ddy(u)
        # Approximate pressure force on the circular surface. The force is
        # sampled slightly outside the masked cylinder; it is a qualitative
        # estimate because a Cartesian mask does not resolve the curved wall exactly.
        p_surface = sample_bilinear(p, x_surface, y_surface)
        cl_pressure = -np.trapz(p_surface * np.sin(theta), theta) / (0.5 * U_in**2 * D)
        cd_pressure = -np.trapz(p_surface * np.cos(theta), theta) / (0.5 * U_in**2 * D)
        speed = np.sqrt(u*u + v*v)
        fluid_speed = speed[fluid]
        row = {
            "step": step,
            "time": step * dt,
            "dimensionless_time_Ut_over_D": step * dt * U_in / D,
            "mean_u": float(np.mean(u[fluid])),
            "max_speed": float(np.max(fluid_speed)),
            "max_abs_vorticity": float(np.max(np.abs(latest_vort[fluid]))),
            "max_abs_divergence": float(np.max(np.abs(divergence_star[fluid]))),
            "Cl_pressure_estimate": float(cl_pressure),
            "Cd_pressure_estimate": float(cd_pressure),
        }
        history.append(row)
        if step % max(output_every * 10, 1) == 0 or step == steps:
            print(f"step {step:5d}/{steps} | t*={row['dimensionless_time_Ut_over_D']:.2f} "
                  f"| max|omega|={row['max_abs_vorticity']:.3f} | Cl~={cl_pressure:.3f}")

# ------------------------------------------------------------
# 7. SAVE NUMERICAL RESULTS
# ------------------------------------------------------------
history_path = OUT / "cylinder_history.csv"
with history_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

vort = ddx(v) - ddy(u)
vort_plot = vort.copy()
vort_plot[solid] = np.nan
np.savez_compressed(OUT / "cylinder_final_fields.npz", x=x, y=y, u=u, v=v, p=p, vorticity=vort)

# A compact text summary makes the run settings and interpretation explicit.
summary_path = OUT / "cylinder_summary.txt"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("Experiment 4: cylinder flow using FDM + pressure projection\n")
    f.write(f"Grid: {nx} x {ny}\nReynolds number: {Re}\n")
    f.write(f"Inlet velocity: {U_in}\nKinematic viscosity: {nu}\nTime step: {dt}\n")
    f.write(f"\nSteps completed: {steps}\nFinal dimensionless time: {steps*dt*U_in/D:.6f}\n")
    f.write("Lift/drag values are pressure-only estimates sampled outside the masked cylinder.\n")
    f.write("This educational FDM solver needs grid/time-step validation for quantitative force reporting.\n")
print("\nSimulation finished. Results folder:", OUT)
print("Summary:", summary_path)

# ------------------------------------------------------------
# 8. PLOT 1 — LIFT COEFFICIENT HISTORY
# ------------------------------------------------------------
time_star = np.array([row["dimensionless_time_Ut_over_D"] for row in history])
cl_history = np.array([row["Cl_pressure_estimate"] for row in history])
cd_history = np.array([row["Cd_pressure_estimate"] for row in history])

plt.figure(figsize=(10, 4))
plt.plot(time_star, cl_history, linewidth=1.4)
plt.xlabel("Dimensionless time, tU/D")
plt.ylabel("Lift coefficient, Cl (pressure estimate)")
plt.title("Cylinder immersed in fluid: lift history (FDM estimate)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "lift_coefficient_history.png", dpi=160)
plt.show()

# ------------------------------------------------------------
# 9. PLOT 2 — VORTICITY CONTOUR / VON KARMAN WAKE
# ------------------------------------------------------------
plt.figure(figsize=(12, 4.2))
levels = np.linspace(-np.nanpercentile(np.abs(vort_plot), 98),
                     np.nanpercentile(np.abs(vort_plot), 98), 41)
contour = plt.contourf(X, Y, vort_plot, levels=levels, cmap="bwr", extend="both")
plt.colorbar(contour, label="Vorticity, dv/dx - du/dy")
# Draw a clean cylinder outline on top of the masked grid cells.
angle = np.linspace(0, 2*np.pi, 250)
plt.fill(cx + R*np.cos(angle), cy + R*np.sin(angle), color="white", zorder=5)
plt.plot(cx + R*np.cos(angle), cy + R*np.sin(angle), color="black", linewidth=1.2, zorder=6)
plt.xlabel("x / D")
plt.ylabel("y / D")
plt.title(f"Vorticity behind cylinder at tU/D = {steps*dt*U_in/D:.1f} (FDM)")
plt.xlim(cx - 2*D, Lx)
plt.ylim(0, Ly)
plt.tight_layout()
plt.savefig(OUT / "vorticity_contour.png", dpi=160)
plt.show()

# ------------------------------------------------------------
# 10. PLOT 3 — FFT OF LIFT HISTORY AND STROUHAL NUMBER
# ------------------------------------------------------------
# Ignore the early transient; use the latter half of the saved lift record.
# If the record is too short, use all samples after the first few points.
start_index = len(cl_history) // 2
cl_signal = cl_history[start_index:]
t_signal = time_star[start_index:]

if len(cl_signal) >= 8:
    # Remove the mean and apply a Hann window to reduce spectral leakage.
    cl_signal = cl_signal - np.mean(cl_signal)
    window = np.hanning(len(cl_signal))
    dt_star = float(np.mean(np.diff(t_signal)))
    spectrum = np.abs(np.fft.rfft(cl_signal * window))
    frequencies = np.fft.rfftfreq(len(cl_signal), d=dt_star)  # frequency in U/D
    # Ignore zero frequency; the strongest remaining peak estimates shedding frequency.
    if len(frequencies) > 1:
        peak_index = 1 + int(np.argmax(spectrum[1:]))
        st_estimate = float(frequencies[peak_index])
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, spectrum, linewidth=1.3)
        plt.axvline(st_estimate, linestyle="--", linewidth=1.2,
                    label=f"Peak = St ≈ {st_estimate:.4f}")
        plt.xlim(0, min(1.0, frequencies[-1]))
        plt.xlabel("Frequency (U/D)")
        plt.ylabel("|FFT(Cl)|")
        plt.title("Lift-coefficient spectrum: dominant peak estimates Strouhal number")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT / "lift_fft_strouhal.png", dpi=160)
        plt.show()
        print(f"Estimated Strouhal number from lift FFT: {st_estimate:.4f}")
    else:
        st_estimate = float("nan")
        print("Not enough FFT frequency bins to estimate Strouhal number.")
else:
    st_estimate = float("nan")
    print("Not enough saved lift samples for FFT. Increase steps or reduce output_every.")

# ------------------------------------------------------------
# 11. SAVE SUMMARY WITH FFT RESULT AND PACKAGE THE OUTPUTS
# ------------------------------------------------------------
with summary_path.open("a", encoding="utf-8") as f:
    f.write(f"\nEstimated Strouhal number from pressure-lift FFT: {st_estimate:.6f}\n")
    f.write("Note: this value is a numerical estimate, not a validated benchmark.\n")

print("\nGenerated files:")
for file in sorted(OUT.iterdir()):
    if file.is_file():
        print(f"{file.name:32s} {file.stat().st_size/1024:.1f} KB")

archive = ROOT / "experiment_04_fdm_results.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file in sorted(OUT.iterdir()):
        if file.is_file():
            zf.write(file, file.relative_to(OUT))
print("Results archive:", archive)
