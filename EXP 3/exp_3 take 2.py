
# CENTRIFUGAL PUMP: TRANSIENT POLAR FDM WITH PUMP SHUTDOWN

from pathlib import Path
'''path makes workingwith file and folder paths easier.
It lets you construct paths using Python objects instead of
manually joining strings.'''
import shutil#Provides file and folder operations, including copying and deleting folders.
import subprocess#Allows Python to start and control another program or process
import sys#Provides access to Python's runtime information and command-line arguments.
import zipfile#Allows Python to create, read, and extract ZIP archives.
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
'''display() shows rich outputs inside a notebook, while Markdown formats notebook
output as Markdown, such as headings, explanations, and formatted results.'''
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

IN_COLAB = False
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    colab_files = None
'''IN_COLAB = False: initially assumes the script is not running in Google Colab.



try:: tells Python to attempt an operation that might fail.



from google.colab import files as colab_files: attempts to import Colab's file-handling utility.



IN_COLAB = True: if the import succeeds, the code recognizes the Colab environment.



except ImportError:: handles the case where the Colab module is unavailable.



colab_files = None: records that the Colab file utility cannot be used.'''


# ============================================================
# 1. RUNTIME HELPERS
# ============================================================

def run_solver(script_path, args, output_dir, timeout=1800):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(script_path),
        *[str(v) for v in args]
    ]
    

    print("Running:", " ".join(command))

    result = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    (output_dir / "solver_stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    (output_dir / "solver_stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode:
        raise RuntimeError(
            result.stderr.strip()
            or f"Solver exited with code {result.returncode}"
        )

    return output_dir


def read_csv(path):
    return np.atleast_1d(
        np.genfromtxt(
            path, delimiter=",", names=True,
            dtype=None, encoding="utf-8"
        )
    )


def show_summary(path):
    summary_text = Path(path).read_text(
        encoding="utf-8"
    ).strip()

    display(Markdown(
        "### Solver summary\n\n```\n"
        + summary_text
        + "\n```"
    ))

    return summary_text


def make_zip(output_dir, number):
    output_dir = Path(output_dir)

    archive = output_dir.parent / (
        f"experiment_{number:02d}_fdm_results.zip"
    )

    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED
    ) as z:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(output_dir))

    print("Results archive:", archive)
    return archive


ROOT = (
    Path("/content/cfd_lab_exp03_fdm")
    if IN_COLAB
    else Path.cwd() / "_colab_runtime_exp03_fdm"
)

CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "outputs"

CODE_DIR.mkdir(parents=True, exist_ok=True)

print(
    "Runtime:",
    "Google Colab" if IN_COLAB else "local notebook"
)
print("Working folder:", ROOT)


# ============================================================
# 2. TRANSIENT POLAR FDM SOLVER
# ============================================================

solver_source = r'''
from __future__ import annotations

import csv
import math
import sys
import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def main():

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------
    # nr, ntheta, rpm, Re, Tin, Twall, swirl, wall_htc,
    # loss_fraction, shutdown_time, dt

    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    ntheta = int(sys.argv[2]) if len(sys.argv) > 2 else 160

    rpm0 = float(sys.argv[3]) if len(sys.argv) > 3 else 3000.0
    Re = float(sys.argv[4]) if len(sys.argv) > 4 else 500.0
    Tin = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0
    Twall = float(sys.argv[6]) if len(sys.argv) > 6 else 330.0
    swirl = float(sys.argv[7]) if len(sys.argv) > 7 else 0.85
    wall_htc = float(sys.argv[8]) if len(sys.argv) > 8 else 120.0
    loss_fraction = float(sys.argv[9]) if len(sys.argv) > 9 else 0.04

    shutdown_time = float(sys.argv[10]) if len(sys.argv) > 10 else 5.0
    dt = float(sys.argv[11]) if len(sys.argv) > 11 else 0.05

    if nr < 5 or ntheta < 8:
        raise ValueError("Use nr >= 5 and ntheta >= 8.")

    if rpm0 <= 0 or Re <= 0 or dt <= 0 or shutdown_time <= 0:
        raise ValueError("RPM, Re, dt and shutdown time must be positive.")

    if not 0 <= swirl <= 1.5:
        raise ValueError("Swirl must be between 0 and 1.5.")

    if wall_htc < 0 or loss_fraction < 0:
        raise ValueError("HTC and loss fraction cannot be negative.")

    # --------------------------------------------------------
    # GEOMETRY AND FLUID PROPERTIES
    # --------------------------------------------------------

    r1 = 0.030
    r2 = 0.120
    b = 0.010

    rho = 997.0
    mu = 0.0010
    cp = 4180.0
    k = 0.60
    g = 9.81

    alpha = k / (rho * cp)

    Omega0 = rpm0 * 2.0 * math.pi / 60.0

    # Hydraulic diameter approximation for a narrow passage
    Dh = 2.0 * b

    # Initial inlet velocity inferred from Reynolds number
    ur10 = Re * mu / (rho * Dh)

    r = np.linspace(r1, r2, nr)
    theta = np.linspace(0.0, 2.0 * math.pi, ntheta,
                        endpoint=False)

    dr = r[1] - r[0]
    dtheta = 2.0 * math.pi / ntheta

    rr, tt = np.meshgrid(r, theta, indexing="ij")

    # --------------------------------------------------------
    # SHUTDOWN MODEL
    # --------------------------------------------------------
    # Exponential rundown is a reduced-order approximation:
    #
    # Omega(t) = Omega0 * exp(-t/tau)
    #
    # Mean flow is assumed to follow the same decay factor.
    # This is a prescribed shutdown law, not a torque-resolved
    # motor/rotor simulation.

    tau = shutdown_time / 5.0

    # Stop the numerical transient after approximately five
    # time constants, or at the requested shutdown time.
    nsteps = max(1, int(math.ceil(shutdown_time / dt)))
    times = np.linspace(0.0, nsteps * dt, nsteps + 1)

    def decay_factor(time):
        return math.exp(-time / tau)

    def make_velocity(time):
        factor = decay_factor(time)

        Omega = Omega0 * factor
        ur1 = ur10 * factor

        ur1d = ur1 * r1 / r
        utheta1d = swirl * Omega * r
        wtheta1d = utheta1d - Omega * r

        ur = ur1d[:, None] * np.ones((1, ntheta))
        utheta = utheta1d[:, None] * np.ones((1, ntheta))
        wtheta = wtheta1d[:, None] * np.ones((1, ntheta))

        uabs_x = ur * np.cos(tt) - utheta * np.sin(tt)
        uabs_y = ur * np.sin(tt) + utheta * np.cos(tt)

        urel_x = ur * np.cos(tt) - wtheta * np.sin(tt)
        urel_y = ur * np.sin(tt) + wtheta * np.cos(tt)

        return (
            factor, Omega, ur1, ur1d, utheta1d, wtheta1d,
            ur, utheta, wtheta,
            uabs_x, uabs_y, urel_x, urel_y
        )

    # --------------------------------------------------------
    # INITIAL TEMPERATURE FIELD
    # --------------------------------------------------------

    # Initially the passage is uniformly at inlet temperature.
    T = np.full((nr, ntheta), Tin, dtype=float)

    # Wall heat-transfer coefficient in the reduced energy model.
    beta = 2.0 * wall_htc / (rho * cp * b)

    # History includes the initial state and every time step.
    history = []

    def index(i, j):
        return i * ntheta + j

    # --------------------------------------------------------
    # TRANSIENT ENERGY EQUATION
    # --------------------------------------------------------
    # dT/dt + ur*dT/dr + (wtheta/r)*dT/dtheta
    # = alpha*Laplacian(T)
    #   + beta*(Twall - T) + q_source
    #
    # Backward Euler in time; central diffusion and upwind
    # advection in space. Angular direction is periodic.
    #
    # Radial boundary:
    #   inner boundary: prescribed Tin while flow is present
    #   outer boundary: zero radial temperature gradient
    #
    # Once flow is negligible, both radial boundaries use
    # zero-gradient conditions, allowing the temperature field
    # to relax toward the wall temperature.

    def solve_temperature(Told, time, ur1d, wtheta1d,
                          Omega, flow_factor):

        A = lil_matrix(
            (nr * ntheta, nr * ntheta),
            dtype=float
        )
        rhs = np.zeros(nr * ntheta)

        flowing = flow_factor > 1.0e-4

        # Inner radial boundary
        for j in range(ntheta):
            row = index(0, j)

            if flowing:
                A[row, row] = 1.0
                rhs[row] = Tin
            else:
                # dT/dr = 0, first-order one-sided form
                A[row, index(0, j)] = 1.0
                A[row, index(1, j)] = -1.0
                rhs[row] = 0.0

        # Outer radial boundary: zero gradient
        for j in range(ntheta):
            row = index(nr - 1, j)

            A[row, index(nr - 1, j)] = 1.0
            A[row, index(nr - 2, j)] = -1.0
            rhs[row] = 0.0

        for i in range(1, nr - 1):

            ri = r[i]

            vr = ur1d[i]
            va = wtheta1d[i] / ri

            # Radial diffusion, including (1/r)*dT/dr
            aw = (
                alpha / dr**2
                - alpha / (2.0 * ri * dr)
                + max(vr, 0.0) / dr
            )

            ae = (
                alpha / dr**2
                + alpha / (2.0 * ri * dr)
                + max(-vr, 0.0) / dr
            )

            # Angular diffusion and upwind convection
            ass = (
                alpha / (ri**2 * dtheta**2)
                + max(va, 0.0) / dtheta
            )

            an = (
                alpha / (ri**2 * dtheta**2)
                + max(-va, 0.0) / dtheta
            )

            # Reduced-order dissipative heating.
            # The source is proportional to the prescribed
            # flow speed and local rotor-relative speed squared.
            ur_abs = abs(ur1d[i])
            relative_speed = abs(
                (swirl - 1.0) * Omega * ri
            )

            q_source = (
                loss_fraction * ur_abs
                * relative_speed**2 / (b * cp)
            )

            for j in range(ntheta):

                row = index(i, j)

                A[row, row] = (
                    1.0 / dt
                    + aw + ae + ass + an + beta
                )

                A[row, index(i - 1, j)] = -aw
                A[row, index(i + 1, j)] = -ae

                A[row, index(i, (j - 1) % ntheta)] = -ass
                A[row, index(i, (j + 1) % ntheta)] = -an

                rhs[row] = (
                    Told[i, j] / dt
                    + beta * Twall
                    + q_source
                )

        A = A.tocsr()
        Tnew = spsolve(A, rhs).reshape(nr, ntheta)

        if not np.all(np.isfinite(Tnew)):
            raise FloatingPointError(
                "Non-finite temperature values detected."
            )

        residual = float(
            np.max(np.abs(A @ Tnew.ravel() - rhs))
        )

        return Tnew, residual

    # --------------------------------------------------------
    # TIME LOOP
    # --------------------------------------------------------

    for step, time in enumerate(times):

        (
            factor, Omega, ur1, ur1d, utheta1d, wtheta1d,
            ur, utheta, wtheta,
            uabs_x, uabs_y, urel_x, urel_y
        ) = make_velocity(time)

        # The initial temperature is already defined.
        if step > 0:
            T, residual = solve_temperature(
                T, time, ur1d, wtheta1d, Omega, factor
            )
        else:
            residual = 0.0

        outlet_temp = float(np.mean(T[-1, :]))
        min_temp = float(np.min(T))
        max_temp = float(np.max(T))

        # Annular passage mass flow approximation
        mass_flow = rho * ur1 * (2.0 * math.pi * r1 * b)

        mean_speed = float(np.mean(np.sqrt(ur**2 + utheta**2)))

        history.append([
            step, time, factor, rpm0 * factor,
            ur1, mean_speed, residual,
            min_temp, max_temp, outlet_temp, mass_flow
        ])

        if step % max(1, nsteps // 10) == 0:
            print(
                f"t={time:7.3f} s | "
                f"flow fraction={factor:.5f} | "
                f"mean speed={mean_speed:.5e} m/s | "
                f"Tout={outlet_temp:.3f} K | "
                f"residual={residual:.3e}"
            )

    # --------------------------------------------------------
    # FINAL PRESSURE FIELD
    # --------------------------------------------------------

    (
        factor, Omega, ur1, ur1d, utheta1d, wtheta1d,
        ur, utheta, wtheta,
        uabs_x, uabs_y, urel_x, urel_y
    ) = make_velocity(times[-1])

    # Integrate radial pressure gradient:
    # dp/dr = rho*utheta^2/r - 12*mu*ur/b^2
    #
    # Gauge pressure is zero at the inner radius.

    rhs_p = (
        rho * utheta1d**2 / r
        - 12.0 * mu * ur1d / b**2
    )

    pressure = np.zeros(nr)
    pressure[1:] = np.cumsum(
        0.5 * dr * (rhs_p[:-1] + rhs_p[1:])
    )

    pressure_exact = (
        0.5 * rho * swirl**2 * Omega**2
        * (r**2 - r1**2)
        - 12.0 * mu * ur1 * r1
        * np.log(r / r1) / b**2
    )

    pressure_error = float(
        np.max(np.abs(pressure - pressure_exact))
    )

    # A periodic grid does not duplicate theta=2*pi.
    # Report the temperature jump across the periodic seam.
    periodic_seam_jump = float(
        np.max(np.abs(T[:, 0] - T[:, -1]))
    )

    # Second-order one-sided outlet gradient check
    outlet_gradient = float(
        np.max(np.abs(
            (3.0*T[-1, :] - 4.0*T[-2, :] + T[-3, :])
            / (2.0 * dr)
        ))
    )

    mass_flow_initial = rho * ur10 * (2.0 * math.pi * r1 * b)
    pump_head = float(
        (pressure[-1] - pressure[0]) / (rho * g)
    )

    heat_gain = float(
        mass_flow * cp * (T[-1, :].mean() - Tin)
    )

    # --------------------------------------------------------
    # SAVE FIELDS
    # --------------------------------------------------------

    np.savez(
        "pump_fdm_fields.npz",
        r=np.repeat(r[:, None], ntheta, axis=1),
        theta=np.repeat(theta[None, :], nr, axis=0),
        temperature=T,
        pressure=pressure,
        pressure_exact=pressure_exact,
        u_abs_x=uabs_x,
        u_abs_y=uabs_y,
        u_rel_x=urel_x,
        u_rel_y=urel_y,
        time=times[-1],
        time_history=times,
        temperature_initial=np.full((nr, ntheta), Tin),
        shutdown_factor=factor
    )

    # Transient history, not just a single residual
    with open(
        "pump_fdm_history.csv", "w",
        newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "iteration", "time_s", "flow_fraction",
            "rotor_rpm", "inlet_radial_velocity_m_per_s",
            "mean_speed_m_per_s", "residual_linf",
            "minimum_temperature", "maximum_temperature",
            "outlet_temperature", "mass_flow_rate_kg_per_s"
        ])

        writer.writerows(history)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        "pump_fdm_summary.txt", "w", encoding="utf-8"
    ) as f:

        f.write("Centrifugal pump transient polar FDM model\n")
        f.write("Model type: reduced-order annular passage\n")
        f.write("Shutdown model: prescribed exponential rundown\n")
        f.write("Time integration: backward Euler\n")
        f.write("Advection: first-order upwind\n")
        f.write("Diffusion: central finite differences\n")
        f.write("Angular boundary: periodic\n")
        f.write("Outer thermal boundary: zero radial gradient\n")
        f.write("Pressure: radial reduced-order approximation\n\n")

        f.write(f"grid_nr {nr}\n")
        f.write(f"grid_ntheta {ntheta}\n")
        f.write(f"dr_m {dr:.12e}\n")
        f.write(f"dtheta_rad {dtheta:.12e}\n")
        f.write(f"initial_rpm {rpm0:.12g}\n")
        f.write(f"final_rpm {rpm0 * factor:.12e}\n")
        f.write(f"Re_initial {Re:.12g}\n")
        f.write(f"shutdown_time_s {shutdown_time:.12g}\n")
        f.write(f"final_time_s {times[-1]:.12g}\n")
        f.write(f"time_step_s {dt:.12g}\n")
        f.write(f"decay_time_constant_s {tau:.12g}\n")
        f.write(f"final_flow_fraction {factor:.12e}\n")
        f.write(f"initial_mass_flow_kg_per_s {mass_flow_initial:.12e}\n")
        f.write(f"final_mass_flow_kg_per_s {mass_flow:.12e}\n")

        f.write(f"r1_m {r1:.12e}\n")
        f.write(f"r2_m {r2:.12e}\n")
        f.write(f"passage_depth_m {b:.12e}\n")
        f.write(f"Tin_K {Tin:.12e}\n")
        f.write(f"Twall_K {Twall:.12e}\n")
        f.write(f"Tout_final_K {T[-1, :].mean():.12e}\n")
        f.write(f"final_minimum_temperature_K {T.min():.12e}\n")
        f.write(f"final_maximum_temperature_K {T.max():.12e}\n")

        f.write(f"final_pressure_rise_Pa {pressure[-1]-pressure[0]:.12e}\n")
        f.write(f"final_pump_head_m {pump_head:.12e}\n")
        f.write(f"final_heat_gain_W {heat_gain:.12e}\n")
        f.write(f"max_pressure_fd_vs_exact_error_Pa {pressure_error:.12e}\n")
        f.write(f"periodic_seam_temperature_jump_K {periodic_seam_jump:.12e}\n")
        f.write(f"outlet_neumann_check_K_per_m {outlet_gradient:.12e}\n")
        f.write(f"final_linear_system_residual_linf {history[-1][6]:.12e}\n")
        f.write("model_scope teaching_level_reduced_order_not_blade_resolved_CFD\n")

    print("\nTransient shutdown simulation completed.")
    print("Wrote pump_fdm_fields.npz")
    print("Wrote pump_fdm_history.csv")
    print("Wrote pump_fdm_summary.txt")


if __name__ == "__main__":
    main()
'''

SCRIPT = CODE_DIR / "fdm_centrifugal_pump.py"
SCRIPT.write_text(solver_source, encoding="utf-8")

print("Transient solver written to:", SCRIPT)


# ============================================================
# 3. SELECT RUN MODE AND PARAMETERS
# ============================================================

RUN_MODE = "custom"  #@param ["quick", "lab", "custom"]

nr = 80  #@param {type:"integer"}
ntheta = 160  #@param {type:"integer"}

rpm = 3000.0  #@param {type:"number"}
Re = 500.0  #@param {type:"number"}

T_in = 300.0  #@param {type:"number"}
T_wall = 330.0  #@param {type:"number"}

swirl = 0.85  #@param {type:"number"}
wall_HTC = 120.0  #@param {type:"number"}
loss_fraction = 0.04  #@param {type:"number"}

shutdown_time = 5.0  #@param {type:"number"}
dt = 0.05  #@param {type:"number"}

QUICK_ARGS = [
    20, 32, 1500, 300, 300, 330, 0.8, 100, 0.04,
    2.0, 0.05
]

LAB_ARGS = [
    80, 160, 2000, 500, 300, 330, 0.85, 120, 0.04,
    5.0, 0.05
]

CUSTOM_ARGS = [
    nr, ntheta, rpm, Re, T_in, T_wall, swirl,
    wall_HTC, loss_fraction, shutdown_time, dt
]

ARGS = (
    QUICK_ARGS if RUN_MODE == "quick"
    else LAB_ARGS if RUN_MODE == "lab"
    else CUSTOM_ARGS
)

print("Run mode:", RUN_MODE)
print("Arguments:", ARGS)

OUT = run_solver(SCRIPT, ARGS, OUTPUT_DIR)


# ============================================================
# 4. READ SUMMARY AND HISTORY
# ============================================================

summary = show_summary(OUT / "pump_fdm_summary.txt")
history = read_csv(OUT / "pump_fdm_history.csv")

print(
    "Final linear-system residual:",
    f"{history['residual_linf'][-1]:.6e}"
)

print(
    "Final outlet temperature:",
    f"{history['outlet_temperature'][-1]:.6f} K"
)


# ============================================================
# 5. LOAD FINAL FIELDS
# ============================================================

fields = np.load(OUT / "pump_fdm_fields.npz")

r_field = fields["r"]
theta_field = fields["theta"]

x_field = r_field * np.cos(theta_field)
y_field = r_field * np.sin(theta_field)

temperature = fields["temperature"]
pressure = fields["pressure"]

ux = fields["u_rel_x"]
uy = fields["u_rel_y"]

speed = np.sqrt(ux**2 + uy**2)


# ============================================================
# 6. ORIGINAL THREE-PANEL PLOTS
# ============================================================

fig, axes = plt.subplots(
    1, 3, figsize=(17, 4.8), constrained_layout=True
)

im = axes[0].pcolormesh(
    x_field, y_field, temperature,
    shading="auto", cmap="inferno"
)
axes[0].set_title("Final Temperature [K]")
axes[0].set_aspect("equal")
fig.colorbar(im, ax=axes[0])

axes[1].plot(
    r_field[:, 0], pressure,
    "o-", ms=3, label="FDM"
)
axes[1].plot(
    r_field[:, 0], fields["pressure_exact"],
    "--", label="Closed-form comparison"
)
axes[1].set(
    xlabel="Radius [m]",
    ylabel="Gauge pressure [Pa]",
    title="Final Radial Pressure Rise"
)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

skip_r = max(1, r_field.shape[0] // 10)
skip_t = max(1, r_field.shape[1] // 16)

axes[2].pcolormesh(
    x_field, y_field, speed,
    shading="auto", cmap="viridis"
)

axes[2].quiver(
    x_field[::skip_r, ::skip_t],
    y_field[::skip_r, ::skip_t],
    ux[::skip_r, ::skip_t],
    uy[::skip_r, ::skip_t],
    color="white", scale=20
)

axes[2].set_title("Final Relative Speed and Vectors")
axes[2].set_aspect("equal")

plt.show()


# ============================================================
# 7. NEW TRANSIENT DECAY PLOTS
# ============================================================

fig, axes = plt.subplots(
    1, 3, figsize=(16, 4.5), constrained_layout=True
)

time = history["time_s"]
flow_fraction = history["flow_fraction"]
rpm_history = history["rotor_rpm"]
speed_history = history["mean_speed_m_per_s"]
outlet_history = history["outlet_temperature"]

axes[0].plot(time, flow_fraction, label="Flow fraction")
axes[0].set(
    xlabel="Time [s]",
    ylabel="Fraction of initial value",
    title="Pump Flow Decay"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(time, rpm_history, label="Rotor speed")
axes[1].plot(
    time, speed_history,
    label="Mean passage speed [m/s]"
)
axes[1].set(
    xlabel="Time [s]",
    title="Rotor and Flow Shutdown"
)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(
    time, outlet_history,
    label="Outlet temperature"
)
axes[2].axhline(
    T_wall if RUN_MODE == "custom" else 330.0,
    linestyle="--",
    label="Wall temperature"
)
axes[2].set(
    xlabel="Time [s]",
    ylabel="Temperature [K]",
    title="Thermal Response During Shutdown"
)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.show()


# ============================================================
# 8. PACKAGE RESULTS
# ============================================================

DOWNLOAD_RESULTS = True  #@param {type:"boolean"}

RESULTS_ZIP = make_zip(OUTPUT_DIR, 3)

if DOWNLOAD_RESULTS and IN_COLAB:
    colab_files.download(str(RESULTS_ZIP))
else:
    print("ZIP saved at:", RESULTS_ZIP)
    print("In Colab, set DOWNLOAD_RESULTS=True to download it.")

print("Output files:")
print(sorted(path.name for path in OUTPUT_DIR.iterdir()))
