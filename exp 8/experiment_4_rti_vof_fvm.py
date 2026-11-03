import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized

nx, ny = 100, 200
Lx, Ly = 2.0, 4.0
dx, dy = Lx / nx, Ly / ny
rho_light, rho_heavy, g, nu = 1.0, 1.2, 9.81, 0.005
rho_ref = 0.5 * (rho_light + rho_heavy)
t_end = 2.1
x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dy
interface = 1.0 + 0.04 * np.cos(2 * np.pi * x / Lx)
alpha = np.clip((y[:, None] + dy / 2 - interface[None, :]) / dy, 0, 1)
u = np.zeros((ny, nx))
v = np.zeros((ny + 1, nx))

Lx_op = diags([np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1]).tolil()
Lx_op[0, -1] = Lx_op[-1, 0] = 1
Ly_op = diags([np.ones(ny - 1), -2 * np.ones(ny), np.ones(ny - 1)], [-1, 0, 1]).tolil()
Ly_op[0, 0] = Ly_op[-1, -1] = -1
lap = kron(eye(ny), Lx_op / dx**2) + kron(Ly_op / dy**2, eye(nx))
lap = lap.tolil()
lap[0, :] = 0
lap[0, 0] = 1
solve_pressure = factorized(lap.tocsc())

snapshots = [(0.0, alpha.copy())]
initial_mass = alpha.sum() * dx * dy
t = 0.0
steps = 0
maximum_divergence = 0.0
while t < t_end - 1e-12:
    adv_rate = np.max(np.abs(u)) / dx + np.max(np.abs(v)) / dy
    dt = min(0.003, 0.3 / max(adv_rate, 1e-12),
             0.45 / (2 * nu * (1/dx**2 + 1/dy**2)), t_end - t)

    fx = u * np.where(u >= 0, np.roll(alpha, 1, axis=1), alpha)
    lower = np.vstack((alpha[:1], alpha))
    upper = np.vstack((alpha, alpha[-1:]))
    fy = v * np.where(v >= 0, lower, upper)
    alpha = alpha - dt * ((np.roll(fx, -1, axis=1) - fx) / dx
                          + (fy[1:] - fy[:-1]) / dy)

    ue = 0.5 * (u + np.roll(u, -1, axis=1))
    uw = 0.5 * (u + np.roll(u, 1, axis=1))
    un = np.roll(u, -1, axis=0)
    us = np.roll(u, 1, axis=0)
    vn = 0.5 * (v[1:] + np.roll(v[1:], 1, axis=1))
    vs = 0.5 * (v[:-1] + np.roll(v[:-1], 1, axis=1))
    un[-1] = u[-1]
    us[0] = u[0]
    flux_ue = ue * np.where(ue >= 0, u, np.roll(u, -1, axis=1))
    flux_uw = uw * np.where(uw >= 0, np.roll(u, 1, axis=1), u)
    flux_un = vn * np.where(vn >= 0, u, un)
    flux_us = vs * np.where(vs >= 0, us, u)
    conv_u = (flux_ue - flux_uw) / dx + (flux_un - flux_us) / dy
    lap_u = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
    lap_u += (un - 2*u + us) / dy**2
    u_star = u + dt * (-conv_u + nu * lap_u)

    vm = v[1:-1]
    ve = 0.5 * (np.roll(u[:-1], -1, axis=1) + np.roll(u[1:], -1, axis=1))
    vw = 0.5 * (u[:-1] + u[1:])
    vne = 0.5 * (vm + v[2:])
    vsw = 0.5 * (vm + v[:-2])
    flux_ve = ve * np.where(ve >= 0, vm, np.roll(vm, -1, axis=1))
    flux_vw = vw * np.where(vw >= 0, np.roll(vm, 1, axis=1), vm)
    flux_vn = vne * np.where(vne >= 0, vm, v[2:])
    flux_vs = vsw * np.where(vsw >= 0, v[:-2], vm)
    conv_v = (flux_ve - flux_vw) / dx + (flux_vn - flux_vs) / dy
    lap_v = (np.roll(vm, -1, axis=1) - 2*vm + np.roll(vm, 1, axis=1)) / dx**2
    lap_v += (v[2:] - 2*vm + v[:-2]) / dy**2
    buoyancy = -g * (rho_heavy - rho_light) * (alpha[:-1] + alpha[1:] - 1) / (2*rho_ref)
    v_star = v.copy()
    v_star[1:-1] = vm + dt * (-conv_v + nu * lap_v + buoyancy)

    divergence = (np.roll(u_star, -1, axis=1) - u_star) / dx
    divergence += (v_star[1:] - v_star[:-1]) / dy
    rhs = (rho_ref / dt * divergence).ravel()
    rhs[0] = 0
    p = solve_pressure(rhs).reshape(ny, nx)
    u = u_star - (dt / (rho_ref * dx)) * (p - np.roll(p, 1, axis=1))
    v = v_star
    v[1:-1] -= (dt / (rho_ref * dy)) * (p[1:] - p[:-1])

    current_div = (np.roll(u, -1, axis=1) - u) / dx
    current_div += (v[1:] - v[:-1]) / dy
    maximum_divergence = max(maximum_divergence, np.max(np.abs(current_div)))
    if alpha.min() < -1e-9 or alpha.max() > 1+1e-9:
        raise RuntimeError("Volume fraction left physical bounds")
    t += dt
    steps += 1
    if t >= len(snapshots) * (t_end/4) - 1e-12:
        snapshots.append((t, alpha.copy()))

mass = alpha.sum() * dx * dy
print(f"Grid={nx}x{ny}; steps={steps}; time={t:.4f}")
print(f"Initial heavy-fluid area={initial_mass:.12f}")
print(f"Final heavy-fluid area={mass:.12f}")
print(f"Relative mass error={abs(mass - initial_mass)/initial_mass:.3e}")
print(f"Volume fraction range=[{alpha.min():.6f}, {alpha.max():.6f}]")
print(f"Maximum projected divergence={maximum_divergence:.3e}")
print(f"Max speed={max(np.max(np.abs(u)), np.max(np.abs(v))):.6f}")

fig, axs = plt.subplots(1, len(snapshots), figsize=(13, 4.8), sharex=True, sharey=True)
for ax, (ti, ai) in zip(axs, snapshots):
    ax.imshow(ai, origin="lower", extent=(0, Lx, 0, Ly), cmap="Blues", vmin=0, vmax=1)
    ax.contour(x, y, ai, levels=[0.5], colors="k", linewidths=0.8)
    ax.set_title(f"t = {ti:.2f} s")
    ax.set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
fig.suptitle("Experiment 4: Rayleigh-Taylor instability (VOF, heavy fluid in blue)")
fig.tight_layout()
fig.savefig("experiment_4_rti_output.png", dpi=180)
plt.close(fig)
