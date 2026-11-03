import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dstn, idstn

N = 88
RE = 88.0
LID_SPEED = 1.0
FINAL_TIME = 23.0
LENGTH = 1.0

h = LENGTH / (N - 1)
nu = LID_SPEED * LENGTH / RE
dt_target = min(0.30 * h / LID_SPEED, 0.20 * h**2 / nu)
steps = int(np.ceil(FINAL_TIME / dt_target))
dt = FINAL_TIME / steps
x = np.linspace(0.0, LENGTH, N)
y = np.linspace(0.0, LENGTH, N)
omega = np.zeros((N, N))
psi = np.zeros_like(omega)
k = np.arange(1, N - 1)
eigenvalues = 4.0 / h**2 * (np.sin(np.pi * k[:, None] / (2 * (N - 1)))**2 + np.sin(np.pi * k[None, :] / (2 * (N - 1)))**2)

def solve_streamfunction(w):
    spectral = dstn(w[1:-1, 1:-1], type=1, norm="ortho")
    p = np.zeros_like(w)
    p[1:-1, 1:-1] = idstn(spectral / eigenvalues, type=1, norm="ortho")
    return p

def wall_vorticity(w, p):
    w[0, 1:-1] = -2.0 * p[1, 1:-1] / h**2
    w[-1, 1:-1] = -2.0 * p[-2, 1:-1] / h**2 - 2.0 * LID_SPEED / h
    w[1:-1, 0] = -2.0 * p[1:-1, 1] / h**2
    w[1:-1, -1] = -2.0 * p[1:-1, -2] / h**2

def rhs(w, p):
    c = w[1:-1, 1:-1]
    e, wst = w[1:-1, 2:], w[1:-1, :-2]
    n, s = w[2:, 1:-1], w[:-2, 1:-1]
    ue = (p[2:, 2:] + p[2:, 1:-1] - p[:-2, 2:] - p[:-2, 1:-1]) / (4 * h)
    uw = (p[2:, 1:-1] + p[2:, :-2] - p[:-2, 1:-1] - p[:-2, :-2]) / (4 * h)
    vn = -(p[2:, 2:] + p[1:-1, 2:] - p[2:, :-2] - p[1:-1, :-2]) / (4 * h)
    vs = -(p[1:-1, 2:] + p[:-2, 2:] - p[1:-1, :-2] - p[:-2, :-2]) / (4 * h)
    fe = ue * np.where(ue >= 0, c, e)
    fw = uw * np.where(uw >= 0, wst, c)
    fn = vn * np.where(vn >= 0, c, n)
    fs = vs * np.where(vs >= 0, s, c)
    convection = (fe - fw + fn - fs) / h
    diffusion = nu * (e + wst + n + s - 4 * c) / h**2
    return diffusion - convection, np.max(np.abs((ue - uw + vn - vs) / h))

max_face_div = 0.0
for iteration in range(1, steps + 1):
    psi = solve_streamfunction(omega)
    wall_vorticity(omega, psi)
    rate, face_div = rhs(omega, psi)
    omega[1:-1, 1:-1] += dt * rate
    max_face_div = max(max_face_div, face_div)

psi = solve_streamfunction(omega)
wall_vorticity(omega, psi)
u = np.zeros_like(psi)
v = np.zeros_like(psi)
u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
u[-1, :] = LID_SPEED
rate, final_face_div = rhs(omega, psi)
steady_residual = float(np.max(np.abs(rate)))

ghia_y = np.array([0.0, .1016, .1719, .2813, .4531, .5, .6172, .7344, .8516, .9531, .9688, 1.0])
ghia_u = np.array([0.0, -.06434, -.10150, -.15662, -.21090, -.20581, -.13641, .00332, .23151, .68717, .78871, 1.0])
ghia_x = np.array([0.0, .0625, .0938, .1563, .2266, .5, .8047, .8594, .9063, .9531, .9688, 1.0])
ghia_v = np.array([0.0, .09233, .12317, .16077, .17507, .05454, -.24533, -.22445, -.16914, -.08864, -.05906, 0.0])
uc = np.interp(ghia_y, y, u[:, N // 2])
vc = np.interp(ghia_x, x, v[N // 2, :])
rmse_u = np.sqrt(np.mean((uc - ghia_u)**2))
rmse_v = np.sqrt(np.mean((vc - ghia_v)**2))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].contour(x, y, psi, levels=22, colors="black", linewidths=.7)
axes[0].set(xlabel="x / L", ylabel="y / L", title="Streamlines (streamfunction)", aspect="equal")
axes[1].plot(u[:, N // 2], y, label="Numerical")
axes[1].plot(ghia_u, ghia_y, "o", ms=4, label="Ghia et al.")
axes[1].set(xlabel="u / U", ylabel="y / L", title="Vertical centreline")
axes[1].legend(fontsize=8)
axes[2].plot(x, v[N // 2, :], label="Numerical")
axes[2].plot(ghia_x, ghia_v, "o", ms=4, label="Ghia et al.")
axes[2].set(xlabel="x / L", ylabel="v / U", title="Horizontal centreline")
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig("experiment_9_cavity_output.png", dpi=160)
plt.close(fig)
print(f"Re={RE:.0f}, N={N}, steps={steps}, dt={dt:.6g}, final_time={FINAL_TIME:g}")
print(f"u_center={u[N//2, N//2]:.6f}, v_center={v[N//2, N//2]:.6f}")
print(f"Ghia_u_RMSE={rmse_u:.6f}, Ghia_v_RMSE={rmse_v:.6f}")
print(f"max_FV_flux_divergence={max_face_div:.3e}, steady_vorticity_rate={steady_residual:.3e}")
print("figure=experiment_9_cavity_output.png")
