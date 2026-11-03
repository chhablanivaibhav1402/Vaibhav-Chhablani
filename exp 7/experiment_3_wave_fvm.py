import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

L, H, g, nx = 20.0, 1.0, 9.81, 500
A0, width, t_end = 0.10, 0.40, 0.80
dx = L / nx
c = np.sqrt(g * H)
dt = t_end / int(np.ceil(t_end * c / (0.8 * dx)))
lam = c * dt / dx
x = (np.arange(nx) + 0.5) * dx
initial = A0 * np.exp(-((x - L / 2) / width) ** 2)

r_exp = initial.copy()
l_exp = initial.copy()
r_imp = initial.copy()
l_imp = initial.copy()

M_r = lil_matrix((nx, nx))
M_l = lil_matrix((nx, nx))
for i in range(nx):
    M_r[i, i] = 1 + lam
    M_r[i, (i - 1) % nx] = -lam
    M_l[i, i] = 1 + lam
    M_l[i, (i + 1) % nx] = -lam
solve_r = factorized(M_r.tocsc())
solve_l = factorized(M_l.tocsc())

for _ in range(round(t_end / dt)):
    r_exp = r_exp - lam * (r_exp - np.roll(r_exp, 1))
    l_exp = l_exp - lam * (l_exp - np.roll(l_exp, -1))
    r_imp = solve_r(r_imp)
    l_imp = solve_l(l_imp)

eta_exp = 0.5 * (r_exp + l_exp)
eta_imp = 0.5 * (r_imp + l_imp)
eta_exact = 0.5 * A0 * (
    np.exp(-((x - L / 2 - c * t_end) / width) ** 2)
    + np.exp(-((x - L / 2 + c * t_end) / width) ** 2)
)
vol_exp = dx * np.sum(eta_exp)
vol_imp = dx * np.sum(eta_imp)
vol_0 = dx * np.sum(initial)
l2_exp = np.sqrt(dx * np.sum((eta_exp - eta_exact) ** 2))
l2_imp = np.sqrt(dx * np.sum((eta_imp - eta_exact) ** 2))

print(f"N={nx}; dx={dx:.5f}; dt={dt:.7f}; CFL={lam:.6f}")
print(f"Time={t_end:.3f}; wave speed={c:.6f}")
print(f"Volume initial={vol_0:.12f}")
print(f"Volume explicit={vol_exp:.12f}; relative error={abs(vol_exp-vol_0)/vol_0:.3e}")
print(f"Volume implicit={vol_imp:.12f}; relative error={abs(vol_imp-vol_0)/vol_0:.3e}")
print(f"L2 explicit={l2_exp:.8f}; L2 implicit={l2_imp:.8f}")

plt.figure(figsize=(9, 4.8))
plt.plot(x, initial, ":", color="0.4", lw=1.5, label="Initial surface")
plt.plot(x, eta_exact, "k-", lw=2, label="Exact at t=0.8 s")
plt.plot(x, eta_exp, color="#126782", lw=1.6, label="FVM explicit")
plt.plot(x, eta_imp, color="#ae6033", lw=1.6, label="FVM implicit")
plt.xlabel("Horizontal distance x (m)")
plt.ylabel("Free-surface displacement eta (m)")
plt.title("Experiment 3: Propagation of a small wave on still water")
plt.grid(alpha=0.22)
plt.legend()
plt.tight_layout()
plt.savefig("experiment_3_wave_output.png", dpi=180)
plt.close()
