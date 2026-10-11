
import numpy as np
import matplotlib.pyplot as plt

Lx = 1.0
Ly = 1.0
Nx = 12
Ny = 12

dx = Lx / Nx
dy = Ly / Ny
alpha = 0.01

T_left = 100.0
T_right = 0.0
T_bottom = 0.0
T_top = 0.0

t_end = 1000.0

# ---------------- EXPLICIT FVM ----------------
# ---------------- EXPLICIT TIME STEP ----------------

dt_limit = 0.3 / (
    alpha * (1.0 / dx**2 + 1.0 / dy**2)
)

steps_exp = int(np.ceil(t_end / dt_limit))

# Ensure the explicit steps are divisible by 10
steps_exp = int(np.ceil(steps_exp / 10)) * 10

dt_exp = t_end / steps_exp
rx = alpha * dt_exp / dx**2
ry = alpha * dt_exp / dy**2

T_exp = np.zeros((Ny, Nx))

for n in range(steps_exp):
    old = T_exp.copy()
    new = old.copy()

    for j in range(Ny):
        for i in range(Nx):
            P = old[j, i]

            if i == 0:
                west = 2.0 * (T_left - P)
            else:
                west = old[j, i - 1] - P

            if i == Nx - 1:
                east = 2.0 * (T_right - P)
            else:
                east = old[j, i + 1] - P

            if j == 0:
                south = 2.0 * (T_bottom - P)
            else:
                south = old[j - 1, i] - P

            if j == Ny - 1:
                north = 2.0 * (T_top - P)
            else:
                north = old[j + 1, i] - P

            new[j, i] = (
                P + rx * (east + west)
                + ry * (north + south)
            )

    T_exp = new

# ---------------- IMPLICIT FVM ----------------
# ---------------- IMPLICIT TIME STEP ----------------

steps_imp = steps_exp // 10
dt_imp = 10.0 * dt_exp
rx = alpha * dt_imp / dx**2
ry = alpha * dt_imp / dy**2

M = Nx * Ny
A = np.zeros((M, M))
source = np.zeros(M)

for j in range(Ny):
    for i in range(Nx):
        p = j * Nx + i
        aP = 1.0

        if i == 0:
            aP += 2.0 * rx
            source[p] += 2.0 * rx * T_left
        else:
            aP += rx
            A[p, p - 1] = -rx

        if i == Nx - 1:
            aP += 2.0 * rx
            source[p] += 2.0 * rx * T_right
        else:
            aP += rx
            A[p, p + 1] = -rx

        if j == 0:
            aP += 2.0 * ry
            source[p] += 2.0 * ry * T_bottom
        else:
            aP += ry
            A[p, p - Nx] = -ry

        if j == Ny - 1:
            aP += 2.0 * ry
            source[p] += 2.0 * ry * T_top
        else:
            aP += ry
            A[p, p + Nx] = -ry

        A[p, p] = aP

T_imp = np.zeros((Ny, Nx))

for n in range(steps_imp):
    b = T_imp.ravel() + source
    T_imp = np.linalg.solve(A, b).reshape(Ny, Nx)

# ---------------- PLOTS ----------------

plt.figure()
plt.imshow(
    T_exp, origin="lower",
    extent=[0, Lx, 0, Ly], aspect="equal"
)
plt.colorbar(label="Temperature")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Explicit FVM at t = 1000")
plt.show()

plt.figure()
plt.imshow(
    T_imp, origin="lower",
    extent=[0, Lx, 0, Ly], aspect="equal"
)
plt.colorbar(label="Temperature")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Implicit FVM at t = 1000")
plt.show()

# ---------------- NUMERICAL VALUES ----------------

print("Explicit time step =", dt_exp)
print("Explicit number of steps =", steps_exp)
print("Implicit time step =", dt_imp)
print("Implicit number of steps =", steps_imp)

print(
    "Explicit rx + ry =",
    alpha * dt_exp / dx**2 + alpha * dt_exp / dy**2
)

print(
    "Implicit rx + ry =",
    alpha * dt_imp / dx**2 + alpha * dt_imp / dy**2
)

print(
    "Maximum difference =",
    np.max(np.abs(T_exp - T_imp))
)

print(
    "RMS difference =",
    np.sqrt(np.mean((T_exp - T_imp)**2))
)

np.set_printoptions(precision=4, suppress=True)

print("\nFinal Explicit Temperature Matrix:")
print(T_exp)

print("\nFinal Implicit Temperature Matrix:")
print(T_imp)

print("\nImplicit Maximum Temperature =", np.max(T_imp))
print("Implicit Minimum Temperature =", np.min(T_imp))
print("Implicit Mean Temperature =", np.mean(T_imp))
