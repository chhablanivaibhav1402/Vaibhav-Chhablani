import numpy as np
import matplotlib.pyplot as plt

H = 1.0
U_bottom = 0.0
U_top = 1.0
nu = 0.01
N = 20

dy = H / N
y = (np.arange(N) + 0.5) * dy

t_end = 1000.0

# ---------------- EXPLICIT FVM ----------------

dt_limit = 0.3 * dy**2 / nu

steps_exp = int(np.ceil(t_end / dt_limit))

# Make explicit steps divisible by 10
steps_exp = int(np.ceil(steps_exp / 10)) * 10

dt_exp = t_end / steps_exp
Fo_exp = nu * dt_exp / dy**2

u_exp = np.zeros(N)

for n in range(steps_exp):
    old = u_exp.copy()

    u_exp[1:-1] = old[1:-1] + Fo_exp * (
        old[2:] - 2.0 * old[1:-1] + old[:-2]
    )

    u_exp[0] = old[0] + Fo_exp * (
        old[1] - 3.0 * old[0] + 2.0 * U_bottom
    )

    u_exp[-1] = old[-1] + Fo_exp * (
        old[-2] - 3.0 * old[-1] + 2.0 * U_top
    )

# ---------------- IMPLICIT FVM ----------------

steps_imp = steps_exp // 10
dt_imp = 10.0 * dt_exp

Fo_imp = nu * dt_imp / dy**2

A = np.zeros((N, N))

for i in range(N):
    if i == 0:
        A[i, i] = 1.0 + 3.0 * Fo_imp
        A[i, i + 1] = -Fo_imp

    elif i == N - 1:
        A[i, i] = 1.0 + 3.0 * Fo_imp
        A[i, i - 1] = -Fo_imp

    else:
        A[i, i] = 1.0 + 2.0 * Fo_imp
        A[i, i - 1] = -Fo_imp
        A[i, i + 1] = -Fo_imp

u_imp = np.zeros(N)

for n in range(steps_imp):
    b = u_imp.copy()

    b[0] += 2.0 * Fo_imp * U_bottom
    b[-1] += 2.0 * Fo_imp * U_top

    u_imp = np.linalg.solve(A, b)

# ---------------- EXACT SOLUTION ----------------

u_exact = U_bottom + (U_top - U_bottom) * y / H

# ---------------- PLOTS ----------------

plt.plot(u_exp, y, "o-", label="Explicit FVM")
plt.plot(u_imp, y, "s--", label="Implicit FVM")
plt.plot(u_exact, y, "k-", label="Exact")

plt.xlabel("Velocity")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.title("Couette Flow at t = 1000")
plt.show()

# ---------------- NUMERICAL VALUES ----------------

print("Total simulation time =", t_end)

print("Explicit dt =", dt_exp)
print("Explicit steps =", steps_exp)

print("Implicit dt =", dt_imp)
print("Implicit steps =", steps_imp)

print("Explicit Fo =", Fo_exp)
print("Implicit Fo =", Fo_imp)

print("Explicit steady error =",
      np.max(np.abs(u_exp - u_exact)))

print("Implicit steady error =",
      np.max(np.abs(u_imp - u_exact)))

print("Maximum difference =",
      np.max(np.abs(u_exp - u_imp)))

print("RMS difference =",
      np.sqrt(np.mean((u_exp - u_imp)**2)))

np.set_printoptions(precision=5, suppress=True)

print("\nExplicit velocity values:")
print(u_exp)

print("\nImplicit velocity values:")
print(u_imp)

print("\nExact velocity values:")
print(u_exact)

print("\nMaximum implicit velocity =", np.max(u_imp))
print("Minimum implicit velocity =", np.min(u_imp))
print("Mean implicit velocity =", np.mean(u_imp))
