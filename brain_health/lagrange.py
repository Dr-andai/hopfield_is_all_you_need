import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---- Parameters ----
m0 = 1.0        # baseline inertia
alpha = 0.5    # exposure sensitivity
k = 1.0        # homeostatic strength
lam = 1.0      # exposure forcing
gamma = 0.3    # dissipation

A = 2.0        # max exposure
tau = 10.0     # exposure timescale

# ---- Exposome trajectory ----
def exposome(t):
    e = A * (1 - np.exp(-t / tau))
    e_dot = (A / tau) * np.exp(-t / tau)
    return e, e_dot

# ---- ODE system ----
def brain_exposome_ode(t, x):
    b, v = x
    e, e_dot = exposome(t)

    m = m0 * (1 + alpha * e**2)

    dv = -(1 / m) * (
        2 * m0 * alpha * e * e_dot * v
        + gamma * v
        + k * b
        + lam * e
    )

    return [v, dv]

# ---- Solve ----
t_span = (0, 50)
t_eval = np.linspace(*t_span, 1000)
x0 = [0.5, 0.0]  # initial brain state

sol = solve_ivp(brain_exposome_ode, t_span, x0, t_eval=t_eval)

# ---- Plot ----
plt.figure()
plt.plot(sol.t, sol.y[0], label="Brain state b(t)")
plt.xlabel("Time")
plt.ylabel("Brain health")
plt.legend()
plt.show()


plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel("b")
plt.ylabel("db/dt")
plt.title("Phase space")
plt.show()
