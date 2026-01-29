import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---- MODEL PARAMETERS ----
# Brain System
m0 = 1.0      # Baseline neural inertia (e.g., metabolic capacity)
k = 0.8       # Homeostatic restoring force (brain resilience)
gamma = 0.15  # Dissipation (metabolic cost, entropy)

# Exposure Coupling
alpha = 0.15  # Sensitivity of inertia to exposure (weaker, more stable)
lam = 0.25    # Direct forcing strength of exposure on brain state

# Exposome Trajectory (Chronic stressor)
A = 2.0       # Max exposure level
tau = 20.0    # Exposure accumulation timescale

def exposome(t):
    """Returns exposure level e(t) and its time derivative e_dot(t)."""
    e = A * (1 - np.exp(-t / tau))
    e_dot = (A / tau) * np.exp(-t / tau)
    return e, e_dot

def brain_exposome_ode(t, x):
    """
    ODE derived from a Lagrangian framework.
    States: x = [b, v] where b = brain state, v = db/dt.
    """
    b, v = x
    e, e_dot = exposome(t)
    m = m0 * (1 + alpha * e)  # Exposure-modulated inertia

    # Euler-Lagrange type equation with dissipation
    dv = -(1/m) * (m0 * alpha * e_dot * v + gamma * v + k * b + lam * e)
    db = v
    return [db, dv]

# ---- SIMULATION ----
t_span = (0, 50)
t_eval = np.linspace(*t_span, 500)
x0 = [0.5, 0.0]  # Initial state

sol = solve_ivp(brain_exposome_ode, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-8)

# ---- CRITICAL STEP: Define a Real-World Neuroimaging Phenotype ----
# Let's assume the abstract brain state 'b' represents normalized hippocampal volume.
# A healthy young baseline (b=0.5) degrades towards 0 with age/disease.
# We create a simulated "measured" phenotype with some observation noise.
np.random.seed(42)  # For reproducibility
measurement_noise = 0.02
simulated_hippocampal_volume = sol.y[0] + np.random.randn(len(sol.t)) * measurement_noise

# ---- VISUALIZATION ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Panel 1: Simulated Neuroimaging Phenotype
axes[0, 0].plot(sol.t, simulated_hippocampal_volume, 'b-', alpha=0.7, label='Noisy "Measurement"')
axes[0, 0].plot(sol.t, sol.y[0], 'k--', linewidth=1.5, label='Model Trajectory (True)')
axes[0, 0].set_xlabel('Time (e.g., Years)')
axes[0, 0].set_ylabel('Normalized Hippocampal Volume')
axes[0, 0].set_title('Panel 1: Simulated Longitudinal Neuroimaging Data')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Phase Portrait
axes[0, 1].plot(sol.y[0], sol.y[1], 'g-')
axes[0, 1].set_xlabel('Brain State b')
axes[0, 1].set_ylabel('Rate of Change v')
axes[0, 1].set_title('Panel 2: Phase Space (b vs v)')
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: CORRECTED - Exposome Input
# >>> THE FIX: Calculate exposure values using the defined function <<<
e_vals = np.array([exposome(t)[0] for t in sol.t])  # Get just e(t), not e_dot
axes[1, 0].plot(sol.t, e_vals, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Exposure e(t)')
axes[1, 0].set_title('Panel 3: Exposome Input (e.g., Cumulative PM2.5)')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Effective Inertia
m_vals = m0 * (1 + alpha * e_vals)
axes[1, 1].plot(sol.t, m_vals, 'm-', linewidth=2)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Effective Inertia m(t)')
axes[1, 1].set_title('Panel 4: Exposure-Modulated Inertia')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---- Quantitative Output ----
# Calculate the "Brain Age Gap": deviation from the starting healthy state.
initial_volume = simulated_hippocampal_volume[0]
final_volume = simulated_hippocampal_volume[-1]
volume_loss = initial_volume - final_volume
volume_loss_percent = (volume_loss / initial_volume) * 100

print("=== SIMULATION RESULTS ===")
print(f"Initial 'Hippocampal Volume': {initial_volume:.3f}")
print(f"Final 'Hippocampal Volume':   {final_volume:.3f}")
print(f"Absolute Volume Loss:         {volume_loss:.3f}")
print(f"Percentage Volume Loss:       {volume_loss_percent:.2f}%")
print("\nInterpretation: This simulates how a chronic exposure (Panel 3) could")
print("drive measurable brain atrophy (Panel 1) via dynamic equations.")