import numpy as np
import matplotlib.pyplot as plt

class BiologicalPruningSynapse:
    """A class to represent a single synapse and its pruning dynamics."""
    def __init__(self, initial_strength, neuron_id):
        self.strength = initial_strength
        self.neuron_id = neuron_id
        self.prune_tag = 0.0  # Accumulated "eat me" signal
        self.active = True
        self.activity_history = []

    def update_activity(self, time_step):
        """Simulate activity based on strength plus noise."""
        # Base activity is proportional to strength, plus noise and a periodic signal
        base_activity = self.strength * 0.5
        noise = np.random.normal(0, 0.1)
        periodic = 0.2 * np.sin(time_step * 0.5 + self.neuron_id)
        activity = max(0, base_activity + noise + periodic)  # No negative activity
        self.activity_history.append(activity)
        return activity

    def update_prune_tag(self, avg_neuron_activity):
        """Update the 'eat me' signal tag. Less active synapses accumulate more tag."""
        recent_activity = np.mean(self.activity_history[-5:]) if self.activity_history else 0
        if avg_neuron_activity > 0:
            # Normalized inactivity ratio: lower relative activity -> faster tag accumulation
            activity_ratio = recent_activity / avg_neuron_activity
            inactivity_signal = max(0, 1.0 - activity_ratio)  # 1 if silent, 0 if at/above average
            self.prune_tag += inactivity_signal * 0.05  # Accumulation rate
            self.prune_tag *= 0.95  # Satural decay
        return self.prune_tag

    def attempt_prune(self, microglial_activity_level):
        """Stochastic pruning event based on tag level and microglial presence."""
        if not self.active:
            return False
        # Probability of pruning increases with tag and general microglial activity
        prune_probability = min(0.5, self.prune_tag * microglial_activity_level * 0.1)
        if np.random.random() < prune_probability:
            self.active = False
            self.strength = 0.0
            return True
        return False

# --- Simulation Parameters ---
np.random.seed(42)  # For reproducibility
num_synapses = 300
simulation_steps = 200
microglial_activity = 1.0  # Can be varied to model different pruning phases

# --- Create a population of synapses ---
# Simulate synapses belonging to 3 different neurons
synapses = []
for neuron_id in range(3):
    for _ in range(num_synapses // 3):
        initial_strength = np.random.exponential(scale=1.0)
        synapses.append(BiologicalPruningSynapse(initial_strength, neuron_id))

# --- Track statistics over time ---
active_counts = []
strength_means = []
prune_tags_avg = []

# --- Main Simulation Loop ---
for step in range(simulation_steps):
    active_synapses = [s for s in synapses if s.active]
    
    # 1. Calculate average activity per neuron for competition
    neuron_activities = {}
    for s in active_synapses:
        act = s.update_activity(step)
        neuron_activities[s.neuron_id] = neuron_activities.get(s.neuron_id, []) + [act]
    avg_neuron_activity = {nid: np.mean(acts) for nid, acts in neuron_activities.items()}
    
    # 2. Update prune tags based on relative activity
    current_prune_tags = []
    for s in active_synapses:
        tag = s.update_prune_tag(avg_neuron_activity.get(s.neuron_id, 0.1))
        current_prune_tags.append(tag)
    
    # 3. Attempt pruning events
    prune_events_this_step = 0
    for s in active_synapses:
        if s.attempt_prune(microglial_activity):
            prune_events_this_step += 1
    
    # 4. Record statistics
    active_counts.append(len(active_synapses))
    strength_means.append(np.mean([s.strength for s in active_synapses]))
    prune_tags_avg.append(np.mean(current_prune_tags))
    
    # Optional: Print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}: Active synapses: {len(active_synapses)}, Pruning events this step: {prune_events_this_step}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Number of active synapses over time
axes[0, 0].plot(active_counts, linewidth=2)
axes[0, 0].set_title('Synapse Survival (Pruning Curve)')
axes[0, 0].set_xlabel('Simulation Step')
axes[0, 0].set_ylabel('Active Synapses')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Average synaptic strength of survivors
axes[0, 1].plot(strength_means, linewidth=2, color='orange')
axes[0, 1].set_title('Average Strength of Surviving Synapses')
axes[0, 1].set_xlabel('Simulation Step')
axes[0, 1].set_ylabel('Mean Strength')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Average "prune tag" level over time
axes[1, 0].plot(prune_tags_avg, linewidth=2, color='red')
axes[1, 0].set_title('Average "Prune Tag" Level')
axes[1, 0].set_xlabel('Simulation Step')
axes[1, 0].set_ylabel('Tag Level')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final distribution of synapse strengths
final_strengths = [s.strength for s in synapses if s.active]
axes[1, 1].hist(final_strengths, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Final Strength Distribution (Survivors)')
axes[1, 1].set_xlabel('Synaptic Strength')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Final Simulation Summary ---
print(f"\n=== Simulation Complete ===")
print(f"Initial synapses: {num_synapses}")
print(f"Final active synapses: {active_counts[-1]}")
print(f"Pruning rate: {100 * (1 - active_counts[-1] / num_synapses):.1f}%")
final_avg_strength = np.mean([s.strength for s in synapses if s.active])
print(f"Final average strength of survivors: {final_avg_strength:.3f}")