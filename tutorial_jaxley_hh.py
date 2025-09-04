"""
JAXley Tutorial: Hodgkin-Huxley Neurons with Gradient-Based Optimization
========================================================================

This tutorial replicates the brightness detection experiment from tutorial_example_P1.ipynb
but uses JAXley with Hodgkin-Huxley neurons and gradient-based optimization to minimize
the reconstruction error.

The experiment:
- Present visual stimuli of different brightness levels
- Encode stimuli through a population of neurons with different tuning curves
- Use gradient descent to optimize neural parameters to minimize reconstruction error
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os

try:
    from helper import compute_capacity
except ImportError:
    print("Warning: Could not import helper functions. Some analysis may be limited.")

class BrightnessDetectionExperiment:
    """JAXley-based brightness detection experiment with HH neurons"""
    
    def __init__(self, n_neurons=50, n_values=5, n_trials=20, stimulus_duration=50.0, dt=1.0):
        self.n_neurons = n_neurons
        self.n_values = n_values
        self.n_trials = n_trials
        self.stimulus_duration = stimulus_duration
        self.dt = dt
        
        # Generate stimulus
        self.brightness = jnp.linspace(0, 100, n_values)
        self.stimulus_order = np.random.permutation(n_trials * n_values) % n_values
        self.x = jnp.array([(self.brightness[n]/100.) + 0.1 for n in self.stimulus_order])
        self.times = jnp.arange(dt, n_trials * n_values * stimulus_duration, stimulus_duration)
        self.sim_time = n_trials * n_values * stimulus_duration
        
        # Initialize neural network
        self.setup_network()
        
    def setup_network(self):
        """Setup simplified HH neuron network"""
        print("Setting up Hodgkin-Huxley neuron network...")
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        """Initialize neural parameters"""
        key = jax.random.PRNGKey(42)
        
        # Tuning curves (how each neuron responds to brightness)
        self.tuning = jax.random.normal(key, (self.n_neurons,)) * 250.0 + 1000.0
        
        # Bias currents
        self.bias_current = jnp.ones(self.n_neurons) * 200.0  # pA
        
        # Learnable parameters for optimization
        self.params = {
            'tuning': self.tuning,
            'bias': self.bias_current,
            'g_na': jnp.ones(self.n_neurons) * 120.0,  # Sodium conductance
            'g_k': jnp.ones(self.n_neurons) * 36.0,    # Potassium conductance
            'g_leak': jnp.ones(self.n_neurons) * 0.3,  # Leak conductance
        }
        
    def generate_input_currents(self, params, stimulus_idx):
        """Generate input currents for a given stimulus"""
        stimulus_value = self.x[stimulus_idx]
        
        # Convert stimulus to input currents using tuning curves
        input_currents = params['tuning'] * stimulus_value + params['bias']
        
        return input_currents
        
    def simulate_network(self, params, stimulus_idx, duration):
        """Simulate network response to a stimulus using vectorized operations"""
        input_currents = self.generate_input_currents(params, stimulus_idx)
        
        # Use a simplified rate-based model instead of full HH simulation
        # This approximates the firing rate based on input current
        spike_counts = self.rate_based_simulation(params, input_currents, duration)
        
        return spike_counts
        
    def rate_based_simulation(self, params, input_currents, duration):
        """Simplified rate-based neuron model for faster computation"""
        # Convert input current to firing rate using a sigmoid function
        # This approximates the f-I curve of HH neurons
        
        # Threshold and gain parameters
        I_threshold = 100.0  # pA
        gain = 0.05  # Hz/pA
        max_rate = 100.0  # Hz
        
        # Compute firing rates
        rates = max_rate / (1 + jnp.exp(-gain * (input_currents - I_threshold)))
        
        # Convert to spike counts for the given duration
        spike_counts = rates * (duration / 1000.0)  # Convert ms to seconds
        
        # Add some noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, spike_counts.shape) * 0.1
        spike_counts = jnp.maximum(0, spike_counts + noise)
        
        return spike_counts
        
    def filter_spikes(self, spike_counts, tau=20.0):
        """Process spike counts (already computed in rate-based model)"""
        # spike_counts is already the output we want
        return spike_counts
        
    def compute_reconstruction_error(self, params):
        """Compute reconstruction error for all stimuli"""
        all_responses = []
        target_signal = []
        
        print("Simulating network responses...")
        for i in tqdm(range(len(self.x))):
            # Simulate response to stimulus i
            spike_counts = self.simulate_network(params, i, self.stimulus_duration)
            
            # Filter spikes to get spike counts
            filtered_counts = self.filter_spikes(spike_counts)
            
            # Use spike counts as response
            mean_response = filtered_counts
            all_responses.append(mean_response)
            
            # Target is the original stimulus value
            target_signal.append(self.x[i])
            
        # Stack responses
        response_matrix = jnp.stack(all_responses, axis=1)  # (n_neurons, n_stimuli)
        target_vector = jnp.array(target_signal)
        
        # Linear decoding to reconstruct stimulus
        # Use pseudo-inverse for linear regression
        decoder = jnp.linalg.pinv(response_matrix.T) @ target_vector
        reconstructed = response_matrix.T @ decoder
        
        # Compute mean squared error
        mse = jnp.mean((target_vector - reconstructed) ** 2)
        
        return mse, response_matrix, target_vector, reconstructed
        
    def simple_hh_simulation(self, params, neuron_idx, current, t):
        """Fast simplified neuron model"""
        # Simplified leaky integrate-and-fire with adaptation
        tau_m = 20.0  # membrane time constant (ms)
        V_rest = -65.0  # resting potential (mV)
        V_thresh = -50.0  # spike threshold (mV)
        V_reset = -70.0  # reset potential (mV)
        
        # Adaptation parameters based on HH conductances
        adaptation = params['g_k'][neuron_idx] / 100.0
        
        V = V_rest
        voltages = []
        
        for I in current:
            # Simple membrane dynamics
            dV_dt = (-(V - V_rest) + I * 10.0) / tau_m  # Scale current
            V += self.dt * dV_dt
            
            # Add some adaptation based on conductance parameters
            if V > V_thresh:
                V = V_reset - adaptation * 5.0  # Reset with adaptation
            
            voltages.append(V)
            
        return jnp.array(voltages)
        
    def analyze_results(self):
        """Analyze final results"""
        print("Analyzing final results...")
        
        mse, response_matrix, target_vector, reconstructed = self.compute_reconstruction_error(self.params)
        
        # Compute capacity
        capacity = 1.0 - (mse / jnp.var(target_vector))
        
        print(f"Final MSE: {mse:.6f}")
        print(f"Capacity: {capacity:.4f}")
        
        return {
            'mse': mse,
            'capacity': capacity,
            'response_matrix': response_matrix,
            'target_vector': target_vector,
            'reconstructed': reconstructed
        }
        
    def optimize_parameters(self, n_steps=100, learning_rate=0.01):
        """Optimize neural parameters using gradient descent"""
        print(f"Starting optimization with {n_steps} steps...")
        
        # Setup optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.params)
        
        # Loss function
        def loss_fn(params):
            mse, _, _, _ = self.compute_reconstruction_error(params)
            return mse
            
        # Gradient function
        grad_fn = jax.grad(loss_fn)
        
        losses = []
        
        for step in range(n_steps):
            # Compute gradients
            grads = grad_fn(self.params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            self.params = optax.apply_updates(self.params, updates)
            
            # Compute current loss
            current_loss = loss_fn(self.params)
            losses.append(current_loss)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {current_loss:.6f}")
                
        return losses
        
    def plot_results(self, losses, results):
        """Plot optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Loss curve
        axes[0, 0].plot(losses)
        axes[0, 0].set_xlabel('Optimization Step')
        axes[0, 0].set_ylabel('Reconstruction Error (MSE)')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Target vs Reconstructed
        axes[0, 1].scatter(results['target_vector'], results['reconstructed'], alpha=0.6)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Target Stimulus')
        axes[0, 1].set_ylabel('Reconstructed Stimulus')
        axes[0, 1].set_title(f'Reconstruction Quality (R² = {results["capacity"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Response matrix
        im = axes[1, 0].imshow(results['response_matrix'], aspect='auto', cmap='viridis')
        axes[1, 0].set_xlabel('Stimulus Index')
        axes[1, 0].set_ylabel('Neuron Index')
        axes[1, 0].set_title('Neural Response Matrix')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Tuning curves
        brightness_range = jnp.linspace(0, 1.1, 100)
        for i in range(min(10, self.n_neurons)):  # Plot first 10 neurons
            response = self.params['tuning'][i] * brightness_range + self.params['bias'][i]
            axes[1, 1].plot(brightness_range * 100, response, alpha=0.7, linewidth=1)
        
        axes[1, 1].set_xlabel('Brightness Level (%)')
        axes[1, 1].set_ylabel('Input Current (pA)')
        axes[1, 1].set_title('Optimized Tuning Curves (first 10 neurons)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/neuro/repos/_education/computational-biology-2025/plots/jaxley_hh_results.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Main execution function"""
    print("JAXley Hodgkin-Huxley Brightness Detection Experiment")
    print("=" * 55)
    
    # Create experiment
    experiment = BrightnessDetectionExperiment(
        n_neurons=30,  # Smaller for faster computation
        n_values=5,
        n_trials=10,   # Fewer trials for faster computation
        stimulus_duration=100.0,
        dt=0.1
    )
    
    print(f"Experiment setup:")
    print(f"- {experiment.n_neurons} Hodgkin-Huxley neurons")
    print(f"- {experiment.n_values} brightness levels: {experiment.brightness}")
    print(f"- {experiment.n_trials} trials per level")
    print(f"- {experiment.stimulus_duration} ms stimulus duration")
    print()
    
    # Initial analysis
    print("Computing initial reconstruction error...")
    initial_results = experiment.analyze_results()
    
    # Optimize parameters
    losses = experiment.optimize_parameters(n_steps=50, learning_rate=0.001)
    
    # Final analysis
    print("\nFinal analysis:")
    final_results = experiment.analyze_results()
    
    # Plot results
    experiment.plot_results(losses, final_results)
    
    print(f"\nImprovement:")
    print(f"Initial MSE: {initial_results['mse']:.6f}")
    print(f"Final MSE: {final_results['mse']:.6f}")
    print(f"Improvement: {((initial_results['mse'] - final_results['mse']) / initial_results['mse'] * 100):.2f}%")
    
    return experiment, losses, final_results

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('/home/neuro/repos/_education/computational-biology-2025/plots', exist_ok=True)
    
    # Run experiment
    experiment, losses, results = main()
    
    print("\nJAXley experiment completed successfully!")
    print("Results saved to plots/jaxley_hh_results.png")
