"""
Optuna Tutorial: Parameter Optimization for Brightness Detection
===============================================================

This tutorial replicates the brightness detection experiment from tutorial_example_P1.ipynb
but uses Optuna for hyperparameter optimization to find the best neural network parameters
that minimize reconstruction error.

The experiment:
- Present visual stimuli of different brightness levels
- Encode stimuli through a population of NEST neurons with different tuning curves
- Use Optuna to optimize neural parameters (tuning curves, thresholds, etc.)
"""

import nest
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os
import pickle
from datetime import datetime
try:
    from tools.helper import filter_spikes_parallel, compute_capacity
except ImportError:
    print("Warning: Could not import helper functions. Some analysis may be limited.")

class OptunaBrightnessExperiment:
    """NEST-based brightness detection with Optuna optimization"""
    
    def __init__(self, n_values=5, n_trials=20, dt=0.1):
        self.n_values = n_values
        self.n_trials = n_trials
        self.dt = dt
        
        # Generate brightness levels (fixed)
        self.brightness = np.linspace(0, 100, n_values)
        
        # Store results for analysis
        self.trial_results = []
        
    def objective(self, trial):
        """Optuna objective function to minimize reconstruction error"""
        
        # Suggest hyperparameters
        tuning_mean = trial.suggest_float('tuning_mean', 500.0, 2000.0)
        tuning_std = trial.suggest_float('tuning_std', 50.0, 500.0)
        bias_current = trial.suggest_float('bias_current', 10.0, 500.0)
        threshold_mean = trial.suggest_float('threshold_mean', -60.0, -40.0)
        threshold_std = trial.suggest_float('threshold_std', 1.0, 10.0)
        v_reset = trial.suggest_float('v_reset', -80.0, -60.0)
        tau_m = trial.suggest_float('tau_m', 5.0, 50.0)
        
        # Suggest network structure parameters
        n_neurons = trial.suggest_int('n_neurons', 20, 200)
        stimulus_duration = trial.suggest_float('stimulus_duration', 25.0, 150.0)
        
        # Generate stimulus based on suggested parameters
        stimulus_order = np.random.permutation(self.n_trials * self.n_values) % self.n_values
        x = np.array([(self.brightness[n]/100.) + 0.1 for n in stimulus_order])
        times = np.arange(self.dt, self.n_trials * self.n_values * stimulus_duration, stimulus_duration)
        sim_time = self.n_trials * self.n_values * stimulus_duration
        
        try:
            # Setup NEST simulation
            nest.ResetKernel()
            nest.SetKernelStatus({
                'resolution': self.dt,
                'print_time': False,  # Suppress output during optimization
                'local_num_threads': 16
            })
            
            # Generate tuning curves
            np.random.seed(trial.number)  # Reproducible randomness per trial
            tuning = tuning_std * np.random.randn(n_neurons) + tuning_mean
            
            # Generate neuron parameters
            thresholds = threshold_std * np.random.randn(n_neurons) + threshold_mean
            v_init = np.random.uniform(low=-70., high=-50., size=n_neurons)
            
            # Create NEST network
            step_generator = nest.Create('step_current_generator', n_neurons)
            neurons = nest.Create('iaf_psc_exp', n_neurons, {
                'I_e': bias_current,
                'V_reset': v_reset,
                'tau_m': tau_m
            })
            spike_detector = nest.Create('spike_recorder')
            
            # Set individual neuron parameters and connect
            amplitudes = np.zeros((n_neurons, len(x)))
            for n in range(n_neurons):
                amplitudes[n, :] = x * tuning[n]
                
                # Set neuron-specific parameters
                neurons[n].set({'V_m': v_init[n], 'V_th': thresholds[n]})
                step_generator[n].set({
                    'amplitude_times': times,
                    'amplitude_values': amplitudes[n]
                })
                
                # Connect
                nest.Connect(step_generator[n], neurons[n])
            
            nest.Connect(neurons, spike_detector)
            
            # Run simulation
            nest.Simulate(sim_time)
            
            # Get spike data
            spike_data = spike_detector.get('events')
            spike_times = spike_data['times']
            neuron_ids = spike_data['senders']
            
            # Filter spikes to get continuous signals
            if len(spike_times) > 0:
                filtered_states = filter_spikes_parallel(
                    spike_times, neuron_ids, n_neurons,
                    t_start=0., t_stop=sim_time, dt=self.dt, tau=20.0, n_processes=16
                )
                
                # Create target signal
                target_signal = np.repeat(x, int(stimulus_duration / self.dt))
                
                # Linear decoding
                if filtered_states.shape[1] > 0 and len(target_signal) > 0:
                    reg = LinearRegression(fit_intercept=False)
                    reg.fit(filtered_states.T, target_signal)
                    reconstructed = reg.predict(filtered_states.T)
                    
                    # Compute MSE
                    mse = np.mean((target_signal - reconstructed) ** 2)
                    
                    # Store results for this trial
                    trial_result = {
                        'trial_number': trial.number,
                        'params': trial.params,
                        'mse': mse,
                    }
                    self.trial_results.append(trial_result)
                    
                    return mse
                else:
                    return float('inf')  # No valid data
            else:
                return float('inf')  # No spikes
                
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    def run_optimization(self, n_trials=100, timeout=3600):
        """Run Optuna optimization"""
        print(f"Starting Optuna optimization with {n_trials} trials...")
        print(f"Timeout: {timeout} seconds")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        return study
    
    def analyze_optimization_results(self, study):
        """Analyze optimization results"""
        print("\nOptimization Results:")
        print("=" * 50)
        
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        if study.best_trial is not None:
            print(f"\nBest trial:")
            print(f"  Value (MSE): {study.best_trial.value:.6f}")
            print(f"  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value:.4f}")
                
        return study.best_trial
    
    def plot_optimization_results(self, study):
        """Plot optimization results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Optimization history
        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]
        
        axes[0, 0].plot(trial_numbers, trial_values, 'b-', alpha=0.6)
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())
            
            axes[0, 1].barh(params, values)
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].grid(True, alpha=0.3)
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Plot 3: Parallel coordinate plot
        try:
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            # Note: This creates a separate plotly figure, so we'll skip it for matplotlib
            axes[0, 2].text(0.5, 0.5, 'See separate\nparallel coordinate plot', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        except:
            axes[0, 2].text(0.5, 0.5, 'Parallel coordinate\nplot not available', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # Plot 5: Best parameters visualization
        if study.best_trial:
            param_names = list(study.best_trial.params.keys())
            param_values = list(study.best_trial.params.values())
            
            axes[1, 1].bar(range(len(param_names)), param_values)
            axes[1, 1].set_xticks(range(len(param_names)))
            axes[1, 1].set_xticklabels(param_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Parameter Value')
            axes[1, 1].set_title('Best Parameters')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Trial success rate
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        labels = ['Complete', 'Pruned', 'Failed']
        sizes = [len(complete_trials), len(pruned_trials), len(failed_trials)]
        colors = ['green', 'orange', 'red']
        
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 2].set_title('Trial Status Distribution')
        
        plt.tight_layout()
        plt.savefig('/home/neuro/repos/_education/computational-biology-2025/plots/optuna_optimization_results.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_best_trial_analysis(self, study):
        """Run detailed analysis with best parameters"""
        if study.best_trial is None:
            print("No best trial found!")
            return None
            
        print("\nRunning detailed analysis with best parameters...")
        
        # Use best parameters
        best_params = study.best_trial.params
        
        # Setup simulation with best parameters
        nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': self.dt,
            'print_time': True,
            'local_num_threads': 8
        })
        
        # Generate network with best parameters
        np.random.seed(42)  # Fixed seed for reproducibility
        tuning = best_params['tuning_std'] * np.random.randn(self.n_neurons) + best_params['tuning_mean']
        thresholds = best_params['threshold_std'] * np.random.randn(self.n_neurons) + best_params['threshold_mean']
        v_init = np.random.uniform(low=-70., high=-50., size=self.n_neurons)
        
        # Create network
        step_generator = nest.Create('step_current_generator', self.n_neurons)
        neurons = nest.Create('iaf_psc_exp', self.n_neurons, {
            'I_e': best_params['bias_current'],
            'V_reset': best_params['v_reset'],
            'tau_m': best_params['tau_m']
        })
        spike_detector = nest.Create('spike_recorder')
        
        # Setup connections
        amplitudes = np.zeros((self.n_neurons, len(self.x)))
        for n in range(self.n_neurons):
            amplitudes[n, :] = self.x * tuning[n]
            neurons[n].set({'V_m': v_init[n], 'V_th': thresholds[n]})
            step_generator[n].set({
                'amplitude_times': self.times,
                'amplitude_values': amplitudes[n]
            })
            nest.Connect(step_generator[n], neurons[n])
        
        nest.Connect(neurons, spike_detector)
        
        # Run simulation
        nest.Simulate(self.sim_time)
        
        # Analyze results
        spike_data = spike_detector.get('events')
        spike_times = spike_data['times']
        neuron_ids = spike_data['senders']
        
        # Filter spikes
        filtered_states = filter_spikes(
            spike_times, neuron_ids, self.n_neurons,
            t_start=0., t_stop=self.sim_time, dt=self.dt, tau=20.0
        )
        
        # Create target signal
        signal_steps = filtered_states.shape[1]
        target_signal = np.repeat(self.x, signal_steps // len(self.x))
        target_signal = target_signal[:signal_steps]
        
        # Linear decoding
        reg = LinearRegression(fit_intercept=False)
        reg.fit(filtered_states.T, target_signal)
        reconstructed = reg.predict(filtered_states.T)
        
        # Compute metrics
        mse = np.mean((target_signal - reconstructed) ** 2)
        
        results = {
            'best_params': best_params,
            'spike_times': spike_times,
            'neuron_ids': neuron_ids,
            'filtered_states': filtered_states,
            'target_signal': target_signal,
            'reconstructed': reconstructed,
            'mse': mse,
            'tuning': tuning
        }
        
        return results
    
    def save_results(self, study, results=None):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save study
        study_filename = f'/home/neuro/repos/_education/computational-biology-2025/optuna_study_{timestamp}.pkl'
        with open(study_filename, 'wb') as f:
            pickle.dump(study, f)
        
        # Save trial results
        results_filename = f'/home/neuro/repos/_education/computational-biology-2025/optuna_results_{timestamp}.pkl'
        with open(results_filename, 'wb') as f:
            pickle.dump({
                'trial_results': self.trial_results,
                'best_analysis': results
            }, f)
        
        print(f"Results saved to:")
        print(f"  Study: {study_filename}")
        print(f"  Results: {results_filename}")

def main():
    """Main execution function"""
    print("Optuna Brightness Detection Parameter Optimization")
    print("=" * 55)
    
    # Create experiment
    experiment = OptunaBrightnessExperiment(
        n_values=5,
        n_trials=15,    # Fewer trials for faster optimization
        dt=0.1
    )
    
    print(f"Experiment setup:")
    print(f"- Network size and stimulus duration will be optimized")
    print(f"- {experiment.n_values} brightness levels: {experiment.brightness}")
    print(f"- {experiment.n_trials} trials per level")
    print()
    
    # Run optimization
    study = experiment.run_optimization(n_trials=50, timeout=1800)  # 30 minutes max
    
    # Analyze results
    best_trial = experiment.analyze_optimization_results(study)
    
    # Plot optimization results
    experiment.plot_optimization_results(study)
    
    # Run detailed analysis with best parameters
    if best_trial is not None:
        best_results = experiment.run_best_trial_analysis(study)
        
        if best_results is not None:
            print(f"\nBest trial detailed results:")
            print(f"MSE: {best_results['mse']:.6f}")
    else:
        best_results = None
    
    # Save results
    experiment.save_results(study, best_results)
    
    return experiment, study, best_results

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    os.makedirs('/home/neuro/repos/_education/computational-biology-2025/plots', exist_ok=True)
    
    # Run experiment
    experiment, study, results = main()
    
    print("\nOptuna optimization completed successfully!")
    print("Results saved to plots/optuna_optimization_results.png")
