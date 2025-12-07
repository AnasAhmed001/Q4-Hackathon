---
title: Chapter 9 - Sim-to-Real Transfer Strategies
description: Explore techniques and strategies for transferring AI models and behaviors from simulation to real humanoid robots.
sidebar_position: 31
---

# Chapter 9 - Sim-to-Real Transfer Strategies

Sim-to-real transfer is the process of taking AI models, control policies, and behaviors developed in simulation and successfully deploying them on physical robots. This is particularly challenging for humanoid robots due to their complex dynamics, multiple degrees of freedom, and the significant differences between simulated and real environments. This chapter explores various techniques and strategies to bridge the reality gap and achieve effective sim-to-real transfer.

## 9.1 Introduction to Sim-to-Real Transfer

The reality gap refers to the differences between simulation and the real world that can cause policies learned in simulation to fail when deployed on physical robots. These differences include:

- **Dynamics mismatch**: Differences in friction, inertia, and contact models
- **Sensor noise**: Real sensors have noise, delays, and imperfections
- **Actuator limitations**: Real actuators have delays, limited torque, and backlash
- **Environmental variations**: Lighting, textures, and object properties differ
- **Model inaccuracies**: Simplified models in simulation vs. complex real systems

### 9.1.1 Challenges for Humanoid Robots

Humanoid robots face unique sim-to-real challenges:

- **Balance and stability**: Small modeling errors can cause catastrophic falls
- **Complex kinematics**: Many joints with potential for modeling errors
- **Contact dynamics**: Foot-ground interactions are critical and hard to model
- **Sensor fusion**: Multiple sensor types with different characteristics
- **Real-time constraints**: Balance control requires high-frequency updates

## 9.2 Domain Randomization Techniques

Domain randomization is a key technique for improving sim-to-real transfer by training policies in diverse simulated environments:

### 9.2.1 Visual Domain Randomization

```python
import numpy as np
import cv2
import random

class VisualDomainRandomizer:
    def __init__(self):
        self.lighting_params = {
            'intensity_range': (0.5, 2.0),
            'color_temperature_range': (5000, 8000),
            'position_variance': (0.2, 0.2, 0.2)
        }

        self.material_params = {
            'albedo_range': (0.1, 0.9),
            'roughness_range': (0.1, 0.9),
            'metallic_range': (0.0, 1.0)
        }

    def randomize_lighting(self, scene):
        """Randomize lighting conditions in the scene"""
        # Randomize light intensity
        intensity_factor = random.uniform(
            self.lighting_params['intensity_range'][0],
            self.lighting_params['intensity_range'][1]
        )

        # Randomize light position
        position_offset = np.random.uniform(
            -np.array(self.lighting_params['position_variance']),
            np.array(self.lighting_params['position_variance'])
        )

        # Apply changes to scene
        scene.light_intensity *= intensity_factor
        scene.light_position += position_offset

        return scene

    def randomize_materials(self, objects):
        """Randomize material properties of objects"""
        for obj in objects:
            # Randomize albedo (base color)
            obj.albedo = np.random.uniform(
                self.material_params['albedo_range'][0],
                self.material_params['albedo_range'][1],
                size=3
            )

            # Randomize roughness
            obj.roughness = random.uniform(
                self.material_params['roughness_range'][0],
                self.material_params['roughness_range'][1]
            )

            # Randomize metallic
            obj.metallic = random.uniform(
                self.material_params['metallic_range'][0],
                self.material_params['metallic_range'][1]
            )

        return objects

    def add_sensor_noise(self, image):
        """Add realistic sensor noise to simulate real camera characteristics"""
        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
        image = image.astype(np.float32) + gaussian_noise

        # Add salt and pepper noise
        if random.random() < 0.01:  # 1% chance of salt&pepper
            s_vs_p = 0.5
            amount = 0.004
            noisy = image.copy()

            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
            noisy[coords[0], coords[1]] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
            noisy[coords[0], coords[1]] = 0

            image = noisy

        # Add motion blur
        if random.random() < 0.1:  # 10% chance of motion blur
            kernel_size = random.randint(2, 5)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            image = cv2.filter2D(image, -1, kernel)

        return np.clip(image, 0, 1).astype(np.float32)

class PhysicsDomainRandomizer:
    def __init__(self):
        self.dynamics_params = {
            'friction_range': (0.4, 0.9),
            'mass_variance': 0.1,  # ±10% mass variation
            'inertia_variance': 0.1,  # ±10% inertia variation
            'damping_range': (0.01, 0.1),
            'com_offset_range': (0.001, 0.005)  # Center of mass offset
        }

    def randomize_robot_dynamics(self, robot):
        """Randomize robot dynamics parameters"""
        for link in robot.links:
            # Randomize friction coefficients
            link.friction_static = random.uniform(
                self.dynamics_params['friction_range'][0],
                self.dynamics_params['friction_range'][1]
            )
            link.friction_dynamic = link.friction_static * random.uniform(0.8, 1.0)

            # Randomize mass with variance
            mass_variation = random.uniform(
                1 - self.dynamics_params['mass_variance'],
                1 + self.dynamics_params['mass_variance']
            )
            link.mass *= mass_variation

            # Randomize inertia with variance
            inertia_variation = random.uniform(
                1 - self.dynamics_params['inertia_variance'],
                1 + self.dynamics_params['inertia_variance']
            )
            link.inertia *= inertia_variation

            # Add center of mass offset
            com_offset = np.random.uniform(
                -np.array(self.dynamics_params['com_offset_range']),
                np.array(self.dynamics_params['com_offset_range'])
            )
            link.center_of_mass += com_offset

        return robot

    def randomize_contact_properties(self, contacts):
        """Randomize contact properties for more realistic simulation"""
        for contact in contacts:
            # Randomize contact stiffness and damping
            contact.stiffness = random.uniform(1e4, 1e6)
            contact.damping = random.uniform(1e2, 1e4)

            # Randomize friction anisotropy
            contact.friction_direction_1 = np.random.uniform(-1, 1, 3)
            contact.friction_direction_1 /= np.linalg.norm(contact.friction_direction_1)
            contact.friction_coefficient_1 = random.uniform(0.3, 0.9)
            contact.friction_coefficient_2 = contact.friction_coefficient_1 * random.uniform(0.5, 1.5)

        return contacts
```

### 9.2.2 Adaptive Domain Randomization

```python
class AdaptiveDomainRandomizer:
    def __init__(self, initial_range_factor=1.0, max_range_factor=5.0):
        self.range_factor = initial_range_factor
        self.max_range_factor = max_range_factor
        self.performance_history = []
        self.update_threshold = 10  # Update every N episodes

    def update_randomization_range(self, current_performance):
        """Adaptively adjust randomization range based on performance"""
        self.performance_history.append(current_performance)

        if len(self.performance_history) >= self.update_threshold:
            recent_performance = self.performance_history[-self.update_threshold:]
            performance_std = np.std(recent_performance)

            # If performance is stable (low variance), increase randomization
            if performance_std < 0.05:  # Threshold for stability
                self.range_factor = min(self.range_factor * 1.1, self.max_range_factor)
                print(f"Increasing randomization range factor to {self.range_factor:.2f}")
            # If performance is unstable (high variance), decrease randomization
            elif performance_std > 0.2:  # Threshold for instability
                self.range_factor = max(self.range_factor * 0.9, 1.0)
                print(f"Decreasing randomization range factor to {self.range_factor:.2f}")

            # Keep only recent history
            self.performance_history = self.performance_history[-self.update_threshold:]

    def get_randomized_parameter(self, nominal_value, variation_range):
        """Get a randomized parameter value based on current range factor"""
        variation = variation_range * self.range_factor
        return random.uniform(nominal_value - variation, nominal_value + variation)

class CurriculumDomainRandomizer:
    def __init__(self, num_stages=5):
        self.num_stages = num_stages
        self.current_stage = 0
        self.stage_progress = 0.0

    def advance_curriculum(self, success_rate):
        """Advance curriculum based on success rate"""
        if success_rate > 0.8 and self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            self.stage_progress = 0.0
            print(f"Advancing to curriculum stage {self.current_stage + 1}")
        elif success_rate < 0.5 and self.current_stage > 0:
            self.current_stage -= 1
            print(f"Regressing to curriculum stage {self.current_stage + 1}")

    def get_current_randomization_level(self):
        """Get current level of randomization based on curriculum stage"""
        return self.current_stage / (self.num_stages - 1)  # Normalized 0-1
```

## 9.3 System Identification and Model Adaptation

### 9.3.1 System Identification for Humanoid Robots

```python
import numpy as np
from scipy.optimize import minimize
from scipy import signal

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.parameters = {}
        self.identification_data = []

    def collect_identification_data(self, joint_positions, joint_velocities,
                                  joint_torques, external_forces=None):
        """Collect data for system identification"""
        data_point = {
            'q': joint_positions.copy(),
            'dq': joint_velocities.copy(),
            'ddq': np.zeros_like(joint_positions),  # Will be computed
            'tau': joint_torques.copy(),
            'f_ext': external_forces.copy() if external_forces is not None else np.zeros_like(joint_positions)
        }

        # Estimate accelerations using finite differences if not provided
        if len(self.identification_data) >= 2:
            dt = 0.001  # Assuming 1kHz control rate
            prev_dq = self.identification_data[-1]['dq']
            data_point['ddq'] = (joint_velocities - prev_dq) / dt

        self.identification_data.append(data_point)

    def dynamics_regression_matrix(self, q, dq, ddq):
        """Construct regression matrix for rigid body dynamics"""
        n = len(q)  # Number of joints
        Y = np.zeros((n, 10 * n))  # Regression matrix

        # For each joint, compute the contribution to the regression matrix
        for i in range(n):
            # This is a simplified example - full implementation would be more complex
            # and involve computing the actual dynamics terms
            Y[i, i*10:(i+1)*10] = self.compute_joint_regression_terms(
                q, dq, ddq, i
            )

        return Y

    def compute_joint_regression_terms(self, q, dq, ddq, joint_idx):
        """Compute regression terms for a specific joint"""
        # Simplified regression vector - in practice this would be much more complex
        # involving Coriolis, centrifugal, gravitational, and inertial terms
        phi = np.zeros(10)

        # Example terms (simplified):
        phi[0] = ddq[joint_idx]  # Inertial term
        phi[1] = dq[joint_idx]**2  # Centrifugal term
        phi[2] = dq[joint_idx] * dq[(joint_idx + 1) % len(q)]  # Coriolis term
        phi[3] = np.sin(q[joint_idx])  # Gravity-related term
        phi[4] = np.cos(q[joint_idx])  # Gravity-related term
        # ... more terms would be added for a complete model

        return phi

    def identify_parameters(self):
        """Identify robot dynamics parameters using collected data"""
        if len(self.identification_data) < 100:  # Need sufficient data
            raise ValueError("Insufficient data for system identification")

        # Construct the full regression problem
        Y_full = []
        tau_full = []

        for data_point in self.identification_data:
            Y = self.dynamics_regression_matrix(
                data_point['q'],
                data_point['dq'],
                data_point['ddq']
            )
            Y_full.append(Y)
            tau_full.append(data_point['tau'])

        Y_matrix = np.vstack(Y_full)
        tau_vector = np.hstack(tau_full)

        # Solve the least squares problem: Y * theta = tau
        # where theta are the parameters to identify
        try:
            parameters, residuals, rank, s = np.linalg.lstsq(
                Y_matrix, tau_vector, rcond=None
            )

            # Store identified parameters
            self.parameters = {
                'mass_matrix_coeffs': parameters[:len(parameters)//3],
                'coriolis_coeffs': parameters[len(parameters)//3:2*len(parameters)//3],
                'gravity_coeffs': parameters[2*len(parameters)//3:]
            }

            print(f"Parameter identification completed. Residuals: {residuals}")
            return self.parameters

        except np.linalg.LinAlgError:
            print("Parameter identification failed - system may be unobservable")
            return None

class ModelAdaptor:
    def __init__(self, simulation_model, real_robot_interface):
        self.sim_model = simulation_model
        self.real_robot = real_robot_interface
        self.parameter_offset = {}
        self.compensation_controller = None

    def adapt_model_parameters(self, identified_params):
        """Adapt simulation model based on identified real robot parameters"""
        # Calculate parameter offsets
        for param_name, real_value in identified_params.items():
            sim_value = self.sim_model.get_parameter(param_name)
            self.parameter_offset[param_name] = real_value - sim_value

        # Update simulation model with adapted parameters
        for param_name, offset in self.parameter_offset.items():
            new_value = self.sim_model.get_parameter(param_name) + offset
            self.sim_model.set_parameter(param_name, new_value)

    def learn_compensation_policy(self):
        """Learn a compensation policy to handle model errors"""
        # Collect data comparing sim and real responses
        sim_responses = []
        real_responses = []

        for _ in range(1000):  # Collect 1000 data points
            # Apply same input to both sim and real
            test_input = np.random.uniform(-1, 1, size=self.sim_model.num_joints)

            sim_response = self.sim_model.forward_dynamics(test_input)
            real_response = self.real_robot.measure_response(test_input)

            sim_responses.append(sim_response)
            real_responses.append(real_response)

        # Train a neural network to predict the compensation needed
        self.compensation_controller = self.train_compensation_network(
            sim_responses, real_responses
        )

    def train_compensation_network(self, sim_data, real_data):
        """Train a neural network to predict compensation"""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class CompensationNet(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size * 2, 256),  # Sim + Real inputs
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size)
                )

            def forward(self, sim_state, real_state):
                combined_input = torch.cat([sim_state, real_state], dim=-1)
                return self.network(combined_input)

        # Convert to tensors
        sim_tensor = torch.FloatTensor(sim_data)
        real_tensor = torch.FloatTensor(real_data)

        # Simple training loop (simplified)
        model = CompensationNet(self.sim_model.num_joints, self.sim_model.num_joints)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(sim_tensor, real_tensor)
            loss = criterion(pred, real_tensor - sim_tensor)  # Learn the difference
            loss.backward()
            optimizer.step()

        return model
```

## 9.4 Robust Control and Adaptive Control

### 9.4.1 Robust Control for Humanoid Balance

```python
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy import signal

class RobustBalanceController:
    def __init__(self, robot_model, uncertainty_bounds):
        self.robot = robot_model
        self.uncertainty_bounds = uncertainty_bounds
        self.feedback_gain = None
        self.robustness_margin = 0.1

    def design_robust_controller(self, nominal_model, desired_poles=None):
        """Design a robust controller using H-infinity or mu-synthesis"""
        # Linearize the system around the nominal operating point
        A, B, C, D = self.linearize_system(nominal_model)

        # Add uncertainty weighting
        A_uncertain = self.add_uncertainty_weighting(A)

        # Design controller using LQR with robustness considerations
        Q = np.eye(A.shape[0]) * 10  # State weighting (more on position errors)
        R = np.eye(B.shape[1]) * 0.1  # Control weighting (less control effort)

        # Solve Riccati equation for robust LQR
        P = solve_continuous_are(A_uncertain, B, Q, R)

        # Compute feedback gain
        self.feedback_gain = np.linalg.inv(R) @ B.T @ P

        return self.feedback_gain

    def linearize_system(self, model_state):
        """Linearize the humanoid robot dynamics around current state"""
        # This would involve computing the Jacobian of the nonlinear dynamics
        # Simplified example for a 2D inverted pendulum model of humanoid
        # (x, theta, dx, dtheta) where x is position, theta is body angle

        # State: [x, theta, dx, dtheta]
        # Control: [F_x, T] where F_x is horizontal force, T is torque
        A = np.array([
            [0, 0, 1, 0],           # dx/dt = dx
            [0, 0, 0, 1],           # dtheta/dt = dtheta
            [0, -9.81, 0, 0],       # d²x/dt² = f(x, theta, dx, dtheta)
            [0, 0, 0, 0]            # d²theta/dt² = f(x, theta, dx, dtheta)
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [1/mass, 0],
            [0, 1/inertia]
        ])

        C = np.eye(4)  # All states measurable
        D = np.zeros((4, 2))

        return A, B, C, D

    def add_uncertainty_weighting(self, A):
        """Add uncertainty weighting to system matrix"""
        # Add uncertainty based on identified bounds
        uncertainty_matrix = np.random.uniform(
            -self.uncertainty_bounds,
            self.uncertainty_bounds,
            A.shape
        )
        return A + uncertainty_matrix

    def compute_control(self, state_error, measured_state):
        """Compute robust control action"""
        if self.feedback_gain is None:
            raise ValueError("Controller not designed yet")

        # Apply feedback with robustness considerations
        control_action = -self.feedback_gain @ state_error

        # Add integral action for disturbance rejection
        control_action += self.integral_action(state_error)

        # Apply control saturation based on actuator limits
        control_action = np.clip(
            control_action,
            -self.robot.max_torque,
            self.robot.max_torque
        )

        return control_action

    def integral_action(self, error, dt=0.001):
        """Integral action for disturbance rejection"""
        if not hasattr(self, 'integral_error'):
            self.integral_error = np.zeros_like(error)

        self.integral_error += error * dt
        integral_gain = 0.1  # Tuned for humanoid balance
        return integral_gain * self.integral_error

class AdaptiveController:
    def __init__(self, initial_params=None):
        self.params = initial_params or self.initialize_parameters()
        self.param_history = []
        self.adaptation_rate = 0.01

    def initialize_parameters(self):
        """Initialize controller parameters"""
        return {
            'Kp': np.eye(6) * 100,  # Proportional gain for 6 DOF
            'Kd': np.eye(6) * 10,   # Derivative gain
            'Ki': np.eye(6) * 1,    # Integral gain
            'model_params': np.ones(10)  # Dynamics model parameters
        }

    def update_parameters(self, tracking_error, state_error, dt=0.001):
        """Update controller parameters based on tracking performance"""
        # Parameter adaptation law (simplified)
        # This would typically use Lyapunov-based adaptation
        param_adjustment = self.adaptation_rate * np.outer(
            tracking_error,
            state_error
        )

        # Update each parameter set
        for key in self.params:
            if isinstance(self.params[key], np.ndarray):
                self.params[key] += param_adjustment

        # Keep track of parameter history for analysis
        self.param_history.append(self.params.copy())

    def compute_adaptive_control(self, desired_state, current_state, dt=0.001):
        """Compute control using adaptive parameters"""
        # State error
        state_error = desired_state - current_state

        # PID control with adaptive gains
        proportional = self.params['Kp'] @ state_error[:6]  # Position errors
        derivative = self.params['Kd'] @ state_error[6:]    # Velocity errors (if available)

        # Integral action
        if not hasattr(self, 'integrated_error'):
            self.integrated_error = np.zeros(6)
        self.integrated_error += state_error[:6] * dt
        integral = self.params['Ki'] @ self.integrated_error

        control_output = proportional + derivative + integral

        return control_output
```

## 9.5 Reality Gap Bridging Techniques

### 9.5.1 Simulated Annealing for Policy Adaptation

```python
import numpy as np
import random
from typing import Callable, Any

class PolicyAdaptor:
    def __init__(self, simulation_policy, real_robot_interface):
        self.sim_policy = simulation_policy
        self.real_robot = real_robot_interface
        self.adaptation_params = {}

    def adapt_policy_temperature(self, initial_temp=100.0, cooling_rate=0.95):
        """Use simulated annealing to adapt policy for real robot"""
        current_params = self.sim_policy.get_parameters().copy()
        current_performance = self.evaluate_on_real_robot(current_params)

        temperature = initial_temp
        best_params = current_params.copy()
        best_performance = current_performance

        iteration = 0
        while temperature > 1.0:
            # Generate neighbor solution by perturbing parameters
            neighbor_params = self.perturb_parameters(current_params, temperature)

            # Evaluate on real robot
            neighbor_performance = self.evaluate_on_real_robot(neighbor_params)

            # Accept or reject the neighbor
            if (neighbor_performance > current_performance or
                random.random() < np.exp((neighbor_performance - current_performance) / temperature)):
                current_params = neighbor_params.copy()
                current_performance = neighbor_performance

                if neighbor_performance > best_performance:
                    best_params = neighbor_params.copy()
                    best_performance = neighbor_performance

            # Cool down
            temperature *= cooling_rate
            iteration += 1

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Temp: {temperature:.2f}, "
                      f"Best Performance: {best_performance:.4f}")

        # Update policy with best parameters found
        self.sim_policy.set_parameters(best_params)
        return best_params, best_performance

    def perturb_parameters(self, params, temperature):
        """Perturb policy parameters based on current temperature"""
        perturbed_params = params.copy()

        # Add noise proportional to temperature
        noise_scale = temperature * 0.01  # Scale factor
        for key in perturbed_params:
            if isinstance(perturbed_params[key], np.ndarray):
                noise = np.random.normal(0, noise_scale, perturbed_params[key].shape)
                perturbed_params[key] += noise

        return perturbed_params

    def evaluate_on_real_robot(self, params):
        """Evaluate policy parameters on real robot"""
        # Temporarily set parameters
        original_params = self.sim_policy.get_parameters()
        self.sim_policy.set_parameters(params)

        try:
            # Execute policy on real robot and measure performance
            performance = self.real_robot.execute_policy(
                self.sim_policy,
                num_episodes=5
            )
        except Exception as e:
            print(f"Error evaluating on real robot: {e}")
            performance = -1.0  # Penalty for failed execution

        # Restore original parameters
        self.sim_policy.set_parameters(original_params)

        return performance

class TransferLearningAdaptor:
    def __init__(self, source_policy, target_robot):
        self.source_policy = source_policy
        self.target_robot = target_robot
        self.feature_extractor = None
        self.task_mapping = {}

    def learn_feature_mapping(self, source_data, target_data):
        """Learn mapping between source and target robot features"""
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor

        # Align data dimensions
        min_samples = min(len(source_data), len(target_data))
        X = np.array(source_data[:min_samples])
        y = np.array(target_data[:min_samples])

        # Train mapping function
        self.feature_mapping_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42
        )
        self.feature_mapping_model.fit(X, y)

        return self.feature_mapping_model

    def adapt_policy_for_target(self):
        """Adapt source policy for target robot using learned mapping"""
        # Get source policy features
        source_features = self.source_policy.extract_features()

        # Map to target robot feature space
        target_features = self.feature_mapping_model.predict([source_features])[0]

        # Create adapted policy for target robot
        adapted_policy = self.target_robot.create_policy_from_features(target_features)

        return adapted_policy
```

### 9.5.2 Domain Adaptation Networks

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=256):
        super().__init__()

        # Feature extractor shared between domains
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )

        # Task-specific classifier for source domain
        self.source_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # For control output
        )

        # Task-specific classifier for target domain
        self.target_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # For control output
        )

        # Domain classifier to distinguish source vs target
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0 for source, 1 for target
        )

    def forward(self, x, domain_label=None):
        features = self.feature_extractor(x)

        if domain_label == 'source':
            task_output = self.source_classifier(features)
        elif domain_label == 'target':
            task_output = self.target_classifier(features)
        else:
            # If no domain specified, return both
            source_out = self.source_classifier(features)
            target_out = self.target_classifier(features)
            domain_out = self.domain_classifier(features)
            return source_out, target_out, domain_out

        return task_output

class DomainAdaptationTrainer:
    def __init__(self, model):
        self.model = model
        self.task_criterion = nn.MSELoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train(self, source_loader, target_loader, num_epochs=100):
        """Train with domain adaptation"""
        for epoch in range(num_epochs):
            total_loss = 0
            total_task_loss = 0
            total_domain_loss = 0

            for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
                self.optimizer.zero_grad()

                # Source domain: task loss + domain loss
                source_features = self.model.feature_extractor(source_data)
                source_task_pred = self.model.source_classifier(source_features)
                source_task_loss = self.task_criterion(source_task_pred, source_labels)

                # Domain classification loss for source
                source_domain_pred = self.model.domain_classifier(source_features)
                source_domain_labels = torch.zeros(source_data.size(0), dtype=torch.long)
                source_domain_loss = self.domain_criterion(source_domain_pred, source_domain_labels)

                # Target domain: only domain loss (no labels available)
                target_features = self.model.feature_extractor(target_data)
                target_domain_pred = self.model.domain_classifier(target_features)
                target_domain_labels = torch.ones(target_data.size(0), dtype=torch.long)
                target_domain_loss = self.domain_criterion(target_domain_pred, target_domain_labels)

                # Total domain loss
                domain_loss = source_domain_loss + target_domain_loss

                # Total loss
                total_batch_loss = source_task_loss + domain_loss

                total_batch_loss.backward()
                self.optimizer.step()

                total_loss += total_batch_loss.item()
                total_task_loss += source_task_loss.item()
                total_domain_loss += domain_loss.item()

            print(f"Epoch {epoch}: Task Loss: {total_task_loss:.4f}, "
                  f"Domain Loss: {total_domain_loss:.4f}")
```

## 9.6 Validation and Testing Strategies

### 9.6.1 Progressive Validation Framework

```python
class ProgressiveValidationFramework:
    def __init__(self, robot_model, real_robot_interface):
        self.robot_model = robot_model
        self.real_robot = real_robot_interface
        self.validation_levels = [
            'component_level',
            'subsystem_level',
            'full_system_level',
            'real_world_level'
        ]
        self.current_level = 0
        self.pass_thresholds = [0.9, 0.85, 0.8, 0.75]  # Success rates

    def validate_progressively(self):
        """Validate at each level before proceeding to the next"""
        results = {}

        for i, level in enumerate(self.validation_levels):
            print(f"Validating at {level}...")

            success_rate = self.run_validation_level(level)
            results[level] = success_rate

            print(f"{level} validation success rate: {success_rate:.3f}")

            if success_rate < self.pass_thresholds[i]:
                print(f"Validation failed at {level}. Success rate {success_rate:.3f} "
                      f"below threshold {self.pass_thresholds[i]:.3f}")
                return results, False  # Validation failed

            print(f"Passed {level} validation. Proceeding to next level...")

        return results, True  # All levels passed

    def run_validation_level(self, level):
        """Run validation at specified level"""
        if level == 'component_level':
            return self.validate_components()
        elif level == 'subsystem_level':
            return self.validate_subsystems()
        elif level == 'full_system_level':
            return self.validate_full_system()
        elif level == 'real_world_level':
            return self.validate_real_world()

    def validate_components(self):
        """Validate individual components (joints, sensors, etc.)"""
        success_count = 0
        total_tests = 100

        for i in range(total_tests):
            # Test individual joint control
            joint_idx = random.randint(0, self.robot_model.num_joints - 1)
            target_position = random.uniform(-1.5, 1.5)

            success = self.test_joint_control(joint_idx, target_position)
            if success:
                success_count += 1

        return success_count / total_tests

    def validate_subsystems(self):
        """Validate subsystems (balance, navigation, manipulation)"""
        success_count = 0
        total_tests = 50

        for i in range(total_tests):
            test_type = random.choice(['balance', 'navigation', 'manipulation'])

            if test_type == 'balance':
                success = self.test_balance_stability()
            elif test_type == 'navigation':
                success = self.test_navigation_accuracy()
            elif test_type == 'manipulation':
                success = self.test_manipulation_precision()

            if success:
                success_count += 1

        return success_count / total_tests

    def validate_full_system(self):
        """Validate integrated system performance"""
        # Run complex multi-step tasks
        success_count = 0
        total_tasks = 20

        for i in range(total_tasks):
            task_success = self.execute_complex_task()
            if task_success:
                success_count += 1

        return success_count / total_tasks

    def validate_real_world(self):
        """Validate in actual real-world conditions"""
        # This would involve testing in the actual deployment environment
        success_count = 0
        total_tests = 10

        for i in range(total_tests):
            real_world_success = self.execute_real_world_task()
            if real_world_success:
                success_count += 1

        return success_count / total_tests

    def test_joint_control(self, joint_idx, target_position):
        """Test individual joint control accuracy"""
        # Command joint to target position
        self.robot_model.set_joint_target(joint_idx, target_position)

        # Simulate or execute on real robot
        current_position = self.robot_model.get_joint_position(joint_idx)

        # Check if within tolerance
        tolerance = 0.1  # radians
        return abs(current_position - target_position) < tolerance

    def test_balance_stability(self):
        """Test balance control stability"""
        # Implement balance stability test
        # This would involve perturbing the robot and checking recovery
        return True  # Placeholder

    def execute_complex_task(self):
        """Execute a complex multi-step task"""
        # Implement complex task execution
        return True  # Placeholder

    def execute_real_world_task(self):
        """Execute task in real world environment"""
        # This would be the actual test on the physical robot
        return True  # Placeholder

class UncertaintyQuantification:
    def __init__(self, model_ensemble_size=5):
        self.ensemble_size = model_ensemble_size
        self.models = []

    def train_uncertainty_ensemble(self, training_data):
        """Train ensemble of models to quantify uncertainty"""
        for i in range(self.ensemble_size):
            # Train individual model with different initialization/random seed
            model = self.create_model()
            model.train(training_data, seed=i)
            self.models.append(model)

    def predict_with_uncertainty(self, input_data):
        """Get prediction with uncertainty estimate"""
        predictions = []

        for model in self.models:
            pred = model.predict(input_data)
            predictions.append(pred)

        # Calculate mean and uncertainty (variance)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.var(predictions, axis=0)

        return mean_pred, uncertainty

    def adapt_behavior_based_on_uncertainty(self, state, uncertainty):
        """Adapt robot behavior based on uncertainty level"""
        high_uncertainty_threshold = 0.5

        if np.max(uncertainty) > high_uncertainty_threshold:
            # Be more conservative when uncertain
            return self.conservative_behavior(state)
        else:
            # Use normal behavior when confident
            return self.normal_behavior(state)

    def conservative_behavior(self, state):
        """Conservative behavior for high uncertainty"""
        # Reduce speed, increase safety margins, etc.
        return "conservative_action"

    def normal_behavior(self, state):
        """Normal behavior for low uncertainty"""
        return "normal_action"
```

## 9.7 Practical Implementation Guidelines

### 9.7.1 Transfer Pipeline

```python
class SimToRealTransferPipeline:
    def __init__(self, simulation_env, real_robot, policy):
        self.sim_env = simulation_env
        self.real_robot = real_robot
        self.policy = policy
        self.transfer_steps = [
            'model_verification',
            'parameter_identification',
            'domain_randomization',
            'policy_adaptation',
            'validation',
            'deployment'
        ]

    def execute_transfer_pipeline(self):
        """Execute the complete sim-to-real transfer pipeline"""
        results = {}

        for step in self.transfer_steps:
            print(f"Executing transfer step: {step}")

            if step == 'model_verification':
                success = self.verify_simulation_model()
            elif step == 'parameter_identification':
                success = self.identify_real_parameters()
            elif step == 'domain_randomization':
                success = self.apply_domain_randomization()
            elif step == 'policy_adaptation':
                success = self.adapt_policy()
            elif step == 'validation':
                success = self.validate_transfer()
            elif step == 'deployment':
                success = self.deploy_to_real_robot()

            results[step] = success
            print(f"Step {step}: {'SUCCESS' if success else 'FAILED'}")

            if not success:
                print(f"Transfer pipeline failed at step: {step}")
                return results, False

        print("Sim-to-real transfer pipeline completed successfully!")
        return results, True

    def verify_simulation_model(self):
        """Verify that simulation model matches real robot"""
        # Compare kinematic and dynamic properties
        sim_properties = self.sim_env.get_robot_properties()
        real_properties = self.real_robot.get_properties()

        # Check mass, inertia, joint limits, etc.
        mass_error = abs(sim_properties.mass - real_properties.mass) / real_properties.mass
        if mass_error > 0.1:  # 10% tolerance
            print(f"Mass error too high: {mass_error:.3f}")
            return False

        return True

    def identify_real_parameters(self):
        """Identify real robot parameters"""
        identifier = SystemIdentifier(self.real_robot)

        # Collect identification data
        for _ in range(1000):  # Collect 1000 data points
            joint_pos = np.random.uniform(-1, 1, size=self.real_robot.num_joints)
            joint_vel = np.random.uniform(-2, 2, size=self.real_robot.num_joints)
            torques = self.real_robot.measure_torques(joint_pos, joint_vel)

            identifier.collect_identification_data(joint_pos, joint_vel, torques)

        # Identify parameters
        params = identifier.identify_parameters()
        if params is None:
            return False

        # Update simulation model
        self.sim_env.update_robot_parameters(params)
        return True

    def apply_domain_randomization(self):
        """Apply domain randomization to policy"""
        randomizer = VisualDomainRandomizer()
        physics_randomizer = PhysicsDomainRandomizer()

        # Train policy with randomization
        for episode in range(10000):
            # Randomize environment
            randomized_scene = randomizer.randomize_lighting(self.sim_env.scene)
            randomized_robot = physics_randomizer.randomize_robot_dynamics(self.sim_env.robot)

            # Train policy in randomized environment
            self.policy.train_step(randomized_scene, randomized_robot)

        return True

    def adapt_policy(self):
        """Adapt policy for real robot"""
        adaptor = PolicyAdaptor(self.policy, self.real_robot)
        adapted_params, performance = adaptor.adapt_policy_temperature()

        # Accept adaptation if performance is reasonable
        return performance > 0.5

    def validate_transfer(self):
        """Validate transfer on real robot"""
        validator = ProgressiveValidationFramework(
            self.sim_env.robot_model,
            self.real_robot
        )
        results, success = validator.validate_progressively()
        return success

    def deploy_to_real_robot(self):
        """Deploy policy to real robot"""
        try:
            self.real_robot.load_policy(self.policy)
            self.real_robot.calibrate_sensors()
            self.real_robot.initialize_control_system()
            return True
        except Exception as e:
            print(f"Deployment failed: {e}")
            return False
```

## 9.8 Best Practices and Lessons Learned

### 9.8.1 Key Success Factors

1. **Start Simple**: Begin with basic behaviors and gradually increase complexity
2. **Model Validation**: Verify simulation models match real robot characteristics
3. **Sufficient Randomization**: Use extensive domain randomization during training
4. **Iterative Refinement**: Continuously adapt and refine based on real-world performance
5. **Safety First**: Implement robust safety mechanisms to prevent damage
6. **Data Collection**: Collect extensive data from both simulation and reality
7. **Modular Design**: Keep components modular for easier adaptation and debugging

### 9.8.2 Common Pitfalls to Avoid

1. **Overfitting to Simulation**: Ensure policies generalize beyond training conditions
2. **Ignoring Sensor Noise**: Account for real sensor characteristics in training
3. **Insufficient Modeling**: Don't ignore important physical effects
4. **Rigid Control**: Use adaptive control strategies for changing conditions
5. **Inadequate Validation**: Test thoroughly across multiple scenarios
6. **Poor Error Handling**: Implement robust error recovery mechanisms

## Summary

Sim-to-real transfer remains one of the most challenging aspects of humanoid robotics, requiring careful consideration of modeling, control, and adaptation strategies. Success depends on a combination of accurate simulation, robust control design, appropriate domain randomization, and systematic validation approaches. The techniques covered in this chapter, from domain randomization to system identification and adaptive control, provide a comprehensive toolkit for bridging the reality gap. The key to successful transfer lies in understanding and addressing the specific challenges of humanoid robot dynamics, balance control, and multi-modal sensing. In the next chapter, we will explore the integration of all these components into a complete humanoid robot system and provide an assessment of the AI-Robot Brain module.