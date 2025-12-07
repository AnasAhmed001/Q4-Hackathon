---
title: Chapter 5 - Unity ML-Agents Integration
description: Learn how to integrate Unity ML-Agents for AI training in digital twin environments for humanoid robots.
sidebar_position: 17
---

# Chapter 5 - Unity ML-Agents Integration

Unity ML-Agents (Machine Learning Agents) is a powerful toolkit that enables the use of deep reinforcement learning and other machine learning techniques to train intelligent agents within Unity environments. For humanoid robotics, ML-Agents provides an excellent platform for training complex behaviors such as walking, balance, manipulation, and navigation in a high-fidelity digital twin environment.

## 5.1 Introduction to Unity ML-Agents

ML-Agents allows you to train agents using reinforcement learning, imitation learning, or other optimization techniques directly within Unity. This is particularly valuable for humanoid robots, where training in simulation can be safer, faster, and more cost-effective than training on physical hardware.

### 5.1.1 Key Concepts

- **Agent**: The entity that learns to take actions in the environment
- **Environment**: The Unity scene where the agent learns
- **Academy**: The manager that controls the simulation and training process
- **Brain**: The decision-making component (though this concept is evolving in newer versions)
- **Observations**: Information the agent receives from the environment
- **Actions**: Decisions the agent makes to interact with the environment
- **Rewards**: Feedback that guides the agent's learning

## 5.2 Setting up ML-Agents in Unity

### 5.2.1 Installation

First, install the ML-Agents toolkit in your Unity project:

1. Open Unity Hub and create a new 3D project or open an existing one
2. Open the Package Manager (Window > Package Manager)
3. Click the + button and select "Add package from git URL..."
4. Enter: `com.unity.ml-agents`
5. Also install: `com.unity.ml-agents.extensions` for additional tools

### 5.2.2 Python Dependencies

Install the Python ML-Agents package:

```bash
pip install mlagents
```

## 5.3 Creating a Humanoid Training Environment

Let's create a simple environment for training a humanoid robot to walk:

### 5.3.1 Basic Scene Setup

First, create a simple scene with:
- A ground plane
- A humanoid robot model (or simple articulated body)
- Target objects or goals

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class HumanoidAgent : Agent
{
    [Header("Humanoid Specific")]
    public Transform target;
    public float moveSpeed = 3f;
    public float rotationSpeed = 100f;

    private Rigidbody rb;
    private float previousDistanceToTarget;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent and target positions
        this.transform.position = new Vector3(Random.Range(-5f, 5f), 1f, Random.Range(-5f, 5f));
        target.position = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));

        previousDistanceToTarget = Vector3.Distance(this.transform.position, target.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add observations about the agent's state
        sensor.AddObservation(this.transform.position);
        sensor.AddObservation(target.position);
        sensor.AddObservation(this.transform.rotation.eulerAngles);
        sensor.AddObservation(rb.velocity);
        sensor.AddObservation(rb.angularVelocity);

        // Add joint angles and velocities if available
        // sensor.AddObservation(jointAngles);
        // sensor.AddObservation(jointVelocities);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Process actions received from the brain
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actions.ContinuousActions[0];
        controlSignal.z = actions.ContinuousActions[1];

        // Apply actions to the agent
        rb.AddForce(controlSignal * moveSpeed);

        // Calculate distance to target
        float distanceToTarget = Vector3.Distance(this.transform.position, target.position);

        // Give reward based on progress
        if (distanceToTarget < previousDistanceToTarget)
        {
            SetReward(0.1f); // Positive reward for getting closer
        }
        else
        {
            SetReward(-0.1f); // Negative reward for moving away
        }

        // Bonus reward for reaching target
        if (distanceToTarget < 2f)
        {
            SetReward(1.0f);
            EndEpisode();
        }

        // Penalty for falling
        if (this.transform.position.y < 0.5f)
        {
            SetReward(-1.0f);
            EndEpisode();
        }

        previousDistanceToTarget = distanceToTarget;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing (WASD keys)
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

### 5.3.2 Academy Configuration

Create an Academy component to manage the training environment:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class HumanoidAcademy : Academy
{
    public override void InitializeAcademy()
    {
        // Initialize academy-specific variables
    }

    public override void AcademyStep()
    {
        // Called every step of the simulation
    }

    public override void AcademyReset()
    {
        // Called when the academy resets
    }
}
```

## 5.4 Advanced Humanoid Training Concepts

### 5.4.1 Joint Control for Humanoid Robots

For more complex humanoid robots with multiple joints, you'll need to implement joint control:

```csharp
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AdvancedHumanoidAgent : Agent
{
    [Header("Joint Configuration")]
    public HingeJoint[] joints;
    public float[] jointForces = new float[10]; // Adjust based on number of joints

    [Header("Body Parts")]
    public Transform body;
    public Transform[] limbs;

    private Rigidbody rb;

    public override void CollectObservations(VectorSensor sensor)
    {
        // Body position and rotation
        sensor.AddObservation(body.position);
        sensor.AddObservation(body.rotation);

        // Joint angles and velocities
        foreach (var joint in joints)
        {
            sensor.AddObservation(joint.angle);
            sensor.AddObservation(joint.velocity);
        }

        // Body velocity and angular velocity
        sensor.AddObservation(rb.velocity);
        sensor.AddObservation(rb.angularVelocity);

        // Relative positions of limbs
        foreach (var limb in limbs)
        {
            sensor.AddObservation(limb.position - body.position);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int actionIndex = 0;

        // Apply actions to each joint
        for (int i = 0; i < joints.Length && actionIndex < actions.ContinuousActions.Length; i++)
        {
            float action = Mathf.Clamp(actions.ContinuousActions[actionIndex], -1f, 1f);
            actionIndex++;

            JointMotor motor = joints[i].motor;
            motor.targetVelocity = action * jointForces[i];
            joints[i].motor = motor;
        }

        // Calculate reward based on balance and movement
        CalculateReward();
    }

    private void CalculateReward()
    {
        // Reward for staying upright
        float upReward = Mathf.Clamp01(Vector3.Dot(body.up, Vector3.up));
        SetReward(upReward * 0.1f);

        // Additional rewards based on specific tasks
        // (e.g., moving forward, maintaining balance, etc.)
    }
}
```

### 5.4.2 Curriculum Learning

Implement curriculum learning to gradually increase task difficulty:

```csharp
public class HumanoidCurriculum : MonoBehaviour
{
    public float currentCurriculumLevel = 0f;
    public float curriculumThreshold = 0.8f; // Performance threshold to advance
    private float performanceWindow = 100f; // Number of episodes to average
    private float[] recentScores = new float[100];
    private int scoreIndex = 0;

    public void UpdateCurriculum(float currentScore)
    {
        recentScores[scoreIndex] = currentScore;
        scoreIndex = (scoreIndex + 1) % recentScores.Length;

        // Calculate average performance
        float averageScore = 0f;
        for (int i = 0; i < recentScores.Length; i++)
        {
            averageScore += recentScores[i];
        }
        averageScore /= recentScores.Length;

        // Advance curriculum if performance is good enough
        if (averageScore > curriculumThreshold && currentCurriculumLevel < 10f)
        {
            currentCurriculumLevel++;
            Debug.Log($"Curriculum advanced to level {currentCurriculumLevel}");
            // Adjust environment difficulty based on level
            AdjustEnvironmentDifficulty(currentCurriculumLevel);
        }
    }

    private void AdjustEnvironmentDifficulty(float level)
    {
        // Increase terrain complexity, add obstacles, etc.
        // based on curriculum level
    }
}
```

## 5.5 Training Configuration

Create a training configuration file (`trainer_config.yaml`) to define the training parameters:

```yaml
behaviors:
  HumanoidLearning:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 6144
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 512
      num_layers: 2
      vis_encode_type: simple
      memory_size: 256
    max_steps: 5000000
    time_horizon: 1000
    summary_freq: 10000
```

## 5.6 Running Training

To start training:

```bash
mlagents-learn trainer_config.yaml --run-id=humanoid-walking-v1
```

To run inference with a trained model:

```bash
mlagents-learn trainer_config.yaml --run-id=humanoid-walking-v1 --load
```

## 5.7 Best Practices for Humanoid Training

### 5.7.1 Reward Engineering
- Design reward functions that encourage desired behaviors
- Balance different reward components (balance, movement, efficiency)
- Avoid reward hacking by carefully designing the reward structure

### 5.7.2 Action Space Design
- Use continuous action spaces for smooth control
- Consider the number of joints when designing action dimensions
- Implement action clipping to prevent extreme movements

### 5.7.3 Observation Space Design
- Include relevant state information (positions, velocities, angles)
- Normalize observations for better training stability
- Consider using relative positions instead of absolute ones

### 5.7.4 Simulation-to-Reality Transfer
- Apply domain randomization to improve generalization
- Include sensor noise and actuator delays in simulation
- Test on physical robots once simulation performance is satisfactory

## 5.8 Integration with ROS 2

To integrate Unity ML-Agents with ROS 2, you can use the Unity ROS TCP Connector or ROS# to send sensor data and receive actions:

```csharp
using UnityEngine;
using System.Collections;
using RosSharp;

public class UnityROSBridge : MonoBehaviour
{
    private RosSocket rosSocket;

    void Start()
    {
        rosSocket = new RosSocket("ws://localhost:9090"); // Connect to ROS bridge
    }

    void SendSensorData()
    {
        // Send sensor data from Unity to ROS
        var sensorData = new SensorData();
        sensorData.position = transform.position;
        sensorData.orientation = transform.rotation;

        rosSocket.CallService("/unity_sensor_data", sensorData, (response) => {
            // Handle response
        });
    }

    void ReceiveActions()
    {
        // Subscribe to ROS topics for actions
        rosSocket.Subscribe<RobotAction>("/unity_robot_action", (action) => {
            // Apply action to Unity agent
            ApplyActionToAgent(action);
        });
    }

    void ApplyActionToAgent(RobotAction action)
    {
        // Convert ROS action to Unity control
        // Apply to joints or rigidbodies
    }
}
```

## Summary

Unity ML-Agents provides a powerful platform for training humanoid robots in digital twin environments. By properly designing reward functions, observation spaces, and action spaces, you can train complex behaviors in simulation that can later be transferred to physical robots. In the next chapter, we will explore high-fidelity rendering pipelines in Unity for creating realistic visual environments.