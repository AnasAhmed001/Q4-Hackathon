---
title: Module 2 Assessment - Digital Twin Fundamentals
description: Assessment for Module 2 covering Gazebo physics, SDF, Unity ML-Agents, rendering pipelines, and ROS-Unity bridge.
sidebar_position: 21
---

# Module 2 Assessment - Digital Twin Fundamentals

This assessment evaluates your understanding of the core concepts covered in Module 2: The Digital Twin. You will be required to demonstrate practical skills in setting up Gazebo simulations, creating Unity environments, integrating sensors, and establishing communication between ROS 2 and Unity.

## Assessment Structure

The Module 2 assessment consists of both practical exercises and theoretical questions to be completed within a **2-hour time frame**. You will need access to a system with Gazebo, Unity, and ROS 2 installed.

## Learning Objectives Covered

By completing this assessment, you should demonstrate the ability to:
1.  Configure Gazebo physics properties for realistic humanoid simulation
2.  Create and modify SDF files for complex scenes
3.  Implement and test physics properties like friction and inertia
4.  Integrate and visualize data from multiple sensor plugins (ray, camera, IMU)
5.  Apply sensor noise models for realistic simulation
6.  Create high-fidelity Unity scenes for humanoid robots
7.  Implement ML-Agents for basic AI training in Unity
8.  Configure advanced rendering pipelines (PBR, HDR)
9.  Establish bidirectional communication between ROS 2 and Unity
10. Validate simulation results against expected behavior
11. Implement collision detection and contact dynamics

## Practical Exercises (70 points)

### Exercise 1: Gazebo World Creation (20 points)

**Objective**: Create a Gazebo world with a humanoid robot model and at least two sensor plugins.

1.  Create a new SDF world file `training_world.sdf` that includes:
    *   A humanoid robot model (you can use a simplified version or the one from Module 1)
    *   A ground plane with realistic friction properties (mu=0.8)
    *   At least two obstacles (box and cylinder)
    *   A ramp or inclined surface for testing
2.  Add the following sensor plugins to the robot:
    *   A camera sensor with 640x480 resolution and 60-degree FOV
    *   An IMU sensor with realistic noise parameters
3.  Configure physics parameters for realistic humanoid simulation:
    *   Set appropriate mass and inertial properties for each link
    *   Configure friction coefficients for feet (mu=0.9)
    *   Set simulation time step to 0.001s
4.  Launch the world in Gazebo and verify that the robot remains stable on the ground plane.
5.  Demonstrate that sensor data is being published by checking topics with `ros2 topic list` and `ros2 topic echo`.

### Exercise 2: Unity ML-Agents Training Environment (20 points)

**Objective**: Create a Unity scene with ML-Agents for training a simple humanoid behavior.

1.  Create a Unity scene with:
    *   A simple humanoid model (articulated with joints)
    *   A target object that moves to random positions
    *   A ground plane
2.  Implement an ML-Agent that controls the humanoid to reach the target:
    *   Define appropriate observations (robot pose, target position, joint angles)
    *   Define actions for controlling joints or movement
    *   Implement a reward function that encourages reaching the target while maintaining balance
3.  Configure the ML-Agent with:
    *   Continuous action space with appropriate bounds
    *   Proper observation normalization
    *   Appropriate memory size if needed for temporal decisions
4.  Set up training parameters in a configuration file for PPO training
5.  Run a brief training session (at least 10,000 steps) and observe the learning progress.

### Exercise 3: High-Fidelity Rendering (15 points)

**Objective**: Configure advanced rendering in Unity for realistic humanoid simulation.

1.  Set up an HDRP or URP project with:
    *   Physically Based Rendering materials for the humanoid robot
    *   High Dynamic Range lighting with realistic sun intensity (100,000 lux)
    *   Image-based lighting using a high-quality environment map
2.  Implement post-processing effects:
    *   Bloom for realistic light scattering
    *   Color grading for realistic appearance
    *   Depth of field (optional but recommended)
3.  Create a rendering pipeline that outputs both RGB and depth images suitable for computer vision applications
4.  Add realistic sensor noise simulation to the rendered images
5.  Capture sample images demonstrating the high-fidelity rendering.

### Exercise 4: ROS-Unity Bridge Communication (15 points)

**Objective**: Establish bidirectional communication between ROS 2 and Unity.

1.  Set up the ROS-TCP-Connector in Unity:
    *   Configure Unity to connect to ROS 2 running on localhost:10000
    *   Create a publisher that sends joint states from Unity to ROS 2
    *   Create a subscriber that receives joint commands from ROS 2
2.  Implement communication for:
    *   Publishing camera images from Unity to a ROS 2 topic
    *   Subscribing to a ROS 2 topic to control the humanoid's base movement
    *   Broadcasting TF transforms from Unity
3.  Create a simple ROS 2 node that:
    *   Subscribes to the joint states published from Unity
    *   Publishes joint commands to move the Unity humanoid
    *   Subscribes to camera images and displays them
4.  Demonstrate bidirectional communication by controlling the Unity humanoid from ROS 2 and observing the state feedback.

## Theoretical Questions (30 points)

### Question 1 (10 points)

Explain the differences between Physically Based Rendering (PBR) and traditional rendering approaches. Describe how PBR properties (metallic, roughness, normal maps) affect the visual appearance of a humanoid robot model in Unity. Provide specific examples of how these properties would differ for metallic parts versus fabric clothing.

### Question 2 (10 points)

Compare and contrast collision detection approaches in Gazebo versus Unity. Discuss the importance of contact dynamics for humanoid robot simulation, and explain how parameters like ERP (Error Reduction Parameter) and CFM (Constraint Force Mixing) affect the stability and realism of collisions in Gazebo.

### Question 3 (10 points)

Describe the key components and workflow of the Unity ML-Agents toolkit. Explain how to design an effective reward function for training a humanoid robot to walk, considering factors like forward progress, balance, energy efficiency, and obstacle avoidance. Discuss potential challenges in reward engineering and how to address them.

## Grading Criteria

*   **Functionality (40 points)**: All practical exercises must run without errors and produce the expected output.
*   **Code Quality (10 points)**: Unity scripts and ROS nodes should be well-structured, readable, and follow best practices.
*   **SDF Correctness (5 points)**: The SDF world file must be syntactically correct and represent the described environment.
*   **Physics Realism (5 points)**: Physics properties and parameters should be appropriate for humanoid robot simulation.
*   **Rendering Quality (5 points)**: Unity rendering should demonstrate proper use of PBR, HDR, and post-processing effects.
*   **Communication Implementation (5 points)**: ROS-Unity bridge should function correctly with bidirectional data flow.
*   **Theoretical Understanding (30 points)**: Responses to theoretical questions must be accurate, clear, and demonstrate a solid understanding of the concepts.

## Submission Requirements

1.  Create a package containing:
    *   The complete Unity project folder (or a link to a shared project)
    *   All SDF files created for Gazebo
    *   ROS 2 nodes and launch files
    *   ML-Agents training configuration files
    *   A `README.md` file with:
        *   Instructions on how to run each exercise
        *   Screenshots of key results (Unity scene, Gazebo simulation, etc.)
        *   Answers to the theoretical questions
2.  Archive the entire package as a ZIP file named `module2_assessment_yourname.zip`.
3.  Submit the ZIP file via the designated platform.

## Evaluation Process

1.  The instructor will extract your ZIP file and review the Unity project.
2.  They will launch your Gazebo world and verify sensor integration.
3.  They will run your ROS-Unity bridge and test bidirectional communication.
4.  They will evaluate your ML-Agents implementation and training results.
5.  They will review your code and configurations for quality and correctness.
6.  They will assess your answers to the theoretical questions.
7.  Your grade will be calculated based on the grading criteria above.

## Resources Allowed

*   Official Gazebo documentation
*   Unity ML-Agents documentation
*   ROS 2 Humble Hawksbill documentation
*   Your personal notes and textbook chapters
*   Online search engines for syntax and API reference

## Resources Not Allowed

*   Direct collaboration with other students during the assessment period
*   Sharing code directly with other students
*   Pre-written solutions from online repositories (unless explicitly allowed by the instructor)