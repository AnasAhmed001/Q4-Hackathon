---
title: Module 1 Assessment - ROS 2 Fundamentals
description: Assessment for Module 1 covering ROS 2 architecture, nodes, topics, services, actions, rclpy, URDF, and launch files.
sidebar_position: 11
---

# Module 1 Assessment - ROS 2 Fundamentals

This assessment evaluates your understanding of the core concepts covered in Module 1: The Robotic Nervous System. You will be required to demonstrate practical skills in setting up a ROS 2 workspace, creating and using nodes, topics, services, actions, and working with URDF models.

## Assessment Structure

The Module 1 assessment consists of both practical exercises and theoretical questions to be completed within a **2-hour time frame**.

## Learning Objectives Covered

By completing this assessment, you should demonstrate the ability to:
1.  Set up and configure a ROS 2 workspace.
2.  Create and run basic ROS 2 nodes using `rclpy`.
3.  Implement publisher-subscriber communication.
4.  Implement service-server and service-client communication.
5.  Implement action-server and action-client communication.
6.  Define and use custom message types.
7.  Define and use custom service types.
8.  Define and use custom action types.
9.  Create and interpret URDF files for robot models.
10. Use launch files to manage complex robot setups.
11. Use parameters to configure node behavior.

## Practical Exercises (70 points)

### Exercise 1: Custom Message and Publisher-Subscriber (20 points)

**Objective**: Create a custom message type and implement a publisher-subscriber pair.

1.  Create a custom message `RobotStatus.msg` in a new package `robot_msgs`:
    ```
    string robot_name
    bool is_operational
    float64 battery_level
    int32 error_code
    ```
2.  Create a publisher node `status_publisher.py` that publishes `RobotStatus` messages every 2 seconds with:
    *   `robot_name`: "TestRobot"
    *   `is_operational`: True
    *   `battery_level`: Random value between 20.0 and 100.0
    *   `error_code`: 0
3.  Create a subscriber node `status_subscriber.py` that subscribes to the topic and prints the received `robot_name` and `battery_level`.
4.  Create a launch file `status_demo.launch.py` to launch both nodes.
5.  Run the launch file and verify the output.

### Exercise 2: Custom Service (20 points)

**Objective**: Create a custom service and implement server-client communication.

1.  Create a custom service `SetRobotMode.srv` in the `robot_msgs` package:
    ```
    # Request
    string mode
    ---
    # Response
    bool success
    string message
    ```
2.  Create a service server `robot_mode_server.py` that accepts "idle", "active", or "maintenance" as valid modes. If the mode is valid, set `success` to `True` and `message` to "Mode set successfully". Otherwise, set `success` to `False` and `message` to "Invalid mode".
3.  Create a service client `robot_mode_client.py` that sends a request with the mode "active" and prints the response.
4.  Create a launch file `service_demo.launch.py` to launch both nodes.
5.  Run the launch file and verify the output.

### Exercise 3: URDF Model (15 points)

**Objective**: Create a URDF model of a simple wheeled robot with a camera.

1.  Create a new package `wheeled_robot_description`.
2.  Create a URDF file `wheeled_robot.urdf` that defines:
    *   A base link (box shape, 0.5x0.3x0.2 m)
    *   Two wheel links (cylinder shape, radius 0.1 m, length 0.05 m)
    *   Two revolute joints connecting the wheels to the base (fixed Z-height, allow rotation around X-axis)
    *   A camera link (box shape, 0.05x0.05x0.05 m) attached to the front of the base
    *   A fixed joint connecting the camera to the base
3.  Create a launch file `view_wheeled_robot.launch.py` that uses `robot_state_publisher` and `rviz2` to visualize the robot.
4.  Launch the file and verify the robot model appears correctly in RViz2.

### Exercise 4: Launch File with Parameters (15 points)

**Objective**: Use launch files and parameter files to configure node behavior.

1.  Create a config directory in your `wheeled_robot_description` package.
2.  Create a parameter file `robot_config.yaml`:
    ```yaml
    /**:
      ros__parameters:
        use_sim_time: false
        log_level: "info"

    robot_state_publisher:
      ros__parameters:
        publish_frequency: 50.0
        use_tf_static: true
    ```
3.  Modify the launch file from Exercise 3 to load this parameter file.
4.  Add a launch argument to the launch file to allow changing the `publish_frequency` parameter from the command line.
5.  Verify that the parameters are loaded correctly by running the launch file.

## Theoretical Questions (30 points)

### Question 1 (10 points)

Explain the difference between a `revolute` joint and a `continuous` joint in URDF. Provide an example of where each type might be used in a humanoid robot.

### Question 2 (10 points)

Describe the role of the `robot_state_publisher` node in ROS 2. How does it use joint state information to publish transforms?

### Question 3 (10 points)

Compare and contrast ROS 2 topics, services, and actions. Provide a specific use case for each communication pattern in the context of controlling a humanoid robot.

## Grading Criteria

*   **Functionality (40 points)**: All practical exercises must run without errors and produce the expected output.
*   **Code Quality (15 points)**: Code should be well-structured, readable, and follow Python best practices (e.g., PEP 8).
*   **URDF Correctness (10 points)**: The URDF model must be syntactically correct and represent the described robot structure.
*   **Launch File Usage (5 points)**: Launch files must correctly start the required nodes and manage parameters.
*   **Theoretical Understanding (30 points)**: Responses to theoretical questions must be accurate, clear, and demonstrate a solid understanding of the concepts.

## Submission Requirements

1.  Create a new ROS 2 workspace named `module1_assessment_ws`.
2.  Place all your custom packages (`robot_msgs`, `wheeled_robot_description`, etc.) inside the `src` directory of this workspace.
3.  Include all source files (`.py`, `.msg`, `.srv`, `.urdf`, `.yaml`, `.launch.py`).
4.  Include a `README.md` file with:
    *   Instructions on how to build and run each exercise.
    *   Answers to the theoretical questions.
5.  Archive the entire workspace directory as a ZIP file named `module1_assessment_yourname.zip`.
6.  Submit the ZIP file via the designated platform.

## Evaluation Process

1.  The instructor will download and extract your ZIP file.
2.  They will build your workspace using `colcon build`.
3.  They will source the workspace and run each launch file to verify the practical exercises.
4.  They will review your code and URDF files for correctness and quality.
5.  They will evaluate your answers to the theoretical questions.
6.  Your grade will be calculated based on the grading criteria above.

## Resources Allowed

*   Official ROS 2 Humble Hawksbill documentation
*   Your personal notes and textbook chapters
*   Online search engines for ROS 2 syntax and API reference

## Resources Not Allowed

*   Direct collaboration with other students
*   Sharing code directly with other students
*   Pre-written solutions from online repositories (unless explicitly allowed by the instructor)