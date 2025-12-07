---
title: Chapter 9 - Launch Files and Parameter Management
description: Learn how to use ROS 2 launch files to manage complex robot setups and configure parameters for your humanoid robot.
sidebar_position: 9
---

# Chapter 9 - Launch Files and Parameter Management

As our ROS 2 projects grow, especially when working with complex robots like humanoids, managing multiple nodes, their configurations, and dependencies becomes increasingly challenging. Launch files provide a powerful way to define and execute complex launch procedures, while parameter management allows us to configure node behavior dynamically without recompiling code.

## 9.1 Introduction to Launch Files

Launch files are Python scripts that define how ROS 2 nodes should be started, configured, and managed. They are typically stored in the `launch/` directory of a ROS package and use the `.py` extension. The `launch` library provides the necessary tools to define these procedures.

### 9.1.1 Basic Launch File Structure

Here's a simple example of a launch file (`robot_bringup.launch.py`) that starts a robot state publisher and a joint state publisher GUI:

```python
# launch/robot_bringup.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    urdf_path = LaunchConfiguration('urdf_path')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch arguments
    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value='/path/to/robot.urdf',
        description='Path to the URDF file'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Define the robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': open(urdf_path.perform(context=None)).read()},
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Define the joint state publisher GUI node
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription([
        urdf_path_arg,
        use_sim_time_arg,
        robot_state_publisher,
        joint_state_publisher_gui
    ])
```

### 9.1.2 Key Launch Concepts

*   **`LaunchDescription`**: The top-level container that holds all the actions to be executed when the launch file is run.
*   **`Node`**: Represents a ROS 2 node to be launched. You specify the package name, executable name, and other properties.
*   **`DeclareLaunchArgument`**: Allows you to define arguments that can be passed to the launch file from the command line, making your launch files flexible.
*   **`LaunchConfiguration`**: Used to reference the values of launch arguments within the launch file.
*   **Actions**: There are various actions available, such as `LogInfo`, `ExecuteProcess`, `TimerAction`, etc., for more complex launch procedures.

## 9.2 Launching with Launch Files

To launch the above file, you would typically use the `ros2 launch` command:

```bash
# Launch with default arguments
ros2 launch my_robot_bringup robot_bringup.launch.py

# Launch with custom arguments
ros2 launch my_robot_bringup robot_bringup.launch.py urdf_path:=/custom/path/to/robot.urdf use_sim_time:=true
```

## 9.3 Parameter Management

Parameters in ROS 2 allow nodes to be configured dynamically. They can be set in several ways:

### 9.3.1 Setting Parameters in Launch Files

As shown in the example above, you can pass parameters directly to a node within the launch file:

```python
robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    name='robot_state_publisher',
    parameters=[
        {'robot_description': open(urdf_path.perform(context=None)).read()},
        {'use_sim_time': use_sim_time},
        {'publish_frequency': 50.0}  # Set publish frequency to 50 Hz
    ],
    output='screen'
)
```

### 9.3.2 Parameter Files (YAML)

You can also store parameters in YAML files, which is often more convenient for large configurations. Create a file like `config/robot_params.yaml`:

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    use_sim_time: false
    log_level: "info"

robot_state_publisher:
  ros__parameters:
    publish_frequency: 50.0
    use_tf_static: true
    ignore_timestamp: false

my_controller_node:
  ros__parameters:
    kp: 1.0
    ki: 0.1
    kd: 0.05
    max_velocity: 1.0
    control_frequency: 100.0
```

Then, load this file in your launch file:

```python
from ament_index_python.packages import get_package_share_directory
import os

# Inside generate_launch_description()
params_file = os.path.join(
    get_package_share_directory('my_robot_bringup'),
    'config',
    'robot_params.yaml'
)

robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    name='robot_state_publisher',
    parameters=[params_file],
    output='screen'
)
```

### 9.3.3 Command Line Parameters

Parameters can also be set directly from the command line when running a node:

```bash
ros2 run my_package my_node --ros-args --param publish_frequency:=60.0 --param log_level:=debug
```

## 9.4 Practical Example: Launching a Humanoid Robot

For a humanoid robot, a launch file might start multiple nodes simultaneously:

*   `robot_state_publisher`: Publishes the robot's joint states as transforms.
*   `joint_state_publisher_gui`: Provides a GUI to manually set joint states (useful for testing).
*   `controller_manager`: Manages hardware interfaces and controllers.
*   Specific sensor nodes (IMU, cameras, etc.).
*   Custom behavior nodes.

```python
# launch/humanoid_bringup.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get paths
    pkg_share = get_package_share_directory('my_humanoid_description')
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'humanoid.rviz'])

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(PathJoinSubstitution([pkg_share, 'urdf', 'humanoid.urdf']).perform(context=None)).read()
        }],
        output='screen'
    )

    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # RViz2 for visualization
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
```

## Conclusion

Launch files and parameter management are essential tools for organizing and configuring complex ROS 2 systems, particularly for humanoid robots with many components. Launch files allow you to start multiple nodes with specific configurations in a single command, while parameter files provide a flexible way to adjust node behavior without recompiling. In the next chapter, we will create a comprehensive tutorial on setting up a ROS 2 workspace and building your first humanoid robot package.

---
