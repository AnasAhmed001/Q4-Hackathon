---
title: Chapter 10 - ROS 2 Workspace Tutorial
description: A hands-on tutorial to create and configure a ROS 2 workspace for your humanoid robot project.
sidebar_position: 10
---

# Chapter 10 - ROS 2 Workspace Tutorial

This chapter provides a comprehensive, hands-on tutorial for setting up a ROS 2 workspace specifically tailored for a humanoid robot project. We'll create a package, add a URDF model, and set up launch files and parameters. By the end, you'll have a functional workspace that can be used for simulation and visualization.

## 10.1 Prerequisites

Before starting this tutorial, ensure you have:
*   ROS 2 Humble Hawksbill installed on Ubuntu 22.04 LTS.
*   Basic familiarity with the Linux command line.
*   Python 3.10+ installed.

## 10.2 Creating the Workspace

First, create a new directory for your ROS 2 workspace:

```bash
mkdir -p ~/ros2_humanoid_ws/src
cd ~/ros2_humanoid_ws
```

Your workspace structure will look like this:
```
~/ros2_humanoid_ws/
├── src/
│   └── (source packages will go here)
```

## 10.3 Creating a Robot Description Package

Let's create a package to hold our robot's URDF model. Navigate to the `src` directory:

```bash
cd ~/ros2_humanoid_ws/src
```

Use the `ros2 pkg create` command to create a new package named `simple_humanoid_description`:

```bash
ros2 pkg create --build-type ament_cmake simple_humanoid_description
```

This command creates a basic package structure:

```
src/simple_humanoid_description/
├── CMakeLists.txt
├── package.xml
├── include/simple_humanoid_description/
├── src/
├── launch/
├── config/
├── urdf/
└── meshes/
```

## 10.4 Adding the URDF Model

Create a `urdf` directory inside your package:

```bash
mkdir ~/ros2_humanoid_ws/src/simple_humanoid_description/urdf
```

Now, create a simple URDF file for a basic humanoid. Create `simple_humanoid.urdf` in the `urdf` directory:

```bash
nano ~/ros2_humanoid_ws/src/simple_humanoid_description/urdf/simple_humanoid.urdf
```

Paste the following URDF content:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link (Pelvis) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1" />
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1" />
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
  </link>

  <!-- Torso Link -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link" />
    <child link="torso_link" />
    <origin xyz="0 0 0.15" rpy="0 0 0" />
  </joint>

  <link name="torso_link">
    <visual>
      <geometry>
        <box size="0.1 0.2 0.3" />
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.2 0.3" />
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
  </link>

  <!-- Head Link -->
  <joint name="head_joint" type="fixed">
    <parent link="torso_link" />
    <child link="head_link" />
    <origin xyz="0 0 0.2" rpy="0 0 0" />
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.075" />
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.075" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso_link" />
    <child link="left_upper_arm_link" />
    <origin xyz="0 0.15 0.1" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />
  </joint>

  <link name="left_upper_arm_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm_link" />
    <child link="left_lower_arm_link" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="0" effort="10" velocity="1" />
  </joint>

  <link name="left_lower_arm_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <!-- Right Arm (mirrored) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso_link" />
    <child link="right_upper_arm_link" />
    <origin xyz="0 -0.15 0.1" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />
  </joint>

  <link name="right_upper_arm_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm_link" />
    <child link="right_lower_arm_link" />
    <origin xyz="0 0 -0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="0" effort="10" velocity="1" />
  </joint>

  <link name="right_lower_arm_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.025" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link" />
    <child link="left_upper_leg_link" />
    <origin xyz="0 0.05 -0.1" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />
  </joint>

  <link name="left_upper_leg_link">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg_link" />
    <child link="left_lower_leg_link" />
    <origin xyz="0 0 -0.3" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="0" effort="10" velocity="1" />
  </joint>

  <link name="left_lower_leg_link">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.4" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <!-- Right Leg (mirrored) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link" />
    <child link="right_upper_leg_link" />
    <origin xyz="0 -0.05 -0.1" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />
  </joint>

  <link name="right_upper_leg_link">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg_link" />
    <child link="right_lower_leg_link" />
    <origin xyz="0 0 -0.3" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="0" effort="10" velocity="1" />
  </joint>

  <link name="right_lower_leg_link">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.4" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>
</robot>
```

## 10.5 Creating a Launch File

Now, let's create a launch file to visualize our robot. First, create the `launch` directory:

```bash
mkdir ~/ros2_humanoid_ws/src/simple_humanoid_description/launch
```

Create a launch file `view_robot.launch.py`:

```bash
nano ~/ros2_humanoid_ws/src/simple_humanoid_description/launch/view_robot.launch.py
```

Add the following Python code:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('simple_humanoid_description')

    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    urdf_path = LaunchConfiguration('urdf_path')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value=os.path.join(pkg_share, 'urdf', 'simple_humanoid.urdf'),
        description='Absolute path to robot urdf file'
    )

    # Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': open(urdf_path.perform(context=None)).read()}
        ],
        output='screen'
    )

    # Joint State Publisher GUI Node (for manual joint control)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz2 Node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        urdf_path_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
```

## 10.6 Building and Running the Workspace

Navigate back to the root of your workspace and build it:

```bash
cd ~/ros2_humanoid_ws
source /opt/ros/humble/setup.bash  # Source ROS 2 environment
colcon build --packages-select simple_humanoid_description
```

Source the workspace's setup file:

```bash
source install/setup.bash
```

Now, launch the visualization:

```bash
ros2 launch simple_humanoid_description view_robot.launch.py
```

This command will start RViz2, the robot state publisher, and the joint state publisher GUI. You should see your simple humanoid robot model in RViz, and you can use the joint state publisher GUI to manually adjust the joint angles and see the robot move.

## Conclusion

In this tutorial, we've successfully created a ROS 2 workspace, defined a simple humanoid robot model using URDF, and set up a launch file to visualize it. This forms the foundation for more complex robot projects. In the next chapter, we will create an assessment page for Module 1, summarizing the key concepts and providing practical exercises.

---
