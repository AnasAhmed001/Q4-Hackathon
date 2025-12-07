---
title: Chapter 7 - URDF Structure for Humanoid Robots
description: Explore the Unified Robot Description Format (URDF) and its application in modeling humanoid robots within ROS 2.
sidebar_position: 7
---

# Chapter 7 - URDF Structure for Humanoid Robots

The Unified Robot Description Format (URDF) is an XML format used in ROS to describe all aspects of a robot. It's crucial for visualizing robots in tools like RViz, simulating them in Gazebo, and enabling motion planning. For humanoid robots, URDF provides a structured way to define their complex articulated bodies, including links (rigid bodies) and joints (connections between links).

In this chapter, we will delve into the fundamental structure of URDF files, focusing on how they are used to represent humanoid robots.

## 7.1 Anatomy of a URDF File

A URDF file is structured around `<robot>` tags, which contain `<link>` and `<joint>` elements.

```xml
<robot name="my_humanoid_robot">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1" />
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1" />
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01" />
    </inertial>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso_joint" type="fixed">
    <parent link="base_link" />
    <child link="torso_link" />
    <origin xyz="0 0 0.15" rpy="0 0 0" />
  </joint>

  <!-- Torso Link -->
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
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01" />
    </inertial>
  </link>

  <!-- ... more links and joints for arms, legs, head ... -->

</robot>
```

### 7.1.1 The `<link>` Element

A `<link>` element represents a rigid body part of the robot. It can contain:
*   **`<visual>`**: Defines the visual properties, such as the shape, size, color, and texture. This is what you see in RViz.
    *   `<geometry>`: Specifies the primitive shape (box, cylinder, sphere) or a mesh file (`.dae`, `.stl`).
    *   `<material>`: Defines the color (RGBA) or texture.
*   **`<collision>`**: Defines the physical properties used for collision detection in simulations. It often mirrors the visual geometry but can be simplified for performance.
*   **`<inertial>`**: Defines the mass and inertia tensor, critical for realistic physics simulation.

### 7.1.2 The `<joint>` Element

A `<joint>` element connects two links together, defining their relative motion. Key attributes and elements include:
*   **`name`**: A unique identifier for the joint.
*   **`type`**: Specifies the type of motion allowed:
    *   `revolute`: A single axis of rotation with joint limits (e.g., elbow, knee).
    *   `continuous`: A single axis of rotation without joint limits (e.g., spinning wheel).
    *   `prismatic`: A single axis of translation with joint limits (e.g., linear actuator).
    *   `fixed`: No motion allowed (e.g., connecting a camera to a robot body).
    *   `planar`: Motion in a plane.
    *   `floating`: Full 6-DOF motion.
*   **`<parent link="link_name" />`**: Specifies the name of the parent link.
*   **`<child link="link_name" />`**: Specifies the name of the child link.
*   **`<origin xyz="X Y Z" rpy="ROLL PITCH YAW" />`**: Defines the transform from the parent link's origin to the child link's origin. `xyz` are translation offsets, and `rpy` are rotation offsets (Roll, Pitch, Yaw in radians).
*   **`<axis xyz="X Y Z" />`**: Defines the axis of rotation or translation for revolute, continuous, and prismatic joints.
*   **`<limit lower="L" upper="U" effort="E" velocity="V" />`**: Defines the lower and upper bounds for revolute and prismatic joints, and the maximum effort and velocity.

## 7.2 Modeling a Simple Humanoid

Let's consider how we would structure a URDF for a very simple humanoid robot.

```xml
<robot name="simple_humanoid">

  <!-- Base Link (Pelvis) -->
  <link name="pelvis_link">
    <!-- ... visual, collision, inertial for pelvis ... -->
  </link>

  <!-- Torso Joint (Fixed) -->
  <joint name="pelvis_to_torso_joint" type="fixed">
    <parent link="pelvis_link" />
    <child link="torso_link" />
    <origin xyz="0 0 0.1" rpy="0 0 0" />
  </joint>

  <!-- Torso Link -->
  <link name="torso_link">
    <!-- ... visual, collision, inertial for torso ... -->
  </link>

  <!-- Left Shoulder Joint (Revolute) -->
  <joint name="left_shoulder_pitch_joint" type="revolute">
    <parent link="torso_link" />
    <child link="left_upper_arm_link" />
    <origin xyz="0 0.1 0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" /> <!-- Pitch axis -->
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />
  </joint>

  <!-- Left Upper Arm Link -->
  <link name="left_upper_arm_link">
    <!-- ... visual, collision, inertial ... -->
  </link>

  <!-- Left Elbow Joint (Revolute) -->
  <joint name="left_elbow_pitch_joint" type="revolute">
    <parent link="left_upper_arm_link" />
    <child link="left_forearm_link" />
    <origin xyz="0 0 0.2" rpy="0 0 0" />
    <axis xyz="0 1 0" /> <!-- Pitch axis -->
    <limit lower="-1.57" upper="0" effort="10" velocity="1" />
  </joint>

  <!-- Left Forearm Link -->
  <link name="left_forearm_link">
    <!-- ... visual, collision, inertial ... -->
  </link>

  <!-- ... repeat for right arm, left leg, right leg, head ... -->

</robot>
```

This example demonstrates how each part of the humanoid (links) is connected by joints, defining its degrees of freedom. The `origin` and `axis` elements are crucial for correctly positioning and orienting the robot's components.

## 7.3 Best Practices for URDF

*   **Modularity**: Break down complex robots into smaller, reusable URDF components if possible.
*   **Clarity**: Use clear and descriptive names for links and joints.
*   **Consistency**: Maintain consistent coordinate frames (e.g., Z-up).
*   **Simplification**: Use simplified collision geometries for performance in simulations.
*   **Validation**: Always validate your URDF files with tools like `check_urdf` to catch syntax errors.

## Conclusion

Understanding URDF is foundational for working with robots in ROS 2. By mastering the concepts of links, joints, and their properties, you can accurately model complex humanoid robots for visualization, simulation, and control. In the next chapter, we will explore different types of joints and how they form kinematic chains in humanoid robot design.

---
