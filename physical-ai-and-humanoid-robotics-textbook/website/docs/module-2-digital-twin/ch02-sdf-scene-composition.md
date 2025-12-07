---
title: Chapter 2 - SDF Format and Scene Composition
description: Learn how to use the Simulation Description Format (SDF) to create complex scenes for humanoid robot simulation in Gazebo.
sidebar_position: 14
---

# Chapter 2 - SDF Format and Scene Composition

The Simulation Description Format (SDF) is an XML-based language used to describe environments, robots, and objects in Gazebo. Understanding SDF is crucial for creating detailed and accurate simulation scenes for humanoid robots. This chapter will guide you through the structure of SDF files and demonstrate how to compose complex scenes.

## 2.1 SDF Structure Overview

An SDF file has a hierarchical structure with the following main elements:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="world_name">
    <!-- World-specific elements -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <model name="model_name">
      <!-- Model-specific elements -->
      <link name="link_name">
        <!-- Link-specific elements -->
        <visual name="visual_name">
          <!-- Visual properties -->
        </visual>
        <collision name="collision_name">
          <!-- Collision properties -->
        </collision>
        <inertial>
          <!-- Inertial properties -->
        </inertial>
      </link>
      <joint name="joint_name" type="revolute">
        <!-- Joint-specific elements -->
      </joint>
    </model>

    <light name="light_name">
      <!-- Light properties -->
    </light>

    <plugin name="plugin_name" filename="libplugin.so">
      <!-- Plugin-specific elements -->
    </plugin>
  </world>
</sdf>
```

## 2.2 Creating a Complex Scene for Humanoid Robots

Let's create a more complex scene that includes a humanoid robot, obstacles, and environmental elements:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_training_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Example humanoid robot (simplified) -->
    <model name="simple_humanoid">
      <!-- Base link -->
      <link name="base_link">
        <pose>0 0 1.0 0 0 0</pose>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01</iyy>
            <iyz>0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
      </link>

      <!-- Torso -->
      <link name="torso">
        <inertial>
          <mass>2.0</mass>
          <inertia>
            <ixx>0.02</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.02</iyy>
            <iyz>0</iyz>
            <izz>0.02</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.1 0.3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.1 0.3</size>
            </box>
          </geometry>
        </collision>
      </link>

      <joint name="base_to_torso" type="fixed">
        <parent>base_link</parent>
        <child>torso</child>
        <pose>0 0 0.1 0 0 0</pose>
      </joint>

      <!-- Additional links and joints would continue here -->
    </model>

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>-2 1 0.3 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Ramp for training -->
    <model name="ramp">
      <pose>3 -1 0 0 0.2 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## 2.3 Key SDF Elements for Humanoid Robots

### 2.3.1 Model Element
The `<model>` element represents a complete robot or object. For humanoid robots, models typically contain multiple links connected by joints.

### 2.3.2 Link Element
The `<link>` element represents a rigid body. Each part of a humanoid robot (torso, limbs, head) is typically a separate link. Each link contains:
- `<visual>`: How the link appears in the simulation
- `<collision>`: How the link interacts physically with other objects
- `<inertial>`: Mass and inertia properties for physics simulation

### 2.3.3 Joint Element
The `<joint>` element connects two links. For humanoid robots, common joint types include:
- `revolute`: Single axis rotation with limits (like elbows, knees)
- `continuous`: Single axis rotation without limits
- `prismatic`: Single axis translation
- `fixed`: No movement between links

### 2.3.4 Geometry Types
SDF supports various geometry types for visual and collision elements:
- `<box>`: Rectangular prism
- `<cylinder>`: Cylindrical shape
- `<sphere>`: Spherical shape
- `<mesh>`: Complex shapes from external files (STL, DAE)
- `<plane>`: Infinite plane (useful for ground)

## 2.4 Physics Configuration for Humanoid Simulation

For humanoid robots, careful physics configuration is essential:

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>  <!-- Small step size for stability -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time simulation -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- High update rate -->
  <ode>
    <solver>
      <type>quick</type>  <!-- Fast solver for real-time simulation -->
      <iters>10</iters>  <!-- Solver iterations -->
      <sor>1.0</sor>  <!-- Successive over-relaxation parameter -->
    </solver>
    <constraints>
      <cfm>0.0</cfm>  <!-- Constraint force mixing -->
      <erp>0.2</erp>  <!-- Error reduction parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## 2.5 Best Practices for SDF in Humanoid Robotics

1. **Modularity**: Create reusable SDF components for common robot parts
2. **Parameterization**: Use SDF models with parameters to create variations
3. **Performance**: Use simple collision geometries where possible
4. **Accuracy**: Ensure inertial properties match the physical robot
5. **Validation**: Test SDF files in Gazebo before complex simulation

## 2.6 Creating and Using SDF Files

To create an SDF file for your humanoid robot:
1. Plan the robot's structure (links and joints)
2. Define each link with appropriate visual, collision, and inertial properties
3. Connect links with appropriate joints
4. Test the SDF file in Gazebo
5. Iterate based on simulation behavior

## Summary

SDF is the foundation for creating complex simulation environments in Gazebo. Understanding its structure and elements is crucial for developing accurate digital twins of humanoid robots. In the next chapter, we will explore physics properties and how to achieve realistic simulation for humanoid robots.