---
title: Chapter 3 - Physics Properties and Realism
description: Explore physics properties in Gazebo including gravity, friction, inertia, and how to achieve basic realism for humanoid robot simulation.
sidebar_position: 15
---

# Chapter 3 - Physics Properties and Realism

Creating realistic physics simulation is crucial for humanoid robots, as their complex dynamics and interactions with the environment require accurate modeling of physical properties. This chapter focuses on configuring physics properties in Gazebo to achieve basic realism for humanoid robot simulation.

## 3.1 Understanding Physics in Gazebo

Gazebo uses a physics engine to simulate the motion and interactions of objects in the virtual world. The accuracy of these simulations depends on how well the physical properties of objects are defined. For humanoid robots, this includes mass distribution, friction coefficients, and inertial properties.

### 3.1.1 Physics Engine Configuration

The physics engine is configured at the world level in SDF files. Here's a detailed breakdown of key parameters:

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.0</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

- **`max_step_size`**: The maximum time step for the physics simulation. Smaller values provide more accurate but slower simulation.
- **`real_time_factor`**: Controls how fast the simulation runs relative to real time (1.0 = real-time).
- **`real_time_update_rate`**: How many times per second the physics engine updates.
- **`iters`**: Number of iterations for the constraint solver (more iterations = more accurate but slower).
- **`erp`**: Error reduction parameter (higher values correct errors faster but can cause instability).
- **`cfm`**: Constraint force mixing parameter (affects constraint stiffness).

## 3.2 Mass and Inertial Properties

For humanoid robots, accurate mass and inertial properties are critical for realistic movement and balance. Each link in the robot must have properly defined inertial properties.

### 3.2.1 Mass Distribution

The mass of each link should reflect the physical robot as closely as possible:

```xml
<inertial>
  <mass>1.5</mass>  <!-- Mass in kilograms -->
  <inertia>
    <ixx>0.01</ixx>  <!-- Moment of inertia about X-axis -->
    <ixy>0.0</ixy>   <!-- Product of inertia XY -->
    <ixz>0.0</ixz>   <!-- Product of inertia XZ -->
    <iyy>0.02</iyy>  <!-- Moment of inertia about Y-axis -->
    <iyz>0.0</iyz>   <!-- Product of inertia YZ -->
    <izz>0.015</izz> <!-- Moment of inertia about Z-axis -->
  </inertia>
</inertial>
```

For a humanoid robot, typical mass values might be:
- Head: 0.5-1.0 kg
- Torso: 5-10 kg
- Upper arm: 0.5-1.0 kg
- Lower arm: 0.3-0.7 kg
- Upper leg: 1.0-2.0 kg
- Lower leg: 0.8-1.5 kg

### 3.2.2 Calculating Inertial Properties

For simple geometric shapes, you can calculate inertial properties using standard formulas:

- **Box** (length l, width w, height h, mass m):
  - ixx = m*(w² + h²)/12
  - iyy = m*(l² + h²)/12
  - izz = m*(l² + w²)/12

- **Cylinder** (radius r, height h, mass m):
  - ixx = m*(3*r² + h²)/12
  - iyy = m*(3*r² + h²)/12
  - izz = m*r²/2

- **Sphere** (radius r, mass m):
  - ixx = iyy = izz = 2*m*r²/5

## 3.3 Friction Properties

Friction is essential for humanoid robots to maintain grip and stability. Gazebo supports two types of friction: ODE friction and bullet friction.

### 3.3.1 ODE Friction

```xml
<collision name="collision">
  <geometry>
    <box>
      <size>0.1 0.1 0.1</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>0.5</mu>        <!-- Primary friction coefficient -->
        <mu2>0.5</mu2>      <!-- Secondary friction coefficient -->
        <fdir1>1 0 0</fdir1> <!-- Direction of primary friction -->
      </ode>
    </friction>
  </surface>
</collision>
```

For humanoid robots, friction coefficients typically range from:
- Rubber feet: 0.8-1.0 (high grip)
- Plastic/metal feet: 0.3-0.6 (moderate grip)
- Socks/bare feet: 0.1-0.3 (low grip)

### 3.3.2 Bullet Friction

```xml
<surface>
  <friction>
    <bullet>
      <friction>0.5</friction>
      <friction2>0.5</friction2>
      <fdir1>1 0 0</fdir1>
      <rolling_friction>0.01</rolling_friction>
    </bullet>
  </friction>
</surface>
```

## 3.4 Joint Properties

Joints in humanoid robots require careful configuration of limits, stiffness, and damping to match real-world behavior.

### 3.4.1 Joint Limits

```xml
<joint name="knee_joint" type="revolute">
  <parent>upper_leg</parent>
  <child>lower_leg</child>
  <axis>
    <xyz>0 1 0</xyz>
    <limit>
      <lower>-2.0</lower>    <!-- Lower limit in radians -->
      <upper>0.5</upper>     <!-- Upper limit in radians -->
      <effort>30</effort>    <!-- Maximum effort in Nm -->
      <velocity>2.0</velocity> <!-- Maximum velocity in rad/s -->
    </limit>
  </axis>
</joint>
```

### 3.4.2 Joint Dynamics

```xml
<axis>
  <xyz>0 1 0</xyz>
  <dynamics>
    <damping>0.1</damping>   <!-- Damping coefficient -->
    <friction>0.05</friction> <!-- Static friction -->
    <spring_reference>0</spring_reference>
    <spring_stiffness>0</spring_stiffness>
  </dynamics>
</axis>
```

## 3.5 Creating Realistic Humanoid Models

Here's an example of a more realistic humanoid leg with proper physics properties:

```xml
<link name="foot">
  <inertial>
    <mass>0.8</mass>
    <inertia>
      <ixx>0.001</ixx>
      <ixy>0.0</ixy>
      <ixz>0.0</ixz>
      <iyy>0.002</iyy>
      <iyz>0.0</iyz>
      <izz>0.002</izz>
    </inertia>
  </inertial>
  <visual name="visual">
    <geometry>
      <box>
        <size>0.2 0.1 0.05</size>
      </box>
    </geometry>
    <material>
      <ambient>0.3 0.3 0.3 1</ambient>
      <diffuse>0.3 0.3 0.3 1</diffuse>
    </material>
  </visual>
  <collision name="collision">
    <geometry>
      <box>
        <size>0.2 0.1 0.05</size>
      </box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>
          <mu2>0.8</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
</link>
```

## 3.6 Tuning for Stability

For stable humanoid simulation:

1. **Use appropriate time steps**: Start with 0.001s and adjust if needed
2. **Balance solver iterations**: Higher iterations improve stability but reduce performance
3. **Set realistic mass ratios**: Ensure no link is disproportionately light/heavy
4. **Adjust ERP and CFM**: Fine-tune for contact stability
5. **Validate with simple tests**: Test balance and basic movements before complex behaviors

## 3.7 Validation Techniques

To validate physics realism:

1. **Static balance**: Check if the robot can stand without falling
2. **Joint range verification**: Ensure joint limits prevent impossible configurations
3. **Mass distribution check**: Verify the center of mass is reasonable
4. **Contact behavior**: Test how the robot interacts with surfaces

## Summary

Accurate physics properties are fundamental to creating realistic humanoid robot simulations in Gazebo. Properly configured mass, inertia, friction, and joint properties enable the robot to behave similarly to its physical counterpart. In the next chapter, we will explore sensor plugins and how to visualize their data streams in Gazebo.