---
title: Chapter 4 - Sensor Plugins and Data Visualization
description: Learn how to integrate and visualize data from various sensor plugins in Gazebo, including ray, camera, and IMU sensors for humanoid robots.
sidebar_position: 16
---

# Chapter 4 - Sensor Plugins and Data Visualization

Sensors are crucial components of humanoid robots, providing essential feedback about the robot's state and its environment. In Gazebo, sensor plugins simulate various types of sensors, allowing you to test perception algorithms and sensor fusion techniques in a virtual environment. This chapter covers integrating different sensor types and visualizing their data streams.

## 4.1 Overview of Gazebo Sensors

Gazebo supports a wide range of sensor types that are commonly used in humanoid robots:

- **Camera sensors**: For visual perception and object recognition
- **Ray/LIDAR sensors**: For distance measurement and mapping
- **IMU sensors**: For orientation and acceleration data
- **Force/Torque sensors**: For contact force measurement
- **GPS sensors**: For position estimation
- **Contact sensors**: For detecting collisions

## 4.2 Camera Sensors

Camera sensors simulate RGB cameras, depth cameras, or multi-camera systems. Here's how to add a camera to a humanoid robot:

```xml
<sensor name="head_camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <topic>head_camera/image_raw</topic>
  <camera>
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <plugin filename="gz-sim-camera-system" name="gz::sim::systems::Camera">
    <camera_name>head_camera</camera_name>
    <frame_name>head_camera_frame</frame_name>
  </plugin>
</sensor>
```

### 4.2.1 Depth Camera

For depth perception, you can use a depth camera:

```xml
<sensor name="depth_camera" type="depth_camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <topic>depth_camera/image_raw</topic>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <output_type>depthmap</output_type>
</sensor>
```

## 4.3 Ray/LIDAR Sensors

Ray sensors (including LIDAR) are used for distance measurement and environment mapping:

```xml
<sensor name="laser_scanner" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
  <topic>scan</topic>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle> <!-- -π radians -->
        <max_angle>3.14159</max_angle>   <!-- π radians -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
</sensor>
```

### 4.3.1 Multi-line LIDAR

For humanoid robots, you might want a 2D or 3D LIDAR:

```xml
<sensor name="3d_lidar" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
  <topic>points2</topic>
  <ray>
    <scan>
      <horizontal>
        <samples>512</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1.0</resolution>
        <min_angle>-0.2618</min_angle> <!-- -15 degrees -->
        <max_angle>0.2618</max_angle>  <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
</sensor>
```

## 4.4 IMU Sensors

IMU sensors provide orientation and acceleration data, crucial for humanoid robot balance and navigation:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <visualize>false</visualize>
  <topic>imu/data</topic>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## 4.5 Force/Torque Sensors

Force/torque sensors are useful for measuring contact forces, especially important for humanoid robots during walking or manipulation:

```xml
<sensor name="ft_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>ft_sensor/wrench</topic>
</sensor>
```

## 4.6 Adding Sensors to a Humanoid Robot Model

Here's how to add multiple sensors to a humanoid robot's head link:

```xml
<link name="head">
  <!-- ... other link elements ... -->

  <!-- Camera sensor -->
  <sensor name="head_camera" type="camera">
    <pose>0.05 0 0 0 0 0</pose> <!-- Position in front of head -->
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
    <topic>head_camera/image_raw</topic>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
  </sensor>

  <!-- IMU sensor -->
  <sensor name="head_imu" type="imu">
    <pose>0 0 0 0 0 0</pose>
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <topic>head_imu/data</topic>
  </sensor>
</link>
```

## 4.7 Visualizing Sensor Data in Gazebo

Gazebo provides visualization for many sensor types:

- **Camera sensors**: Visualize the camera view in the GUI
- **Ray sensors**: Show the laser scan rays and detected points
- **IMU sensors**: No direct visualization, but data can be monitored

To visualize sensor data, ensure the `<visualize>` tag is set to `true` in the sensor definition.

## 4.8 Sensor Noise Modeling

Real sensors have noise and inaccuracies. Gazebo allows you to model this:

```xml
<camera>
  <!-- ... other camera properties ... -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev>
  </noise>
</camera>
```

For IMU sensors, as shown earlier, you can add noise to both angular velocity and linear acceleration measurements.

## 4.9 Accessing Sensor Data in ROS 2

To access sensor data from your ROS 2 nodes, you need to subscribe to the appropriate topics:

```python
# Example Python code to subscribe to camera data
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'head_camera/image_raw',
            self.camera_callback,
            10)
        self.cv_bridge = CvBridge()

    def camera_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Process the image
        # ...
```

## 4.10 Sensor Fusion Concepts

For humanoid robots, sensor fusion combines data from multiple sensors to improve perception accuracy:

- **Visual-Inertial Odometry**: Combines camera and IMU data
- **Kalman Filters**: Fuses multiple sensor readings probabilistically
- **Extended Kalman Filters**: Handles non-linear sensor models

## Summary

Sensor plugins are essential for creating realistic humanoid robot simulations in Gazebo. By properly configuring camera, ray, IMU, and other sensors, you can test perception algorithms and sensor fusion techniques in a virtual environment. In the next chapter, we will explore Unity ML-Agents integration for AI training in digital twins.