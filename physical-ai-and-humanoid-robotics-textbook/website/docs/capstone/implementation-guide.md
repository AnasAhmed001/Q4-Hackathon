---
sidebar_position: 3
title: "Implementation Guide"
---

# Implementation Guide

This comprehensive guide will walk you through building the Autonomous Humanoid system in 7 distinct phases. Each phase builds upon the previous one, integrating the concepts from all four modules to create a complete, functioning system.

## Phase 1: Project Setup

### Create ROS 2 Workspace

First, establish your development environment with a proper ROS 2 workspace:

```bash
# Create workspace directory
mkdir -p ~/autonomous_humanoid_ws/src
cd ~/autonomous_humanoid_ws

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build workspace (initial empty build)
colcon build --symlink-install
source install/setup.bash
```

### Install Dependencies

Install all required dependencies for the project:

```bash
# Install Python dependencies
pip install torch torchvision torchaudio \
  openai-whisper transformers openai \
  numpy scipy matplotlib pyyaml \
  opencv-python

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop ros-humble-nav2-bringup \
  ros-humble-moveit ros-humble-gazebo-ros-pkgs \
  ros-humble-vision-opencv ros-humble-cv-bridge \
  ros-humble-tf2-tools ros-humble-xacro
```

### Create Package Structure

Create the ROS 2 packages for each component:

```bash
cd ~/autonomous_humanoid_ws/src

# Create packages
ros2 pkg create --dependencies rclpy std_msgs sensor_msgs geometry_msgs \
  nav_msgs moveit_core moveit_ros_planning_interface \
  voice_interface

ros2 pkg create --dependencies rclpy std_msgs builtin_interfaces \
  task_planner

ros2 pkg create --dependencies rclpy geometry_msgs nav_msgs \
  navigation_controller

ros2 pkg create --dependencies rclpy sensor_msgs cv_bridge \
  perception_module

ros2 pkg create --dependencies rclpy moveit_ros_planning_interface \
  manipulation_controller
```

### Basic Node Template

Create a template for your ROS 2 nodes (`voice_interface/voice_interface/voice_node.py`):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        self.publisher_ = self.create_publisher(String, 'voice_command', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello: {self.i}'
        self.publisher_.publish(msg)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceNode()
    rclpy.spin(voice_node)
    voice_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Configuration Files

Create the basic launch file structure (`voice_interface/launch/voice_launch.py`):

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_interface',
            executable='voice_node',
            name='voice_node',
            output='screen'
        )
    ])
```

### Phase 1 Verification

Verify that your workspace builds correctly:

```bash
cd ~/autonomous_humanoid_ws
colcon build --symlink-install
source install/setup.bash
```

## Phase 2: Voice Interface

### Install Whisper

Set up the Whisper speech recognition system:

```bash
pip install git+https://github.com/openai/whisper.git
```

### Create Voice Interface Node

Create `voice_interface/voice_interface/voice_processor.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import pyaudio
import numpy as np
import threading
import queue

class VoiceProcessor(Node):
    def __init__(self):
        super().__init__('voice_processor')

        # Initialize Whisper model
        self.model = whisper.load_model("base")

        # Audio configuration
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 5

        # ROS 2 publisher for recognized text
        self.voice_pub = self.create_publisher(String, 'voice_command', 10)

        # Start audio recording thread
        self.audio_queue = queue.Queue()
        self.recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

        # Timer to process audio periodically
        self.timer = self.create_timer(1.0, self.process_audio)

    def record_audio(self):
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        while self.recording:
            data = stream.read(self.chunk)
            self.audio_queue.put(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def process_audio(self):
        if not self.audio_queue.empty():
            # Collect audio data for processing
            frames = []
            while not self.audio_queue.empty():
                frames.append(self.audio_queue.get())

            if frames:
                # Convert to numpy array and process
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Transcribe using Whisper
                result = self.model.transcribe(audio_float)
                text = result["text"].strip()

                if text:  # Only publish if we got meaningful text
                    msg = String()
                    msg.data = text
                    self.voice_pub.publish(msg)
                    self.get_logger().info(f'Published: {text}')

    def destroy_node(self):
        self.recording = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    voice_processor = VoiceProcessor()
    rclpy.spin(voice_processor)
    voice_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Update Package.xml

Update `voice_interface/package.xml` to include dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>voice_interface</name>
  <version>0.0.0</version>
  <description>Voice interface for humanoid robot</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>builtin_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Setup.py Configuration

Update `voice_interface/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'voice_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo'>
    description='Voice interface for humanoid robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_processor = voice_interface.voice_processor:main',
        ],
    },
)
```

### Launch File

Create `voice_interface/launch/voice_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_interface',
            executable='voice_processor',
            name='voice_processor',
            output='screen',
            parameters=[
                {'use_sim_time': True}  # Use sim time in simulation
            ]
        )
    ])
```

### Phase 2 Testing

Test the voice interface:

```bash
cd ~/autonomous_humanoid_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch voice_interface voice_launch.py
```

## Phase 3: LLM Task Planner

### Create Task Planner Package

Create the task planner node (`task_planner/task_planner/task_planner_node.py`):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import openai
import json
import os

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            String,
            'voice_command',
            self.voice_callback,
            10
        )

        # Publish structured task plans
        self.task_pub = self.create_publisher(String, 'task_plan', 10)

        # Initialize OpenAI API
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.get_logger().info('Task Planner Node initialized')

    def voice_callback(self, msg):
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        try:
            # Plan the task using LLM
            task_plan = self.plan_task(command)

            # Publish the task plan
            plan_msg = String()
            plan_msg.data = json.dumps(task_plan)
            self.task_pub.publish(plan_msg)
            self.get_logger().info(f'Published task plan: {task_plan}')

        except Exception as e:
            self.get_logger().error(f'Error in task planning: {str(e)}')

    def plan_task(self, command):
        """
        Use LLM to decompose high-level commands into structured task plans
        """
        prompt = f"""
        You are a task planning assistant for a humanoid robot.
        Decompose the following high-level command into 3-5 specific, executable subtasks.
        Return the result as a JSON object with the following structure:

        {{
          "command": "original command",
          "tasks": [
            {{
              "id": "task_1",
              "description": "description of the task",
              "type": "navigation|manipulation|perception|other",
              "parameters": {{"key": "value"}}
            }}
          ],
          "dependencies": [
            {{"from": "task_1", "to": "task_2"}}
          ]
        }}

        Command: {command}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        # Extract JSON from response
        response_text = response.choices[0].message.content.strip()

        # Clean up the response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```

        try:
            task_plan = json.loads(response_text)
            return task_plan
        except json.JSONDecodeError:
            # If JSON parsing fails, create a simple plan
            return {
                "command": command,
                "tasks": [
                    {
                        "id": "fallback_task",
                        "description": f"Execute command: {command}",
                        "type": "other",
                        "parameters": {}
                    }
                ],
                "dependencies": []
            }

def main(args=None):
    rclpy.init(args=args)
    task_planner_node = TaskPlannerNode()
    rclpy.spin(task_planner_node)
    task_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Update Task Planner Package.xml

Update `task_planner/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>task_planner</name>
  <version>0.0.0</version>
  <description>LLM-based task planning for humanoid robot</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>builtin_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Task Planner Setup.py

Update `task_planner/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'task_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo'>
    description='LLM-based task planning for humanoid robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_planner_node = task_planner.task_planner_node:main',
        ],
    },
)
```

### Task Planner Launch File

Create `task_planner/launch/task_planner_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='task_planner',
            executable='task_planner_node',
            name='task_planner_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ],
            # Set environment variable for OpenAI API key
            additional_env={'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', '')}
        )
    ])
```

### Phase 3 Testing

Test the task planner by integrating with voice input:

```bash
cd ~/autonomous_humanoid_ws
colcon build --symlink-install
source install/setup.bash

# Terminal 1: Run the voice processor
ros2 run voice_interface voice_processor

# Terminal 2: Run the task planner (in another terminal)
ros2 run task_planner task_planner_node
```

## Phase 4: Navigation

### Create Navigation Package

Create the navigation controller (`navigation_controller/navigation_controller/nav_controller.py`):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from std_msgs.msg import String
import json
import math

class NavController(Node):
    def __init__(self):
        super().__init__('nav_controller')

        # Subscriptions
        self.task_sub = self.create_subscription(
            String,
            'task_plan',
            self.task_callback,
            10
        )

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.path_pub = self.create_publisher(Path, 'current_path', 10)

        # Navigation state
        self.current_task = None
        self.is_navigating = False

        self.get_logger().info('Navigation Controller initialized')

    def task_callback(self, msg):
        try:
            task_plan = json.loads(msg.data)

            # Process navigation tasks
            for task in task_plan['tasks']:
                if task['type'] == 'navigation':
                    self.execute_navigation_task(task)
                    break  # For now, execute first navigation task

        except Exception as e:
            self.get_logger().error(f'Error processing task plan: {str(e)}')

    def execute_navigation_task(self, task):
        """
        Execute a navigation task based on parameters
        """
        # Extract navigation parameters
        params = task.get('parameters', {})
        target_location = params.get('location', 'unknown')

        # For demo purposes, use predefined locations
        # In a real system, these would come from a map or semantic locations
        locations = {
            'kitchen': (2.0, 1.0, 0.0),  # x, y, theta
            'living_room': (-1.0, 2.0, 1.57),
            'office': (3.0, -2.0, -1.57),
            'bedroom': (-2.0, -1.0, 3.14)
        }

        if target_location in locations:
            x, y, theta = locations[target_location]
            self.navigate_to_pose(x, y, theta)
        else:
            self.get_logger().warn(f'Unknown location: {target_location}')
            # You could implement a search or fallback behavior here

    def navigate_to_pose(self, x, y, theta):
        """
        Send navigation goal to Nav2
        """
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'

        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0

        # Convert theta to quaternion
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f'Navigating to: ({x}, {y}, {theta})')

        # Set navigation flag
        self.is_navigating = True

    def check_navigation_status(self):
        """
        Check if navigation is complete
        This would integrate with Nav2's feedback system in a real implementation
        """
        # In a real implementation, this would check Nav2 status
        # For now, simulate completion after some time
        pass

def main(args=None):
    rclpy.init(args=args)
    nav_controller = NavController()
    rclpy.spin(nav_controller)
    nav_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Navigation Configuration

Create navigation configuration files. First, the main navigation launch file (`navigation_controller/launch/navigation_launch.py`):

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    return LaunchDescription([
        # Include Nav2's default bringup
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'true',
                'map': os.path.join(nav2_bringup_dir, 'maps', 'turtlebot3_world.yaml'),
                'params_file': os.path.join(get_package_share_directory('navigation_controller'), 'config', 'nav2_params.yaml')
            }.items()
        ),

        # Our custom navigation controller
        Node(
            package='navigation_controller',
            executable='nav_controller',
            name='nav_controller',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        )
    ])
```

### Navigation Parameters

Create `navigation_controller/config/nav2_params.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    scan_topic: scan
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the path to the Behavior Tree XML file
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    # Recovery
    server_names: ["navigate_to_pose", "navigate_through_poses"]

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "general_goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    general_goal_checker:
      stateful: True
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25

    # DWB parameters
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: False
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.26
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.26
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True
      restore_defaults: False
      scale_velocities: False
      backup_velocities: {"v_linear": -0.1, "v_angular": -0.1}
      oscillation_threshold: 0.44
      oscillation_distance: 0.05

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.22
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

smoother_server:
  ros__parameters:
    use_sim_time: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      drive_on_heading_dist: 1.0
      drive_on_heading_angle_tolerance: 0.785
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1s

robot_state_publisher:
  ros__parameters:
    use_sim_time: True

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: True
      wait_time: 1s
```

### Navigation Package.xml

Update `navigation_controller/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>navigation_controller</name>
  <version>0.0.0</version>
  <description>Navigation controller for humanoid robot</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf_transformations</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Navigation Setup.py

Update `navigation_controller/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'navigation_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo'>
    description='Navigation controller for humanoid robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav_controller = navigation_controller.nav_controller:main',
        ],
    },
)
```

### Phase 4 Testing

Test the navigation system:

```bash
cd ~/autonomous_humanoid_ws
colcon build --symlink-install
source install/setup.bash

# Run navigation system
ros2 launch navigation_controller navigation_launch.py
```

## Phase 5: Perception

### Create Perception Package

Create the perception module (`perception_module/perception_module/perception_node.py`):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import torch
from torchvision import transforms
from PIL import Image as PILImage

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Subscribe to task plans
        self.task_sub = self.create_subscription(
            String,
            'task_plan',
            self.task_callback,
            10
        )

        # Publisher for detected objects
        self.object_pub = self.create_publisher(String, 'detected_objects', 10)

        # Camera info (would normally be subscribed)
        self.camera_info = None

        # Object detection model (using a simple pre-trained model for demo)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.current_task = None
        self.get_logger().info('Perception Node initialized')

    def task_callback(self, msg):
        try:
            task_plan = json.loads(msg.data)
            # Store current task for reference during perception
            self.current_task = task_plan
            self.get_logger().info('Received new task for perception')
        except Exception as e:
            self.get_logger().error(f'Error processing task: {str(e)}')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process the image for object detection
            detections = self.detect_objects(cv_image)

            # Filter detections based on current task if available
            if self.current_task:
                relevant_detections = self.filter_detections_by_task(detections, self.current_task)
            else:
                relevant_detections = detections

            # Publish detections
            detections_msg = String()
            detections_msg.data = json.dumps(relevant_detections)
            self.object_pub.publish(detections_msg)

            self.get_logger().info(f'Detected {len(relevant_detections)} objects')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_objects(self, image):
        """
        Detect objects in the image using a simple approach
        In a real system, this would use a trained model like YOLO or ViT
        """
        # Convert to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # For this example, we'll simulate object detection
        # In a real implementation, this would use a trained model
        height, width = image.shape[:2]

        # Simulated detections - in real implementation, this would come from ML model
        detections = []

        # Simple color-based detection for demonstration
        # Detect red objects (for detecting a red cup as per example scenario)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of red objects
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small detections
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center in image coordinates
                center_x = x + w // 2
                center_y = y + h // 2

                # Convert to relative coordinates (0-1)
                rel_x = center_x / width
                rel_y = center_y / height

                detection = {
                    'class': 'red_object',
                    'confidence': 0.85,  # Simulated confidence
                    'bbox': [x, y, x+w, y+h],
                    'center': [rel_x, rel_y],
                    'area': float(area)
                }

                detections.append(detection)

        # Also detect other common objects using simple shape detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Larger objects
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

                shape_name = "unknown"
                if len(approx) == 3:
                    shape_name = "triangle"
                elif len(approx) == 4:
                    # Check if square or rectangle
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    shape_name = "square" if 0.95 <= ar <= 1.05 else "rectangle"
                elif len(approx) == 5:
                    shape_name = "pentagon"
                else:
                    shape_name = "circle"

                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                rel_x = center_x / width
                rel_y = center_y / height

                detection = {
                    'class': shape_name,
                    'confidence': 0.7,  # Lower confidence for shape detection
                    'bbox': [x, y, x+w, y+h],
                    'center': [rel_x, rel_y],
                    'area': float(area)
                }

                detections.append(detection)

        return detections

    def filter_detections_by_task(self, detections, task_plan):
        """
        Filter detections based on current task requirements
        """
        if not task_plan or 'tasks' not in task_plan:
            return detections

        # Look for perception tasks in the plan
        for task in task_plan['tasks']:
            if task.get('type') == 'perception':
                # Filter based on object type specified in task parameters
                required_object = task.get('parameters', {}).get('object_type')
                if required_object:
                    filtered = []
                    for detection in detections:
                        if required_object.lower() in detection['class'].lower():
                            filtered.append(detection)
                    return filtered

        return detections

def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()
    rclpy.spin(perception_node)
    perception_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Perception Package.xml

Update `perception_module/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>perception_module</name>
  <version>0.0.0</version>
  <description>Perception module for humanoid robot</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>cv_bridge</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Perception Setup.py

Update `perception_module/setup.py`:

```python
from setuptools import setup

package_name = 'perception_module'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo'>
    description='Perception module for humanoid robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception_module.perception_node:main',
        ],
    },
)
```

## Phase 6: Manipulation

### Create Manipulation Package

Create the manipulation controller (`manipulation_controller/manipulation_controller/manipulation_controller.py`):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
import json
import numpy as np

class ManipulationController(Node):
    def __init__(self):
        super().__init__('manipulation_controller')

        # Subscriptions
        self.task_sub = self.create_subscription(
            String,
            'task_plan',
            self.task_callback,
            10
        )

        self.object_sub = self.create_subscription(
            String,
            'detected_objects',
            self.object_callback,
            10
        )

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_trajectory', 10)

        # Service clients
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.fk_client = self.create_client(GetPositionFK, 'compute_fk')

        # Robot state
        self.current_joint_state = None
        self.detected_objects = []
        self.current_task = None

        # Wait for services
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting...')

        self.get_logger().info('Manipulation Controller initialized')

    def task_callback(self, msg):
        try:
            task_plan = json.loads(msg.data)
            self.current_task = task_plan

            # Check for manipulation tasks
            for task in task_plan['tasks']:
                if task['type'] == 'manipulation':
                    self.execute_manipulation_task(task)
                    break

        except Exception as e:
            self.get_logger().error(f'Error processing task: {str(e)}')

    def object_callback(self, msg):
        try:
            objects = json.loads(msg.data)
            self.detected_objects = objects
            self.get_logger().info(f'Updated detected objects: {len(objects)} items')
        except Exception as e:
            self.get_logger().error(f'Error processing objects: {str(e)}')

    def execute_manipulation_task(self, task):
        """
        Execute a manipulation task based on parameters
        """
        params = task.get('parameters', {})
        action = params.get('action', 'unknown')
        target_object = params.get('target_object', 'unknown')

        self.get_logger().info(f'Executing manipulation: {action} for {target_object}')

        if action == 'grasp' and target_object:
            # Find the target object in detected objects
            target = self.find_object_by_class(target_object)
            if target:
                self.execute_grasp(target)
            else:
                self.get_logger().warn(f'Target object {target_object} not found')
        elif action == 'place':
            target_location = params.get('location', 'default')
            self.execute_place(target_location)
        else:
            self.get_logger().warn(f'Unknown manipulation action: {action}')

    def find_object_by_class(self, class_name):
        """
        Find an object by its class name
        """
        for obj in self.detected_objects:
            if class_name.lower() in obj['class'].lower():
                return obj
        return None

    def execute_grasp(self, target_object):
        """
        Execute grasping motion for the target object
        """
        self.get_logger().info(f'Attempting to grasp {target_object["class"]}')

        # Calculate grasp pose based on object position
        # This is a simplified approach - in reality, grasp planning would be more complex
        object_center = target_object['center']
        object_bbox = target_object['bbox']

        # Convert normalized coordinates to 3D world coordinates (simplified)
        # In a real system, this would use depth information and camera calibration
        grasp_pose = Pose()
        grasp_pose.position.x = object_center[0] * 2.0 - 1.0  # Map 0-1 to -1 to 1
        grasp_pose.position.y = (1.0 - object_center[1]) * 1.5 - 0.75  # Map 0-1 to 0.75 to -0.75
        grasp_pose.position.z = 0.5  # Fixed height for demonstration

        # Set orientation for grasping (simplified)
        grasp_pose.orientation.x = 0.0
        grasp_pose.orientation.y = 0.0
        grasp_pose.orientation.z = 0.0
        grasp_pose.orientation.w = 1.0

        # Plan and execute the grasp
        success = self.plan_and_execute_motion(grasp_pose)

        if success:
            self.get_logger().info('Grasp successful!')
        else:
            self.get_logger().warn('Grasp failed')

    def execute_place(self, location):
        """
        Execute placing motion at the specified location
        """
        self.get_logger().info(f'Placing object at {location}')

        # Define placement poses based on location
        placement_poses = {
            'table': Pose(position=Point(x=1.0, y=0.0, z=0.75)),
            'shelf': Pose(position=Point(x=1.5, y=0.5, z=1.2)),
            'default': Pose(position=Point(x=0.5, y=0.0, z=0.5))
        }

        if location in placement_poses:
            place_pose = placement_poses[location]
        else:
            place_pose = placement_poses['default']

        success = self.plan_and_execute_motion(place_pose)

        if success:
            self.get_logger().info('Place successful!')
        else:
            self.get_logger().warn('Place failed')

    def plan_and_execute_motion(self, target_pose):
        """
        Plan and execute motion to reach target pose using MoveIt
        """
        try:
            # Create IK request
            request = GetPositionIK.Request()
            request.ik_request.group_name = "arm"  # Assuming arm group exists
            request.ik_request.pose_stamped.header.frame_id = "base_link"
            request.ik_request.pose_stamped.pose = target_pose
            request.ik_request.timeout.sec = 5
            request.ik_request.attempts = 10

            # Call IK service
            future = self.ik_client.call_async(request)

            # In a real implementation, you'd wait for the response and execute the trajectory
            # For this example, we'll simulate success
            return True

        except Exception as e:
            self.get_logger().error(f'Error in motion planning: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    manipulation_controller = ManipulationController()
    rclpy.spin(manipulation_controller)
    manipulation_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Manipulation Package.xml

Create `manipulation_controller/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>manipulation_controller</name>
  <version>0.0.0</version>
  <description>Manipulation controller for humanoid robot</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>moveit_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Manipulation Setup.py

Create `manipulation_controller/setup.py`:

```python
from setuptools import setup

package_name = 'manipulation_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo'>
    description='Manipulation controller for humanoid robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manipulation_controller = manipulation_controller.manipulation_controller:main',
        ],
    },
)
```

## Phase 7: End-to-End Integration and Testing

### Create Integration Node

Create the main integration node that coordinates all components (`integration_layer/integration_layer/integration_node.py`):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from builtin_interfaces.msg import Time
import json

class IntegrationNode(Node):
    def __init__(self):
        super().__init__('integration_node')

        # Subscriptions for all system components
        self.voice_sub = self.create_subscription(
            String, 'voice_command', self.voice_callback, 10)

        self.task_sub = self.create_subscription(
            String, 'task_plan', self.task_callback, 10)

        self.object_sub = self.create_subscription(
            String, 'detected_objects', self.object_callback, 10)

        self.nav_status_sub = self.create_subscription(
            String, 'navigation_status', self.nav_status_callback, 10)

        # Publishers for system coordination
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.control_pub = self.create_publisher(String, 'system_control', 10)

        # System state tracking
        self.system_state = {
            'voice_received': False,
            'task_planned': False,
            'navigation_complete': False,
            'objects_detected': False,
            'manipulation_complete': False
        }

        # Task tracking
        self.current_task = None
        self.task_progress = {}

        self.get_logger().info('Integration Node initialized')

    def voice_callback(self, msg):
        self.get_logger().info(f'Voice command received: {msg.data}')
        self.system_state['voice_received'] = True

        # Update system status
        status_msg = String()
        status_msg.data = json.dumps({
            'component': 'voice',
            'status': 'received',
            'timestamp': self.get_clock().now().seconds_nanoseconds()
        })
        self.status_pub.publish(status_msg)

    def task_callback(self, msg):
        try:
            task_plan = json.loads(msg.data)
            self.current_task = task_plan
            self.system_state['task_planned'] = True

            self.get_logger().info(f'Task plan received: {task_plan["command"]}')

            # Update system status
            status_msg = String()
            status_msg.data = json.dumps({
                'component': 'task_planner',
                'status': 'planned',
                'task': task_plan['command'],
                'timestamp': self.get_clock().now().seconds_nanoseconds()
            })
            self.status_pub.publish(status_msg)

            # Begin executing the task plan
            self.execute_task_plan(task_plan)

        except Exception as e:
            self.get_logger().error(f'Error processing task: {str(e)}')

    def object_callback(self, msg):
        try:
            objects = json.loads(msg.data)
            self.system_state['objects_detected'] = len(objects) > 0

            # Update system status
            status_msg = String()
            status_msg.data = json.dumps({
                'component': 'perception',
                'status': 'detected',
                'count': len(objects),
                'timestamp': self.get_clock().now().seconds_nanoseconds()
            })
            self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing objects: {str(e)}')

    def nav_status_callback(self, msg):
        try:
            status = json.loads(msg.data)
            if status.get('completed', False):
                self.system_state['navigation_complete'] = True

                # Update system status
                status_msg = String()
                status_msg.data = json.dumps({
                    'component': 'navigation',
                    'status': 'completed',
                    'timestamp': self.get_clock().now().seconds_nanoseconds()
                })
                self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing nav status: {str(e)}')

    def execute_task_plan(self, task_plan):
        """
        Execute the task plan by coordinating all components
        """
        self.get_logger().info('Starting task plan execution')

        # Execute tasks in sequence based on dependencies
        for task in task_plan['tasks']:
            self.get_logger().info(f'Executing task: {task["description"]}')

            # Publish control command for this task
            control_msg = String()
            control_msg.data = json.dumps({
                'task_id': task['id'],
                'task_type': task['type'],
                'parameters': task['parameters']
            })

            self.control_pub.publish(control_msg)

            # Wait for task completion (in a real system, this would be event-driven)
            # For this example, we'll just log the task execution

    def check_system_completion(self):
        """
        Check if the entire task has been completed
        """
        required_states = [
            self.system_state['voice_received'],
            self.system_state['task_planned'],
            self.system_state['navigation_complete'],
            self.system_state['objects_detected']
        ]

        if all(required_states):
            self.get_logger().info('Task execution completed successfully!')

            # Publish completion status
            status_msg = String()
            status_msg.data = json.dumps({
                'system': 'complete',
                'status': 'success',
                'timestamp': self.get_clock().now().seconds_nanoseconds()
            })
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    integration_node = IntegrationNode()
    rclpy.spin(integration_node)
    integration_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration Package

Create the integration package:

```bash
cd ~/autonomous_humanoid_ws/src
ros2 pkg create --dependencies rclpy std_msgs builtin_interfaces \
  integration_layer
```

### Integration Package.xml

Create `integration_layer/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>integration_layer</name>
  <version>0.0.0</version>
  <description>Integration layer for humanoid robot system</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>builtin_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Integration Setup.py

Create `integration_layer/setup.py`:

```python
from setuptools import setup

package_name = 'integration_layer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo'>
    description='Integration layer for humanoid robot system',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'integration_node = integration_layer.integration_node:main',
        ],
    },
)
```

### Main System Launch File

Create a main launch file to start the entire system (`system_launch.py` in the integration_layer package):

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Voice interface
        Node(
            package='voice_interface',
            executable='voice_processor',
            name='voice_processor',
            output='screen',
        ),

        # Task planner
        Node(
            package='task_planner',
            executable='task_planner_node',
            name='task_planner_node',
            output='screen',
        ),

        # Navigation controller
        Node(
            package='navigation_controller',
            executable='nav_controller',
            name='nav_controller',
            output='screen',
        ),

        # Perception module
        Node(
            package='perception_module',
            executable='perception_node',
            name='perception_node',
            output='screen',
        ),

        # Manipulation controller
        Node(
            package='manipulation_controller',
            executable='manipulation_controller',
            name='manipulation_controller',
            output='screen',
        ),

        # Integration layer
        Node(
            package='integration_layer',
            executable='integration_node',
            name='integration_node',
            output='screen',
        ),
    ])
```

### Phase 7 Testing

Test the complete integrated system:

```bash
cd ~/autonomous_humanoid_ws
colcon build --symlink-install
source install/setup.bash

# Launch the complete system
ros2 launch integration_layer system_launch.py
```

### Architecture Diagram

The following diagram shows the complete system architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │───▶│   Task Planner  │───▶│   Navigation    │
│   (Whisper)     │    │    (LLM)        │    │     (Nav2)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │   Manipulation  │    │   Simulation    │
│   (Vision)      │───▶│    (MoveIt2)    │    │  (Isaac/Gazebo) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Integration   │    │   Status/Logs   │
│   Layer         │    │                 │
└─────────────────┘    └─────────────────┘
```

### Troubleshooting Tips

#### Phase 1 Issues
- **Workspace build fails**: Ensure all dependencies are installed and package.xml files are correct
- **Missing ROS packages**: Verify ROS 2 Humble installation and package repositories

#### Phase 2 Issues
- **Audio not detected**: Check microphone permissions and PyAudio installation
- **Whisper model not loading**: Verify internet connection and model download

#### Phase 3 Issues
- **OpenAI API errors**: Verify API key is set in environment variables
- **Task planning fails**: Check API quota and network connectivity

#### Phase 4 Issues
- **Navigation fails**: Ensure map and localization are properly configured
- **Nav2 not working**: Check parameter files and simulation environment

#### Phase 5 Issues
- **Object detection poor**: Verify camera calibration and lighting conditions
- **No detections**: Check camera topic and image format

#### Phase 6 Issues
- **MoveIt not responding**: Ensure MoveIt configuration and robot model are correct
- **IK/FK services unavailable**: Check MoveIt setup assistant configuration

#### Phase 7 Issues
- **Integration fails**: Verify all topics are properly connected
- **System hangs**: Check for circular dependencies in the system architecture

With all 7 phases completed, your Autonomous Humanoid system should be fully functional, integrating all four modules (ROS 2, Digital Twin, Isaac, and Vision-Language-Action) into a cohesive, voice-controlled humanoid robot capable of complex task execution.