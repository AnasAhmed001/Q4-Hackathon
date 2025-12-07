# Chapter 6: rclpy Python Bindings

## Learning Objectives
*   Understand the role of `rclpy` as the Python client library for ROS 2.
*   Learn the basic structure of `rclpy` applications.
*   Explore key `rclpy` APIs for node creation, communication, and parameter management.
*   Write robust and efficient ROS 2 Python nodes.

## Prerequisites
*   Familiarity with ROS 2 core concepts: nodes, topics, services, actions (Chapters 1-5).
*   Intermediate Python proficiency, including object-oriented programming.

## Theory & Concepts

`rclpy` is the official Python client library for ROS 2, providing a Pythonic interface to all core ROS 2 functionalities. It allows developers to write ROS 2 nodes, publishers, subscribers, service servers, service clients, action servers, and action clients using Python. `rclpy` is built on `rcl` (ROS Client Library) and `rmw` (ROS Middleware Wrapper), which are C libraries, ensuring high performance and seamless integration with the underlying DDS middleware.

### Why `rclpy`?

Python's ease of use, extensive libraries for data processing, and rapid prototyping capabilities make it an excellent choice for many robotics applications. `rclpy` bridges the gap between Python's flexibility and ROS 2's robust communication infrastructure, making it a popular choice for:

*   Rapid prototyping of robot behaviors.
*   Interfacing with high-level AI/ML frameworks.
*   Developing complex control logic and data processing pipelines.
*   Creating user interfaces and logging tools.

### Basic `rclpy` Node Structure

All `rclpy` applications follow a similar structure:

1.  **Import `rclpy` and `Node`**: Start by importing the necessary modules.
2.  **Initialize `rclpy`**: `rclpy.init(args=args)` initializes the ROS 2 client library.
3.  **Create a Node**: Instantiate a class inheriting from `rclpy.node.Node`.
4.  **Spin the Node**: `rclpy.spin(node)` keeps the node alive and allows callbacks (e.g., topic callbacks, timer callbacks) to be executed.
5.  **Shutdown `rclpy`**: `rclpy.shutdown()` cleans up ROS 2 resources.

### Key `rclpy` APIs

*   **`Node` class**: The base class for all `rclpy` nodes. Provides access to ROS 2 functionalities like `create_publisher`, `create_subscription`, `create_service`, `create_client`, `create_timer`, `get_logger`, `set_parameters`, etc.
*   **`rclpy.init()` / `rclpy.shutdown()`**: Manage the lifecycle of the ROS 2 client library.
*   **`rclpy.spin()` / `rclpy.spin_once()` / `rclpy.spin_until_future_complete()`**: Control the execution flow of the node, allowing it to process pending callbacks.
*   **`rclpy.logging`**: The logging interface for ROS 2, allowing nodes to output messages with different severity levels (DEBUG, INFO, WARN, ERROR, FATAL).
*   **QoS Profiles**: As discussed in Chapter 1, `rclpy` allows explicit configuration of Quality of Service profiles for publishers and subscribers to fine-tune communication reliability and performance.

## Hands-on Tutorial: Advanced `rclpy` Features (Parameters and Timers)

In previous chapters, we've seen basic node creation and communication. Here, we'll explore two more essential `rclpy` features: **parameters** for dynamic configuration and **timers** for periodic task execution.

### Step 1: Create a Parameterized Node

Inside `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/` (from Chapter 2), create `parameter_node.py` with the following content:

```python
# parameter_node.py

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

class ParameterNode(Node):

    def __init__(self):
        super().__init__('parameter_node')

        # Declare a parameter with a default value
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('publish_frequency', 1.0) # Frequency in Hz

        # Register a callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Get initial parameter value
        initial_param = self.get_parameter('my_parameter').get_parameter_value().string_value
        initial_freq = self.get_parameter('publish_frequency').get_parameter_value().double_value
        self.get_logger().info(f'Initial parameter: {initial_param}')
        self.get_logger().info(f'Initial frequency: {initial_freq} Hz')

        # Create a timer that uses the frequency parameter
        self.timer = self.create_timer(1.0 / initial_freq, self.timer_callback)
        self.get_logger().info('Parameter node started!')

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'my_parameter':
                self.get_logger().info(f'Parameter 'my_parameter' changed to: {param.value.string_value}')
            elif param.name == 'publish_frequency':
                new_freq = param.value.double_value
                self.get_logger().info(f'Parameter 'publish_frequency' changed to: {new_freq} Hz')
                # Update timer period if frequency changes
                self.timer.timer_period_ns = rclpy.duration.Duration(seconds=1.0/new_freq).nanoseconds
        return SetParametersResult(successful=True)

    def timer_callback(self):
        current_param = self.get_parameter('my_parameter').get_parameter_value().string_value
        self.get_logger().info(f'Timer callback: Current parameter value is {current_param}')

def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()
    rclpy.spin(parameter_node)
    parameter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Make Node Executable and Rebuild `my_ros2_pkg`

Edit `~/ros2_ws/src/my_ros2_pkg/setup.py` again. Add an entry point for your new node:

```python
# setup.py (excerpt)

    entry_points={
        'console_scripts': [
            # ... existing entry points ...
            'parameter_node = my_ros2_pkg.parameter_node:main',
        ],
    },

```

Rebuild only `my_ros2_pkg`:

```bash
cd ~/ros2_ws
colcon build --packages-select my_ros2_pkg
```

### Step 3: Source and Run the Node

Source your workspace:

```bash
source install/setup.bash
```

Run the parameter node in one terminal:

```bash
ros2 run my_ros2_pkg parameter_node
```

### Step 4: Dynamically Change Parameters

Open a second terminal (and source!). You can inspect parameters:

```bash
ros2 param list
ros2 param get /parameter_node my_parameter
```

Now, change a parameter:

```bash
ros2 param set /parameter_node my_parameter 'new_value'
ros2 param set /parameter_node publish_frequency 0.5
```

You should observe the `parameter_node` logging the parameter changes and its timer callback frequency adjusting. This demonstrates dynamic reconfiguration of nodes at runtime.

## Summary

`rclpy` empowers Python developers to fully leverage the ROS 2 ecosystem for building sophisticated robotic applications. Its intuitive API, combined with powerful features like parameters for dynamic configuration and timers for periodic tasks, facilitates rapid development and flexible system design. Mastering `rclpy` is crucial for creating intelligent and responsive robot behaviors.

---
