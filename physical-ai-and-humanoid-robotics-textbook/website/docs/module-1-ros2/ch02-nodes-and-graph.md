# Chapter 2: ROS 2 Nodes and the Computational Graph

## Learning Objectives
*   Define what a ROS 2 node is and its purpose.
*   Understand the concept of the ROS 2 computational graph.
*   Identify different ways nodes can be created and managed.
*   Explain how nodes interact within a distributed system.

## Prerequisites
*   Familiarity with basic ROS 2 architecture concepts (Chapter 1).
*   Intermediate Python proficiency.

## Theory & Concepts

In ROS 2, the fundamental unit of computation is the **node**. A node is an executable process that performs a specific task, such as reading data from a sensor, controlling a motor, or performing complex computations like path planning. By breaking down the robot's software into many small, modular nodes, ROS 2 enables developers to create complex systems more easily, manage dependencies, and encourage code reuse.

### What is a ROS 2 Node?

Each node in a ROS 2 system is an independent process that can communicate with other nodes. This modularity means that if one node crashes, it ideally does not bring down the entire robotic system. Nodes are typically written to do one thing well.

Examples of nodes:
*   A `camera_driver_node` that publishes camera images.
*   A `motor_controller_node` that subscribes to velocity commands and sends them to robot motors.
*   A `navigation_planner_node` that takes sensor data and calculates a path.

### The ROS 2 Computational Graph

The **computational graph** is a network of all the ROS 2 nodes in a system and their connections. It's a dynamic map of how data and commands flow between different parts of your robot's software. Unlike ROS 1, which had a central ROS Master to manage this graph, ROS 2 uses its DDS middleware for decentralized discovery and communication, making the graph more resilient.

Key elements of the computational graph:
*   **Nodes**: The boxes in the graph, representing processes.
*   **Topics, Services, Actions**: The lines connecting the boxes, representing the communication channels.

Visualizing this graph is extremely helpful for understanding the runtime behavior of a complex robot system and for debugging communication issues.

## Hands-on Tutorial: Creating and Introspecting ROS 2 Nodes

In this tutorial, we will create a simple ROS 2 node using `rclpy` (the Python client library) and use ROS 2 command-line tools to introspect (examine) the active nodes and their connections.

### Step 1: Create a ROS 2 Package

First, navigate to your ROS 2 workspace `src` directory (e.g., `~/ros2_ws/src`) and create a new Python package. If you don't have a workspace, please refer to Chapter 10 for setup instructions.

```bash
# Navigate to your ROS 2 workspace src directory
cd ~/ros2_ws/src

# Create a new Python package named 'my_ros2_pkg'
ros2 pkg create --build-type ament_python my_ros2_pkg
```

This command creates a directory structure for your new package, including `setup.py` and `package.xml`.

### Step 2: Write a Simple ROS 2 Node

Inside `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/`, create a new Python file named `minimal_node.py` with the following content:

```python
# minimal_node.py

import rclpy
from rclpy.node import Node

class MinimalNode(Node):

    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node started!')

    # Optionally, add a method that gets called regularly, like a timer
    # def timer_callback(self):
    #    self.get_logger().info('Hello from timer!')

def main(args=None):
    rclpy.init(args=args)
    minimal_node = MinimalNode()
    rclpy.spin(minimal_node) # Keep node alive until Ctrl+C
    minimal_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Make the Node Executable

Edit `~/ros2_ws/src/my_ros2_pkg/setup.py` to add an entry point for your node. Find the `entry_points` dictionary and add the following:

```python
# setup.py (excerpt)

    entry_points={
        'console_scripts': [
            'minimal_node = my_ros2_pkg.minimal_node:main',
        ],
    },

```

### Step 4: Build Your Package

Navigate back to your workspace root (`~/ros2_ws`) and build your package:

```bash
# Build the workspace
cd ~/ros2_ws
colcon build --packages-select my_ros2_pkg
```

### Step 5: Source Your Workspace

After building, you need to source your workspace to make your new node available in your shell:

```bash
# Source the ROS 2 environment and your workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Step 6: Run and Introspect the Node

Now, run your minimal node in one terminal:

```bash
# Run the node
ros2 run my_ros2_pkg minimal_node
```

You should see `[INFO] Minimal node started!` in the terminal.

Open a **second terminal** (and remember to source your workspace again in the new terminal!) to introspect the running node:

```bash
# List active ROS 2 nodes
ros2 node list
```

You should see `/minimal_node` in the output, confirming your node is running.

To see more details about your node:

```bash
# Get info about the minimal_node
ros2 node info /minimal_node
```

This will show you its subscribed topics, published topics, services, and actions (which will be empty for this minimal node).

### Step 7: Visualize the Computational Graph (Optional)

For a graphical representation of the computational graph, you can use `rqt_graph`. This tool is very useful for debugging.

```bash
# Install rqt_graph if you haven't already
sudo apt install ros-humble-rqt-graph

# Run rqt_graph
rqt_graph
```

`rqt_graph` will display a visual representation of your `/minimal_node` and any other active ROS 2 entities.

## Summary

ROS 2 nodes are the modular building blocks of any robotic application, allowing for organized and distributed computation. The computational graph represents the dynamic connections between these nodes, facilitating complex data flow. Using `rclpy` to create nodes and ROS 2 command-line tools for introspection, developers can effectively manage and debug their robotic systems.

---