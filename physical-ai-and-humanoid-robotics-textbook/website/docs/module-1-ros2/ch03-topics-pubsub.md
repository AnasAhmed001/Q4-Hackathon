# Chapter 3: ROS 2 Topics (Publisher/Subscriber Communication)

## Learning Objectives
*   Understand the concept of ROS 2 topics for asynchronous data streaming.
*   Differentiate between publishers and subscribers.
*   Learn how to define custom message types.
*   Implement a simple publisher-subscriber node pair using `rclpy`.
*   Use ROS 2 command-line tools to introspect topic communication.

## Prerequisites
*   Familiarity with ROS 2 nodes and computational graph (Chapter 2).
*   Intermediate Python proficiency.

## Theory & Concepts

Topics are the most common way for nodes in a ROS 2 system to exchange data. They implement a **publisher-subscriber** communication model, which is asynchronous and one-to-many. This means that a node can publish messages to a topic without knowing if any other node is subscribed, and multiple nodes can subscribe to the same topic to receive all messages published to it.

### How Topics Work

1.  **Publisher**: A node that creates and sends messages to a named topic.
2.  **Subscriber**: A node that receives messages from a named topic.
3.  **Topic**: A named channel through which messages flow. Messages are strongly typed, ensuring data consistency.

This decoupled nature allows for flexible and scalable robotic systems. For example, a camera driver node can publish image data to an `/image_raw` topic, and multiple nodes (e.g., an object detection node, a recording node, a display node) can subscribe to this single topic to process the same data independently.

### Message Types

Messages are the data structures transmitted over topics. ROS 2 provides a set of standard message types (e.g., `std_msgs`, `sensor_msgs`, `geometry_msgs`). However, you will often need to define **custom message types** for specific application data. Custom messages are defined using `.msg` files, which are then compiled into source code for different languages.

#### Defining a Custom Message

A custom message file (e.g., `MyCustomMessage.msg`) defines the fields and their types:

```
# MyCustomMessage.msg

int32 my_int_field
string my_string_field
float64[] my_float_array
```

This message would be defined within a ROS 2 package and then built to generate the corresponding Python/C++ message classes.

## Hands-on Tutorial: Implementing a Custom Publisher and Subscriber

In this tutorial, we will create a custom message, then implement a publisher node that sends messages of this type and a subscriber node that receives and prints them.

### Step 1: Create a Custom Message Package (if you don't have one)

If you haven't already, create a dedicated package for your custom messages. Navigate to your ROS 2 workspace `src` directory (e.g., `~/ros2_ws/src`) and create a package:

```bash
# Create a new Python package named 'my_interfaces' for custom messages
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_interfaces
```

### Step 2: Define the Custom Message

Inside `~/ros2_ws/src/my_interfaces/msg`, create a new file named `SensorData.msg` with the following content:

```
# SensorData.msg

int32 id
float32 temperature
string status
```

### Step 3: Configure `package.xml` and `setup.py`

Edit `~/ros2_ws/src/my_interfaces/package.xml` to add build and run dependencies for `rosidl_default_generators` and `rosidl_default_runtime`. Add these lines:

```xml
<!-- package.xml (excerpt) -->

  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
```

Edit `~/ros2_ws/src/my_interfaces/setup.py`. Add the following to `data_files` to ensure the message definition is installed:

```python
# setup.py (excerpt)

from setuptools import setup

package_name = 'my_interfaces'

setup(
    # ... other setup parameters ...
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add this line to install your message definition
        ('share/' + package_name + '/msg', glob(os.path.join('msg', '*.msg'))),
    ],
    # ... rest of setup parameters ...
)

```

And add imports at the top of `setup.py`:

```python
import os
from glob import glob
# ... rest of imports ...
```

### Step 4: Build the Custom Message Package

Navigate back to your workspace root (`~/ros2_ws`) and build your package. This will generate the necessary Python classes for your `SensorData` message.

```bash
# Build the workspace
cd ~/ros2_ws
colcon build --packages-select my_interfaces
```

### Step 5: Source Your Workspace

After building, you need to source your workspace to make your new message type available:

```bash
# Source the ROS 2 environment and your workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Step 6: Write the Publisher Node

Now, let's create the publisher. Inside `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/` (from Chapter 2), create `sensor_publisher.py`:

```python
# sensor_publisher.py

import rclpy
from rclpy.node import Node
from my_interfaces.msg import SensorData # Import your custom message

class SensorPublisher(Node):

    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher_ = self.create_publisher(SensorData, 'sensor_topic', 10)
        self.timer = self.create_timer(1.0, self.publish_sensor_data)
        self.i = 0
        self.get_logger().info('Sensor Publisher node started!')

    def publish_sensor_data(self):
        msg = SensorData()
        msg.id = self.i
        msg.temperature = float(25.0 + (self.i % 5))
        msg.status = f'Operational {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: id={msg.id}, temp={msg.temperature}, status={msg.status}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()
    rclpy.spin(sensor_publisher)
    sensor_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 7: Write the Subscriber Node

Inside the same directory `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/`, create `sensor_subscriber.py`:

```python
# sensor_subscriber.py

import rclpy
from rclpy.node import Node
from my_interfaces.msg import SensorData # Import your custom message

class SensorSubscriber(Node):

    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            SensorData, # Your custom message type
            'sensor_topic',
            self.listener_callback,
            10)
        self.subscription # prevent unused variable warning
        self.get_logger().info('Sensor Subscriber node started!')

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: id={msg.id}, temp={msg.temperature}, status={msg.status}')

def main(args=None):
    rclpy.init(args=args)
    sensor_subscriber = SensorSubscriber()
    rclpy.spin(sensor_subscriber)
    sensor_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 8: Make Nodes Executable and Rebuild `my_ros2_pkg`

Edit `~/ros2_ws/src/my_ros2_pkg/setup.py` again. Add entry points for both your new nodes. Also, add `my_interfaces` to `install_requires` and `ament_python` to `packages`:

```python
# setup.py (excerpt)

from setuptools import setup

package_name = 'my_ros2_pkg'

setup(
    # ... other setup parameters ...
    install_requires=['setuptools', 'my_interfaces'], # Add my_interfaces
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Examples for ROS 2 nodes',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'minimal_node = my_ros2_pkg.minimal_node:main',
            'sensor_publisher = my_ros2_pkg.sensor_publisher:main',
            'sensor_subscriber = my_ros2_pkg.sensor_subscriber:main',
        ],
    },
    packages=[package_name], # Ensure package_name is here
)

```

Now, rebuild only `my_ros2_pkg`:

```bash
cd ~/ros2_ws
colcon build --packages-select my_ros2_pkg
```

### Step 9: Source and Run

Source your workspace again:

```bash
source install/setup.bash
```

Open two terminals. In the first, run the publisher:

```bash
ros2 run my_ros2_pkg sensor_publisher
```

In the second terminal (after sourcing!), run the subscriber:

```bash
ros2 run my_ros2_pkg sensor_subscriber
```

You should see messages being published by the `sensor_publisher` and received by the `sensor_subscriber`, demonstrating successful topic communication with your custom message type.

### Step 10: Introspect Topics

Open a third terminal (and source!) to inspect the topic:

```bash
# List active topics
ros2 topic list

# Get info about your topic
ros2 topic info /sensor_topic

# Echo messages on your topic
ros2 topic echo /sensor_topic
```

## Summary

ROS 2 topics provide a powerful and flexible mechanism for asynchronous, many-to-many data exchange. By defining custom message types, developers can tailor data structures to their specific robotic applications. The publisher-subscriber model, combined with robust introspection tools, forms a cornerstone of ROS 2 communication, enabling complex robotic systems to share information efficiently.

---
