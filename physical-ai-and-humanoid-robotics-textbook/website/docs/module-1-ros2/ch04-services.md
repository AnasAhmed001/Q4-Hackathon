# Chapter 4: ROS 2 Services (Synchronous Request/Reply)

## Learning Objectives
*   Understand the concept of ROS 2 services for synchronous communication.
*   Differentiate services from topics and when to use each.
*   Learn how to define custom service types.
*   Implement a simple service server and client node pair using `rclpy`.
*   Use ROS 2 command-line tools to introspect service communication.

## Prerequisites
*   Familiarity with ROS 2 nodes and custom messages (Chapter 3).
*   Intermediate Python proficiency.

## Theory & Concepts

While topics are ideal for continuous, asynchronous data streams, ROS 2 **services** are used for **synchronous request-reply communication**. This means a client node sends a request to a service server node and then blocks (waits) until it receives a response. Services are suitable for single-shot interactions where an immediate response is expected, such as triggering a specific action, querying a parameter, or performing a one-time computation.

### How Services Work

1.  **Service Server**: A node that provides a specific service. It registers the service with the ROS 2 graph and listens for incoming requests.
2.  **Service Client**: A node that requests a service from a service server. It sends a request and waits for the server's response.
3.  **Service Type**: Defines the structure of the request and response messages. Similar to topics, services use strongly typed messages defined in `.srv` files.

Services are well-suited for commands that have a clear start and end, and where the client needs to know the outcome before proceeding. For example, a `gripper_controller_node` might offer a `close_gripper` service that takes no arguments and returns a boolean indicating success or failure.

#### Defining a Custom Service

A custom service file (e.g., `AddTwoInts.srv`) defines both the request and response sections, separated by `---`:

```
# AddTwoInts.srv

int64 a
int64 b
---
int64 sum
```

Here, `a` and `b` are part of the request, and `sum` is part of the response.

## Hands-on Tutorial: Implementing a Custom Service Server and Client

In this tutorial, we will create a custom service, then implement a service server node that provides this service and a client node that requests it.

### Step 1: Define the Custom Service

Inside your `~/ros2_ws/src/my_interfaces/srv` directory (create the `srv` directory if it doesn't exist), create a new file named `AddTwoInts.srv` with the content shown above:

```
# AddTwoInts.srv

int64 a
int64 b
---
int64 sum
```

### Step 2: Configure `package.xml` and `setup.py` (for `my_interfaces`)

Ensure your `~/ros2_ws/src/my_interfaces/package.xml` has the necessary build and run dependencies for `rosidl_default_generators` and `rosidl_default_runtime`. If you created `my_interfaces` in Chapter 3, these should already be there.

Edit `~/ros2_ws/src/my_interfaces/setup.py`. Add the following to `data_files` to ensure the service definition is installed:

```python
# setup.py (excerpt)

from setuptools import setup

package_name = 'my_interfaces'

setup(
    # ... other setup parameters ...
    packages=[package_name],
    data_files=[
        # ... existing data_files ...
        # Add this line to install your service definition
        ('share/' + package_name + '/srv', glob(os.path.join('srv', '*.srv'))),
    ],
    # ... rest of setup parameters ...
)

```

### Step 3: Build the Custom Service Package

Navigate back to your workspace root (`~/ros2_ws`) and rebuild your `my_interfaces` package. This will generate the necessary Python classes for your `AddTwoInts` service.

```bash
# Build the workspace
cd ~/ros2_ws
colcon build --packages-select my_interfaces
```

### Step 4: Source Your Workspace

After building, you need to source your workspace:

```bash
# Source the ROS 2 environment and your workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Step 5: Write the Service Server Node

Inside `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/` (from Chapter 2), create `add_two_ints_server.py`:

```python
# add_two_ints_server.py

import rclpy
from rclpy.node import Node
from my_interfaces.srv import AddTwoInts # Import your custom service type

class AddTwoIntsService(Node):

    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
        self.get_logger().info('Add Two Ints Service Server started!')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request: a={request.a}, b={request.b}')
        self.get_logger().info(f'Sending response: sum={response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    add_two_ints_service = AddTwoIntsService()
    rclpy.spin(add_two_ints_service)
    add_two_ints_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Write the Service Client Node

Inside the same directory `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/`, create `add_two_ints_client.py`:

```python
# add_two_ints_client.py

import sys
import rclpy
from rclpy.node import Node
from my_interfaces.srv import AddTwoInts # Import your custom service type

class AddTwoIntsClient(Node):

    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()
        self.get_logger().info('Add Two Ints Service Client started!')

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 3:
        print('Usage: ros2 run my_ros2_pkg add_two_ints_client <arg1> <arg2>')
        sys.exit(1)

    add_two_ints_client = AddTwoIntsClient()
    response = add_two_ints_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    add_two_ints_client.get_logger().info(
        f'Result of add_two_ints: for {int(sys.argv[1])} + {int(sys.argv[2])} = {response.sum}')

    add_two_ints_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 7: Make Nodes Executable and Rebuild `my_ros2_pkg`

Edit `~/ros2_ws/src/my_ros2_pkg/setup.py` again. Add entry points for both your new nodes:

```python
# setup.py (excerpt)

    entry_points={
        'console_scripts': [
            'minimal_node = my_ros2_pkg.minimal_node:main',
            'sensor_publisher = my_ros2_pkg.sensor_publisher:main',
            'sensor_subscriber = my_ros2_pkg.sensor_subscriber:main',
            'add_two_ints_server = my_ros2_pkg.add_two_ints_server:main',
            'add_two_ints_client = my_ros2_pkg.add_two_ints_client:main',
        ],
    },

```

Rebuild only `my_ros2_pkg`:

```bash
cd ~/ros2_ws
colcon build --packages-select my_ros2_pkg
```

### Step 8: Source and Run

Source your workspace again:

```bash
source install/setup.bash
```

Open two terminals. In the first, run the service server:

```bash
ros2 run my_ros2_pkg add_two_ints_server
```

In the second terminal (after sourcing!), run the service client with two integer arguments:

```bash
ros2 run my_ros2_pkg add_two_ints_client 5 7
```

You should see the server processing the request and the client receiving the sum in response.

### Step 9: Introspect Services

Open a third terminal (and source!) to inspect the service:

```bash
# List active services
ros2 service list

# Get info about your service
ros2 service info /add_two_ints

# Call the service from the command line
ros2 service call /add_two_ints my_interfaces/srv/AddTwoInts "{a: 10, b: 20}"
```

## Summary

ROS 2 services provide a robust mechanism for synchronous, request-reply communication, essential for triggering specific actions and querying information. By defining custom service types and implementing server-client pairs using `rclpy`, developers can create powerful, interactive robotic functionalities. Service introspection tools further aid in understanding and debugging these interactions within the ROS 2 computational graph.

---
