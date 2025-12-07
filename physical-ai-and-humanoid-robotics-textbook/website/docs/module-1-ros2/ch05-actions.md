# Chapter 5: ROS 2 Actions (Long-Running Tasks)

## Learning Objectives
*   Understand the purpose of ROS 2 actions for long-running, goal-oriented tasks.
*   Differentiate actions from topics and services.
*   Learn the structure of a custom action type.
*   Implement a simple action server and client node pair using `rclpy`.
*   Use ROS 2 command-line tools to introspect action communication.

## Prerequisites
*   Familiarity with ROS 2 nodes, custom messages, and services (Chapter 4).
*   Intermediate Python proficiency.

## Theory & Concepts

ROS 2 **actions** are designed for communication patterns that involve **long-running, goal-oriented tasks** that provide continuous feedback and allow for preemption. They combine the best aspects of topics (continuous feedback) and services (request-reply, goal-oriented) into a single, more sophisticated communication mechanism. Actions are particularly well-suited for tasks like robot navigation (e.g., "go to a goal"), manipulation (e.g., "pick up an object"), or complex sensor processing where the client needs to monitor progress and potentially cancel the operation.

### How Actions Work (Goal, Feedback, Result)

An action involves three primary components, each with its own message type:

1.  **Goal**: Sent by the **Action Client** to the **Action Server**, defining the task to be performed (e.g., target coordinates for navigation).
2.  **Feedback**: Sent by the Action Server to the Action Client, providing intermittent updates on the progress of the long-running task (e.g., current robot position during navigation).
3.  **Result**: Sent by the Action Server to the Action Client upon completion or failure of the task, indicating the final outcome (e.g., whether the robot reached the goal successfully).

This robust communication pattern allows for complex interactions where clients can initiate a task, monitor its progress, and even cancel it if conditions change, offering greater control than simple services.

#### Defining a Custom Action

A custom action file (e.g., `Fibonacci.action`) defines the goal, result, and feedback sections, separated by `---`:

```
# Fibonacci.action

# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] partial_sequence
```

Here, `order` is the goal, `sequence` is the final result, and `partial_sequence` is the continuous feedback.

## Hands-on Tutorial: Implementing a Custom Action Server and Client

In this tutorial, we will create a custom action, then implement an action server node that computes a Fibonacci sequence and an action client node that requests this action.

### Step 1: Define the Custom Action

Inside your `~/ros2_ws/src/my_interfaces/action` directory (create the `action` directory if it doesn't exist), create a new file named `Fibonacci.action` with the content shown above:

```
# Fibonacci.action

# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] partial_sequence
```

### Step 2: Configure `package.xml` and `setup.py` (for `my_interfaces`)

Ensure your `~/ros2_ws/src/my_interfaces/package.xml` has the necessary build and run dependencies for `rosidl_default_generators` and `rosidl_default_runtime`. Add an `exec_depend` for `rosidl_default_runtime` and `build_depend` for `rosidl_default_generators` if they aren't already present from previous chapters.

Edit `~/ros2_ws/src/my_interfaces/setup.py`. Add the following to `data_files` to ensure the action definition is installed:

```python
# setup.py (excerpt)

from setuptools import setup

package_name = 'my_interfaces'

setup(
    # ... other setup parameters ...
    packages=[package_name],
    data_files=[
        # ... existing data_files ...
        # Add this line to install your action definition
        ('share/' + package_name + '/action', glob(os.path.join('action', '*.action'))),
    ],
    # ... rest of setup parameters ...
)

```

### Step 3: Build the Custom Action Package

Navigate back to your workspace root (`~/ros2_ws`) and rebuild your `my_interfaces` package. This will generate the necessary Python classes for your `Fibonacci` action.

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

### Step 5: Write the Action Server Node

Inside `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/` (from Chapter 2), create `fibonacci_action_server.py`:

```python
# fibonacci_action_server.py

import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from my_interfaces.action import Fibonacci # Import your custom action type

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info('Fibonacci Action Server started!')

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info(f'Received goal request: {goal_request.order}')
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        """Callback that is executed when a goal is accepted."""
        self.get_logger().info('Goal accepted, executing...')
        goal_handle.execute()

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute a goal."""
        self.get_logger().info('Executing goal...')

        # Get the goal request
        order = goal_handle.request.order

        # Start generating Fibonacci sequence
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1]
            )
            self.get_logger().info(f'Feedback: {feedback_msg.partial_sequence}')
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1) # Simulate work

        goal_handle.succeed()

        # Populate result message
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)

if __name__ == '__main__':
    main()
```

### Step 6: Write the Action Client Node

Inside the same directory `~/ros2_ws/src/my_ros2_pkg/my_ros2_pkg/`, create `fibonacci_action_client.py`:

```python
# fibonacci_action_client.py

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_interfaces.action import Fibonacci # Import your custom action type

class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')
        self.get_logger().info('Fibonacci Action Client started!')

    def send_goal(self, order):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info(f'Sending goal: {order}')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.partial_sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(int(sys.argv[1])) if len(sys.argv) > 1 else action_client.send_goal(10) # Default order if not provided
    rclpy.spin(action_client)

if __name__ == '__main__':
    import sys
    main(sys.argv)
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
            'fibonacci_action_server = my_ros2_pkg.fibonacci_action_server:main',
            'fibonacci_action_client = my_ros2_pkg.fibonacci_action_client:main',
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

Open two terminals. In the first, run the action server:

```bash
ros2 run my_ros2_pkg fibonacci_action_server
```

In the second terminal (after sourcing!), run the action client, requesting a Fibonacci sequence of a certain order (e.g., 10):

```bash
ros2 run my_ros2_pkg fibonacci_action_client 10
```

You should see the server publishing feedback (partial sequences) and finally the result (complete sequence), while the client receives these updates.

### Step 9: Introspect Actions

Open a third terminal (and source!) to inspect the action:

```bash
# List active actions
ros2 action list

# Get info about your action
ros2 action info /fibonacci
```

## Summary

ROS 2 actions provide a sophisticated mechanism for managing long-running, goal-oriented tasks with continuous feedback and preemption capabilities. By leveraging custom action types and implementing server-client pairs using `rclpy`, developers can create robust and interactive robotic behaviors. Action introspection tools further aid in understanding and debugging these complex interactions within the ROS 2 ecosystem.

---
