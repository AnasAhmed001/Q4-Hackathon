# Chapter 1: ROS 2 Architecture and DDS Middleware

## Learning Objectives
*   Understand the fundamental shift from ROS 1 to ROS 2 architecture.
*   Explain the role of Data Distribution Service (DDS) as the middleware in ROS 2.
*   Identify key architectural components: nodes, topics, services, actions, and parameters.
*   Describe the benefits of ROS 2's distributed, fault-tolerant, and real-time capable design.

## Prerequisites
*   Basic Linux command line familiarity.
*   Intermediate Python proficiency.

## Theory & Concepts

ROS 2 (Robot Operating System 2) represents a significant evolution from its predecessor, ROS 1, designed to address the challenges of modern robotics, including multi-robot systems, real-time control, and security. The most pivotal change in ROS 2 is its adoption of **Data Distribution Service (DDS)** as its core communication middleware.

### The Shift from ROS 1 to ROS 2

ROS 1 relied on a central **ROS Master** for node registration and lookup. While effective for single-robot setups, this centralized architecture introduced single points of failure and limited scalability for multi-robot and highly distributed applications. ROS 2, in contrast, adopts a **decentralized** architecture.

### Data Distribution Service (DDS)

DDS is an open international standard (IEEE 1516) for real-time, peer-to-peer, data-centric communication. It enables anonymous, decoupled, and quality-of-service (QoS) aware data exchange. In ROS 2, DDS handles:

*   **Discovery**: Nodes automatically discover each other without a central server.
*   **Data Transfer**: Efficient and reliable exchange of messages.
*   **Quality of Service (QoS)**: Configurable parameters (e.g., reliability, history, deadline) to meet specific application requirements for real-time performance and data integrity.

This decentralized nature makes ROS 2 inherently more robust, scalable, and suitable for a wider range of robotic applications, from industrial automation to autonomous vehicles.

### Key Architectural Components

ROS 2 builds upon a set of fundamental concepts:

*   **Nodes**: Independent executable processes that perform specific computations (e.g., a camera driver node, a navigation node). Nodes communicate with each other to achieve complex tasks.
*   **Topics**: An asynchronous, publish-subscribe communication mechanism for streaming data. One node publishes messages to a topic, and other nodes subscribe to it to receive those messages.
*   **Services**: A synchronous, request-reply communication mechanism used for individual, one-time requests (e.g., requesting the robot to perform a specific action and waiting for its completion status).
*   **Actions**: An extension of services for long-running, pre-emptable tasks. Actions provide continuous feedback, allowing a client to monitor the progress of a goal and even cancel it if necessary.
*   **Parameters**: Dynamic configuration values that nodes can expose. These allow users to adjust a node's behavior without recompiling the code.

## Hands-on Tutorial: Exploring ROS 2 DDS Configuration

In this tutorial, we will explore how DDS is configured in ROS 2. While DDS works largely transparently, understanding its configuration options is crucial for optimizing performance and reliability in complex setups.

### Step 1: Check your DDS implementation

ROS 2 can use different DDS implementations (e.g., Fast RTPS, Cyclone DDS, RTI Connext). You can check which one is active by examining the `RMW_IMPLEMENTATION` environment variable.

```bash
# Check current RMW implementation
echo $RMW_IMPLEMENTATION
```

If it's not set, ROS 2 typically defaults to Fast RTPS or Cyclone DDS depending on your installation. You can explicitly set it, for example:

```bash
# Set RMW implementation to Cyclone DDS (if installed)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### Step 2: Understanding Quality of Service (QoS) Profiles

DDS offers various QoS profiles that define how data is exchanged. These are critical for managing reliability, latency, and resource usage. Common QoS policies include:

*   **Reliability**: `Reliable` (guaranteed delivery) vs. `Best Effort` (data may be lost).
*   **History**: `Keep Last` (only store the N most recent samples) vs. `Keep All` (store all samples until read).
*   **Durability**: `Transient Local` (new subscribers get last published message) vs. `Volatile` (only live data).
*   **Liveliness**: How often a publisher announces its presence.

These QoS settings can be configured for publishers and subscribers, often within the code or via ROS 2 launch files. For instance, to ensure all messages are received (even if the network is flaky) and new subscribers get the most recent message, you might use `Reliable` and `Transient Local` settings.

### Step 3: Example: Configuring QoS in `rclpy` (Conceptual)

While the actual code implementation will be covered in later chapters, conceptually, when creating a publisher or subscriber in `rclpy`, you would specify QoS profiles. Here's a simplified example of how you might define a reliable publisher:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

class MyPublisher(Node):
    def __init__(self):
        super().__init__('my_publisher')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10, # Keep last 10 messages
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.publisher_ = self.create_publisher(String, 'topic', qos_profile)
        self.timer = self.create_timer(1.0, self.publish_message)
        self.get_logger().info('Publisher created with RELIABLE QoS')

    def publish_message(self):
        msg = String()
        msg.data = 'Hello from ROS 2 DDS!'
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This example demonstrates how QoS profiles are applied to a publisher to ensure reliable delivery and data availability for new subscribers. These concepts are foundational for building robust ROS 2 systems.

## Summary

ROS 2's architecture, built upon the DDS middleware, provides a decentralized, fault-tolerant, and highly configurable communication framework. This shift from ROS 1 enables greater scalability and real-time capabilities for complex robotic applications. Understanding nodes, topics, services, actions, and QoS profiles is crucial for effective ROS 2 development.

---
