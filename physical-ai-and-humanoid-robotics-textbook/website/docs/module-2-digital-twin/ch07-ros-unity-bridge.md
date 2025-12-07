---
title: Chapter 7 - ROS-Unity Bridge Communication
description: Learn how to establish bidirectional communication between ROS 2 and Unity using the ROS-TCP-Connector for humanoid robot digital twins.
sidebar_position: 19
---

# Chapter 7 - ROS-Unity Bridge Communication

Establishing robust communication between ROS 2 and Unity is essential for creating effective digital twins of humanoid robots. The ROS-TCP-Connector provides a bridge that allows bidirectional data exchange between the two environments. This chapter covers setting up and using this bridge for humanoid robot applications.

## 7.1 Introduction to ROS-Unity Communication

The ROS-Unity bridge enables:
- Publishing sensor data from Unity to ROS 2 topics
- Subscribing to ROS 2 topics to control Unity agents
- Calling ROS 2 services from Unity
- Using ROS 2 actions with Unity
- Synchronizing simulation time between environments

## 7.2 Setting up the ROS-TCP-Connector

### 7.2.1 Installing ROS-TCP-Connector in Unity

1. Download the ROS-TCP-Connector package from the Unity Asset Store or GitHub
2. Import it into your Unity project
3. The package includes:
   - ROS Communication Manager
   - Message types for common ROS messages
   - Examples and documentation

### 7.2.2 Installing Python ROS Client

Install the Python client for ROS 2:

```bash
pip install ros2-unity-bridge
```

Or install from source:

```bash
git clone https://github.com/Unity-Technologies/ROS-TCP-Connector.git
cd ROS-TCP-Connector/python
pip install -e .
```

## 7.3 Basic ROS-TCP-Connector Setup

### 7.3.1 Unity Side Configuration

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class ROSUnityBridge : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    private RosConnection ros;

    void Start()
    {
        // Initialize ROS connection
        ros = RosConnection.GetOrCreateInstance();
        ros.RegisteredUri = new System.Uri($"tcp://{rosIPAddress}:{rosPort}");

        // Start ROS connection
        ros.Initialize();
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.Disconnect();
        }
    }
}
```

### 7.3.2 Python ROS Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import socket
import json

class ROSUnityBridgeNode(Node):
    def __init__(self):
        super().__init__('ros_unity_bridge')

        # Create publishers and subscribers
        self.joint_state_publisher = self.create_publisher(JointState, '/unity_joint_states', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/unity_cmd_vel', 10)
        self.unity_command_publisher = self.create_publisher(String, '/unity_commands', 10)

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/robot/joint_states',
            self.joint_state_callback,
            10
        )

        # Setup TCP server for Unity connection
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind(('localhost', 10000))
        self.tcp_server.listen(1)

        self.unity_client = None
        self.get_logger().info('ROS-Unity Bridge initialized')

    def joint_state_callback(self, msg):
        # Forward joint states to Unity
        if self.unity_client:
            joint_data = {
                'type': 'joint_states',
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort)
            }
            self.send_to_unity(json.dumps(joint_data))

    def send_to_unity(self, data):
        try:
            if self.unity_client:
                self.unity_client.send(data.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f'Error sending to Unity: {e}')

def main(args=None):
    rclpy.init(args=args)
    bridge_node = ROSUnityBridgeNode()

    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge_node.tcp_server.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7.4 Publishing Sensor Data from Unity to ROS 2

### 7.4.1 Camera Data Publisher

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections;

public class CameraDataPublisher : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera unityCamera;
    public string rosTopic = "/unity_camera/image_raw";
    public float publishRate = 30f; // Hz

    private RosConnection ros;
    private RenderTexture renderTexture;
    private Texture2D tempTexture;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        publishInterval = 1f / publishRate;

        // Create render texture for camera
        renderTexture = new RenderTexture(640, 480, 24);
        unityCamera.targetTexture = renderTexture;

        tempTexture = new Texture2D(640, 480, TextureFormat.RGB24, false);
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishCameraData();
            lastPublishTime = Time.time;
        }
    }

    void PublishCameraData()
    {
        // Copy render texture to regular texture
        RenderTexture.active = renderTexture;
        tempTexture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tempTexture.Apply();

        // Convert to byte array
        byte[] imageData = tempTexture.EncodeToPNG();

        // Create ROS Image message
        var imageMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.CompressedImageMsg
        {
            format = "png",
            data = imageData
        };

        var sensorMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.ImageMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = "unity_camera"
            },
            height = (uint)tempTexture.height,
            width = (uint)tempTexture.width,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(tempTexture.width * 3),
            data = tempTexture.GetRawTextureData<byte>().ToArray()
        };

        // Publish to ROS topic
        ros.Publish(rosTopic, sensorMsg);
    }
}
```

### 7.4.2 IMU Data Publisher

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class IMUDataPublisher : MonoBehaviour
{
    [Header("IMU Settings")]
    public string rosTopic = "/unity_imu/data";
    public float publishRate = 100f; // Hz
    public float noiseLevel = 0.01f;

    private RosConnection ros;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        publishInterval = 1f / publishRate;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishIMUData();
            lastPublishTime = Time.time;
        }
    }

    void PublishIMUData()
    {
        // Get orientation from transform
        Quaternion orientation = transform.rotation;

        // Get angular velocity (simulate or get from rigidbody)
        Vector3 angularVelocity = GetSimulatedAngularVelocity();

        // Get linear acceleration
        Vector3 linearAcceleration = GetSimulatedLinearAcceleration();

        // Create IMU message with noise
        var imuMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.ImuMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = "unity_imu"
            },
            orientation = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.QuaternionMsg
            {
                x = orientation.x,
                y = orientation.y,
                z = orientation.z,
                w = orientation.w
            },
            angular_velocity = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
            {
                x = angularVelocity.x + Random.Range(-noiseLevel, noiseLevel),
                y = angularVelocity.y + Random.Range(-noiseLevel, noiseLevel),
                z = angularVelocity.z + Random.Range(-noiseLevel, noiseLevel)
            },
            linear_acceleration = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
            {
                x = linearAcceleration.x + Random.Range(-noiseLevel, noiseLevel),
                y = linearAcceleration.y + Random.Range(-noiseLevel, noiseLevel),
                z = linearAcceleration.z + Random.Range(-noiseLevel, noiseLevel)
            }
        };

        ros.Publish(rosTopic, imuMsg);
    }

    Vector3 GetSimulatedAngularVelocity()
    {
        // Simulate angular velocity based on rotation changes
        // In a real implementation, this might come from a Rigidbody
        return new Vector3(
            Random.Range(-0.1f, 0.1f),
            Random.Range(-0.1f, 0.1f),
            Random.Range(-0.1f, 0.1f)
        );
    }

    Vector3 GetSimulatedLinearAcceleration()
    {
        // Simulate linear acceleration
        return new Vector3(
            Random.Range(-0.5f, 0.5f),
            Random.Range(-0.5f, 0.5f),
            Random.Range(-9.8f, -9.3f) // Gravity component
        );
    }
}
```

## 7.5 Subscribing to ROS 2 Topics in Unity

### 7.5.1 Joint State Subscriber

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class JointStateSubscriber : MonoBehaviour
{
    [Header("Joint Control")]
    public string rosTopic = "/robot/joint_commands";
    public HingeJoint[] joints;
    public string[] jointNames; // Must match ROS joint names

    private RosConnection ros;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();

        // Subscribe to joint state topic
        ros.Subscribe<JointStateMsg>(rosTopic, OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Update joints based on received commands
        for (int i = 0; i < jointNames.Length; i++)
        {
            string jointName = jointNames[i];

            // Find the corresponding position in the ROS message
            for (int j = 0; j < jointState.name.Length; j++)
            {
                if (jointState.name[j] == jointName)
                {
                    if (j < jointState.position.Length)
                    {
                        ApplyJointCommand(i, jointState.position[j]);
                    }
                    break;
                }
            }
        }
    }

    void ApplyJointCommand(int jointIndex, double targetPosition)
    {
        if (jointIndex < joints.Length)
        {
            HingeJoint joint = joints[jointIndex];

            // Set target position for the joint
            // Note: This is a simplified approach; real implementation
            // would use PID controllers or joint motors
            JointMotor motor = joint.motor;
            motor.targetVelocity = (float)(targetPosition - joint.angle) * 100f; // Simple PD control
            motor.force = 100f;
            joint.motor = motor;
            joint.useMotor = true;
        }
    }
}
```

### 7.5.2 Command Subscriber for Humanoid Robot

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class HumanoidCommandSubscriber : MonoBehaviour
{
    [Header("Command Settings")]
    public string cmdVelTopic = "/unity_cmd_vel";
    public string poseTopic = "/unity_pose_cmd";
    public float moveSpeed = 2.0f;
    public float rotationSpeed = 50.0f;

    private RosConnection ros;
    private Rigidbody rb;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        rb = GetComponent<Rigidbody>();

        // Subscribe to command topics
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);
        ros.Subscribe<PoseMsg>(poseTopic, OnPoseReceived);
    }

    void OnCmdVelReceived(TwistMsg cmdVel)
    {
        // Apply linear and angular velocities
        Vector3 linearVelocity = new Vector3(
            (float)cmdVel.linear.x,
            (float)cmdVel.linear.y,
            (float)cmdVel.linear.z
        );

        Vector3 angularVelocity = new Vector3(
            (float)cmdVel.angular.x,
            (float)cmdVel.angular.y,
            (float)cmdVel.angular.z
        );

        // Apply to rigidbody or character controller
        if (rb != null)
        {
            rb.velocity = linearVelocity * moveSpeed;
            rb.angularVelocity = angularVelocity * rotationSpeed;
        }
    }

    void OnPoseReceived(PoseMsg pose)
    {
        // Set position and orientation
        transform.position = new Vector3(
            (float)pose.position.x,
            (float)pose.position.y,
            (float)pose.position.z
        );

        transform.rotation = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.y,
            (float)pose.orientation.z,
            (float)pose.orientation.w
        );
    }
}
```

## 7.6 ROS Actions and Services Integration

### 7.6.1 Using ROS Actions in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Actionlib;

public class UnityActionClient : MonoBehaviour
{
    [Header("Action Settings")]
    public string actionName = "/unity_navigation_action";
    public string actionType = "move_base_msgs/MoveBaseAction";

    private RosConnection ros;
    private string actionGoalId;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
    }

    public void SendNavigationGoal(Vector3 targetPosition)
    {
        // Create and send action goal
        var goal = new Unity.Robotics.ROSTCPConnector.MessageTypes.MoveBase_msgs.MoveBaseActionGoalMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = "map"
            },
            goal_id = new Unity.Robotics.ROSTCPConnector.MessageTypes.Actionlib.GoalIDMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                id = "nav_goal_" + Time.time
            },
            goal = new Unity.Robotics.ROSTCPConnector.MessageTypes.MoveBase_msgs.MoveBaseGoalMsg
            {
                target_pose = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.PoseStampedMsg
                {
                    header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
                    {
                        stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                        {
                            sec = (int)Time.time,
                            nanosec = (uint)((Time.time % 1) * 1000000000)
                        },
                        frame_id = "map"
                    },
                    pose = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.PoseMsg
                    {
                        position = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.PointMsg
                        {
                            x = targetPosition.x,
                            y = targetPosition.y,
                            z = targetPosition.z
                        },
                        orientation = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.QuaternionMsg
                        {
                            x = 0, y = 0, z = 0, w = 1 // Default orientation
                        }
                    }
                }
            }
        };

        ros.Send(actionName + "/goal", goal);
    }

    public void CancelGoal()
    {
        var cancelMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Actionlib.GoalIDMsg
        {
            stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
            {
                sec = (int)Time.time,
                nanosec = (uint)((Time.time % 1) * 1000000000)
            },
            id = actionGoalId
        };

        ros.Send(actionName + "/cancel", cancelMsg);
    }
}
```

## 7.7 Advanced ROS-Unity Integration Patterns

### 7.7.1 Time Synchronization

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;

public class TimeSynchronizer : MonoBehaviour
{
    [Header("Time Settings")]
    public bool useSimTime = true;
    public float timeScale = 1.0f;

    private RosConnection ros;
    private double simulationTimeOffset;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();

        if (useSimTime)
        {
            ros.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg>(
                "/clock", OnClockReceived);
        }
    }

    void OnClockReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg clock)
    {
        // Synchronize Unity time with ROS time
        double rosTime = clock.sec + clock.nanosec / 1000000000.0;
        double unityTime = Time.timeAsDouble;

        simulationTimeOffset = rosTime - unityTime;
    }

    public double GetROSTime()
    {
        return Time.timeAsDouble + simulationTimeOffset;
    }
}
```

### 7.7.2 Transform Broadcasting

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Tf2;

public class TransformBroadcaster : MonoBehaviour
{
    [Header("Transform Settings")]
    public string frameId = "unity_robot_base";
    public string parentFrameId = "world";
    public float publishRate = 50f;

    private RosConnection ros;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        publishInterval = 1f / publishRate;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            BroadcastTransform();
            lastPublishTime = Time.time;
        }
    }

    void BroadcastTransform()
    {
        var transformMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Tf2.TFMessageMsg
        {
            transforms = new[]
            {
                new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.TransformStampedMsg
                {
                    header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
                    {
                        stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.TimeMsg
                        {
                            sec = (int)Time.time,
                            nanosec = (uint)((Time.time % 1) * 1000000000)
                        },
                        frame_id = parentFrameId
                    },
                    child_frame_id = frameId,
                    transform = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.TransformMsg
                    {
                        translation = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.Vector3Msg
                        {
                            x = transform.position.x,
                            y = transform.position.y,
                            z = transform.position.z
                        },
                        rotation = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.QuaternionMsg
                        {
                            x = transform.rotation.x,
                            y = transform.rotation.y,
                            z = transform.rotation.z,
                            w = transform.rotation.w
                        }
                    }
                }
            }
        };

        ros.Publish("/tf", transformMsg);
    }
}
```

## 7.8 Best Practices for ROS-Unity Communication

### 7.8.1 Performance Optimization
- Use appropriate publish rates for different data types
- Compress large data like images when possible
- Use efficient data structures and minimize allocations
- Consider using ROS 2 Quality of Service (QoS) settings

### 7.8.2 Error Handling
- Implement connection status monitoring
- Add retry mechanisms for failed connections
- Handle message deserialization errors gracefully
- Log communication issues for debugging

### 7.8.3 Security Considerations
- Use secure network connections when possible
- Validate incoming messages from ROS
- Implement authentication if required
- Consider network segmentation for sensitive data

## Summary

The ROS-Unity bridge enables powerful bidirectional communication between ROS 2 and Unity environments, making it possible to create sophisticated digital twins for humanoid robots. By properly implementing publishers, subscribers, and advanced features like actions and time synchronization, you can create seamless integration between your simulation and control systems. In the next chapter, we will explore collision detection, contact dynamics, and validation techniques for digital twin environments.