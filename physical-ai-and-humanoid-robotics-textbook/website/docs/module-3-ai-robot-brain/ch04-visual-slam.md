---
title: Chapter 4 - Visual SLAM for Humanoid Robots
description: Learn how to implement visual SLAM systems for humanoid robots using Isaac Sim, Isaac ROS, and real-time mapping techniques.
sidebar_position: 26
---

# Chapter 4 - Visual SLAM for Humanoid Robots

Visual Simultaneous Localization and Mapping (SLAM) is a critical capability for humanoid robots, enabling them to understand and navigate their environment without prior knowledge. Visual SLAM combines visual perception with spatial mapping to create a representation of the environment while simultaneously determining the robot's position within it. This chapter explores the implementation of visual SLAM systems specifically tailored for humanoid robots using Isaac Sim for simulation and Isaac ROS for hardware acceleration.

## 4.1 Introduction to Visual SLAM

Visual SLAM is a technique that allows a robot to build a map of an unknown environment while simultaneously localizing itself within that map using visual sensors (cameras). For humanoid robots, visual SLAM provides:

- **Spatial Awareness**: Understanding the 3D structure of the environment
- **Self-Localization**: Knowing the robot's position and orientation
- **Navigation**: Planning paths through the environment
- **Obstacle Avoidance**: Detecting and avoiding obstacles in real-time

### 4.1.1 Visual SLAM vs. Other SLAM Approaches

- **Visual SLAM**: Uses cameras as primary sensors (monocular, stereo, RGB-D)
- **LIDAR SLAM**: Uses LIDAR sensors for precise distance measurements
- **Visual-Inertial SLAM**: Combines visual and IMU data for improved accuracy
- **Multi-Sensor SLAM**: Integrates multiple sensor types for robustness

### 4.1.2 Challenges for Humanoid Robots

Visual SLAM for humanoid robots faces unique challenges:
- **Dynamic Motion**: Humanoid robots have complex movement patterns
- **Height Variations**: Different viewing angles as the robot moves
- **Computational Constraints**: Limited processing power on humanoid platforms
- **Sensor Placement**: Cameras may be placed on moving parts (head, torso)

## 4.2 Visual SLAM Fundamentals

### 4.2.1 Key Components

Visual SLAM systems typically include:

1. **Feature Detection and Matching**: Identifying and tracking visual features
2. **Pose Estimation**: Calculating camera/robot pose from feature correspondences
3. **Mapping**: Building a representation of the environment
4. **Loop Closure**: Detecting revisited locations to correct drift
5. **Optimization**: Refining pose and map estimates over time

### 4.2.2 Common Visual SLAM Algorithms

- **ORB-SLAM**: Feature-based approach using ORB features
- **LSD-SLAM**: Direct method using image intensity
- **SVO**: Semi-direct visual odometry
- **DVO**: Dense visual odometry
- **RTAB-Map**: Appearance-based mapping and localization

## 4.3 Isaac ROS Visual SLAM Implementation

Isaac ROS provides hardware-accelerated visual SLAM capabilities optimized for NVIDIA platforms:

### 4.3.1 Isaac ROS Visual SLAM Node

```python
# Example Isaac ROS Visual SLAM implementation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

class IsaacROSVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_visual_slam_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Create TF broadcaster for pose information
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribe to camera and IMU data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publishers for SLAM outputs
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/visual_slam/map', 10)

        # SLAM state variables
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.latest_imu_data = None
        self.robot_pose = np.eye(4)  # 4x4 transformation matrix
        self.map_points = []

        self.get_logger().info('Isaac ROS Visual SLAM Node initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def imu_callback(self, msg):
        """Process IMU data for visual-inertial fusion"""
        self.latest_imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def image_callback(self, msg):
        """Process incoming image for SLAM"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Perform visual SLAM processing (this would call Isaac ROS accelerated functions)
            pose_update = self.process_visual_slam(cv_image, msg.header.stamp)

            if pose_update is not None:
                # Update robot pose
                self.robot_pose = self.robot_pose @ pose_update

                # Publish pose and odometry
                self.publish_pose(msg.header.stamp, msg.header.frame_id)
                self.publish_odometry(msg.header.stamp, msg.header.frame_id)

                # Broadcast TF transform
                self.broadcast_transform(msg.header.stamp, msg.header.frame_id)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_visual_slam(self, image, timestamp):
        """
        Process visual SLAM using Isaac ROS accelerated functions
        In practice, this would interface with Isaac ROS Visual SLAM node
        """
        # This is a simplified representation
        # Actual implementation would use Isaac ROS Visual SLAM node
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            # Simulate pose update based on visual features
            # In real implementation, this would call Isaac ROS accelerated functions
            pose_change = self.estimate_pose_change(image)
            return pose_change

        return None

    def estimate_pose_change(self, image):
        """
        Estimate pose change using visual features
        This would be replaced by Isaac ROS accelerated feature matching
        """
        # Placeholder for Isaac ROS accelerated feature matching
        # In practice, this would use CUDA-accelerated feature detection and matching
        return np.eye(4)  # Identity matrix as placeholder

    def publish_pose(self, timestamp, frame_id):
        """Publish robot pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = frame_id

        # Extract position and orientation from transformation matrix
        pose_msg.pose.position.x = self.robot_pose[0, 3]
        pose_msg.pose.position.y = self.robot_pose[1, 3]
        pose_msg.pose.position.z = self.robot_pose[2, 3]

        # Convert rotation matrix to quaternion
        rotation_matrix = self.robot_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)

        pose_msg.pose.orientation.w = qw
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz

        self.pose_pub.publish(pose_msg)

    def publish_odometry(self, timestamp, frame_id):
        """Publish odometry information"""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.robot_pose[0, 3]
        odom_msg.pose.pose.position.y = self.robot_pose[1, 3]
        odom_msg.pose.pose.position.z = self.robot_pose[2, 3]

        # Set orientation
        rotation_matrix = self.robot_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)

        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Set covariance (simplified)
        odom_msg.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                   0, 0.1, 0, 0, 0, 0,
                                   0, 0, 0.1, 0, 0, 0,
                                   0, 0, 0, 0.1, 0, 0,
                                   0, 0, 0, 0, 0.1, 0,
                                   0, 0, 0, 0, 0, 0.1]

        self.odom_pub.publish(odom_msg)

    def broadcast_transform(self, timestamp, frame_id):
        """Broadcast TF transform"""
        t = TransformStamped()

        t.header.stamp = timestamp
        t.header.frame_id = frame_id
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.robot_pose[0, 3]
        t.transform.translation.y = self.robot_pose[1, 3]
        t.transform.translation.z = self.robot_pose[2, 3]

        rotation_matrix = self.robot_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)

        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz

        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return qw, qx, qy, qz

def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSVisualSLAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4.3.2 Launch File for Isaac ROS Visual SLAM

```xml
<!-- visual_slam.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Declare launch arguments
    use_rectified_images = LaunchConfiguration('use_rectified_images')
    enable_slam_visualization = LaunchConfiguration('enable_slam_visualization')

    # Declare launch arguments
    declare_use_rectified_images_cmd = DeclareLaunchArgument(
        'use_rectified_images',
        default_value='True',
        description='Use camera rectified images'
    )

    declare_enable_slam_visualization_cmd = DeclareLaunchArgument(
        'enable_slam_visualization',
        default_value='True',
        description='Enable SLAM visualization'
    )

    # Isaac ROS Visual SLAM container
    visual_slam_container = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
                name='visual_slam',
                parameters=[{
                    'enable_rectified_pose': True,
                    'enable_fisheye_distortion': False,
                    'rectified_pose_frame_id': 'base_link',
                    'map_frame_id': 'map',
                    'odom_frame_id': 'odom',
                    'base_frame_id': 'base_link',
                    'enable_observations_view': True,
                    'enable_slam_visualization': enable_slam_visualization,
                    'enable_landmarks_view': True,
                    'enable_metrics_output': True,
                    'input_width': 640,
                    'input_height': 480,
                    'input_rate': 30.0,
                    'map_publish_rate': 1.0,
                }],
                remappings=[
                    ('/visual_slam/image', '/camera/image_rect_color'),
                    ('/visual_slam/camera_info', '/camera/camera_info'),
                    ('/visual_slam/imu', '/imu/data'),
                    ('/visual_slam/tracking/pose_graph/landmarks', '/landmarks'),
                    ('/visual_slam/tracking/pose_graph/optimized_landmarks', '/optimized_landmarks'),
                ],
            ),
        ],
        output='screen',
    )

    # Image format converter for rectification
    image_format_converter_container = ComposableNodeContainer(
        name='image_format_converter_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        condition=IfCondition(use_rectified_images),
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_image_format_converter',
                plugin='nvidia::isaac_ros::image_format_converter::ImageFormatConverterNode',
                name='image_format_converter',
                parameters=[{
                    'source_format': 'rgb8',
                    'dest_format': 'rgba8',
                }],
                remappings=[
                    ('image_raw', '/camera/image_raw'),
                    ('image', '/camera/image_rect_color'),
                ],
            ),
        ],
        output='screen',
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_rectified_images_cmd)
    ld.add_action(declare_enable_slam_visualization_cmd)

    # Add containers
    ld.add_action(visual_slam_container)
    ld.add_action(image_format_converter_container)

    return ld
```

## 4.4 Real-time Performance Optimization

### 4.4.1 Keyframe Selection

For humanoid robots moving through environments, efficient keyframe selection is crucial:

```python
import numpy as np
from scipy.spatial.distance import cdist

class KeyframeSelector:
    def __init__(self, min_translation=0.1, min_rotation=0.1, max_keyframes=100):
        self.min_translation = min_translation
        self.min_rotation = min_rotation
        self.max_keyframes = max_keyframes
        self.keyframes = []
        self.keyframe_poses = []

    def should_add_keyframe(self, current_pose):
        """Determine if current frame should be added as a keyframe"""
        if len(self.keyframe_poses) == 0:
            return True

        # Calculate distance to last keyframe
        last_pose = self.keyframe_poses[-1]
        translation_diff = np.linalg.norm(
            current_pose[:3, 3] - last_pose[:3, 3]
        )

        # Calculate rotation difference
        rotation_diff = self.rotation_distance(current_pose[:3, :3], last_pose[:3, :3])

        # Check if movement exceeds thresholds
        if translation_diff > self.min_translation or rotation_diff > self.min_rotation:
            return True

        return False

    def rotation_distance(self, R1, R2):
        """Calculate rotation distance between two rotation matrices"""
        R_rel = R1 @ R2.T
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return angle

    def add_keyframe(self, pose, image):
        """Add a new keyframe if conditions are met"""
        if self.should_add_keyframe(pose):
            self.keyframes.append(image)
            self.keyframe_poses.append(pose.copy())

            # Limit the number of keyframes
            if len(self.keyframes) > self.max_keyframes:
                self.keyframes.pop(0)
                self.keyframe_poses.pop(0)

            return True
        return False

    def get_recent_keyframes(self, n=5):
        """Get the n most recent keyframes"""
        return self.keyframes[-n:], self.keyframe_poses[-n:]
```

### 4.4.2 Multi-threaded Processing

To maintain real-time performance on humanoid robots:

```python
import threading
import queue
from collections import deque

class RealTimeVisualSLAM:
    def __init__(self):
        self.image_queue = queue.Queue(maxsize=10)
        self.pose_queue = queue.Queue(maxsize=10)
        self.keyframe_queue = queue.Queue(maxsize=5)

        self.slam_lock = threading.Lock()
        self.feature_detector = None  # Initialize with appropriate detector
        self.tracker = None  # Initialize with appropriate tracker

        # Start processing threads
        self.image_processing_thread = threading.Thread(target=self.process_images)
        self.optimization_thread = threading.Thread(target=self.optimize_map)

        self.running = True

    def start_processing(self):
        """Start the processing threads"""
        self.image_processing_thread.start()
        self.optimization_thread.start()

    def process_images(self):
        """Process images in a separate thread"""
        while self.running:
            try:
                # Get image from queue
                image_data = self.image_queue.get(timeout=1.0)

                # Process with Isaac ROS accelerated functions
                with self.slam_lock:
                    features = self.extract_features(image_data['image'])
                    pose_update = self.estimate_motion(
                        features,
                        image_data['timestamp']
                    )

                # Put results in appropriate queues
                if pose_update is not None:
                    self.pose_queue.put({
                        'pose': pose_update,
                        'timestamp': image_data['timestamp']
                    })

                self.image_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in image processing: {e}')

    def optimize_map(self):
        """Run map optimization in background"""
        while self.running:
            try:
                # Perform map optimization using Isaac ROS
                # This would call Isaac ROS optimization functions
                time.sleep(0.1)  # Simulate optimization work
            except Exception as e:
                self.get_logger().error(f'Error in map optimization: {e}')

    def extract_features(self, image):
        """Extract features using Isaac ROS accelerated functions"""
        # Placeholder for Isaac ROS feature extraction
        # In practice, this would use CUDA-accelerated feature detection
        return np.array([])  # Placeholder

    def estimate_motion(self, features, timestamp):
        """Estimate motion between frames"""
        # Placeholder for Isaac ROS motion estimation
        return np.eye(4)  # Placeholder
```

## 4.5 Integration with Isaac Sim for Training

### 4.5.1 Synthetic Data Generation for SLAM

```python
import omni.replicator.core as rep
import numpy as np

def setup_slam_training_data():
    """Set up Isaac Sim Replicator for SLAM training data generation"""

    # Create a camera for SLAM data collection
    with rep.new_layer():
        # Stereo camera setup for depth estimation
        left_camera = rep.create.camera(
            position=(-0.05, 0, 1.5),  # Slightly offset for stereo
            rotation=(0, 0, 0)
        )

        right_camera = rep.create.camera(
            position=(0.05, 0, 1.5),   # Slightly offset for stereo
            rotation=(0, 0, 0)
        )

        # Create render products
        left_render = rep.create.render_product(left_camera, (640, 480))
        right_render = rep.create.render_product(right_camera, (640, 480))

        # Add randomization for domain adaptation
        def randomize_environment():
            # Randomize lighting conditions
            lights = rep.get.lights()
            with lights:
                rep.modify.pose(
                    position=rep.distribution.uniform((-10, -10, 5), (10, 10, 15))
                )
                rep.modify.attribute(
                    "intensity", rep.distribution.normal(5000, 1000)
                )

            # Randomize object positions and properties
            objects = rep.get.prims_from_path("/World/Objects/*")
            with objects:
                rep.modify.pose(
                    position=rep.distribution.uniform((-5, -5, 0), (5, 5, 2)),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 3.14))
                )

            return lights, objects

        rep.randomizer.extents_generator(randomize_environment)

        # Annotators for SLAM training data
        with rep.trigger.on_frame():
            # Left camera data
            rep.annotators.camera_annotator(
                render_product=left_render,
                name="left_rgb",
                annotator="RgbdSchema"
            )

            # Right camera data
            rep.annotators.camera_annotator(
                render_product=right_render,
                name="right_rgb",
                annotator="RgbdSchema"
            )

            # Depth data for ground truth
            rep.annotators.camera_annotator(
                render_product=left_render,
                name="depth",
                annotator="DistanceToImagePlane"
            )

            # Pose data for ground truth
            rep.annotators.camera_annotator(
                render_product=left_render,
                name="camera_pose",
                annotator="CameraAnnotation"
            )

# Execute the SLAM training data setup
setup_slam_training_data()
```

## 4.6 Performance Considerations for Humanoid Robots

### 4.6.1 Computational Resource Management

```python
import psutil
import GPUtil
import time

class SLAMResourceMonitor:
    def __init__(self, cpu_threshold=80, gpu_threshold=85, memory_threshold=80):
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.memory_threshold = memory_threshold

    def check_resources(self):
        """Check if system resources are sufficient for SLAM"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        gpus = GPUtil.getGPUs()
        gpu_load = gpus[0].load * 100 if gpus else 0
        gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0

        # Check if any resource exceeds threshold
        if (cpu_percent > self.cpu_threshold or
            gpu_load > self.gpu_threshold or
            memory_percent > self.memory_threshold or
            gpu_memory > self.memory_threshold):

            # Reduce SLAM complexity to preserve resources
            self.reduce_slam_complexity()
            return False

        return True

    def reduce_slam_complexity(self):
        """Reduce SLAM processing complexity"""
        # This would adjust SLAM parameters like:
        # - Reduce feature detection density
        # - Lower image resolution
        # - Increase keyframe intervals
        # - Reduce optimization frequency
        print("Reducing SLAM complexity due to resource constraints")
```

### 4.6.2 Adaptive SLAM Parameters

```python
class AdaptiveSLAMParameters:
    def __init__(self):
        self.feature_density = 1000  # Number of features to track
        self.keyframe_interval = 10  # Process every 10th frame as keyframe
        self.optimization_frequency = 5  # Optimize every 5 keyframes
        self.tracking_threshold = 50   # Minimum features for tracking

    def adapt_to_performance(self, current_fps, tracking_features, processing_time):
        """Adapt SLAM parameters based on current performance"""
        target_fps = 30  # Desired SLAM FPS

        if current_fps < target_fps * 0.8:  # If significantly below target
            # Reduce computational load
            self.feature_density = max(100, int(self.feature_density * 0.8))
            self.keyframe_interval = min(50, self.keyframe_interval + 2)
            self.optimization_frequency = max(1, self.optimization_frequency - 1)
        elif current_fps > target_fps * 1.2:  # If significantly above target
            # Increase quality if resources allow
            self.feature_density = min(5000, int(self.feature_density * 1.1))
            self.keyframe_interval = max(5, self.keyframe_interval - 1)
            self.optimization_frequency = min(10, self.optimization_frequency + 1)

        # Ensure tracking doesn't fail due to too few features
        if tracking_features < self.tracking_threshold:
            self.feature_density = min(3000, self.feature_density + 100)
```

## 4.7 Validation and Testing

### 4.7.1 SLAM Accuracy Metrics

```python
import numpy as np
from scipy.spatial.distance import euclidean

class SLAMValidator:
    def __init__(self):
        self.ground_truth_poses = []
        self.estimated_poses = []
        self.alignment_transform = None

    def add_ground_truth_pose(self, pose):
        """Add ground truth pose from simulation"""
        self.ground_truth_poses.append(pose.copy())

    def add_estimated_pose(self, pose):
        """Add estimated pose from SLAM"""
        self.estimated_poses.append(pose.copy())

    def calculate_metrics(self):
        """Calculate SLAM accuracy metrics"""
        if len(self.ground_truth_poses) != len(self.estimated_poses):
            return None

        # Align estimated trajectory to ground truth
        self.align_trajectories()

        # Calculate trajectory errors
        position_errors = []
        rotation_errors = []

        for gt_pose, est_pose in zip(self.ground_truth_poses, self.estimated_poses):
            # Apply alignment transform to estimated pose
            aligned_est_pose = self.alignment_transform @ est_pose

            # Calculate position error
            pos_err = euclidean(
                gt_pose[:3, 3],
                aligned_est_pose[:3, 3]
            )
            position_errors.append(pos_err)

            # Calculate rotation error
            rot_err = self.rotation_error(
                gt_pose[:3, :3],
                aligned_est_pose[:3, :3]
            )
            rotation_errors.append(rot_err)

        # Calculate statistics
        rmse_position = np.sqrt(np.mean(np.array(position_errors) ** 2))
        rmse_rotation = np.sqrt(np.mean(np.array(rotation_errors) ** 2))
        mean_position_error = np.mean(position_errors)
        max_position_error = np.max(position_errors)

        return {
            'rmse_position': rmse_position,
            'rmse_rotation': rmse_rotation,
            'mean_position_error': mean_position_error,
            'max_position_error': max_position_error,
            'num_poses': len(position_errors)
        }

    def align_trajectories(self):
        """Align estimated trajectory to ground truth using Umeyama algorithm"""
        if len(self.ground_truth_poses) < 3:
            return

        # Extract positions
        gt_positions = np.array([pose[:3, 3] for pose in self.ground_truth_poses])
        est_positions = np.array([pose[:3, 3] for pose in self.estimated_poses])

        # Compute alignment transform using Umeyama algorithm
        self.alignment_transform = self.compute_alignment_transform(
            gt_positions, est_positions
        )

    def compute_alignment_transform(self, P, Q):
        """Compute alignment transform using SVD"""
        # Compute centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # Center the point sets
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Compute covariance matrix
        H = P_centered.T @ Q_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure proper rotation (not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_Q.T - R @ centroid_P.T

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def rotation_error(self, R1, R2):
        """Calculate rotation error between two rotation matrices"""
        R_rel = R1 @ R2.T
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return angle
```

## 4.8 Best Practices for Humanoid Robot Visual SLAM

### 4.8.1 Sensor Configuration

1. **Camera Selection**: Use cameras with appropriate FOV and resolution
2. **Stereo Setup**: Proper baseline for depth estimation
3. **Mounting Position**: Stable mounting on the robot body
4. **Calibration**: Regular calibration of camera intrinsics and extrinsics

### 4.8.2 Environmental Considerations

1. **Feature Richness**: Environments with sufficient visual features
2. **Lighting Conditions**: Adequate and consistent lighting
3. **Texture Variety**: Avoid textureless or repetitive surfaces
4. **Motion Smoothness**: Smooth robot motion for better tracking

### 4.8.3 Performance Optimization

1. **Hardware Acceleration**: Leverage Isaac ROS for GPU acceleration
2. **Multi-threading**: Separate tracking and mapping threads
3. **Keyframe Management**: Efficient keyframe selection and storage
4. **Loop Closure**: Regular loop closure to correct drift

## Summary

Visual SLAM is a fundamental capability for humanoid robots to navigate and understand their environment. With Isaac ROS, you can implement hardware-accelerated visual SLAM that runs efficiently on NVIDIA platforms like the Jetson Orin. Proper integration with Isaac Sim allows for synthetic data generation and testing in diverse scenarios. By implementing adaptive parameters and resource management, you can ensure that visual SLAM runs reliably on humanoid robots with limited computational resources. In the next chapter, we will explore occupancy grid mapping and how to deploy these systems on edge hardware.