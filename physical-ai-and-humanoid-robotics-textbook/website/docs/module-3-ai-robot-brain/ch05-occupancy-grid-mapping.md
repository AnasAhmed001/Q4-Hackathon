---
title: Chapter 5 - Occupancy Grid Mapping
description: Learn how to create and maintain occupancy grid maps for humanoid robots using Isaac Sim and deployment on Jetson platforms.
sidebar_position: 27
---

# Chapter 5 - Occupancy Grid Mapping

Occupancy grid mapping is a fundamental technique in robotics that represents the environment as a grid of cells, each indicating the probability of occupancy. For humanoid robots, accurate occupancy grid maps are essential for navigation, path planning, and obstacle avoidance. This chapter explores the creation, maintenance, and deployment of occupancy grid maps, with a focus on Isaac Sim for simulation and NVIDIA Jetson platforms for edge deployment.

## 5.1 Introduction to Occupancy Grid Mapping

Occupancy grid mapping represents the environment as a 2D or 3D grid where each cell contains a probability value indicating whether that space is occupied by an obstacle. This representation is particularly useful for humanoid robots because:

- **Discrete Representation**: Simplifies complex environments into manageable grid cells
- **Probabilistic Nature**: Handles uncertainty in sensor readings
- **Path Planning Compatibility**: Works well with common path planning algorithms
- **Real-time Updates**: Can be updated efficiently as the robot moves

### 5.1.1 Grid Map Structure

A 2D occupancy grid map typically consists of:
- **Grid Cells**: Discrete locations in space
- **Resolution**: Size of each cell (e.g., 0.05m x 0.05m)
- **Occupancy Values**: Probability of occupancy (0.0 = free, 1.0 = occupied)
- **Metadata**: Origin, dimensions, and resolution information

### 5.1.2 Types of Grid Maps

- **2D Grid Maps**: Planar representation suitable for ground-based navigation
- **3D Grid Maps**: Volumetric representation for complex environments
- **Multi-layer Maps**: Separate layers for different types of information
- **Dynamic Maps**: Include temporal information for moving obstacles

## 5.2 Occupancy Grid Mapping Fundamentals

### 5.2.1 Sensor Model

The sensor model describes how sensor readings update the grid map. For humanoid robots, common sensors include:

- **LIDAR**: Provides precise distance measurements
- **Stereo Cameras**: Generate depth information
- **RGB-D Cameras**: Provide both color and depth
- **Ultrasonic Sensors**: Short-range obstacle detection

```python
import numpy as np
import math

class SensorModel:
    def __init__(self, max_range=10.0, free_threshold=0.3, occupied_threshold=0.6):
        self.max_range = max_range
        self.free_threshold = free_threshold
        self.occupied_threshold = occupied_threshold
        self.p_hit = 0.7
        self.p_miss = 0.4
        self.p_random = 0.05
        self.p_occ = 0.6

    def ray_casting_update(self, grid_map, robot_pose, laser_scan):
        """
        Update grid map using ray casting algorithm
        """
        robot_x, robot_y, robot_theta = robot_pose

        for i, range_reading in enumerate(laser_scan.ranges):
            if not (self.max_range * 0.1 < range_reading < self.max_range):
                continue

            # Calculate angle of this laser beam
            angle = laser_scan.angle_min + i * laser_scan.angle_increment + robot_theta

            # Calculate endpoint of this laser beam
            end_x = robot_x + range_reading * math.cos(angle)
            end_y = robot_y + range_reading * math.sin(angle)

            # Convert to grid coordinates
            grid_end_x = int((end_x - grid_map.origin_x) / grid_map.resolution)
            grid_end_y = int((end_y - grid_map.origin_y) / grid_map.resolution)

            # Calculate start point (robot position)
            grid_start_x = int((robot_x - grid_map.origin_x) / grid_map.resolution)
            grid_start_y = int((robot_y - grid_map.origin_y) / grid_map.resolution)

            # Perform ray tracing to update free space
            self.update_free_space(grid_map, grid_start_x, grid_start_y, grid_end_x, grid_end_y)

            # Update endpoint as occupied
            if 0 <= grid_end_x < grid_map.width and 0 <= grid_end_y < grid_map.height:
                current_log_odds = grid_map.log_odds[grid_end_y, grid_end_x]
                new_log_odds = current_log_odds + self.log_odds(self.p_occ)
                grid_map.log_odds[grid_end_y, grid_end_x] = min(new_log_odds, 50)  # Clamp

    def update_free_space(self, grid_map, start_x, start_y, end_x, end_y):
        """
        Update free space along a ray using Bresenham's algorithm
        """
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x_step = 1 if end_x > start_x else -1
        y_step = 1 if end_y > start_y else -1

        error = dx - dy
        x, y = start_x, start_y

        while x != end_x or y != end_y:
            if 0 <= x < grid_map.width and 0 <= y < grid_map.height:
                current_log_odds = grid_map.log_odds[y, x]
                new_log_odds = current_log_odds + self.log_odds(self.p_free)
                grid_map.log_odds[y, x] = max(new_log_odds, -50)  # Clamp

            # Bresenham's algorithm step
            double_error = 2 * error
            if double_error > -dy:
                error -= dy
                x += x_step
            if double_error < dx:
                error += dx
                y += y_step

    def log_odds(self, probability):
        """Convert probability to log odds"""
        return math.log(probability / (1 - probability))

    @property
    def p_free(self):
        return 1 - self.p_occ
```

### 5.2.2 Map Representation

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GridMap:
    width: int
    height: int
    resolution: float  # meters per cell
    origin_x: float    # world x coordinate of grid[0,0]
    origin_y: float    # world y coordinate of grid[0,0]

    def __post_init__(self):
        # Initialize log odds representation
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        # Initialize probability representation
        self.probability = np.full((self.height, self.width), 0.5, dtype=np.float32)

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.resolution + self.origin_x
        world_y = grid_y * self.resolution + self.origin_y
        return world_x, world_y

    def is_valid_grid(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are within map bounds"""
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height

    def get_probability(self, grid_x: int, grid_y: int) -> float:
        """Get occupancy probability for a cell"""
        if self.is_valid_grid(grid_x, grid_y):
            return self.probability[grid_y, grid_x]
        return 0.5  # Unknown area

    def update_probability(self):
        """Convert log odds to probability"""
        # Convert log odds to probability: p = 1 - 1/(1 + exp(log_odds))
        exp_log_odds = np.exp(self.log_odds)
        self.probability = exp_log_odds / (1 + exp_log_odds)
```

## 5.3 Isaac ROS Occupancy Grid Mapping

Isaac ROS provides optimized occupancy grid mapping capabilities that leverage NVIDIA GPU acceleration:

### 5.3.1 Isaac ROS Occupancy Grid Node

```python
# Example Isaac ROS Occupancy Grid implementation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from visualization_msgs.msg import MarkerArray
from builtin_interfaces.msg import Time

class IsaacOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('isaac_occupancy_grid_node')

        # Parameters
        self.declare_parameter('map_resolution', 0.05)  # meters per cell
        self.declare_parameter('map_width', 40)         # cells
        self.declare_parameter('map_height', 40)        # cells
        self.declare_parameter('map_origin_x', -10.0)   # meters
        self.declare_parameter('map_origin_y', -10.0)   # meters
        self.declare_parameter('update_rate', 5.0)      # Hz

        # Get parameters
        self.resolution = self.get_parameter('map_resolution').value
        self.width = self.get_parameter('map_width').value
        self.height = self.get_parameter('map_height').value
        self.origin_x = self.get_parameter('map_origin_x').value
        self.origin_y = self.get_parameter('map_origin_y').value
        self.update_rate = self.get_parameter('update_rate').value

        # Initialize map
        self.grid_map = np.full((self.height, self.width), -1, dtype=np.int8)  # -1 = unknown
        self.grid_map_metadata = MapMetaData()
        self.grid_map_metadata.resolution = self.resolution
        self.grid_map_metadata.width = self.width
        self.grid_map_metadata.height = self.height
        self.grid_map_metadata.origin = Pose()
        self.grid_map_metadata.origin.position.x = self.origin_x
        self.grid_map_metadata.origin.position.y = self.origin_y
        self.grid_map_metadata.origin.orientation.w = 1.0

        # TF buffer for transform lookups
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.map_metadata_pub = self.create_publisher(MapMetaData, '/map_metadata', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/map_visualization', 10)

        # Timer for periodic map updates
        self.timer = self.create_timer(1.0 / self.update_rate, self.publish_map)

        self.get_logger().info(f'Isaac Occupancy Grid Node initialized with {self.width}x{self.height} map')

    def laser_callback(self, msg):
        """Process laser scan data to update occupancy grid"""
        try:
            # Get robot's transform in map frame
            transform = self.tf_buffer.lookup_transform(
                'map',
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().error(f'Could not transform laser scan: {ex}')
            return

        # Convert laser scan to map coordinates and update grid
        self.update_grid_with_laser(msg, transform)

    def pointcloud_callback(self, msg):
        """Process point cloud data to update occupancy grid"""
        try:
            # Get transform to map frame
            transform = self.tf_buffer.lookup_transform(
                'map',
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().error(f'Could not transform point cloud: {ex}')
            return

        # Convert point cloud to map coordinates and update grid
        self.update_grid_with_pointcloud(msg, transform)

    def update_grid_with_laser(self, laser_msg, transform):
        """Update occupancy grid using laser scan data"""
        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y
        robot_yaw = self.quaternion_to_yaw(transform.transform.rotation)

        for i, range_reading in enumerate(laser_msg.ranges):
            if not (0.1 < range_reading < laser_msg.range_max):
                continue

            # Calculate laser beam angle in map frame
            beam_angle = laser_msg.angle_min + i * laser_msg.angle_increment + robot_yaw

            # Calculate endpoint in world coordinates
            world_x = robot_x + range_reading * math.cos(beam_angle)
            world_y = robot_y + range_reading * math.sin(beam_angle)

            # Convert to grid coordinates
            grid_x = int((world_x - self.origin_x) / self.resolution)
            grid_y = int((world_y - self.origin_y) / self.resolution)

            # Check bounds
            if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
                continue

            # Update cell as occupied
            self.grid_map[grid_y, grid_x] = 100  # 100 = occupied

            # Update free space along the ray
            self.update_free_space_along_ray(robot_x, robot_y, world_x, world_y)

    def update_free_space_along_ray(self, start_x, start_y, end_x, end_y):
        """Update free space along a ray using Bresenham's algorithm"""
        grid_start_x = int((start_x - self.origin_x) / self.resolution)
        grid_start_y = int((start_y - self.origin_y) / self.resolution)
        grid_end_x = int((end_x - self.origin_x) / self.resolution)
        grid_end_y = int((end_y - self.origin_y) / self.resolution)

        dx = abs(grid_end_x - grid_start_x)
        dy = abs(grid_end_y - grid_start_y)
        x_step = 1 if grid_end_x > grid_start_x else -1
        y_step = 1 if grid_end_y > grid_start_y else -1

        error = dx - dy
        x, y = grid_start_x, grid_start_y

        while x != grid_end_x or y != grid_end_y:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Update as free space (but don't overwrite occupied cells)
                if self.grid_map[y, x] < 50:  # Only update if not definitely occupied
                    self.grid_map[y, x] = 0  # 0 = free

            double_error = 2 * error
            if double_error > -dy:
                error -= dy
                x += x_step
            if double_error < dx:
                error += dx
                y += y_step

    def update_grid_with_pointcloud(self, pc_msg, transform):
        """Update occupancy grid using point cloud data"""
        # This would use Isaac ROS accelerated point cloud processing
        # For now, a simplified implementation
        import sensor_msgs.point_cloud2 as pc2

        for point in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
            # Transform point to map frame
            transformed_x = point[0] * transform.transform.rotation.w + transform.transform.translation.x
            transformed_y = point[1] * transform.transform.rotation.w + transform.transform.translation.y

            # Convert to grid coordinates
            grid_x = int((transformed_x - self.origin_x) / self.resolution)
            grid_y = int((transformed_y - self.origin_y) / self.resolution)

            # Check bounds and update as occupied
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.grid_map[grid_y, grid_x] = 100

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_map(self):
        """Publish the occupancy grid map"""
        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = self.grid_map_metadata

        # Flatten grid data for message
        msg.data = self.grid_map.flatten().tolist()

        self.map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacOccupancyGridNode()

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

### 5.3.2 Isaac ROS Map Fusion Node

For humanoid robots with multiple sensors, map fusion is important:

```python
# Example Isaac ROS Map Fusion implementation
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np
from threading import Lock

class IsaacMapFusionNode(Node):
    def __init__(self):
        super().__init__('isaac_map_fusion_node')

        self.map_lock = Lock()

        # Initialize multiple maps for different sensors
        self.laser_map = None
        self.vision_map = None
        self.fused_map = None

        # Subscribers for different sensor maps
        self.laser_map_sub = self.create_subscription(
            OccupancyGrid,
            '/laser_map',
            self.laser_map_callback,
            10
        )

        self.vision_map_sub = self.create_subscription(
            OccupancyGrid,
            '/vision_map',
            self.vision_map_callback,
            10
        )

        # Publisher for fused map
        self.fused_map_pub = self.create_publisher(OccupancyGrid, '/fused_map', 10)

        # Timer for periodic fusion
        self.fusion_timer = self.create_timer(0.2, self.fuse_maps)

    def laser_map_callback(self, msg):
        """Update laser-based occupancy map"""
        with self.map_lock:
            self.laser_map = msg

    def vision_map_callback(self, msg):
        """Update vision-based occupancy map"""
        with self.map_lock:
            self.vision_map = msg

    def fuse_maps(self):
        """Fuse multiple occupancy grids into a single map"""
        with self.map_lock:
            if self.laser_map is None or self.vision_map is None:
                return

            # Check if maps have compatible metadata
            if (self.laser_map.info.width != self.vision_map.info.width or
                self.laser_map.info.height != self.vision_map.info.height or
                abs(self.laser_map.info.resolution - self.vision_map.info.resolution) > 1e-6):
                self.get_logger().error('Map dimensions or resolution do not match')
                return

            # Convert to probability representation for fusion
            laser_prob = np.array(self.laser_map.data).reshape(
                self.laser_map.info.height, self.laser_map.info.width
            ).astype(np.float32) / 100.0

            vision_prob = np.array(self.vision_map.data).reshape(
                self.vision_map.info.height, self.vision_map.info.width
            ).astype(np.float32) / 100.0

            # Fuse maps using probabilistic methods
            # Using Dempster-Shafer theory for sensor fusion
            fused_prob = self.dempster_shafer_fusion(laser_prob, vision_prob)

            # Convert back to occupancy grid format
            fused_data = (fused_prob * 100).astype(np.int8).flatten()

            # Create and publish fused map
            fused_msg = OccupancyGrid()
            fused_msg.header = self.laser_map.header
            fused_msg.info = self.laser_map.info
            fused_msg.data = fused_data.tolist()

            self.fused_map_pub.publish(fused_msg)

    def dempster_shafer_fusion(self, prob1, prob2):
        """
        Fuse two probability maps using Dempster-Shafer theory
        This is a simplified version focusing on combining evidence
        """
        # Convert probabilities to log-odds for easier combination
        def prob_to_log_odds(p):
            p = np.clip(p, 0.001, 0.999)  # Avoid log(0)
            return np.log(p / (1 - p))

        def log_odds_to_prob(log_odds):
            return 1 - 1 / (1 + np.exp(log_odds))

        log_odds1 = prob_to_log_odds(prob1)
        log_odds2 = prob_to_log_odds(prob2)

        # Combine log-odds (assuming independent evidence)
        combined_log_odds = log_odds1 + log_odds2

        # Convert back to probability
        fused_prob = log_odds_to_prob(combined_log_odds)

        # Clip to valid range
        fused_prob = np.clip(fused_prob, 0, 1)

        return fused_prob
```

## 5.4 Optimized Mapping for Jetson Platforms

### 5.4.1 Memory-Efficient Grid Implementation

```python
import numpy as np
import numba
from numba import jit
import threading
import queue

class JetsonOptimizedGridMap:
    def __init__(self, width=400, height=400, resolution=0.05):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2

        # Use int8 for memory efficiency
        self.grid = np.full((height, width), -1, dtype=np.int8)  # -1 = unknown

        # Use queues for multi-threaded updates
        self.update_queue = queue.Queue(maxsize=100)
        self.update_thread = threading.Thread(target=self.process_updates, daemon=True)
        self.update_thread.start()

    @jit(nopython=True)
    def bresenham_ray_casting(self, grid, start_x, start_y, end_x, end_y, resolution, origin_x, origin_y):
        """
        Optimized ray casting using Numba for speed
        """
        grid_start_x = int((start_x - origin_x) / resolution)
        grid_start_y = int((start_y - origin_y) / resolution)
        grid_end_x = int((end_x - origin_x) / resolution)
        grid_end_y = int((end_y - origin_y) / resolution)

        # Bresenham's line algorithm
        dx = abs(grid_end_x - grid_start_x)
        dy = abs(grid_end_y - grid_start_y)
        x_step = 1 if grid_end_x > grid_start_x else -1
        y_step = 1 if grid_end_y > grid_start_y else -1

        error = dx - dy
        x, y = grid_start_x, grid_start_y

        # Update along the ray
        while x != grid_end_x or y != grid_end_y:
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                # Update as free space (only if not occupied)
                if grid[y, x] < 50:
                    grid[y, x] = 0

            double_error = 2 * error
            if double_error > -dy:
                error -= dy
                x += x_step
            if double_error < dx:
                error += dx
                y += y_step

        # Mark endpoint as occupied
        if 0 <= grid_end_x < grid.shape[1] and 0 <= grid_end_y < grid.shape[0]:
            grid[grid_end_y, grid_end_x] = 100

    def update_with_scan(self, robot_pose, scan_ranges, scan_angles):
        """
        Update map with laser scan data using optimized ray casting
        """
        robot_x, robot_y, robot_theta = robot_pose

        for i, range_reading in enumerate(scan_ranges):
            if range_reading < 0.1 or range_reading > 10.0:  # Filter invalid ranges
                continue

            angle = scan_angles[i] + robot_theta
            end_x = robot_x + range_reading * np.cos(angle)
            end_y = robot_y + range_reading * np.sin(angle)

            # Use optimized ray casting
            self.bresenham_ray_casting(
                self.grid, robot_x, robot_y, end_x, end_y,
                self.resolution, self.origin_x, self.origin_y
            )

    def process_updates(self):
        """Process map updates in a separate thread"""
        while True:
            try:
                update_data = self.update_queue.get(timeout=1.0)
                self.apply_update(update_data)
                self.update_queue.task_done()
            except queue.Empty:
                continue

    def apply_update(self, update_data):
        """Apply a single map update"""
        # This would apply the update to the grid
        # Implementation depends on update_data format
        pass
```

### 5.4.2 Multi-resolution Mapping

```python
class MultiResolutionMap:
    def __init__(self, base_resolution=0.05, levels=3):
        self.base_resolution = base_resolution
        self.levels = levels
        self.maps = {}

        # Create maps at different resolutions
        for level in range(levels):
            resolution = base_resolution * (2 ** level)
            width = int(40 / resolution)  # 40m x 40m area
            height = int(40 / resolution)

            self.maps[level] = {
                'resolution': resolution,
                'width': width,
                'height': height,
                'grid': np.full((height, width), -1, dtype=np.int8),
                'origin_x': -20.0,  # Center around robot
                'origin_y': -20.0
            }

    def update_at_all_levels(self, sensor_data):
        """Update maps at all resolutions"""
        for level in range(self.levels):
            self.update_single_level(level, sensor_data)

    def update_single_level(self, level, sensor_data):
        """Update a single resolution level"""
        map_info = self.maps[level]

        # Process sensor data at this resolution
        for point in sensor_data:
            # Convert world coordinates to grid coordinates for this level
            grid_x = int((point[0] - map_info['origin_x']) / map_info['resolution'])
            grid_y = int((point[1] - map_info['origin_y']) / map_info['resolution'])

            if 0 <= grid_x < map_info['width'] and 0 <= grid_y < map_info['height']:
                # Update occupancy probability
                current_val = map_info['grid'][grid_y, grid_x]
                if current_val == -1:  # Unknown
                    map_info['grid'][grid_y, grid_x] = 0  # Free
                elif current_val < 50:  # Free or uncertain
                    map_info['grid'][grid_y, grid_x] = min(100, current_val + 10)  # More occupied
                else:  # Already occupied
                    map_info['grid'][grid_y, grid_x] = min(100, current_val + 5)

    def get_map_at_resolution(self, target_resolution):
        """Get map at or closest to target resolution"""
        best_level = 0
        best_diff = abs(self.maps[0]['resolution'] - target_resolution)

        for level in range(1, self.levels):
            diff = abs(self.maps[level]['resolution'] - target_resolution)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        return self.maps[best_level]
```

## 5.5 Integration with Isaac Sim for Map Training

### 5.5.1 Synthetic Map Generation

```python
import omni.replicator.core as rep
import numpy as np

def setup_synthetic_map_training():
    """Set up Isaac Sim for generating synthetic occupancy maps"""

    # Create a complex environment for map training
    with rep.new_layer():
        # Create random obstacles
        def create_random_obstacles():
            positions = rep.distribution.uniform((-15, -15, 0), (15, 15, 0))
            heights = rep.distribution.uniform(0.1, 2.0)
            widths = rep.distribution.uniform(0.5, 2.0)
            lengths = rep.distribution.uniform(0.5, 2.0)

            obstacles = rep.create.cube(
                position=positions,
                scale=rep.distribution.uniform((0.5, 0.5, 0.1), (2.0, 2.0, 2.0)),
            )

            return obstacles

        # Create random environments
        def create_random_environment():
            # Randomize lighting
            lights = rep.create.light(
                position=rep.distribution.uniform((-20, -20, 5), (20, 20, 15)),
                intensity=rep.distribution.normal(5000, 1000),
                light_type="distant"
            )

            # Randomize ground texture
            ground_material = rep.create.material(
                diffuse_texture=rep.distribution.choice([
                    "textures/ground1.png",
                    "textures/ground2.png",
                    "textures/ground3.png"
                ])
            )

            return lights

        # Randomize the scene
        rep.randomizer.extents_generator(create_random_obstacles)
        rep.randomizer.extents_generator(create_random_environment)

        # Create a robot with sensors
        robot = rep.create.from_usd(
            usd_path="/Isaac/Robots/Carter/carter_navigation.usd",
            position=(0, 0, 0)
        )

        # Add LIDAR sensor to robot
        lidar = rep.create.lidar(
            position=(0.5, 0, 0.5),  # On robot
            rotation=(0, 0, 0),
            sensor_period=1.0/10.0,  # 10 Hz
            horizontal_samples=360,
            horizontal_min_angle=-np.pi,
            horizontal_max_angle=np.pi,
            vertical_samples=1,
            vertical_min_angle=0,
            vertical_max_angle=0,
            range=20.0
        )

        # Annotators for synthetic map generation
        with rep.trigger.on_frame():
            # Generate ground truth occupancy map
            rep.annotators.occupancy_map(
                name="synthetic_occupancy_map",
                resolution=0.1,  # 10cm resolution
                width=400,       # 40m x 40m map
                height=400,
                origin_x=-20.0,
                origin_y=-20.0
            )

            # Generate semantic segmentation for object identification
            rep.annotators.camera_annotator(
                render_product=rep.create.render_product(
                    rep.create.camera(position=(0, 0, 2)),
                    (640, 480)
                ),
                name="semantic_segmentation",
                annotator="SemanticSegmentation"
            )

# Execute the synthetic map training setup
setup_synthetic_map_training()
```

## 5.6 Map Management and Optimization

### 5.6.1 Dynamic Map Expansion

```python
class DynamicGridMap:
    def __init__(self, initial_size=200, resolution=0.05):
        self.resolution = resolution
        self.initial_size = initial_size
        self.grid = np.full((initial_size, initial_size), -1, dtype=np.int8)
        self.origin_x = -initial_size * resolution / 2
        self.origin_y = -initial_size * resolution / 2
        self.width = initial_size
        self.height = initial_size

    def expand_if_needed(self, world_x, world_y):
        """Expand map if robot moves outside current bounds"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)

        margin = 50  # cells of margin

        # Check if we need to expand
        if (grid_x < margin or grid_x >= self.width - margin or
            grid_y < margin or grid_y >= self.height - margin):

            # Calculate new dimensions
            new_width = max(self.width, abs(grid_x) * 2 + margin * 2)
            new_height = max(self.height, abs(grid_y) * 2 + margin * 2)

            # Create new larger grid
            new_grid = np.full((new_height, new_width), -1, dtype=np.int8)

            # Calculate offset to copy old grid to center of new grid
            offset_x = (new_width - self.width) // 2
            offset_y = (new_height - self.height) // 2

            # Copy old grid to new grid
            new_grid[offset_y:offset_y + self.height,
                    offset_x:offset_x + self.width] = self.grid

            # Update map parameters
            self.grid = new_grid
            self.width = new_width
            self.height = new_height
            self.origin_x -= offset_x * self.resolution
            self.origin_y -= offset_y * self.resolution

    def update_with_pose(self, robot_pose, sensor_data):
        """Update map with robot pose and sensor data"""
        robot_x, robot_y, _ = robot_pose

        # Expand map if needed
        self.expand_if_needed(robot_x, robot_y)

        # Update with sensor data
        self.update_with_sensor_data(robot_x, robot_y, sensor_data)

    def update_with_sensor_data(self, robot_x, robot_y, sensor_data):
        """Update map with sensor data"""
        # Convert robot position to grid coordinates
        grid_robot_x = int((robot_x - self.origin_x) / self.resolution)
        grid_robot_y = int((robot_y - self.origin_y) / self.resolution)

        # Process each sensor reading
        for reading in sensor_data:
            range_reading, angle = reading
            if range_reading < 0.1 or range_reading > 20.0:
                continue

            # Calculate endpoint in world coordinates
            world_end_x = robot_x + range_reading * np.cos(angle)
            world_end_y = robot_y + range_reading * np.sin(angle)

            # Convert to grid coordinates
            grid_end_x = int((world_end_x - self.origin_x) / self.resolution)
            grid_end_y = int((world_end_y - self.origin_y) / self.resolution)

            # Update free space along ray
            self.update_free_space(grid_robot_x, grid_robot_y, grid_end_x, grid_end_y)

            # Update endpoint as occupied
            if (0 <= grid_end_x < self.width and 0 <= grid_end_y < self.height):
                self.grid[grid_end_y, grid_end_x] = 100

    def update_free_space(self, start_x, start_y, end_x, end_y):
        """Update free space along a ray"""
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x_step = 1 if end_x > start_x else -1
        y_step = 1 if end_y > start_y else -1

        error = dx - dy
        x, y = start_x, start_y

        while x != end_x or y != end_y:
            if (0 <= x < self.width and 0 <= y < self.height and
                self.grid[y, x] < 50):  # Only update if not occupied
                self.grid[y, x] = 0

            double_error = 2 * error
            if double_error > -dy:
                error -= dy
                x += x_step
            if double_error < dx:
                error += dx
                y += y_step
```

### 5.6.2 Map Compression and Storage

```python
import zlib
import pickle
from typing import Dict, Any

class MapCompressor:
    @staticmethod
    def compress_map(grid_map: np.ndarray) -> bytes:
        """Compress occupancy grid using run-length encoding and zlib"""
        # Convert to bytes
        grid_bytes = grid_map.tobytes()

        # Compress using zlib
        compressed = zlib.compress(grid_bytes)

        return compressed

    @staticmethod
    def decompress_map(compressed_data: bytes, shape: tuple) -> np.ndarray:
        """Decompress occupancy grid"""
        # Decompress
        decompressed = zlib.decompress(compressed_data)

        # Convert back to numpy array
        grid_map = np.frombuffer(decompressed, dtype=np.int8).reshape(shape)

        return grid_map

    @staticmethod
    def save_map(grid_map: np.ndarray, filename: str):
        """Save compressed map to file"""
        compressed = MapCompressor.compress_map(grid_map)

        # Save with metadata
        map_data = {
            'compressed_grid': compressed,
            'shape': grid_map.shape,
            'dtype': str(grid_map.dtype)
        }

        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)

    @staticmethod
    def load_map(filename: str) -> np.ndarray:
        """Load compressed map from file"""
        with open(filename, 'rb') as f:
            map_data = pickle.load(f)

        grid_map = MapCompressor.decompress_map(
            map_data['compressed_grid'],
            map_data['shape']
        )

        return grid_map
```

## 5.7 Validation and Quality Assessment

### 5.7.1 Map Quality Metrics

```python
import numpy as np
from scipy.ndimage import label, binary_opening, binary_closing

class MapQualityAssessor:
    def __init__(self):
        self.metrics = {}

    def assess_map_quality(self, occupancy_grid: np.ndarray) -> Dict[str, float]:
        """Assess various quality metrics of the occupancy grid"""
        # Convert to binary for some metrics (occupied = 1, free = 0, unknown = 0)
        binary_map = (occupancy_grid > 50).astype(int)  # Occupied cells
        free_map = (occupancy_grid < 50).astype(int)    # Free cells

        metrics = {}

        # Coverage metrics
        total_cells = occupancy_grid.size
        known_cells = np.sum(occupancy_grid != -1)
        metrics['coverage_ratio'] = known_cells / total_cells if total_cells > 0 else 0

        # Occupancy statistics
        occupied_cells = np.sum(binary_map)
        metrics['occupancy_ratio'] = occupied_cells / total_cells if total_cells > 0 else 0

        # Connectivity analysis
        labeled_map, num_features = label(binary_map)
        metrics['num_obstacles'] = num_features

        # Map complexity (edge detection)
        from scipy import ndimage
        sobel_x = ndimage.sobel(occupancy_grid.astype(float), axis=0)
        sobel_y = ndimage.sobel(occupancy_grid.astype(float), axis=1)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        metrics['edge_density'] = np.mean(edge_magnitude[occupancy_grid != -1]) if known_cells > 0 else 0

        # Consistency check (how much the map has changed recently)
        # This would require comparing with previous maps

        # Navigability assessment
        navigable_area = np.sum(free_map) / total_cells if total_cells > 0 else 0
        metrics['navigable_ratio'] = navigable_area

        self.metrics = metrics
        return metrics

    def validate_map_consistency(self, current_map: np.ndarray, previous_map: np.ndarray) -> float:
        """Validate consistency between current and previous maps"""
        if current_map.shape != previous_map.shape:
            raise ValueError("Map shapes do not match")

        # Calculate change ratio
        changes = np.sum(current_map != previous_map)
        total_valid = np.sum((current_map != -1) & (previous_map != -1))

        consistency_ratio = 1 - (changes / total_valid if total_valid > 0 else 0)
        return consistency_ratio

    def detect_map_anomalies(self, occupancy_grid: np.ndarray) -> list:
        """Detect potential anomalies in the map"""
        anomalies = []

        # Check for impossible transitions (occupied to free in one scan)
        # This would require temporal analysis

        # Check for isolated occupied cells (likely noise)
        binary_map = (occupancy_grid > 50).astype(int)
        labeled_map, num_features = label(binary_map)

        # Find small connected components (likely noise)
        unique, counts = np.unique(labeled_map, return_counts=True)
        small_components = unique[counts < 10]  # Less than 10 cells
        if len(small_components) > 1:  # Exclude background (0)
            anomalies.append(f"Found {len(small_components)-1} small occupied components (possible noise)")

        return anomalies
```

## 5.8 Best Practices for Humanoid Robot Mapping

### 5.8.1 Performance Optimization

1. **Resolution Selection**: Choose appropriate resolution based on robot size and environment
2. **Update Frequency**: Balance update rate with computational resources
3. **Map Size**: Use dynamic maps that expand as needed
4. **Multi-threading**: Separate sensor processing from map updates
5. **GPU Acceleration**: Leverage Isaac ROS for hardware acceleration

### 5.8.2 Robustness Considerations

1. **Sensor Fusion**: Combine multiple sensor types for better accuracy
2. **Temporal Filtering**: Use temporal information to reduce noise
3. **Dynamic Object Handling**: Distinguish between static and dynamic obstacles
4. **Uncertainty Modeling**: Properly represent uncertainty in the map
5. **Validation**: Continuously validate map quality and consistency

## Summary

Occupancy grid mapping is a fundamental capability for humanoid robots to understand and navigate their environment. With Isaac ROS and optimized implementations, you can create efficient mapping systems that run well on edge platforms like the NVIDIA Jetson. The combination of Isaac Sim for synthetic data generation and Isaac ROS for hardware acceleration enables robust mapping systems that can handle the complex requirements of humanoid robots. Proper map management, including dynamic expansion and compression, ensures that the system can operate effectively in large environments with limited computational resources. In the next chapter, we will explore the Nav2 navigation stack and how to configure it specifically for bipedal humanoid robots.