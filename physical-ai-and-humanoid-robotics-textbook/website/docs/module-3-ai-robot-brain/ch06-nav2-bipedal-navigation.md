---
title: Chapter 6 - Nav2 Navigation for Bipedal Robots
description: Configure and optimize the Nav2 navigation stack specifically for bipedal humanoid robots using behavior trees and custom controllers.
sidebar_position: 28
---

# Chapter 6 - Nav2 Navigation for Bipedal Robots

The Navigation2 (Nav2) stack is the standard navigation framework for ROS 2, providing path planning, obstacle avoidance, and motion control capabilities. For bipedal humanoid robots, Nav2 requires specific configuration and customization to account for the unique kinematic and dynamic constraints of walking robots. This chapter explores how to configure Nav2 for bipedal locomotion, implement custom controllers, and integrate behavior trees for complex navigation tasks.

## 6.1 Introduction to Nav2 for Humanoid Robots

Nav2 is a comprehensive navigation framework that includes:
- **Global Planner**: Path planning from start to goal
- **Local Planner**: Local path following and obstacle avoidance
- **Controller**: Low-level motion control
- **Behavior Trees**: Task orchestration and decision making
- **Costmap Management**: Obstacle representation and safety margins

For bipedal humanoid robots, Nav2 must be adapted to handle:
- **Bipedal Kinematics**: Different from wheeled or tracked robots
- **Balance Constraints**: Maintaining stability during navigation
- **Step-by-Step Motion**: Discrete foot placement rather than continuous motion
- **Dynamic Stability**: Center of mass considerations during walking

### 6.1.1 Challenges for Bipedal Navigation

- **Foot Placement**: Precise footstep planning required
- **Balance Maintenance**: Continuous balance control during movement
- **Turning Mechanisms**: Different turning dynamics compared to wheeled robots
- **Step Height Limitations**: Ability to navigate small obstacles
- **Walking Speed**: Slower speeds affecting navigation behavior

## 6.2 Nav2 Architecture Overview

### 6.2.1 Core Components

```yaml
# Nav2 Core Components
Navigation System:
  ├── Global Planner
  │   ├── A* / Dijkstra / NavFn
  │   └── Path Smoothing
  ├── Local Planner
  │   ├── Trajectory Rollout
  │   ├── Obstacle Avoidance
  │   └── Velocity Control
  ├── Controller
  │   ├── Pure Pursuit
  │   ├── MPC
  │   └── Custom Controllers
  ├── Costmap 2D
  │   ├── Static Layer
  │   ├── Obstacle Layer
  │   ├── Inflation Layer
  │   └── Voxel Layer
  └── Behavior Trees
      ├── Action Nodes
      ├── Condition Nodes
      └── Decorator Nodes
```

### 6.2.2 Nav2 Parameters for Bipedal Robots

```yaml
# Example Nav2 configuration for bipedal humanoid robot
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
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
    set_initial_pose: true
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_back_up_cancel_bt_node

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Bipedal-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.3
      vx_max: 0.3  # Reduced for bipedal stability
      vx_min: -0.1
      vy_max: 0.1
      wz_max: 0.5
      xy_goal_tolerance: 0.1
      yaw_goal_tolerance: 0.1
      state_reset_threshold: 0.5
      ctrl_freq: 20.0
      goal_checker: "simple_goal_checker"
      transform_tolerance: 0.1
      use_vel_controller: true
      motion_model: "DiffDrive"
      reference_state_multiplier: 1.0
      clock_skew_tolerance: 0.1

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: False
      rolling_window: true
      width: 6  # Reduced for bipedal robot
      height: 6
      resolution: 0.05  # High resolution for precise foot placement
      robot_radius: 0.3  # Account for robot's balance margin
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.5  # Larger for bipedal safety
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: False
      robot_radius: 0.3
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
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.5

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

## 6.3 Custom Controllers for Bipedal Locomotion

### 6.3.1 Bipedal Footstep Controller

```python
# Custom controller for bipedal humanoid navigation
import rclpy
from rclpy.node import Node
from nav2_msgs.action import FollowPath
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import numpy as np
import math
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle

class BipedalFootstepController(Node):
    def __init__(self):
        super().__init__('bipedal_footstep_controller')

        # Action server for following paths with footstep planning
        self._action_server = ActionServer(
            self,
            FollowPath,
            'bipedal_follow_path',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.footstep_pub = self.create_publisher(Path, '/footsteps', 10)

        # Robot parameters for bipedal locomotion
        self.step_size = 0.3  # Maximum step size in meters
        self.turn_threshold = 0.1  # Minimum angle for turning
        self.linear_speed = 0.1    # Walking speed in m/s
        self.angular_speed = 0.2   # Turning speed in rad/s

        # Balance and stability parameters
        self.balance_margin = 0.1  # Safety margin for balance
        self.zmp_threshold = 0.05  # Zero Moment Point threshold

        self.get_logger().info('Bipedal Footstep Controller initialized')

    def goal_callback(self, goal_request):
        """Accept or reject path following goal"""
        self.get_logger().info('Received path following request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject goal cancellation"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the path following action"""
        self.get_logger().info('Executing path following')

        feedback_msg = FollowPath.Feedback()
        result = FollowPath.Result()

        path = goal_handle.request.path
        current_pose = self.get_current_pose()

        # Generate footstep plan for the path
        footsteps = self.generate_footsteps(path, current_pose)

        # Publish footsteps for visualization
        self.publish_footsteps(footsteps)

        # Execute footsteps
        success = self.execute_footsteps(footsteps, goal_handle, feedback_msg)

        if success:
            goal_handle.succeed()
            result.error_code = FollowPath.Result.SUCCESS
        else:
            goal_handle.abort()
            result.error_code = FollowPath.Result.FAILURE

        return result

    def generate_footsteps(self, path, start_pose):
        """Generate footstep plan for bipedal locomotion"""
        footsteps = []

        # Start with current pose
        current_pose_2d = self.pose_to_2d(start_pose.pose)
        footsteps.append(current_pose_2d)

        # Process path to create discrete footsteps
        for i in range(len(path.poses) - 1):
            current_waypoint = self.pose_to_2d(path.poses[i].pose)
            next_waypoint = self.pose_to_2d(path.poses[i + 1].pose)

            # Calculate direction and distance
            dx = next_waypoint.x - current_pose_2d.x
            dy = next_waypoint.y - current_pose_2d.y
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)

            # Generate footsteps along the path
            num_steps = int(distance / self.step_size) + 1
            for j in range(num_steps):
                step_x = current_pose_2d.x + (dx * j / num_steps)
                step_y = current_pose_2d.y + (dy * j / num_steps)
                step_yaw = angle  # Maintain direction

                step_pose = PoseStamped()
                step_pose.header.frame_id = "map"
                step_pose.pose.position.x = step_x
                step_pose.pose.position.y = step_y
                step_pose.pose.position.z = 0.0

                # Convert angle to quaternion
                qw = math.cos(step_yaw * 0.5)
                qz = math.sin(step_yaw * 0.5)
                step_pose.pose.orientation.w = qw
                step_pose.pose.orientation.z = qz

                footsteps.append(step_pose)

        return footsteps

    def execute_footsteps(self, footsteps, goal_handle, feedback_msg):
        """Execute the generated footsteps"""
        for i, step in enumerate(footsteps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False

            # Move to the next footstep
            success = self.move_to_footstep(step)

            if not success:
                return False

            # Publish feedback
            feedback_msg.current_pose = step
            goal_handle.publish_feedback(feedback_msg)

            # Small delay between steps for bipedal stability
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.5))

        return True

    def move_to_footstep(self, target_pose):
        """Move robot to a specific footstep pose"""
        # This would implement the actual bipedal locomotion
        # For now, we'll simulate with simple movement commands

        # Calculate required movement
        current_pose = self.get_current_pose()
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        target_yaw = self.quaternion_to_yaw(target_pose.pose.orientation)
        current_yaw = self.quaternion_to_yaw(current_pose.pose.orientation)

        # Simple proportional controller for demonstration
        linear_vel = min(self.linear_speed, math.sqrt(dx*dx + dy*dy) * 0.5)
        angular_vel = (target_yaw - current_yaw) * 0.5

        # Limit angular velocity
        angular_vel = max(-self.angular_speed, min(self.angular_speed, angular_vel))

        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        self.cmd_vel_pub.publish(cmd_vel)

        # Wait for movement to complete (simplified)
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1.0))

        # Stop the robot
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

        return True

    def get_current_pose(self):
        """Get current robot pose (would use TF or odometry in practice)"""
        # Placeholder implementation
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def pose_to_2d(self, pose):
        """Convert 3D pose to 2D representation"""
        pose_2d = PoseStamped()
        pose_2d.pose.position.x = pose.position.x
        pose_2d.pose.position.y = pose.position.y
        pose_2d.pose.orientation = pose.orientation
        return pose_2d

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_footsteps(self, footsteps):
        """Publish footsteps for visualization"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        path_msg.poses = footsteps

        self.footstep_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BipedalFootstepController()

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

### 6.3.2 Balance-Aware Path Planner

```python
# Balance-aware path planning for bipedal robots
import numpy as np
import math
from scipy.spatial.distance import euclidean
from typing import List, Tuple

class BalanceAwarePlanner:
    def __init__(self, robot_width=0.3, step_length=0.3, max_lean=0.1):
        self.robot_width = robot_width
        self.step_length = step_length
        self.max_lean = max_lean  # Maximum acceptable lean angle
        self.support_polygon_margin = robot_width * 0.5

    def plan_balanced_path(self, start: Tuple[float, float, float],
                          goal: Tuple[float, float, float],
                          occupancy_grid) -> List[Tuple[float, float, float]]:
        """
        Plan a path that maintains balance for bipedal locomotion
        """
        path = []

        # Use A* with balance constraints
        open_set = [start]
        closed_set = set()
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        came_from = {}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if self.distance_2d(current, goal) < 0.1:  # Reached goal
                path = self.reconstruct_path(came_from, current)
                return path

            open_set.remove(current)
            closed_set.add(current)

            # Get valid neighbors considering balance
            neighbors = self.get_balanced_neighbors(current, occupancy_grid)

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + self.distance_2d(current, neighbor)

                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)

        return []  # No path found

    def get_balanced_neighbors(self, pose: Tuple[float, float, float],
                              occupancy_grid) -> List[Tuple[float, float, float]]:
        """
        Get valid neighboring poses that maintain balance
        """
        neighbors = []
        x, y, theta = pose

        # Generate potential steps in different directions
        step_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi,
                       -3*math.pi/4, -math.pi/2, -math.pi/4]

        for angle_offset in step_angles:
            # Calculate step position
            step_x = x + self.step_length * math.cos(theta + angle_offset)
            step_y = y + self.step_length * math.sin(theta + angle_offset)

            # Check if step is valid (not in obstacle, maintains balance)
            if self.is_balanced_step(pose, (step_x, step_y, theta), occupancy_grid):
                neighbors.append((step_x, step_y, theta))

        return neighbors

    def is_balanced_step(self, current_pose: Tuple[float, float, float],
                        next_pose: Tuple[float, float, float],
                        occupancy_grid) -> bool:
        """
        Check if a step maintains balance for bipedal locomotion
        """
        current_x, current_y, current_theta = current_pose
        next_x, next_y, next_theta = next_pose

        # Check if next position is free
        grid_x = int(next_x / occupancy_grid.resolution)
        grid_y = int(next_y / occupancy_grid.resolution)

        if (grid_x < 0 or grid_x >= occupancy_grid.width or
            grid_y < 0 or grid_y >= occupancy_grid.height):
            return False

        if occupancy_grid.get_probability(grid_x, grid_y) > 0.7:  # Occupied
            return False

        # Check balance constraints
        step_distance = self.distance_2d(current_pose, next_pose)
        if step_distance > self.step_length * 1.2:  # Too large a step
            return False

        # Check that the step maintains center of mass within support polygon
        # This is a simplified check - real implementation would be more complex
        return True

    def distance_2d(self, pose1: Tuple[float, float, float],
                   pose2: Tuple[float, float, float]) -> float:
        """Calculate 2D Euclidean distance between poses"""
        dx = pose1[0] - pose2[0]
        dy = pose1[1] - pose2[1]
        return math.sqrt(dx*dx + dy*dy)

    def heuristic(self, pose1: Tuple[float, float, float],
                 pose2: Tuple[float, float, float]) -> float:
        """Heuristic function for A*"""
        return self.distance_2d(pose1, pose2)

    def reconstruct_path(self, came_from: dict,
                        current: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path
```

## 6.4 Behavior Trees for Complex Navigation

### 6.4.1 Nav2 Behavior Tree Customization

```xml
<!-- navigate_with_bipedal_recovery.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <PipelineSequence name="NavigateWithReplanning">
            <RateController hz="1.0">
                <RecoveryNode number_of_retries="6" name="NavigateRecovery">
                    <PipelineSequence name="NavigateWithOdometry">
                        <ControllerSelector>
                            <Controller id="FollowPath"/>
                        </ControllerSelector>
                        <GoalCheckerSelector>
                            <GoalChecker id="SimpleGoalChecker"/>
                        </GoalCheckerSelector>
                        <ComputePathToPose>
                            <PlannerSelector>
                                <Planner id="GridBased"/>
                            </PlannerSelector>
                        </ComputePathToPose>
                        <FollowPath>
                            <Controller id="BipedalController"/>
                        </FollowPath>
                    </PipelineSequence>
                    <RecoveryActions>
                        <ClearEntirelyCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                        <ClearEntirelyCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
                    </RecoveryActions>
                </RecoveryNode>
            </RateController>
        </PipelineSequence>
    </BehaviorTree>
</root>
```

### 6.4.2 Custom Behavior Tree Nodes for Bipedal Navigation

```python
# Custom behavior tree nodes for bipedal navigation
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer
import math

class BipedalBalanceChecker(Node):
    def __init__(self):
        super().__init__('bipedal_balance_checker')

        # Subscribe to IMU data for balance checking
        self.imu_sub = self.create_subscription(
            String,  # In practice, this would be sensor_msgs/Imu
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for balance status
        self.balance_pub = self.create_publisher(String, '/balance_status', 10)

        self.balance_threshold = 0.2  # Maximum acceptable tilt
        self.current_balance_status = "BALANCED"

    def imu_callback(self, msg):
        """Process IMU data to check balance"""
        # Parse IMU message (simplified)
        # In practice, this would use sensor_msgs/Imu
        try:
            # Extract roll and pitch from IMU
            roll = msg.roll  # This would come from actual IMU data
            pitch = msg.pitch

            # Check if robot is within balance limits
            tilt_magnitude = math.sqrt(roll*roll + pitch*pitch)

            if tilt_magnitude > self.balance_threshold:
                self.current_balance_status = "UNBALANCED"
                self.publish_balance_warning()
            else:
                self.current_balance_status = "BALANCED"

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def is_balanced(self) -> bool:
        """Check if robot is currently balanced"""
        return self.current_balance_status == "BALANCED"

    def publish_balance_warning(self):
        """Publish balance warning for behavior tree"""
        status_msg = String()
        status_msg.data = "BALANCE_CRITICAL"
        self.balance_pub.publish(status_msg)

class BipedalStepPlanner(Node):
    def __init__(self):
        super().__init__('bipedal_step_planner')

        # Action server for step planning
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'bipedal_step_plan',
            self.execute_step_plan
        )

        # Robot-specific parameters
        self.step_length = 0.3  # Maximum step length
        self.step_width = 0.2   # Maximum step width
        self.max_climb = 0.1    # Maximum step-up height

    def execute_step_plan(self, goal_handle):
        """Execute step-by-step planning for bipedal navigation"""
        self.get_logger().info('Executing bipedal step plan')

        goal = goal_handle.request.pose

        # Plan steps from current position to goal
        step_plan = self.plan_steps_to_goal(goal)

        if not step_plan:
            goal_handle.abort()
            return NavigateToPose.Result()

        # Execute step plan
        success = self.execute_step_sequence(step_plan)

        if success:
            goal_handle.succeed()
            result = NavigateToPose.Result()
            result.error_code = NavigateToPose.Result.SUCCESS
        else:
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.error_code = NavigateToPose.Result.FAILURE

        return result

    def plan_steps_to_goal(self, goal):
        """Plan a sequence of steps to reach the goal"""
        # This would implement a more sophisticated step planning algorithm
        # considering foot placement, balance, and terrain constraints
        steps = []

        # Simplified step planning
        current_pos = self.get_current_position()
        dx = goal.pose.position.x - current_pos.x
        dy = goal.pose.position.y - current_pos.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate number of steps needed
        num_steps = int(distance / self.step_length) + 1

        for i in range(num_steps):
            step_x = current_pos.x + (dx * i / num_steps)
            step_y = current_pos.y + (dy * i / num_steps)

            step = {
                'position': (step_x, step_y),
                'type': 'walk'  # walk, step_up, step_down, turn
            }
            steps.append(step)

        return steps

    def execute_step_sequence(self, steps):
        """Execute a sequence of steps"""
        for step in steps:
            success = self.execute_single_step(step)
            if not success:
                return False
        return True

    def execute_single_step(self, step):
        """Execute a single step"""
        # This would send commands to the robot's walking controller
        # For now, we'll simulate the step execution
        self.get_logger().info(f'Executing step to {step["position"]}')

        # Simulate step execution time
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.5))

        return True

    def get_current_position(self):
        """Get current robot position"""
        # This would typically come from TF or odometry
        from geometry_msgs.msg import Point
        pos = Point()
        pos.x = 0.0
        pos.y = 0.0
        pos.z = 0.0
        return pos
```

## 6.5 Integration with Isaac Sim for Navigation Training

### 6.5.1 Isaac Sim Navigation Environment

```python
# Isaac Sim setup for navigation training
import omni
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np

def setup_bipedal_navigation_environment():
    """Setup Isaac Sim environment for bipedal navigation training"""

    # Initialize world
    world = World(stage_units_in_meters=1.0)
    world.reset()

    # Create ground plane
    ground_plane = world.scene.add_default_ground_plane()

    # Create a simple bipedal robot (simplified representation)
    robot = create_prim(
        prim_path="/World/Robot",
        prim_type="Xform",
        position=np.array([0, 0, 0.8]),  # Start slightly above ground
        orientation=np.array([0, 0, 0, 1]),
        scale=np.array([1.0, 1.0, 1.0])
    )

    # Add basic links for bipedal robot
    torso = create_prim(
        prim_path="/World/Robot/Torso",
        prim_type="Capsule",
        position=np.array([0, 0, 0.5]),
        scale=np.array([0.2, 0.2, 0.3])
    )

    left_leg = create_prim(
        prim_path="/World/Robot/LeftLeg",
        prim_type="Capsule",
        position=np.array([-0.1, 0, 0.2]),
        scale=np.array([0.08, 0.08, 0.4])
    )

    right_leg = create_prim(
        prim_path="/World/Robot/RightLeg",
        prim_type="Capsule",
        position=np.array([0.1, 0, 0.2]),
        scale=np.array([0.08, 0.08, 0.4])
    )

    # Add physics properties
    for link in [torso, left_leg, right_leg]:
        UsdPhysics.RigidBodyAPI.Apply(link)
        UsdPhysics.MassAPI.Apply(link)

    # Create obstacles for navigation
    obstacles = []
    for i in range(5):
        obstacle = create_prim(
            prim_path=f"/World/Obstacle{i}",
            prim_type="Cube",
            position=np.array([2 + i*1.5, np.random.uniform(-2, 2), 0.2]),
            scale=np.array([0.3, 0.3, 0.4])
        )
        obstacles.append(obstacle)

    # Add goal marker
    goal = create_prim(
        prim_path="/World/Goal",
        prim_type="Sphere",
        position=np.array([10, 0, 0.2]),
        scale=np.array([0.3, 0.3, 0.3])
    )

    # Configure camera view
    set_camera_view(eye=np.array([5, 5, 5]), target=np.array([0, 0, 0]))

    # Add sensors for navigation
    # RGB camera
    camera = create_prim(
        prim_path="/World/Robot/Camera",
        prim_type="Camera",
        position=np.array([0, 0, 0.8])
    )

    # LIDAR sensor
    lidar = create_prim(
        prim_path="/World/Robot/LIDAR",
        prim_type="Xform",
        position=np.array([0, 0, 0.8])
    )

    # Reset the world to apply changes
    world.reset()

    return world, robot, obstacles, goal

def run_navigation_training_episode(world, robot, goal_position):
    """Run a single navigation training episode"""

    # Get initial robot position
    initial_position = robot.get_world_pose()[0]

    # Simple navigation loop
    for step in range(1000):  # Max steps per episode
        # Get sensor data
        lidar_data = get_lidar_data()  # Placeholder
        camera_data = get_camera_data()  # Placeholder

        # Process navigation decision (simplified)
        action = simple_navigation_policy(initial_position, goal_position, lidar_data)

        # Apply action to robot
        apply_action_to_robot(robot, action)

        # Check termination conditions
        current_pos = robot.get_world_pose()[0]
        distance_to_goal = np.linalg.norm(current_pos - goal_position)

        if distance_to_goal < 0.5:  # Reached goal
            print("Goal reached!")
            break
        elif step == 999:  # Episode ended
            print("Episode ended without reaching goal")

        # Step the simulation
        world.step(render=True)

def simple_navigation_policy(robot_pos, goal_pos, lidar_data):
    """Simple navigation policy for training"""
    # Calculate direction to goal
    direction = goal_pos - robot_pos
    distance = np.linalg.norm(direction)

    if distance > 0.1:
        direction = direction / distance  # Normalize

        # Simple proportional controller
        linear_vel = min(0.5, distance * 0.5)  # Max 0.5 m/s
        angular_vel = 0.0  # Simplified turning

        return (linear_vel, angular_vel)
    else:
        return (0.0, 0.0)  # Stop when close to goal
```

## 6.6 Performance Optimization for Bipedal Navigation

### 6.6.1 Adaptive Path Planning

```python
class AdaptiveBipedalPlanner:
    def __init__(self):
        self.planning_frequency = 1.0  # Hz
        self.replanning_threshold = 0.5  # meters
        self.balance_margin = 0.1
        self.adaptation_factor = 1.0

    def adapt_planning_parameters(self, robot_state, environment_state):
        """Adapt planning parameters based on current conditions"""

        # Adjust planning frequency based on robot speed
        if robot_state.linear_velocity > 0.2:  # Moving fast
            self.planning_frequency = min(5.0, self.planning_frequency * 1.1)
        else:  # Moving slowly or stopped
            self.planning_frequency = max(0.5, self.planning_frequency * 0.9)

        # Adjust safety margins based on terrain
        if environment_state.rough_terrain:
            self.balance_margin = min(0.3, self.balance_margin * 1.1)
        else:
            self.balance_margin = max(0.1, self.balance_margin * 0.9)

        # Adjust replanning threshold based on obstacle density
        obstacle_density = self.calculate_obstacle_density(environment_state)
        self.replanning_threshold = 0.5 + (obstacle_density * 0.5)

    def calculate_obstacle_density(self, env_state):
        """Calculate local obstacle density around robot"""
        # This would analyze the local costmap to determine obstacle density
        # For now, return a simplified value
        return 0.3  # 30% obstacle density

    def is_replanning_needed(self, current_path, robot_pose):
        """Determine if path replanning is needed"""
        if not current_path:
            return True

        # Check if robot deviated significantly from path
        closest_point = self.find_closest_point_on_path(current_path, robot_pose)
        deviation = self.calculate_deviation(robot_pose, closest_point)

        return deviation > self.replanning_threshold

    def find_closest_point_on_path(self, path, robot_pose):
        """Find the closest point on the path to the robot"""
        min_dist = float('inf')
        closest_point = None

        for pose in path.poses:
            dist = self.calculate_2d_distance(robot_pose, pose.pose)
            if dist < min_dist:
                min_dist = dist
                closest_point = pose.pose

        return closest_point

    def calculate_deviation(self, robot_pose, path_pose):
        """Calculate deviation between robot and path"""
        return self.calculate_2d_distance(robot_pose.pose, path_pose)

    def calculate_2d_distance(self, pose1, pose2):
        """Calculate 2D Euclidean distance between poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)
```

### 6.6.2 Multi-Modal Navigation

```python
class MultiModalBipedalNavigation:
    def __init__(self):
        self.modes = {
            'walking': {'speed': 0.1, 'step_size': 0.3, 'balance_required': True},
            'stepping': {'speed': 0.05, 'step_size': 0.15, 'balance_required': True},
            'climbing': {'speed': 0.02, 'step_size': 0.1, 'balance_required': True},
            'turning': {'speed': 0.05, 'step_size': 0.0, 'balance_required': True}
        }
        self.current_mode = 'walking'

    def select_navigation_mode(self, path_segment, terrain_analysis):
        """Select appropriate navigation mode based on path and terrain"""

        # Analyze path segment characteristics
        path_slope = self.analyze_path_slope(path_segment)
        obstacle_height = self.analyze_obstacle_height(path_segment)
        turn_angle = self.analyze_turn_angle(path_segment)

        # Determine required mode
        if abs(path_slope) > 0.3:  # Steep incline/decline
            return 'climbing'
        elif obstacle_height > 0.1:  # Small obstacles
            return 'stepping'
        elif abs(turn_angle) > 0.5:  # Significant turn
            return 'turning'
        else:
            return 'walking'

    def analyze_path_slope(self, path_segment):
        """Analyze the slope of a path segment"""
        if len(path_segment) < 2:
            return 0.0

        start_z = path_segment[0].pose.position.z
        end_z = path_segment[-1].pose.position.z
        horizontal_distance = self.calculate_horizontal_distance(path_segment[0], path_segment[-1])

        if horizontal_distance == 0:
            return 0.0

        return (end_z - start_z) / horizontal_distance

    def analyze_obstacle_height(self, path_segment):
        """Analyze obstacle height along path"""
        # This would typically come from 3D perception data
        # For now, return a simplified analysis
        return 0.05  # Assume small obstacles

    def analyze_turn_angle(self, path_segment):
        """Analyze the turn angle required for path segment"""
        if len(path_segment) < 2:
            return 0.0

        # Calculate direction vectors
        start_to_mid = self.calculate_direction_vector(path_segment[0], path_segment[len(path_segment)//2])
        mid_to_end = self.calculate_direction_vector(path_segment[len(path_segment)//2], path_segment[-1])

        # Calculate angle between vectors
        dot_product = np.dot(start_to_mid, mid_to_end)
        norms = np.linalg.norm(start_to_mid) * np.linalg.norm(mid_to_end)

        if norms == 0:
            return 0.0

        angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
        return angle

    def calculate_direction_vector(self, pose1, pose2):
        """Calculate direction vector between two poses"""
        dx = pose2.pose.position.x - pose1.pose.position.x
        dy = pose2.pose.position.y - pose1.pose.position.y
        return np.array([dx, dy])

    def calculate_horizontal_distance(self, pose1, pose2):
        """Calculate horizontal distance between poses"""
        dx = pose2.pose.position.x - pose1.pose.position.x
        dy = pose2.pose.position.y - pose1.pose.position.y
        return math.sqrt(dx*dx + dy*dy)
```

## 6.7 Validation and Testing

### 6.7.1 Navigation Performance Metrics

```python
import numpy as np
import math
from typing import List, Tuple

class NavigationValidator:
    def __init__(self):
        self.metrics = {}

    def validate_navigation_performance(self, planned_path: List[Tuple[float, float]],
                                     executed_path: List[Tuple[float, float]],
                                     goal_position: Tuple[float, float]) -> dict:
        """Validate navigation performance with various metrics"""

        metrics = {}

        # Path efficiency
        if len(planned_path) > 1 and len(executed_path) > 1:
            planned_length = self.calculate_path_length(planned_path)
            executed_length = self.calculate_path_length(executed_path)
            metrics['path_efficiency'] = planned_length / max(executed_length, 0.001)

        # Goal accuracy
        final_position = executed_path[-1] if executed_path else (0, 0)
        goal_distance = math.sqrt((final_position[0] - goal_position[0])**2 +
                                 (final_position[1] - goal_position[1])**2)
        metrics['goal_accuracy'] = goal_distance

        # Success rate
        metrics['success'] = goal_distance < 0.5  # Within 0.5m of goal

        # Time to goal (if timestamps are available)
        # metrics['time_to_goal'] = end_time - start_time

        # Deviation from planned path
        if planned_path and executed_path:
            avg_deviation = self.calculate_average_deviation(planned_path, executed_path)
            metrics['average_deviation'] = avg_deviation

        # Number of replans (if tracked)
        # metrics['replans_count'] = replan_count

        self.metrics = metrics
        return metrics

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total length of a path"""
        if len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)

        return total_length

    def calculate_average_deviation(self, planned_path: List[Tuple[float, float]],
                                   executed_path: List[Tuple[float, float]]) -> float:
        """Calculate average deviation from planned path"""
        if not planned_path or not executed_path:
            return float('inf')

        total_deviation = 0.0
        for exec_point in executed_path:
            # Find closest point on planned path
            min_dist = float('inf')
            for plan_point in planned_path:
                dist = math.sqrt((exec_point[0] - plan_point[0])**2 +
                               (exec_point[1] - plan_point[1])**2)
                min_dist = min(min_dist, dist)

            total_deviation += min_dist

        return total_deviation / len(executed_path) if executed_path else 0.0

    def validate_balance_during_navigation(self, balance_data: List[float]) -> dict:
        """Validate robot balance during navigation"""
        if not balance_data:
            return {'balance_valid': False, 'avg_lean': float('inf')}

        avg_lean = np.mean(balance_data)
        max_lean = np.max(np.abs(balance_data))
        balance_valid = max_lean < 0.3  # Threshold for acceptable lean

        return {
            'balance_valid': balance_valid,
            'avg_lean': avg_lean,
            'max_lean': max_lean,
            'lean_std': np.std(balance_data)
        }
```

## 6.8 Best Practices for Bipedal Navigation

### 6.8.1 Configuration Guidelines

1. **Costmap Configuration**:
   - Use higher resolution (0.025-0.05m) for precise foot placement
   - Increase inflation radius for safety margins
   - Adjust robot radius to account for balance envelope

2. **Controller Tuning**:
   - Reduce maximum velocities for stability
   - Implement smooth acceleration/deceleration profiles
   - Use appropriate control frequencies (10-20 Hz)

3. **Path Planning**:
   - Consider step constraints in global planning
   - Implement frequent replanning for dynamic environments
   - Add balance checks during path execution

### 6.8.2 Safety Considerations

1. **Emergency Stop**: Implement immediate stop on balance loss
2. **Safe Homing**: Return to stable pose if navigation fails
3. **Terrain Assessment**: Avoid unstable or unsuitable terrain
4. **Human Safety**: Maintain safe distances from humans

## Summary

Nav2 provides a robust foundation for navigation, but requires specific adaptation for bipedal humanoid robots. Custom controllers, balance-aware planning, and specialized behavior trees enable safe and effective navigation for walking robots. The integration with Isaac Sim allows for comprehensive training and validation of navigation behaviors in simulated environments before deployment. Proper configuration of costmaps, controllers, and behavior trees is essential for achieving stable and reliable bipedal navigation. In the next chapter, we will explore behavior trees in greater detail and how they can be used for complex task planning in humanoid robots.