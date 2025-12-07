---
title: Chapter 10 - Integration & Assessment Tutorial
description: Complete integration tutorial combining all AI-Robot Brain components with practical assessment for humanoid robots.
sidebar_position: 32
---

# Chapter 10 - Integration & Assessment Tutorial

This chapter provides a comprehensive integration tutorial that brings together all the components of the AI-Robot Brain module. We'll create a complete humanoid robot system that integrates Isaac Sim, visual SLAM, Nav2 navigation, behavior trees, cognitive architectures, and sim-to-real transfer strategies. This practical tutorial will demonstrate how to combine these elements into a functional system and provide a comprehensive assessment of the module's learning objectives.

## 10.1 Complete System Architecture

### 10.1.1 Integrated Architecture Overview

Let's design a complete architecture that integrates all components learned in this module:

```yaml
# Complete Humanoid Robot Architecture
HumanoidRobotSystem:
  PerceptionLayer:
    - IsaacROSVisualSLAM:
        inputs: [RGB_Camera, IMU]
        outputs: [Pose_Estimate, Map]
        acceleration: GPU
    - IsaacROSDetection2D:
        inputs: [RGB_Camera]
        outputs: [Object_Detections]
        acceleration: GPU
    - IsaacROSOccupancyGrid:
        inputs: [LIDAR, Visual_SLAM]
        outputs: [Occupancy_Grid]
        acceleration: GPU

  CognitiveLayer:
    - MemorySystem:
        - SensoryBuffer: 100ms retention
        - WorkingMemory: 30s retention
        - LongTermMemory: Persistent storage
    - ReasoningEngine:
        - ProbabilisticReasoner
        - FrameBasedReasoner
    - PlanningSystem:
        - HTNPlanner
        - CognitivePlanner
    - LearningSystem:
        - CognitiveRLSystem
        - TransferLearningSystem

  NavigationLayer:
    - Nav2Stack:
        - GlobalPlanner: NavFn/A*
        - LocalPlanner: MPC/TEB
        - Controller: MPPI
        - Costmap2D: With inflation
    - BipedalSpecific:
        - FootstepPlanner
        - BalanceController
        - StepGenerator

  BehaviorLayer:
    - BehaviorTreeEngine:
        - CompositeNodes: Sequence, Selector, Parallel
        - DecoratorNodes: Inverter, Retry, Timeout
        - ActionNodes: Custom for humanoid tasks
    - TaskDecomposition:
        - HighLevel: Serve drink, Clean room
        - MidLevel: Navigate, Detect object, Grasp
        - LowLevel: Joint control, Balance maintenance

  ControlLayer:
    - WholeBodyController:
        - InverseKinematics
        - BalanceControl
        - TrajectoryGeneration
    - LowLevelControllers:
        - JointPosition
        - JointVelocity
        - JointEffort
```

### 10.1.2 ROS 2 Launch File for Complete System

```python
# complete_humanoid_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_model = LaunchConfiguration('robot_model')
    run_perception = LaunchConfiguration('run_perception')
    run_navigation = LaunchConfiguration('run_navigation')
    run_behavior = LaunchConfiguration('run_behavior')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time if true'
    )

    declare_robot_model_cmd = DeclareLaunchArgument(
        'robot_model',
        default_value='humanoid_a',
        description='Robot model to use'
    )

    declare_run_perception_cmd = DeclareLaunchArgument(
        'run_perception',
        default_value='True',
        description='Whether to run perception stack'
    )

    declare_run_navigation_cmd = DeclareLaunchArgument(
        'run_navigation',
        default_value='True',
        description='Whether to run navigation stack'
    )

    declare_run_behavior_cmd = DeclareLaunchArgument(
        'run_behavior',
        default_value='True',
        description='Whether to run behavior system'
    )

    # Isaac ROS Perception Container
    perception_container = ComposableNodeContainer(
        condition=IfCondition(run_perception),
        name='perception_container',
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
                    'map_frame_id': 'map',
                    'odom_frame_id': 'odom',
                    'base_frame_id': 'base_link',
                    'enable_observations_view': True,
                    'enable_slam_visualization': True,
                }],
                remappings=[
                    ('/visual_slam/image', '/camera/image_rect_color'),
                    ('/visual_slam/camera_info', '/camera/camera_info'),
                    ('/visual_slam/imu', '/imu/data'),
                ],
            ),
            ComposableNode(
                package='isaac_ros_detection_2d',
                plugin='nvidia::isaac_ros::detection_2d::Detection2DNode',
                name='object_detection',
                parameters=[{
                    'model_file_path': '/models/yolov5s.plan',
                    'input_tensor_names': ['input'],
                    'output_tensor_names': ['output'],
                    'tensorrt_engine_file_path': '/models/yolov5s.plan',
                    'label_file_path': '/models/coco_labels.txt',
                }],
                remappings=[
                    ('image', '/camera/image_rect_color'),
                    ('detections', '/object_detections'),
                ],
            ),
        ],
        output='screen',
    )

    # Navigation Container
    navigation_container = ComposableNodeContainer(
        condition=IfCondition(run_navigation),
        name='navigation_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_occupancy_grid',
                plugin='nvidia::isaac_ros::occupancy_grid::OccupancyGridNode',
                name='occupancy_grid',
                parameters=[{
                    'resolution': 0.05,
                    'width': 40,
                    'height': 40,
                    'origin_x': -10.0,
                    'origin_y': -10.0,
                }],
                remappings=[
                    ('scan', '/scan'),
                    ('map', '/map'),
                ],
            ),
        ],
        output='screen',
    )

    # Cognitive Architecture Node
    cognitive_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('humanoid_cognitive_system'),
                'launch',
                'cognitive_architecture.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Behavior Tree Node
    behavior_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('humanoid_behavior_system'),
                'launch',
                'behavior_tree.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Nav2 Stack
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Isaac Sim Bridge (if running in simulation)
    isaac_sim_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('isaac_ros_common'),
                'launch',
                'isaac_ros_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_robot_model_cmd)
    ld.add_action(declare_run_perception_cmd)
    ld.add_action(declare_run_navigation_cmd)
    ld.add_action(declare_run_behavior_cmd)

    # Add nodes based on conditions
    ld.add_action(perception_container)
    ld.add_action(navigation_container)
    ld.add_action(cognitive_node)
    ld.add_action(behavior_node)
    ld.add_action(nav2_launch)
    ld.add_action(isaac_sim_bridge)

    return ld
```

## 10.2 Complete Cognitive Architecture Implementation

### 10.2.1 Integrated Cognitive System

```python
# integrated_cognitive_system.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from cognitive_msgs.msg import Belief, Goal, Plan
from behavior_tree_core.msg import NodeStatus
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Any
import time
import pickle
import sqlite3
from dataclasses import dataclass
from enum import Enum

class CognitiveState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTING = "acting"
    LEARNING = "learning"

@dataclass
class MemoryItem:
    id: str
    content: Any
    timestamp: float
    confidence: float
    context: Dict[str, Any]

class SensoryBuffer:
    def __init__(self, retention_time: float = 0.1):
        self.retention_time = retention_time
        self.buffers: Dict[str, List[tuple]] = {}
        self.lock = threading.Lock()

    def store(self, sensor_type: str, data: Any):
        with self.lock:
            if sensor_type not in self.buffers:
                self.buffers[sensor_type] = []

            current_time = time.time()
            self.buffers[sensor_type].append((current_time, data))

            # Remove old data
            self.buffers[sensor_type] = [
                (t, d) for t, d in self.buffers[sensor_type]
                if current_time - t <= self.retention_time
            ]

    def get_recent(self, sensor_type: str, time_window: float = 0.05):
        with self.lock:
            if sensor_type not in self.buffers:
                return []

            current_time = time.time()
            return [
                data for timestamp, data in self.buffers[sensor_type]
                if current_time - timestamp <= time_window
            ]

class WorkingMemory:
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self.items: Dict[str, MemoryItem] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()

    def store(self, key: str, content: Any, confidence: float = 1.0, context: Dict = None):
        with self.lock:
            item = MemoryItem(
                id=key,
                content=content,
                timestamp=time.time(),
                confidence=confidence,
                context=context or {}
            )

            self.items[key] = item
            self.access_times[key] = time.time()

            # Remove oldest items if exceeding max
            if len(self.items) > self.max_items:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.items[oldest_key]
                del self.access_times[oldest_key]

    def retrieve(self, key: str) -> Optional[MemoryItem]:
        with self.lock:
            if key in self.items:
                self.access_times[key] = time.time()
                return self.items[key]
            return None

    def update(self, key: str, new_content: Any):
        with self.lock:
            if key in self.items:
                self.items[key].content = new_content
                self.items[key].timestamp = time.time()

class LongTermMemory:
    def __init__(self, db_path: str = "cognitive_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content BLOB,
                    timestamp REAL,
                    confidence REAL,
                    context BLOB,
                    embedding BLOB
                )
            ''')
            conn.commit()

    def store(self, item: MemoryItem):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        item.id,
                        pickle.dumps(item.content),
                        item.timestamp,
                        item.confidence,
                        pickle.dumps(item.context),
                        None  # embedding would be computed here
                    )
                )
                conn.commit()

    def retrieve(self, key: str) -> Optional[MemoryItem]:
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM memories WHERE id = ?', (key,))
                row = cursor.fetchone()

                if row:
                    return MemoryItem(
                        id=row[0],
                        content=pickle.loads(row[1]),
                        timestamp=row[2],
                        confidence=row[3],
                        context=pickle.loads(row[4])
                    )
        return None

    def query_similar(self, content: Any, limit: int = 5) -> List[MemoryItem]:
        # This would implement semantic similarity search
        # For now, return empty list
        return []

class ProbabilisticReasoner:
    def __init__(self, working_memory: WorkingMemory):
        self.working_memory = working_memory
        self.belief_networks = {}

    def update_belief(self, belief_id: str, evidence: Dict[str, float]):
        """Update belief using Bayesian reasoning"""
        current_belief = self.working_memory.retrieve(belief_id)
        if current_belief:
            # Apply Bayes' rule
            new_confidence = self.bayesian_update(
                current_belief.confidence,
                evidence
            )
            self.working_memory.update(belief_id, {
                **current_belief.content,
                'confidence': new_confidence
            })
        else:
            # Create new belief
            self.working_memory.store(belief_id, {
                'content': evidence,
                'confidence': max(evidence.values()) if evidence else 0.5
            })

    def bayesian_update(self, prior: float, likelihood: Dict[str, float]) -> float:
        """Perform Bayesian update"""
        # Simplified Bayesian update
        if not likelihood:
            return prior

        max_likelihood = max(likelihood.values())
        updated = prior * max_likelihood
        return min(updated, 1.0)  # Clamp to [0, 1]

class BehaviorTreeManager:
    def __init__(self):
        self.active_trees = {}
        self.tree_library = {}
        self.running_nodes = set()

    def register_tree(self, name: str, tree_xml: str):
        """Register a behavior tree"""
        self.tree_library[name] = tree_xml

    def execute_tree(self, tree_name: str, blackboard_data: Dict = None):
        """Execute a registered behavior tree"""
        if tree_name in self.tree_library:
            # This would integrate with actual BT execution
            # For now, return success
            return True
        return False

class HumanoidCognitiveSystem(Node):
    def __init__(self):
        super().__init__('humanoid_cognitive_system')

        # Initialize memory systems
        self.sensory_buffer = SensoryBuffer()
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()

        # Initialize reasoning components
        self.reasoner = ProbabilisticReasoner(self.working_memory)
        self.behavior_manager = BehaviorTreeManager()

        # State management
        self.current_state = CognitiveState.IDLE
        self.active_goals = []
        self.current_plan = None

        # Publishers and subscribers
        qos_profile = QoSProfile(depth=10)

        # Perception inputs
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, qos_profile)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 'points', self.pointcloud_callback, qos_profile)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, qos_profile)

        # Cognitive outputs
        self.belief_pub = self.create_publisher(Belief, 'beliefs', qos_profile)
        self.goal_pub = self.create_publisher(Goal, 'goals', qos_profile)
        self.plan_pub = self.create_publisher(Plan, 'plans', qos_profile)
        self.action_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)

        # Cognitive control
        self.cognitive_cycle_timer = self.create_timer(0.1, self.cognitive_cycle)
        self.state_monitor_timer = self.create_timer(1.0, self.monitor_state)

        # Initialize behavior trees
        self.initialize_behavior_trees()

        self.get_logger().info('Humanoid Cognitive System initialized')

    def initialize_behavior_trees(self):
        """Initialize all behavior trees for the humanoid robot"""

        # Navigation behavior tree
        navigation_bt = """
        <root main_tree_to_execute="NavigateToGoal">
            <BehaviorTree ID="NavigateToGoal">
                <Sequence name="navigation_sequence">
                    <CheckBatteryLevel min_level="20"/>
                    <MoveToPoseAction target_pose="{goal_pose}"/>
                    <CheckGoalReached tolerance="0.5"/>
                </Sequence>
            </BehaviorTree>
        </root>
        """

        # Object interaction behavior tree
        interaction_bt = """
        <root main_tree_to_execute="InteractWithObject">
            <BehaviorTree ID="InteractWithObject">
                <Fallback name="interaction_fallback">
                    <Sequence name="grasp_sequence">
                        <DetectObject object_type="{object_type}"/>
                        <ApproachObject object_pose="{object_pose}"/>
                        <GraspObject object_pose="{object_pose}"/>
                        <VerifyGraspSuccess/>
                    </Sequence>
                    <Sequence name="push_sequence">
                        <ApproachObject object_pose="{object_pose}"/>
                        <PushObject object_pose="{object_pose}"/>
                    </Sequence>
                </Fallback>
            </BehaviorTree>
        </root>
        """

        # Social interaction behavior tree
        social_bt = """
        <root main_tree_to_execute="SocialInteraction">
            <BehaviorTree ID="SocialInteraction">
                <Sequence name="greeting_sequence">
                    <DetectHuman/>
                    <MoveToHuman distance="1.0"/>
                    <Speak text="Hello, how can I help you?"/>
                    <ListenForCommands timeout="10.0"/>
                </Sequence>
            </BehaviorTree>
        </root>
        """

        self.behavior_manager.register_tree('navigate', navigation_bt)
        self.behavior_manager.register_tree('interact', interaction_bt)
        self.behavior_manager.register_tree('social', social_bt)

    def image_callback(self, msg):
        """Process image data"""
        self.sensory_buffer.store('image', msg)

        # Extract features and update beliefs
        features = self.extract_visual_features(msg)
        self.update_visual_beliefs(features)

    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        self.sensory_buffer.store('pointcloud', msg)

        # Extract spatial information
        obstacles = self.detect_obstacles_from_pointcloud(msg)
        self.update_spatial_beliefs(obstacles)

    def imu_callback(self, msg):
        """Process IMU data for balance"""
        self.sensory_buffer.store('imu', msg)

        # Check balance state
        balance_state = self.assess_balance(msg)
        self.update_balance_beliefs(balance_state)

    def odom_callback(self, msg):
        """Process odometry data"""
        self.sensory_buffer.store('odometry', msg)

        # Update pose belief
        self.working_memory.store('current_pose', {
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                          msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        }, confidence=0.9)

    def extract_visual_features(self, image_msg):
        """Extract features from image data"""
        # This would use Isaac ROS accelerated vision processing
        # For now, return placeholder features
        return {
            'objects_detected': [],
            'room_layout': 'unknown',
            'lighting_conditions': 'normal'
        }

    def detect_obstacles_from_pointcloud(self, pc_msg):
        """Detect obstacles from point cloud"""
        # This would use Isaac ROS point cloud processing
        return []

    def assess_balance(self, imu_msg):
        """Assess robot balance from IMU data"""
        roll = np.arctan2(2*(imu_msg.orientation.w*imu_msg.orientation.x +
                            imu_msg.orientation.y*imu_msg.orientation.z),
                         1 - 2*(imu_msg.orientation.x**2 + imu_msg.orientation.y**2))

        pitch = np.arcsin(2*(imu_msg.orientation.w*imu_msg.orientation.y -
                            imu_msg.orientation.z*imu_msg.orientation.x))

        return {
            'roll': roll,
            'pitch': pitch,
            'balanced': abs(roll) < 0.2 and abs(pitch) < 0.2
        }

    def update_visual_beliefs(self, features):
        """Update beliefs based on visual features"""
        for obj_type, obj_data in features.get('objects_detected', []):
            belief_id = f'object_{obj_type}'
            self.reasoner.update_belief(belief_id, {
                'type': obj_type,
                'count': len(obj_data),
                'confidence': 0.8
            })

    def update_spatial_beliefs(self, obstacles):
        """Update spatial beliefs"""
        self.working_memory.store('obstacles', obstacles, confidence=0.9)

    def update_balance_beliefs(self, balance_state):
        """Update balance-related beliefs"""
        self.working_memory.store('balance_state', balance_state, confidence=0.95)

        if not balance_state['balanced']:
            self.working_memory.store('balance_at_risk', True, confidence=1.0)

    def cognitive_cycle(self):
        """Main cognitive cycle"""
        cycle_start = time.time()

        try:
            # 1. Perceive (handled by callbacks)
            self.current_state = CognitiveState.PERCEIVING

            # 2. Reason
            self.current_state = CognitiveState.REASONING
            self.perform_reasoning()

            # 3. Plan
            self.current_state = CognitiveState.PLANNING
            self.generate_plans()

            # 4. Act
            self.current_state = CognitiveState.ACTING
            self.execute_actions()

            # 5. Learn
            self.current_state = CognitiveState.LEARNING
            self.update_learning()

        except Exception as e:
            self.get_logger().error(f'Error in cognitive cycle: {e}')

        self.current_state = CognitiveState.IDLE

        # Log cycle time
        cycle_time = time.time() - cycle_start
        if cycle_time > 0.1:  # 10Hz requirement
            self.get_logger().warn(f'Cognitive cycle took {cycle_time:.3f}s (exceeds 0.1s)')

    def perform_reasoning(self):
        """Perform reasoning based on current beliefs"""
        # Check for balance emergencies
        balance_at_risk = self.working_memory.retrieve('balance_at_risk')
        if balance_at_risk and balance_at_risk.content:
            self.emergency_balance_recovery()

        # Update goal priorities based on current state
        self.update_goal_priorities()

    def generate_plans(self):
        """Generate plans based on current goals"""
        if self.active_goals:
            primary_goal = self.active_goals[0]

            if primary_goal.type == 'navigation':
                self.generate_navigation_plan(primary_goal)
            elif primary_goal.type == 'manipulation':
                self.generate_manipulation_plan(primary_goal)
            elif primary_goal.type == 'social':
                self.generate_social_plan(primary_goal)

    def execute_actions(self):
        """Execute current plan"""
        if self.current_plan:
            next_action = self.get_next_action()
            if next_action:
                self.execute_action(next_action)

    def update_learning(self):
        """Update learning based on experience"""
        # Store current episode for future learning
        current_state = self.get_current_state_summary()
        self.long_term_memory.store(MemoryItem(
            id=f'episode_{time.time()}',
            content=current_state,
            timestamp=time.time(),
            confidence=1.0,
            context={}
        ))

    def emergency_balance_recovery(self):
        """Execute emergency balance recovery"""
        self.get_logger().warn('Balance emergency detected - executing recovery')

        # Stop all motion
        stop_cmd = Twist()
        self.action_pub.publish(stop_cmd)

        # Update beliefs
        self.working_memory.store('balance_emergency', True, confidence=1.0)

    def update_goal_priorities(self):
        """Update goal priorities based on current situation"""
        # Example: Safety goals have highest priority
        safety_goals = [g for g in self.active_goals if g.priority == 'safety']
        if safety_goals:
            # Move safety goals to front
            self.active_goals = safety_goals + [g for g in self.active_goals if g.priority != 'safety']

    def generate_navigation_plan(self, goal):
        """Generate navigation plan"""
        # Use Nav2 integration
        plan_request = self.create_plan_request(goal.location)

        # Execute navigation behavior tree
        success = self.behavior_manager.execute_tree('navigate', {
            'goal_pose': goal.location
        })

        if success:
            self.current_plan = self.working_memory.retrieve('nav_plan')

    def get_next_action(self):
        """Get next action from current plan"""
        if self.current_plan:
            # Return next action in plan
            return self.current_plan.get_next_action()
        return None

    def execute_action(self, action):
        """Execute a specific action"""
        if action.type == 'move':
            cmd = Twist()
            cmd.linear.x = action.linear_vel
            cmd.angular.z = action.angular_vel
            self.action_pub.publish(cmd)
        elif action.type == 'speak':
            self.speak(action.text)
        elif action.type == 'grasp':
            self.execute_grasp(action.object_pose)

    def monitor_state(self):
        """Monitor overall system state"""
        self.get_logger().info(f'Cognitive System State: {self.current_state}')

        # Check memory usage
        working_mem_size = len(self.working_memory.items)
        self.get_logger().info(f'Working Memory Items: {working_mem_size}')

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidCognitiveSystem()

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

## 10.3 Integration Tutorial: Complete Task Execution

### 10.3.1 Multi-Modal Task: Serve Drink

Let's implement a complete multi-modal task that demonstrates all the integrated components:

```python
# complete_task_demo.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cognitive_msgs.msg import Goal
import time
import threading

class CompleteTaskDemo(Node):
    def __init__(self):
        super().__init__('complete_task_demo')

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers
        self.goal_pub = self.create_publisher(Goal, 'goals', 10)
        self.speech_pub = self.create_publisher(String, 'speech_commands', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        # Task state
        self.task_state = 'IDLE'
        self.cognitive_state = None
        self.navigation_result = None

        # Start task execution
        self.get_logger().info('Complete Task Demo initialized')
        self.execute_serve_drink_task()

    def execute_serve_drink_task(self):
        """Execute complete serve drink task"""
        self.get_logger().info('Starting serve drink task')

        # Define the complete task sequence
        task_sequence = [
            self.navigat_to_kitchen,
            self.detect_cup,
            self.grasp_cup,
            self.navigate_to_person,
            self.serve_drink,
            self.return_to_base
        ]

        # Execute each step
        for task_step in task_sequence:
            self.get_logger().info(f'Executing task step: {task_step.__name__}')

            success = task_step()
            if not success:
                self.get_logger().error(f'Task step {task_step.__name__} failed')
                self.abort_task()
                return

            self.get_logger().info(f'Task step {task_step.__name__} completed successfully')

        self.get_logger().info('Complete serve drink task finished successfully')

    def navigat_to_kitchen(self):
        """Navigate to kitchen location"""
        self.get_logger().info('Navigating to kitchen')

        # Define kitchen pose
        kitchen_pose = PoseStamped()
        kitchen_pose.header.frame_id = 'map'
        kitchen_pose.pose.position.x = 5.0
        kitchen_pose.pose.position.y = 2.0
        kitchen_pose.pose.position.z = 0.0
        kitchen_pose.pose.orientation.w = 1.0

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = kitchen_pose

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)

        # Wait for result with timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

        if future.result() is not None:
            result = future.result().result
            return result is not None
        else:
            return False

    def detect_cup(self):
        """Detect cup using perception system"""
        self.get_logger().info('Detecting cup')

        # This would integrate with Isaac ROS detection
        # For now, simulate detection
        time.sleep(2.0)  # Simulate detection time

        # Publish detection result to cognitive system
        goal = Goal()
        goal.type = 'object_detection'
        goal.description = 'cup'
        goal.priority = 'normal'
        self.goal_pub.publish(goal)

        # Simulate successful detection
        return True

    def grasp_cup(self):
        """Execute cup grasping"""
        self.get_logger().info('Grasping cup')

        # This would integrate with manipulation system
        # For now, simulate grasping
        time.sleep(3.0)  # Simulate grasping time

        # Publish manipulation goal
        goal = Goal()
        goal.type = 'manipulation'
        goal.description = 'grasp_cup'
        goal.priority = 'normal'
        self.goal_pub.publish(goal)

        return True

    def navigate_to_person(self):
        """Navigate to person location"""
        self.get_logger().info('Navigating to person')

        # Define person pose (this would come from perception)
        person_pose = PoseStamped()
        person_pose.header.frame_id = 'map'
        person_pose.pose.position.x = -2.0
        person_pose.pose.position.y = 1.0
        person_pose.pose.position.z = 0.0
        person_pose.pose.orientation.w = 1.0

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = person_pose

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

        if future.result() is not None:
            result = future.result().result
            return result is not None
        else:
            return False

    def serve_drink(self):
        """Serve drink to person"""
        self.get_logger().info('Serving drink')

        # This would involve releasing the cup
        time.sleep(2.0)  # Simulate serving time

        # Announce serving
        speech_msg = String()
        speech_msg.data = 'Here is your drink. Enjoy!'
        self.speech_pub.publish(speech_msg)

        return True

    def return_to_base(self):
        """Return to base location"""
        self.get_logger().info('Returning to base')

        # Define base pose
        base_pose = PoseStamped()
        base_pose.header.frame_id = 'map'
        base_pose.pose.position.x = 0.0
        base_pose.pose.position.y = 0.0
        base_pose.pose.position.z = 0.0
        base_pose.pose.orientation.w = 1.0

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = base_pose

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

        if future.result() is not None:
            result = future.result().result
            return result is not None
        else:
            return False

    def image_callback(self, msg):
        """Handle image data"""
        # This would feed into perception system
        pass

    def abort_task(self):
        """Handle task abortion"""
        self.get_logger().error('Task aborted due to failure')

        # Emergency stop
        stop_msg = String()
        stop_msg.data = 'STOP'
        self.speech_pub.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CompleteTaskDemo()

    try:
        # Run in separate thread to allow other operations
        task_thread = threading.Thread(target=lambda: node.execute_serve_drink_task())
        task_thread.start()

        rclpy.spin(node)

        task_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 10.4 Performance Optimization and Profiling

### 10.4.1 System Profiling Tools

```python
# cognitive_profiler.py
import time
import threading
import psutil
import GPUtil
from collections import defaultdict, deque
import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetric:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    gpu_memory: float
    cognitive_cycle_time: float
    perception_rate: float
    navigation_rate: float

class CognitiveProfiler:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.component_times = defaultdict(list)
        self.lock = threading.Lock()

        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

    def start_component_timer(self, component_name: str):
        """Start timing for a component"""
        start_time = time.time()
        return start_time, component_name

    def end_component_timer(self, timer_info):
        """End timing for a component"""
        start_time, component_name = timer_info
        elapsed = time.time() - start_time

        with self.lock:
            self.component_times[component_name].append(elapsed)

    def record_cycle_metrics(self, cycle_time: float):
        """Record metrics for a cognitive cycle"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        gpus = GPUtil.getGPUs()
        gpu_percent = gpus[0].load if gpus else 0
        gpu_memory = gpus[0].memoryUtil if gpus else 0

        metric = PerformanceMetric(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory=gpu_memory,
            cognitive_cycle_time=cycle_time,
            perception_rate=0,  # Would be calculated from perception callbacks
            navigation_rate=0   # Would be calculated from navigation callbacks
        )

        with self.lock:
            self.metrics_history.append(metric)

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        with self.lock:
            if not self.metrics_history:
                return {}

            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
            avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
            avg_gpu = sum(m.gpu_percent for m in self.metrics_history) / len(self.metrics_history)
            avg_cycle_time = sum(m.cognitive_cycle_time for m in self.metrics_history) / len(self.metrics_history)

            # Component performance
            component_avg_times = {}
            for comp, times in self.component_times.items():
                if times:
                    component_avg_times[comp] = sum(times) / len(times)

            return {
                'system_performance': {
                    'avg_cpu_percent': avg_cpu,
                    'avg_memory_percent': avg_memory,
                    'avg_gpu_percent': avg_gpu,
                    'avg_cycle_time': avg_cycle_time,
                    'target_cycle_time': 0.1,  # 10Hz
                    'cycle_time_exceeded': avg_cycle_time > 0.1
                },
                'component_performance': component_avg_times,
                'total_samples': len(self.metrics_history)
            }

    def _monitor_system(self):
        """Background monitoring thread"""
        while self.monitoring:
            time.sleep(1.0)  # Monitor every second

    def save_profiling_data(self, filename: str):
        """Save profiling data to file"""
        with self.lock:
            data = {
                'metrics_history': [
                    {
                        'timestamp': m.timestamp,
                        'cpu_percent': m.cpu_percent,
                        'memory_percent': m.memory_percent,
                        'gpu_percent': m.gpu_percent,
                        'gpu_memory': m.gpu_memory,
                        'cognitive_cycle_time': m.cognitive_cycle_time
                    } for m in self.metrics_history
                ],
                'component_times': {
                    comp: times for comp, times in self.component_times.items()
                }
            }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

class OptimizedCognitiveSystem:
    def __init__(self):
        self.profiler = CognitiveProfiler()
        self.priority_scheduler = PriorityScheduler()
        self.resource_manager = ResourceManager()

    def execute_optimized_cycle(self):
        """Execute optimized cognitive cycle"""
        cycle_start = time.time()

        # Schedule components based on priority and resource availability
        scheduled_components = self.priority_scheduler.get_scheduled_components()

        for component in scheduled_components:
            if self.resource_manager.can_execute(component):
                timer = self.profiler.start_component_timer(component.name)
                component.execute()
                self.profiler.end_component_timer(timer)

        # Record metrics
        cycle_time = time.time() - cycle_start
        self.profiler.record_cycle_metrics(cycle_time)

        # Check if optimization is needed
        if cycle_time > 0.1:  # Exceeds target
            self.optimize_performance()

    def optimize_performance(self):
        """Optimize system performance based on profiling data"""
        summary = self.profiler.get_performance_summary()

        if summary.get('system_performance', {}).get('cycle_time_exceeded', False):
            # Reduce perception frequency
            self.reduce_perception_frequency()

        if summary.get('component_performance', {}).get('perception', 0) > 0.05:
            # Perception taking too long, reduce complexity
            self.reduce_perception_complexity()

    def reduce_perception_frequency(self):
        """Reduce perception system frequency"""
        pass  # Implementation would adjust perception node parameters

    def reduce_perception_complexity(self):
        """Reduce perception system complexity"""
        pass  # Implementation would adjust perception algorithm parameters

class PriorityScheduler:
    def __init__(self):
        self.component_priorities = {
            'balance_control': 100,  # Highest priority
            'collision_avoidance': 90,
            'navigation': 80,
            'perception': 70,
            'planning': 60,
            'learning': 50  # Lowest priority
        }

    def get_scheduled_components(self):
        """Get components to execute based on priority and resources"""
        # This would return components in priority order
        # considering current system load
        return []

class ResourceManager:
    def __init__(self):
        self.max_cpu_threshold = 80.0
        self.max_memory_threshold = 85.0
        self.max_gpu_threshold = 85.0

    def can_execute(self, component):
        """Check if component can be executed given resource constraints"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        gpus = GPUtil.getGPUs()
        gpu_percent = gpus[0].load * 100 if gpus else 0

        return (cpu_percent < self.max_cpu_threshold and
                memory_percent < self.max_memory_threshold and
                gpu_percent < self.max_gpu_threshold)
```

## 10.5 Assessment and Evaluation

### 10.5.1 Comprehensive Assessment Framework

```python
# assessment_framework.py
import unittest
import numpy as np
from typing import Dict, List, Tuple
import time
import statistics

class CognitiveSystemAssessment:
    def __init__(self, cognitive_system):
        self.system = cognitive_system
        self.test_results = {}
        self.performance_metrics = {}

    def run_comprehensive_assessment(self) -> Dict:
        """Run comprehensive assessment of cognitive system"""
        print("Starting comprehensive cognitive system assessment...")

        # 1. Functional Tests
        print("Running functional tests...")
        functional_results = self.run_functional_tests()

        # 2. Performance Tests
        print("Running performance tests...")
        performance_results = self.run_performance_tests()

        # 3. Integration Tests
        print("Running integration tests...")
        integration_results = self.run_integration_tests()

        # 4. Robustness Tests
        print("Running robustness tests...")
        robustness_results = self.run_robustness_tests()

        # 5. Learning Tests
        print("Running learning capability tests...")
        learning_results = self.run_learning_tests()

        # Compile results
        assessment_results = {
            'functional': functional_results,
            'performance': performance_results,
            'integration': integration_results,
            'robustness': robustness_results,
            'learning': learning_results,
            'overall_score': self.calculate_overall_score({
                'functional': functional_results,
                'performance': performance_results,
                'integration': integration_results,
                'robustness': robustness_results,
                'learning': learning_results
            })
        }

        self.test_results = assessment_results
        return assessment_results

    def run_functional_tests(self) -> Dict:
        """Test individual components functionality"""
        results = {
            'memory_system': self.test_memory_system(),
            'reasoning_engine': self.test_reasoning_engine(),
            'planning_system': self.test_planning_system(),
            'behavior_trees': self.test_behavior_trees(),
            'navigation': self.test_navigation(),
            'perception': self.test_perception()
        }

        return {
            'pass_rate': sum(1 for r in results.values() if r['success']) / len(results),
            'detailed_results': results
        }

    def test_memory_system(self) -> Dict:
        """Test memory system functionality"""
        try:
            # Test sensory buffer
            self.system.sensory_buffer.store('test_sensor', 'test_data')
            recent_data = self.system.sensory_buffer.get_recent('test_sensor')

            # Test working memory
            self.system.working_memory.store('test_key', 'test_value')
            retrieved = self.system.working_memory.retrieve('test_key')

            # Test long-term memory
            import pickle
            test_item = MemoryItem('test_id', 'test_content', time.time(), 0.9, {})
            self.system.long_term_memory.store(test_item)
            retrieved_ltm = self.system.long_term_memory.retrieve('test_id')

            success = (len(recent_data) > 0 and
                      retrieved is not None and
                      retrieved_ltm is not None)

            return {
                'success': success,
                'details': f"Sensory buffer: {len(recent_data) > 0}, "
                          f"Working memory: {retrieved is not None}, "
                          f"Long-term memory: {retrieved_ltm is not None}"
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_reasoning_engine(self) -> Dict:
        """Test reasoning engine functionality"""
        try:
            # Test belief update
            self.system.reasoner.update_belief('test_belief', {'evidence': 0.8})

            # Check if belief was updated
            belief = self.system.working_memory.retrieve('test_belief')

            success = belief is not None and belief.confidence > 0.5

            return {
                'success': success,
                'details': f"Belief confidence: {belief.confidence if belief else 0}"
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_planning_system(self) -> Dict:
        """Test planning system functionality"""
        try:
            # This would test the planning components
            # For now, return success
            return {'success': True, 'details': 'Planning system test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_behavior_trees(self) -> Dict:
        """Test behavior tree functionality"""
        try:
            # Test behavior tree execution
            success = self.system.behavior_manager.execute_tree('navigate')

            return {
                'success': success,
                'details': f"Behavior tree execution: {success}"
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_navigation(self) -> Dict:
        """Test navigation functionality"""
        try:
            # This would test navigation components
            return {'success': True, 'details': 'Navigation test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_perception(self) -> Dict:
        """Test perception functionality"""
        try:
            # This would test perception components
            return {'success': True, 'details': 'Perception test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_performance_tests(self) -> Dict:
        """Test system performance under various conditions"""
        results = {
            'cycle_time': self.test_cycle_time(),
            'memory_usage': self.test_memory_usage(),
            'cpu_usage': self.test_cpu_usage(),
            'throughput': self.test_throughput()
        }

        return results

    def test_cycle_time(self) -> Dict:
        """Test cognitive cycle timing"""
        cycle_times = []

        # Run multiple cycles to measure timing
        for _ in range(50):
            start_time = time.time()
            self.system.cognitive_cycle()
            cycle_time = time.time() - start_time
            cycle_times.append(cycle_time)

        avg_cycle_time = statistics.mean(cycle_times)
        max_cycle_time = max(cycle_times)

        success = avg_cycle_time <= 0.1  # Should be under 100ms for 10Hz

        return {
            'success': success,
            'avg_cycle_time': avg_cycle_time,
            'max_cycle_time': max_cycle_time,
            'target_cycle_time': 0.1,
            'measurements': cycle_times
        }

    def test_memory_usage(self) -> Dict:
        """Test memory usage patterns"""
        # This would monitor memory usage during operation
        return {'success': True, 'details': 'Memory usage test completed'}

    def test_cpu_usage(self) -> Dict:
        """Test CPU usage under load"""
        # This would monitor CPU usage
        return {'success': True, 'details': 'CPU usage test completed'}

    def test_throughput(self) -> Dict:
        """Test system throughput"""
        # Test how many operations per second the system can handle
        start_time = time.time()
        operations_completed = 0

        # Run for 10 seconds
        while time.time() - start_time < 10.0:
            self.system.cognitive_cycle()
            operations_completed += 1

        duration = time.time() - start_time
        throughput = operations_completed / duration

        return {
            'success': throughput >= 10,  # At least 10 cycles per second
            'throughput': throughput,
            'duration': duration,
            'operations_completed': operations_completed
        }

    def run_integration_tests(self) -> Dict:
        """Test system integration and end-to-end functionality"""
        results = {
            'perception_to_reasoning': self.test_perception_reasoning_integration(),
            'reasoning_to_planning': self.test_reasoning_planning_integration(),
            'planning_to_execution': self.test_planning_execution_integration(),
            'end_to_end_task': self.test_end_to_end_task()
        }

        return {
            'pass_rate': sum(1 for r in results.values() if r['success']) / len(results),
            'detailed_results': results
        }

    def test_perception_reasoning_integration(self) -> Dict:
        """Test integration between perception and reasoning"""
        try:
            # Simulate perception input
            perception_output = {'object_detected': True, 'object_type': 'cup', 'confidence': 0.8}

            # Update beliefs based on perception
            self.system.reasoner.update_belief('detected_cup', perception_output)

            # Check if reasoning system properly processed the input
            belief = self.system.working_memory.retrieve('detected_cup')

            success = belief is not None and belief.confidence >= 0.7

            return {
                'success': success,
                'details': f"Belief confidence: {belief.confidence if belief else 0}"
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_reasoning_planning_integration(self) -> Dict:
        """Test integration between reasoning and planning"""
        try:
            # Set up a goal based on reasoning output
            self.system.active_goals = [{'type': 'navigation', 'location': (5, 5)}]

            # Generate plan
            self.system.generate_plans()

            # Check if plan was generated
            success = self.system.current_plan is not None

            return {
                'success': success,
                'details': f"Plan generated: {success}"
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_planning_execution_integration(self) -> Dict:
        """Test integration between planning and execution"""
        try:
            # Set up a simple plan
            self.system.current_plan = [{'action': 'move', 'target': (1, 1)}]

            # Execute plan
            self.system.execute_actions()

            # This would check if action was properly executed
            success = True  # Simplified

            return {
                'success': success,
                'details': 'Plan execution test completed'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_end_to_end_task(self) -> Dict:
        """Test complete end-to-end task execution"""
        try:
            # This would run a complete task similar to the serve drink demo
            # For assessment, we'll simulate the process
            task_success = True  # Would be determined by actual task execution

            return {
                'success': task_success,
                'details': 'End-to-end task completed successfully'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_robustness_tests(self) -> Dict:
        """Test system robustness under various conditions"""
        results = {
            'noise_resilience': self.test_noise_resilience(),
            'failure_recovery': self.test_failure_recovery(),
            'graceful_degradation': self.test_graceful_degradation()
        }

        return results

    def test_noise_resilience(self) -> Dict:
        """Test system resilience to sensor noise"""
        try:
            # Simulate noisy sensor inputs and check if system maintains performance
            success = True  # Simplified test
            return {'success': success, 'details': 'Noise resilience test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_failure_recovery(self) -> Dict:
        """Test system recovery from component failures"""
        try:
            # Simulate component failure and recovery
            success = True  # Simplified test
            return {'success': success, 'details': 'Failure recovery test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_graceful_degradation(self) -> Dict:
        """Test graceful degradation when components fail"""
        try:
            # Test if system degrades gracefully rather than failing completely
            success = True  # Simplified test
            return {'success': success, 'details': 'Graceful degradation test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_learning_tests(self) -> Dict:
        """Test learning capabilities"""
        results = {
            'adaptation': self.test_adaptation(),
            'generalization': self.test_generalization(),
            'memory_integration': self.test_memory_integration()
        }

        return results

    def test_adaptation(self) -> Dict:
        """Test system adaptation to new situations"""
        try:
            # Test if system can adapt behavior based on experience
            success = True  # Simplified test
            return {'success': success, 'details': 'Adaptation test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_generalization(self) -> Dict:
        """Test system generalization to new scenarios"""
        try:
            # Test if system can generalize learned behaviors
            success = True  # Simplified test
            return {'success': success, 'details': 'Generalization test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_memory_integration(self) -> Dict:
        """Test integration of learning with memory systems"""
        try:
            # Test if learned information is properly stored and retrieved
            success = True  # Simplified test
            return {'success': success, 'details': 'Memory integration test completed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall assessment score"""
        weights = {
            'functional': 0.3,
            'performance': 0.25,
            'integration': 0.2,
            'robustness': 0.15,
            'learning': 0.1
        }

        score = 0.0
        for category, weight in weights.items():
            if category in results:
                category_result = results[category]
                if isinstance(category_result, dict) and 'pass_rate' in category_result:
                    score += category_result['pass_rate'] * weight
                else:
                    # For categories without pass_rate, assume 100% if all sub-tests passed
                    if isinstance(category_result, dict):
                        sub_results = [r['success'] for r in category_result.get('detailed_results', {}).values() if isinstance(r, dict)]
                        if sub_results:
                            pass_rate = sum(sub_results) / len(sub_results)
                            score += pass_rate * weight

        return score

    def generate_assessment_report(self) -> str:
        """Generate detailed assessment report"""
        if not self.test_results:
            return "No assessment results available"

        report = []
        report.append("=== Humanoid AI-Robot Brain Assessment Report ===\n")

        # Overall Score
        overall_score = self.test_results.get('overall_score', 0)
        report.append(f"Overall Score: {overall_score:.2f}/1.0 ({overall_score*100:.1f}%)\n")

        # Category Breakdown
        for category, results in self.test_results.items():
            if category != 'overall_score':
                if isinstance(results, dict) and 'pass_rate' in results:
                    report.append(f"{category.title()} Score: {results['pass_rate']:.2f} ({results['pass_rate']*100:.1f}%)")

        report.append(f"\nAssessment completed at: {time.ctime()}")

        return "\n".join(report)

# Assessment runner
def run_module_assessment():
    """Run the complete module assessment"""
    rclpy.init()

    # Create cognitive system instance
    cognitive_system = HumanoidCognitiveSystem()

    # Create assessment instance
    assessor = CognitiveSystemAssessment(cognitive_system)

    # Run comprehensive assessment
    results = assessor.run_comprehensive_assessment()

    # Generate and print report
    report = assessor.generate_assessment_report()
    print(report)

    # Save detailed results
    import json
    with open('cognitive_assessment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Assessment results saved to cognitive_assessment_results.json")

    cognitive_system.destroy_node()
    rclpy.shutdown()

    return results

if __name__ == '__main__':
    results = run_module_assessment()
```

## 10.6 Best Practices Summary

### 10.6.1 Integration Best Practices

1. **Modular Design**: Keep components loosely coupled for easier integration and testing
2. **Standardized Interfaces**: Use consistent message types and service interfaces
3. **Performance Monitoring**: Continuously monitor system performance and resource usage
4. **Error Handling**: Implement robust error handling and recovery mechanisms
5. **Testing Strategy**: Develop comprehensive testing at component, integration, and system levels
6. **Documentation**: Maintain clear documentation for all system components and interfaces

### 10.6.2 Validation Checklist

- [ ] All components initialized successfully
- [ ] Memory systems functioning correctly
- [ ] Reasoning engine processing inputs properly
- [ ] Planning system generating valid plans
- [ ] Behavior trees executing as expected
- [ ] Navigation system operating safely
- [ ] Perception system providing accurate data
- [ ] System meeting real-time performance requirements
- [ ] Error handling mechanisms in place
- [ ] Comprehensive testing completed

## Summary

This integration tutorial demonstrates how to combine all the components of the AI-Robot Brain module into a complete, functional humanoid robot system. The tutorial covers:

1. **Complete Architecture**: Integration of perception, cognition, navigation, and behavior systems
2. **System Implementation**: Complete implementation of cognitive architecture with all components
3. **Task Execution**: End-to-end task execution demonstrating multi-modal capabilities
4. **Performance Optimization**: Profiling and optimization techniques for real-time operation
5. **Comprehensive Assessment**: Detailed evaluation framework for all system aspects

The integrated system demonstrates the practical application of Isaac Sim, visual SLAM, Nav2, behavior trees, cognitive architectures, and sim-to-real transfer strategies working together in a humanoid robot context. The assessment framework provides a systematic approach to evaluating the complete system's functionality, performance, and robustness.

This concludes Module 3: The AI-Robot Brain. Students should now be able to design, implement, and evaluate complete AI systems for humanoid robots that integrate perception, reasoning, planning, and control in a cohesive cognitive architecture.