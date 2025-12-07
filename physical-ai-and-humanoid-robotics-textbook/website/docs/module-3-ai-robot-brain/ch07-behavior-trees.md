---
title: Chapter 7 - Behavior Trees for Task Planning
description: Learn how to design and implement behavior trees for complex task planning in humanoid robots using ROS 2 and custom action nodes.
sidebar_position: 29
---

# Chapter 7 - Behavior Trees for Task Planning

Behavior trees are a powerful tool for representing and executing complex tasks in robotics, particularly for humanoid robots that need to perform sequences of actions while responding to environmental changes. Unlike traditional finite state machines, behavior trees offer better modularity, reusability, and maintainability for complex robotic behaviors. This chapter explores the theory and implementation of behavior trees for humanoid robot task planning.

## 7.1 Introduction to Behavior Trees

Behavior trees are hierarchical structures that represent tasks as a tree of nodes, where each node returns one of three states: SUCCESS, FAILURE, or RUNNING. This structure allows for complex decision-making and task execution while maintaining clear, readable logic.

### 7.1.1 Why Behavior Trees for Humanoid Robots?

Humanoid robots require sophisticated task planning capabilities due to their:
- **Complex kinematics**: Multiple degrees of freedom requiring coordinated movement
- **Diverse capabilities**: Manipulation, navigation, interaction, and communication
- **Dynamic environments**: Need to adapt to changing conditions
- **Multi-step tasks**: Complex sequences involving multiple skills

### 7.1.2 Basic Behavior Tree Concepts

- **Root Node**: The entry point of the behavior tree
- **Composite Nodes**: Control flow nodes (Sequence, Selector, Parallel)
- **Decorator Nodes**: Modify behavior of child nodes (Inverter, Retry, Timeout)
- **Leaf Nodes**: Execute specific actions or conditions (Action Nodes, Condition Nodes)

## 7.2 Behavior Tree Node Types

### 7.2.1 Composite Nodes

Composite nodes are the backbone of behavior trees, controlling the execution flow of their children:

```cpp
// Example C++ implementation of composite nodes
#include <vector>
#include <memory>

enum class NodeStatus {
    SUCCESS,
    FAILURE,
    RUNNING
};

class BTNode {
public:
    virtual NodeStatus tick() = 0;
    virtual void reset() {}
    virtual ~BTNode() = default;
};

class SequenceNode : public BTNode {
protected:
    std::vector<std::shared_ptr<BTNode>> children_;
    size_t current_child_;

public:
    SequenceNode() : current_child_(0) {}

    NodeStatus tick() override {
        for (; current_child_ < children_.size(); ++current_child_) {
            NodeStatus status = children_[current_child_]->tick();

            if (status == NodeStatus::FAILURE) {
                current_child_ = 0;
                return NodeStatus::FAILURE;
            } else if (status == NodeStatus::RUNNING) {
                return NodeStatus::RUNNING;
            }
            // If SUCCESS, continue to next child
        }

        current_child_ = 0; // Reset for next tick
        return NodeStatus::SUCCESS;
    }

    void reset() override {
        current_child_ = 0;
        for (auto& child : children_) {
            child->reset();
        }
    }

    void add_child(std::shared_ptr<BTNode> child) {
        children_.push_back(child);
    }
};

class SelectorNode : public BTNode {
protected:
    std::vector<std::shared_ptr<BTNode>> children_;
    size_t current_child_;

public:
    SelectorNode() : current_child_(0) {}

    NodeStatus tick() override {
        for (; current_child_ < children_.size(); ++current_child_) {
            NodeStatus status = children_[current_child_]->tick();

            if (status == NodeStatus::SUCCESS) {
                current_child_ = 0;
                return NodeStatus::SUCCESS;
            } else if (status == NodeStatus::RUNNING) {
                return NodeStatus::RUNNING;
            }
            // If FAILURE, continue to next child
        }

        current_child_ = 0; // Reset for next tick
        return NodeStatus::FAILURE;
    }

    void reset() override {
        current_child_ = 0;
        for (auto& child : children_) {
            child->reset();
        }
    }

    void add_child(std::shared_ptr<BTNode> child) {
        children_.push_back(child);
    }
};
```

### 7.2.2 Decorator Nodes

Decorator nodes modify the behavior of their single child:

```cpp
class InverterNode : public BTNode {
protected:
    std::shared_ptr<BTNode> child_;

public:
    InverterNode(std::shared_ptr<BTNode> child) : child_(child) {}

    NodeStatus tick() override {
        NodeStatus status = child_->tick();

        switch (status) {
            case NodeStatus::SUCCESS:
                return NodeStatus::FAILURE;
            case NodeStatus::FAILURE:
                return NodeStatus::SUCCESS;
            default:
                return status; // RUNNING stays RUNNING
        }
    }

    void reset() override {
        child_->reset();
    }
};

class RetryNode : public BTNode {
protected:
    std::shared_ptr<BTNode> child_;
    int max_attempts_;
    int current_attempts_;

public:
    RetryNode(std::shared_ptr<BTNode> child, int max_attempts)
        : child_(child), max_attempts_(max_attempts), current_attempts_(0) {}

    NodeStatus tick() override {
        NodeStatus status = child_->tick();

        if (status == NodeStatus::FAILURE) {
            if (current_attempts_ < max_attempts_) {
                current_attempts_++;
                child_->reset();
                return NodeStatus::RUNNING; // Continue retrying
            }
            current_attempts_ = 0;
            return NodeStatus::FAILURE;
        } else if (status == NodeStatus::SUCCESS) {
            current_attempts_ = 0;
            return NodeStatus::SUCCESS;
        }

        return status; // RUNNING
    }

    void reset() override {
        current_attempts_ = 0;
        child_->reset();
    }
};
```

## 7.3 Behavior Trees in ROS 2

ROS 2 provides the `behaviortree_cpp` library for implementing behavior trees. Here's how to create custom action nodes for humanoid robot tasks:

### 7.3.1 Setting up Behavior Tree Package

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(humanoid_behavior_trees)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)

add_executable(bt_node src/bt_node.cpp)
target_link_libraries(bt_node ${BT_LIBRARIES})
ament_target_dependencies(bt_node
  rclcpp
  behaviortree_cpp
  geometry_msgs
  nav2_msgs
)

install(TARGETS bt_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### 7.3.2 Custom Action Nodes for Humanoid Robots

```cpp
// humanoid_action_nodes.cpp
#include "behaviortree_cpp/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace BT;

class MoveToPoseAction : public BT::AsyncActionNode
{
public:
    MoveToPoseAction(const std::string& name, const BT::NodeConfig& config)
        : BT::AsyncActionNode(name, config), goal_sent_(false)
    {
        node_ = config.blackboard->template get<rclcpp::Node::SharedPtr>("node");
        client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
            node_, "navigate_to_pose");
    }

    static BT::NodeConfig metadata()
    {
        BT::NodeConfig config;
        config.blackboard_key = "node";
        return config;
    }

    BT::NodeStatus tick() override
    {
        if (!goal_sent_) {
            // Get target pose from input port
            geometry_msgs::msg::PoseStamped goal_pose;
            if (!getInput("target_pose", goal_pose)) {
                RCLCPP_ERROR(node_->get_logger(), "Missing required input [target_pose]");
                return BT::NodeStatus::FAILURE;
            }

            // Send navigation goal
            auto goal = nav2_msgs::action::NavigateToPose::Goal();
            goal.pose = goal_pose;

            auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
            send_goal_options.result_callback = [this](const auto& result) {
                std::lock_guard<std::mutex> lock(mutex_);
                result_received_ = true;
                if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                    result_status_ = BT::NodeStatus::SUCCESS;
                } else {
                    result_status_ = BT::NodeStatus::FAILURE;
                }
            };

            goal_handle_future_ = client_->async_send_goal(goal, send_goal_options);
            goal_sent_ = true;
        }

        // Check for result
        std::lock_guard<std::mutex> lock(mutex_);
        if (result_received_) {
            goal_sent_ = false;
            result_received_ = false;
            return result_status_;
        }

        return BT::NodeStatus::RUNNING;
    }

    void halt() override
    {
        goal_sent_ = false;
        result_received_ = false;
        BT::AsyncActionNode::halt();
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr client_;
    rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::SharedPtr goal_handle_future_;

    bool goal_sent_;
    bool result_received_ = false;
    BT::NodeStatus result_status_;
    std::mutex mutex_;
};

class DetectObjectAction : public BT::SyncActionNode
{
public:
    DetectObjectAction(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config)
    {
        node_ = config.blackboard->template get<rclcpp::Node::SharedPtr>("node");
        detection_client_ = node_->create_client<your_msgs::srv::DetectObjects>("detect_objects");
    }

    static BT::NodeConfig metadata()
    {
        BT::NodeConfig config;
        config.blackboard_key = "node";
        config.input_ports["object_type"] = "Type of object to detect";
        config.output_ports["object_pose"] = "Detected object pose";
        return config;
    }

    BT::NodeStatus tick() override
    {
        std::string object_type;
        if (!getInput("object_type", object_type)) {
            RCLCPP_ERROR(node_->get_logger(), "Missing required input [object_type]");
            return BT::NodeStatus::FAILURE;
        }

        // Create detection request
        auto request = std::make_shared<your_msgs::srv::DetectObjects::Request>();
        request->object_type = object_type;

        // Synchronous call (in practice, you might want async)
        while (!detection_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(node_->get_logger(), "Interrupted while waiting for service");
                return BT::NodeStatus::FAILURE;
            }
            RCLCPP_INFO(node_->get_logger(), "Service not available, waiting again...");
        }

        auto result = detection_client_->async_send_request(request);
        auto future_result = result.wait_for(std::chrono::seconds(5));

        if (future_result == std::future_status::ready) {
            auto response = result.get();
            if (response->success) {
                // Set output
                geometry_msgs::msg::Pose object_pose = response->object_poses[0];
                setOutput("object_pose", object_pose);
                return BT::NodeStatus::SUCCESS;
            } else {
                return BT::NodeStatus::FAILURE;
            }
        } else {
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Client<your_msgs::srv::DetectObjects>::SharedPtr detection_client_;
};

// Register custom nodes
static const char* xml_text = R"(
<root main_tree_to_execute = "MainTree" >
    <BehaviorTree ID="MainTree">
        <Sequence name="root_sequence">
            <MoveToPoseAction target_pose="{waypoint_1}"/>
            <DetectObjectAction object_type="red_cup" output="cup_pose"/>
            <Sequence>
                <MoveToPoseAction target_pose="{cup_pose}"/>
                <GraspObjectAction object_pose="{cup_pose}"/>
            </Sequence>
        </Sequence>
    </BehaviorTree>
</root>
)";

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("bt_node");

    // Create factory and register custom nodes
    BT::BehaviorTreeFactory factory;
    factory.registerNodeType<MoveToPoseAction>("MoveToPoseAction");
    factory.registerNodeType<DetectObjectAction>("DetectObjectAction");

    // Add node to blackboard
    auto blackboard = BT::Blackboard::create();
    blackboard->set("node", node);

    // Create and tick tree
    auto tree = factory.createTreeFromText(xml_text, blackboard);

    rclcpp::Rate rate(10); // 10 Hz
    while (rclcpp::ok()) {
        tree.tickRoot();
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
```

## 7.4 Complex Humanoid Robot Behavior Trees

### 7.4.1 Multi-Modal Task Execution

```xml
<!-- complex_task_bt.xml -->
<root main_tree_to_execute="HumanoidTask">
    <BehaviorTree ID="HumanoidTask">
        <Fallback name="main_fallback">
            <!-- Emergency stop sequence -->
            <Sequence name="emergency_check">
                <CheckEmergencyStop/>
                <StopRobot/>
            </Sequence>

            <!-- Normal task execution -->
            <Sequence name="normal_execution">
                <!-- Navigate to task location -->
                <Fallback name="navigation">
                    <Sequence name="primary_navigation">
                        <CheckBatteryLevel min_level="20"/>
                        <MoveToPoseAction target_pose="{task_location}"/>
                    </Sequence>
                    <Sequence name="backup_navigation">
                        <MoveToPoseAction target_pose="{backup_location}"/>
                        <RequestHumanAssistance/>
                    </Sequence>
                </Fallback>

                <!-- Execute task with error handling -->
                <RetryUntilSucceed name="task_execution" num_attempts="3">
                    <Sequence name="single_task_attempt">
                        <DetectObjectAction object_type="{target_object}"/>
                        <CheckObjectReachable object_pose="{detected_object}"/>
                        <Fallback name="grasp_strategy">
                            <Sequence name="precise_grasp">
                                <MoveToGraspPose grasp_pose="{precise_grasp_pose}"/>
                                <ExecuteGrasp object_pose="{detected_object}"/>
                            </Sequence>
                            <Sequence name="safe_grasp">
                                <MoveToGraspPose grasp_pose="{safe_grasp_pose}"/>
                                <ExecuteGrasp object_pose="{detected_object}"/>
                            </Sequence>
                        </Fallback>
                    </Sequence>
                </RetryUntilSucceed>

                <!-- Transport object -->
                <Sequence name="transport">
                    <CheckObjectHeld/>
                    <MoveToPoseAction target_pose="{delivery_location}"/>
                    <ExecuteRelease object_pose="{delivery_pose}"/>
                </Sequence>
            </Sequence>
        </Fallback>
    </BehaviorTree>
</root>
```

### 7.4.2 Adaptive Behavior Trees

```cpp
// Adaptive behavior tree that modifies based on context
class AdaptiveBehaviorTree
{
private:
    std::shared_ptr<BT::Tree> tree_;
    std::shared_ptr<BT::Blackboard> blackboard_;
    rclcpp::Node::SharedPtr node_;

    // Context variables that affect tree structure
    double battery_level_;
    task_complexity_t current_task_complexity_;
    environment_type_t current_environment_;

public:
    AdaptiveBehaviorTree(rclcpp::Node::SharedPtr node)
        : node_(node), battery_level_(100.0)
    {
        blackboard_ = BT::Blackboard::create();
        blackboard_->set("node", node_);

        update_tree_structure();
    }

    void update_tree_structure()
    {
        // Modify tree based on current context
        if (battery_level_ < 30.0) {
            // Use energy-efficient paths
            modify_for_low_battery();
        } else if (current_task_complexity_ == HIGH_COMPLEXITY) {
            // Use more sophisticated approaches
            modify_for_complex_task();
        } else if (current_environment_ == DYNAMIC) {
            // Increase safety margins and check frequency
            modify_for_dynamic_environment();
        }
    }

    void modify_for_low_battery()
    {
        // Switch to energy-efficient navigation
        blackboard_->set("navigation_mode", "energy_efficient");
        blackboard_->set("max_speed", 0.5); // Reduce speed to save energy
    }

    void modify_for_complex_task()
    {
        // Increase perception accuracy
        blackboard_->set("detection_threshold", 0.9); // Higher confidence required
        blackboard_->set("planning_timeout", 30.0);   // Allow more planning time
    }

    void modify_for_dynamic_environment()
    {
        // Increase safety checks
        blackboard_->set("replanning_frequency", 5.0); // More frequent replanning
        blackboard_->set("safety_margin", 0.8);        // Larger safety margin
    }

    BT::NodeStatus execute_tick()
    {
        update_context();
        update_tree_structure();
        return tree_->rootNode()->executeTick();
    }

private:
    void update_context()
    {
        // Update context variables from sensors and system state
        battery_level_ = get_battery_level();
        current_task_complexity_ = get_current_task_complexity();
        current_environment_ = get_current_environment_type();
    }

    double get_battery_level()
    {
        // Get battery level from system
        return 85.0; // Placeholder
    }

    task_complexity_t get_current_task_complexity()
    {
        // Determine task complexity
        return MEDIUM_COMPLEXITY; // Placeholder
    }

    environment_type_t get_current_environment_type()
    {
        // Determine environment type
        return STATIC; // Placeholder
    }
};
```

## 7.5 Behavior Tree Design Patterns for Humanoid Robots

### 7.5.1 Perception-Action Loop

```xml
<!-- perception_action_loop.xml -->
<root main_tree_to_execute="PerceptionActionLoop">
    <BehaviorTree ID="PerceptionActionLoop">
        <ReactiveSequence name="perception_action_cycle">
            <!-- Continuous perception -->
            <Parallel success_threshold="1" failure_threshold="1">
                <Sequence name="perception_tasks">
                    <UpdateSensorData/>
                    <DetectObstacles/>
                    <TrackHumans/>
                    <MonitorEnvironmentChanges/>
                </Sequence>

                <!-- Action execution with interruption -->
                <Fallback name="action_layer">
                    <CheckForInterrupts/>
                    <ExecuteCurrentAction/>
                </Fallback>
            </Parallel>
        </ReactiveSequence>
    </BehaviorTree>
</root>
```

### 7.5.2 Hierarchical Task Decomposition

```cpp
// Hierarchical task decomposition example
class TaskDecomposition
{
public:
    // High-level task: Serve drink
    std::shared_ptr<BT::Tree> create_serve_drink_tree()
    {
        auto factory = BT::BehaviorTreeFactory();

        // Register all action nodes
        register_action_nodes(factory);

        // Create tree for serving drink
        std::string serve_drink_xml = R"(
        <root main_tree_to_execute="ServeDrink">
            <BehaviorTree ID="ServeDrink">
                <Sequence name="serve_drink_sequence">
                    <MoveToPoseAction target_pose="{kitchen_location}"/>
                    <DetectObjectAction object_type="cup" output="cup_pose"/>
                    <MoveToPoseAction target_pose="{cup_approach_pose}"/>
                    <GraspObjectAction object_pose="{cup_pose}"/>
                    <MoveToPoseAction target_pose="{person_location}"/>
                    <ReleaseObjectAction object_pose="{delivery_pose}"/>
                    <MoveToPoseAction target_pose="{safe_pose}"/>
                </Sequence>
            </BehaviorTree>
        </root>
        )";

        return std::make_shared<BT::Tree>(factory.createTreeFromText(serve_drink_xml));
    }

    // Mid-level task: Pick up object
    std::shared_ptr<BT::Tree> create_pickup_object_tree()
    {
        std::string pickup_xml = R"(
        <root main_tree_to_execute="PickupObject">
            <BehaviorTree ID="PickupObject">
                <Sequence name="pickup_sequence">
                    <CheckObjectReachable object_pose="{target_pose}"/>
                    <ApproachObject object_pose="{target_pose}"/>
                    <GraspObjectAction object_pose="{target_pose}"/>
                    <VerifyGraspSuccess/>
                </Sequence>
            </BehaviorTree>
        </root>
        )";

        auto factory = BT::BehaviorTreeFactory();
        register_action_nodes(factory);
        return std::make_shared<BT::Tree>(factory.createTreeFromText(pickup_xml));
    }

private:
    void register_action_nodes(BT::BehaviorTreeFactory& factory)
    {
        factory.registerNodeType<MoveToPoseAction>("MoveToPoseAction");
        factory.registerNodeType<DetectObjectAction>("DetectObjectAction");
        factory.registerNodeType<GraspObjectAction>("GraspObjectAction");
        factory.registerNodeType<ReleaseObjectAction>("ReleaseObjectAction");
        factory.registerNodeType<CheckObjectReachable>("CheckObjectReachable");
        factory.registerNodeType<ApproachObject>("ApproachObject");
        factory.registerNodeType<VerifyGraspSuccess>("VerifyGraspSuccess");
        factory.registerNodeType<CheckForInterrupts>("CheckForInterrupts");
    }
};
```

## 7.6 Debugging and Visualization

### 7.6.1 Behavior Tree Visualization

```python
# Python script for visualizing behavior trees
import pydot
from behaviortree import BehaviorTree, Sequence, Selector, Action

def visualize_behavior_tree(bt, filename="behavior_tree.png"):
    """Create a visual representation of the behavior tree"""
    graph = pydot.Dot(graph_type='digraph', rankdir='TB')

    def add_node_recursive(node, parent_node=None):
        # Create node in graph
        node_id = str(id(node))
        node_label = f"{node.__class__.__name__}\\n{getattr(node, 'name', '')}"

        if hasattr(node, 'status'):
            status = node.status
            color = 'lightgreen' if status == 'SUCCESS' else 'lightcoral' if status == 'FAILURE' else 'lightyellow'
        else:
            color = 'lightblue'

        graph_node = pydot.Node(node_id, label=node_label, style='filled', fillcolor=color)
        graph.add_node(graph_node)

        # Connect to parent if exists
        if parent_node:
            parent_id = str(id(parent_node))
            edge = pydot.Edge(parent_id, node_id)
            graph.add_edge(edge)

        # Add children
        if hasattr(node, 'children'):
            for child in node.children:
                add_node_recursive(child, node)

    add_node_recursive(bt.root)
    graph.write_png(filename)

# Example usage during execution
def monitor_and_visualize():
    """Monitor tree execution and visualize periodically"""
    import time

    bt = create_humanoid_task_tree()

    for i in range(100):  # Run for 100 ticks
        status = bt.tick()
        if i % 10 == 0:  # Visualize every 10 ticks
            visualize_behavior_tree(bt, f"bt_state_{i}.png")
        time.sleep(0.1)  # 10 Hz
```

### 7.6.2 Logging and Monitoring

```cpp
// Enhanced action node with logging
class LoggedActionNode : public BT::AsyncActionNode
{
protected:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_;
    rclcpp::Time start_time_;
    std::string action_name_;

public:
    LoggedActionNode(const std::string& name,
                     const BT::NodeConfig& config,
                     const std::string& action_name)
        : BT::AsyncActionNode(name, config), action_name_(action_name)
    {
        node_ = config.blackboard->template get<rclcpp::Node::SharedPtr>("node");
        logger_ = node_->get_logger();
    }

    BT::NodeStatus tick() override
    {
        if (status() == BT::NodeStatus::IDLE) {
            start_time_ = node_->now();
            RCLCPP_INFO(logger_, "Starting action: %s", action_name_.c_str());
        }

        BT::NodeStatus status = execute_action();

        if (status != BT::NodeStatus::RUNNING) {
            auto duration = node_->now() - start_time_;
            std::string result = (status == BT::NodeStatus::SUCCESS) ? "SUCCESS" : "FAILURE";
            RCLCPP_INFO(logger_, "Action %s completed with %s in %.2f seconds",
                       action_name_.c_str(), result.c_str(),
                       duration.seconds());
        }

        return status;
    }

    virtual BT::NodeStatus execute_action() = 0;

    void halt() override
    {
        RCLCPP_WARN(logger_, "Action %s was halted", action_name_.c_str());
        BT::AsyncActionNode::halt();
    }
};

// Specialized logged action for humanoid navigation
class LoggedNavigateAction : public LoggedActionNode
{
private:
    rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr client_;
    bool goal_sent_;
    rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::SharedPtr goal_handle_;

public:
    LoggedNavigateAction(const std::string& name, const BT::NodeConfig& config)
        : LoggedActionNode(name, config, "NavigateToPose"), goal_sent_(false)
    {
        client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
            node_, "navigate_to_pose");
    }

    BT::NodeStatus execute_action() override
    {
        if (!goal_sent_) {
            geometry_msgs::msg::PoseStamped goal_pose;
            if (!getInput("target_pose", goal_pose)) {
                RCLCPP_ERROR(logger_, "Missing target pose for navigation");
                return BT::NodeStatus::FAILURE;
            }

            auto goal = nav2_msgs::action::NavigateToPose::Goal();
            goal.pose = goal_pose;

            auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
            send_goal_options.result_callback = [this](const auto& result) {
                if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                    RCLCPP_INFO(logger_, "Navigation goal succeeded");
                    // Set internal state to success
                } else {
                    RCLCPP_WARN(logger_, "Navigation goal failed");
                    // Set internal state to failure
                }
            };

            goal_handle_ = client_->async_send_goal(goal, send_goal_options);
            goal_sent_ = true;

            RCLCPP_INFO(logger_, "Sent navigation goal to (%.2f, %.2f)",
                       goal_pose.pose.position.x, goal_pose.pose.position.y);
        }

        // Check if goal is still executing
        if (goal_handle_ && goal_handle_->get_status() == rclcpp_action::GoalStatus::STATUS_EXECUTING) {
            return BT::NodeStatus::RUNNING;
        }

        // Goal has finished
        goal_sent_ = false;
        // Return appropriate status based on result
        return BT::NodeStatus::SUCCESS; // Simplified
    }
};
```

## 7.7 Performance Optimization

### 7.7.1 Efficient Tree Ticking

```cpp
class OptimizedBehaviorTree
{
private:
    std::shared_ptr<BT::Tree> tree_;
    std::vector<BT::TreeNode*> active_nodes_;  // Nodes that returned RUNNING
    std::chrono::steady_clock::time_point last_tick_time_;
    double max_tick_rate_;  // Hz

public:
    OptimizedBehaviorTree(std::shared_ptr<BT::Tree> tree, double max_rate = 50.0)
        : tree_(tree), max_tick_rate_(max_rate)
    {
        update_active_nodes_list();
    }

    BT::NodeStatus tick_optimized()
    {
        // Rate limiting
        auto now = std::chrono::steady_clock::now();
        auto min_interval = std::chrono::duration<double>(1.0 / max_tick_rate_);
        if (now - last_tick_time_ < min_interval) {
            std::this_thread::sleep_until(last_tick_time_ + min_interval);
        }
        last_tick_time_ = now;

        // Only tick nodes that were running
        BT::NodeStatus root_status = tick_active_subtree();

        // Update active nodes list for next tick
        update_active_nodes_list();

        return root_status;
    }

private:
    BT::NodeStatus tick_active_subtree()
    {
        // For simplicity, tick the whole tree
        // In practice, you'd only tick the relevant subtrees
        return tree_->rootNode()->executeTick();
    }

    void update_active_nodes_list()
    {
        active_nodes_.clear();
        collect_running_nodes(tree_->rootNode());
    }

    void collect_running_nodes(BT::TreeNode* node)
    {
        if (node->status() == BT::NodeStatus::RUNNING) {
            active_nodes_.push_back(node);
        }

        // Collect children if this is a control node
        if (auto control_node = dynamic_cast<BT::ControlNode*>(node)) {
            for (size_t i = 0; i < control_node->childrenCount(); ++i) {
                collect_running_nodes(control_node->child(i));
            }
        }
    }
};
```

### 7.7.2 Memory Management

```cpp
// Memory-efficient behavior tree implementation
class MemoryEfficientBT
{
private:
    // Use object pooling to reduce allocations
    std::queue<std::unique_ptr<BTNode>> node_pool_;
    std::mutex pool_mutex_;

    // Pre-allocated nodes for common operations
    std::vector<std::unique_ptr<BTNode>> pre_allocated_nodes_;

public:
    template<typename NodeType, typename... Args>
    std::unique_ptr<NodeType> acquire_node(Args&&... args)
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        if (!node_pool_.empty()) {
            auto node = std::unique_ptr<NodeType>(
                static_cast<NodeType*>(node_pool_.front().release()));
            node_pool_.pop();
            node->reset(); // Reset to clean state
            return node;
        }

        // Create new node if pool is empty
        return std::make_unique<NodeType>(std::forward<Args>(args)...);
    }

    void release_node(std::unique_ptr<BTNode> node)
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        node->reset(); // Ensure clean state
        node_pool_.push(std::move(node));
    }

    void pre_allocate_nodes(size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            pre_allocated_nodes_.push_back(std::make_unique<SequenceNode>());
        }
    }
};
```

## 7.8 Integration with Planning Systems

### 7.8.1 Hierarchical Planning Integration

```cpp
// Integration with high-level planner
class HierarchicalPlanner
{
private:
    std::shared_ptr<BT::Tree> behavior_tree_;
    std::shared_ptr<HighLevelPlanner> high_level_planner_;
    std::shared_ptr<TaskDecomposer> task_decomposer_;

public:
    void execute_high_level_task(const Task& high_level_task)
    {
        // Decompose high-level task into executable actions
        auto primitive_tasks = task_decomposer_->decompose(high_level_task);

        // Create behavior tree for the task sequence
        auto task_tree = create_task_tree(primitive_tasks);

        // Execute the behavior tree
        execute_behavior_tree(task_tree);
    }

    std::shared_ptr<BT::Tree> create_task_tree(const std::vector<PrimitiveTask>& tasks)
    {
        // Build behavior tree from primitive tasks
        auto factory = BT::BehaviorTreeFactory();
        register_action_nodes(factory);

        // Create XML representation of task sequence
        std::string xml = build_task_xml(tasks);

        return std::make_shared<BT::Tree>(factory.createTreeFromText(xml));
    }

    void execute_behavior_tree(std::shared_ptr<BT::Tree> tree)
    {
        // Execute with monitoring and error handling
        while (rclcpp::ok()) {
            BT::NodeStatus status = tree->rootNode()->executeTick();

            if (status == BT::NodeStatus::SUCCESS) {
                RCLCPP_INFO(node_->get_logger(), "Task completed successfully");
                break;
            } else if (status == BT::NodeStatus::FAILURE) {
                RCLCPP_ERROR(node_->get_logger(), "Task failed, initiating recovery");
                handle_failure(tree);
                break;
            }

            // Check for preemption
            if (check_for_preemption()) {
                RCLCPP_INFO(node_->get_logger(), "Task preempted");
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    std::string build_task_xml(const std::vector<PrimitiveTask>& tasks)
    {
        std::string xml = R"(<root main_tree_to_execute="TaskSequence"><BehaviorTree ID="TaskSequence"><Sequence>)";

        for (const auto& task : tasks) {
            xml += "<" + task.action_type + " ";
            for (const auto& param : task.parameters) {
                xml += param.first + "=\"" + param.second + "\" ";
            }
            xml += "/>";
        }

        xml += "</Sequence></BehaviorTree></root>";
        return xml;
    }

    void handle_failure(std::shared_ptr<BT::Tree> tree)
    {
        // Implement failure recovery strategies
        // - Retry current action
        // - Execute recovery behavior tree
        // - Request human assistance
        // - Switch to safe state
    }

    bool check_for_preemption()
    {
        // Check if a higher priority task has arrived
        return false; // Simplified
    }
};
```

## 7.9 Best Practices for Humanoid Robot Behavior Trees

### 7.9.1 Design Principles

1. **Modularity**: Create reusable sub-trees for common behaviors
2. **Error Handling**: Always include fallbacks and recovery behaviors
3. **State Management**: Properly manage state between tree executions
4. **Performance**: Optimize for real-time execution constraints
5. **Debugging**: Include logging and visualization capabilities

### 7.9.2 Common Patterns

1. **Monitor Pattern**: Continuous monitoring with interruption capability
2. **Try-Catch Pattern**: Attempt action with recovery fallback
3. **Selector-Sequence Pattern**: Try alternatives until one succeeds
4. **Parallel Pattern**: Execute multiple tasks concurrently with coordination

## Summary

Behavior trees provide a powerful framework for implementing complex task planning in humanoid robots. Their hierarchical structure, clear execution semantics, and modularity make them ideal for representing the diverse and complex behaviors required by humanoid robots. By properly implementing custom action nodes, integrating with ROS 2 systems, and following best practices for design and optimization, you can create robust and maintainable behavior trees that enable sophisticated robotic behaviors. In the next chapter, we will explore cognitive architectures for embodied agents, building on the task planning foundation established here.