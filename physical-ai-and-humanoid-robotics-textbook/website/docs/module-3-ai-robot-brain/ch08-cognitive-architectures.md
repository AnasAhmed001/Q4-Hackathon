---
title: Chapter 8 - Cognitive Architectures for Embodied Agents
description: Explore cognitive architectures that enable humanoid robots to perceive, reason, and act intelligently in complex environments.
sidebar_position: 30
---

# Chapter 8 - Cognitive Architectures for Embodied Agents

Cognitive architectures provide the foundational framework for creating intelligent embodied agents that can perceive, reason, learn, and act in complex environments. For humanoid robots, cognitive architectures must integrate perception, memory, reasoning, planning, and action selection into a cohesive system that enables sophisticated behavior. This chapter explores the design and implementation of cognitive architectures specifically tailored for humanoid robots.

## 8.1 Introduction to Cognitive Architectures

A cognitive architecture is a theoretical framework that describes the structure and function of intelligent systems. Unlike simple reactive systems, cognitive architectures incorporate:

- **Memory systems** for storing and retrieving information
- **Reasoning mechanisms** for making decisions and drawing inferences
- **Learning capabilities** for adapting to new situations
- **Perception-action loops** for interacting with the environment
- **Goal management** for directing behavior toward objectives

### 8.1.1 Key Components of Cognitive Architectures

```cpp
// Basic structure of a cognitive architecture
class CognitiveArchitecture {
protected:
    std::unique_ptr<PerceptionSystem> perception_system_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<ReasoningEngine> reasoning_engine_;
    std::unique_ptr<PlanningSystem> planning_system_;
    std::unique_ptr<ActionSelection> action_selection_;
    std::unique_ptr<LearningSystem> learning_system_;

public:
    virtual void sense() = 0;
    virtual void perceive() = 0;
    virtual void reason() = 0;
    virtual void plan() = 0;
    virtual void act() = 0;
    virtual void learn() = 0;

    void cycle() {
        sense();
        perceive();
        reason();
        plan();
        act();
        learn();
    }
};
```

### 8.1.2 Requirements for Humanoid Robot Cognitive Architectures

Humanoid robots have specific requirements that influence cognitive architecture design:

- **Real-time constraints**: Must operate within timing constraints for stability and safety
- **Embodied cognition**: Physical form affects cognitive processes
- **Multi-modal perception**: Integration of various sensor modalities
- **Social interaction**: Understanding and responding to human social cues
- **Dynamic environments**: Adaptation to changing environmental conditions

## 8.2 Memory Systems in Cognitive Architectures

### 8.2.1 Memory Hierarchy

Cognitive architectures typically implement a memory hierarchy similar to human memory:

```cpp
class MemorySystem {
public:
    // Sensory memory - immediate sensory buffer
    class SensoryBuffer {
    private:
        std::unordered_map<std::string, std::vector<SensorData>> buffers_;
        std::chrono::milliseconds retention_time_{100ms};

    public:
        void store(const std::string& sensor_type, const SensorData& data) {
            buffers_[sensor_type].push_back(data);
            // Remove old data based on retention time
            auto now = std::chrono::steady_clock::now();
            buffers_[sensor_type].erase(
                std::remove_if(buffers_[sensor_type].begin(), buffers_[sensor_type].end(),
                    [this, now](const SensorData& d) {
                        return (now - d.timestamp) > retention_time_;
                    }),
                buffers_[sensor_type].end()
            );
        }

        std::vector<SensorData> get(const std::string& sensor_type) {
            return buffers_[sensor_type];
        }
    };

    // Working memory - active information for current tasks
    class WorkingMemory {
    private:
        std::unordered_map<std::string, std::any> active_items_;
        std::chrono::seconds expiration_time_{30s};

    public:
        template<typename T>
        void store(const std::string& key, const T& value) {
            active_items_[key] = value;
            // Set expiration timestamp
        }

        template<typename T>
        std::optional<T> retrieve(const std::string& key) {
            auto it = active_items_.find(key);
            if (it != active_items_.end()) {
                try {
                    return std::any_cast<T>(it->second);
                } catch (const std::bad_any_cast& e) {
                    return std::nullopt;
                }
            }
            return std::nullopt;
        }

        void cleanup_expired() {
            // Remove expired items
        }
    };

    // Long-term memory - persistent storage
    class LongTermMemory {
    private:
        std::unordered_map<std::string, std::vector<MemoryItem>> memory_nodes_;
        sqlite3* db_;

    public:
        void store(const std::string& key, const MemoryItem& item) {
            // Store in both in-memory cache and persistent storage
            memory_nodes_[key].push_back(item);
            store_in_db(key, item);
        }

        std::vector<MemoryItem> retrieve(const std::string& key,
                                       const std::string& context = "") {
            // Retrieve from cache first, then from DB if needed
            auto it = memory_nodes_.find(key);
            if (it != memory_nodes_.end()) {
                return it->second;
            }
            return retrieve_from_db(key, context);
        }

        std::vector<MemoryItem> query(const std::string& pattern) {
            // Semantic search through long-term memory
            return semantic_search(pattern);
        }

    private:
        void store_in_db(const std::string& key, const MemoryItem& item);
        std::vector<MemoryItem> retrieve_from_db(const std::string& key,
                                               const std::string& context);
        std::vector<MemoryItem> semantic_search(const std::string& pattern);
    };

private:
    SensoryBuffer sensory_buffer_;
    WorkingMemory working_memory_;
    LongTermMemory long_term_memory_;
};
```

### 8.2.2 Episodic Memory for Humanoid Robots

Episodic memory stores sequences of experiences that can be replayed for learning:

```cpp
struct Episode {
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    std::vector<SensorObservation> sensor_data;
    std::vector<Action> executed_actions;
    std::vector<Goal> active_goals;
    std::vector<Context> environmental_context;
    double reward;
    bool success;
};

class EpisodicMemory {
private:
    std::vector<Episode> episodes_;
    size_t max_episodes_{1000};
    std::mutex mutex_;

public:
    void store_episode(const Episode& episode) {
        std::lock_guard<std::mutex> lock(mutex_);
        episodes_.push_back(episode);
        if (episodes_.size() > max_episodes_) {
            episodes_.erase(episodes_.begin()); // Remove oldest
        }
    }

    std::vector<Episode> retrieve_similar_episodes(const Episode& query,
                                                 size_t max_count = 5) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::pair<double, Episode>> similarities;

        for (const auto& episode : episodes_) {
            double similarity = calculate_episode_similarity(query, episode);
            similarities.emplace_back(similarity, episode);
        }

        // Sort by similarity (descending)
        std::sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<Episode> result;
        for (size_t i = 0; i < std::min(max_count, similarities.size()); ++i) {
            result.push_back(similarities[i].second);
        }

        return result;
    }

    std::vector<Action> retrieve_successful_actions_for_context(const Context& context) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Action> successful_actions;
        for (const auto& episode : episodes_) {
            if (episode.success && is_context_similar(episode.environmental_context, context)) {
                successful_actions.insert(successful_actions.end(),
                                        episode.executed_actions.begin(),
                                        episode.executed_actions.end());
            }
        }

        return successful_actions;
    }

private:
    double calculate_episode_similarity(const Episode& a, const Episode& b) {
        // Calculate similarity based on context, goals, and outcomes
        double context_similarity = calculate_context_similarity(a.environmental_context,
                                                               b.environmental_context);
        double goal_similarity = calculate_goal_similarity(a.active_goals, b.active_goals);
        double outcome_similarity = (a.success == b.success) ? 1.0 : 0.0;

        return 0.4 * context_similarity + 0.4 * goal_similarity + 0.2 * outcome_similarity;
    }

    double calculate_context_similarity(const std::vector<Context>& a,
                                      const std::vector<Context>& b) {
        // Implementation depends on context representation
        // This could involve semantic similarity, spatial proximity, etc.
        return 0.5; // Placeholder
    }

    double calculate_goal_similarity(const std::vector<Goal>& a, const std::vector<Goal>& b) {
        // Calculate similarity between goal sets
        return 0.5; // Placeholder
    }

    bool is_context_similar(const std::vector<Context>& a, const Context& b) {
        // Check if context b is similar to any context in a
        return false; // Placeholder
    }
};
```

## 8.3 Perception Systems for Cognitive Integration

### 8.3.1 Multi-Modal Perception Integration

```cpp
class MultiModalPerceptionSystem {
private:
    std::unordered_map<std::string, std::unique_ptr<SensorInterface>> sensors_;
    std::unique_ptr<PerceptualProcessor> perceptual_processor_;
    std::unique_ptr<AttentionMechanism> attention_mechanism_;

    // Working memory for current percepts
    WorkingMemory* working_memory_;

public:
    MultiModalPerceptionSystem(WorkingMemory* wm) : working_memory_(wm) {
        initialize_sensors();
        perceptual_processor_ = std::make_unique<DeepPerceptualProcessor>();
        attention_mechanism_ = std::make_unique<SelectiveAttention>();
    }

    void sense_and_perceive() {
        // Collect raw sensor data
        auto raw_data = collect_sensor_data();

        // Apply attention mechanism to select relevant information
        auto attended_data = attention_mechanism_->select_relevant(raw_data);

        // Process attended data to form percepts
        auto percepts = perceptual_processor_->process(attended_data);

        // Store percepts in working memory
        store_percepts_in_memory(percepts);

        // Update saliency maps for next attention cycle
        update_saliency_maps(percepts);
    }

private:
    std::unordered_map<std::string, SensorData> collect_sensor_data() {
        std::unordered_map<std::string, SensorData> raw_data;

        for (auto& [sensor_name, sensor] : sensors_) {
            raw_data[sensor_name] = sensor->get_data();
        }

        return raw_data;
    }

    void store_percepts_in_memory(const std::vector<Percept>& percepts) {
        for (const auto& percept : percepts) {
            // Store in working memory with appropriate metadata
            working_memory_->store(percept.type + "_" + percept.id, percept);

            // Update saliency based on percept importance
            update_saliency(percept);
        }
    }

    void update_saliency(const Percept& percept) {
        // Update attention maps based on percept saliency
        // This could involve object importance, motion, etc.
    }

    void initialize_sensors() {
        // Initialize different sensor types
        sensors_["rgb_camera"] = std::make_unique<RGBCamera>();
        sensors_["depth_camera"] = std::make_unique<DepthCamera>();
        sensors_["lidar"] = std::make_unique<Lidar>();
        sensors_["imu"] = std::make_unique<IMU>();
        sensors_["microphone"] = std::make_unique<Microphone>();
        sensors_["tactile"] = std::make_unique<TactileSensors>();
    }
};

// Attention mechanism for selecting relevant information
class SelectiveAttention {
private:
    cv::Mat visual_saliency_map_;
    std::vector<AudioEvent> audio_saliency_events_;
    std::chrono::time_point<std::chrono::steady_clock> last_update_;

public:
    std::unordered_map<std::string, SensorData> select_relevant(
        const std::unordered_map<std::string, SensorData>& raw_data) {

        std::unordered_map<std::string, SensorData> attended_data;

        // Process visual attention
        if (raw_data.count("rgb_camera")) {
            auto attended_visual = process_visual_attention(raw_data.at("rgb_camera"));
            attended_data["rgb_camera"] = attended_visual;
        }

        // Process audio attention
        if (raw_data.count("microphone")) {
            auto attended_audio = process_audio_attention(raw_data.at("microphone"));
            attended_data["microphone"] = attended_audio;
        }

        // Process tactile attention
        if (raw_data.count("tactile")) {
            auto attended_tactile = process_tactile_attention(raw_data.at("tactile"));
            attended_data["tactile"] = attended_tactile;
        }

        return attended_data;
    }

private:
    SensorData process_visual_attention(const SensorData& visual_data) {
        // Use saliency maps to focus on important visual regions
        // This could involve object detection, motion detection, etc.
        return visual_data; // Placeholder
    }

    SensorData process_audio_attention(const SensorData& audio_data) {
        // Focus on relevant audio sources (e.g., speaker direction)
        return audio_data; // Placeholder
    }

    SensorData process_tactile_attention(const SensorData& tactile_data) {
        // Focus on important tactile events (e.g., contact, pressure changes)
        return tactile_data; // Placeholder
    }
};
```

## 8.4 Reasoning and Inference Systems

### 8.4.1 Probabilistic Reasoning

```cpp
class ProbabilisticReasoner {
private:
    std::unordered_map<std::string, std::unique_ptr<BayesianNetwork>> knowledge_bases_;
    WorkingMemory* working_memory_;
    EpisodicMemory* episodic_memory_;

public:
    ProbabilisticReasoner(WorkingMemory* wm, EpisodicMemory* em)
        : working_memory_(wm), episodic_memory_(em) {
        initialize_knowledge_bases();
    }

    std::vector<Belief> perform_inference(const Query& query) {
        std::vector<Belief> results;

        // Use different knowledge bases for different types of queries
        auto kb = get_relevant_knowledge_base(query.domain);
        if (kb) {
            auto inference_result = kb->infer(query);
            results.insert(results.end(),
                         inference_result.begin(),
                         inference_result.end());
        }

        // Incorporate episodic memory for contextual reasoning
        auto episodic_inferences = use_episodic_memory(query);
        results.insert(results.end(),
                     episodic_inferences.begin(),
                     episodic_inferences.end());

        return results;
    }

    void update_beliefs(const std::vector<Observation>& observations) {
        for (const auto& obs : observations) {
            auto existing_belief = working_memory_->retrieve<Belief>(obs.type);
            if (existing_belief) {
                // Update belief using Bayes' rule
                auto updated_belief = update_belief(*existing_belief, obs);
                working_memory_->store(obs.type, updated_belief);
            } else {
                // Create new belief
                Belief new_belief = create_belief_from_observation(obs);
                working_memory_->store(obs.type, new_belief);
            }
        }
    }

private:
    std::unique_ptr<BayesianNetwork> get_relevant_knowledge_base(const std::string& domain) {
        auto it = knowledge_bases_.find(domain);
        return (it != knowledge_bases_.end()) ? std::move(it->second) : nullptr;
    }

    std::vector<Belief> use_episodic_memory(const Query& query) {
        // Retrieve similar episodes and extract relevant beliefs
        Episode query_episode;
        query_episode.active_goals = {query.goal};

        auto similar_episodes = episodic_memory_->retrieve_similar_episodes(query_episode);

        std::vector<Belief> beliefs;
        for (const auto& episode : similar_episodes) {
            // Extract beliefs from episode context
            for (const auto& context : episode.environmental_context) {
                beliefs.push_back(context.to_belief());
            }
        }

        return beliefs;
    }

    Belief update_belief(const Belief& prior, const Observation& obs) {
        // Apply Bayes' rule: P(H|E) = P(E|H) * P(H) / P(E)
        double likelihood = calculate_likelihood(obs, prior);
        double posterior = likelihood * prior.confidence / calculate_evidence(obs);

        return Belief{
            prior.content,
            posterior,
            std::chrono::steady_clock::now()
        };
    }

    void initialize_knowledge_bases() {
        // Initialize domain-specific Bayesian networks
        knowledge_bases_["navigation"] = std::make_unique<NavigationBayesNet>();
        knowledge_bases_["object_manipulation"] = std::make_unique<ManipulationBayesNet>();
        knowledge_bases_["social_interaction"] = std::make_unique<SocialBayesNet>();
    }
};

// Knowledge representation using frames
struct Frame {
    std::string name;
    std::unordered_map<std::string, std::any> slots;
    std::vector<Frame> subframes;
    std::vector<Rule> attached_rules;

    template<typename T>
    void set_slot(const std::string& slot_name, const T& value) {
        slots[slot_name] = value;
    }

    template<typename T>
    std::optional<T> get_slot(const std::string& slot_name) const {
        auto it = slots.find(slot_name);
        if (it != slots.end()) {
            try {
                return std::any_cast<T>(it->second);
            } catch (const std::bad_any_cast& e) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
};

class FrameBasedReasoner {
private:
    std::unordered_map<std::string, Frame> frame_memory_;
    std::vector<Rule> production_rules_;

public:
    void add_frame(const Frame& frame) {
        frame_memory_[frame.name] = frame;
    }

    std::vector<Frame> retrieve_frames(const std::string& pattern) {
        std::vector<Frame> matches;

        for (const auto& [name, frame] : frame_memory_) {
            if (std::regex_match(name, std::regex(pattern))) {
                matches.push_back(frame);
            }
        }

        return matches;
    }

    void apply_production_rules() {
        for (const auto& rule : production_rules_) {
            if (rule.condition_met()) {
                rule.execute();
            }
        }
    }
};
```

## 8.5 Planning and Decision Making

### 8.5.1 Hierarchical Task Network (HTN) Planning

```cpp
struct Task {
    std::string name;
    std::vector<std::any> parameters;
    std::vector<Task> subtasks;
    bool is_primitive;
    std::function<bool()> preconditions;
    std::function<void()> effects;
};

class HTNPlanner {
private:
    std::unordered_map<std::string, std::function<std::vector<Task>()>> methods_;
    std::unordered_map<std::string, std::function<bool()>> operators_;

public:
    void add_method(const std::string& task_name,
                   std::function<std::vector<Task>()> method) {
        methods_[task_name] = method;
    }

    void add_operator(const std::string& op_name,
                     std::function<bool()> precondition,
                     std::function<void()> effect) {
        operators_[op_name] = [precondition, effect]() {
            if (precondition()) {
                effect();
                return true;
            }
            return false;
        };
    }

    std::vector<Task> plan(const Task& goal_task) {
        return decompose_task(goal_task);
    }

private:
    std::vector<Task> decompose_task(const Task& task) {
        if (task.is_primitive) {
            return {task};
        }

        auto method_it = methods_.find(task.name);
        if (method_it != methods_.end()) {
            auto subtasks = method_it->second();
            std::vector<Task> plan;

            for (const auto& subtask : subtasks) {
                auto subtask_plan = decompose_task(subtask);
                plan.insert(plan.end(),
                          subtask_plan.begin(),
                          subtask_plan.end());
            }

            return plan;
        }

        throw std::runtime_error("No method found for task: " + task.name);
    }
};

// Integration with cognitive architecture
class CognitivePlanner {
private:
    std::unique_ptr<HTNPlanner> htn_planner_;
    std::unique_ptr<ProbabilisticReasoner> reasoner_;
    WorkingMemory* working_memory_;

public:
    CognitivePlanner(WorkingMemory* wm, EpisodicMemory* em)
        : working_memory_(wm) {
        htn_planner_ = std::make_unique<HTNPlanner>();
        reasoner_ = std::make_unique<ProbabilisticReasoner>(wm, em);

        initialize_planning_methods();
    }

    std::vector<Action> generate_plan(const Goal& goal) {
        // Use reasoning to determine best approach
        auto reasoning_results = reasoner_->perform_inference(
            Query{goal.domain, goal.description, goal}
        );

        // Select appropriate high-level task
        Task high_level_task = select_task_from_reasoning(goal, reasoning_results);

        // Decompose using HTN planning
        auto primitive_tasks = htn_planner_->plan(high_level_task);

        // Convert to executable actions
        std::vector<Action> actions;
        for (const auto& task : primitive_tasks) {
            if (auto action = task_to_action(task)) {
                actions.push_back(*action);
            }
        }

        return actions;
    }

private:
    Task select_task_from_reasoning(const Goal& goal,
                                  const std::vector<Belief>& beliefs) {
        // Use beliefs to select most appropriate task
        // This could involve utility calculations, risk assessment, etc.
        return Task{goal.description, {}, {}, true, nullptr, nullptr};
    }

    std::optional<Action> task_to_action(const Task& task) {
        // Convert primitive task to executable action
        if (task.name == "move_to_location") {
            auto location = std::any_cast<geometry_msgs::msg::Point>(task.parameters[0]);
            return Action{ActionType::NAVIGATE, location};
        } else if (task.name == "grasp_object") {
            auto object = std::any_cast<ObjectInfo>(task.parameters[0]);
            return Action{ActionType::GRASP, object};
        }
        // Add more task-to-action mappings
        return std::nullopt;
    }

    void initialize_planning_methods() {
        // Define methods for high-level tasks
        htn_planner_->add_method("serve_drink", [this]() {
            return std::vector<Task>{
                Task{"navigate_to_kitchen", {}, {}, true, nullptr, nullptr},
                Task{"detect_cup", {}, {}, true, nullptr, nullptr},
                Task{"grasp_cup", {}, {}, true, nullptr, nullptr},
                Task{"navigate_to_person", {}, {}, true, nullptr, nullptr},
                Task{"serve_drink_to_person", {}, {}, true, nullptr, nullptr}
            };
        });

        htn_planner_->add_method("navigate_to_location", [this]() {
            return std::vector<Task>{
                Task{"plan_path", {}, {}, true, nullptr, nullptr},
                Task{"execute_navigation", {}, {}, true, nullptr, nullptr}
            };
        });
    }
};
```

## 8.6 Learning Systems Integration

### 8.6.1 Reinforcement Learning Integration

```cpp
class CognitiveRLSystem {
private:
    std::unique_ptr<DeepQNetwork> dqn_;
    std::unique_ptr<PolicyNetwork> policy_network_;
    std::unique_ptr<ValueNetwork> value_network_;

    WorkingMemory* working_memory_;
    EpisodicMemory* episodic_memory_;

    double learning_rate_{0.001};
    double discount_factor_{0.99};
    double epsilon_{0.1};
    size_t replay_buffer_size_{10000};

    struct Transition {
        State state;
        Action action;
        double reward;
        State next_state;
        bool terminal;
    };

    std::deque<Transition> replay_buffer_;

public:
    CognitiveRLSystem(WorkingMemory* wm, EpisodicMemory* em)
        : working_memory_(wm), episodic_memory_(em) {
        dqn_ = std::make_unique<DeepQNetwork>();
        policy_network_ = std::make_unique<PolicyNetwork>();
        value_network_ = std::make_unique<ValueNetwork>();
    }

    Action select_action(const State& state, bool is_training = true) {
        Action action;

        if (is_training && (double)rand() / RAND_MAX < epsilon_) {
            // Exploration: random action
            action = sample_random_action();
        } else {
            // Exploitation: use learned policy
            action = select_best_action(state);
        }

        return action;
    }

    void learn_from_experience(const State& state, const Action& action,
                             double reward, const State& next_state, bool terminal) {
        // Store transition in replay buffer
        replay_buffer_.push_back({state, action, reward, next_state, terminal});

        if (replay_buffer_.size() > replay_buffer_size_) {
            replay_buffer_.pop_front();
        }

        // Sample batch from replay buffer
        auto batch = sample_batch(32);  // batch size

        // Update networks
        update_networks(batch);

        // Store in episodic memory for higher-level learning
        store_episode_for_memory(state, action, reward, next_state, terminal);
    }

    void incorporate_prior_knowledge() {
        // Use episodic memory to initialize policy
        auto similar_episodes = episodic_memory_->retrieve_similar_episodes(
            Episode{}  // current episode context
        );

        for (const auto& episode : similar_episodes) {
            // Use successful episode transitions for imitation learning
            for (size_t i = 0; i < episode.executed_actions.size(); ++i) {
                if (i + 1 < episode.sensor_data.size()) {
                    auto state = convert_sensor_to_state(episode.sensor_data[i]);
                    auto next_state = convert_sensor_to_state(episode.sensor_data[i + 1]);
                    auto action = episode.executed_actions[i];

                    // Add to replay buffer with high priority
                    replay_buffer_.push_front({
                        state, action, episode.reward, next_state,
                        (i == episode.executed_actions.size() - 1)
                    });
                }
            }
        }
    }

private:
    std::vector<Transition> sample_batch(size_t batch_size) {
        std::vector<Transition> batch;
        std::vector<size_t> indices;

        for (size_t i = 0; i < replay_buffer_.size(); ++i) {
            indices.push_back(i);
        }

        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (size_t i = 0; i < std::min(batch_size, indices.size()); ++i) {
            batch.push_back(replay_buffer_[indices[i]]);
        }

        return batch;
    }

    void update_networks(const std::vector<Transition>& batch) {
        // Update Q-network using DQN algorithm
        // This would involve computing target Q-values and updating network weights
    }

    void store_episode_for_memory(const State& state, const Action& action,
                                double reward, const State& next_state, bool terminal) {
        // Convert to episode format and store
        Episode episode;
        episode.timestamp = std::chrono::system_clock::now();
        episode.sensor_data = {convert_state_to_sensor(state)};
        episode.executed_actions = {action};
        episode.reward = reward;
        episode.success = terminal && reward > 0;

        episodic_memory_->store_episode(episode);
    }
};
```

### 8.6.2 Transfer Learning and Multi-Task Learning

```cpp
class TransferLearningSystem {
private:
    std::unordered_map<std::string, std::unique_ptr<NeuralNetwork>> task_networks_;
    std::unique_ptr<SharedRepresentation> shared_representation_;
    EpisodicMemory* episodic_memory_;

public:
    TransferLearningSystem(EpisodicMemory* em) : episodic_memory_(em) {
        shared_representation_ = std::make_unique<SharedRepresentation>();
    }

    void enable_transfer_learning(const std::string& source_task,
                                const std::string& target_task) {
        // Transfer knowledge from source to target task
        if (task_networks_.count(source_task)) {
            // Extract relevant features from source network
            auto source_features = extract_features(task_networks_[source_task].get());

            // Initialize target network with transferred features
            initialize_with_transferred_features(target_task, source_features);
        }
    }

    void perform_multi_task_learning(const std::vector<Task>& tasks) {
        for (const auto& task : tasks) {
            // Share representations across tasks
            auto shared_features = shared_representation_->get_features();

            // Train task-specific heads while preserving shared representations
            train_task_specific_head(task.name, shared_features);
        }
    }

    double evaluate_transfer_effectiveness(const std::string& source_task,
                                         const std::string& target_task) {
        // Use episodic memory to evaluate transfer effectiveness
        auto source_episodes = episodic_memory_->retrieve_similar_episodes(
            Episode{.active_goals = {Goal{source_task, ""}}}
        );

        auto target_episodes = episodic_memory_->retrieve_similar_episodes(
            Episode{.active_goals = {Goal{target_task, ""}}}
        );

        // Calculate similarity and potential for transfer
        return calculate_transfer_similarity(source_episodes, target_episodes);
    }

private:
    std::vector<Feature> extract_features(NeuralNetwork* network) {
        // Extract features from intermediate layers
        return network->get_intermediate_features();
    }

    void initialize_with_transferred_features(const std::string& task_name,
                                            const std::vector<Feature>& features) {
        // Initialize new network with transferred features
        task_networks_[task_name] = std::make_unique<NeuralNetwork>();
        task_networks_[task_name]->set_initial_features(features);
    }

    double calculate_transfer_similarity(const std::vector<Episode>& source_episodes,
                                       const std::vector<Episode>& target_episodes) {
        // Calculate similarity between task contexts and outcomes
        // This could involve semantic similarity, environmental context, etc.
        return 0.5; // Placeholder
    }
};
```

## 8.7 Implementation Example: SOAR-like Architecture

### 8.7.1 Production System Implementation

```cpp
struct ProductionRule {
    std::string name;
    std::vector<Condition> conditions;
    std::vector<Action> actions;
    int priority{0};

    bool matches(const WorkingMemory& wm) const {
        for (const auto& condition : conditions) {
            if (!condition.evaluate(wm)) {
                return false;
            }
        }
        return true;
    }
};

class ProductionSystem {
private:
    std::vector<ProductionRule> rules_;
    WorkingMemory* working_memory_;
    std::mutex rules_mutex_;

public:
    void add_rule(const ProductionRule& rule) {
        std::lock_guard<std::mutex> lock(rules_mutex_);
        rules_.push_back(rule);
    }

    void run_cycle() {
        std::vector<ProductionRule> applicable_rules;

        // Match phase: find applicable rules
        for (const auto& rule : rules_) {
            if (rule.matches(*working_memory_)) {
                applicable_rules.push_back(rule);
            }
        }

        if (!applicable_rules.empty()) {
            // Select phase: choose rule to fire
            auto selected_rule = select_rule(applicable_rules);

            // Apply phase: execute selected rule
            apply_rule(selected_rule);
        }
    }

private:
    ProductionRule select_rule(const std::vector<ProductionRule>& applicable) {
        // Select rule based on priority, recency, etc.
        auto it = std::max_element(applicable.begin(), applicable.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.priority < b.priority;
                                 });
        return *it;
    }

    void apply_rule(const ProductionRule& rule) {
        for (const auto& action : rule.actions) {
            execute_action(action);
        }
    }

    void execute_action(const Action& action) {
        // Execute action and update working memory
        switch (action.type) {
            case ActionType::ADD_FACT:
                working_memory_->store(action.fact_key, action.fact_value);
                break;
            case ActionType::REMOVE_FACT:
                working_memory_->remove(action.fact_key);
                break;
            case ActionType::MODIFY_FACT:
                working_memory_->update(action.fact_key, action.fact_value);
                break;
        }
    }
};
```

### 8.7.2 Complete Cognitive Architecture Example

```cpp
class HumanoidCognitiveArchitecture : public CognitiveArchitecture {
private:
    std::unique_ptr<MultiModalPerceptionSystem> perception_system_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<ProbabilisticReasoner> reasoner_;
    std::unique_ptr<CognitivePlanner> planner_;
    std::unique_ptr<CognitiveRLSystem> rl_system_;
    std::unique_ptr<ProductionSystem> production_system_;

    WorkingMemory working_memory_;
    EpisodicMemory episodic_memory_;

    std::chrono::time_point<std::chrono::steady_clock> last_cycle_time_;
    std::chrono::milliseconds cycle_period_{100ms}; // 10 Hz

public:
    HumanoidCognitiveArchitecture() {
        memory_system_ = std::make_unique<MemorySystem>();
        perception_system_ = std::make_unique<MultiModalPerceptionSystem>(&working_memory_);
        reasoner_ = std::make_unique<ProbabilisticReasoner>(&working_memory_, &episodic_memory_);
        planner_ = std::make_unique<CognitivePlanner>(&working_memory_, &episodic_memory_);
        rl_system_ = std::make_unique<CognitiveRLSystem>(&working_memory_, &episodic_memory_);
        production_system_ = std::make_unique<ProductionSystem>();

        initialize_production_rules();
        last_cycle_time_ = std::chrono::steady_clock::now();
    }

    void sense() override {
        perception_system_->sense_and_perceive();
    }

    void perceive() override {
        // Process percepts and update beliefs
        auto percepts = working_memory_.retrieve<std::vector<Percept>>("current_percepts");
        if (percepts) {
            reasoner_->update_beliefs(convert_percepts_to_observations(*percepts));
        }
    }

    void reason() override {
        // Apply production rules
        production_system_->run_cycle();

        // Perform probabilistic inference
        auto queries = working_memory_.retrieve<std::vector<Query>>("pending_queries");
        if (queries) {
            for (const auto& query : *queries) {
                auto inferences = reasoner_->perform_inference(query);
                for (const auto& inference : inferences) {
                    working_memory_.store("inference_" + inference.id, inference);
                }
            }
        }
    }

    void plan() override {
        auto goals = working_memory_.retrieve<std::vector<Goal>>("active_goals");
        if (goals && !goals->empty()) {
            auto plan = planner_->generate_plan(goals->front());
            working_memory_.store("current_plan", plan);
        }
    }

    void act() override {
        auto current_plan = working_memory_.retrieve<std::vector<Action>>("current_plan");
        if (current_plan && !current_plan->empty()) {
            auto action = select_action_from_plan(*current_plan);
            execute_action(action);
        }
    }

    void learn() override {
        // Update learning systems
        rl_system_->incorporate_prior_knowledge();

        // Store current episode
        store_current_episode();
    }

    void run() {
        while (true) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_cycle_time_ >= cycle_period_) {
                cycle();
                last_cycle_time_ = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    Action select_action_from_plan(const std::vector<Action>& plan) {
        // Select next action from plan
        auto current_step = working_memory_.retrieve<size_t>("plan_step");
        if (!current_step) {
            working_memory_.store("plan_step", size_t(0));
            current_step = 0;
        }

        if (*current_step < plan.size()) {
            auto action = plan[*current_step];
            working_memory_.store("plan_step", (*current_step) + 1);
            return action;
        }

        // Plan completed, remove it
        working_memory_.remove("current_plan");
        working_memory_.remove("plan_step");
        return Action{ActionType::WAIT, nullptr};
    }

    void store_current_episode() {
        Episode episode;
        episode.timestamp = std::chrono::system_clock::now();

        // Collect current state, actions, and outcomes
        auto percepts = working_memory_.retrieve<std::vector<Percept>>("current_percepts");
        auto actions = working_memory_.retrieve<std::vector<Action>>("executed_actions");
        auto goals = working_memory_.retrieve<std::vector<Goal>>("active_goals");

        if (percepts) episode.sensor_data = convert_percepts_to_observations(*percepts);
        if (actions) episode.executed_actions = *actions;
        if (goals) episode.active_goals = *goals;

        // Determine success based on goal achievement
        episode.success = check_goals_achieved();
        episode.reward = calculate_episode_reward();

        episodic_memory_.store_episode(episode);
    }

    void initialize_production_rules() {
        // Example production rules for humanoid behavior
        production_system_->add_rule({
            "avoid_obstacle",
            {Condition("obstacle_detected", true), Condition("navigation_active", true)},
            {Action{ActionType::ADJUST_PATH, nullptr}},
            10
        });

        production_system_->add_rule({
            "greet_person",
            {Condition("person_detected", true), Condition("social_mode", true)},
            {Action{ActionType::SPEAK, "Hello!"}},
            5
        });

        production_system_->add_rule({
            "maintain_balance",
            {Condition("balance_at_risk", true)},
            {Action{ActionType::ADJUST_POSTURE, nullptr}},
            100
        });
    }
};
```

## 8.8 Integration with ROS 2

### 8.8.1 ROS 2 Cognitive Architecture Node

```python
# cognitive_architecture_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped
from cognitive_msgs.msg import Belief, Goal, Plan
from cognitive_msgs.srv import QueryBeliefs, UpdateGoals

class CognitiveArchitectureNode(Node):
    def __init__(self):
        super().__init__('cognitive_architecture_node')

        # Initialize cognitive architecture components
        self.cognitive_system = HumanoidCognitiveArchitecture()

        # Publishers for cognitive outputs
        self.belief_pub = self.create_publisher(Belief, 'beliefs', 10)
        self.plan_pub = self.create_publisher(Plan, 'plans', 10)
        self.action_pub = self.create_publisher(String, 'actions', 10)

        # Subscribers for sensor inputs
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 'points', self.pointcloud_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Service servers
        self.belief_query_srv = self.create_service(
            QueryBeliefs, 'query_beliefs', self.query_beliefs_callback)
        self.goal_update_srv = self.create_service(
            UpdateGoals, 'update_goals', self.update_goals_callback)

        # Timer for cognitive cycle
        self.cycle_timer = self.create_timer(0.1, self.cognitive_cycle)  # 10 Hz

        self.get_logger().info('Cognitive Architecture Node initialized')

    def image_callback(self, msg):
        """Process image data and update cognitive system"""
        # Convert ROS Image to cognitive system format
        image_data = self.convert_ros_image_to_cognitive_format(msg)

        # Update working memory with visual information
        self.cognitive_system.update_visual_input(image_data)

    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        pointcloud_data = self.convert_ros_pointcloud_to_cognitive_format(msg)
        self.cognitive_system.update_spatial_input(pointcloud_data)

    def imu_callback(self, msg):
        """Process IMU data for balance and orientation"""
        imu_data = self.convert_ros_imu_to_cognitive_format(msg)
        self.cognitive_system.update_balance_input(imu_data)

    def cognitive_cycle(self):
        """Execute one cycle of the cognitive architecture"""
        try:
            # Run cognitive architecture cycle
            self.cognitive_system.cycle()

            # Publish current beliefs
            beliefs = self.cognitive_system.get_current_beliefs()
            for belief in beliefs:
                belief_msg = self.convert_belief_to_ros(belief)
                self.belief_pub.publish(belief_msg)

            # Publish current plan
            plan = self.cognitive_system.get_current_plan()
            if plan:
                plan_msg = self.convert_plan_to_ros(plan)
                self.plan_pub.publish(plan_msg)

            # Publish actions
            actions = self.cognitive_system.get_pending_actions()
            for action in actions:
                action_msg = self.convert_action_to_ros(action)
                self.action_pub.publish(action_msg)

        except Exception as e:
            self.get_logger().error(f'Cognitive cycle error: {e}')

    def query_beliefs_callback(self, request, response):
        """Handle belief queries"""
        try:
            beliefs = self.cognitive_system.query_beliefs(request.query)
            response.beliefs = [self.convert_belief_to_ros(b) for b in beliefs]
            response.success = True
        except Exception as e:
            self.get_logger().error(f'Belief query error: {e}')
            response.success = False

        return response

    def update_goals_callback(self, request, response):
        """Handle goal updates"""
        try:
            goals = [self.convert_ros_goal_to_cognitive(g) for g in request.goals]
            self.cognitive_system.update_goals(goals)
            response.success = True
        except Exception as e:
            self.get_logger().error(f'Goal update error: {e}')
            response.success = False

        return response

    def convert_ros_image_to_cognitive_format(self, ros_image):
        """Convert ROS Image message to cognitive system format"""
        # Implementation depends on specific cognitive system requirements
        return {}

    def convert_belief_to_ros(self, belief):
        """Convert cognitive belief to ROS message"""
        msg = Belief()
        msg.name = belief.name
        msg.confidence = belief.confidence
        msg.timestamp = self.get_clock().now().to_msg()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = CognitiveArchitectureNode()

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

## 8.9 Evaluation and Validation

### 8.9.1 Cognitive Architecture Metrics

```cpp
class CognitiveArchitectureEvaluator {
public:
    struct PerformanceMetrics {
        double task_success_rate{0.0};
        double average_completion_time{0.0};
        double memory_utilization{0.0};
        double reasoning_efficiency{0.0};
        double learning_rate{0.0};
        std::chrono::steady_clock::time_point evaluation_time;
    };

    PerformanceMetrics evaluate_performance(const std::vector<Episode>& episodes) {
        PerformanceMetrics metrics;

        if (episodes.empty()) {
            return metrics;
        }

        // Calculate task success rate
        size_t successful_episodes = 0;
        double total_time = 0.0;

        for (const auto& episode : episodes) {
            if (episode.success) {
                successful_episodes++;
            }
            total_time += std::chrono::duration<double>(episode.completion_time).count();
        }

        metrics.task_success_rate = static_cast<double>(successful_episodes) / episodes.size();
        metrics.average_completion_time = total_time / episodes.size();

        // Calculate reasoning efficiency
        metrics.reasoning_efficiency = calculate_reasoning_efficiency(episodes);

        // Calculate learning rate
        metrics.learning_rate = calculate_learning_improvement(episodes);

        metrics.evaluation_time = std::chrono::steady_clock::now();

        return metrics;
    }

    void log_performance_metrics(const PerformanceMetrics& metrics) {
        std::stringstream ss;
        ss << "Cognitive Architecture Performance:\n";
        ss << "  Task Success Rate: " << (metrics.task_success_rate * 100) << "%\n";
        ss << "  Avg Completion Time: " << metrics.average_completion_time << "s\n";
        ss << "  Reasoning Efficiency: " << metrics.reasoning_efficiency << "\n";
        ss << "  Learning Rate: " << metrics.learning_rate << "\n";

        RCLCPP_INFO(rclcpp::get_logger("cognitive_eval"), "%s", ss.str().c_str());
    }

private:
    double calculate_reasoning_efficiency(const std::vector<Episode>& episodes) {
        // Calculate efficiency as successful inferences per unit time
        size_t total_inferences = 0;
        double total_time = 0.0;

        for (const auto& episode : episodes) {
            // Count reasoning operations and time spent reasoning
            // This would depend on the specific implementation
        }

        return total_inferences / std::max(total_time, 1.0);
    }

    double calculate_learning_improvement(const std::vector<Episode>& episodes) {
        // Calculate improvement over time
        if (episodes.size() < 2) {
            return 0.0;
        }

        // Compare early performance to later performance
        size_t early_count = episodes.size() / 4;
        size_t late_count = episodes.size() / 4;

        double early_success_rate = calculate_success_rate(
            std::vector<Episode>(episodes.begin(), episodes.begin() + early_count));
        double late_success_rate = calculate_success_rate(
            std::vector<Episode>(episodes.end() - late_count, episodes.end()));

        return late_success_rate - early_success_rate;
    }

    double calculate_success_rate(const std::vector<Episode>& episodes) {
        size_t success_count = 0;
        for (const auto& episode : episodes) {
            if (episode.success) {
                success_count++;
            }
        }
        return static_cast<double>(success_count) / episodes.size();
    }
};
```

## 8.10 Best Practices for Cognitive Architecture Design

### 8.10.1 Design Principles

1. **Modularity**: Separate cognitive components for independent development and testing
2. **Scalability**: Design to handle increasing complexity and sensor modalities
3. **Real-time constraints**: Ensure cognitive processes meet timing requirements
4. **Robustness**: Handle sensor failures and unexpected situations gracefully
5. **Explainability**: Maintain traceability of decisions for debugging and trust

### 8.10.2 Implementation Guidelines

1. **Memory Management**: Use efficient data structures and memory pools
2. **Threading**: Separate perception, reasoning, and action threads appropriately
3. **Communication**: Use efficient message passing between components
4. **Logging**: Maintain detailed logs for debugging and analysis
5. **Testing**: Implement comprehensive unit and integration tests

## Summary

Cognitive architectures provide the essential framework for creating intelligent humanoid robots capable of complex, adaptive behavior. By integrating perception, memory, reasoning, planning, and learning systems, cognitive architectures enable robots to operate effectively in dynamic, unstructured environments. The implementation of these architectures requires careful consideration of real-time constraints, memory management, and modularity to ensure scalable and maintainable systems. The integration with ROS 2 provides a robust foundation for deployment in real robotic systems. In the next chapter, we will explore sim-to-real transfer strategies that enable cognitive architectures trained in simulation to operate effectively on physical humanoid robots.