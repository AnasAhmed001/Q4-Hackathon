---
title: Chapter 2 - Synthetic Data Generation & Domain Randomization
description: Learn how to generate synthetic training data using NVIDIA Isaac Sim and domain randomization techniques for humanoid robot AI.
sidebar_position: 24
---

# Chapter 2 - Synthetic Data Generation & Domain Randomization

Synthetic data generation is a critical component of modern AI development for robotics, especially for humanoid robots that require vast amounts of training data to learn complex behaviors and perception tasks. NVIDIA Isaac Sim provides powerful tools for creating high-quality synthetic data with domain randomization techniques that improve the generalization of AI models when deployed to real-world scenarios.

## 2.1 Introduction to Synthetic Data in Robotics

Synthetic data refers to artificially generated data that mimics real-world observations. For humanoid robots, this includes:
- Visual data (images, depth maps, point clouds)
- Sensor data (IMU, force/torque, joint encoders)
- Environmental data (object positions, lighting conditions)
- Ground truth data (object poses, semantic segmentation)

The advantages of synthetic data include:
- **Cost efficiency**: No need for expensive real-world data collection
- **Safety**: Training in virtual environments without physical risk
- **Control**: Ability to create specific scenarios and conditions
- **Volume**: Generation of massive datasets quickly
- **Variety**: Creation of diverse scenarios and edge cases

## 2.2 Domain Randomization Concepts

Domain randomization is a technique that involves randomizing various aspects of the simulation environment to force the AI model to learn features that are invariant to these changes. This improves the model's ability to generalize to real-world conditions.

### 2.2.1 Types of Randomization

1. **Visual Randomization**:
   - Lighting conditions (position, intensity, color)
   - Camera parameters (FOV, noise, distortion)
   - Material properties (color, texture, reflectance)
   - Backgrounds and environments

2. **Physical Randomization**:
   - Friction coefficients
   - Mass and inertia properties
   - Joint dynamics (damping, stiffness)
   - Environmental properties (gravity, wind)

3. **Geometric Randomization**:
   - Object sizes and shapes
   - Placement and positioning
   - Scene layouts and configurations

## 2.3 Isaac Sim Synthetic Data Tools

Isaac Sim provides several tools and extensions for synthetic data generation:

### 2.3.1 Isaac Sim Synthetic Data Extension

This extension provides:
- Camera calibration tools
- Multi-camera setup capabilities
- Sensor fusion utilities
- Annotation generation (bounding boxes, segmentation masks)
- Data export in standard formats

### 2.3.2 Isaac Sim Replicator

Isaac Sim Replicator is a powerful tool for generating large-scale synthetic datasets with domain randomization:

```python
import omni.replicator.core as rep

# Define a basic camera setup
with rep.new_layer():
    # Create a camera
    camera = rep.create.camera(
        position=(0, 0, 1),
        rotation=(0, 0, 0)
    )

    # Create a light
    lights = rep.create.light(
        position=rep.distribution.uniform((-5, -5, -5), (5, 5, 5)),
        intensity=rep.distribution.normal(3000, 500),
        light_type="disk"
    )

    # Create objects with random properties
    with rep.randomizer.frequency(2):
        def randomize_objects():
            # Randomize object positions
            cubes = rep.get.prims_from_path("/World/Cube")
            with cubes:
                rep.modify.pose(
                    position=rep.distribution.uniform((-1, -1, 0), (1, 1, 0.5)),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 3.14))
                )

            # Randomize materials
            materials = rep.get.materials()
            with materials:
                rep.randomizer.color(
                    color=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                )

            return cubes

    # Assign the randomization function
    rep.randomizer.extents_generator(randomize_objects)

# Run the replicator
with rep.trigger.on_frame(num_frames=100):
    rep.orchestrator.run()
```

## 2.4 Implementing Domain Randomization for Humanoid Robots

### 2.4.1 Visual Domain Randomization

For humanoid robots that rely on vision-based perception, visual domain randomization is crucial:

```python
import omni.replicator.core as rep
import numpy as np

def setup_visual_domain_randomization():
    # Randomize lighting conditions
    with rep.new_layer():
        # Randomize environment lighting
        lights = rep.create.light(
            position=rep.distribution.uniform((-10, -10, 5), (10, 10, 15)),
            intensity=rep.distribution.normal(10000, 2000),
            light_type="distant"
        )

        # Randomize light color temperature
        with lights:
            rep.modify.attribute("color", rep.distribution.uniform((0.8, 0.8, 0.8), (1.0, 1.0, 1.0)))

    # Randomize camera properties
    with rep.new_layer():
        camera = rep.create.camera(
            position=(1, 0, 1.5),
            rotation=(0, 0, 0)
        )

        # Add noise to camera
        render_product = rep.create.render_product(camera, (640, 480))
        rep.modify.rand_dist_offset(
            render_product=render_product,
            offset=rep.distribution.uniform(0.0, 0.1)
        )

    # Randomize material properties of objects in the scene
    def randomize_materials():
        # Get all materials in the scene
        materials = rep.get.materials()
        with materials:
            # Randomize base color
            rep.randomizer.color(
                color=rep.distribution.uniform((0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
            )

            # Randomize roughness for PBR materials
            rep.modify.attribute("roughness", rep.distribution.uniform(0.1, 0.9))

            # Randomize metallic for PBR materials
            rep.modify.attribute("metallic", rep.distribution.uniform(0.0, 1.0))

        return materials

    rep.randomizer.extents_generator(randomize_materials)

# Apply the domain randomization
setup_visual_domain_randomization()
```

### 2.4.2 Physical Domain Randomization

For humanoid robots, physical domain randomization helps with sim-to-real transfer:

```python
import omni.replicator.core as rep
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

def setup_physical_domain_randomization():
    # Randomize friction coefficients
    def randomize_friction():
        # Get all rigid bodies in the scene
        rigid_bodies = rep.get.rigid_bodies()
        with rigid_bodies:
            # Randomize friction values
            rep.randomizer.physics_material(
                static_friction=rep.distribution.uniform(0.4, 0.9),
                dynamic_friction=rep.distribution.uniform(0.3, 0.8),
                restitution=rep.distribution.uniform(0.0, 0.2)
            )

        return rigid_bodies

    rep.randomizer.extents_generator(randomize_friction)

    # Randomize mass properties
    def randomize_mass():
        # Get robot links and randomize their mass within reasonable bounds
        robot_links = rep.get.prims_from_path("/World/Robot//")
        with robot_links:
            # Apply small randomization to mass (±10%)
            base_mass = rep.get.mass()
            randomized_mass = base_mass * rep.distribution.uniform(0.9, 1.1)
            rep.modify.mass(mass=randomized_mass)

        return robot_links

    rep.randomizer.extents_generator(randomize_mass)

# Apply physical domain randomization
setup_physical_domain_randomization()
```

## 2.5 Synthetic Data Generation Pipeline

### 2.5.1 Multi-Sensor Data Collection

Humanoid robots often use multiple sensors simultaneously. Isaac Sim can generate synchronized data from various sensors:

```python
import omni.replicator.core as rep
from omni.isaac.synthetic_utils import plot
import carb

def setup_multi_sensor_synthetic_data():
    # Create multiple sensors
    with rep.new_layer():
        # RGB camera
        rgb_camera = rep.create.camera(
            position=(1.0, 0.0, 1.5),
            rotation=(0, 0, 0)
        )

        # Depth camera
        depth_camera = rep.create.camera(
            position=(1.0, 0.0, 1.5),
            rotation=(0, 0, 0)
        )

        # LIDAR sensor
        lidar = rep.create.lidar(
            position=(0.5, 0.0, 1.2),
            rotation=(0, 0, 0)
        )

    # Create render products for each sensor
    rgb_render = rep.create.render_product(rgb_camera, (640, 480))
    depth_render = rep.create.render_product(depth_camera, (640, 480))
    lidar_render = rep.create.render_product(lidar, (640, 480))

    # Annotators for different data types
    with rep.trigger.on_frame():
        # RGB data
        rep.annotators.camera_data(
            render_product=rgb_render,
            name="rgb_data"
        )

        # Depth data
        rep.annotators.camera_data(
            render_product=depth_render,
            name="depth_data"
        )

        # Semantic segmentation
        rep.annotators.camera_annotator(
            render_product=rgb_render,
            name="semantic_segmentation",
            annotator="SemanticSegmentation"
        )

        # Bounding boxes
        rep.annotators.camera_annotator(
            render_product=rgb_render,
            name="bounding_boxes",
            annotator="BoundingBox2DTight"
        )

# Execute the multi-sensor data collection
setup_multi_sensor_synthetic_data()
```

### 2.5.2 Data Annotation and Labeling

Isaac Sim provides various annotation types that are essential for training perception models:

```python
import omni.replicator.core as rep
from omni.replicator import writers

def setup_annotations():
    # Set up render product
    camera = rep.create.camera(position=(0, 0, 2))
    render_product = rep.create.render_product(camera, (640, 480))

    # Semantic segmentation
    with rep.trigger.on_frame():
        rep.annotators.camera_annotator(
            render_product=render_product,
            name="semantic",
            annotator="SemanticSegmentation"
        )

    # Instance segmentation
    with rep.trigger.on_frame():
        rep.annotators.camera_annotator(
            render_product=render_product,
            name="instance",
            annotator="InstanceSegmentation"
        )

    # 2D bounding boxes
    with rep.trigger.on_frame():
        rep.annotators.camera_annotator(
            render_product=render_product,
            name="bbox2d",
            annotator="BoundingBox2DTight"
        )

    # 3D bounding boxes
    with rep.trigger.on_frame():
        rep.annotators.camera_annotator(
            render_product=render_product,
            name="bbox3d",
            annotator="BoundingBox3D"
        )

    # Depth data
    with rep.trigger.on_frame():
        rep.annotators.camera_annotator(
            render_product=render_product,
            name="depth",
            annotator="DistanceToImagePlane"
        )

# Configure the annotation system
setup_annotations()
```

## 2.6 Advanced Domain Randomization Techniques

### 2.6.1 Adaptive Domain Randomization

Instead of randomizing all parameters equally, adaptive domain randomization adjusts the randomization based on the model's performance:

```python
class AdaptiveDomainRandomizer:
    def __init__(self):
        self.performance_history = []
        self.randomization_ranges = {
            'light_intensity': (1000, 10000),
            'friction': (0.3, 0.9),
            'object_size': (0.5, 1.5)
        }
        self.current_ranges = self.randomization_ranges.copy()

    def update_randomization(self, current_performance):
        self.performance_history.append(current_performance)

        # If performance is plateauing, increase randomization range
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            if np.std(recent_performance) < 0.01:  # Performance is plateauing
                self.increase_randomization_range()

    def increase_randomization_range(self):
        for param, (min_val, max_val) in self.current_ranges.items():
            range_width = max_val - min_val
            new_min = max(0, min_val - range_width * 0.1)
            new_max = max_val + range_width * 0.1
            self.current_ranges[param] = (new_min, new_max)

    def get_randomization_distribution(self, param_name):
        min_val, max_val = self.current_ranges[param_name]
        return rep.distribution.uniform(min_val, max_val)

# Usage example
adaptive_randomizer = AdaptiveDomainRandomizer()

def randomize_with_adaptation():
    # Use adaptive randomizer for scene setup
    lights = rep.create.light(
        position=rep.distribution.uniform((-5, -5, 5), (5, 5, 10)),
        intensity=adaptive_randomizer.get_randomization_distribution('light_intensity'),
        light_type="distant"
    )

    return lights
```

### 2.6.2 Curriculum Learning with Domain Randomization

Implementing curriculum learning where the difficulty of randomization increases over time:

```python
class CurriculumDomainRandomizer:
    def __init__(self):
        self.curriculum_stage = 0
        self.max_stages = 5
        self.randomization_schedules = {
            'lighting': [0.1, 0.3, 0.5, 0.7, 0.9],  # Variance in lighting
            'textures': [0.0, 0.2, 0.4, 0.7, 1.0],  # Texture variation
            'physics': [0.0, 0.1, 0.3, 0.5, 0.8]   # Physics parameter variation
        }

    def get_current_randomization_level(self, parameter):
        if self.curriculum_stage >= self.max_stages:
            return self.randomization_schedules[parameter][-1]
        return self.randomization_schedules[parameter][self.curriculum_stage]

    def advance_curriculum(self):
        if self.curriculum_stage < self.max_stages - 1:
            self.curriculum_stage += 1

curriculum_randomizer = CurriculumDomainRandomizer()

def setup_curriculum_randomization():
    # Start with minimal randomization and increase over time
    with rep.new_layer():
        # Lighting randomization based on curriculum stage
        lights = rep.create.light(
            position=rep.distribution.uniform(
                (-5 * curriculum_randomizer.get_current_randomization_level('lighting'),
                 -5 * curriculum_randomizer.get_current_randomization_level('lighting'),
                 5),
                (5 * curriculum_randomizer.get_current_randomization_level('lighting'),
                 5 * curriculum_randomizer.get_current_randomization_level('lighting'),
                 10)
            ),
            intensity=rep.distribution.normal(
                5000,
                1000 * curriculum_randomizer.get_current_randomization_level('lighting')
            )
        )

    return lights
```

## 2.7 Data Export and Format Standards

Isaac Sim can export synthetic data in various formats compatible with popular ML frameworks:

```python
import omni.replicator.core as rep
from omni.replicator import writers

def setup_data_export():
    # Create render product
    camera = rep.create.camera(position=(0, 0, 2))
    render_product = rep.create.render_product(camera, (640, 480))

    # Set up different export writers
    with rep.trigger.on_frame():
        # RGB data writer
        rep.WriterRegistry.get("RgbSchema").attach([render_product])

        # Semantic segmentation writer
        rep.WriterRegistry.get("SemanticSegmentationSchema").attach([render_product])

        # Bounding box writer
        rep.WriterRegistry.get("BoundingBox2D").attach([render_product])

    # Configure export settings
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="./synthetic_data",
        rgb=True,
        semantic_segmentation=True,
        bounding_box_2d_tight=True,
        joint_state=True,
        physics_callback=True
    )

    # Attach to render product
    writer.attach([render_product])
```

## 2.8 Validation and Quality Assurance

### 2.8.1 Synthetic vs. Real Data Comparison

To validate synthetic data quality, compare statistical properties:

```python
import numpy as np
import matplotlib.pyplot as plt

def validate_synthetic_data_quality(synthetic_data, real_data):
    """
    Compare statistical properties of synthetic and real data
    """
    # Compare color distributions
    syn_colors = np.array(synthetic_data['colors'])
    real_colors = np.array(real_data['colors'])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(syn_colors.flatten(), bins=50, alpha=0.5, label='Synthetic', density=True)
    plt.hist(real_colors.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    plt.title('Color Distribution')
    plt.legend()

    # Compare texture properties
    syn_textures = synthetic_data['textures']
    real_textures = real_data['textures']

    plt.subplot(1, 3, 2)
    plt.hist(syn_textures.flatten(), bins=50, alpha=0.5, label='Synthetic', density=True)
    plt.hist(real_textures.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    plt.title('Texture Distribution')
    plt.legend()

    # Compare geometric properties
    syn_shapes = synthetic_data['shapes']
    real_shapes = real_data['shapes']

    plt.subplot(1, 3, 3)
    plt.hist(syn_shapes.flatten(), bins=50, alpha=0.5, label='Synthetic', density=True)
    plt.hist(real_shapes.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    plt.title('Shape Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('data_comparison.png')

    # Calculate statistical distances
    from scipy import stats

    color_ks_stat, color_p_value = stats.ks_2samp(
        syn_colors.flatten(), real_colors.flatten()
    )

    print(f"Color distribution KS statistic: {color_ks_stat:.4f}")
    print(f"Color distribution p-value: {color_p_value:.4f}")

    return color_ks_stat < 0.1  # Return True if distributions are similar enough
```

## 2.9 Best Practices for Effective Domain Randomization

### 2.9.1 Randomization Guidelines

1. **Start Simple**: Begin with minimal randomization and gradually increase complexity
2. **Preserve Physics**: Ensure randomization doesn't break physical laws or robot stability
3. **Monitor Training**: Track model performance to adjust randomization appropriately
4. **Validate Results**: Test models on real data to ensure sim-to-real transfer
5. **Document Changes**: Keep track of randomization parameters for reproducibility

### 2.9.2 Performance Optimization

- Use appropriate scene complexity for training speed
- Optimize rendering settings for faster data generation
- Use GPU acceleration effectively
- Implement efficient data storage and retrieval systems

## Summary

Synthetic data generation and domain randomization are essential techniques for training AI models for humanoid robots. Isaac Sim provides powerful tools for creating diverse, high-quality synthetic datasets that can improve model generalization and enable effective sim-to-real transfer. By implementing systematic domain randomization strategies, you can train more robust perception and control systems for humanoid robots. In the next chapter, we will explore Isaac ROS hardware acceleration and how it integrates with the broader Isaac ecosystem.