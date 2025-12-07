import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      link: {
        type: 'doc',
        id: 'intro/index',
      },
      items: [
        'intro/course-overview',
        'intro/prerequisites',
        'intro/assessments',
      ],
    },
    {
      type: 'category',
      label: 'Getting Started',
      link: {
        type: 'doc',
        id: 'getting-started/hardware-requirements',
      },
      items: [
        'getting-started/software-setup',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 (The Robotic Nervous System)',
      link: {
        type: 'doc',
        id: 'module-1-ros2/index',
      },
      items: [
        'module-1-ros2/ch01-ros2-architecture',
        'module-1-ros2/ch02-nodes-and-graph',
        'module-1-ros2/ch03-topics-pubsub',
        'module-1-ros2/ch04-services',
        'module-1-ros2/ch05-actions',
        'module-1-ros2/ch06-rclpy-python',
        'module-1-ros2/ch07-urdf-structure',
        'module-1-ros2/ch08-joint-types',
        'module-1-ros2/ch09-launch-files',
        'module-1-ros2/ch10-workspace-tutorial',
        'module-1-ros2/assessment',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      link: {
        type: 'doc',
        id: 'module-2-digital-twin/index',
      },
      items: [
        'module-2-digital-twin/ch01-gazebo-fundamentals',
        'module-2-digital-twin/ch02-sdf-scene-composition',
        'module-2-digital-twin/ch03-physics-properties',
        'module-2-digital-twin/ch04-sensor-plugins',
        'module-2-digital-twin/ch05-unity-mlagents',
        'module-2-digital-twin/ch06-rendering-pipelines',
        'module-2-digital-twin/ch07-ros-unity-bridge',
        'module-2-digital-twin/ch08-collision-validation',
        'module-2-digital-twin/assessment',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac Platform)',
      link: {
        type: 'doc',
        id: 'module-3-ai-robot-brain/index',
      },
      items: [
        'module-3-ai-robot-brain/ch01-isaac-sim-fundamentals',
        'module-3-ai-robot-brain/ch02-synthetic-data-generation',
        'module-3-ai-robot-brain/ch03-isaac-ros-acceleration',
        'module-3-ai-robot-brain/ch04-visual-slam',
        'module-3-ai-robot-brain/ch05-occupancy-grid-mapping',
        'module-3-ai-robot-brain/ch06-nav2-bipedal-navigation',
        'module-3-ai-robot-brain/ch07-behavior-trees',
        'module-3-ai-robot-brain/ch08-cognitive-architectures',
        'module-3-ai-robot-brain/ch09-sim-to-real-transfer',
        'module-3-ai-robot-brain/ch10-integration-tutorial',
        'module-3-ai-robot-brain/assessment',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      link: {
        type: 'doc',
        id: 'module-4-vision-language-action/index',
      },
      items: [
        'module-4-vision-language-action/ch01-whisper-integration',
        'module-4-vision-language-action/ch02-llm-prompt-engineering',
        'module-4-vision-language-action/ch03-natural-language-to-ros-actions',
        'module-4-vision-language-action/ch04-vision-transformers',
        'module-4-vision-language-action/ch05-multimodal-fusion',
        'module-4-vision-language-action/ch06-real-time-inference',
        'module-4-vision-language-action/ch07-end-to-end-vla-systems',
        'module-4-vision-language-action/ch08-integration-tutorial',
        'module-4-vision-language-action/assessment',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      link: {
        type: 'doc',
        id: 'capstone/index',
      },
      items: [
        'capstone/requirements',
        'capstone/implementation-guide',
        'capstone/testing-validation',
        'capstone/submission',
      ],
    },
  ],
};

export default sidebars;