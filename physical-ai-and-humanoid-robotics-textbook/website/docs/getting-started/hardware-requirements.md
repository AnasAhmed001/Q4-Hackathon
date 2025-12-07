# Hardware Requirements

To effectively engage with the practical aspects of this textbook, understanding the hardware requirements is crucial. We've outlined three tiers of hardware, catering to different budgets and learning objectives. While some simulations can be run on basic systems, advanced modules (especially those involving NVIDIA Isaac Sim) will require more robust hardware.

## Tier 1: High-Performance Workstation (Recommended for full experience)

This tier provides the optimal experience for all modules, particularly those requiring heavy simulation and AI model training.

*   **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher. An RTX GPU with ray tracing capabilities is essential for NVIDIA Isaac Sim.
*   **CPU**: Intel Core i7 13th Gen / AMD Ryzen 9 equivalent or newer.
*   **RAM**: 64GB DDR5 (minimum recommended for complex simulations).
*   **OS**: Ubuntu 22.04 LTS (primary development environment).
*   **Storage**: 1TB NVMe SSD (minimum, for large datasets and multiple software installations).
*   **Typical Cost**: ~$2500 - $4000

## Tier 2: Edge AI Kit (Recommended for real-world deployment focus)

This tier is ideal for students focusing on deploying AI models to edge devices and real-time robotics. It complements a workstation or cloud setup.

*   **Brain**: NVIDIA Jetson Orin Nano Developer Kit (8GB) or Jetson Orin NX/AGX.
*   **Vision**: Intel RealSense D435i depth camera (or similar).
*   **Audio**: ReSpeaker USB Mic Array v2.0 (or similar for voice commands).
*   **Storage**: 128GB high-speed microSD card or NVMe SSD (for Jetson).
*   **Typical Cost**: ~$700 (excluding workstation)

## Tier 3: Cloud Alternative (For students without high-end local hardware)

For those without access to a high-performance local workstation, cloud GPU instances offer a viable alternative. However, be aware of potential latency issues for real-time control and interaction tasks.

*   **Instance**: AWS g5.2xlarge (or equivalent with NVIDIA A10G GPU, 24GB VRAM).
*   **Cost**: Approximately $1.50/hour (around $205 per quarter if used ~5 hours/day).
*   **Limitations**: Potential network latency for real-time robot control; ensure stable internet connection.

**Note**: While some module assessments *must* involve physical hardware (Tier 2), primarily for real-world deployment and validation, much of the initial development and simulation can be performed on Tier 1 or Tier 3 hardware.