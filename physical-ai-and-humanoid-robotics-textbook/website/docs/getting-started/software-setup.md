# Software Setup

Establishing a robust development environment is key to a successful learning experience in Physical AI and Humanoid Robotics. This section outlines the essential software components and their installation instructions.

## Operating System: Ubuntu 22.04 LTS

**Primary OS**: All tutorials and code examples in this textbook are developed and tested on **Ubuntu 22.04 LTS (Jammy Jellyfish)**. It is highly recommended that you use this specific version to ensure compatibility and avoid common setup issues.

*   **Installation**: If you don't have Ubuntu 22.04 LTS, you can:
    *   Install it as your primary operating system.
    *   Set up a dual-boot system alongside your existing OS.
    *   Use a virtual machine (e.g., VirtualBox, VMware) with Ubuntu 22.04 LTS.
    *   **WSL2/Docker**: While not the primary recommended setup due to potential complexities with GPU pass-through and ROS 2 networking, advanced users may explore Windows Subsystem for Linux 2 (WSL2) or Docker containers with Ubuntu 22.04 for development. Ensure your GPU drivers are correctly configured for these environments if hardware acceleration is needed.

## ROS 2 Humble Hawksbill

**Robotics Middleware**: This textbook exclusively uses **ROS 2 Humble Hawksbill (LTS)**, which is the Long Term Support release compatible with Ubuntu 22.04 LTS. All ROS 2 code examples and tutorials will assume this distribution.

*   **Installation**: Follow the official ROS 2 Humble installation guide for Ubuntu (typically via `apt`). Ensure you install the `ros-humble-desktop` package for a complete development environment.

    ```bash
    sudo apt update && sudo apt install locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8

    sudo apt install software-properties-common
    sudo add-apt-repository universe

    sudo apt update && sudo apt install curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt update
    sudo apt upgrade

    sudo apt install ros-humble-desktop

    # Source ROS 2 setup file (add to .bashrc for persistence)
    source /opt/ros/humble/setup.bash
    ```

## Python 3.10+

**Programming Language**: All code examples are written in **Python 3.10 or newer**, utilizing `rclpy` for ROS 2 interactions. Ensure your system's default Python version or your virtual environment is set accordingly.

*   **Virtual Environments**: It is highly recommended to use Python virtual environments (e.g., `venv`, `conda`) for managing project-specific dependencies.

    ```bash
    sudo apt install python3-venv
    python3 -m venv ~/ros2_ws/install/python_venv # Example for a ROS 2 workspace
    source ~/ros2_ws/install/python_venv/bin/activate
    pip install -U pip
    ```

## Other Tools

*   **Git**: Pre-installed on most Linux systems. Ensure you have a recent version (`git --version`).
*   **VS Code**: A popular IDE with excellent support for Python, C++, ROS, and Markdown. Install recommended extensions for enhanced development.
*   **NVIDIA Drivers**: If using an NVIDIA GPU (Tier 1 or 2), ensure the latest proprietary drivers are installed and correctly configured, especially for CUDA and OptiX (for Isaac Sim).

---