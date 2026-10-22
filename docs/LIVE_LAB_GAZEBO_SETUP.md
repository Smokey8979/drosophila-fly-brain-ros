# Gazebo Virtual World Setup & Interface Specification

## 1. Overview

This document specifies the setup, configuration, and interface definition for the 3D **Gazebo Virtual World** designed for closed-loop learning experiments with the embodied *Drosophila melanogaster* connectome.

The simulation environment couples a simulated four-wheel skid-steer / differential vehicle in Gazebo with the persistent spiking Drosophila neural runtime.

---

## 2. Gazebo Robot Model: `four_wheel_fly_robot`

The virtual robot model is defined in both SDF format (`fly_brain_ros/models/four_wheel_fly_robot/model.sdf`) and URDF format (`fly_brain_ros/models/four_wheel_fly_robot/robot.urdf`).

### 2.1 Physical & Kinematic Parameters
- **Chassis Dimensions**: $0.40\text{ m (L)} \times 0.30\text{ m (W)} \times 0.12\text{ m (H)}$
- **Chassis Mass**: $5.0\text{ kg}$ ($I_{xx}=0.05, I_{yy}=0.08, I_{zz}=0.10\text{ kg}\cdot\text{m}^2$)
- **Wheel Geometry**: 4 cylindrical rubber wheels with radius $r = 0.085\text{ m}$ ($85\text{ mm}$), width $w = 0.05\text{ m}$ ($50\text{ mm}$), mass $m_w = 0.8\text{ kg}$ each.
- **Track Width (Wheel Separation)**: $B = 0.36\text{ m}$ ($360\text{ mm}$)
- **Wheelbase (Axle Separation)**: $L = 0.24\text{ m}$ ($240\text{ mm}$)
- **Friction Coefficients**: Surface friction $\mu_1 = 1.0$, $\mu_2 = 0.8$ (providing realistic lateral skid-steer turning resistance).

### 2.2 Simulated Sensors
1. **Planar Panoramic LiDAR (Optic / Looming Sensory Emulation)**:
   - Mounted at $(0.10, 0.0, 0.10)\text{ m}$ relative to base link.
   - Field of View: $270^\circ$ azimuth ($-135^\circ$ to $+135^\circ$), matching the horizontal visual coverage of the fruit fly compound eye.
   - 180 azimuthal ray samples @ $20\text{ Hz}$.
   - Range: $0.10\text{ m}$ to $15.0\text{ m}$.
   - ROS 2 Topic: `/scan` (`sensor_msgs/LaserScan`).
2. **Odometry & Wheel Velocity Encoders**:
   - Computes dead-reckoning pose $(x, y, \theta)$ and linear/angular velocities $(v_x, \omega_z)$.
   - ROS 2 Topic: `/odom` (`nav_msgs/Odometry`).

### 2.3 Actuation Plugins
- **Plugin**: `libgazebo_ros_diff_drive.so`
- **Subscribed Topic**: `/cmd_vel` (`geometry_msgs/Twist`)
- **Control Interface**: Maps linear velocity $v_x$ and angular velocity $\omega_z$ to individual left and right wheel angular speeds:
  $$\omega_{left} = \frac{v_x - \frac{B}{2}\omega_z}{r}$$
  $$\omega_{right} = \frac{v_x + \frac{B}{2}\omega_z}{r}$$

---

## 3. Host System Audit & Dependency Findings

During the Phase 0 audit of the host environment:
1. **ROS 2 Distribution**: System `/opt/ros/humble` was referenced by existing workspace setups (`ros2_ws`), but is not currently present on the filesystem.
2. **Gazebo Installation**: System apt does not have `gazebo` or `ignition-gazebo` installed.
3. **Hardware / Disk Constraints**: The host has an Intel Core i5-10310U CPU with no NVIDIA GPU, and the root filesystem is **$97\%$ full ($8.3\text{ GB}$ free)**. Installing a full ROS 2 desktop + Gazebo 11 from conda-forge requires $10-15\text{ GB}$ of unpack cache, which risks exhausting available disk space.
4. **Architecture Solution**:
   - We provide the complete, valid Gazebo SDF/URDF assets, worlds, and launch files (`live_lab_gazebo.launch.py`) for standard ROS 2/Gazebo deployments.
   - To provide the required **immediate live interactive graphical experience** on this host, we implement a **native real-time graphical simulator and dashboard** running natively on `DISPLAY=:0` (and accessible via a real-time web interface), maintaining identical continuous kinematics, sensor raycasting, and neural closed-loop dynamics.
