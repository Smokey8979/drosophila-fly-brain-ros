# Drosophila Fly Brain: Whole-Brain Connectome ROS 2 Autonomous Robot Controller

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20Hawksbill-22314E.svg?logo=ros)](https://docs.ros.org/en/humble/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Fortress%206.16%2B-FF6F00.svg)](https://gazebosim.org/docs/fortress)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CPU%20LIF-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FlyWire Connectome](https://img.shields.io/badge/FlyWire-v783%20(138%2C639%20Neurons)-4CAF50.svg)](https://flywire.ai/)
[![Tests](https://img.shields.io/badge/Tests-142%2F142%20Passing-brightgreen.svg)](file:///test/)
[![License](https://img.shields.io/badge/License-GPL--2.0--or--later-blue.svg)](LICENSE)

An embodied, biologically authentic cyber-physical robotics platform integrating the complete whole-brain connectome of the adult fruit fly (*Drosophila melanogaster*, FlyWire v783 with **138,639 neurons** and **~5,000,000 synaptic connections**) into a closed-loop autonomous navigation and learning controller for differential / 4-wheel mobile robots.

The system couples real-time 360° LiDAR sensory raycasting and 3D Gazebo physics with Leaky Integrate-and-Fire (LIF) neural dynamics, biological descending motor circuits, 3-factor Hebbian reinforcement learning, homeostatic internal affect tracking, atomic cryptographic state persistence, and a real-time 6-window interactive web laboratory dashboard.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites & Tested Environment](#prerequisites--tested-environment)
- [Quickstart: Installation & First Run](#quickstart-installation--first-run)
- [The Interactive Live Lab Web Dashboard](#the-interactive-live-lab-web-dashboard)
- [One-Command Runner (`run.sh`)](#one-command-runner-runsh)
- [Biological Connectome Mapping & Key Neurons](#biological-connectome-mapping--key-neurons)
- [Automated Verification & Test Suite](#automated-verification--test-suite)
- [Troubleshooting & FAQs](#troubleshooting--faqs)
- [License & References](#license--references)

---

## System Architecture

```text
                               +-------------------------------------------------------------+
                               |              Gazebo Fortress 3D Simulation                  |
                               |    - Learning Lab Arena (80m x 80m, walls, obstacles)      |
                               |    - 4-Wheel Skid-Steer Drosophila Robot Chassis            |
                               |    - 360-degree LiDAR Sensor & Differential Drive Actuators |
                               +------------------------------+------------------------------+
                                                              |  ros_gz_bridge
                                                              v
+-------------------------------------------------------------------------------------------------------------------------+
|                                              ROS 2 Node (fly_brain_node)                                                |
|                                                                                                                         |
|    +----------------------+       +------------------------------------+       +------------------------------------+   |
|    |    SensoryEncoder    | ----> |          FlyBrainEngine            | ----> |            MotorDecoder            |   |
|    |                      |       |                                    |       |                                    |   |
|    | - 360-deg LiDAR scan |       | - 138,639 LIF Spiking Neurons      |       | - P9 activity -> Linear Vx         |   |
|    | - Distance extraction|       | - 5M Sparse CSR Synapses           |       | - DNa asymmetry -> Angular Wz      |   |
|    | - Poisson spike rates|       | - 0.1ms biological integration dt  |       | - MDN moonwalker -> Reverse Vx     |   |
|    | - Looming LC4 L/R    |       | - Delay buffer tensor (19 slots)   |       | - Differential 4-wheel kinematics  |   |
|    +----------------------+       +-----------------+------------------+       +-----------------+------------------+   |
|                                                     |                                            |                      |
|                                                     v                                            v                      |
|                                   +------------------------------------+       +------------------------------------+   |
|                                   |  Learning & Internal State Subsys  |       |          SafetySupervisor          |   |
|                                   |                                    |       |                                    |   |
|                                   | - 3-Factor Hebbian Plasticity      |       | - Invariant 0.25m Hard Stop Clamp  |   |
|                                   | - PAM Dopamine (+Reward/Sugar)     |       | - Maximum speed & yaw clamps       |   |
|                                   | - PPL1 Dopamine (-Punish/Shock)    |       | - Acceleration rate limiters       |   |
|                                   | - Homeostatic Energy & Fatigue     |       | - Sub-500ms Watchdog Heartbeat     |   |
|                                   | - Episodic Memory & Consolidation  |       | - Deliberate AWAKE Motion Gating   |   |
|                                   +------------------------------------+       +-----------------+------------------+   |
|                                                                                                  |                      |
+--------------------------------------------------------------------------------------------------|----------------------+
                                                                                                   v  /cmd_vel
                                                                                 +------------------------------------+
                                                                                 |  Chassis Motor Drivers & Odometry  |
                                                                                 +------------------------------------+
                                                                                                   |
                                                                    HTTP REST / WebSocket Telemetry| (Port 5000)
                                                                                                   v
+-------------------------------------------------------------------------------------------------------------------------+
|                                    Live Laboratory Web Dashboard (http://127.0.0.1:5000)                                |
|                                                                                                                         |
|  [Window 1: Arena & Robot]     [Window 2: 3D Connectome]      [Window 3: Sensor-Motor Timeline]                         |
|  - Dynamic 2D/3D Arena Grid    - Three.js WebGL Point Cloud   - Real-time Poisson input vs Motor output latency         |
|  - Real-time LiDAR Beams       - 138,639 Colored Neurons      - Dynamic time-correlated sliding chart                   |
|  - Pan, Zoom, Follow-Robot     - 64-bit FlyWire ID Inspector                                                            |
|                                                                                                                         |
|  [Window 4: Learning & Memory] [Window 5: Affect & Stress]    [Window 6: Lifecycle Sleep/Wake]                          |
|  - Plastic Synapse Weights     - Homeostatic Energy Level     - Deliberate Awake default startup                        |
|  - Sugar Reward / Shock Punish - Stress & Sensory Fatigue     - Atomic SHA-256 Checkpoint Commit & Wake                 |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## Repository Structure

```text
fly_brain_ros/
├── package.xml                       # ROS 2 package specification (ament_python)
├── setup.py                          # Python setuptools packaging (modules + static assets)
├── setup.cfg                         # Package script installation rules
├── requirements.txt                  # Python dependencies with strict compatibility bounds
├── requirements.sh                   # Automated multi-step system, pip, and dataset installer
├── requirments.sh                    # Convenience alias symlink to requirements.sh
├── run.sh                            # One-command interactive runner with graceful shutdown
├── config/
│   └── fly_brain_params.yaml         # ROS 2 node parameters (gains, thresholds, rates)
├── launch/
│   ├── live_lab_full.launch.py       # Unified single launch: Gazebo + Bridge + Node + Web Dashboard
│   └── sim_stack.launch.py           # Simulation stack launch (used by lifecycle supervisor)
├── models/
│   └── four_wheel_fly_robot/         # SDF model, meshes, sensors, and chassis kinematics
├── worlds/
│   └── learning_lab_arena.world      # Gazebo Fortress physics arena with perimeter boundaries
├── fly_brain_ros/                    # Core ROS 2 Python package
│   ├── brain_engine.py               # PyTorch LIF spiking connectome engine (138,639 neurons)
│   ├── sensory_encoder.py            # LaserScan/LiDAR depth -> Poisson visual & antennal rates
│   ├── motor_decoder.py              # Biological descending spikes -> differential 4-wheel twist
│   ├── safety_supervisor.py          # Watchdog, velocity limits, invariant 0.25m obstacle stop
│   ├── learning_subsystem.py         # 3-factor Hebbian plasticity, PAM/PPL1 dopamine, episodic memory
│   ├── internal_state.py             # Homeostatic energy, stress, fatigue, and affect tracking
│   ├── checkpoint_config.py          # Cryptographic SHA-256 verification and atomic state persistence
│   ├── connectome_spatial_layout.py  # 3D spatial coordinate atlas generator for 138k neurons
│   ├── neuron_metadata.py            # Biological FlyWire metadata, 64-bit ID registry, neuropils
│   ├── lifecycle_supervisor.py       # Child simulation process manager (Sleep/Wake state machine)
│   ├── fly_brain_node.py             # Main ROS 2 Node orchestrating sensor-brain-motor closed loop
│   ├── mock_chassis.py               # Standalone 4-wheel kinematic physics simulator
│   └── standalone_demo.py            # Lightweight CLI demo running brain engine without ROS 2
├── live_lab/                         # Interactive Web Dashboard
│   ├── web_server.py                 # Flask REST + WebSocket API server (port 5000)
│   ├── interactive_experiments.py    # Automated curricula (A-F: forward, turn, avoid, memory)
│   ├── laboratory_gui.py             # Local Tkinter experimental GUI
│   └── static/
│       └── three.min.js              # Bundled Three.js WebGL library (zero external CDN dependency)
├── virtual_world/                    # Embodied biological learning simulation layer
│   ├── arena.py                      # Virtual 2D/3D arena with odor plumes, landmarks, food/shock
│   ├── sensors.py                    # Vectorized sensory models (antennae, compound eye raycaster)
│   ├── actuators.py                  # Descending-to-kinematic motor decoder
│   ├── closed_loop_agent.py          # Embodied agent coupling brain engine, synapses, and arena
│   ├── visual_circuit.py             # Optic lobe motion detector (Elementary Motion Detectors)
│   └── plasticity.py                 # Dopaminergic synaptic weight modulation
├── test/                             # Comprehensive test suite (142 unit & integration tests)
│   ├── test_3d_connectome_viewer_telemetry.py         # WebGL 3D point cloud & FlyWire ID inspection
│   ├── test_checkpoint_authoritative_verification.py  # SHA-256 hash & schema 2.x persistence
│   ├── test_decoder.py                                # Motor decoding (P9, DNa, MDN kinematics)
│   ├── test_encoder.py                                # Sensory encoding (LiDAR -> Poisson rates)
│   ├── test_end_to_end.py                             # Full 138k closed-loop execution cycle
│   ├── test_hard_stop_invariance.py                   # Invariant 0.25m obstacle override
│   ├── test_learning_and_memory.py                    # Plastic synapses & episodic consolidation
│   ├── test_neural_telemetry.py                       # Connectome telemetry & neuron lookup
│   ├── test_persistence_checkpoints.py                # Delay buffer & weights round-trip
│   ├── test_reinforcement_learning_controls.py        # PAM reward & PPL1 punishment API
│   ├── test_ros_node_integration.py                   # ROS 2 topics, lifecycle, and callbacks
│   ├── test_safety.py                                 # Safety supervisor clamp invariance
│   ├── test_simulated_robot_arena_visualization.py    # Window 1 arena, camera, LiDAR tests
│   ├── test_sleep_wake_lifecycle.py                   # Supervisor sleep/wake state transitions
│   ├── test_sleep_wake_motion_gate.py                 # Motion gating and zero-command clamp
│   └── test_startup_lifecycle_motion_gate.py          # Deliberate AWAKE startup enforcement
└── docs/                             # Comprehensive architecture & integration documentation
```

---

## Prerequisites & Tested Environment

The system has been designed, validated, and verified under:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish) x86_64
- **ROS 2 Distribution**: ROS 2 Humble Hawksbill (`ros-humble-desktop` or `ros-humble-ros-base`)
- **Physics Simulator**: Gazebo Fortress 6.16+ (`ros-humble-ros-gz`)
- **Python**: 3.10.12+ (aligned with ROS 2 Humble system Python `/usr/bin/python3`)
- **PyTorch**: 2.0+ (CPU or CUDA; uses PyTorch sparse CSR tensors)
- **NumPy**: Must be `>=1.21.5, <2.0.0` (NumPy 2.x breaks PyTorch C-API umath extensions on Python 3.10)
- **RAM**: Minimum 8 GB (16 GB recommended for whole-brain 138k neural simulation)

---

## Quickstart: Installation & First Run

### Step 1: Clone the Repository into your ROS 2 Workspace

```bash
mkdir -p ~/robo_fly_ros_ws/src
cd ~/robo_fly_ros_ws/src
git clone https://github.com/Smokey8979/drosophila-fly-brain-ros.git fly_brain_ros
cd fly_brain_ros
```

### Step 2: Run the Automated Setup Script

The automated setup script installs required system libraries, installs compatible Python packages, links or downloads the 138k Drosophila connectome dataset from upstream, generates the 3D spatial layout coordinates, and initializes the baseline persistent checkpoint:

```bash
./requirements.sh
```

*(Note: `./requirments.sh` is provided as an alias for spelling convenience).*

### Step 3: Build the ROS 2 Workspace

```bash
cd ~/robo_fly_ros_ws
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Step 4: Launch the Full Simulation

From the repository directory:

```bash
cd ~/robo_fly_ros_ws/src/fly_brain_ros
./run.sh
```

To run without Gazebo 3D GUI client (ideal for servers or low CPU/GPU overhead):

```bash
./run.sh headless:=true
```

### Step 5: Open the Interactive Web Dashboard

Once launched, open your web browser to:

👉 **`http://127.0.0.1:5000`**

---

## The Interactive Live Lab Web Dashboard

The web dashboard is organized into 6 synchronized, real-time telemetry windows:

```
+------------------------------------+------------------------------------+
|  Window 1: Simulated Robot & Arena |  Window 2: 3D Connectome Viewer    |
|  - Dynamic 2D/3D physics arena     |  - WebGL Three.js point cloud      |
|  - Live robot footprint & heading  |  - 138,639 neurons color by region |
|  - Real-time 360-deg LiDAR beams   |  - Click-to-inspect FlyWire 64-bit |
|  - Pan, Zoom, Follow-Robot controls|  - Real-time spike rate heatmap    |
+------------------------------------+------------------------------------+
|  Window 3: Sensor-Motor Timeline   |  Window 4: Reinforcement Learning  |
|  - Visual looming threat frequency |  - Plastic synaptic weight matrix  |
|  - Forward drive vs turning yaw    |  - PAM Reward & PPL1 Punishment    |
|  - Millisecond sensor-motor latency|  - Episodic memory event history   |
+------------------------------------+------------------------------------+
|  Window 5: Simulated Affect/State  |  Window 6: Lifecycle Sleep & Wake  |
|  - Homeostatic energy expenditure  |  - Deliberate AWAKE status display |
|  - Internal stress & sensor fatigue|  - Sleep memory consolidation     |
|  - Motor readiness indicator       |  - Cryptographic SHA-256 state save|
+------------------------------------+------------------------------------+
```

### 1. Window 1: Simulated Robot & Arena
- **Dynamic Camera Controls**: Pan by dragging the canvas, Zoom with mouse wheel or +/- buttons, Reset View to re-center the arena.
- **Follow Robot**: Toggleable button that locks the viewport onto the moving robot as it explores the arena.
- **Accurate LiDAR Beams**: Live raycasted beams extending from the robot chassis to arena walls and obstacles, color-coded green (clear), yellow (caution), and red (danger).
- **Clean Visual Aesthetics**: Unwanted circular markers removed; only authentic physical obstacles, perimeter walls, sensor origins, and robot trajectory trails are rendered.

### 2. Window 2: Scientific 3D Connectome Viewer
- **Whole-Brain Point Cloud**: Renders all 138,639 neurons in 3D WebGL (Three.js) based on biological spatial coordinates.
- **Neuropil Segmentation**: Neurons are color-coded by biological compartment (Optic Lobes, Central Complex, Mushroom Body, Subesophageal Zone, Antennal Lobes, Descending System).
- **FlyWire 64-bit Inspection**: Click on any neuron or enter an ID/alias to inspect its 64-bit string FlyWire ID (e.g., `720575940618956972`), biological label, functional classification, coordinates, and live spike rate.

### 3. Window 3: Time-Correlated Sensory-to-Motor Timeline
- Visualizes incoming sensory Poisson rate spikes (LC4 looming aversion, P9 forward drive) correlated against published chassis linear velocity ($V_x$) and angular rate ($W_z$).

### 4. Window 4: Reinforcement Learning & Memory
- **Interactive Teaching**: Deliver instantaneous PAM dopaminergic reward (simulating sucrose taste contact) or PPL1 dopaminergic punishment (simulating obstacle collision or electric shock).
- **Synaptic Plasticity**: Monitors real-time weight adaptation governed by 3-factor Hebbian learning:
  $$\Delta W_{ij} = \eta \cdot (\text{DA} - \text{DA}_0) \cdot \text{Pre}_i \cdot \text{Post}_j - \lambda W_{ij}$$

### 5. Window 5: Simulated Internal Affect & Stress
- Tracks simulated biological metabolic energy expenditure, sensory fatigue, obstacle stress, and overall motor arousal.

### 6. Window 6: Lifecycle Sleep/Wake Controls
- **Sleep**: Gates motor output immediately to $0.0\,\text{m/s}$, consolidates short-term episodic memory traces into permanent synaptic structure, and saves a cryptographically verified atomic checkpoint.
- **Wake**: Seamlessly re-awakens the neural pipeline from the persistent checkpoint, ensuring zero state regression across restarts.
- **Deliberate Awake Default**: The robot starts AWAKE by design upon launching, separating long-term model parameters from the transient operational state.

---

## One-Command Runner (`run.sh`)

The root `run.sh` script manages process initialization, DDS discovery, logging, and graceful shutdown:

```bash
./run.sh [OPTIONS]
```

### Available Options

| Option | Default | Description |
|---|---|---|
| `headless:=true` | `false` | Runs Gazebo Fortress in server mode (`-s`) without 3D GUI client |
| `use_sim_time:=true` | `false` | Synchronizes ROS 2 clock with Gazebo simulation clock |
| `dashboard_port:=PORT` | `5000` | Overrides the HTTP web dashboard listening port |
| `--help`, `-h` | - | Displays usage instructions and available options |

### Graceful Safe Shutdown

To stop the simulation safely without corrupting brain weights or leaving orphaned background processes:

- **Option A (Interactive)**: Type `close` and press **Enter** in the runner terminal.
- **Option B (Signal)**: Press **Ctrl+C**.

The shutdown sequence executes the following operations:
1. Halts robot motion through the safety-supervised command path ($V_x = 0, W_z = 0$).
2. Commands the lifecycle supervisor to trigger neural sleep and memory consolidation.
3. Flushes an atomic checkpoint to `checkpoints/fly_brain_checkpoint.json`.
4. Computes and validates the SHA-256 cryptographic digest of the written checkpoint.
5. Terminates child process groups cleanly (`ros_gz_bridge`, `ign gazebo`, `fly_brain_node`, `web_server`).

---

## Biological Connectome Mapping & Key Neurons

Every key sensory, interneuron, and motor driver in this controller maps to authentic biological neurons documented in the FlyWire whole-brain connectome:

| Biological Symbol | 64-bit FlyWire ID | Macro-Neuropil | Biological Function & Role in Controller |
|---|---|---|---|
| **DNa01 (Left)** | `720575940628236165` | Descending / GNG | Steering motor driver: drives left yaw rotation |
| **DNa01 (Right)**| `720575940618956972` | Descending / GNG | Steering motor driver: drives right yaw rotation |
| **DNa02 (Left)** | `720575940632943265` | Descending / GNG | Sharp ipsilateral steering control |
| **DNa02 (Right)**| `720575940627798317` | Descending / GNG | Sharp contralateral steering control |
| **P9 (Left)** | `720575940624021596` | Descending / GNG | Forward walking drive: proportional to linear velocity $V_x$ |
| **P9 (Right)** | `720575940626356740` | Descending / GNG | Forward walking drive: proportional to linear velocity $V_x$ |
| **MDN (1..4)** | `720575940624239857`+ | Descending / GNG | Moonwalker Descending Neurons: reverse locomotion & halt |
| **Giant Fiber (1/2)**| `720575940623238612`+ | Descending / GNG | Fast escape jump response upon critical looming stimulus |
| **LC4 (Left/Right)**| 22 authentic IDs | Optic Lobe (Lobula) | Visual Looming Threat Detectors: converts LiDAR range to Poisson spikes |
| **PAM Clusters** | 12 authentic IDs | Mushroom Body | Dopaminergic Reward Neurons: activates 3-factor Hebbian LTP |
| **PPL1 Clusters**| 8 authentic IDs | Mushroom Body | Dopaminergic Punishment Neurons: activates LTD / aversion |
| **Sugar GRNs** | 8 authentic IDs | SEZ (Subesophageal) | Gustatory Receptor Neurons: detects sweet rewards |
| **Bitter GRNs** | 6 authentic IDs | SEZ (Subesophageal) | Gustatory Receptor Neurons: detects deterrents/toxins |
| **JO_EV** | 10 authentic IDs | Antennal Lobe | Johnston's Organ mechanosensory wind/collision detectors |

---

## Automated Verification & Test Suite

The repository contains **142 automated unit and integration tests** verifying all sub-components:

```bash
cd ~/robo_fly_ros_ws/src/fly_brain_ros
unset RMW_IMPLEMENTATION
python3 -m pytest -p no:anyio test/
```

### Test Suite Breakdown

| Test Module | Tests | Verified Subsystems & Behavioral Guarantees |
|---|---|---|
| `test_simulated_robot_arena_visualization.py` | 25 | Window 1 camera pan/zoom/follow, dynamic grid, LiDAR rays, marker cleanup |
| `test_safety.py` | 16 | Velocity bounds, acceleration limits, watchdog timeouts, zero-motion clamp |
| `test_startup_lifecycle_motion_gate.py` | 10 | Enforces deliberate AWAKE startup default; rejects stale startup sleep |
| `test_checkpoint_authoritative_verification.py`| 11 | Schema 2.x completeness, SHA-256 verification, atomic file replacement |
| `test_ros_node_integration.py` | 10 | ROS 2 topics (`/scan`, `/cmd_vel`), QoS contracts, heartbeat monitoring |
| `test_3d_connectome_viewer_telemetry.py` | 10 | 3D Float32 binary buffer streaming, FlyWire ID lookup, neuropil queries |
| `test_neural_telemetry.py` | 10 | Real-time spike rate sampling, active subnetwork identification |
| `test_reinforcement_learning_controls.py` | 9 | PAM reward and PPL1 punishment REST endpoints and weight changes |
| `test_sleep_wake_motion_gate.py` | 8 | Immediate motor clamping during sleep; graceful restoration during wake |
| `test_learning_and_memory.py` | 7 | Proactive safety margins, episodic memory consolidation during sleep |
| `test_persistence_checkpoints.py` | 7 | Delay buffer head pointer persistence, state serialization round-trip |
| `test_hard_stop_invariance.py` | 6 | Invariant $0.25\,\text{m}$ hardware override regardless of neural commands |
| `test_encoder.py` | 4 | LiDAR range to LC4/P9 Poisson conversion and spatial binning |
| `test_decoder.py` | 4 | P9 linear gain, DNa turning gain, and MDN reverse walking kinematics |
| `test_sleep_wake_lifecycle.py` | 4 | Supervisor state machine transitions (AWAKE -> SLEEP -> WAKE) |
| `test_end_to_end.py` | 1 | Complete closed-loop step (LiDAR -> Encoder -> Brain -> Decoder -> Chassis) |
| **Total** | **142** | **100% Passed (0 Failures, 0 Errors)** |

---

## Troubleshooting & FAQs

### Q1: `librmw_cyclonedds_cpp.so: cannot open shared object file`
**Cause**: The environment variable `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` is set, but CycloneDDS is not installed.  
**Fix**: Run `unset RMW_IMPLEMENTATION` in your terminal, or place `unset RMW_IMPLEMENTATION` in your `~/.bashrc`. Both `run.sh` and `requirements.sh` handle this automatically.

### Q2: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
**Cause**: PyTorch C-API extensions on Python 3.10 / Ubuntu 22.04 LTS require NumPy 1.x.  
**Fix**: Run `pip install "numpy>=1.21.5,<2.0.0"`. `requirements.txt` strictly enforces this bound.

### Q3: `Port 5000 is already in use`
**Cause**: An earlier simulation instance or other web application is holding port 5000.  
**Fix**: Specify an alternate port when running:
```bash
./run.sh dashboard_port:=5001
```
Or terminate the holding process: `lsof -ti:5000 | xargs kill -9 2>/dev/null || true`.

### Q4: Running on a Remote Server or Cloud VM Without a Monitor
**Cause**: Gazebo 3D client requires an X11/GLX display server.  
**Fix**: Launch in headless mode:
```bash
./run.sh headless:=true
```
The physics engine, LiDAR raycasting, neural brain, and web dashboard will run normally at full speed. Access the dashboard from your local browser by port-forwarding: `ssh -L 5000:localhost:5000 user@server_ip`.

---

## License & References

### License
This project is licensed under the **GNU General Public License v2.0 or later** ([GPL-2.0-or-later](LICENSE)).

### Scientific Citations & Acknowledgments
The biological wiring and whole-brain connectome utilized in this project are based on the seminal work of the FlyWire Consortium and the Princeton Neuroscience Institute:

1. **Dorkenwald, S. et al.** (2024). *Neuronal wiring diagram of an adult brain*. **Nature**, 634, 124–138. [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y).
2. **Schlegel, P. et al.** (2024). *A consensus cell type atlas of the adult Drosophila melanogaster brain*. **Nature**, 634, 139–152. [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5).
3. Connectome dataset materialization and PyTorch sparse matrix reference provided by Eon Systems PBC: [github.com/eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain).
