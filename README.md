# fly_brain_ros: ROS 2 Drosophila Connectome Controller for 4-Wheel Skid-Steer Robots

This ROS 2 package integrates the biological whole-brain connectome of the adult fruit fly (*Drosophila melanogaster*, FlyWire v783 with 138k neurons and ~5M synapses) into an autonomous four-wheel differential / skid-steer vehicle control loop.

---

## 1. Package Architecture

```text
fly_brain_ros/
├── package.xml                       # ROS 2 package manifest
├── setup.py                          # Python build script
├── config/
│   └── fly_brain_params.yaml         # Node parameters (gains, thresholds, paths)
├── launch/
│   └── fly_brain_closed_loop.launch.py # ROS 2 launch specification
├── fly_brain_ros/
│   ├── brain_engine.py               # Persistent PyTorch LIF connectome engine
│   ├── sensory_encoder.py            # LaserScan/Distance -> Poisson sensory rates
│   ├── motor_decoder.py              # Descending motor spikes -> 4-wheel cmd_vel
│   ├── safety_supervisor.py          # Watchdog, velocity limits, obstacle override
│   ├── fly_brain_node.py             # ROS 2 Node orchestrator with async worker
│   └── mock_chassis.py               # 4-wheel robot physics & LiDAR simulation
└── test/
    ├── test_encoder.py               # Sensory encoding unit tests
    ├── test_decoder.py               # Motor decoding unit tests
    ├── test_safety.py                # Safety supervisor unit tests
    └── test_end_to_end.py            # Complete closed-loop simulation test
```

---

## 2. Tested & Verified Environment

- **OS**: Ubuntu 22.04 LTS (x86_64)
- **Python**: 3.10 (in `brain-fly` conda environment)
- **Core Libs**: PyTorch 2.4.1+cpu, Brian2 2.9.0, Pandas 2.3.3, PyArrow 25.0.1, NumPy 1.26.4
- **Chassis Target**: 4-wheel skid-steer / differential vehicle ($B = 0.36\text{ m}, r = 0.085\text{ m}$).

---

## 3. How to Run the Tests and Demos

### Run Unit Tests:
```bash
cd /home/om/Desktop/robo_fly/fly_brain_ros
PYTHONPATH=. /home/om/miniconda3/envs/brain-fly/bin/python test/test_encoder.py
PYTHONPATH=. /home/om/miniconda3/envs/brain-fly/bin/python test/test_decoder.py
PYTHONPATH=. /home/om/miniconda3/envs/brain-fly/bin/python test/test_safety.py
```

### Run End-to-End Closed-Loop Test:
```bash
cd /home/om/Desktop/robo_fly/fly_brain_ros
PYTHONPATH=. /home/om/miniconda3/envs/brain-fly/bin/python test/test_end_to_end.py
```

### Build with Colcon (in a ROS 2 workspace):
```bash
colcon build --packages-select fly_brain_ros
source install/setup.bash
ros2 launch fly_brain_ros fly_brain_closed_loop.launch.py
```
