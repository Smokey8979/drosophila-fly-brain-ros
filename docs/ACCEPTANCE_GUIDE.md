# Drosophila Fly Brain Acceptance & Manual Verification Guide

## 1. System Overview & Architecture

The **Drosophila Fly Brain Robot Simulation** integrates a 138,639-neuron biological connectome (derived from Shiu et al. 2024 / FlyWire MaleCNS v783) with Gazebo Fortress physics, ROS 2 Humble control nodes, a Three.js WebGL 3D anatomical viewer, an interactive 2D arena and robot visualization, an internal affect model, bounded reinforcement learning, and persistent sleep/wake lifecycles.

---

## 2. Quickstart: One-Command Execution

### Launch Environment
Start the entire simulation, Gazebo physics, ROS bridge, whole-brain neural engine, and live dashboard using a single command:

```bash
cd /home/om/Desktop/robo_fly
./run.sh
```

For headless execution (simulator running without GUI display):
```bash
./run.sh headless:=true
```

The script will build prerequisites if needed, start the launch stack, wait for system readiness, and open `http://127.0.0.1:5000`.

### Safe Shutdown Verification
To cleanly shut down the simulation without corrupting neural state, simply type `close` into the running terminal and press Enter (or press `Ctrl+C`):

```text
close
```

Expected terminal output:
```text
============================================================
 Drosophila Fly Brain — Shutdown Summary
============================================================
 Neural State Saved:     YES (Cryptographically Verified)
 Checkpoint Path:        /home/om/Desktop/robo_fly/checkpoints/fly_brain_checkpoint.json
 Checkpoint Size:        ~540 KB
 SHA-256 Digest:         <64-hex SHA-256 hash>
 Schema Version:         v2.0.0
 Shutdown Trigger:       USER_CLOSE (or SIGINT)
 Owned Processes:        Terminated cleanly
============================================================
```

---

## 3. Web Dashboard Acceptance Procedures

Open **`http://127.0.0.1:5000`** in Google Chrome or any WebGL2-compatible browser.

### A. Window 1: Simulated Robot & Arena Professional Visualization
Window 1 provides an interactive, engineering-grade 2D robotics visualization of the robot in its environment.

1. **Interactive Camera Controls**:
   * **Mouse Wheel Zoom**: Smoothly zooms in or out while strictly preserving the exact world coordinate beneath the mouse cursor. Clamped safely between `2.0 px/m` (macro 150m field view) and `120.0 px/m` (micro chassis inspection).
   * **Click-and-Drag Pan**: Click and drag anywhere on the canvas to inspect distant arena sections. Panning smoothly moves the camera without snapping back, and automatically disengages camera tracking (`Follow: OFF`).
   * **Double-Click Centering**: Double-click on any world point or obstacle to smoothly center the camera at that coordinate.
   * **Keyboard Shortcuts**:
     * `+` or `=`: Zoom In
     * `-` or `_`: Zoom Out
     * `F` or `f`: Toggle Follow Robot mode
     * `C` or `c`: Center on Robot
     * `R` or `r`: Reset View
     * `A` or `a`: Fit Arena in viewport

2. **Camera & Layer Toolbar**:
   * **`Zoom In (+)` / `Zoom Out (-)`**: Precise geometric scaling with instant HUD readout.
   * **`Reset`**: Restores default camera center $(0.0, -6.0)$, scale `16.0 px/m`, and re-engages robot tracking (`Follow: ON`).
   * **`Center`**: Immediately re-centers the viewport on the current $(X, Y)$ position of the robot without altering zoom level.
   * **`Follow: ON/OFF`**: Toggles automatic tracking. When enabled, camera centers continuously on the robot; when disabled, the camera remains locked in place while the robot navigates.
   * **`Fit Arena`**: Automatically computes optimal zoom scale and centers at $(0.0, 0.0)$ to fit the entire $20\text{m}\times 20\text{m}$ arena with safety margin into the canvas.
   * **Layer Toggles (`Rays`, `Hits`, `Trail`, `Grid`, `Walls`)**: Instant toggle buttons to independently show or hide LiDAR scan beams, obstacle hit points, odometry trail history, dynamic coordinate grid lines, and physical arena boundary walls.

3. **Dynamic Infinite World Grid**:
   * **Continuous Extents**: Grid lines dynamically calculate from visible world bounds, expanding without bounds as the robot travels. The robot and environment never disappear or clip beyond boundaries.
   * **Adaptive Step Sizing**: Automatically adapts grid subdivisions based on zoom level (e.g. 20m/5m for macro view, 5m/1m for standard view, 1m/0.2m for close zoom).
   * **World Origin Indicator**: Origin $(0, 0)$ is rendered with distinct Red $+X$ (East) and Green $+Y$ (North) coordinate crosshairs.

4. **Accurate LiDAR Geometry & Obstacle Discrimination**:
   * **Sensor Origin Offset**: Ray origins accurately originate from $(x + 0.10\cos\theta, y + 0.10\sin\theta)$, correctly reflecting the physical $+0.10\text{m}$ forward LiDAR mount in `model.sdf`.
   * **Hit vs. Free-Space Distinction**: Obstacle returns ($r < 0.98\cdot r_{max}$) render as distinct glowing red hit markers; free-space scan lines render as subtle, non-distracting translucent beams.
   * **Stale Scan Handling**: If scan data ceases for $>1.5\text{s}$, rays gracefully dim with a prominent `STALE` status indicator in the HUD.

5. **Authentic 4-Wheel Robot Footprint & Live Telemetry HUD**:
   * **Chassis & Wheel Model**: Drawn to exact physical dimensions ($0.40\text{m}$ length, $0.30\text{m}$ width) with 4 distinct drive wheels at $(\pm 0.12\text{m}, \pm 0.18\text{m})$, cylindrical sensor turret at $+0.10\text{m}$, and bright forward heading indicator.
   * **Telemetry HUD Overlay**: Real-time on-canvas display of Robot Pose $(X, Y, \theta)$, Linear velocity $v$, Angular velocity $\omega$, Wheel velocities $(v_L, v_R)$, Motion state badges (`AWAKE`, `MOVING`, `ASLEEP`), LiDAR statistics (total rays, valid hits, freshness), Safety state, and viewport scale.

---

### B. Scientific 3D Connectome Viewer
1. **Biological Neuropil Segmentation**:
   - The viewer renders all 138,639 neurons partitioned into 7 authentic anatomical neuropils:
     * **Optic Lobes (Left/Right)**: Lobula, Medulla, and LC4 looming motion detectors (Cyan).
     * **Central Complex (CX)**: Protocerebral bridge and ellipsoid body for heading orientation (Gold).
     * **Antennal / Johnston's Organ (AL/JO)**: Mechanosensory antennal inputs (Purple).
     * **Subesophageal Zone (SEZ)**: Taste gustatory receptors and feeding motor neurons (Coral).
     * **Descending Motor Tracts (DN)**: DNa01 steering, P9 walking, MDN reverse, and Giant Fiber escape neurons (Bright Green).
     * **Central Brain Intrinsic (CB)**: Protocerebrum and mushroom bodies (Muted Slate).
2. **Interactive 3D Camera Controls**:
   - **Rotate**: Left-click drag.
   - **Pan**: Right-click drag.
   - **Zoom**: Mouse wheel.
   - **Preset Views**: Click **Front (Anterior)**, **Dorsal (Top)**, **Lateral (Side)**, or **Reset Camera** buttons.
3. **Real-Time Spike Propagation**:
   - As the simulated robot approaches obstacles or turns, active spiking neurons flash bright neon gold with exponential decay.

---

### C. Exact 64-bit FlyWire Neuron Inspector
1. In the **FlyWire Neuron Search** box in the left sidebar, enter an authentic 64-bit ID or biological symbol:
   * Symbol: `DNa01_left` (or ID: `720575940644438551`)
   * Symbol: `P9_ODN1_left` (or ID: `720575940620300308`)
   * Symbol: `LC4_left_01` (or ID: `720575940605598892`)
2. Click **Inspect** or press Enter.
3. **Verify Precision**:
   - Ensure the rendered ID in the inspector displays exact string `720575940644438551` and **NOT** truncated to `720575940644438500`.
   - The 3D viewer automatically flies the camera to the anatomical coordinate of the neuron and marks it with a glowing inspection indicator ring.
   - Physiological metrics (membrane voltage in mV, functional group, and pre/post synaptic partner weights) are rendered.

---

### D. Sleep / Wake Lifecycle & Hard Motion Gate
1. Click **Put to Sleep** in the Lifecycle panel.
   * **Visual Indicator**: Dashboard changes state to `"ASLEEP"` with purple night styling.
   * **Motion Gate Verification**: Robot physical velocity drops to strictly `0.0 m/s`. Teleoperation commands and neural motor outputs are clamped to zero at the authoritative `/cmd_vel` gate.
   * **Checkpoint Flush**: Memory consolidation is triggered and an atomic checkpoint is committed to `/home/om/Desktop/robo_fly/checkpoints/fly_brain_checkpoint.json`.
2. Click **Wake Brain**:
   * Dashboard returns to `"AWAKE"` state with green status badge.
   * Neural simulation and motion safely resume.

---

### E. Reinforcement Learning Console
1. **Manual Reward**: Click **+ Reward (+0.5)** or **Max Reward (+1.0)**.
   * **Affect Response**: Valence Affect meter increases, Confidence increases, Environmental Vigilance settles.
   * **Policy Tuning**: Proactive steer distance expands (`delta_steer_dist > 0.0`), allowing earlier avoidance around obstacles.
2. **Manual Punishment**: Click **- Punish (-0.5)** or **Max Punish (-1.0)**.
   * **Affect Response**: Vigilance and Uncertainty increase, Valence Affect decreases.
   * **Policy Tuning**: Clutter speed factor scales down (`speed_factor < 1.0`), slowing the robot in dangerous proximity.
3. **Reset Baseline**: Click **Reset Policy** to restore neutral parameters.

---

## 4. Authoritative Checkpoint Verification

The single authoritative checkpoint is stored at:
```text
/home/om/Desktop/robo_fly/checkpoints/fly_brain_checkpoint.json
```

Verify its cryptographic integrity and completeness at any time:
```bash
python3 -c "
import json, hashlib
with open('/home/om/Desktop/robo_fly/checkpoints/fly_brain_checkpoint.json') as f:
    data = json.load(f)
h = data['manifest'].pop('sha256')
calc = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
assert h == calc, 'Hash mismatch!'
print(f'VALID CHECKPOINT: Schema {data[\"manifest\"][\"schema_version\"]}, SHA-256: {h}')
"
```

---

## 5. Automated Test Suite Execution

Run the complete 137-test automated verification suite:
```bash
cd /home/om/Desktop/robo_fly/fly_brain_ros
PYTHONPATH="/home/om/.local/lib/python3.10/site-packages:$PYTHONPATH" python3 -m pytest -p no:anyio test/
```

### Complete Test Suite Inventory (137 Tests / 100% Passing):
1. `test_simulated_robot_arena_visualization.py` (20 tests): Camera zoom clamping, cursor-centered mouse-wheel zoom invariant, pan world delta invariance, follow robot toggle, fit arena scaling, adaptive grid steps, dynamic infinite bounds, red +X / green +Y axes, authentic 4-wheel robot footprint, LiDAR sensor offset (+0.10m), hit discrimination, stale scan detection, `/api/state` lidar and arena bounds schema contracts.
2. `test_startup_lifecycle_motion_gate.py` (10 tests): Clean startup default AWAKE state, persistent checkpoint separation, DDS discovery sleep reject, Rule 0 motion gating, fresh sensor gate, diagnostic motion state transitions.
3. `test_checkpoint_authoritative_verification.py` (11 tests): Authoritative path resolution, atomic tmp/bak replacements, SHA-256 integrity, 138,639 sparse weights round-trip.
4. `test_sleep_wake_motion_gate.py` (8 tests): Hard Rule 0 velocity clamp, teleop rejection during sleep, memory consolidation, ROS bridge state transitions.
5. `test_reinforcement_learning_controls.py` (9 tests): Reward/punishment bounds, affect modulation, episodic memory logging, policy persistence.
6. `test_3d_connectome_viewer_telemetry.py` (10 tests): Spatial projection layout, 2.2MB binary coordinate stream, 7 neuropils, 64-bit ID string preservation, Three.js assets.
7. `test_neural_telemetry.py` (10 tests): Exact FlyWire ID biological queries, sparse CSR connectivity extraction.
8. `test_hard_stop_invariance.py` (6 tests): Obstacle safety invariant under all conditions.
9. `test_persistence_checkpoints.py` (7 tests): Checkpoint continuation fidelity.
10. `test_safety.py` (16 tests): Emergency stop, speed governors, proximity safety latch.
11. `test_ros_node_integration.py` (10 tests): Full ROS 2 node lifecycle and topic bridges.
12. `test_learning_and_memory.py` (7 tests): Experience-based parameter bounds.
13. `test_sleep_wake_lifecycle.py` (4 tests): Lifecycle state supervisor.
14. `test_decoder.py` (4 tests): Neural population decoding.
15. `test_encoder.py` (4 tests): Sensory Poisson encoding.
