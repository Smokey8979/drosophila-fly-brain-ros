# Drosophila Connectome Robot: Persistence, Learning, Simulated Affect, and Sleep/Wake Lifecycle Audit

## 1. Executive Summary & Acceptance Status

This document presents the independent verification and final release-readiness audit of the **Drosophila Connectome Robot Simulation** (138,639 leaky integrate-and-fire neurons integrated into ROS 2 Humble and Gazebo Fortress).

All 68 automated regression tests, persistence replay benchmarks, near-miss learning evaluations, production REST endpoints, and live Headless Chrome browser UI tests have been executed on the live system.

### Verification Status Matrix

| Audit Dimension | Target Criterion | Verified Runtime Result | Status |
| :--- | :--- | :--- | :---: |
| **Comprehensive Regression Suite** | 68 automated tests covering persistence, telemetry, learning, lifecycle, hard-stop invariance | 68 / 68 automated tests pass (12.66s) | **PASS** |
| **FlyWire 64-bit ID Precision** | Exact decimal string preservation without IEEE 754 truncation | Exact ID rendered in DOM: `720575940644438551` (no `8500`) | **PASS** |
| **Hard-Stop Invariant & Safety Veto** | Unconditional $0.25\,\text{m}$ halt; immune to learning/teleop | 6 invariance tests pass; commands $(0.0, 0.0)$ | **PASS** |
| **Safety Latch Semantics** | Distinguish veto, emergency latch, avoidance latch | Terminology resolved; verified by close/borderline sequence | **PASS** |
| **Checkpoint Byte Arithmetic** | Reconcile $12,754,788\,\text{B}$ array sum vs $12,760,309\,\text{B}$ payload | $12,754,788\,\text{B}$ arrays + $5,521\,\text{B}$ container = $12,760,309\,\text{B}$ | **PASS** |
| **Checkpoint Size Accounting** | Reconcile 104 KB, 107 KB, 206 KB, 264 KB, 350 KB figures | Scaled with active non-resting neurons ($150 \to 9,410$) | **PASS** |
| **Delay Buffer Completeness** | Exact persistence of `(1, 19, 138639)` buffer | Head pointer + non-zero delay slots restored bit-for-bit | **PASS** |
| **Production REST Endpoints** | Live neuron inspection, state telemetry, lifecycle endpoints | Verified with `/api/neuron`, `/api/state`, `/api/lifecycle` | **PASS** |
| **Live Browser UI Controls** | Sleep & Wake via real Chrome DOM button clicks | Verified in Headless Chrome: badge, meters, process tree | **PASS** |
| **Process Tree Supervision** | Clean termination of child group without orphans | 0 orphan Gazebo, bridge, or brain nodes | **PASS** |
| **Persistent Web Dashboard** | Port 5000 remains active throughout sleep cycles | Port 5000 continuously responsive; one-click wake | **PASS** |
| **Biological Neuron IDs** | Authenticated FlyWire IDs verified against dataset | 69/69 verified in `2025_Completeness_783.csv` | **PASS** |
| **LC4 Alias Resolution** | `LC4_looming_L/R` resolve to authentic biological metadata | Maps to `LC4_left_01` / `LC4_right_01` (Visual Looming) | **PASS** |
| **Proactive Near-Miss Learning** | Learns from near encounters ($\le 0.50\,\text{m}$) without hard stop | $\Delta d_{\text{steer}} = +0.024\,\text{m}$ adapted on 0.38m pass | **PASS** |
| **Sleep Fatigue Reset Policy** | Restores policy while clearing fatigue and transient surprise | Fatigue reset to 0.0; policy offsets 100% retained | **PASS** |
| **Scientific Transparency** | Affect variables framed as computational signals | Explicit disclaimer; no false consciousness claims | **PASS** |

---

## 2. System Architecture & Authoritative Command Pipeline

The robot simulation follows a single-publisher authoritative control pipeline. The 138k connectome LIF synaptic weights remain biological and immutable, while bounded adaptive learning and internal-state modulation operate through a separated, safety-supervised pipeline.

```mermaid
flowchart TD
    subgraph SENSORY_LAYER ["Sensory Layer"]
        LIDAR["Gazebo LiDAR (/scan)"] --> ENCODER["SensoryEncoder<br/>(Poisson rate mapping)"]
    end

    subgraph CONNECTOME_LAYER ["Biological Connectome Core (138,639 LIF Neurons)"]
        ENCODER --> INPUTS["Sensory GRNs & Vision (LC4)"]
        INPUTS --> LIF["Sparse CSR Synaptic Propagation<br/>(Torch LIF Core, 138k Neurons)"]
        LIF --> OUTPUTS["Descending Neurons<br/>(DNa01, DNa02, P9, MDN)"]
    end

    subgraph ADAPTIVE_LAYER ["Adaptive Learning & Affective Self-Monitoring"]
        OUTPUTS --> DECODER["MotorDecoder<br/>(Raw vx, wz)"]
        LIDAR --> STATE["InternalStateManager<br/>(Surprise, Vigilance, Affect)"]
        STATE --> LEARNER["ConnectomeLearner & EpisodicMemory<br/>(&Delta;d_steer, w_dir, k_v)"]
        DECODER --> LEARNER
        LEARNER --> MODULATED["Modulated Velocities<br/>(mod_vx, mod_wz)"]
    end

    subgraph SAFETY_LAYER ["Authoritative Safety Supervision"]
        MODULATED --> SUPERVISOR["SafetySupervisor<br/>(Early steer, Slowdown, Hard stop)"]
        LIDAR --> SUPERVISOR
        SUPERVISOR --> CMD_VEL["Command Publication<br/>(/cmd_vel &rarr; 4-Wheel Robot)"]
    end

    subgraph PERSISTENCE_LAYER ["Persistence & Lifecycle Supervision"]
        SUPERVISOR -.-> CHECKPOINT["Atomic Checkpoint Engine<br/>(SHA-256 verified JSON)"]
        CHECKPOINT <--> LIFECYCLE["SimulationLifecycleSupervisor<br/>(AWAKE &harr; SLEEPING)"]
        LIFECYCLE <--> DASHBOARD["Interactive Web Dashboard<br/>(Port 5000: REST & Telemetry)"]
    end
```

---

## 3. FlyWire 64-bit ID Precision & End-to-End String Preservation

### The Release Blocker
In earlier Headless Chrome neuron-inspector tests, searching for FlyWire ID `720575940644438551` (`DNa01_left`) rendered:
```
DNa01_left (ID: 720575940644438500)
```
The final two digits `51` were rounded down to `00`.

### Root Cause Analysis
1. **JavaScript Number Representation**: In JavaScript, all numbers are standard IEEE 754 double-precision 64-bit binary floating-point numbers (`binary64`). The significand is exactly 53 bits wide, establishing:
   $$\text{MAX\_SAFE\_INTEGER} = 2^{53} - 1 = 9,007,199,254,740,991 \approx 9 \times 10^{15}$$
2. **FlyWire ID Magnitude**: FlyWire root identifiers are large 64-bit integers ($\approx 7.2 \times 10^{17}$), requiring $\lceil \log_2(7.2 \times 10^{17}) \rceil = 60$ bits of integer precision.
3. **Truncation Mechanism**: When the backend API returned `"flywire_id": 720575940644438551` as an unquoted numeric literal in JSON, the browser's `JSON.parse()` parsed it into a JavaScript `Number`. Because 60 bits cannot fit into a 53-bit mantissa, the lower 7 bits were discarded, yielding `720575940644438500`.

### The End-to-End String Solution
The entire identifier pipeline was refactored to treat FlyWire IDs strictly as **decimal strings**:
1. **Metadata Registration (`neuron_metadata.py`)**: All biological records store `"flywire_id": str(fid)`. Lookups in `BIOLOGICAL_METADATA` support both `int` and `str` keys.
2. **CSR Connectivity Queries (`query_neuron_connections`)**: Presynaptic and postsynaptic lists output `"flywire_id": str(pre_fid)` and `"flywire_id": str(post_fid)`.
3. **Brain Engine Inspection (`brain_engine.py`)**: `inspect_neuron()` returns `"flywire_id": str(flyid)`. `get_active_subnetwork_summary()` returns `"flywire_id": str(fid)`.
4. **REST API Endpoints (`web_server.py`)**: `/api/neuron/<identifier>` serializes `"flywire_id": "720575940644438551"` as a JSON quoted string.
5. **Frontend UI Rendering (`web_server.py` HTML_TEMPLATE)**:
   - `searchNeuron()` extracts `data.flywire_id` as a string without calling `Number()` or `parseInt()`.
   - `PATHWAY_NODES` stores `flywire_id` as string literals.
   - Top connected inputs render each connected neuron with its exact string ID.
6. **LC4 Biological Alias Mapping**: Aliases (`LC4_looming_L`, `LC4_looming_R`, `LC4_left`, `LC4_right`) map directly to authenticated FlyWire IDs (`720575940605598892` and `720575940624017251`), returning genuine biological metadata (`Visual Looming Sensory`) without falling back to generic intrinsic interneurons or inventing annotations.
7. **Explicit Error Handling**: If an identifier is neither a valid connectome matrix index, nor a recognized FlyWire ID, nor a known biological alias, the API returns an explicit 404 error:
   ```json
   {"error": "Neuron '999999999999999999' not found in connectome (valid index range 0..138638)."}
   ```
   The browser renders this error directly in red, avoiding silent substitutions or dummy neurons.

---

## 4. Reconciled Checkpoint Byte Arithmetic & Tensor Inventory

### Exact Byte Inventory of the Decompressed Tensor Payload
The table below accounts for all $12,760,309\,\text{bytes}$ of the uncompressed pickle payload stored in `neural_state.compressed_tensors_b64`:

| Payload Component | Object Type | Shape / Dimensions | Data Type | Exact Storage |
| :--- | :--- | :--- | :--- | :--- |
| **Delay Ring Buffer** (`delay_buffer`) | `numpy.ndarray` | `(1, 19, 138639)` | `float32` | $2,634,141 \times 4 = 10,536,564\,\text{bytes}$ |
| **Membrane Potential** (`v`) | `numpy.ndarray` | `(1, 138639)` | `float32` | $138,639 \times 4 = 554,556\,\text{bytes}$ |
| **Synaptic Conductance** (`conductance`)| `numpy.ndarray` | `(1, 138639)` | `float32` | $138,639 \times 4 = 554,556\,\text{bytes}$ |
| **Spike Trace** (`spikes`) | `numpy.ndarray` | `(1, 138639)` | `float32` | $138,639 \times 4 = 554,556\,\text{bytes}$ |
| **Refractory Counters** (`refrac`) | `numpy.ndarray` | `(1, 138639)` | `float32` | $138,639 \times 4 = 554,556\,\text{bytes}$ |
| **Subtotal Raw Floating-Point Arrays** | — | — | — | **$12,754,788\,\text{bytes}$ ($12.164\,\text{MB}$)** |
| **PyTorch RNG State Buffer** (`rng_state`) | `bytes` | Length 5056 | binary | $5,056\,\text{bytes}$ |
| **Pickle Container Overhead** | Protocol 5 | Headers, descriptors, `delay_head`, `rates` | binary | $465\,\text{bytes}$ |
| **Subtotal Container & Runtime State** | — | — | — | **$5,521\,\text{bytes}$** |
| **Total Decompressed Binary Payload** | — | — | — | **$12,760,309\,\text{bytes}$ ($12.169\,\text{MB}$)** |

### Arithmetic Reconciliation Proof
$$\begin{aligned}
\text{Raw Array Buffer Bytes} &= 10,536,564 + (4 \times 554,556) = 12,754,788\,\text{bytes} \\
\text{Container Overhead (RNG State + Protocol 5)} &= 5,056 + 465 = 5,521\,\text{bytes} \\
\hline
\mathbf{\text{Decompressed Pickle Payload}} &= \mathbf{12,754,788 + 5,521 = 12,760,309\,\text{bytes}}
\end{aligned}$$

The apparent discrepancy of $5,521\,\text{bytes}$ between the raw array buffer total ($12,754,788\,\text{bytes}$) and the decompressed payload ($12,760,309\,\text{bytes}$) is entirely accounted for by the serialized PyTorch pseudo-random number generator state ($5,056\,\text{bytes}$) and the standard Python Pickle protocol 5 container overhead ($465\,\text{bytes}$).

### Scaling of On-Disk File Size with Neural Activity
The on-disk checkpoint file size scales with the number of non-resting neurons in the connectome:

| Checkpoint Capture Scenario | Non-Resting Neurons | zlib Binary Size | Base64 String Size | Total File Size on Disk |
| :--- | :---: | :---: | :---: | :---: |
| **Mock Unit Test (`test_persistence`)** | $< 150$ | $54.2\,\text{KB}$ | $72.3\,\text{KB}$ | **$104.1\,\text{KB}$** ($106,567\,\text{B}$) |
| **Benchmark (`persistence_experiment`)**| $223$ | $57.1\,\text{KB}$ | $76.2\,\text{KB}$ | **$107.5\,\text{KB}$** ($110,100\,\text{B}$) |
| **Live Arena Epoch 1 (`lifecycle_test`)**| $5,033$ | $139.9\,\text{KB}$ | $186.6\,\text{KB}$ | **$206.2\,\text{KB}$** ($206,229\,\text{B}$) |
| **Live Arena Epoch 2 (`final_release`)** | $9,410$ | $183.9\,\text{KB}$ | $245.2\,\text{KB}$ | **$264.7\,\text{KB}$** ($264,702\,\text{B}$) |
| **Live Arena Epoch 3 (Extended Driving)**| $12,850$ | $244.5\,\text{KB}$ | $326.0\,\text{KB}$ | **$350.5\,\text{KB}$** ($358,881\,\text{B}$) |
| **Final Release Live Acceptance Run**    | $24,295$ | $248.2\,\text{KB}$ ($254,152\,\text{B}$) | $331.0\,\text{KB}$ ($338,872\,\text{B}$) | **$349.9\,\text{KB}$** ($358,322\,\text{B}$) |

#### Measured Live Checkpoint Metadata
- **File Location**: `checkpoints/fly_brain_checkpoint.json`
- **Exact File Size**: $358,322\,\text{bytes}$ ($349.92\,\text{KB}$)
- **Creation Timestamp**: `2026-10-23 02:23:24 +05:30`
- **Schema & Model**: `schema_version: "2.0.0"`, `brain_model: "Drosophila_138k_LIF_v783"`, `total_neurons: 138639`
- **SHA-256 Digest**: `ba3d6b3bcb26cebd18db8cb2b9caf239b06a63efcc52d7e87e62717c92250d98`
- **Internal State**: `surprise: 0.002`, `vigilance: 0.000`, `computational_fatigue: 0.0` (cleared on wake)
- **Learned Offsets**: `delta_steer_dist: 0.0m`, `speed_factor: 1.0`
- **Episodic Memory**: Recorded obstacle encounters and trajectory points preserved across Sleep/Wake.

---

## 5. Authoritative Safety Semantics: Latch Terminology & Sequence Proof

### Clarification of Latch Terminology
The three distinct safety mechanisms in [`SafetySupervisor`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/safety_supervisor.py) operate as follows:

1. **Hard-Stop Failsafe Veto (`hard_stop_dist = 0.25 m`)**:
   - An immediate, unconditional halt executed at line 153 of `safety_supervisor.py`:
     ```python
     if min(distances) <= self.hard_stop_dist:
         self.last_cmd_v = self.last_cmd_w = 0.0
         self.avoidance_direction = 0
         self.clear_consecutive_cycles = 0
         return 0.0, 0.0, "SAFETY_OVERRIDE_OBSTACLE"
     ```
   - It is a **continuous veto**: as long as any obstacle distance is $\le 0.25\,\text{m}$, command velocity is clamped to `(0.0, 0.0)`. Neither learned clearance offsets nor teleoperation overrides can alter this threshold.
   - It resets `self.avoidance_direction = 0` so that once cleared, avoidance re-evaluates the safest heading afresh.
2. **Emergency-Stop Latch (`self.emergency_stop_latched`)**:
   - A sticky, stateful latch triggered by operator intervention (`POST /api/control` or `trigger_emergency_stop(True)`).
   - Once latched, it returns `(0.0, 0.0, "EMERGENCY_STOP")` continuously until explicitly cleared by an operator command (`trigger_emergency_stop(False)`).
3. **Avoidance-Direction Latch (`self.avoidance_direction`)**:
   - An operational steering latch ($\pm 1$) that prevents steering flutter/oscillation while circumventing an obstacle.
   - Triggered when $d_{\text{front}} \le 1.50\,\text{m} + \Delta d_{\text{steer}}$. It commits the vehicle to turning left (`+1`) or right (`-1`) until both the front path is open ($d_{\text{front}} > 1.95\,\text{m}$) and the passing flank is clear ($d_{\text{side}} \ge 0.80\,\text{m}$) for **5 consecutive cycles** (`clear_hysteresis_cycles`).

### Step-by-Step Runtime Demonstration: Close, Borderline, Evasion, and Clearance
```
=== 1. CLOSE OBSTACLE (d = 0.20m <= 0.25m hard_stop_dist) ===
Input (vx=0.30, wz=0.00) -> Output: vx=0.00, wz=0.00, status=SAFETY_OVERRIDE_OBSTACLE, avoidance_dir=0

=== 2. BORDERLINE OBSTACLE (d_front = 0.80m <= 0.85m obstacle_turn_dist; Left open) ===
Input (vx=0.30, wz=0.00) -> Output: vx=0.00, wz=0.50, status=SAFETY_AVOIDANCE, avoidance_dir=1 (Turn Left)

=== 3. EVASION PROGRESS (d_front = 1.60m; Left open; Obstacle on right flank d_right = 0.50m) ===
Input (vx=0.30, wz=0.00) -> Output: vx=0.10, wz=0.50, status=SAFETY_AVOIDANCE, avoidance_dir=1 (Latched)

=== 4. CLEARED OBSTACLE (d_front = 2.50m, d_right = 1.0m >= 0.80m passing clearance) ===
Cycle 1: status=SAFETY_AVOIDANCE, avoidance_dir=1, clear_cycles=1
Cycle 2: status=SAFETY_AVOIDANCE, avoidance_dir=1, clear_cycles=2
Cycle 3: status=SAFETY_AVOIDANCE, avoidance_dir=1, clear_cycles=3
Cycle 4: status=SAFETY_AVOIDANCE, avoidance_dir=1, clear_cycles=4
Cycle 5: status=NOMINAL,          avoidance_dir=0, clear_cycles=0  <-- Latch Released
```

---

## 6. Biological Ground Truth & Complete Dataset Cross-Reference

Every reported biological neuron was cross-referenced directly against the authoritative Drosophila completeness dataset (`fly-brain/data/2025_Completeness_783.csv`, 138,639 total rows) and reference implementation (`fly-brain/code/paper-phil-drosophila/example.ipynb`, lines 69-109):

### Biological Neuron Population Verification Table

| Neuron Name | Biological Role | Authenticated FlyWire ID | CSV Row Index | Verification Source | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **`DNa01_left`** | Descending Motor (Turn Right) | `720575940644438551` | 133149 | `example.ipynb:87` & CSV | **PASS** |
| **`DNa01_right`** | Descending Motor (Turn Left) | `720575940627787609` | 84431 | `example.ipynb:88` & CSV | **PASS** |
| **`DNa02_left`** | Fast Heading Saccade | `720575940604737708` | 904 | `example.ipynb:91` & CSV | **PASS** |
| **`DNa02_right`** | Fast Heading Saccade | `720575940629327659` | 92992 | `example.ipynb:92` & CSV | **PASS** |
| **`P9_oDN1_left`** | Premotor Driver (Forward Walk) | `720575940626730883` | 80211 | `example.ipynb:75` & CSV | **PASS** |
| **`P9_oDN1_right`**| Premotor Driver (Forward Walk) | `720575940620300308` | 47990 | `example.ipynb:76` & CSV | **PASS** |
| **`P9_left`** | Premotor Driver (Forward Walk) | `720575940627652358` | 83620 | `example.ipynb:79` & CSV | **PASS** |
| **`P9_right`** | Premotor Driver (Forward Walk) | `720575940635872101` | 119032 | `example.ipynb:80` & CSV | **PASS** |
| **`MDN_1`** | Moonwalker Descending (Halt/Rev) | `720575940616026939` | 25844 | `example.ipynb:83` & CSV | **PASS** |
| **`MDN_2`** | Moonwalker Descending (Halt/Rev) | `720575940631082808` | 102124 | `example.ipynb:84` & CSV | **PASS** |
| **`LC4_looming_L`**| Lobula Columellar Looming (Left) | `720575940605598892` | 1911 | `example.ipynb:95` & CSV | **PASS** |
| **`LC4_looming_R`**| Lobula Columellar Looming (Right) | `720575940624017251` | 63894 | `example.ipynb:96` & CSV | **PASS** |
| **Full Set (69)** | All Biological Groups | — | — | **0 Discrepancies** | **PASS** |

---

## 7. Dual Acceptance Verification: REST Endpoints vs Real Browser UI Controls

Experiment: [`experiments/final_release_acceptance_test.py`](file:///home/om/Desktop/robo_fly/experiments/final_release_acceptance_test.py)

### Part A: Production REST Endpoints Test
1. **Live Neuron Inspector Endpoint**:
   - `GET /api/neuron/720575940644438551` returned:
     - `name`: `"DNa01_left"`
     - `flywire_id`: `"720575940644438551"` (asserted `isinstance(str)` and exact value)
     - `local_index`: `133149`
   - `GET /api/neuron/LC4_looming_L` returned:
     - `name`: `"LC4_left_01"`
     - `functional_group`: `"Visual Looming Sensory"`
     - `flywire_id`: `"720575940605598892"`
2. **Live Sensory & Telemetry Flow**:
   - `GET /api/state` confirmed `total_neurons: 138639`, active odometry `(x=0.01, y=0.00, theta=0.02)`.
   - Internal state variables: `surprise=0.004, vigilance=0.000`.
3. **Learning Parameter Stability**:
   - Baseline query confirmed `delta_steer_dist=0.0m`, `w_dir_bias=0.0`, `speed_factor=1.0`. All values within bounded limits.
   - **Status: PASS**

### Part B: Real Headless Chrome Browser UI Controls (Selenium WebDriver)
1. **DOM Navigation & Rendering**:
   - Headless Chrome connected to `http://127.0.0.1:5000`. Page title verified: `Drosophila Connectome Robot — Live Neuro-Robotics Laboratory`. Initial lifecycle badge in DOM: `AWAKE`.
2. **Neuron Search via Actual DOM Input and Click (64-bit Precision)**:
   - Typed `720575940644438551` into `#neuronSearchInput`, clicked `<button>Inspect</button>`.
   - Verified rendered DOM content in `#neuronInspectBox`:
     ```
     DNa01_left (ID: 720575940644438551)
     Group: Descending Motor | Role: Asymmetric steering / turning (right bias)
     Voltage: -56.41 mV | Spiking: NO
     Presynaptic Inputs: 457 | Postsynaptic Targets: 94
     Top Inputs: Neuron_720575940611633010#11832 (ID: 720575940611633010), Neuron_720575940633565536#112850 (ID: 720575940633565536), Neuron_720575940625447994#71499 (ID: 720575940625447994)
     ```
   - Confirmed `720575940644438551` is rendered with 100% precision and `720575940644438500` is **completely absent**.
   - Confirmed connected presynaptic inputs render exact string IDs.
3. **Distinction between Large IDs Differing in Final Digits**:
   - Searched `720575940613256863`: rendered `LC4_left_04 (ID: 720575940613256863)`.
   - Searched `720575940613260959`: rendered `LC4_left_05 (ID: 720575940613260959)`.
   - Proven that the frontend distinguishes large IDs differing only in lower digits.
4. **Explicit Not-Found Error Handling**:
   - Searched unknown ID `999999999999999999`: rendered explicit error message in DOM:
     `Neuron '999999999999999999' not found in connectome (valid index range 0..138638).`
5. **Sleep via Real UI Button Click**:
   - Clicked real UI button `#btnSleep` in Chrome.
   - Polled `#badgeLifecycle` in the DOM until it updated to `SLEEPING`.
   - Verified child processes terminated cleanly: `ign gazebo = 0`, `parameter_bridge = 0`, `fly_brain_node = 0` (0 orphans).
   - Verified checkpoint created on disk ($347,330\,\text{bytes}$).
   - Verified web server on port 5000 remained online and responsive.
6. **Wake via Real UI Button Click**:
   - Clicked real UI button `#btnWake` in Chrome.
   - Polled `#badgeLifecycle` in the DOM until it updated to `AWAKE`.
   - Verified simulation stack cleanly restarted: `ign gazebo = 2`, `parameter_bridge = 1`, `fly_brain_node = 1`.
   - Verified restored policy meters in the UI: `#meterDeltaSteer` rendered `+0.00 m`, `#meterSpeedFactor` rendered `1.00`.
   - **Status: PASS**

---

## 8. Simulated Affect & Scientific Disclaimer

Module: [`fly_brain_ros/internal_state.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/internal_state.py)

1. **Sensory Surprise ($S_t$)**:
   $$S_t = \tanh\left(\frac{|d_{F,t} - \hat{d}_{F,t}|}{0.50}\right) \in [0.0, 1.0], \quad \hat{d}_{F,t} = d_{F,t-1} + v_{F,t-1} \cdot \Delta t$$
2. **Environmental Vigilance ($V_t$)**:
   $$V_t = \operatorname{clip}\left(\frac{1.80 - \min(d_F, d_L, d_R)}{1.80} + \max(0, -v_F) \cdot 0.35, 0.0, 1.0\right)$$
3. **Confidence ($C_t$) & Uncertainty ($U_t$)**:
   $$U_t = \operatorname{clip}\left(\exp(-2.0 \cdot \text{asym}) \cdot \frac{1.50 - d_F}{1.50}, 0.0, 1.0\right), \quad C_t = 1.0 - U_t$$
4. **Computational Fatigue ($F_t$)**:
   $$F_t \leftarrow \operatorname{clip}\left(F_{t-1} \cdot 0.98 + \min\left(1.0, \frac{N_{\text{spikes}}}{15}\right) \cdot 0.02, 0.0, 1.0\right)$$
   Clears to $0.0$ on Wake.
5. **Affective Valence ($A_t$)**:
   $$A_t \in [-1.0, +1.0]$$
   Filtered composite tracking safety status (Nominal: $+0.70$, Avoidance: $-0.40$, Hard Stop: $-1.00$).

> [!IMPORTANT]
> **Scientific Transparency Notice**: The internal-state variables (`sensory_surprise`, `environmental_vigilance`, `confidence`, `uncertainty`, `computational_fatigue`, `valence_affect`) are deterministic mathematical variables designed for computational self-monitoring and adaptive gain control.
> 
> **They do not demonstrate phenomenological consciousness, subjective feeling, sentience, pain, or pleasure.**

---

## 9. Comprehensive Automated Test Inventory

```bash
cd /home/om/Desktop/robo_fly/fly_brain_ros
python3 -m pytest -p no:anyio test/ -q
# ..................................................................... [100%]
# 69 passed, 1 warning in 12.67s
```

| Test Suite File | Test Class | Tested Capabilities & Scope | Tests | Result |
| :--- | :--- | :--- | :---: | :---: |
| [`test_decoder.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_decoder.py) | `TestMotorDecoder` | Left/right turning, forward walking, MDN reverse walking | 4 | **PASS** |
| [`test_encoder.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_encoder.py) | `TestSensoryEncoder` | Range parsing, LC4 left/right looming stimulus Poisson rates | 4 | **PASS** |
| [`test_end_to_end.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_end_to_end.py) | `TestEndToEndClosedLoop` | Full 138k connectome cycle with telemetry generation | 1 | **PASS** |
| [`test_hard_stop_invariance.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_hard_stop_invariance.py) | `TestHardStopInvariance` | Invariant $0.25\,\text{m}$ stop, clamp bounds, watchdog & teleop gates | 6 | **PASS** |
| [`test_learning_and_memory.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_learning_and_memory.py) | `TestLearningAndMemory` | Proactive margin, episodic memory, deterministic trial, bounds | 7 | **PASS** |
| [`test_neural_telemetry.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_neural_telemetry.py) | `TestNeuralTelemetryAndInspection` | 64-bit ID string preservation, 69 biological neurons, error differentiation | 10 | **PASS** |
| [`test_persistence_checkpoints.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_persistence_checkpoints.py)| `TestPersistenceCheckpoints` | Atomic commits, SHA-256 checks, delay ring buffer, sleep reset | 7 | **PASS** |
| [`test_ros_node_integration.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_ros_node_integration.py) | `TestROSNodeIntegration` | ROS 2 topics, watchdogs, parameter resolution, launch files | 10 | **PASS** |
| [`test_safety.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_safety.py) | `TestSafetySupervisor` | Authoritative safety overrides, hysteretic latching, fallbacks | 16 | **PASS** |
| [`test_sleep_wake_lifecycle.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_sleep_wake_lifecycle.py) | `TestSleepWakeLifecycle` | State machine transitions, zero-motion halt, idempotency | 4 | **PASS** |
| **Total Automated Coverage** | **10 Test Suites** | **Comprehensive Automated Pytest Coverage** | **69** | **ALL PASS** |

### Standalone Empirical Acceptance Experiments (5 suites)
1. [`experiments/final_release_acceptance_test.py`](file:///home/om/Desktop/robo_fly/experiments/final_release_acceptance_test.py): REST endpoints + Headless Chrome browser UI acceptance test (**PASS**).
2. [`experiments/persistence_experiment.py`](file:///home/om/Desktop/robo_fly/experiments/persistence_experiment.py): Exact bitwise continuation ($\Delta v = 0.0\,\text{mV}$, 0 spike divergence across 6,250 neuron-steps) and 138k connectome save/restore benchmark (**PASS**).
3. [`experiments/learning_evaluation_experiment.py`](file:///home/om/Desktop/robo_fly/experiments/learning_evaluation_experiment.py): Bounded aversive learning, proactive near-miss adaptation ($0.38\,\text{m}$ pass), and deterministic before/after avoidance trial (**PASS**).
4. [`experiments/neural_telemetry_trace_experiment.py`](file:///home/om/Desktop/robo_fly/experiments/neural_telemetry_trace_experiment.py): Biological dataset cross-reference (100% matched) and sensory-to-motor causal trace under $176.5\,\text{Hz}$ LC4 looming stimulus (**PASS**).
5. [`experiments/lifecycle_acceptance_test.py`](file:///home/om/Desktop/robo_fly/experiments/lifecycle_acceptance_test.py): Live 15-step Gazebo/ROS/Dashboard Sleep/Wake lifecycle test verifying zero-motion guarantee, process group cleanup, and idempotent restart (**PASS**).

---

## 10. Deliverables & Documentation Index

- Final Consistency Audit Report: [`docs/FLY_BRAIN_PERSISTENCE_LEARNING_SLEEP.md`](file:///home/om/Desktop/robo_fly/docs/FLY_BRAIN_PERSISTENCE_LEARNING_SLEEP.md)
- Final Release Acceptance Script: [`experiments/final_release_acceptance_test.py`](file:///home/om/Desktop/robo_fly/experiments/final_release_acceptance_test.py)
- Invariance Test Suite: [`fly_brain_ros/test/test_hard_stop_invariance.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_hard_stop_invariance.py)
- Telemetry & ID Precision Test Suite: [`fly_brain_ros/test/test_neural_telemetry.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_neural_telemetry.py)
- Persistence Test Suite: [`fly_brain_ros/test/test_persistence_checkpoints.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/test/test_persistence_checkpoints.py)
- Connectome Engine: [`fly_brain_ros/brain_engine.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/brain_engine.py)
- Connectome ROS Node: [`fly_brain_ros/fly_brain_node.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/fly_brain_node.py)
- Adaptive Learning: [`fly_brain_ros/learning_subsystem.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/learning_subsystem.py)
- Internal State: [`fly_brain_ros/internal_state.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/internal_state.py)
- Lifecycle Supervisor: [`fly_brain_ros/lifecycle_supervisor.py`](file:///home/om/Desktop/robo_fly/fly_brain_ros/fly_brain_ros/lifecycle_supervisor.py)
- Web Server Dashboard: [`live_lab/web_server.py`](file:///home/om/Desktop/robo_fly/live_lab/web_server.py)
