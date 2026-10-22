# Live Neural Activity Dashboard Specification

## 1. Executive Summary

This document specifies the graphical architecture, visual telemetry pipelines, and real-time synchronization of the **Live Brain Visualization and Learning Dashboard**.

The dashboard renders the real internal state of the running *Drosophila melanogaster* connectome simulation in real time, guaranteeing that every spike, firing rate, synaptic update, and wheel command reflects authentic computation rather than simulated or mock animation.

---

## 2. Dashboard Visual Panels & Layout

The dashboard is structured into three coordinated visual panels:

```text
+---------------------------------------------------------------------------------------------------+
|                           DROSOPHILA VIRTUAL BRAIN LEARNING LABORATORY                            |
+------------------------------------+----------------------------------+---------------------------+
| PANEL 1: VIRTUAL WORLD ARENA       | PANEL 2: LIVE NEURAL DASHBOARD   | PANEL 3: TEACHING PANEL   |
|                                    |                                  |                           |
| [2D/3D Continuous Physics View]    | [Spike Raster (KCs, LC4, DNs)]   | [Mode Selector]           |
| - 4-Wheel Skid-Steer Robot         |   Kenyon Cells:   || | ||| |     |   ( ) Mode A: Fixed Base  |
| - Left/Right Wheel Rotations       |   LC4 Looming:    |    ||        |   (X) Mode B: Plasticity  |
| - Real-time LiDAR Raycast Beams    |   DNp09 (Forward):||||||||||     |   ( ) Mode C: Ext Decoder |
| - Green Sucrose Reward Zone        |   DNa01 (Turn L): ||  |          |                           |
| - Red Bitter Hazard Zone           |   DNa02 (Turn R):   ||  |        | [Movement Controls]       |
| - Central Obstacles                |                                  |   [^] [v] [<] [>] [STOP]  |
| - Dynamic Trajectory Trail         | [Plastic Synapses (KC -> MBON)]  |                           |
|                                    |   Approach W_app: [======    ]   | [Feedback Controls]       |
|                                    |   Avoidance W_av: [===       ]   |   [ REWARD (+1.0) ]       |
|                                    |                                  |   [ PUNISH (-1.0) ]       |
|                                    | [Dopamine Neuromodulation]       |                           |
|                                    |   PAM (Sweet): [   0.00 Hz   ]   | [Workflow Actions]        |
|                                    |   PPL1 (Shock):[   0.00 Hz   ]   |   [ TEACHING MODE ]       |
|                                    |                                  |   [ AUTONOMOUS MODE ]     |
|                                    | [Central Complex Compass]        |   [ SAVE CHECKPOINT ]     |
|                                    |   E-PG Wedge Heading Attractor   |   [ RESET EPISODE ]       |
|                                    |                                  |                           |
|                                    | [Decoded Wheel Kinematics]       | [Telemetry Display]       |
|                                    |   Speed v: 5.5 cm/s, Turn: 0.0   |   Episode: 1, Step: 245   |
|                                    |   Left Wheel: 5.5, Right: 5.5    |   Reward: +2.0, Shock: 0  |
+------------------------------------+----------------------------------+---------------------------+
```

---

## 3. Real-Time Telemetry Data Pipelines

### 3.1 Spiking Neural Population Monitoring
The dashboard samples neural events at the simulation timestep ($dt = 0.02\text{ s}$, $50\text{ Hz}$):
1. **Kenyon Cells (KCs, ~2,000 neurons)**:
   - Samples active sparse odor representation ($s_{KC} > 0.1$).
   - Renders a multi-channel scrolling raster plot showing active sensory channels.
2. **Lobula Columnar (LC4 Looming Neurons)**:
   - Displays instantaneous firing rates of left and right optical looming populations.
   - Highlights looming threat events in red/amber when an obstacle approaches.
3. **Descending Motor Neurons**:
   - `DNp09`: Forward walking driver firing rate (Hz).
   - `DNa01_left` / `DNa01_right`: Bilateral steering rates (Hz).

### 3.2 Plastic Synaptic Weight Heatmap & Histograms
Whenever three-factor Hebbian plasticity occurs:
- The dashboard extracts the current weight vectors $\mathbf{W}_{KC \to MBON\_app}$ and $\mathbf{W}_{KC \to MBON\_av}$.
- Computes mean synaptic strength, divergence $\Delta W = W_{app} - W_{av}$, and bounds compliance ($[W_{min}, W_{max}] = [0.05, 1.00]$).
- Renders dynamic horizontal gauge bars showing current behavioral valence for Odor A and Odor B.

### 3.3 Neuromodulatory Dopamine Gauges
- **PAM Cluster Dopamine ($D_{reward}$)**: Gauged in green; activates when the agent ingests sucrose or receives an instructor reward.
- **PPL1 Cluster Dopamine ($D_{punish}$)**: Gauged in red; activates when the agent encounters bitter shock or receives an instructor penalty.

### 3.4 Central Complex Compass Ring Attractor
- Renders a polar / radial wedge display of the 16 Ellipsoid Body (E-PG) compass wedges.
- The active population bump rotates in real time with the agent's angular velocity and physical orientation $\theta$, visually demonstrating persistent internal spatial representation.

---

## 4. Performance & Synchronization Guarantees

1. **Non-Blocking Architecture**:
   - The simulation loop runs on a dedicated high-precision timer ($50\text{ Hz}$).
   - The GUI canvas utilizes double-buffering to prevent visual flickering.
2. **Anti-Mock Guarantee**:
   - All displayed spikes, firing rates, and weights originate from `EmbodiedFlyAgent.step()`.
   - If the brain engine is paused or halted, all dashboard plots freeze immediately.
