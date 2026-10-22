# Interactive Laboratory Experiments: Specifications & Protocols

## 1. Overview

This document specifies the six interactive experiments (Experiments A through F) designed to evaluate teacher-demonstrated learning, synaptic plasticity, memory retention, and cognitive adaptation in the Live Virtual Brain Laboratory.

Each experiment compares behavior across the three defined modes:
- **Mode A**: Baseline Fixed Connectome (Control)
- **Mode B**: Mushroom Body Dopamine Plasticity (Biological STDP)
- **Mode C**: Adaptive External Readout Decoder (Technological Baseline)

---

## 2. Interactive Experiment Protocols

### Experiment A: Learn to Move Forward
- **Objective**: Train a stationary agent to initiate and sustain forward locomotion upon sensory trigger.
- **Untrained Baseline (Mode A)**: Agent remains at baseline wander speed ($1.5\text{ cm/s}$); when encountering appetitive cues, it lacks learned motivation to accelerate.
- **Teaching Intervention**: Teacher commands `MOVE FORWARD` while presenting Odor A, followed by `REWARD (+1.0)`.
- **Learning Update**: PAM-cluster dopamine release depresses $W_{KC(A) \to MBON\_av}$ and potentiates $W_{KC(A) \to MBON\_app}$, boosting downstream `DNp09` firing.
- **Autonomous Evaluation**: Teacher override removed. When Odor A is presented, the autonomous agent increases forward velocity from $1.5\text{ cm/s}$ to $> 5.0\text{ cm/s}$ without manual input.

### Experiment B: Learn to Turn
- **Objective**: Teach the agent to execute a directional turn (e.g. Turn Left) in response to a specific environmental trigger.
- **Untrained Baseline**: Naive agent splits turns symmetrically ($50\%$ left, $50\%$ right) when presented with a symmetric visual/olfactory stimulus.
- **Teaching Intervention**: Teacher commands `TURN LEFT` in the presence of Odor A, reinforcing with `REWARD`.
- **Autonomous Evaluation**: Upon stimulus presentation in autonomous mode, the agent demonstrates a leftward steering bias ($\omega > +1.5\text{ rad/s}$), steering reliably toward the taught direction.

### Experiment C: Learn to Stop / Avoid
- **Objective**: Condition an aversive deceleration or stopping response to a warning signal.
- **Teaching Intervention**: Teacher commands `STOP` when approaching Hazard Zone B, followed by `PUNISH (-1.0)`.
- **Learning Update**: PPL1-cluster dopamine release depresses approach synapses and potentiates avoidance synapses for Odor B.
- **Autonomous Evaluation**: When navigating toward Hazard B in autonomous mode, the agent decelerates, halts forward progress ($v < 0.5\text{ cm/s}$), and executes an evasive reorientation.

### Experiment D: Obstacle Response (Multi-Modal Looming Escape)
- **Objective**: Evaluate whether the agent navigates around a physical barrier to reach a target.
- **Mechanism**: Visual compound eye detects optical looming expansion $L_i$, activating contralateral LC4 escape turns while Central Complex compass maintains goal heading.
- **Evaluation Metric**: Clearance distance to obstacle ($> 2.0\text{ cm}$) and arrival at goal.

### Experiment E: Memory Retention Across Resets
- **Objective**: Verify that acquired behaviors survive complete bodily and episodic resets.
- **Protocol**:
  1. Complete training for Experiment A & B.
  2. Save checkpoint (`save_checkpoint`).
  3. Reset the virtual world and dynamic membrane potentials.
  4. Load checkpoint into a freshly initialized agent.
  5. Test zero-shot autonomous choice without any new teacher feedback.
- **Metric**: Preference Index $PI > +0.50$ on Trial 1 post-reset.

### Experiment F: Reversal Learning & Cognitive Flexibility
- **Objective**: Assess the agent's ability to extinguish out-of-date memories when environmental contingencies reverse.
- **Protocol**:
  1. Former reward odor (Odor A) is paired with punishment (`PUNISH`).
  2. Former hazard odor (Odor B) is paired with reward (`REWARD`).
  3. Measure the number of trials required to invert the Preference Index ($PI < -0.30 \to PI > +0.30$).
- **Biological Correlate**: Validates that heterosynaptic plasticity avoids catastrophic interference and supports dynamic behavioral re-adaptation.
