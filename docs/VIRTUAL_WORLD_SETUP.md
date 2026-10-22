# Virtual World Specification & Setup Guide

## 1. Overview

This document specifies the architecture, mathematical models, coordinate frames, sensory fields, and physics equations of the **FlyArena** virtual environment.

FlyArena is an embodied simulation environment purpose-built for closed-loop experiments with the *Drosophila melanogaster* connectome and spiking neural models.

---

## 2. Coordinate System & Arena Geometry

The arena is defined as a bounded continuous 2D/3D workspace:
- **Dimensions**: Rectangle with width $X_{max} = 100.0\text{ cm}$ and height $Y_{max} = 100.0\text{ cm}$ centered at $(0, 0)$ ($[-50, 50] \times [-50, 50]$ cm).
- **Coordinate Frame**: Standard right-handed Cartesian frame where:
  - $+X$ points East.
  - $+Y$ points North.
  - Heading angle $\theta \in [-\pi, \pi)$ measured counter-clockwise from $+X$.
- **Boundary Conditions**: Elastic or inelastic boundary restitution with friction. When the agent contacts an arena boundary, normal velocity is damped:
  $$v_{\perp}(t^+) = -\epsilon \cdot v_{\perp}(t^-) \quad (\epsilon = 0.05)$$
  and tangential velocity is subject to Coulomb friction:
  $$v_{\parallel}(t^+) = (1 - \mu) \cdot v_{\parallel}(t^-) \quad (\mu = 0.2)$$

---

## 3. Agent Kinematics & Physical Parameters

The virtual fly agent is modeled as an embodied rigid body with biological dimensions and mass:
- **Length**: $L_{body} = 2.5\text{ mm} = 0.25\text{ cm}$
- **Antennal Span**: $d_{ant} = 0.5\text{ mm} = 0.05\text{ cm}$ between left and right olfactory sensilla.
- **Mass**: $m = 1.2\text{ mg} = 1.2 \times 10^{-6}\text{ kg}$
- **State Vector**:
  $$\mathbf{s}(t) = \begin{bmatrix} x(t) \\ y(t) \\ \theta(t) \\ v(t) \\ \omega(t) \end{bmatrix}$$
- **Equations of Motion**:
  $$\frac{dx}{dt} = v(t) \cos(\theta(t))$$
  $$\frac{dy}{dt} = v(t) \sin(\theta(t))$$
  $$\frac{d\theta}{dt} = \omega(t)$$
  $$\frac{dv}{dt} = \frac{1}{\tau_v} \left( v_{target}(a_{fwd}) - v(t) \right)$$
  $$\frac{d\omega}{dt} = \frac{1}{\tau_\omega} \left( \omega_{target}(a_{turn}) - \omega(t) \right)$$
  where $\tau_v = 0.05\text{ s}$ and $\tau_\omega = 0.03\text{ s}$ represent neuromuscular time constants.

---

## 4. Multi-Modal Sensory Models

### 4.1 Bilateral Olfactory Antennae
The agent possesses left and right olfactory antennae positioned at:
$$\mathbf{p}_{ant, L}(t) = \begin{bmatrix} x(t) + \frac{L_{body}}{2}\cos\theta - \frac{d_{ant}}{2}\sin\theta \\ y(t) + \frac{L_{body}}{2}\sin\theta + \frac{d_{ant}}{2}\cos\theta \end{bmatrix}$$
$$\mathbf{p}_{ant, R}(t) = \begin{bmatrix} x(t) + \frac{L_{body}}{2}\cos\theta + \frac{d_{ant}}{2}\sin\theta \\ y(t) + \frac{L_{body}}{2}\sin\theta - \frac{d_{ant}}{2}\cos\theta \end{bmatrix}$$

For an odor source $k$ at position $\mathbf{p}_k = (x_k, y_k)$ with emission rate $Q_k$, the concentration at position $\mathbf{p}$ follows a Gaussian plume with turbulent stochastic noise:
$$C_k(\mathbf{p}) = \frac{Q_k}{2\pi \sigma_k^2} \exp\left( -\frac{\|\mathbf{p} - \mathbf{p}_k\|^2}{2\sigma_k^2} \right) \cdot \left( 1 + \eta_{turb}(t) \right)$$
where $\eta_{turb} \sim \mathcal{N}(0, \sigma_{noise}^2)$ captures fluid intermittent fluctuations.
- **Bilateral Gradient**:
  $$\Delta C_k = C_k(\mathbf{p}_{ant, L}) - C_k(\mathbf{p}_{ant, R})$$
  This spatial gradient drives differential firing in left and right Projection Neurons (PNs), guiding tropotactic turning.

### 4.2 Panoramic Compound Eye Visual Array
The compound eye is modeled as an array of $N_{om} = 16$ ommatidia covering a $270^\circ$ azimuth ($[-135^\circ, +135^\circ]$ relative to agent heading $\theta$):
$$\phi_i = \theta + \left( -\frac{3\pi}{4} + \frac{3\pi \cdot i}{2(N_{om}-1)} \right), \quad i \in \{0, \dots, N_{om}-1\}$$
- Each ommatidium performs continuous raycasting into the arena, detecting:
  1. Distance to nearest obstacle/wall: $d_i \in [0, d_{max}]$ ($d_{max} = 30.0\text{ cm}$).
  2. Relative optical flow / looming expansion:
     $$L_i = -\frac{1}{d_i} \frac{\partial d_i}{\partial t}$$
     Positive $L_i$ indicates an approaching or looming object, which directly stimulates **LC4 (Lobula Columnar 4)** looming-detection neurons.

### 4.3 Gustatory Proboscis Sensor
Located at the anterior tip $\mathbf{p}_{prob} = (x + \frac{L_{body}}{2}\cos\theta, y + \frac{L_{body}}{2}\sin\theta)$:
- Detects substrate chemical concentration upon physical contact:
  - **Sucrose / Sugar Patch ($G_{sugar}$)**: Evaluated when $\|\mathbf{p}_{prob} - \mathbf{p}_{food}\| \le R_{food}$. Triggers sweet Gustatory Receptor Neurons (GRNs) and PAM dopaminergic reward signaling ($R = +1.0$).
  - **Quinine / Bitter / Shock Patch ($G_{bitter}$)**: Evaluated when $\|\mathbf{p}_{prob} - \mathbf{p}_{hazard}\| \le R_{hazard}$. Triggers bitter GRNs and PPL1 dopaminergic punishment signaling ($R = -1.0$).

### 4.4 Proprioceptive / Vestibular Compass
Measures internal angular velocity $\omega(t)$ and heading $\theta(t)$, providing input to the Central Complex (Protocerebral Bridge and Ellipsoid Body E-PG ring attractor) to maintain continuous internal directional orientation.

---

## 5. Actuator Interfaces

The agent's motion is driven by biological motor outputs decoded from descending neurons:
1. **Forward Velocity ($v$)**: Decoded from bilateral activity of Descending Neuron P9 (`DNp09` / `P9_oDN1`):
   $$v_{target} = v_{base} + k_{p9} \cdot \left( r(DNp09_L) + r(DNp09_R) \right)$$
   where $v_{base} \approx 2.0\text{ cm/s}$ is spontaneous exploration walking speed, and $r(\cdot)$ denotes instantaneous spike rate.
2. **Angular Velocity ($\omega$)**: Decoded from asymmetric activity of bilateral steering descending neurons `DNa01` and `DNa02`:
   $$\omega_{target} = k_{dna} \cdot \left[ \left( r(DNa01_R) + r(DNa02_R) \right) - \left( r(DNa01_L) + r(DNa02_L) \right) \right]$$
   where positive $\omega$ produces counter-clockwise (leftward) turning.
3. **Proboscis Extension Reflex (PER)**: Decoded from feeding subesophageal zone motor neurons; extends the proboscis to consume food when food is encountered.

---

## 6. Software Architecture

FlyArena is implemented in the `virtual_world` module:
- `virtual_world/arena.py`: Top-level environment simulating physical state, odor fields, landmarks, food/shock patches, and step execution.
- `virtual_world/sensors.py`: Vectorized sensory models (antennae, compound eye raycaster, gustatory contacts, proprioception).
- `virtual_world/actuators.py`: Biological descending-to-kinematics motor decoder.
- `virtual_world/closed_loop_agent.py`: Persistent agent tying sensory encoder, spiking neural network engine, plastic synapses, and motor decoder into a single unified step loop.
