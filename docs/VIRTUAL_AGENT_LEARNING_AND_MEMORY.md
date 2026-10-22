# Virtual Agent Learning, Memory, and Plasticity Mechanisms

## 1. Biological Foundation: The Drosophila Mushroom Body

In *Drosophila melanogaster*, associative learning and adaptive decision-making are primarily mediated by the **Mushroom Body (MB)** and its associated **Dopaminergic Neurons (DANs)** and **Mushroom Body Output Neurons (MBONs)** (Aso et al., 2014; Hige et al., 2015; Cognigni et al., 2018).

The biological circuit operates on the principle of **compartmentalized heterosynaptic plasticity**:
1. **Kenyon Cells (KCs)** receive combinatorial olfactory inputs from Projection Neurons (PNs) and form sparse, high-dimensional sensory representations.
2. **Mushroom Body Output Neurons (MBONs)** integrate KC inputs. MBONs are segregated into functional compartments that either promote **Approach** (e.g. $\gamma 5\beta'2a$) or **Avoidance** (e.g. $\gamma 1pedc, \gamma 2\alpha'1$).
3. **Dopaminergic Neurons (DANs)** innervate specific MB lobes:
   - **PAM cluster DANs**: Encode appetitive rewards (sucrose, feeding).
   - **PPL1 cluster DANs**: Encode aversive punishment (quinine, mechanical trauma, electric shock).

---

## 2. Dopamine-Modulated Three-Factor Plasticity Rule

Synaptic plasticity between Kenyon Cells and Mushroom Body Output Neurons does not follow simple two-factor Hebbian rules. Instead, it requires a **three-factor learning rule** involving presynaptic activity, postsynaptic state, and neuromodulatory dopamine release:

$$\Delta W_{ij}(t) = \eta \cdot e_i(t) \cdot y_j(t) \cdot D(t)$$

### 2.1 Presynaptic Eligibility Trace
When a Kenyon cell $i$ fires, it generates an eligibility trace that decays exponentially with time constant $\tau_e = 1.0\text{ s}$:
$$\tau_e \frac{d e_i(t)}{dt} = -e_i(t) + s_i(t)$$
where $s_i(t) \in \{0, 1\}$ is the instantaneous KC spike indicator.

### 2.2 Neuromodulatory Gating
Dopaminergic signaling represents reinforcement valence:
- **Appetitive Reinforcement ($D_{reward}$)**: Triggered by sucrose contact:
  $$D_{reward}(t) = \alpha_{PAM} \cdot G_{sugar}(t)$$
- **Aversive Reinforcement ($D_{punish}$)**: Triggered by shock/quinine contact:
  $$D_{punish}(t) = \alpha_{PPL1} \cdot G_{bitter}(t)$$

### 2.3 Synaptic Weight Updates
Biological evidence indicates that reinforcement selectively depresses or potentiates the opposing behavioral channel:
1. **Aversive Conditioning (Shock / Bitter)**:
   Depresses synapses onto approach-promoting MBONs and potentiates synapses onto avoidance-promoting MBONs:
   $$\Delta W_{KC_i \to MBON_{approach}} = -\eta_{LTD} \cdot e_i(t) \cdot D_{punish}(t)$$
   $$\Delta W_{KC_i \to MBON_{avoid}} = +\eta_{LTP} \cdot e_i(t) \cdot D_{punish}(t)$$
2. **Appetitive Conditioning (Sugar Reward)**:
   Depresses synapses onto avoidance-promoting MBONs and potentiates synapses onto approach-promoting MBONs:
   $$\Delta W_{KC_i \to MBON_{avoid}} = -\eta_{LTD} \cdot e_i(t) \cdot D_{reward}(t)$$
   $$\Delta W_{KC_i \to MBON_{approach}} = +\eta_{LTP} \cdot e_i(t) \cdot D_{reward}(t)$$

### 2.4 Weight Bounds & Normalization
To prevent runaway excitation or synaptic collapse, weights are strictly bounded:
$$W_{min} \le W_{ij}(t) \le W_{max} \quad (W_{min} = 0.05, W_{max} = 1.0, W_{baseline} = 0.50)$$

### 2.5 Passive Memory Decay & Extinction
Unreinforced exposures or passage of time lead to gradual homeostatic relaxation towards baseline:
$$\frac{dW_{ij}}{dt}_{decay} = -\lambda_{decay} (W_{ij}(t) - W_{baseline})$$
where $\lambda_{decay} = 0.001\text{ s}^{-1}$.

---

## 3. Circuit Mapping: From MBONs to Steering Behavior

The behavioral valence $V(t)$ for an odor is computed as the competition between approach and avoidance MBON firing rates:
$$V(t) = r(MBON_{approach}) - r(MBON_{avoid})$$
- If $V(t) > 0$ (positive valence, learned attraction):
  - Agent turns toward the higher odor concentration:
    $$\Delta \omega_{MB} = +k_{tropo} \cdot V(t) \cdot \Delta C_{odor}$$
- If $V(t) < 0$ (negative valence, learned aversion):
  - Agent turns away from the odor source:
    $$\Delta \omega_{MB} = -k_{tropo} \cdot |V(t)| \cdot \Delta C_{odor}$$

These valence signals directly project to descending steering neurons `DNa01` and `DNa02`.

---

## 4. Trainable Readout / Liquid State Machine (LSM) Layer

In addition to the biological MB plasticity rule, we implement an adaptive **readout decoder** operating on descending neural populations:
- Collects continuous neural state vectors $\mathbf{r}(t) \in \mathbb{R}^{K}$ (firing rates across MBONs, Central Complex, and Descending Neurons).
- Trains linear readout weights $\mathbf{W}_{out}$ using recursive least squares (RLS) or ridge regression to optimize motor trajectory smoothness and obstacle clearance.

---

## 5. Persistence and Checkpointing Specification

To enable multi-stage curricula and memory retention testing:
- **`save_checkpoint(filepath)`**: Serializes all plastic synaptic weight matrices, eligibility traces, baseline offsets, and episode history into JSON/PT files.
- **`load_checkpoint(filepath)`**: Restores synaptic weights into a freshly initialized agent, allowing retention evaluation without re-training.
- **`reset_states(reset_weights=False)`**: Clears dynamic membrane potentials and pose while preserving acquired synaptic memories.
