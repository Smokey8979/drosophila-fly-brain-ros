# Fly Brain Setup, Dependencies, and Execution Guide

## 1. Environment and Hardware Verification

### 1.1 Host Compute Hardware
- **Host OS**: Ubuntu 22.04.5 LTS (Linux kernel `6.8.0-138-generic`, x86_64)
- **CPU**: Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz (4 physical cores, 8 threads, 6 MiB L3 cache)
- **Host GPU**: Intel Corporation CometLake-U GT2 [UHD Graphics] (Integrated GPU)
- **CUDA / NVIDIA Hardware**: **NONE DETECTED**. No dedicated NVIDIA GPU or CUDA runtime is present.
- **Disk Storage**: `/dev/nvme0n1p2` with ~10 GB free space (96% utilized).

### 1.2 Backend Compatibility Matrix

| Backend | Platform / Device | Status on This Machine | Reason / Evidence |
|---|---|---|---|
| **Brian2 (CPU)** | C++ Standalone / Multi-core CPU | **VERIFIED & WORKING** | Runs via `g++ 11.4` and `brian2 2.9.0`. Verified in 8.05s total runtime for 0.1s simulation. |
| **PyTorch (CPU)** | CPU Tensors / Multi-threading | **VERIFIED & WORKING** | Runs via `torch 2.4.1+cpu`. Verified in 32.4s simulation time for 1,000 steps. |
| **Brian2CUDA** | NVIDIA GPU (CUDA Standalone) | **UNAVAILABLE** | Blocked: No NVIDIA GPU / CUDA toolkit installed. |
| **NEST GPU** | NVIDIA GPU (Custom `user_m1` CUDA) | **UNAVAILABLE** | Blocked: Requires NVIDIA GPU and source compilation with nvcc. |
| **GeNN (PyGeNN)**| NVIDIA GPU (CUDA) | **UNAVAILABLE** | Blocked: `run_genn.py` uses `genn_wrapper.CUDABackend()`. |
| **Brian2GeNN** | NVIDIA GPU (GeNN 4.x CUDA) | **UNAVAILABLE** | Blocked: Requires GeNN CUDA compiler and `brian2<2.6`. |

---

## 2. Environment Setup (Reproducible Instructions)

Because no NVIDIA hardware exists on this system, attempting to install CUDA PyTorch or Brian2CUDA downloads gigabytes of non-functional binary blobs. The verified CPU environment is built cleanly as follows:

```bash
# 1. Create a dedicated Python 3.10 environment
conda create -n brain-fly -y -c conda-forge --override-channels python=3.10 pip

# 2. Install core data science dependencies
/home/om/miniconda3/envs/brain-fly/bin/pip install \
    "numpy==1.26.4" \
    pandas \
    pyarrow \
    scipy \
    joblib \
    tqdm \
    matplotlib

# 3. Install Brian 2 for C++ standalone execution
/home/om/miniconda3/envs/brain-fly/bin/pip install brian2

# 4. Install lightweight PyTorch CPU wheel (186 MB vs 3.5 GB CUDA wheel)
curl -4 -L -o /tmp/torch-2.4.1+cpu-cp310-cp310-linux_x86_64.whl \
    "https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp310-cp310-linux_x86_64.whl"
/home/om/miniconda3/envs/brain-fly/bin/pip install /tmp/torch-2.4.1+cpu-cp310-cp310-linux_x86_64.whl
rm -f /tmp/torch-2.4.1+cpu-cp310-cp310-linux_x86_64.whl
```

### Verification Command:
```bash
/home/om/miniconda3/envs/brain-fly/bin/python -c "
import torch, brian2, pandas, pyarrow, scipy
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('Brian2:', brian2.__version__)
"
```
**Output**:
```text
Torch: 2.4.1+cpu CUDA: False
Brian2: 2.9.0
```

---

## 3. Minimal Baseline Simulation Runs

### 3.1 Brian2 CPU Execution
```bash
cd /home/om/Desktop/robo_fly/fly-brain
/home/om/miniconda3/envs/brain-fly/bin/python main.py --brian2-cpu --t_run 0.1 --n_run 1 --no_log_file
```
**Verified Measurement**:
- Data Loading: `0.498s`
- Neuron & Synapse Creation: `0.327s`
- C++ Code Generation & Compilation: `5.674s`
- Simulation Execution ($0.1\text{s}$ simulated): `0.965s`
- Spike Extraction & Parquet Write: `0.470s`
- Total Elapsed: `8.053s`
- Realtime Ratio: `0.104x` (10 seconds wall-clock per 1 second brain time)
- Output: `data/results/brian2cpp_t0.1s_n1.parquet` (1,632 spikes across 323 active neurons)

### 3.2 PyTorch CPU Execution
```bash
cd /home/om/Desktop/robo_fly/fly-brain
/home/om/miniconda3/envs/brain-fly/bin/python main.py --pytorch --t_run 0.1 --n_run 1 --no_log_file
```
**Verified Measurement**:
- Weight Matrix Assembly & CSR Conversion: `8.091s` (cached to `data/weight_csr.pkl`)
- Simulation Execution ($1,000\text{ steps} \times 0.1\text{ms}$): `32.399s`
- Average Step Latency: `32.4 ms` per discrete biological step on CPU
- Output: `data/results/pytorch_t0.1s_n1.parquet` (1,438 spikes across 314 active neurons)

---

## 4. Ground Truth Parity Verification

Cross-validation between Brian2 CPU (canonical Shiu et al. ground truth) and PyTorch CPU:

```bash
cd /home/om/Desktop/robo_fly/fly-brain
/home/om/miniconda3/envs/brain-fly/bin/python code/compare_ground_truth.py --t_run 0.1 --n_run 1
```

**Measured Parity**:
- Active Neuron Overlap (Jaccard Index): **0.9072** (90.7% identical neuron population)
- Shared Active Neurons: **303** neurons
- Firing Rate Pearson Correlation: **0.9256** (92.6% correlation)
- Spike Count Ratio: **0.8811**
- Median Firing Rate Difference: **20.0%**
