# Fly Brain Repository Map

## 1. Overview & Provenance

- **Repository**: [https://github.com/eonsystemspbc/fly-brain.git](https://github.com/eonsystemspbc/fly-brain.git)
- **Local Location**: `/home/om/Desktop/robo_fly/fly-brain`
- **Revision**: `a3db62f9436074e485c0278290c2164ed6150808` (Branch: `main`)
- **License**: GNU General Public License v2.0 or later (`GPL-2.0-or-later`), with upstream MIT license for `code/paper-phil-drosophila/` components.
- **Scientific Foundation**: Leaky Integrate-and-Fire (LIF) whole-brain computational connectome model based on Shiu et al., *A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor processing* (bioRxiv 2023, Nature 2024).
- **Biological Data**: FlyWire whole-brain connectome v783 (138,639 neurons, ~5 million synapses), with legacy v630 data preserved for archive comparisons.

---

## 2. Directory Structure

```text
/home/om/Desktop/robo_fly/fly-brain/
├── .git/                                   # Git version control metadata
├── .gitattributes                          # Git attributes
├── .gitignore                              # Git ignore rules (caches, parquet outputs, logs)
├── LICENSE                                 # GPL-2.0 license file
├── README.md                               # Project documentation and benchmark overview
├── environment.yml                         # Conda environment definition for main backends (brain-fly)
├── environment-brian2genn.yml              # Conda environment definition for Brian2GeNN backend
├── main.py                                 # Main CLI entrypoint for benchmarks and experiments
├── code/
│   ├── benchmark.py                        # Central orchestrator: config, logging, dispatcher, CSV output
│   ├── compare_backend_to_brian2.py        # Parity analysis between single backend and Brian2 CPU
│   ├── compare_ground_truth.py             # Validation against Brian2 ground truth (Jaccard, correlation)
│   ├── compare_spike_outputs.py            # Pairwise multi-backend spike comparison utility
│   ├── run_brian2_cuda.py                  # Runner for Brian2 CPU and Brian2CUDA GPU backends
│   ├── run_brian2_genn.py                  # Runner for Brian2GeNN backend (isolated env)
│   ├── run_genn.py                         # Runner for native PyGeNN 5.4.0 CUDA backend
│   ├── run_nestgpu.py                      # Runner for NEST GPU backend (custom user_m1 kernel)
│   ├── run_pytorch.py                      # Runner for PyTorch backend (CPU and CUDA GPU)
│   └── paper-phil-drosophila/              # Upstream reference model from Philip Shiu et al.
│       ├── LICENSE                         # MIT License for Shiu et al. code
│       ├── example.ipynb                   # Tutorial notebook: activation, silencing, rate analysis
│       ├── figures.ipynb                   # Notebook reproducing paper figures
│       ├── model.py                        # Brian2 reference implementation of LIF network
│       └── utils.py                        # Analysis helpers (load_exps, get_rate)
├── data/
│   ├── 2025_Completeness_783.csv           # FlyWire v783 neuron list and completeness metadata (138,640 rows)
│   ├── 2025_Connectivity_783.parquet       # FlyWire v783 synaptic connectivity graph (pre, post, weight)
│   ├── benchmark-results.csv               # Accumulated benchmark timing and spike metrics
│   ├── ground-truth-comparison.json        # Cached validation metrics vs Brian2 CPU ground truth
│   ├── sez_neurons.pickle                  # Pre-extracted Subesophageal Zone neuron IDs for analysis
│   ├── archive/
│   │   ├── 2023_Completeness_630.csv       # Legacy FlyWire v630 neuron list
│   │   └── 2023_Connectivity_630.parquet  # Legacy FlyWire v630 synaptic connectivity
│   └── results/                            # Simulation output directory
│       └── nature_2026_07/                 # Publication benchmark metadata and manifests
└── scripts/
    ├── setup_WSL_CUDA.sh                   # Environment setup script for WSL2/CUDA/Miniconda
    └── nestgpu_source_files/               # Custom NEST GPU model kernel files
        ├── pythonlib/
        │   └── nestgpu.py                  # Patched NEST GPU Python interface
        └── src/
            ├── user_m1.cu                  # Custom CUDA kernel for LIF + alpha synapse
            └── user_m1.h                   # Header for custom NEST GPU neuron model
```

---

## 3. Systematic Module Inventory

### 3.1 Entry Points and Orchestration

#### `main.py`
- **Purpose**: CLI entrypoint to configure and launch simulation experiments and benchmark suites.
- **Input Data**: Command-line arguments: `--t_run`, `--n_run`, `--rounds`, `--round-start`, `--run-label`, `--paper`, `--log_file`, `--no_log_file`, `--disable-spike-io`, `--experiment`, backend selectors (`--brian2-cpu`, `--brian2cuda-gpu`, `--pytorch`, `--nestgpu`, `--genn`, `--brian2genn`).
- **Output Data**: Dispatches simulation runs, triggers logging and CSV output.
- **Interfaces**: Calls `benchmark.run_benchmarks()`.
- **Dependencies**: `sys`, `os`, `argparse`, `pathlib`, `code.benchmark`.
- **Role in ROS 2 Integration**: Serves as reference for parameter parsing, but ROS 2 requires persistent in-memory execution rather than batch CLI execution.

#### `code/benchmark.py`
- **Purpose**: Central orchestration engine. Defines parameter spaces, maintains persistent benchmark results (`benchmark-results.csv` and `manifest.csv`), provides logging (`BenchmarkLogger`), and dispatches to backend runners.
- **Input Data**: Configuration dictionaries, experiment specifications (`sugar`, `p9`), paths to connectome data.
- **Output Data**: `data/benchmark-results.csv`, `data/results/<run_label>/manifest.csv`, console and file logs.
- **Public Functions/Classes**:
  - `BenchmarkLogger`: Dual-output logging (console + timestamped file).
  - `get_experiment(name)`: Retrieves experiment dictionary defining stimulated neuron FlyWire IDs and frequencies.
  - `save_result_csv(backend_name, result)`: Records execution metrics (setup time, sim time, spike count, realtime ratio).
  - `run_benchmarks(...)`: Main dispatcher loop over rounds, trials, and backends.
- **Dependencies**: `csv`, `datetime`, `pathlib`, `pandas`, backend modules (`run_brian2_cuda`, `run_pytorch`, `run_nestgpu`, `run_genn`, `run_brian2_genn`).
- **Role in ROS 2 Integration**: Defines canonical experiment structure and data paths.

---

### 3.2 Framework Simulation Runners

#### `code/run_pytorch.py`
- **Purpose**: PyTorch-based execution engine for the LIF fruit-fly brain model. Implements tensorized batched execution with support for both CPU and CUDA GPU.
- **Input Data**: `2025_Completeness_783.csv`, `2025_Connectivity_783.parquet`, cached sparse matrices (`weight_coo.pkl`, `weight_csr.pkl`).
- **Output Data**: Per-spike timing parquet files (`<backend>_t<t_run>s_n<n_run>.parquet`), execution timing dictionaries.
- **Public Classes & Interfaces**:
  - `PoissonSpikeGenerator(nn.Module)`: Computes Bernoulli spike emission from input firing rates.
  - `AlphaSynapse(nn.Module)`: Conductance updates with exponential decay and ring buffer delay.
  - `LIFNeuron(nn.Module)`: Leaky integrate-and-fire membrane potential integration with `ATan` surrogate gradient.
  - `AlphaLIF(nn.Module)`: Composite neuron combining synapse dynamics, refractory clock, and LIF reset.
  - `TorchModel(nn.Module)`: Connectome-level model combining Poisson driver, sparse recurrent synaptic matrix multiplication (`torch.matmul(spikes, weights.T)`), and AlphaLIF.
  - `get_hash_tables(comp_path)`: Returns mapping dictionaries `flyid2i` and `i2flyid`.
  - `get_weights(conn_path, comp_path, wt_dir, csr=True)`: Builds and caches `torch.sparse_csr_tensor`.
  - `run_single_benchmark(...)`: Executes single trial batch.
  - `run_all_benchmarks(...)`: Orchestrates full parameter sweep.
- **Dependencies**: `torch`, `torch.nn`, `numpy`, `pandas`, `pyarrow`, `pickle`, `benchmark`.
- **Role in ROS 2 Integration**: **CRITICAL PRIMARY CANDIDATE**. Unlike Brian2 standalone or NEST GPU, PyTorch maintains explicit tensor state (`v`, `conductance`, `delay_buffer`, `refrac`, `spikes`) across discrete timesteps. This allows persistent stepping (`forward(dt)`) inside a ROS 2 timer loop without recompilation or process teardown. Works natively on CPU when CUDA hardware is unavailable.

#### `code/run_brian2_cuda.py`
- **Purpose**: Executes the canonical LIF model using Brian 2 runtime mode on multi-core CPU (via `joblib`) or Brian2CUDA standalone mode on NVIDIA GPUs.
- **Input Data**: Connectome CSV/Parquet, experiment parameters.
- **Output Data**: Parquet files with spike trains, benchmark performance metrics.
- **Public Functions**: `create_network`, `_run_trial_cpu`, `run_single_benchmark`, `run_all_benchmarks`.
- **Dependencies**: `brian2`, `brian2cuda` (GPU only), `joblib`, `pandas`, `benchmark`.
- **Role in ROS 2 Integration**: Serves as biological ground-truth reference. Standalone compilation mode (`set_device('cuda_standalone')` or `cpp_standalone`) generates static C++ code that executes from start to finish (`device.run()`), making live step-by-step ROS 2 closed-loop streaming difficult without substantial C++ harness modifications.

#### `code/run_nestgpu.py`
- **Purpose**: GPU-accelerated simulation using NEST GPU and a custom CUDA neuron kernel (`user_m1`).
- **Input Data**: Connectome CSV/Parquet, custom neuron parameters.
- **Output Data**: Spike timings, benchmark CSV metrics.
- **Public Functions**: `_configure_nestgpu_environment`, `run_single_benchmark`, `run_all_benchmarks`.
- **Dependencies**: `nestgpu` (compiled from source), CUDA toolkit, `numpy`, `pandas`.
- **Hardware Requirement**: NVIDIA GPU with CUDA.
- **Role in ROS 2 Integration**: In-process reset is impossible (runs in separate subprocesses per trial); unsuited for reactive real-time robotic loops.

#### `code/run_genn.py`
- **Purpose**: Native PyGeNN 5.4.0 implementation with batched CUDA execution.
- **Input Data**: Connectome CSV/Parquet, GeNN model specifications.
- **Output Data**: Parquet spike files, timing metrics.
- **Public Functions**: `_import_pygenn`, `build_genn_model`, `run_single_benchmark`, `run_all_benchmarks`.
- **Dependencies**: `pygenn`, CUDA toolkit 12.x.
- **Hardware Requirement**: NVIDIA GPU with CUDA.
- **Role in ROS 2 Integration**: Fast for batched offline evaluation on GPU, but requires NVIDIA hardware not present on this host.

#### `code/run_brian2_genn.py`
- **Purpose**: Brian2GeNN 1.7.0 standalone device runner.
- **Input Data**: Connectome CSV/Parquet.
- **Dependencies**: GeNN 4.x, Brian2 < 2.6 (isolated conda environment `brain-fly-brian2genn`).
- **Hardware Requirement**: NVIDIA GPU with CUDA.

---

### 3.3 Reference Model & Analysis Utilities

#### `code/paper-phil-drosophila/model.py`
- **Purpose**: Original reference code from Philip Shiu et al. (2023/2024).
- **Public Functions**:
  - `create_model(path_comp, path_con, params)`: Instantiates Brian2 `NeuronGroup` and `Synapses`.
  - `poi(neu, exc, exc2, params)`: Attaches Poisson input generators to target neurons.
  - `silence(slnc, syn)`: Sets outgoing synaptic weights of target neurons to zero.
  - `run_trial(...)`: Runs single episode.
  - `run_exp(...)`: Executes multi-trial experiment and saves parquet results.
  - `get_spk_trn(spk_mon)`: Extracts spike timing dictionary.
  - `construct_dataframe(...)`: Formats spikes into tabular pandas DataFrame.

#### `code/paper-phil-drosophila/utils.py`
- **Purpose**: Post-simulation spike analysis.
- **Public Functions**:
  - `load_exps(l_pqt)`: Loads and concatenates parquet spike files.
  - `get_rate(df, t_run, n_run, flyid2name)`: Calculates mean firing rate (Hz) and standard deviation per neuron across trials.

#### `code/paper-phil-drosophila/example.ipynb` & `figures.ipynb`
- **Purpose**: Scientific validation notebooks demonstrating biological input/output pathways:
  - Descending motor pathways: P9 / P9_oDN1 (forward locomotion), DNa01 / DNa02 (left/right turning), MDN (backward walking/escape), Giant Fiber (escape jump).
  - Sensory pathways: LC4 (visual loom/obstacle detection), GRNs (sugar and bitter taste), JO (auditory/wind mechanosensation), Or56a (olfaction).

---

### 3.4 Verification and Parity Utilities

#### `code/compare_ground_truth.py`
- Compares non-Brian2 backends against Brian2 CPU ground truth.
- Computes Jaccard overlap, precision, recall, and Pearson correlation of per-neuron firing rates.

#### `code/compare_spike_outputs.py` & `code/compare_backend_to_brian2.py`
- Cross-framework parity scripts for publication benchmarks.

---

## 4. Complete Data Inventory

| File Path | Format | Size | Records / Dimensions | Purpose |
|---|---|---|---|---|
| `data/2025_Completeness_783.csv` | CSV | 3.5 MB | 138,640 rows (138,639 neurons) | Primary index mapping FlyWire ID to dense integer neuron index [0..138638]. |
| `data/2025_Connectivity_783.parquet` | Parquet | 98.4 MB | ~5.0 million rows | Synapse graph: `Presynaptic_Index`, `Postsynaptic_Index`, `Excitatory x Connectivity` (signed weight). |
| `data/archive/2023_Completeness_630.csv` | CSV | 3.2 MB | 127,978 neurons | Legacy v630 completeness metadata. |
| `data/archive/2023_Connectivity_630.parquet` | Parquet | 84.6 MB | ~4.3 million synapses | Legacy v630 connectivity graph. |
| `data/sez_neurons.pickle` | Pickle | 5.1 KB | List of integers | Subesophageal zone neuron IDs for feeding/proboscis analysis. |
| `data/benchmark-results.csv` | CSV | 164 KB | Tabular timing log | Historical execution timing logs across backends and parameter grids. |
| `data/results/nature_2026_07/manifest.csv` | CSV | 40 KB | Manifest of benchmark runs | Manifest file documenting paper benchmark runs. |
