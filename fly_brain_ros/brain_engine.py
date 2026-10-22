"""
Persistent PyTorch LIF Brain Engine for the Drosophila Connectome.

Maintains the 138k-neuron state in memory across control cycles, allowing
incremental stepping without recompilation or reinitialization.
"""
import time as _time_module
import time
from typing import Dict, Any, List, Optional, Tuple

import sys
import pickle
from pathlib import Path
from time import perf_counter as time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure fly-brain and package modules can be imported if needed
repo_code_dir = Path(__file__).resolve().parents[2] / 'fly-brain' / 'code'
if str(repo_code_dir) not in sys.path:
    sys.path.insert(0, str(repo_code_dir))

pkg_dir = Path(__file__).resolve().parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

try:
    from neuron_metadata import get_neuron_info, query_neuron_connections, BIOLOGICAL_METADATA
except ImportError:
    try:
        from fly_brain_ros.neuron_metadata import get_neuron_info, query_neuron_connections, BIOLOGICAL_METADATA
    except ImportError:
        def get_neuron_info(fid, idx=None):
            return {"flywire_id": fid, "name": f"Neuron_{fid}", "functional_group": "Connectome Interneuron", "hemisphere": "Brain", "role": "LIF integration"}
        def query_neuron_connections(idx, w, o, m, max_top=10):
            return [], [], 0, 0
        BIOLOGICAL_METADATA = {}


class PoissonSpikeGenerator(nn.Module):
    """Generates Poisson-distributed spikes from input firing rates (Hz)."""

    def __init__(self, dt_ms, scale=250.0, device='cpu'):
        super().__init__()
        self.prob_scale = dt_ms / 1000.0
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None, active_indices=None):
        """Generate Bernoulli spikes, optionally sampling active inputs only."""
        if active_indices is None:
            return (
                torch.bernoulli(
                    rates * self.prob_scale, generator=generator
                ) * self.scale
            )

        spikes = torch.zeros_like(rates)

        if active_indices.numel() == 0:
            return spikes

        active_rates = rates.index_select(1, active_indices)
        active_spikes = (
            torch.bernoulli(
                active_rates * self.prob_scale,
                generator=generator,
            ) * self.scale
        )
        spikes.index_copy_(1, active_indices, active_spikes)
        return spikes


class AlphaSynapse(nn.Module):
    """Conductance dynamics with exponential decay and transmission delay."""

    def __init__(self, batch_size, num_neurons, dt_ms, tau_syn=5.0, delay_ms=1.8, device='cpu'):
        super().__init__()
        self.time_factor = dt_ms / tau_syn
        self.steps_delay = max(1, int(round(delay_ms / dt_ms)))
        self.size = num_neurons
        self.batch = batch_size
        self.device = device
        self.delay_head = 0

    def state_init(self):
        self.delay_head = 0
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(self.batch, self.steps_delay + 1, self.size, device=self.device)
        return conductance, delay_buffer

    def forward(self, input_current, conductance, delay_buffer, refrac_mask):
        """Advance synaptic delay using a circular buffer."""
        delayed_input = delay_buffer[:, self.delay_head, :]

        conductance_new = (
            conductance * (1.0 - self.time_factor)
            + delayed_input * refrac_mask
        )

        # Reuse the slot just consumed instead of shifting the full buffer.
        delay_buffer[:, self.delay_head, :] = input_current
        self.delay_head = (
            self.delay_head + 1
        ) % delay_buffer.shape[1]

        return conductance_new, delay_buffer


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron integration with hard reset."""

    def __init__(self, batch_size, num_neurons, dt_ms,
                 v_0=-52.0, v_rst=-52.0, v_th=-45.0, tau_mem=20.0, device='cpu'):
        super().__init__()
        self.size = num_neurons
        self.dt = dt_ms
        self.v_0 = v_0
        self.v_rst = v_rst
        self.v_th = v_th
        self.time_factor = dt_ms / tau_mem
        self.device = device
        self.batch = batch_size

    def state_init(self):
        v = torch.full((self.batch, self.size), self.v_0, device=self.device)
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, conductance, voltage_stim, v):
        v = v + voltage_stim
        v = v + self.time_factor * (conductance - (v - self.v_0))
        spikes = (v >= self.v_th).float()
        v = torch.where(spikes > 0, torch.full_like(v, self.v_rst), v)
        return spikes, v


class AlphaLIF(nn.Module):
    """Composite neuron layer managing synaptic conductance, LIF, and refractory clock."""

    def __init__(self, batch_size, num_neurons, dt_ms, exc_indices=None,
                 t_rfc_ms=2.2, device='cpu'):
        super().__init__()
        self.size = num_neurons
        self.synapse = AlphaSynapse(batch_size, num_neurons, dt_ms, device=device)
        self.neuron = LIFNeuron(batch_size, num_neurons, dt_ms, device=device)
        base_refrac = int(round(t_rfc_ms / dt_ms))

        self.refrac_steps = torch.full((num_neurons,), base_refrac, dtype=torch.long, device=device)
        if exc_indices is not None and len(exc_indices) > 0:
            self.refrac_steps[exc_indices] = 0

    def state_init(self):
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.refrac_steps.unsqueeze(0).repeat(self.neuron.batch, 1).float()
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self, recurrent_input, voltage_stim, conductance, delay_buffer, spikes, v, refrac):
        refrac = torch.where(spikes > 0, torch.zeros_like(refrac), refrac + 1.0)
        refrac_mask = (refrac >= self.refrac_steps.unsqueeze(0)).float()

        conductance_new, delay_buffer = self.synapse(
            recurrent_input, conductance, delay_buffer, refrac_mask
        )
        spikes, v_new = self.neuron(conductance, voltage_stim, v)
        conductance_new = torch.where(spikes > 0, torch.zeros_like(conductance_new), conductance_new)
        return conductance_new, delay_buffer, spikes, v_new, refrac


class FlyBrainEngine:
    """High-performance persistent wrapper for the Drosophila connectome."""

    def __init__(self, comp_path, conn_path, cache_dir=None, dt_ms=0.1, device='cpu'):
        self.comp_path = Path(comp_path)
        self.conn_path = Path(conn_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.conn_path.parent
        self.dt_ms = dt_ms
        self.device = device

        self.num_neurons = 0
        self.flyid2i = {}
        self.i2flyid = {}
        self.weights = None
        self.model = None

        # State tensors
        self.conductance = None
        self.delay_buffer = None
        self.spikes = None
        self.v = None
        self.refrac = None
        self.rates = None

        self.poisson = None
        self._active_rate_index_set = set()
        self.active_rate_indices = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self.w_scale = 0.275
        self.is_initialized = False

    def initialize(self, exc_flywire_ids=None):
        """Loads connectome and builds persistent model and states."""
        t0 = time()
        df_comp = pd.read_csv(self.comp_path, index_col=0)
        self.flyid2i = {j: i for i, j in enumerate(df_comp.index)}
        self.i2flyid = {i: j for j, i in self.flyid2i.items()}
        self.num_neurons = len(df_comp)

        # Load or construct sparse CSR weight tensor
        csr_cache = self.cache_dir / 'weight_csr.pkl'
        if csr_cache.exists():
            with open(csr_cache, 'rb') as f:
                self.weights = pickle.load(f)
        else:
            df_con = pd.read_parquet(self.conn_path)
            idx = [df_con['Postsynaptic_Index'].to_list(), df_con['Presynaptic_Index'].to_list()]
            val = df_con['Excitatory x Connectivity'].to_list()
            coo = torch.sparse_coo_tensor(idx, val, (self.num_neurons, self.num_neurons)).to(torch.float32)
            self.weights = coo.to_sparse_csr()
            with open(csr_cache, 'wb') as f:
                pickle.dump(self.weights, f)

        self.weights = self.weights.to(device=self.device)

        # Reverse-index connections for event-driven propagation.
        # Original weights[post, pre] becomes outgoing_weights[pre, post].
        outgoing_cache = self.cache_dir / "weight_outgoing_csr.pkl"

        if outgoing_cache.exists():
            with open(outgoing_cache, "rb") as f:
                self.outgoing_weights = pickle.load(f)
        else:
            self.outgoing_weights = (
                self.weights.transpose(0, 1).to_sparse_csr()
            )
            with open(outgoing_cache, "wb") as f:
                pickle.dump(self.outgoing_weights.cpu(), f)

        self.outgoing_weights = self.outgoing_weights.to(
            device=self.device
        )


        exc_indices = []
        if exc_flywire_ids:
            exc_indices = [self.flyid2i[fid] for fid in exc_flywire_ids if fid in self.flyid2i]

        self.neurons = AlphaLIF(
            batch_size=1,
            num_neurons=self.num_neurons,
            dt_ms=self.dt_ms,
            exc_indices=exc_indices,
            device=self.device
        )
        self.poisson = PoissonSpikeGenerator(self.dt_ms, scale=250.0, device=self.device)

        self.reset_state()
        self.rates = torch.zeros(1, self.num_neurons, device=self.device)
        self._active_rate_index_set.clear()
        self.active_rate_indices = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self.is_initialized = True
        return time() - t0

    def reset_state(self):
        """Resets all membrane voltages, conductances, and buffers in microseconds."""
        self.conductance, self.delay_buffer, self.spikes, self.v, self.refrac = self.neurons.state_init()

    def set_rates_by_id(self, rate_dict):
        """Update input rates and maintain indices of positive-rate neurons."""
        indices_changed = False

        for flyid, rate in rate_dict.items():
            if flyid not in self.flyid2i:
                continue

            idx = self.flyid2i[flyid]
            rate_value = float(rate)

            if not (0.0 <= rate_value < float("inf")):
                raise ValueError(
                    f"Invalid firing rate for FlyWire ID {flyid}: "
                    f"{rate_value}"
                )

            was_active = idx in self._active_rate_index_set
            is_active = rate_value > 0.0

            self.rates[0, idx] = rate_value

            if is_active and not was_active:
                self._active_rate_index_set.add(idx)
                indices_changed = True
            elif was_active and not is_active:
                self._active_rate_index_set.remove(idx)
                indices_changed = True

        if indices_changed:
            self.active_rate_indices = torch.tensor(
                sorted(self._active_rate_index_set),
                dtype=torch.long,
                device=self.device,
            )

    def _event_driven_propagation(self, active_indices):
        """Accumulate outgoing synaptic input from active presynaptic neurons."""
        weighted_spikes = torch.zeros_like(self.spikes)

        if not active_indices:
            return weighted_spikes

        row_ptr = self.outgoing_weights.crow_indices()
        post_indices = self.outgoing_weights.col_indices()
        edge_weights = self.outgoing_weights.values()

        source_spikes = self.spikes[0]
        target_input = weighted_spikes[0]

        for pre_idx in active_indices:
            start = int(row_ptr[pre_idx].item())
            end = int(row_ptr[pre_idx + 1].item())

            if start == end:
                continue

            destinations = post_indices[start:end]
            weights = edge_weights[start:end]
            contributions = weights * source_spikes[pre_idx]

            target_input.index_add_(
                0, destinations, contributions
            )

        return weighted_spikes

    def step(self, num_steps=1):
        """Advances the simulation by num_steps discrete timesteps.
        
        Returns:
            accumulated_spikes: tensor of shape (num_neurons,) containing spike counts
        """
        profile_total_start = _time_module.perf_counter()
        profile = {
            "setup_sec": 0.0,
            "sensory_input_sec": 0.0,
            "active_count_sec": 0.0,
            "propagation_sec": 0.0,
            "recurrent_input_sec": 0.0,
            "neuron_update_sec": 0.0,
            "output_accumulation_sec": 0.0,
        }

        t0 = _time_module.perf_counter()
        accumulated_spikes = torch.zeros(
            self.num_neurons, device=self.device
        )
        profile["setup_sec"] += _time_module.perf_counter() - t0

        zero_spike_steps = 0
        active_counts = []

        with torch.no_grad():
            for _ in range(num_steps):
                t0 = _time_module.perf_counter()
                p_spikes = self.poisson(
                    self.rates,
                    active_indices=self.active_rate_indices,
                )
                v_stim = self.w_scale * p_spikes
                profile["sensory_input_sec"] += _time_module.perf_counter() - t0

                # Find active presynaptic neurons once.
                t0 = _time_module.perf_counter()
                active_indices = torch.nonzero(
                    self.spikes[0], as_tuple=False
                ).flatten().tolist()

                active_count = len(active_indices)
                active_counts.append(active_count)
                profile["active_count_sec"] += (
                    _time_module.perf_counter() - t0
                )

                # Propagate only outgoing connections from active neurons.
                t0 = _time_module.perf_counter()

                if active_count == 0:
                    zero_spike_steps += 1
                    weighted_spikes = torch.zeros_like(self.spikes)
                else:
                    weighted_spikes = self._event_driven_propagation(
                        active_indices
                    )

                profile["propagation_sec"] += (
                    _time_module.perf_counter() - t0
                )

                t0 = _time_module.perf_counter()
                recurrent_in = self.w_scale * weighted_spikes
                profile["recurrent_input_sec"] += _time_module.perf_counter() - t0

                t0 = _time_module.perf_counter()
                self.conductance, self.delay_buffer, self.spikes, self.v, self.refrac = self.neurons(
                    recurrent_in, v_stim, self.conductance, self.delay_buffer,
                    self.spikes, self.v, self.refrac
                )
                profile["neuron_update_sec"] += _time_module.perf_counter() - t0

                t0 = _time_module.perf_counter()
                accumulated_spikes += self.spikes[0]
                profile["output_accumulation_sec"] += _time_module.perf_counter() - t0

        profile_total_sec = _time_module.perf_counter() - profile_total_start
        profile["other_sec"] = max(
            0.0, profile_total_sec - sum(profile.values())
        )
        profile["total_sec"] = profile_total_sec

        self.last_step_stats = {
            "steps": int(num_steps),
            "zero_spike_steps": zero_spike_steps,
            "mean_active_spikes": (
                sum(active_counts) / len(active_counts) if active_counts else 0.0
            ),
            "max_active_spikes": max(active_counts, default=0),
            **{
                f"profile_{key}": round(value, 6)
                for key, value in profile.items()
            },
        }

        return accumulated_spikes.cpu().numpy()

    def inspect_neuron(self, identifier: Any) -> Dict[str, Any]:
        """Inspect detailed physiological state and connectivity for a single neuron."""
        if not self.is_initialized:
            return {"error": "Connectome not initialized.", "query": str(identifier)}

        ident_str = str(identifier).strip()
        val = None
        is_numeric = False
        try:
            val = int(ident_str)
            is_numeric = True
        except (ValueError, TypeError):
            from neuron_metadata import resolve_neuron_name_or_alias
            val = resolve_neuron_name_or_alias(ident_str)

        if val is None:
            return {
                "error": f"Neuron identifier '{ident_str}' is neither a recognized biological neuron name/alias, nor a valid local index (0..{self.num_neurons-1}), nor an authenticated FlyWire ID.",
                "query": ident_str,
            }

        if val in self.flyid2i:
            local_idx = self.flyid2i[val]
            flyid = val
        elif 0 <= val < self.num_neurons:
            local_idx = val
            flyid = self.i2flyid.get(val, val)
        else:
            if is_numeric and (val >= 10**9 or len(ident_str) >= 10):
                return {
                    "error": f"FlyWire ID '{ident_str}' not found in the {self.num_neurons}-neuron connectome dataset.",
                    "query": ident_str,
                }
            else:
                return {
                    "error": f"Connectome local matrix index {val} is out of bounds (valid range: 0 to {self.num_neurons-1}).",
                    "query": ident_str,
                }

        meta = get_neuron_info(flyid, local_idx)

        with torch.no_grad():
            cur_v = float(self.v[0, local_idx].item()) if self.v is not None else -52.0
            cur_cond = float(self.conductance[0, local_idx].item()) if self.conductance is not None else 0.0
            is_spiking = bool(self.spikes[0, local_idx].item() > 0) if self.spikes is not None else False
            rate_hz = float(self.rates[0, local_idx].item()) if self.rates is not None else 0.0

        pre_list, post_list, tot_pre, tot_post = query_neuron_connections(
            local_idx, self.weights, self.outgoing_weights, self.i2flyid, max_top=10
        )

        return {
            "flywire_id": str(flyid),
            "local_index": int(local_idx),
            "name": meta["name"],
            "functional_group": meta["functional_group"],
            "hemisphere": meta["hemisphere"],
            "role": meta["role"],
            "membrane_voltage_mv": round(cur_v, 2),
            "conductance": round(cur_cond, 4),
            "is_spiking": is_spiking,
            "input_rate_hz": round(rate_hz, 1),
            "total_presynaptic_count": tot_pre,
            "total_postsynaptic_count": tot_post,
            "top_presynaptic_inputs": pre_list,
            "top_postsynaptic_targets": post_list,
        }

    def get_active_subnetwork_summary(self, max_active: int = 25) -> Dict[str, Any]:
        """Provides a lightweight summary of recently active neurons and biological pathways."""
        if not self.is_initialized or self.spikes is None:
            return {"active_count_step": 0, "active_nodes": [], "active_input_rates": {}}

        with torch.no_grad():
            active_indices = torch.nonzero(self.spikes[0], as_tuple=False).flatten().tolist()

        active_nodes = []
        for idx in active_indices[:max_active]:
            fid = self.i2flyid.get(idx, idx)
            meta = get_neuron_info(fid, idx)
            v = float(self.v[0, idx].item())
            active_nodes.append({
                "local_index": int(idx),
                "flywire_id": str(fid),
                "name": meta["name"],
                "group": meta["functional_group"],
                "voltage": round(v, 2),
            })

        return {
            "active_count_step": len(active_indices),
            "active_nodes": active_nodes,
            "active_input_rates": {
                str(self.i2flyid[idx]): round(float(self.rates[0, idx].item()), 1)
                for idx in self._active_rate_index_set
            },
        }

    def save_dynamic_state(self) -> Dict[str, Any]:
        """Serialize dynamic LIF neural variables for atomic checkpointing.
        
        Saves human-readable summary diagnostics alongside a zlib-compressed,
        lossless snapshot of all dynamic neural tensors (v, conductance, delay buffer,
        spikes, refractory counters, delay head, active rates, and PyTorch RNG state)
        to guarantee bitwise-exact dynamical state continuity across restarts.
        """
        if not self.is_initialized or self.v is None:
            return {}

        with torch.no_grad():
            v_tensor = self.v[0]
            non_resting_mask = torch.abs(v_tensor - (-52.0)) > 0.05
            non_resting_indices = torch.nonzero(non_resting_mask, as_tuple=False).flatten().tolist()

            v_sparse = {
                str(idx): round(float(v_tensor[idx].item()), 3)
                for idx in non_resting_indices[:500]
            }

            active_rates_dict = {
                str(self.i2flyid[idx]): float(self.rates[0, idx].item())
                for idx in self._active_rate_index_set
            }

            delay_head = int(self.neurons.synapse.delay_head) if hasattr(self.neurons.synapse, 'delay_head') else 0

            # Lossless state serialization via compressed binary payload
            import zlib, pickle, base64
            tensors_payload = {
                'v': self.v.cpu().numpy(),
                'conductance': self.conductance.cpu().numpy() if self.conductance is not None else None,
                'delay_buffer': self.delay_buffer.cpu().numpy() if self.delay_buffer is not None else None,
                'spikes': self.spikes.cpu().numpy() if self.spikes is not None else None,
                'refrac': self.refrac.cpu().numpy() if self.refrac is not None else None,
                'delay_head': delay_head,
                'active_rates': active_rates_dict,
                'rng_state': torch.get_rng_state().numpy().tobytes(),
            }
            raw_pkl = pickle.dumps(tensors_payload, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_b64 = base64.b64encode(zlib.compress(raw_pkl, level=3)).decode('ascii')
            rng_b64 = base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode('ascii')

            return {
                "v_mean": round(float(v_tensor.mean().item()), 3),
                "v_min": round(float(v_tensor.min().item()), 3),
                "v_max": round(float(v_tensor.max().item()), 3),
                "non_resting_count": len(non_resting_indices),
                "v_sparse": v_sparse,
                "active_rates": active_rates_dict,
                "delay_head": delay_head,
                "compressed_tensors_b64": compressed_b64,
                "rng_state_b64": rng_b64,
            }

    def restore_dynamic_state(self, state_dict: Dict[str, Any]):
        """Restore dynamic neural state from validated checkpoint."""
        if not self.is_initialized or not state_dict:
            return

        with torch.no_grad():
            comp_b64 = state_dict.get("compressed_tensors_b64")
            if comp_b64:
                try:
                    import zlib, pickle, base64
                    raw_bytes = zlib.decompress(base64.b64decode(comp_b64))
                    payload = pickle.loads(raw_bytes)
                    if payload.get('v') is not None:
                        self.v = torch.from_numpy(payload['v']).to(device=self.device)
                    if payload.get('conductance') is not None:
                        self.conductance = torch.from_numpy(payload['conductance']).to(device=self.device)
                    if payload.get('delay_buffer') is not None:
                        self.delay_buffer = torch.from_numpy(payload['delay_buffer']).to(device=self.device)
                    if payload.get('spikes') is not None:
                        self.spikes = torch.from_numpy(payload['spikes']).to(device=self.device)
                    if payload.get('refrac') is not None:
                        self.refrac = torch.from_numpy(payload['refrac']).to(device=self.device)
                    delay_head = payload.get('delay_head', 0)
                    if hasattr(self.neurons.synapse, 'delay_head'):
                        self.neurons.synapse.delay_head = int(delay_head)
                    active_rates = payload.get('active_rates', {})
                    if active_rates:
                        self.set_rates_by_id(active_rates)
                    if payload.get('rng_state'):
                        torch.set_rng_state(torch.ByteTensor(list(payload['rng_state'])))
                    return
                except Exception:
                    pass

            # Fallback for minimal/sparse checkpoints
            self.reset_state()
            v_sparse = state_dict.get("v_sparse", {})
            for idx_str, v_val in v_sparse.items():
                idx = int(idx_str)
                if 0 <= idx < self.num_neurons:
                    self.v[0, idx] = float(v_val)

            delay_head = state_dict.get("delay_head", 0)
            if hasattr(self.neurons.synapse, 'delay_head'):
                self.neurons.synapse.delay_head = int(delay_head)

            active_rates = state_dict.get("active_rates", {})
            if active_rates:
                self.set_rates_by_id(active_rates)

            rng_b64 = state_dict.get("rng_state_b64")
            if rng_b64:
                import base64
                torch.set_rng_state(torch.ByteTensor(list(base64.b64decode(rng_b64))))

