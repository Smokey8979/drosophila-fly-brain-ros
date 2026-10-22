"""
Biological Neuron Metadata & Connectome Topology Mapping for Drosophila Melanogaster.

Provides authenticated biological FlyWire IDs, anatomical annotations, cell classes,
and functional descriptions from:
  - Shiu et al., Nature (2024): Whole-brain connectome LIF model.
  - Ache et al., Neuron (2019): Lobula Columellar (LC4) looming detection.
  - Bidaye et al., Science (2014): Moonwalker descending neurons (MDN).
  - Suver et al., J. Neurosci (2016): DNa01/DNa02 turning descending neurons.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch

# ---------------------------------------------------------------------------
# Authenticated Biological Neuron Populations
# ---------------------------------------------------------------------------

# 1. Forward Locomotion Descending Drivers
P9_ODN1_LEFT = 720575940626730883
P9_ODN1_RIGHT = 720575940620300308
P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

# 2. Asymmetric Steering / Turning Descending Neurons
DNA01_LEFT = 720575940644438551
DNA02_LEFT = 720575940604737708
DNA01_RIGHT = 720575940627787609
DNA02_RIGHT = 720575940629327659

# 3. Backward Walking / Startle Moonwalker Descending Neurons (MDN)
MDN_1 = 720575940616026939
MDN_2 = 720575940631082808
MDN_3 = 720575940640331472
MDN_4 = 720575940610236514

# 4. Fast Escape Jump Descending Neurons
GIANT_FIBER_1 = 720575940622838154
GIANT_FIBER_2 = 720575940632499757

# 5. Proboscis Motor Neurons (Feeding / Consummatory)
MN9_LEFT = 720575940660219265
MN9_RIGHT = 720575940618238523

# 6. Antennal Grooming Descending Neurons
ADN1_LEFT = 720575940624319124
ADN1_RIGHT = 720575940616185531

# 7. Lobula Columellar (LC4) Visual Looming / Threat Detection Neurons
LC4_LEFT = [
    720575940605598892, 720575940611134833, 720575940612580977,
    720575940613256863, 720575940613260959, 720575940614914107,
    720575940615462587, 720575940617176321, 720575940617266722,
    720575940618807105, 720575940620795728, 720575940622108001,
]

LC4_RIGHT = [
    720575940624017251, 720575940625038090, 720575940625934973,
    720575940625991043, 720575940626605200, 720575940626626895,
    720575940628454522, 720575940628462340, 720575940630851036,
    720575940638496720, 720575940603637438, 720575940610522009,
]

# 8. Sugar Gustatory Receptor Neurons (Appetitive Reinforcement / Taste)
SUGAR_GRNS = [
    720575940616885538, 720575940630233916, 720575940639332736,
    720575940632889389, 720575940617000768, 720575940632425919,
    720575940637568838, 720575940629176663, 720575940621502051,
    720575940638202345, 720575940612670570, 720575940611875570,
]

# 9. Bitter Gustatory Receptor Neurons (Aversive Reinforcement / Danger)
BITTER_GRNS = [
    720575940619072513, 720575940646212996, 720575940622298631,
    720575940642088333, 720575940627692048, 720575940617239197,
    720575940618682526, 720575940604714528, 720575940603266592,
]

# 10. Johnston's Organ (Mechanosensory / Wind / Vibration)
JO_EV = [
    720575940645106376, 720575940615272415, 720575940619869120,
    720575940620257345, 720575940620382889, 720575940630834683,
]


# Master Dictionary of Biological Metadata
BIOLOGICAL_METADATA: Dict[Any, Dict[str, Any]] = {}
BIOLOGICAL_NAME_MAP: Dict[str, int] = {}

# Common biological aliases referencing authoritative 64-bit FlyWire IDs
BIOLOGICAL_ALIASES: Dict[str, int] = {
    # LC4 Visual Looming (Left hemifield)
    "lc4_looming_l": LC4_LEFT[0],
    "lc4_looming_left": LC4_LEFT[0],
    "lc4_left": LC4_LEFT[0],
    "lc4_l": LC4_LEFT[0],
    # LC4 Visual Looming (Right hemifield)
    "lc4_looming_r": LC4_RIGHT[0],
    "lc4_looming_right": LC4_RIGHT[0],
    "lc4_right": LC4_RIGHT[0],
    "lc4_r": LC4_RIGHT[0],
    # Descending Moonwalker halts
    "mdn_left": MDN_1,
    "mdn_right": MDN_2,
    # Forward locomotion drivers
    "p9_odn_left": P9_ODN1_LEFT,
    "p9_odn_right": P9_ODN1_RIGHT,
}


def _register(fid: int, name: str, functional_group: str, hemisphere: str, role: str):
    record = {
        "flywire_id": str(fid),
        "name": name,
        "functional_group": functional_group,
        "hemisphere": hemisphere,
        "role": role,
    }
    BIOLOGICAL_METADATA[fid] = record
    BIOLOGICAL_METADATA[str(fid)] = record
    BIOLOGICAL_NAME_MAP[name.lower()] = fid


def resolve_neuron_name_or_alias(name_or_alias: str) -> Optional[int]:
    """Resolves biological neuron names and common aliases to their exact 64-bit FlyWire ID integer."""
    cleaned = str(name_or_alias).strip().lower()
    if cleaned in BIOLOGICAL_NAME_MAP:
        return BIOLOGICAL_NAME_MAP[cleaned]
    if cleaned in BIOLOGICAL_ALIASES:
        return BIOLOGICAL_ALIASES[cleaned]
    return None


# Register all known neurons
_register(P9_ODN1_LEFT, "P9_oDN1_left", "Descending Motor", "Left", "Forward locomotion descending drive")
_register(P9_ODN1_RIGHT, "P9_oDN1_right", "Descending Motor", "Right", "Forward locomotion descending drive")
_register(P9_LEFT, "P9_left", "Premotor Driver", "Left", "Forward walking initiation interneuron")
_register(P9_RIGHT, "P9_right", "Premotor Driver", "Right", "Forward walking initiation interneuron")

_register(DNA01_LEFT, "DNa01_left", "Descending Motor", "Left", "Asymmetric steering / turning (right bias)")
_register(DNA02_LEFT, "DNa02_left", "Descending Motor", "Left", "Asymmetric steering / turning (right bias)")
_register(DNA01_RIGHT, "DNa01_right", "Descending Motor", "Right", "Asymmetric steering / turning (left bias)")
_register(DNA02_RIGHT, "DNa02_right", "Descending Motor", "Right", "Asymmetric steering / turning (left bias)")

_register(MDN_1, "MDN_1", "Descending Motor", "Bilateral", "Moonwalker backward walking & emergency stop")
_register(MDN_2, "MDN_2", "Descending Motor", "Bilateral", "Moonwalker backward walking & emergency stop")
_register(MDN_3, "MDN_3", "Descending Motor", "Bilateral", "Moonwalker backward walking & emergency stop")
_register(MDN_4, "MDN_4", "Descending Motor", "Bilateral", "Moonwalker backward walking & emergency stop")

_register(GIANT_FIBER_1, "Giant_Fiber_1", "Escape Circuit", "Left", "Fast ballistic escape jump descending command")
_register(GIANT_FIBER_2, "Giant_Fiber_2", "Escape Circuit", "Right", "Fast ballistic escape jump descending command")

_register(MN9_LEFT, "MN9_left", "Feeding Motor", "Left", "Proboscis extension motor neuron (feeding)")
_register(MN9_RIGHT, "MN9_right", "Feeding Motor", "Right", "Proboscis extension motor neuron (feeding)")

_register(ADN1_LEFT, "aDN1_left", "Grooming Motor", "Left", "Antennal grooming behavior command")
_register(ADN1_RIGHT, "aDN1_right", "Grooming Motor", "Right", "Antennal grooming behavior command")

for i, fid in enumerate(LC4_LEFT):
    _register(fid, f"LC4_left_{i+1:02d}", "Visual Looming Sensory", "Left", "Lobula Columellar left visual field expansion detector")

for i, fid in enumerate(LC4_RIGHT):
    _register(fid, f"LC4_right_{i+1:02d}", "Visual Looming Sensory", "Right", "Lobula Columellar right visual field expansion detector")

for i, fid in enumerate(SUGAR_GRNS):
    _register(fid, f"Sugar_GRN_{i+1:02d}", "Gustatory Sensory", "Labellum", "Appetitive sucrose gustatory receptor neuron")

for i, fid in enumerate(BITTER_GRNS):
    _register(fid, f"Bitter_GRN_{i+1:02d}", "Gustatory Sensory", "Labellum", "Aversive bitter/toxic tastant receptor neuron")

for i, fid in enumerate(JO_EV):
    _register(fid, f"JO_EV_{i+1:02d}", "Mechanosensory", "Antenna", "Johnston's Organ vibrational/wind displacement sensory neuron")


def get_neuron_info(flywire_id: Any, local_index: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve biological metadata for a FlyWire ID, falling back to connectome index."""
    fid_str = str(flywire_id).strip()

    # Attempt numerical parse or alias resolution
    try:
        fid_int = int(fid_str)
    except ValueError:
        resolved = resolve_neuron_name_or_alias(fid_str)
        if resolved is not None:
            fid_int = resolved
            fid_str = str(resolved)
        else:
            fid_int = None

    if fid_int is not None and fid_int in BIOLOGICAL_METADATA:
        info = dict(BIOLOGICAL_METADATA[fid_int])
        info["flywire_id"] = str(info["flywire_id"])
        if local_index is not None:
            info["local_index"] = int(local_index)
        return info

    # Generic connectome intrinsic neuron
    idx_str = f"#{local_index}" if local_index is not None else ""
    return {
        "flywire_id": fid_str,
        "name": f"Neuron_{fid_str}{idx_str}",
        "functional_group": "Connectome Intrinsic Interneuron",
        "hemisphere": "Central Brain",
        "role": "Recurrent sparse LIF connectome integration",
        "local_index": local_index,
    }


def query_neuron_connections(
    local_idx: int,
    weights_csr: Any,
    outgoing_csr: Any,
    i2flyid: Dict[int, int],
    max_top: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """Extract presynaptic incoming inputs and postsynaptic outgoing targets from CSR tensors.
    
    Returns:
        (presynaptic_list, postsynaptic_list, total_presynaptic_count, total_postsynaptic_count)
    """
    presynaptic_list: List[Dict[str, Any]] = []
    postsynaptic_list: List[Dict[str, Any]] = []

    # 1. Incoming presynaptic connections from weights_csr
    # weights_csr[local_idx, :] contains presynaptic neurons that project to local_idx
    row_ptr_in = weights_csr.crow_indices()
    col_idx_in = weights_csr.col_indices()
    values_in = weights_csr.values()

    start_in = int(row_ptr_in[local_idx].item())
    end_in = int(row_ptr_in[local_idx + 1].item())
    total_pre = end_in - start_in

    if total_pre > 0:
        pre_cols = col_idx_in[start_in:end_in].tolist()
        pre_vals = values_in[start_in:end_in].tolist()
        # Sort by absolute weight magnitude
        paired = sorted(zip(pre_cols, pre_vals), key=lambda x: abs(x[1]), reverse=True)[:max_top]
        for pre_c, w in paired:
            pre_fid = i2flyid.get(pre_c, pre_c)
            meta = get_neuron_info(pre_fid, pre_c)
            presynaptic_list.append({
                "local_index": int(pre_c),
                "flywire_id": str(pre_fid),
                "name": meta["name"],
                "group": meta["functional_group"],
                "weight": round(float(w), 4),
            })

    # 2. Outgoing postsynaptic connections from outgoing_csr
    # outgoing_csr[local_idx, :] contains postsynaptic neurons that local_idx projects to
    row_ptr_out = outgoing_csr.crow_indices()
    col_idx_out = outgoing_csr.col_indices()
    values_out = outgoing_csr.values()

    start_out = int(row_ptr_out[local_idx].item())
    end_out = int(row_ptr_out[local_idx + 1].item())
    total_post = end_out - start_out

    if total_post > 0:
        post_cols = col_idx_out[start_out:end_out].tolist()
        post_vals = values_out[start_out:end_out].tolist()
        paired = sorted(zip(post_cols, post_vals), key=lambda x: abs(x[1]), reverse=True)[:max_top]
        for post_c, w in paired:
            post_fid = i2flyid.get(post_c, post_c)
            meta = get_neuron_info(post_fid, post_c)
            postsynaptic_list.append({
                "local_index": int(post_c),
                "flywire_id": str(post_fid),
                "name": meta["name"],
                "group": meta["functional_group"],
                "weight": round(float(w), 4),
            })

    return presynaptic_list, postsynaptic_list, total_pre, total_post
