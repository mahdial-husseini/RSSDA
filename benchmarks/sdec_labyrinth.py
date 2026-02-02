"""
Labyrinth Domain Driver for Semi-Decentralized Multi-Agent Planning

This module implements the labyrinth (graph search) benchmark domain for evaluating
the RSSDA algorithm. Two agents navigate a graph to locate a hidden target, with
optional noisy observations and state-based synchronization triggers.

Domain Description:
    - Two agents start at designated positions in a graph
    - A target is hidden at one of several possible locations
    - Agents receive observations about their position and potentially the target
    - Goal: Locate the target with maximum expected reward within the planning horizon

Execution Modes:
    - Semi-decentralized (default): Uses RS-SDA* with state-based sync triggers
    - Fully decentralized: Uses RS-MAA* without any synchronization
    - Fully centralized: Plans as if agents can always communicate
    - Noisy mode: Adds sensor noise to target detection and a commitment action (dig)

Key Features:
    - Cached model loading for efficient repeated experiments
    - Policy visualization for debugging and analysis
    - Full simulation mode for averaging over all possible targets
    - Support for custom synchronization trigger sets

Usage:
    python sdec_labyrinth_approx.py <benchmark_id> [options]

    Options:
        --horizon H         Planning horizon (default: from benchmark)
        --maxit M           Max A* iterations (default: inf)
        --decentralized     Run in fully decentralized mode
        --centralized       Run in fully centralized mode
        --noisy             Enable noisy observations
        --fullsim           Run simulation over all targets
        --visualize         Generate policy visualization .txt file

Experiment Reproducibility (Configuration Support):
    - Modify solver_config parameters in def run_labyrinth per paper (for semi-decentralized and centralized solvers; decentralized has dec_pomdp = OriginalDecPOMDP(). 
    -   algorithm="approximate",
        TI1=True,                     # turns on interleaving as desired {True, False}
        TI2=True,                     # pruning; default True
        TI3=True,                     # tail heuristics; default True
        TI4=False,                    # sliding window clustering; default False
        iter_limit=2000,              # L in paper, modify as desired
        rec_limit=1,                  # r in paper, modify as desired
        heuristic_type="HYBRID",      # heuristic choice {QMDP, HYBRID}, QMDP for speed, HBRID for accuracy
        tail_heuristic_type="HYBRID", # heuristic choice {QMDP, HYBRID}, QMDP for speed, HBRID for accuracy
        hybrid_r=1,                   # r but hybrid component (more general than QMDP) {1, 2}, but recommend 2 for accuracy, 1 for speed
        max_clusters=2,               # k in paper for sliding window, modify as desired

Examples (ensure configuration modified as desired, per the above):  
    - "python sdec_labyrinth_approx.py chamber_3d_015 9": runs chamber_3d_015 benchmark for horizon 9, one time
    - "python sdec_labyrinth_approx.py 1 9 5 --noisy": runs the noisy version of Labyrinth 1 for horizon 9, five times (with statistical output)
    - "python sdec_labyrinth_approx.py 5 6 2 --fullsim --decentralized": runs a Labyrinth 5 for horizon 6 for two full loops of iterations through every possible target node

Important Notes:
    Exact solvers should almost always use tight heuristics (heuristic_type = "POMDP")
    Approximate solvers should almost always use loose heuristics (heuristic_type = "HYBRID"/"QMDP")

Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

import sys
import os
import time
import random
import math
import json
from array import array

# Add parent directories to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)  # Parent of benchmarks/
sys.path.insert(0, _script_dir)  # For labyrinth_cache.py
sys.path.insert(0, _root_dir)  # For RSSDA.py
sys.path.insert(0, os.path.join(_root_dir, 'baselines'))  # For decPOMDP.py

try:
    from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple, MemoryLimitExceeded
except ImportError:
    print("Warning: Could not import SDecPOMDP solver. Running in test mode.")
    SDecPOMDP = None
    SDecPOMDPModel = None
    int_tuple = tuple

# Import original decPOMDP for fully decentralized mode
try:
    from decPOMDP import DecPOMDP as OriginalDecPOMDP, MemoryLimitExceeded as DecPOMDPMemoryLimitExceeded
except ImportError:
    print("Warning: Could not import original decPOMDP. Decentralized mode may not work.")
    OriginalDecPOMDP = None
    DecPOMDPMemoryLimitExceeded = None

# Import caching utilities
try:
    from labyrinth_cache import (
        load_cached_labyrinth,
        load_cached_qmdp,
        load_cached_pdict,
        precompute_all,
        precompute_pdict,
        create_config_from_cache,
        create_loader_from_cache,
        apply_sync_knowledge_propagation,
        apply_noisy_detection,
        load_cached_noisy_labyrinth,
        precompute_noisy_all
    )
except ImportError:
    print("Warning: Could not import labyrinth_cache. Cache functionality disabled.")
    load_cached_labyrinth = None
    apply_sync_knowledge_propagation = None
    apply_noisy_detection = None
    load_cached_qmdp = None
    load_cached_pdict = None
    precompute_all = None
    precompute_pdict = None
    create_config_from_cache = None
    create_loader_from_cache = None
    load_cached_noisy_labyrinth = None
    precompute_noisy_all = None

# ============================================================================
#                           USER CONFIGURATION
# ============================================================================
# Modify the parameters below to control solver behavior.
# ============================================================================

# --- Core Algorithm ---
ALGORITHM = "approximate"       # "exact" or "approximate" (enables TI approximations)

# --- Heuristic Type ---
# Controls upper-bound heuristic for A* search guidance.
#   "QMDP"   - Loose/fast: ignores observations (dot product of belief and state values)
#   "POMDP"  - Tight/exact: full centralized POMDP value function
#   "HYBRID" - Runs exact POMDP for first HYBRID_R steps, then QMDP
# Rule of thumb: Use "POMDP" for exact algorithm, "QMDP" or "HYBRID" for approximate.
HEURISTIC_TYPE = "HYBRID"
HYBRID_R = 1                   # Steps of exact POMDP before switching to QMDP (HYBRID mode only)

# --- Decentralized Heuristic Search ---
# When computing decentralized component heuristics, we run a bounded A* search.
# These parameters control early termination of that inner search.
MAXIT = 200                     # Max A* iterations for decentralized heuristic computation
ALPHA = 0.2                     # Early termination: stop if value <= upper - alpha*|upper|
IE_MIN2 = 3                     # Min depth of information-sharing stages for decentralized heuristic

# --- Approximation Techniques (TI Flags) ---
# Enable these for faster but approximate solutions. Requires ALGORITHM = "approximate".
TI1 = False  # Interleaving Planning/Execution: prune branches via consensus voting
TI2 = True   # Progress-based Pruning: limit per-entity exploration budget
TI3 = True   # Tail Approximation: use heuristics for final REC_LIMIT stages
TI4 = True  # Max Clustering: cluster based on L1 distance between beliefs, weighted by probability mass

# --- TI1: Interleaving Parameters ---
# Consensus voting among top nodes to detect centralized stages early.
SCORE_LIMIT = 20                # Number of top policy nodes to sample for voting
CEN_THRESHOLD = 0.6             # Weighted vote threshold to force centralization
SM_TEMPERATURE = 0.6            # Softmax temperature for node weights (lower = focus on best)
ADAPTIVE_CHECK = 100            # Minimum iterations between TI1 consensus checks

# --- TI2: Progress Pruning ---
# Total iteration budget; per-entity budget B = ITER_LIMIT / (nagents + 1).
# Prunes nodes when exploration exceeds entity's fair share.
ITER_LIMIT = 1000

# --- TI3: Tail Approximation ---
# When remaining horizon <= REC_LIMIT, use heuristic value instead of exact expansion.
REC_LIMIT = 2
TAIL_HEURISTIC_TYPE = "HYBRID"

# --- TI4: Max Clustering ---
# Cluster into MAX_Clusters based on combination of L1 distance between resulting beliefs and probability mass
MAX_CLUSTERS = 2

# ============================================================================
#                        END USER CONFIGURATION
# ============================================================================

# ==========================================
# Noisy mode helpers
# ==========================================

def format_noisy_action(act, act_per_agent):
    """Format action name for drilling mode (has DRILL action)."""
    if act == 0:
        return "WAIT"
    elif act == act_per_agent - 1:
        return "DRILL"
    else:
        return f"MV({act})"

def decode_noisy_obs(obs, num_nodes):
    """Decode noisy observation into (position, sensor_str)."""
    position = obs // 2
    sensor = obs % 2
    sensor_str = "B" if sensor == 1 else "S"  # Beep or Silence
    return position, sensor_str

def decode_deterministic_obs(obs):
    """Decode deterministic observation into (position, found).

    Observation encoding: o = position * 2 + found
    where found=1 means agent has discovered the target.
    """
    position = obs // 2
    found = obs % 2
    return position, found

def translate_sync_trigger_to_drilling(base_sync_trigger, num_nodes, num_targets):
    """Translate sync_trigger from base labyrinth encoding to drilling mode encoding.

    Base labyrinth state: s = u1 * (N * T * 4) + u2 * (T * 4) + t_idx * 4 + f1 * 2 + f2
    Drilling state:       s = u1 * (N * T) + u2 * T + t_idx

    For drilling mode, we want to trigger at the same physical positions (u1, u2)
    for ALL target indices, since we don't know which target is correct.
    """
    N, T = num_nodes, num_targets
    drilling_triggers = set()

    for base_state in base_sync_trigger:
        # Decode base state to get position (u1, u2) - ignore t_idx, f1, f2
        temp = base_state // 2  # skip f2
        temp = temp // 2        # skip f1
        temp = temp // T        # skip t_idx
        u2 = temp % N
        u1 = temp // N

        # For drilling mode, trigger at position (u1, u2) for ALL targets
        # This ensures synchronization happens regardless of which target is correct
        for t in range(T):
            drilling_state = u1 * (N * T) + u2 * T + t
            drilling_triggers.add(drilling_state)

    return list(sorted(drilling_triggers))

# ==========================================
# Conversion utilities for decPOMDP.py
# ==========================================

def flat_to_pdict_transitions(T_flat, nactions, nstates):
    """Convert flat transition array to pdict format for decPOMDP.py.

    decPOMDP.py expects: transitions[act*nstates + s] = (indices_array, values_array)
    where indices are next states and values are probabilities.

    Supports both dense arrays and sparse dicts.
    """
    nsq = nstates * nstates
    is_sparse = isinstance(T_flat, dict)
    result = []
    for act in range(nactions):
        for s in range(nstates):
            indices = []
            values = []
            for snew in range(nstates):
                idx = act * nsq + s * nstates + snew
                val = T_flat.get(idx, 0.0) if is_sparse else T_flat[idx]
                if val > 0:
                    indices.append(snew)
                    values.append(val)
            result.append((array('i', indices), array('d', values)))
    return result


def flat_to_pdict_obs(O_flat, nactions, nstates, nobs):
    """Convert flat observation array to pdict format for decPOMDP.py.

    decPOMDP.py expects: obs[act*nstates + snew] = (indices_array, values_array)
    where indices are observations and values are probabilities.

    Supports both dense arrays and sparse dicts.
    """
    nso = nstates * nobs
    is_sparse = isinstance(O_flat, dict)
    result = []
    for act in range(nactions):
        for snew in range(nstates):
            indices = []
            values = []
            for o in range(nobs):
                idx = act * nso + snew * nobs + o
                val = O_flat.get(idx, 0.0) if is_sparse else O_flat[idx]
                if val > 0:
                    indices.append(o)
                    values.append(val)
            result.append((array('i', indices), array('d', values)))
    return result


# ==========================================
# Policy Visualization
# ==========================================

def build_action_destination_map(loader, config, T):
    """
    Build a mapping from (node, action) -> destination_node for each agent.

    Returns:
        action_dest: dict mapping (node, action_idx, agent_idx) -> destination_node
                     for single-agent actions (action 0 = WAIT)
    """
    action_dest = {}
    num_nodes = loader.num_nodes
    act_per_agent = config.act_per_agent
    nsq = config.nstates * config.nstates
    nstates = config.nstates
    is_sparse = isinstance(T, dict)

    for node in range(num_nodes):
        # For agent 1, check from position (node, 0)
        ref_state = loader.tuple_to_state(node, 0, 0, 0, 0)
        for a in range(act_per_agent):
            if a == 0:
                action_dest[(node, a, 0)] = node  # WAIT stays at same node
                continue
            ja = a + 0 * act_per_agent  # Agent 1 acts, agent 2 waits
            idx_base = ja * nsq + ref_state * nstates
            dest = node  # Default to same node if no transition found
            for s_next in range(nstates):
                p = T.get(idx_base + s_next, 0.0) if is_sparse else T[idx_base + s_next]
                if p > 0.5:
                    nu1, _, _, _, _ = loader.state_to_tuple(s_next)
                    if nu1 != -1:
                        dest = nu1
                    break
            action_dest[(node, a, 0)] = dest

        # For agent 2, check from position (0, node)
        ref_state = loader.tuple_to_state(0, node, 0, 0, 0)
        for a in range(act_per_agent):
            if a == 0:
                action_dest[(node, a, 1)] = node  # WAIT stays at same node
                continue
            ja = 0 + a * act_per_agent  # Agent 1 waits, agent 2 acts
            idx_base = ja * nsq + ref_state * nstates
            dest = node
            for s_next in range(nstates):
                p = T.get(idx_base + s_next, 0.0) if is_sparse else T[idx_base + s_next]
                if p > 0.5:
                    _, nu2, _, _, _ = loader.state_to_tuple(s_next)
                    if nu2 != -1:
                        dest = nu2
                    break
            action_dest[(node, a, 1)] = dest

    return action_dest


def generate_policy_visualization(policy, clustering, cent_vector, cen_dists_map, clustering_cen,
                                   config, loader, T, dec_pomdp, output_file=None):
    """
    Generate a comprehensive human-readable visualization of a policy.

    Args:
        policy: The policy structure from multi_agent_astar
        clustering: Observation-to-cluster mappings for decentralized stages
        cent_vector: Boolean list indicating centralized (True) or decentralized (False) per stage
        cen_dists_map: Maps stage -> list of belief indices for centralized execution
        clustering_cen: Clustering for centralized stages
        config: LabyrinthConfig
        loader: LabyrinthLoader with graph topology
        T: Transition matrix
        dec_pomdp: The solver instance (for belief lookups)
        output_file: Optional file path to write the visualization

    Returns:
        String containing the policy visualization
    """
    lines = []

    # Build action destination map
    action_dest = build_action_destination_map(loader, config, T)

    # Header
    lines.append("=" * 80)
    lines.append(f"POLICY VISUALIZATION - Labyrinth {config.bid}")
    lines.append("=" * 80)
    lines.append("")

    # Graph Topology
    lines.append("GRAPH TOPOLOGY")
    lines.append("-" * 40)
    lines.append(f"Nodes: {loader.num_nodes} (0 to {loader.num_nodes - 1})")
    lines.append(f"Start Node: {loader.start_node}")
    lines.append(f"Possible Targets: {loader.targets}")
    lines.append("")

    # Build adjacency from action_dest
    adjacency = {n: set() for n in range(loader.num_nodes)}
    for (node, act, agent), dest in action_dest.items():
        if dest != node and dest != "?":
            adjacency[node].add(dest)

    lines.append("Adjacency List:")
    for node in range(loader.num_nodes):
        neighbors = sorted(adjacency[node])
        lines.append(f"  Node {node}: -> {neighbors if neighbors else '(no outgoing edges)'}")
    lines.append("")

    # Action Mapping per Node
    lines.append("ACTION MAPPINGS (per node)")
    lines.append("-" * 40)
    for node in range(loader.num_nodes):
        agent1_actions = []
        agent2_actions = []
        for a in range(config.act_per_agent):
            dest1 = action_dest.get((node, a, 0), node)
            dest2 = action_dest.get((node, a, 1), node)
            if a == 0:
                agent1_actions.append(f"a{a}=WAIT")
                agent2_actions.append(f"a{a}=WAIT")
            else:
                if dest1 != node:
                    agent1_actions.append(f"a{a}->N{dest1}")
                if dest2 != node:
                    agent2_actions.append(f"a{a}->N{dest2}")
        lines.append(f"  Node {node}: Agent1[{', '.join(agent1_actions)}] Agent2[{', '.join(agent2_actions)}]")
    lines.append("")

    # Policy Structure
    lines.append("POLICY STRUCTURE")
    lines.append("-" * 40)
    lines.append(f"Horizon: {len(policy)} stages")
    lines.append(f"Centralization Vector: {cent_vector}")
    lines.append("")

    # Detailed Policy per Stage
    for stage_idx, stage_policy in enumerate(policy):
        is_centralized = cent_vector[stage_idx] if stage_idx < len(cent_vector) else False
        stage_type = "CENTRALIZED" if is_centralized else "DECENTRALIZED"

        lines.append(f"STAGE {stage_idx + 1} [{stage_type}]")
        lines.append("-" * 40)

        dec_part = stage_policy[0]  # [[agent1_actions], [agent2_actions]]
        cen_part = stage_policy[1]  # [[joint_action1], [joint_action2], ...]

        if is_centralized:
            # Centralized stage - show joint actions for each belief cluster
            lines.append("  Joint Actions (one per belief cluster):")
            for cluster_idx, joint_actions in enumerate(cen_part):
                if not joint_actions:
                    continue
                ja = joint_actions[0]
                a1 = ja % config.act_per_agent
                a2 = ja // config.act_per_agent
                lines.append(f"    Cluster {cluster_idx}: JointAction={ja} (Agent1=a{a1}, Agent2=a{a2})")

            # Show belief cluster info if available
            if stage_idx < len(cen_dists_map) and cen_dists_map[stage_idx]:
                lines.append(f"  Belief Clusters: {len(cen_dists_map[stage_idx])} cluster(s)")
        else:
            # Decentralized stage - show per-agent action tables
            lines.append("  Agent 1 Actions (indexed by observation history cluster):")
            if dec_part and len(dec_part) > 0:
                agent1_actions = dec_part[0]
                for oh_idx, act in enumerate(agent1_actions):
                    if act == -1:
                        lines.append(f"    OH[{oh_idx}]: N/A (centralized path)")
                    elif act == -2:
                        lines.append(f"    OH[{oh_idx}]: UNASSIGNED")
                    else:
                        lines.append(f"    OH[{oh_idx}]: a{act}")

            lines.append("  Agent 2 Actions (indexed by observation history cluster):")
            if dec_part and len(dec_part) > 1:
                agent2_actions = dec_part[1]
                for oh_idx, act in enumerate(agent2_actions):
                    if act == -1:
                        lines.append(f"    OH[{oh_idx}]: N/A (centralized path)")
                    elif act == -2:
                        lines.append(f"    OH[{oh_idx}]: UNASSIGNED")
                    else:
                        lines.append(f"    OH[{oh_idx}]: a{act}")

            # Show clustering (observation to next OH mapping)
            if stage_idx < len(clustering) and clustering[stage_idx]:
                lines.append("  Observation Clustering (current_OH, obs -> next_OH):")
                for agent_idx, agent_clustering in enumerate(clustering[stage_idx]):
                    if agent_clustering:
                        lines.append(f"    Agent {agent_idx + 1}:")
                        for oh_idx, obs_mapping in enumerate(agent_clustering):
                            if obs_mapping:
                                mappings = [f"o{obs}->OH{next_oh}" for obs, next_oh in enumerate(obs_mapping) if next_oh is not None]
                                if mappings:
                                    lines.append(f"      OH[{oh_idx}]: {', '.join(mappings[:8])}{'...' if len(mappings) > 8 else ''}")

        lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append("POLICY SUMMARY")
    lines.append("=" * 80)
    num_cen = sum(1 for v in cent_vector if v)
    num_dec = len(cent_vector) - num_cen
    lines.append(f"Total Stages: {len(policy)}")
    lines.append(f"Centralized Stages: {num_cen}")
    lines.append(f"Decentralized Stages: {num_dec}")
    lines.append("")

    # Legend
    lines.append("LEGEND")
    lines.append("-" * 40)
    lines.append("  a0 = WAIT action")
    lines.append("  a1, a2, ... = MOVE actions (destination depends on current node)")
    lines.append("  OH[i] = Observation History cluster index")
    lines.append("  JointAction = a1 + a2 * act_per_agent")
    lines.append("  N/A = This cluster follows centralized execution")
    lines.append("  UNASSIGNED = Action not yet determined (partial policy)")
    lines.append("  Observation o = position * 2 + found_flag (found=1 means target discovered)")
    lines.append("")

    result = "\n".join(lines)

    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"Policy visualization written to: {output_file}")

    return result


# ==========================================
# Usage Notes
# ==========================================

# Standard usage (uses cached data, auto-generates if missing)
# python sdec_labyrinth_approx.py 1 20

# Multiple simulations
# python sdec_labyrinth_approx.py 1 20 5

# With verbose output
# python sdec_labyrinth_approx.py 1 20 --verbose

# ==========================================
# Configuration & Constants
# ==========================================

class LabyrinthConfig:
    # Default detection probability for drilling mode (modify this value for testing)
    DEFAULT_DETECTION_PROB = 0.90

    def __init__(self, bid, horizon, maxit, ie_min2, alpha, replan_at_all_syncs=False, decentralized=False, centralized=False, noisy=False, detection_prob=None):
        self.bid = bid
        self.horizon = horizon
        self.maxit = maxit
        self.ie_min2 = ie_min2
        self.alpha = alpha
        self.replan_at_all_syncs = replan_at_all_syncs
        self.decentralized = decentralized
        self.centralized = centralized
        self.noisy = noisy

        # Set detection probability
        if noisy:
            self.detection_prob = detection_prob if detection_prob is not None else self.DEFAULT_DETECTION_PROB
            print(f"DRILLING mode enabled: detection probability = {self.detection_prob}")
        else:
            self.detection_prob = 1.0  # deterministic detection when not noisy

        self.nagents = 2
        self.act_per_agent = None  # auto-detected from .data
        self.nacts = None  # auto-detected from .data
        self.nacts_factor = None  # auto-detected from .data

        # Track if using mode-specific data file (knowledge propagation baked in)
        self.uses_mode_specific_data = False
        self.load_metadata()

        # Load Triggers based on mode:
        # - Decentralized: empty trigger (no sync points)
        # - Centralized: all states are sync points
        # - Semi-decentralized (default): LOS-based sync states from config
        if decentralized:
            self.state_trigger = []
            self.action_trigger = []
            self.obs_trigger = []
            print(f"Running in DECENTRALIZED mode (no sync triggers)")
        elif centralized:
            # Exclude sink state (nstates - 1) from triggers
            self.state_trigger = list(range(self.nstates - 1))
            self.action_trigger = []
            self.obs_trigger = []
            print(f"Running in CENTRALIZED mode (all {self.nstates - 1} non-sink states are sync triggers)")
        else:
            trigger_path = os.path.join(_root_dir, "labyrinth_benchmarks", "trigger_config.json")
            if not os.path.exists(trigger_path):
                raise FileNotFoundError(f"Trigger config not found: {trigger_path}")

            with open(trigger_path, "r") as f:
                triggers = json.load(f)

            if str(bid) not in triggers:
                raise KeyError(f"Benchmark ID {bid} not found in {trigger_path}")

            trigger_data = triggers[str(bid)]

            # Initialize defaults
            self.state_trigger = []
            self.action_trigger = []
            self.obs_trigger = []

            # Handle Legacy Format (List of ints = States only)
            if isinstance(trigger_data, list):
                self.state_trigger = trigger_data
            
            # Handle New Format (Dictionary)
            elif isinstance(trigger_data, dict):
                self.state_trigger = trigger_data.get("states", [])
                self.action_trigger = trigger_data.get("actions", [])
                self.obs_trigger = trigger_data.get("observations", [])
            
            else:
                print(f"Warning: Unknown trigger format for {bid}. No triggers loaded.")

    def _get_data_file_path(self):
        """
        Get the appropriate data file path based on mode.

        Mode-specific data files have knowledge propagation baked into transitions:
        - decentralized: labyrinth_{bid}.data (no propagation)
        - semi_decentralized: labyrinth_{bid}_semi_decentralized.data (LOS-based)
        - centralized: labyrinth_{bid}_centralized.data (always propagates)

        Falls back to base file if mode-specific file doesn't exist.
        """
        base_file = os.path.join(_root_dir, "labyrinth_benchmarks", f"labyrinth_{self.bid}.data")

        if self.centralized:
            mode_file = os.path.join(_root_dir, "labyrinth_benchmarks", f"labyrinth_{self.bid}_centralized.data")
            if os.path.exists(mode_file):
                self.uses_mode_specific_data = True
                return mode_file
        elif not self.decentralized:  # semi-decentralized (default)
            mode_file = os.path.join(_root_dir, "labyrinth_benchmarks", f"labyrinth_{self.bid}_semi_decentralized.data")
            if os.path.exists(mode_file):
                self.uses_mode_specific_data = True
                return mode_file

        # Decentralized mode or fallback to base file
        return base_file

    def load_metadata(self):
        filename = self._get_data_file_path()
        if not os.path.exists(filename):
            print(f"Error: {filename} not found.")
            sys.exit(1)

        # Store the file path for later use by LabyrinthLoader
        self.data_file_path = filename

        if self.uses_mode_specific_data:
            mode_str = "centralized" if self.centralized else "semi_decentralized"
            print(f"Using mode-specific data file: {filename} (knowledge propagation baked in)")

        max_s = 0
        max_o = 0
        max_a_single = 0  
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("#"): continue 
                parts = line.split()
                if not parts: continue
                
                if parts[0] == 'T':
                    a1 = int(parts[1])
                    a2 = int(parts[2])
                    max_a_single = max(max_a_single, a1, a2)
                    max_s = max(max_s, int(parts[3]), int(parts[4]))
                    
                elif parts[0] == 'O':
                    max_o = max(max_o, int(parts[4]), int(parts[5]))
        
        # Auto-detect Action Space
        self.act_per_agent = max_a_single + 1 
        self.nacts = self.act_per_agent ** self.nagents
        self.nacts_factor = [self.act_per_agent, self.act_per_agent]
        
        # Auto-detect State/Obs Space
        # Add 1 for sink state
        self.nstates = max_s + 2 
        self.sink_state = self.nstates - 1
        
        self.obs_per_agent = max_o + 1
        self.nobs = self.obs_per_agent ** self.nagents
        self.nobs_factor = [self.obs_per_agent, self.obs_per_agent]
        
        self.nsq = self.nstates ** 2
        self.nso = self.nstates * self.nobs

        print(f"Loaded Labyrinth {self.bid}: {self.nstates} states.")
        print(f"Auto-detected Action Space: {self.act_per_agent} actions per agent.")


class LabyrinthLoader:
    """
    Loads labyrinth data using the FULL state encoding:
        s_idx = u1 * (N * T * 4) + u2 * (T * 4) + t_idx * 4 + found1 * 2 + found2
    """

    def __init__(self, config, start_node=0):
        self.c = config
        self.start_node = start_node

        # Derive num_nodes from state space
        total_states = self.c.nstates - 1  # Exclude sink
        self.num_nodes = self._find_num_nodes(total_states)
        self.num_targets = self.num_nodes - 1
        self.targets = [i for i in range(self.num_nodes) if i != self.start_node]
        
        # New: Store edges for topology
        self.edges_list = []

        print(f"Derived: {self.num_nodes} nodes, {self.num_targets} possible targets")

        # Storage for transitions, observations, rewards
        # Use SPARSE dict for transit to avoid 60+ GB memory for large problems
        self.transit = {}  # Sparse: only store non-zero entries
        self.obs = {}  # Sparse: only store non-zero entries
        self.reward = [0.0] * (self.c.nstates * self.c.nacts)
        self.init_beliefs = [0.0] * self.c.nstates

    def _find_num_nodes(self, total_states):
        """Solve N from total_states = 4 * N^2 * (N-1)"""
        for n in range(2, 100):
            if 4 * n * n * (n - 1) == total_states:
                return n
        return self.c.obs_per_agent

    def state_to_tuple(self, s_idx):
        """Decode state index to (u1, u2, t_idx, found1, found2)."""
        if s_idx == self.c.sink_state:
            return -1, -1, -1, -1, -1

        N = self.num_nodes
        T = self.num_targets

        found2 = s_idx % 2
        temp = s_idx // 2
        found1 = temp % 2
        temp = temp // 2
        t_idx = temp % T
        temp = temp // T
        u2 = temp % N
        u1 = temp // N
        return u1, u2, t_idx, found1, found2

    def tuple_to_state(self, u1, u2, t_idx, found1, found2):
        """Encode (u1, u2, t_idx, found1, found2) to state index."""
        N = self.num_nodes
        T = self.num_targets
        return u1 * (N * T * 4) + u2 * (T * 4) + t_idx * 4 + found1 * 2 + found2

    def _build_valid_joint_actions_per_state(self):
        """
        Build mapping: state -> list of valid joint action indices.

        Optimized approach: Filter based on PHYSICAL MOVEMENT of each agent.
        An action is valid if it causes actual movement (position change).
        We compute validity per position pair (u1, u2), not per full state,
        since movement depends only on position, not on t_idx/found flags.
        """
        nacts = self.c.nacts
        act_per_agent = self.c.act_per_agent

        # Step 1: Build valid single-agent actions per node based on physical movement
        valid_single_actions = {}

        for node in range(self.num_nodes):
            valid_a1 = {0}  # WAIT is always valid
            valid_a2 = {0}

            # Check agent 1's actions from position (node, 0)
            ref_state_a1 = self.tuple_to_state(node, 0, 0, 0, 0)
            for a1 in range(1, act_per_agent):
                ja = a1 + 0 * act_per_agent
                idx = ja * self.c.nsq + ref_state_a1 * self.c.nstates
                for s_next in range(self.c.nstates):
                    if self.transit.get(idx + s_next, 0.0) > 0.5:
                        nu1, _, _, _, _ = self.state_to_tuple(s_next)
                        if nu1 != node:
                            valid_a1.add(a1)
                        break

            # Check agent 2's actions from position (0, node)
            ref_state_a2 = self.tuple_to_state(0, node, 0, 0, 0)
            for a2 in range(1, act_per_agent):
                ja = 0 + a2 * act_per_agent
                idx = ja * self.c.nsq + ref_state_a2 * self.c.nstates
                for s_next in range(self.c.nstates):
                    if self.transit.get(idx + s_next, 0.0) > 0.5:
                        _, nu2, _, _, _ = self.state_to_tuple(s_next)
                        if nu2 != node:
                            valid_a2.add(a2)
                        break

            valid_single_actions[node] = (valid_a1, valid_a2)

        # Store single-agent action masks for decentralized planning
        # Format: {position: [list of valid actions]} for each agent
        self.valid_actions_per_position = [
            {node: sorted(valid_single_actions[node][0]) for node in range(self.num_nodes)},  # Agent 1
            {node: sorted(valid_single_actions[node][1]) for node in range(self.num_nodes)}   # Agent 2
        ]

        # Step 2: Build valid joint actions per position pair
        valid_ja_per_position = {}
        for u1 in range(self.num_nodes):
            for u2 in range(self.num_nodes):
                valid_a1, _ = valid_single_actions[u1]
                _, valid_a2 = valid_single_actions[u2]
                valid_ja = [a1 + a2 * act_per_agent for a1 in valid_a1 for a2 in valid_a2]
                valid_ja_per_position[(u1, u2)] = valid_ja

        # Step 3: Map states to their position-based valid actions
        valid_ja_per_state = {}
        for s in range(self.c.nstates):
            if s == self.c.sink_state:
                valid_ja_per_state[s] = [0]
                continue

            u1, u2, _, _, _ = self.state_to_tuple(s)
            if u1 == -1:
                valid_ja_per_state[s] = [0]
            else:
                valid_ja_per_state[s] = valid_ja_per_position[(u1, u2)]

        # Print statistics
        total_actions = self.c.nstates * nacts
        valid_count = sum(len(v) for v in valid_ja_per_state.values())
        avg_per_pos = sum(len(v) for v in valid_ja_per_position.values()) / max(1, len(valid_ja_per_position))
        print(f"Action Masks: {valid_count}/{total_actions} ({100*valid_count/total_actions:.1f}%) - avg {avg_per_pos:.1f}/position")

        return valid_ja_per_state

    def load_data(self):
        # Use the data file path from config (supports mode-specific files)
        default_path = os.path.join(_root_dir, "labyrinth_benchmarks", f"labyrinth_{self.c.bid}.data")
        filename = getattr(self.c, 'data_file_path', default_path)
        print(f"Loading {filename}...")

        self.reward = [-1.0] * (self.c.nstates * self.c.nacts)
        
        edges = set()

        with open(filename, "r") as data:
            for line in data:
                d = line.split()
                if not d: continue

                row_type = d[0]

                if row_type == "T":
                    # T <a1> <a2> <s_from> <s_to> <prob>
                    a1, a2 = int(d[1]), int(d[2])
                    s_from, s_to = int(d[3]), int(d[4])
                    prob = float(d[5])

                    act = a1 + self.c.act_per_agent * a2
                    self.transit[act * self.c.nsq + s_from * self.c.nstates + s_to] = prob
                    
                    if prob > 0:
                        u1, u2, _, _, _ = self.state_to_tuple(s_from)
                        nu1, nu2, _, _, _ = self.state_to_tuple(s_to)
                        
                        # If Agent 1 moved (and valid state)
                        if u1 != -1 and nu1 != -1 and u1 != nu1:
                            edges.add((u1, nu1))
                        
                        # If Agent 2 moved
                        if u2 != -1 and nu2 != -1 and u2 != nu2:
                            edges.add((u2, nu2))

                elif row_type == "O":
                    # O <a1> <a2> <s_end> <o1> <o2> <prob>
                    a1, a2 = int(d[1]), int(d[2])
                    s_end = int(d[3])
                    o1, o2 = int(d[4]), int(d[5])
                    prob = float(d[6])

                    act = a1 + self.c.act_per_agent * a2
                    o = o1 + self.c.obs_per_agent * o2
                    self.obs[act * self.c.nso + s_end * self.c.nobs + o] = prob

        self.edges_list = list(edges)
        print(f"Graph Topology Extracted: {len(self.edges_list)} edges found.")

        # Goal Condition Setup
        sink = self.c.sink_state
        goal_states = set()
        for s in range(self.c.nstates - 1):
            u1, u2, t_idx, found1, found2 = self.state_to_tuple(s)
            if (u1 == self.start_node and found1 == 1) or \
               (u2 == self.start_node and found2 == 1):
                goal_states.add(s)

        # Iterate only over non-zero transitions (sparse optimization)
        # This avoids O(nstates^2 * nacts) iteration for large problems
        transitions_to_redirect = []
        for flat_idx, prob in self.transit.items():
            if prob <= 0:
                continue
            # Decode flat index: flat_idx = a * nsq + s * nstates + s_next
            a = flat_idx // self.c.nsq
            remainder = flat_idx % self.c.nsq
            s = remainder // self.c.nstates
            s_next = remainder % self.c.nstates
            if s_next in goal_states and s_next != sink:
                transitions_to_redirect.append((flat_idx, a, s, s_next, prob))

        for flat_idx, a, s, s_next, prob in transitions_to_redirect:
            self.reward[a * self.c.nstates + s] = 100.0
            del self.transit[flat_idx]  # Remove transition to goal state
            sink_idx = a * self.c.nsq + s * self.c.nstates + sink
            self.transit[sink_idx] = self.transit.get(sink_idx, 0.0) + prob

        # Sink State Maintenance
        for a in range(self.c.nacts):
            self.reward[a * self.c.nstates + sink] = 0.0
            self.transit[a * self.c.nsq + sink * self.c.nstates + sink] = 1.0
            self.obs[a * self.c.nso + sink * self.c.nobs + 0] = 1.0

        # Initial Belief
        self.init_beliefs = [0.0] * self.c.nstates
        prob_per_target = 1.0 / self.num_targets

        for t_idx in range(self.num_targets):
            s_idx = self.tuple_to_state(self.start_node, self.start_node, t_idx, 0, 0)
            if s_idx < self.c.nstates:
                self.init_beliefs[s_idx] = prob_per_target

        # Build action masks for optimization (after transitions are loaded)
        self.valid_joint_actions_per_state = self._build_valid_joint_actions_per_state()

        return self.transit, self.obs, self.reward, self.init_beliefs

# ==========================================
# Simulation Helper
# ==========================================

def step_environment(config, transit, obs_matrix, current_state, joint_action):
    """Execute one step in the environment.

    Handles both deterministic and probabilistic transitions/observations.
    """
    # Transition (sample from distribution)
    start_idx = joint_action * config.nsq + current_state * config.nstates
    next_state = current_state

    # Check if transit/obs are sparse dicts or dense arrays
    transit_is_sparse = isinstance(transit, dict)
    obs_is_sparse = isinstance(obs_matrix, dict)

    # Build transition distribution
    trans_probs = []
    trans_states = []
    for s_next in range(config.nstates):
        idx = start_idx + s_next
        p = transit.get(idx, 0.0) if transit_is_sparse else transit[idx]
        if p > 0:
            trans_probs.append(p)
            trans_states.append(s_next)

    if trans_probs:
        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(trans_probs):
            cumsum += p
            if r < cumsum:
                next_state = trans_states[i]
                break
        else:
            next_state = trans_states[-1]

    # Observation (sample from distribution)
    start_idx_o = joint_action * config.nso + next_state * config.nobs
    joint_obs = 0

    # Build observation distribution
    obs_probs = []
    obs_ids = []
    for o in range(config.nobs):
        idx_o = start_idx_o + o
        p = obs_matrix.get(idx_o, 0.0) if obs_is_sparse else obs_matrix[idx_o]
        if p > 0:
            obs_probs.append(p)
            obs_ids.append(o)

    if obs_probs:
        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(obs_probs):
            cumsum += p
            if r < cumsum:
                joint_obs = obs_ids[i]
                break
        else:
            joint_obs = obs_ids[-1]

    return next_state, joint_obs

# ==========================================
# Decentralized Mode (using original decPOMDP.py)
# ==========================================

def run_labyrinth_decentralized(config, verbose=True):
    """
    Run the labyrinth simulation using the original decPOMDP.py algorithm.
    This is the fully decentralized mode with no sync triggers.
    """
    if OriginalDecPOMDP is None:
        print("Original decPOMDP solver not available. Exiting.")
        return 0

    # Load cached data
    cache_data = load_cached_labyrinth(config.bid)
    if cache_data is None:
        if verbose:
            print(f"Cache not found for labyrinth {config.bid}. Generating...")
        precompute_all(config.bid, config.horizon)
        cache_data = load_cached_labyrinth(config.bid)

    cached_config = create_config_from_cache(
        cache_data, config.horizon, config.maxit, config.ie_min2, config.alpha
    )
    loader = create_loader_from_cache(cache_data, cached_config)

    T = cache_data['T']
    O = cache_data['O']
    R = cache_data['R']
    init_b = cache_data['init_beliefs']

    # Load pdict from cache (fast) or convert on-the-fly (slow fallback)
    pdict_data = load_cached_pdict(config.bid) if load_cached_pdict is not None else None
    if pdict_data is not None:
        if verbose:
            print("Using cached pdict format for decPOMDP.py")
        T_pdict = pdict_data['T_pdict']
        O_pdict = pdict_data['O_pdict']
    else:
        # Fallback: generate pdict cache if possible, otherwise convert on-the-fly
        if precompute_pdict is not None:
            if verbose:
                print("Pdict cache not found. Generating (one-time cost)...")
            pdict_data = precompute_pdict(config.bid, cache_data)
            T_pdict = pdict_data['T_pdict']
            O_pdict = pdict_data['O_pdict']
        else:
            if verbose:
                print("Converting data to pdict format for decPOMDP.py (slow)...")
            T_pdict = flat_to_pdict_transitions(T, cached_config.nacts, cached_config.nstates)
            O_pdict = flat_to_pdict_obs(O, cached_config.nacts, cached_config.nstates, cached_config.nobs)

    if verbose:
        print(f"Starting positions: Agent1={loader.start_node}, Agent2={loader.start_node}")

    # Create original DecPOMDP solver
    dec_pomdp = OriginalDecPOMDP(
        nagents=2,
        nstates=cached_config.nstates,
        nactions=cached_config.nacts,
        nobs=cached_config.nobs,
        transitions=T_pdict,
        obs=O_pdict,
        rewards=list(R),
        init_beliefs=list(init_b),
        nacts_factor=cached_config.nacts_factor,
        nobs_factor=cached_config.nobs_factor,
        maxh=config.horizon,
        cluster_type="lossless", # "lossless", "finite_memory_cluster
        maxit=config.maxit,
        q_depth=config.ie_min2,
        alpha=config.alpha,
        iter_limit="inf", # "inf",
        maxrec="inf", # "inf"
        memory=None, # None
        heuristic="MDP",    # Use MDP heuristic (more stable)
        rec_type="MDP",     # Use MDP for terminal heuristic
        p_threshold_cluster=0,
        p_threshold_expand=0,
        policyvalfound=-math.inf,
        output=verbose
    )
    # Set required attributes not handled by constructor
    dec_pomdp.decentralized = False
    dec_pomdp.onesided = False

    # Sample true state
    states = list(range(cached_config.nstates))
    true_state = random.choices(states, weights=init_b, k=1)[0]

    u1, u2, t_idx, found1, found2 = loader.state_to_tuple(true_state)
    target_node = loader.targets[t_idx] if t_idx >= 0 and t_idx < len(loader.targets) else -1

    print(f"Labyrinth {cached_config.bid} | Start: U1={u1} U2={u2} Target={target_node} K1={found1} K2={found2}")
    print(f"\n--- Planning Phase (H: {config.horizon}) ---")

    # Run the solver
    t0 = time.time()
    try:
        val, _, _ = dec_pomdp.multi_agent_astar(config.horizon)
        elapsed = time.time() - t0

        print(f"Planning Time: {elapsed:.4f}s | Exp. Value: {val:.4f}")
        print(f"Expected Value (decentralized): {val:.4f}")

        return val
    except DecPOMDPMemoryLimitExceeded as e:
        elapsed = time.time() - t0
        print(f"Result: MO")
        print(f"Memory limit exceeded: {e}")
        print(f"Planning Time: {elapsed:.4f}s")
        return "MO"


# ==========================================
# Main Execution Loop
# ==========================================

def run_labyrinth(config, verbose=True, fixed_target_idx=None, visualize_policy=False):
    """
    Run the labyrinth simulation.
    Target is sampled uniformly from all non-start nodes unless fixed_target_idx is specified.
    Uses cached data for fast initialization (auto-generates cache if missing).

    Args:
        config: LabyrinthConfig object
        verbose: Whether to print detailed output
        fixed_target_idx: If specified, use this target index (0 to num_targets-1) instead of sampling
        visualize_policy: If True, generate and save a human-readable policy visualization
    """
    if SDecPOMDP is None:
        print("DecPOMDP  solver not available. Exiting.")
        return 0

    # Try to load cache first (fast path)
    cache_data = None
    qmdp_data = None
    use_cache = load_cached_labyrinth is not None
    used_noisy_cache = False  # Track if we loaded from noisy cache

    # Determine cache key
    cache_bid = config.bid

    if use_cache:
        if config.noisy and load_cached_noisy_labyrinth is not None:
            # Load from noisy cache (has different action/observation spaces)
            cache_data = load_cached_noisy_labyrinth(config.bid, config.detection_prob)
            if cache_data is None:
                if verbose:
                    print(f"Noisy cache not found for labyrinth {config.bid}. Generating (one-time cost)...")
                precompute_noisy_all(config.bid, config.detection_prob)
                cache_data = load_cached_noisy_labyrinth(config.bid, config.detection_prob)
            used_noisy_cache = True
            # No QMDP cache for noisy (would need separate precomputation)
            qmdp_data = None
        else:
            # Load from standard cache (using mode-specific cache key if applicable)
            cache_data = load_cached_labyrinth(cache_bid)
            if cache_data is None:
                if verbose:
                    print(f"Cache not found for labyrinth {cache_bid}. Generating (one-time cost)...")
                # Pass mode flags to generate cache from correct data file
                precompute_all(config.bid, config.horizon,
                              decentralized=config.decentralized,
                              centralized=config.centralized)
                cache_data = load_cached_labyrinth(cache_bid)

            qmdp_data = load_cached_qmdp(cache_bid, config.horizon)
            if qmdp_data is None:
                if verbose:
                    print(f"QMDP cache not found for horizon {config.horizon}. Generating...")
                from labyrinth_cache import precompute_qmdp
                qmdp_data = precompute_qmdp(cache_bid, config.horizon, cache_data)

    # Create config and loader from cache (fast) or from files (slow fallback)
    if use_cache and cache_data is not None:
        # Fast path: use cached data
        cached_config = create_config_from_cache(
            cache_data, config.horizon, config.maxit, config.ie_min2, config.alpha,
            replan_at_all_syncs=config.replan_at_all_syncs
        )
        loader = create_loader_from_cache(cache_data, cached_config)
        T = cache_data['T']
        O = cache_data['O']
        R = cache_data['R']
        init_b = cache_data['init_beliefs']
        active_config = cached_config
    else:
        # Slow fallback: load from files (should only happen for small problems or debugging)
        print("  WARNING: Loading from file without cache. This may be slow for large problems.")
        loader = LabyrinthLoader(config)
        T, O, R, init_b = loader.load_data()
        active_config = config

    # Apply noisy detection FIRST if enabled
    # Skip if we loaded from noisy cache (noise is already baked in)
    # This must happen BEFORE sync propagation so detection uncertainty is preserved
    if config.noisy and not used_noisy_cache and apply_noisy_detection is not None:
        T, R = apply_noisy_detection(
            T, R, config.detection_prob,
            loader.targets, loader.num_nodes, loader.num_targets,
            active_config.nacts, active_config.nstates,
            start_node=loader.start_node
        )

    # Apply knowledge propagation based on sync_trigger
    # Skip if using mode-specific data files (propagation already baked in)
    # Skip for drilling mode (no found flags to propagate)
    # This modifies T and R to propagate found flags at sync states
    skip_propagation = getattr(config, 'uses_mode_specific_data', False)
    is_drilling_mode = used_noisy_cache and cache_data['config'].get('drilling_mode', False)
    applied_propagation = False
    drilling_sync_trigger = None
    if is_drilling_mode:
        print("  Skipping knowledge propagation (drilling mode has no found flags)")
        # Translate sync_trigger from base labyrinth encoding to drilling mode encoding
        if config.state_trigger and len(config.state_trigger) > 0:
            drilling_sync_trigger = translate_sync_trigger_to_drilling(
                config.state_trigger, loader.num_nodes, loader.num_targets
            )
            # Verify translated indices are within bounds
            max_trigger = max(drilling_sync_trigger) if drilling_sync_trigger else -1
            sink = active_config.sink_state
            print(f"  Translated {len(config.state_trigger)} base sync_triggers to {len(drilling_sync_trigger)} drilling sync_triggers")
            print(f"  DEBUG: max_trigger={max_trigger}, sink_state={sink}, nstates={active_config.nstates}")
            if max_trigger >= active_config.nstates:
                print(f"  ERROR: sync_trigger index {max_trigger} >= nstates {active_config.nstates}!")
                drilling_sync_trigger = [s for s in drilling_sync_trigger if s < active_config.nstates]
                print(f"  Filtered to {len(drilling_sync_trigger)} valid triggers")
        else:
            # Decentralized mode: empty sync triggers
            drilling_sync_trigger = []
            print("  Using empty sync_trigger (decentralized drilling mode)")
    elif not skip_propagation and config.state_trigger and len(config.state_trigger) > 0 and apply_sync_knowledge_propagation is not None:
        T, R = apply_sync_knowledge_propagation(
            T, R, config.state_trigger,
            loader.num_nodes, loader.num_targets,
            active_config.nacts, active_config.nstates,
            start_node=loader.start_node
        )
        applied_propagation = True
    elif skip_propagation:
        print("  Skipping runtime knowledge propagation (baked into data file)")

    # Update cached_data if any transformations were applied
    # Skip for drilling mode (cache already has correct data, no propagation applied)
    if applied_propagation or (config.noisy and not is_drilling_mode):
        if cache_data is not None:
            # Check if T is sparse (dict) or dense (numpy array)
            if isinstance(T, dict):
                # Sparse format - rebuild CSR matrices from modified dict
                import numpy as np
                from scipy import sparse
                nactions = active_config.nacts
                nstates = active_config.nstates
                nsq = nstates * nstates

                T_csr_list = []
                for a in range(nactions):
                    rows, cols, data = [], [], []
                    for idx, val in T.items():
                        if val <= 0:
                            continue
                        a_idx = idx // nsq
                        if a_idx == a:
                            rem = idx % nsq
                            s = rem // nstates
                            sp = rem % nstates
                            rows.append(s)
                            cols.append(sp)
                            data.append(val)
                    T_csr = sparse.csr_matrix((data, (rows, cols)), shape=(nstates, nstates), dtype=np.float64)
                    T_csr_list.append(T_csr)

                cache_data['T_csr_list'] = T_csr_list
                cache_data['T'] = T
                cache_data['T_np'] = None  # Not available in sparse mode
            else:
                # Dense format - rebuild both T_np and T_csr_list from modified T
                import numpy as np
                from scipy import sparse
                nactions = active_config.nacts
                nstates = active_config.nstates

                T_np = T.reshape(nactions, nstates, nstates)
                cache_data['T_np'] = T_np
                cache_data['T'] = T

                # Rebuild T_csr_list for sparse QMDP computation
                T_csr_list = [sparse.csr_matrix(T_np[a]) for a in range(nactions)]
                cache_data['T_csr_list'] = T_csr_list
            cache_data['R'] = R
            cache_data['R_np'] = R.reshape(active_config.nacts, active_config.nstates)
            # Recompute QMDP with modified T and R
            qmdp_data = None

    if verbose:
        print(f"Starting positions: Agent1={loader.start_node}, Agent2={loader.start_node}")
        print(f"  DEBUG: active_config.nstates={active_config.nstates}, sink_state={active_config.sink_state}")

    # 1. Instantiate Model
    model = SDecPOMDPModel(
        nagents=active_config.nagents,
        nstates=active_config.nstates,
        nactions=active_config.nacts,
        nobs=active_config.nobs,
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=active_config.nacts_factor,
        nobs_factor=active_config.nobs_factor,
        cached_data=cache_data,
        sync_states=drilling_sync_trigger if is_drilling_mode else config.state_trigger,
        sync_actions=config.action_trigger,
        sync_observations=config.obs_trigger

    )
    
    # Inject action masks into model (optional, but keeps things tidy)
    model.valid_actions_per_state = loader.valid_joint_actions_per_state
    model.valid_actions_per_position = loader.valid_actions_per_position

    # 2. Instantiate Config (uses top-level USER CONFIGURATION constants)
    solver_config = RSSDAConfig(
        maxh=active_config.horizon,
        maxit=active_config.maxit,
        IEmin2=active_config.ie_min2,
        alpha=active_config.alpha,
        algorithm=ALGORITHM,
        TI1=TI1,
        TI2=TI2,
        TI3=TI3,
        TI4=TI4,
        score_limit=SCORE_LIMIT,
        cen_threshold=CEN_THRESHOLD,
        sm_temperature=SM_TEMPERATURE,
        iter_limit=ITER_LIMIT,
        rec_limit=REC_LIMIT,
        heuristic_type=HEURISTIC_TYPE,
        tail_heuristic_type=TAIL_HEURISTIC_TYPE,
        hybrid_r=HYBRID_R,
        max_clusters=MAX_CLUSTERS,
        adaptive_check=ADAPTIVE_CHECK,
        output=verbose
    )

    # 3. Instantiate Solver
    sdec_pomdp = SDecPOMDP(
        model=model,
        config=solver_config,
        qmdp_data=qmdp_data
    )

    print(f"Configuration: algorithm={sdec_pomdp.algorithm}, TI1={sdec_pomdp.TI1}, TI2={sdec_pomdp.TI2}, TI3={sdec_pomdp.TI3}, TI4={sdec_pomdp.TI4}, "
          f"iter_limit={sdec_pomdp.iter_limit}, rec_limit={sdec_pomdp.rec_limit}, heuristic_type={sdec_pomdp.heuristic_type}, "
          f"tail_heuristic_type={sdec_pomdp.tail_heuristic_type}, maxit={sdec_pomdp.maxit}, max_clusters={sdec_pomdp.max_clusters}, "
          f"ie_min2={sdec_pomdp.IEmin2}, hybrid_r={sdec_pomdp.hybrid_r}")
    print(f"Position action masks enabled: {sdec_pomdp.use_position_action_masks}")

    current_belief_idx = sdec_pomdp.dist_dict[int_tuple(init_b)]

    if getattr(loader, 'drilling_mode', False) and verbose:
        import numpy as np
        belief = sdec_pomdp.dists[current_belief_idx]
        h = active_config.horizon
        print(f"DEBUG QMDP: Horizon={h}, belief sum={sum(belief):.4f}")
        qmdp_vals = []
        for ja in range(active_config.nacts):
            q_val = np.dot(belief, sdec_pomdp.qmdp_Q[h, ja])
            qmdp_vals.append(q_val)
        print(f"DEBUG QMDP: max Q-value={max(qmdp_vals):.2f}, min Q-value={min(qmdp_vals):.2f}")
        best_action = max(range(len(qmdp_vals)), key=lambda x: qmdp_vals[x])
        print(f"DEBUG QMDP: best joint action={best_action}, value={qmdp_vals[best_action]:.2f}")

    # Determine true state: either use fixed target or sample uniformly
    # Check if drilling mode (3-tuple) vs standard mode (5-tuple)
    is_drilling_mode = getattr(loader, 'drilling_mode', False)

    if fixed_target_idx is not None:
        # Use the specified target index
        if fixed_target_idx < 0 or fixed_target_idx >= loader.num_targets:
            print(f"Error: fixed_target_idx {fixed_target_idx} out of range [0, {loader.num_targets-1}]")
            return 0
        if is_drilling_mode:
            true_state = loader.tuple_to_state(loader.start_node, loader.start_node, fixed_target_idx)
        else:
            true_state = loader.tuple_to_state(loader.start_node, loader.start_node, fixed_target_idx, 0, 0)
        t_idx = fixed_target_idx
    else:
        # Sample uniformly from initial belief
        states = list(range(active_config.nstates))
        true_state = random.choices(states, weights=init_b, k=1)[0]

    # Decode state and display start info
    if is_drilling_mode:
        u1, u2, t_idx = loader.state_to_tuple(true_state)
        target_node = loader.targets[t_idx] if t_idx >= 0 and t_idx < len(loader.targets) else -1
        if verbose:
            print(f"Drilling Labyrinth {active_config.bid} | Start: U1={u1} U2={u2} Target={target_node}")
    else:
        u1, u2, t_idx, found1, found2 = loader.state_to_tuple(true_state)
        target_node = loader.targets[t_idx] if t_idx >= 0 and t_idx < len(loader.targets) else -1
        print(f"Labyrinth {active_config.bid} | Start: U1={u1} U2={u2} Target={target_node} K1={found1} K2={found2}")

    current_horizon = active_config.horizon
    total_reward = 0
    total_plan_time = 0
    step_global = 0
    termination_flag = False

    while not termination_flag:
        print(f"\n--- Planning Phase (H: {current_horizon}) ---")
        sdec_pomdp.cluster_dict.clear()

        t0 = time.time()
        try:
            val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = sdec_pomdp.multi_agent_astar(
                current_horizon, init_beliefs=current_belief_idx)
        except MemoryLimitExceeded as e:
            print(f"Result: MO")
            print(f"Memory limit exceeded: {e}")
            return "MO"

        plan_time = time.time() - t0
        total_plan_time += plan_time
        print(f"Planning Time: {plan_time:.4f}s | Exp. Value: {val:.4f} | Policy: {policy} | Stage-Wise Centralization: {cent_vector}")

        if val == -math.inf or policy is None:
            print("Planner failed (Time out or no solution).")
            break

        # Generate policy visualization if requested (only on first planning phase)
        if visualize_policy and step_global == 0:
            output_file = f"policy_labyrinth_{config.bid}_h{current_horizon}.txt"
            viz = generate_policy_visualization(
                policy, clustering, cent_vector, cen_dists_map, clustering_cen,
                active_config, loader, T, sdec_pomdp, output_file=output_file
            )
            if verbose:
                print("\n" + viz)

        steps_to_execute = len(policy)
        
        # Truncate execution if policy is explicitly partial (indicated by cent_vector)
        # This preserves "current behavior" for partial policies regardless of the flag.
        if sdec_pomdp.algorithm == "approximate" and sdec_pomdp.TI1:
            try:
                relative_idx = cent_vector[1:].index(True)
                first_sync_idx = relative_idx + 1
                limit = first_sync_idx + 1
                if limit < steps_to_execute:
                    steps_to_execute = limit
            except ValueError:
                pass

        steps_to_execute = min(steps_to_execute, current_horizon)

        current_oh = [0] * active_config.nagents
        steps_executed = 0

        for step_local in range(steps_to_execute):
            # ============================================
            # NORMAL RSSDA POLICY EXECUTION
            # ============================================

            # Determine step type
            is_centralized_step = False
            c_ptr = -1
            if step_local < len(cen_dists_map):
                dists_at_step = cen_dists_map[step_local]
                if current_belief_idx in dists_at_step:
                    is_centralized_step = True
                    c_ptr = dists_at_step.index(current_belief_idx)

            # If flag is True, we stop BEFORE executing any non-initial sync step
            if active_config.replan_at_all_syncs and is_centralized_step and step_local > 0:
                if verbose:
                    print("Sync trigger encountered (runtime). Replanning...")
                break

            step_global += 1
            steps_executed += 1

            # Extract Action
            if is_centralized_step:
                if c_ptr < len(policy[step_local][1]):
                    joint_act = policy[step_local][1][c_ptr][0]
                else:
                    if verbose:
                        print("CRITICAL: Centralized policy pointer out of bounds.")
                    termination_flag = True
                    break
                act1 = joint_act % active_config.act_per_agent
                act2 = joint_act // active_config.act_per_agent
            else:
                act1 = policy[step_local][0][0][current_oh[0]]
                act2 = policy[step_local][0][1][current_oh[1]]
                joint_act = act1 + (act2 * active_config.act_per_agent)

            if act1 < 0 or act2 < 0:
                if verbose:
                    print(f"!!! CRITICAL: Planner returned invalid action {act1}/{act2} !!!")
                termination_flag = True
                break

            # Step Environment
            next_state, joint_obs = step_environment(active_config, T, O, true_state, joint_act)

            # Compute reward from R matrix for (state, action) pair
            # In drilling mode: +100 for correct drill, -200 for wrong drill, -1 for move/wait
            # Look up reward from R matrix (supports both dict and array formats)
            R_idx = joint_act * active_config.nstates + true_state
            if isinstance(R, dict):
                step_reward = R.get(R_idx, -1.0)
            else:
                step_reward = R[R_idx]
            total_reward += step_reward

            # Display
            # Extract agent positions from observation (o1=v1, o2=v2)
            o1 = joint_obs % active_config.obs_per_agent
            o2 = joint_obs // active_config.obs_per_agent

            # Check if drilling mode (3-tuple state) vs standard mode (5-tuple state)
            is_drilling_mode = getattr(loader, 'drilling_mode', False)

            if is_drilling_mode:
                # Drilling mode: state = (u1, u2, t_idx) - NO found flags
                u1, u2, t_idx = loader.state_to_tuple(true_state)
                k_str = ""  # No knowledge flags in drilling mode

                if step_reward > 50:
                    next_state_str = f"[DRILL SUCCESS +100]"
                elif step_reward < -150:
                    next_state_str = f"[DRILL FAILURE -200]"
                elif next_state == active_config.sink_state:
                    next_state_str = "Sink"
                else:
                    nu1, nu2, nt_idx = loader.state_to_tuple(next_state)
                    next_state_str = f"({nu1},{nu2})"
            else:
                # Standard mode: state = (u1, u2, t_idx, f1, f2)
                u1, u2, t_idx, f1, f2 = loader.state_to_tuple(true_state)

                if step_reward > 50:
                    next_state_str = f"({o1},{o2}) [GOAL REACHED]"
                    k_str = "K:(1,1)"
                elif next_state == active_config.sink_state:
                    next_state_str = "Sink"
                    k_str = f"K:({f1},{f2})"
                else:
                    nu1, nu2, nt_idx, nf1, nf2 = loader.state_to_tuple(next_state)
                    next_state_str = f"({nu1},{nu2})"
                    k_str = f"K:({nf1},{nf2})"

                # Format action names (different for drilling mode which has DRILL)
                if used_noisy_cache:
                    act1_str = format_noisy_action(act1, active_config.act_per_agent)
                    act2_str = format_noisy_action(act2, active_config.act_per_agent)
                    # Decode observations: obs = position * 2 + sensor
                    pos1, sens1 = decode_noisy_obs(o1, loader.num_nodes)
                    pos2, sens2 = decode_noisy_obs(o2, loader.num_nodes)
                    obs_str = f"({pos1}{sens1},{pos2}{sens2})"
                else:
                    act1_str = "WAIT" if act1 == 0 else f"MOVE({act1})"
                    act2_str = "WAIT" if act2 == 0 else f"MOVE({act2})"
                    # Decode observations: obs = position * 2 + found
                    pos1, found1_obs = decode_deterministic_obs(o1)
                    pos2, found2_obs = decode_deterministic_obs(o2)
                    found1_str = "F" if found1_obs == 1 else "-"
                    found2_str = "F" if found2_obs == 1 else "-"
                    obs_str = f"({pos1}{found1_str},{pos2}{found2_str})"

                step_type = '[SYNC]' if is_centralized_step else '[DEC]'
                print(f"Step {step_global} {step_type}: Pos({u1},{u2})->{next_state_str} "
                      f"[{act1_str},{act2_str}] {k_str} Obs:{obs_str} Rew:{step_reward} S':{next_state}")

            if step_reward > 50:
                if verbose:
                    is_drilling_mode = getattr(loader, 'drilling_mode', False)
                    if is_drilling_mode:
                        print("DRILL SUCCESS: Target found!")
                    else:
                        print("MISSION SUCCESS: Reward collected.")
                termination_flag = True
                break

            if step_reward < -150:
                # Drill failure (drilling mode only)
                if verbose:
                    print("DRILL FAILURE: Wrong location!")
                termination_flag = True
                break

            # Update Belief
            sparse_transitions = sdec_pomdp.get_terminal(current_belief_idx, joint_act)
            
            next_belief_idx = -1
            
            # Linear scan is efficient here (k is small, usually 1-5 transitions)
            for o, p, d in sparse_transitions:
                if o == joint_obs:
                    next_belief_idx = d
                    break
            
            if next_belief_idx == -1:
                if verbose:
                    print(f"CRITICAL: Impossible observation {joint_obs} for current belief!")
                termination_flag = True
                break

            current_belief_idx = next_belief_idx
            true_state = next_state

            # Update History
            if step_local < steps_to_execute - 1:
                o1 = joint_obs % active_config.obs_per_agent
                o2 = joint_obs // active_config.obs_per_agent
                try:
                    if is_centralized_step:
                        if step_local < len(clustering_cen) and len(clustering_cen[step_local]) > 0:
                            if c_ptr < len(clustering_cen[step_local][0]):
                                current_oh[0] = clustering_cen[step_local][0][c_ptr][o1]
                                current_oh[1] = clustering_cen[step_local][1][c_ptr][o2]
                    else:
                        # Check both: (1) clustering exists for this step, AND (2) it has entries for both agents
                        if step_local < len(clustering) and len(clustering[step_local]) >= active_config.nagents:
                            current_oh[0] = clustering[step_local][0][current_oh[0]][o1]
                            current_oh[1] = clustering[step_local][1][current_oh[1]][o2]
                        elif step_local < len(clustering) and len(clustering[step_local]) == 0:
                            # Planner bug: decentralized execution but no clustering data
                            # This happens when dec_split was ~0 but actions were still populated
                            if verbose:
                                print(f"  [PLANNER INCONSISTENCY] Step {step_local}: decentralized policy exists but clustering is empty")
                            raise IndexError(f"Empty clustering at step {step_local}")
                except IndexError:
                    if verbose:
                        print(f"Warning: History update index error at step {step_local}. Re-planning.")
                    break

        current_horizon -= steps_executed
        if current_horizon <= 0:
            termination_flag = True
        else:
            sdec_pomdp.maxh = current_horizon

    print(f"Total Reward: {total_reward} | Total Plan Time: {total_plan_time:.4f}s")
    return total_reward, total_plan_time


def compute_statistics(results):
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(results) / n
    if n == 1:
        return mean, 0.0, 0.0
    variance = sum((x - mean) ** 2 for x in results) / (n - 1)
    std_dev = math.sqrt(variance)
    std_error = std_dev / math.sqrt(n)
    return mean, std_dev, std_error


def run_fullsim(config, num_trials=1, verbose=False):
    """
    Run a full simulation campaign: iterates through every possible target node.
    If num_trials > 1, runs multiple trials for each target and averages the results
    to smooth out sensor noise variance (crucial for --noisy mode).

    Args:
        config: LabyrinthConfig object
        num_trials: Number of times to run each specific target (default 1)
        verbose: Whether to print detailed output per simulation

    Returns:
        dict with keys: 'mean_reward', 'std_error', 'avg_time', 'results', 'times', 'num_targets'
    """
    if SDecPOMDP is None:
        print("SDecPOMDP solver not available. Exiting.")
        return None

    # Load cache to determine number of targets
    cache_data = load_cached_labyrinth(config.bid)
    if cache_data is None:
        print(f"Cache not found for labyrinth {config.bid}. Generating...")
        precompute_all(config.bid, config.horizon)
        cache_data = load_cached_labyrinth(config.bid)

    cached_config = create_config_from_cache(
        cache_data, config.horizon, config.maxit, config.ie_min2, config.alpha,
        replan_at_all_syncs=config.replan_at_all_syncs
    )
    loader = create_loader_from_cache(cache_data, cached_config)
    num_targets = loader.num_targets

    print(f"\n{'='*60}")
    print(f"FULL SIMULATION: Labyrinth {config.bid}")
    print(f"Running {num_targets} targets, {num_trials} trials per target")
    print(f"Horizon: {config.horizon}, Targets: {loader.targets}")
    print(f"{'='*60}\n")

    # Suppress inner verbose output if running multiple trials to prevent log flooding
    inner_verbose = verbose
    if num_trials > 1 and verbose:
        print("Notice: Suppressing step-by-step output for inner trials (num_trials > 1).")
        inner_verbose = False

    # Store the averaged result for each target
    target_averages = []
    target_avg_times = []

    # Store every single raw result for deeper analysis if needed
    all_raw_results = []

    # Stratified outcome tracking per target node
    # Keys: target_node, Values: {'found': count, 'no_dig': count, 'wrong_target': count, 'rewards': []}
    stratified_outcomes = {}

    for t_idx in range(num_targets):
        target_node = loader.targets[t_idx]
        print(f"--- Target {t_idx+1}/{num_targets} (Node {target_node}) | Running {num_trials} trial(s) ---")

        current_target_rewards = []
        current_target_times = []

        # Initialize stratified tracking for this target
        stratified_outcomes[target_node] = {
            'found': 0,
            'no_dig': 0,
            'wrong_target': 0,
            'rewards': []
        }

        for trial in range(num_trials):
            # Create fresh config for each simulation to ensure clean state
            sim_config = LabyrinthConfig(
                config.bid, config.horizon, config.maxit,
                config.ie_min2, config.alpha, config.replan_at_all_syncs,
                config.decentralized, config.centralized,
                config.noisy, config.detection_prob
            )

            # Run simulation
            reward, plan_time = run_labyrinth(sim_config, verbose=inner_verbose, fixed_target_idx=t_idx)

            current_target_rewards.append(reward)
            current_target_times.append(plan_time)
            all_raw_results.append(reward)

            # Classify outcome based on reward
            # +100 (reward > 50): correct drill (target found)
            # -200 (reward < -150): wrong drill (wrong target)
            # Otherwise: no drill attempted (horizon exhausted with move/wait actions)
            stratified_outcomes[target_node]['rewards'].append(reward)
            if reward > 50:
                stratified_outcomes[target_node]['found'] += 1
            elif reward < -150:
                stratified_outcomes[target_node]['wrong_target'] += 1
            else:
                stratified_outcomes[target_node]['no_dig'] += 1

        # Compute averages for this specific target
        avg_reward = sum(current_target_rewards) / num_trials
        avg_time = sum(current_target_times) / num_trials

        target_averages.append(avg_reward)
        target_avg_times.append(avg_time)

        # Store average reward in stratified outcomes
        stratified_outcomes[target_node]['avg_reward'] = avg_reward

        # Print summary for this target
        if num_trials > 1:
            # Calculate local variance for this target
            variance = sum((x - avg_reward) ** 2 for x in current_target_rewards) / (num_trials - 1)
            std_dev = math.sqrt(variance)
            print(f"Target {target_node}: Avg Reward={avg_reward:.2f} (StdDev={std_dev:.2f}), Avg Time={avg_time:.2f}s")
            # Print raw values if they differ significantly (e.g., a mix of success and failure)
            if std_dev > 10.0:
                print(f"    Raw trials: {current_target_rewards}")
            print("")
        else:
            print(f"Target {target_node}: Reward={avg_reward}, Plan Time={avg_time:.2f}s\n")

    # Compute statistics across the population of targets (using the averaged values)
    mean_reward, std_dev, std_error = compute_statistics(target_averages)
    avg_total_time = sum(target_avg_times) / len(target_avg_times) if target_avg_times else 0.0

    # Compute aggregate outcome counts
    total_found = sum(s['found'] for s in stratified_outcomes.values())
    total_no_dig = sum(s['no_dig'] for s in stratified_outcomes.values())
    total_wrong = sum(s['wrong_target'] for s in stratified_outcomes.values())
    total_trials = num_targets * num_trials

    print(f"\n{'='*60}")
    print(f"FULL SIMULATION RESULTS ({num_targets} targets, {num_trials} trials/target)")
    print(f"{'='*60}")
    print(f"Mean Reward (of averages): {mean_reward:.4f}")
    print(f"Std Dev (between targets): {std_dev:.4f}")
    print(f"Std Error:                 {std_error:.4f}")
    print(f"Avg Plan Time:             {avg_total_time:.2f}s")
    print(f"Total Run Time:            {sum(target_avg_times) * num_trials:.2f}s")
    print(f"{'='*60}")

    # Stratified outcomes report
    print(f"\n{'='*60}")
    print(f"STRATIFIED OUTCOMES BY TARGET NODE")
    print(f"{'='*60}")
    print(f"{'Node':<8} {'Found':<8} {'No Dig':<10} {'Wrong':<8} {'Avg Reward':<12}")
    print(f"{'-'*60}")
    for target_node in sorted(stratified_outcomes.keys()):
        outcome = stratified_outcomes[target_node]
        print(f"{target_node:<8} {outcome['found']:<8} {outcome['no_dig']:<10} {outcome['wrong_target']:<8} {outcome['avg_reward']:<12.2f}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<8} {total_found:<8} {total_no_dig:<10} {total_wrong:<8}")
    print(f"\nOutcome Rates (across {total_trials} total trials):")
    print(f"  Found:        {total_found:4d} ({100*total_found/total_trials:.1f}%)")
    print(f"  No Dig:       {total_no_dig:4d} ({100*total_no_dig/total_trials:.1f}%)")
    print(f"  Wrong Target: {total_wrong:4d} ({100*total_wrong/total_trials:.1f}%)")
    print(f"{'='*60}\n")

    return {
        'mean_reward': mean_reward,
        'std_error': std_error,
        'std_dev': std_dev,
        'avg_plan_time': avg_total_time,
        'results': target_averages,
        'all_raw_results': all_raw_results,
        'plan_times': target_avg_times,
        'num_targets': num_targets,
        'targets': loader.targets,
        'stratified_outcomes': stratified_outcomes,
        'outcome_totals': {
            'found': total_found,
            'no_dig': total_no_dig,
            'wrong_target': total_wrong,
            'total_trials': total_trials
        }
    }

def run_multiple_simulations(config, num_simulations, verbose=False):
    results = []
    plan_times = []
    if config.decentralized:
        mode_str = "DECENTRALIZED (decPOMDP.py)"
    elif config.centralized:
        mode_str = "CENTRALIZED (all states sync)"
    else:
        mode_str = "SEMI-DECENTRALIZED"
    print(f"\n{'='*60}\nRunning {num_simulations} simulation(s) for Labyrinth {config.bid} ({mode_str})\n{'='*60}\n")

    for i in range(num_simulations):
        print(f"--- Simulation {i+1}/{num_simulations} ---")
        sim_config = LabyrinthConfig(config.bid, config.horizon, config.maxit,
                                     config.ie_min2, config.alpha, config.replan_at_all_syncs,
                                     config.decentralized, config.centralized,
                                     config.noisy, config.detection_prob)
        if config.decentralized:
            total_reward = run_labyrinth_decentralized(sim_config, verbose=verbose)
            plan_time = 0.0  # Decentralized mode doesn't track plan time
        else:
            total_reward, plan_time = run_labyrinth(sim_config, verbose=verbose)
        results.append(total_reward)
        plan_times.append(plan_time)
        print(f"Simulation {i+1} Total Reward: {total_reward}\n")

    mean, std_dev, std_error = compute_statistics(results)
    avg_plan_time = sum(plan_times) / len(plan_times) if plan_times else 0.0
    print(f"\n{'='*60}\nRESULTS SUMMARY ({num_simulations} simulations)\n{'='*60}")
    print(f"Mean: {mean:.4f}, StdDev: {std_dev:.4f}, StdErr: {std_error:.4f}")
    print(f"Avg Plan Time: {avg_plan_time:.4f}s\n{'='*60}\n")
    return results, mean, std_dev, std_error


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sdec_labyrinth_approx.py <benchmark_id> <horizon> [num_sims] [maxit] [IEmin2] [alpha] [--verbose] [--replan_syncs] [--decentralized] [--centralized] [--fullsim] [--target X] [--noisy] [--policy]")
        print("\nModes:")
        print("  (default)        Semi-decentralized: uses LOS-based sync triggers from trigger_config.json")
        print("  --decentralized  Fully decentralized: no sync triggers, agents cannot share knowledge")
        print("  --centralized    Fully centralized: all states are sync triggers, agents always share knowledge")
        print("\nDrilling mode (--noisy flag):")
        print("  --noisy          Enable drilling labyrinth with noisy sensors and terminal DRILL action")
        print("                   - DRILL at target: +100 (WIN), DRILL at non-target: -200 (LOSS)")
        print("\nSimulation options:")
        print("  --fullsim        Run one simulation for each possible target node, return averaged results")
        print("                   (If num_sims > 1, runs multiple trials per target and averages them)")
        print("  --target X       Run a single simulation with a fixed target node ID (e.g., --target 9)")
        print("\nDebug/Analysis options:")
        print("  --policy         Generate human-readable policy visualization file")
        sys.exit(1)

    bid = sys.argv[1]
    h = int(sys.argv[2])

    # Filter out --target and its value before positional argument parsing
    filtered_argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--target':
            i += 2  # Skip --target and its value
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    # Parse num_sims (defaults to 1)
    # In --fullsim mode, this becomes "trials per target"
    num_sims = int(filtered_argv[3]) if len(filtered_argv) > 3 and not filtered_argv[3].startswith('--') else 1

    arg_idx = 4 if len(filtered_argv) > 3 and not filtered_argv[3].startswith('--') else 3
    maxit = int(filtered_argv[arg_idx]) if len(filtered_argv) > arg_idx and not filtered_argv[arg_idx].startswith('--') else MAXIT
    arg_idx += 1
    ie_min2_default = max(IE_MIN2, h - 3)
    ie_min2 = int(filtered_argv[arg_idx]) if len(filtered_argv) > arg_idx and not filtered_argv[arg_idx].startswith('--') else ie_min2_default
    arg_idx += 1
    alpha = float(filtered_argv[arg_idx]) if len(filtered_argv) > arg_idx and not filtered_argv[arg_idx].startswith('--') else ALPHA

    verbose = '--verbose' in sys.argv
    replan_syncs = '--replan_syncs' in sys.argv
    decentralized = '--decentralized' in sys.argv
    centralized = '--centralized' in sys.argv
    fullsim = '--fullsim' in sys.argv
    noisy = '--noisy' in sys.argv
    visualize_policy = '--policy' in sys.argv

    # Parse --target X flag
    target_node_id = None
    if '--target' in sys.argv:
        target_flag_idx = sys.argv.index('--target')
        if target_flag_idx + 1 < len(sys.argv):
            try:
                target_node_id = int(sys.argv[target_flag_idx + 1])
            except ValueError:
                print(f"Error: --target requires an integer node ID, got '{sys.argv[target_flag_idx + 1]}'")
                sys.exit(1)
        else:
            print("Error: --target requires a node ID argument")
            sys.exit(1)

    # Validate mutually exclusive flags
    if decentralized and centralized:
        print("Error: --decentralized and --centralized are mutually exclusive.")
        sys.exit(1)

    if fullsim and target_node_id is not None:
        print("Error: --fullsim and --target are mutually exclusive.")
        sys.exit(1)

    conf = LabyrinthConfig(bid, h, maxit, ie_min2, alpha,
                           replan_at_all_syncs=replan_syncs,
                           decentralized=decentralized,
                           centralized=centralized,
                           noisy=noisy)

    if fullsim:
        # Run full simulation over all targets
        # Pass num_sims as num_trials to smooth out sensor noise
        run_fullsim(conf, num_trials=num_sims, verbose=verbose)
    elif target_node_id is not None:
        # Run single simulation with fixed target node
        cache_data = load_cached_labyrinth(conf.bid)
        if cache_data is None:
            precompute_all(conf.bid, conf.horizon)
            cache_data = load_cached_labyrinth(conf.bid)

        cached_config = create_config_from_cache(
            cache_data, conf.horizon, conf.maxit, conf.ie_min2, conf.alpha,
            replan_at_all_syncs=conf.replan_at_all_syncs
        )
        loader = create_loader_from_cache(cache_data, cached_config)

        if target_node_id not in loader.targets:
            print(f"Error: Node {target_node_id} is not a valid target.")
            print(f"Valid target nodes: {loader.targets}")
            sys.exit(1)

        target_idx = loader.targets.index(target_node_id)
        print(f"Using fixed target: node {target_node_id} (target index {target_idx})")
        
        # If num_sims > 1, we can also run multiple trials for this specific target
        if num_sims > 1:
            print(f"Running {num_sims} trials for fixed target {target_node_id}...")
            rewards = []
            for i in range(num_sims):
                r, _ = run_labyrinth(conf, verbose=(verbose and i==0), fixed_target_idx=target_idx, visualize_policy=(visualize_policy and i==0))
                rewards.append(r)
            mean, std, err = compute_statistics(rewards)
            print(f"\nFixed Target {target_node_id} Results ({num_sims} trials):")
            print(f"Mean: {mean:.4f}, StdDev: {std:.4f}")
        else:
            _, _ = run_labyrinth(conf, verbose=verbose, fixed_target_idx=target_idx, visualize_policy=visualize_policy)

    elif num_sims == 1:
        if decentralized and not noisy:
            run_labyrinth_decentralized(conf, verbose=verbose)
        else:
            _, _ = run_labyrinth(conf, verbose=verbose, visualize_policy=visualize_policy)
    else:
        run_multiple_simulations(conf, num_sims, verbose=verbose)