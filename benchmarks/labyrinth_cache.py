"""
Labyrinth Precomputation Cache
Caches expensive initialization data to disk for fast loading.

Uses SPARSE storage for transition/observation matrices to keep cache small.

Usage:
    # First run (or after changes): precompute and save cache
    python labyrinth_cache.py precompute 1 20

    # In your benchmark: load from cache
    from labyrinth_cache import load_cached_labyrinth, load_cached_decpomdp

    # Load preprocessed labyrinth data
    data = load_cached_labyrinth(bid=1)

    # Create DecPOMDP with cached QMDP (much faster)
    dec_pomdp = load_cached_decpomdp(bid=1, horizon=20, config=config, data=data)
"""

import os
import sys
import pickle
import numpy as np
import time
from scipy import sparse

CACHE_DIR = os.path.join(os.path.dirname(__file__), "labyrinth_cache")


def get_cache_path(bid, suffix):
    """Get cache file path for a given benchmark ID."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"labyrinth_{bid}_{suffix}.pkl")


def precompute_labyrinth(bid, decentralized=False, centralized=False):
    """
    Precompute and cache all labyrinth data for a given benchmark ID.
    Uses COO-style sparse storage (indices + values) for efficient storage.
    Also pre-computes transposed arrays and numpy matrices to skip expensive DecPOMDP init.

    Args:
        bid: Benchmark ID (e.g., "chamber_3d_015")
        decentralized: If True, use decentralized mode (no knowledge propagation)
        centralized: If True, use centralized mode (instant knowledge propagation)
        (default is semi-decentralized with LOS-based propagation)
    """
    from sdec_labyrinth import LabyrinthConfig, LabyrinthLoader

    mode_str = "centralized" if centralized else ("decentralized" if decentralized else "semi_decentralized")
    print(f"Precomputing labyrinth {bid} (mode={mode_str})...")
    t0 = time.time()

    # Create minimal config with mode flags to load correct data file
    config = LabyrinthConfig(bid, horizon=1, maxit=1, ie_min2=1, alpha=0.1,
                             decentralized=decentralized, centralized=centralized)
    loader = LabyrinthLoader(config)
    T, O, R, init_b = loader.load_data()

    nactions = config.nacts
    nstates = config.nstates
    nobs = config.nobs
    nsq = config.nsq
    nso = config.nso

    # T and O are now sparse dicts from LabyrinthLoader
    # Convert to COO format (list of (flat_idx, value) tuples)
    T_total_size = nactions * nsq
    O_total_size = nactions * nso

    # T and O are already sparse dicts - just convert to list of tuples
    T_indices = [(i, v) for i, v in T.items() if v > 0]
    O_indices = [(i, v) for i, v in O.items() if v > 0]

    print(f"  T sparsity: {len(T_indices)}/{T_total_size} ({100*len(T_indices)/T_total_size:.6f}%)")
    print(f"  O sparsity: {len(O_indices)}/{O_total_size} ({100*len(O_indices)/O_total_size:.6f}%)")

    # Pre-compute transposed arrays from sparse COO format (memory-efficient)
    print("  Pre-computing transposed arrays from sparse data...")
    t1 = time.time()

    # transitions_transpose: T[a, s2, s1] -> T_transpose[a, s1, s2]
    # Iterate only over non-zero entries from T_indices
    T_transpose_coo = []
    for idx, val in T_indices:
        # Decode: idx = act * nsq + s * nstates + s_next
        act = idx // nsq
        remainder = idx % nsq
        s = remainder // nstates
        s_next = remainder % nstates
        # Transpose: swap s and s_next
        new_idx = act * nsq + s_next * nstates + s
        T_transpose_coo.append((new_idx, val))

    # obs_transpose: O[a, snew, o] -> O_transpose[a, o, snew]
    # Iterate only over non-zero entries from O_indices
    O_transpose_coo = []
    for idx, val in O_indices:
        # Decode: idx = act * nso + snew * nobs + o
        act = idx // nso
        remainder = idx % nso
        snew = remainder // nobs
        o = remainder % nobs
        # Transpose: O[a, snew, o] -> O_transpose[a, o, snew]
        new_idx = act * nobs * nstates + o * nstates + snew
        O_transpose_coo.append((new_idx, val))

    print(f"    Transposed arrays computed in {time.time()-t1:.2f}s")

    # Pre-compute SPARSE CSR matrices (memory-efficient for large problems)
    # This replaces the dense numpy arrays that caused 18+ GB memory errors
    print("  Pre-computing SPARSE CSR matrices...")
    t2 = time.time()

    # Build sparse CSR matrices for T: one per action, shape (nstates, nstates)
    T_csr_list = []
    for a in range(nactions):
        rows, cols, data = [], [], []
        for idx, val in T_indices:
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

    # Build sparse CSR matrices for O: one per action, shape (nstates, nobs)
    O_csr_list = []
    for a in range(nactions):
        rows, cols, data = [], [], []
        for idx, val in O_indices:
            a_idx = idx // nso
            if a_idx == a:
                rem = idx % nso
                s = rem // nobs
                o = rem % nobs
                rows.append(s)
                cols.append(o)
                data.append(val)
        O_csr = sparse.csr_matrix((data, (rows, cols)), shape=(nstates, nobs), dtype=np.float64)
        O_csr_list.append(O_csr)

    # R is small, keep as dense numpy array
    R_np = np.array(R, dtype=np.float64).reshape(nactions, nstates)

    # Calculate memory savings
    total_nnz_T = sum(m.nnz for m in T_csr_list)
    total_nnz_O = sum(m.nnz for m in O_csr_list)
    sparse_T_bytes = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in T_csr_list)
    sparse_O_bytes = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in O_csr_list)
    dense_T_bytes = nactions * nstates * nstates * 8
    dense_O_bytes = nactions * nstates * nobs * 8

    print(f"    Sparse matrices created in {time.time()-t2:.2f}s")
    print(f"    T: {total_nnz_T} non-zeros, {sparse_T_bytes/(1024**2):.2f} MiB (vs {dense_T_bytes/(1024**3):.2f} GiB dense)")
    print(f"    O: {total_nnz_O} non-zeros, {sparse_O_bytes/(1024**2):.2f} MiB (vs {dense_O_bytes/(1024**2):.2f} MiB dense)")

    # Store all relevant data
    cache_data = {
        'config': {
            'bid': config.bid,
            'nagents': config.nagents,
            'nstates': config.nstates,
            'nacts': config.nacts,
            'nobs': config.nobs,
            'act_per_agent': config.act_per_agent,
            'obs_per_agent': config.obs_per_agent,
            'nacts_factor': config.nacts_factor,
            'nobs_factor': config.nobs_factor,
            'nsq': config.nsq,
            'nso': config.nso,
            'sink_state': config.sink_state,
            'state_trigger': config.state_trigger,
        },
        'loader': {
            'num_nodes': loader.num_nodes,
            'num_targets': loader.num_targets,
            'targets': loader.targets,
            'start_node': loader.start_node,
            'edges_list': loader.edges_list,
        },
        'T_coo': T_indices,  # COO format: list of (idx, value)
        'O_coo': O_indices,  # COO format: list of (idx, value)
        'T_transpose_coo': T_transpose_coo,  # COO for transposed T
        'O_transpose_coo': O_transpose_coo,  # COO for transposed O
        'T_size': T_total_size,  # Total flat size (nactions * nsq)
        'O_size': O_total_size,  # Total flat size (nactions * nso)
        'T_transpose_size': nactions * nsq,
        'O_transpose_size': nactions * nobs * nstates,
        'T_csr_list': T_csr_list,  # SPARSE CSR matrices (replaces T_np)
        'O_csr_list': O_csr_list,  # SPARSE CSR matrices (replaces O_np)
        'R_np': R_np,  # Pre-computed numpy array (small, keep dense)
        'R': R,  # Keep as list (small)
        'init_beliefs': init_b,  # Keep as list (small)
        'valid_actions_per_state': loader.valid_joint_actions_per_state,
        'valid_actions_per_position': loader.valid_actions_per_position,  # Single-agent action masks for decentralized planning
        'sparse': True,  # Flag indicating v5 sparse format
    }

    # Use mode-specific cache key if using mode-specific data file
    if config.uses_mode_specific_data:
        if centralized:
            cache_bid = f"{bid}_centralized"
        else:
            cache_bid = f"{bid}_semi_decentralized"
    else:
        cache_bid = bid

    cache_path = get_cache_path(cache_bid, "data_v5")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cached labyrinth {cache_bid}: {size_mb:.1f} MB in {elapsed:.2f}s")
    print(f"  States: {config.nstates}, Actions: {config.nacts}, Obs: {config.nobs}")

    return cache_data


def precompute_qmdp(bid, max_horizon, cache_data=None):
    """
    Precompute QMDP Q-values for a given labyrinth and horizon.
    """
    if cache_data is None:
        cache_data = load_cached_labyrinth(bid)

    print(f"Precomputing QMDP for labyrinth {bid}, horizon {max_horizon}...")
    t0 = time.time()

    cfg = cache_data['config']
    nactions = cfg['nacts']
    nstates = cfg['nstates']
    nsq = cfg['nsq']

    # Get R (rewards) - small, always available
    if 'R_np' in cache_data:
        R = cache_data['R_np']
    else:
        R = np.array(cache_data['R'], dtype=np.float64).reshape(nactions, nstates)

    # Use sparse matrices for QMDP computation (memory-efficient)
    if 'T_csr_list' in cache_data and cache_data.get('sparse', False):
        # Sparse QMDP computation
        T_csr_list = cache_data['T_csr_list']
        qmdp_Q = np.zeros((max_horizon + 1, nactions, nstates), dtype=np.float64)

        for h in range(1, max_horizon + 1):
            v_prev = np.max(qmdp_Q[h-1], axis=0)
            for a in range(nactions):
                qmdp_Q[h, a, :] = R[a, :] + T_csr_list[a] @ v_prev
    else:
        # Fallback to dense QMDP computation (for old cache formats)
        # Handle both v3 (COO) format and v1 (dense) format
        if 'T' not in cache_data and 'T_coo' in cache_data:
            # Reconstruct dense T from COO format
            T_flat = [0.0] * cache_data['T_size']
            for idx, val in cache_data['T_coo']:
                T_flat[idx] = val
            cache_data['T'] = T_flat

        T = np.array(cache_data['T'], dtype=np.float64).reshape(nactions, nstates, nstates)

        # Value iteration
        qmdp_Q = np.zeros((max_horizon + 1, nactions, nstates), dtype=np.float64)

        for h in range(1, max_horizon + 1):
            v_prev = np.max(qmdp_Q[h-1], axis=0)
            future_val = np.einsum('asr,r->as', T, v_prev)
            qmdp_Q[h] = R + future_val

    qmdp_data = {
        'qmdp_Q': qmdp_Q,
        'max_horizon': max_horizon,
    }

    cache_path = get_cache_path(bid, f"qmdp_h{max_horizon}")
    with open(cache_path, 'wb') as f:
        pickle.dump(qmdp_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cached QMDP h={max_horizon}: {size_mb:.1f} MB in {elapsed:.2f}s")

    return qmdp_data


def apply_sync_knowledge_propagation(T_input, R_flat, sync_trigger, num_nodes, num_targets, nactions, nstates, start_node=0):
    """
    Modify transition matrix AND rewards to propagate found flags at sync states.

    At sync states, if either agent has found=1, both agents get found=1.
    This implements knowledge sharing that occurs during communication at sync points.

    CRITICAL: Also updates rewards when redirected transitions now lead to goal states.

    Args:
        T_input: Transition data - can be:
                 - Flat numpy array (legacy dense format)
                 - Dict of {flat_idx: prob} (sparse format)
        R_flat: Flat reward array (will be modified)
        sync_trigger: Set of state indices where communication/sync occurs
        num_nodes: Number of nodes in the labyrinth
        num_targets: Number of targets
        nactions: Total number of joint actions
        nstates: Total number of states
        start_node: The start/goal node (default 0)

    Returns:
        (T_modified, R_modified) tuple - T_modified is same type as T_input

    State encoding: s = u1*(N*T*4) + u2*(T*4) + t_idx*4 + found1*2 + found2
    """
    import numpy as np

    if len(sync_trigger) == 0:
        # No sync states - no knowledge propagation
        return T_input, R_flat

    sync_set = set(sync_trigger)

    # Determine if input is sparse (dict) or dense (numpy array)
    is_sparse = isinstance(T_input, dict)

    if is_sparse:
        # SPARSE path - memory efficient for large problems
        T_modified = T_input.copy()  # Dict copy is memory-efficient
    else:
        # DENSE path - legacy format
        if isinstance(T_input, np.ndarray):
            T_modified = T_input.copy()
        else:
            T_modified = np.array(T_input, dtype=np.float64)

    if isinstance(R_flat, np.ndarray):
        R_modified = R_flat.copy()
    else:
        R_modified = np.array(R_flat, dtype=np.float64)

    N = num_nodes
    T = num_targets
    nsq = nstates * nstates

    def get_state_tuple(s_idx):
        """Decode state index to (u1, u2, t_idx, found1, found2)."""
        found2 = s_idx % 2
        temp = s_idx // 2
        found1 = temp % 2
        temp = temp // 2
        t_idx = temp % T
        temp = temp // T
        u2 = temp % N
        u1 = temp // N
        return u1, u2, t_idx, found1, found2

    def get_state_index(u1, u2, t_idx, found1, found2):
        """Encode (u1, u2, t_idx, found1, found2) to state index."""
        return u1 * (N * T * 4) + u2 * (T * 4) + t_idx * 4 + found1 * 2 + found2

    def is_goal_state(u1, u2, found1, found2):
        """Check if state satisfies goal condition."""
        return (u1 == start_node and found1 == 1) or (u2 == start_node and found2 == 1)

    # Sink state is the last state (nstates - 1)
    sink_state = nstates - 1

    # For each action and source state, check destination states
    t_modifications = []  # (old_idx, new_idx, prob, act, s_from, is_new_goal) tuples

    if is_sparse:
        # SPARSE: iterate only over non-zero transitions
        for idx, prob in list(T_modified.items()):
            if prob <= 0:
                continue

            # Decode flat index: idx = act * nsq + s_from * nstates + s_to
            act = idx // nsq
            remainder = idx % nsq
            s_from = remainder // nstates
            s_to = remainder % nstates

            u1, u2, t_idx, found1, found2 = get_state_tuple(s_to)

            # Check if this position is a sync position
            is_sync_position = False
            for f1 in range(2):
                for f2 in range(2):
                    variant = get_state_index(u1, u2, t_idx, f1, f2)
                    if variant in sync_set:
                        is_sync_position = True
                        break
                if is_sync_position:
                    break

            if is_sync_position and (found1 == 1 or found2 == 1):
                # Knowledge propagates: redirect to state with both found=1
                new_s_to = get_state_index(u1, u2, t_idx, 1, 1)

                if new_s_to != s_to:
                    is_new_goal = is_goal_state(u1, u2, 1, 1)
                    was_goal = is_goal_state(u1, u2, found1, found2)

                    if is_new_goal and not was_goal:
                        new_idx = act * nsq + s_from * nstates + sink_state
                        t_modifications.append((idx, new_idx, prob, act, s_from, True))
                    else:
                        new_idx = act * nsq + s_from * nstates + new_s_to
                        t_modifications.append((idx, new_idx, prob, act, s_from, False))
    else:
        # DENSE: iterate over all combinations (original code)
        for act in range(nactions):
            for s_from in range(nstates):
                for s_to in range(nstates):
                    idx = act * nsq + s_from * nstates + s_to
                    prob = T_modified[idx]

                    if prob <= 0:
                        continue

                    u1, u2, t_idx, found1, found2 = get_state_tuple(s_to)

                    # Check if this position is a sync position
                    is_sync_position = False
                    for f1 in range(2):
                        for f2 in range(2):
                            variant = get_state_index(u1, u2, t_idx, f1, f2)
                            if variant in sync_set:
                                is_sync_position = True
                                break
                        if is_sync_position:
                            break

                    if is_sync_position and (found1 == 1 or found2 == 1):
                        # Knowledge propagates: redirect to state with both found=1
                        new_s_to = get_state_index(u1, u2, t_idx, 1, 1)

                        if new_s_to != s_to:
                            is_new_goal = is_goal_state(u1, u2, 1, 1)
                            was_goal = is_goal_state(u1, u2, found1, found2)

                            if is_new_goal and not was_goal:
                                new_idx = act * nsq + s_from * nstates + sink_state
                                t_modifications.append((idx, new_idx, prob, act, s_from, True))
                            else:
                                new_idx = act * nsq + s_from * nstates + new_s_to
                                t_modifications.append((idx, new_idx, prob, act, s_from, False))

    # Apply T modifications and track goal probabilities for weighted rewards
    goal_prob = {}  # (act, s_from) -> probability going to goal
    for old_idx, new_idx, prob, act, s_from, is_new_goal in t_modifications:
        if is_sparse:
            # Sparse: remove old key, add to new key
            if old_idx in T_modified:
                del T_modified[old_idx]
            T_modified[new_idx] = T_modified.get(new_idx, 0.0) + prob
        else:
            T_modified[old_idx] = 0.0
            T_modified[new_idx] += prob
        if is_new_goal:
            key = (act, s_from)
            goal_prob[key] = goal_prob.get(key, 0.0) + prob

    # Apply weighted R modifications
    reward_updates = set()
    for (act, s_from), p_goal in goal_prob.items():
        original_reward = R_modified[act * nstates + s_from]
        R_modified[act * nstates + s_from] = p_goal * 100.0 + (1.0 - p_goal) * original_reward
        reward_updates.add((act, s_from))

    if t_modifications:
        print(f"  Knowledge propagation: {len(t_modifications)} transitions, {len(reward_updates)} rewards updated")

    return T_modified, R_modified


def apply_noisy_detection(T_flat, R_flat, detection_prob, targets, num_nodes, num_targets, nactions, nstates, start_node=0):
    """
    Modify transition matrix to make target detection probabilistic.

    In the standard labyrinth, when an agent is at the target node, detection is deterministic
    (found flag becomes 1 with probability 1.0). This function makes detection probabilistic:
    - With probability detection_prob: found flag becomes 1
    - With probability (1 - detection_prob): found flag stays 0

    Args:
        T_flat: Flat transition data - can be:
                - Flat numpy array (legacy dense format)
                - Dict of {flat_idx: prob} (sparse format)
        R_flat: Flat reward array (may be modified for goal state changes)
        detection_prob: Probability of detecting target when at target node (0.0 to 1.0)
        targets: List of target node indices
        num_nodes: Number of nodes in the labyrinth
        num_targets: Number of targets
        nactions: Total number of joint actions
        nstates: Total number of states
        start_node: The start/goal node (default 0)

    Returns:
        (T_modified, R_modified) tuple - T_modified is same type as T_flat

    State encoding: s = u1*(N*T*4) + u2*(T*4) + t_idx*4 + found1*2 + found2
    """
    if detection_prob >= 1.0:
        # Deterministic detection - no modification needed
        return T_flat, R_flat

    if detection_prob <= 0.0:
        raise ValueError("detection_prob must be > 0")

    # Determine if input is sparse (dict) or dense (numpy array)
    is_sparse = isinstance(T_flat, dict)

    if is_sparse:
        # SPARSE path - memory efficient for large problems
        T_modified = T_flat.copy()  # Dict copy is memory-efficient
    else:
        # DENSE path - legacy format
        if isinstance(T_flat, np.ndarray):
            T_modified = T_flat.copy()
        else:
            T_modified = np.array(T_flat, dtype=np.float64)

    if isinstance(R_flat, np.ndarray):
        R_modified = R_flat.copy()
    else:
        R_modified = np.array(R_flat, dtype=np.float64)

    N = num_nodes
    T = num_targets
    nsq = nstates * nstates
    sink_state = nstates - 1

    def get_state_tuple(s_idx):
        """Decode state index to (u1, u2, t_idx, found1, found2)."""
        if s_idx == sink_state:
            return -1, -1, -1, -1, -1
        found2 = s_idx % 2
        temp = s_idx // 2
        found1 = temp % 2
        temp = temp // 2
        t_idx = temp % T
        temp = temp // T
        u2 = temp % N
        u1 = temp // N
        return u1, u2, t_idx, found1, found2

    def get_state_index(u1, u2, t_idx, found1, found2):
        """Encode (u1, u2, t_idx, found1, found2) to state index."""
        return u1 * (N * T * 4) + u2 * (T * 4) + t_idx * 4 + found1 * 2 + found2

    def is_goal_state(u1, u2, found1, found2):
        """Check if state satisfies goal condition."""
        return (u1 == start_node and found1 == 1) or (u2 == start_node and found2 == 1)

    # Track modifications for reporting
    detection_splits = 0

    # Collect modifications to apply after iteration (to avoid modifying while iterating)
    # Format: (idx_to_clear, modifications_list) where modifications_list = [(idx, prob_delta), ...]
    t_modifications = []

    def process_transition(idx, prob, act, s_from, s_to):
        """Process a single transition, return modifications if it's a detection transition."""
        nonlocal detection_splits

        src_u1, src_u2, src_t_idx, src_found1, src_found2 = get_state_tuple(s_from)
        if src_t_idx < 0:
            return None

        target_node = targets[src_t_idx]

        dst_u1, dst_u2, dst_t_idx, dst_found1, dst_found2 = get_state_tuple(s_to)

        # Skip if target index changed (shouldn't happen in normal transitions)
        if dst_t_idx != src_t_idx:
            return None

        # Check for Agent 1 detection event: found1 goes 0->1 while at target
        agent1_detected = (src_found1 == 0 and dst_found1 == 1 and dst_u1 == target_node)

        # Check for Agent 2 detection event: found2 goes 0->1 while at target
        agent2_detected = (src_found2 == 0 and dst_found2 == 1 and dst_u2 == target_node)

        if not (agent1_detected or agent2_detected):
            return None

        # This is a detection transition - compute split
        modifications = []

        # Compute the "no detection" destination state
        no_detect_found1 = 0 if agent1_detected else dst_found1
        no_detect_found2 = 0 if agent2_detected else dst_found2
        s_to_no_detect = get_state_index(dst_u1, dst_u2, dst_t_idx, no_detect_found1, no_detect_found2)

        if agent1_detected and agent2_detected:
            # Both agents at target simultaneously - joint probability
            p_both_detect = detection_prob * detection_prob
            p_only_a1 = detection_prob * (1 - detection_prob)
            p_only_a2 = (1 - detection_prob) * detection_prob
            p_neither = (1 - detection_prob) * (1 - detection_prob)

            # Create states for each outcome
            s_to_only_a1 = get_state_index(dst_u1, dst_u2, dst_t_idx, 1, 0)
            s_to_only_a2 = get_state_index(dst_u1, dst_u2, dst_t_idx, 0, 1)

            # Compute indices
            idx_both = act * nsq + s_from * nstates + s_to
            idx_only_a1 = act * nsq + s_from * nstates + s_to_only_a1
            idx_only_a2 = act * nsq + s_from * nstates + s_to_only_a2
            idx_neither = act * nsq + s_from * nstates + s_to_no_detect

            # Original transition is cleared, new ones added
            modifications.append((idx_both, prob * p_both_detect))
            modifications.append((idx_only_a1, prob * p_only_a1))
            modifications.append((idx_only_a2, prob * p_only_a2))
            modifications.append((idx_neither, prob * p_neither))

        else:
            # Single agent detection
            p_detect = detection_prob
            p_no_detect = 1.0 - detection_prob

            # Original gets p_detect, new one gets p_no_detect
            modifications.append((idx, prob * p_detect))
            idx_no_detect = act * nsq + s_from * nstates + s_to_no_detect
            modifications.append((idx_no_detect, prob * p_no_detect))

        detection_splits += 1
        return (idx, modifications)

    if is_sparse:
        # SPARSE path: iterate only over non-zero transitions
        for idx, prob in list(T_modified.items()):
            if prob <= 0:
                continue

            # Decode flat index: idx = act * nsq + s_from * nstates + s_to
            act = idx // nsq
            remainder = idx % nsq
            s_from = remainder // nstates
            s_to = remainder % nstates

            # Skip sink state transitions
            if s_from == sink_state or s_to == sink_state:
                continue

            result = process_transition(idx, prob, act, s_from, s_to)
            if result is not None:
                t_modifications.append(result)
    else:
        # DENSE path: iterate over all combinations (original code)
        for act in range(nactions):
            for s_from in range(nstates - 1):  # Exclude sink state
                for s_to in range(nstates - 1):  # Exclude sink state
                    idx = act * nsq + s_from * nstates + s_to
                    prob = T_modified[idx]

                    if prob <= 0:
                        continue

                    result = process_transition(idx, prob, act, s_from, s_to)
                    if result is not None:
                        t_modifications.append(result)

    # Apply modifications
    for idx_to_clear, modifications in t_modifications:
        if is_sparse:
            # Sparse: remove original entry, add new entries
            if idx_to_clear in T_modified:
                del T_modified[idx_to_clear]
            for new_idx, new_prob in modifications:
                if new_prob > 0:
                    T_modified[new_idx] = T_modified.get(new_idx, 0.0) + new_prob
        else:
            # Dense: set to zero and add
            T_modified[idx_to_clear] = 0.0
            for new_idx, new_prob in modifications:
                T_modified[new_idx] += new_prob

    print(f"  Noisy detection: {detection_splits} transitions split (detection_prob={detection_prob})")

    return T_modified, R_modified


def load_cached_labyrinth(bid, fast_mode=True):
    """
    Load cached labyrinth data. Returns None if cache doesn't exist.

    Args:
        bid: Benchmark ID
        fast_mode: If True, skip COO->list reconstruction (use sparse/numpy arrays only).
                   If False, reconstruct all Python lists for full compatibility.

    Returns:
        Cache data dict containing:
        - T_csr_list: List of sparse CSR matrices for T (one per action)
        - O_csr_list: List of sparse CSR matrices for O (one per action)
        - R_np: Dense numpy array for rewards
        - config, loader info, etc.
    """
    # Try v5 (with sparse CSR matrices) first - most memory efficient
    cache_path = get_cache_path(bid, "data_v5")
    if os.path.exists(cache_path):
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        t1 = time.time()
        print(f"  (v5 sparse pickle load in {t1-t0:.2f}s)")

        # Sparse matrices are ready to use directly
        # T_csr_list and O_csr_list are already in the cache
        data['T_transpose'] = None
        data['O_transpose'] = None

        # For compatibility, store T and O as sparse dicts (memory-efficient)
        # This avoids creating 18+ GB dense arrays for large problems
        if fast_mode:
            # Store as sparse dict: {flat_idx: prob}
            data['T'] = {idx: val for idx, val in data['T_coo']}
            data['O'] = {idx: val for idx, val in data['O_coo']}
            data['T_is_sparse'] = True  # Flag to indicate sparse format

        print(f"  (v5 sparse load in {time.time()-t1:.2f}s, total {time.time()-t0:.2f}s)")
        return data

    # Try v4 (with pre-computed dense numpy arrays) - fallback
    cache_path = get_cache_path(bid, "data_v4")
    if os.path.exists(cache_path):
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        t1 = time.time()
        print(f"  (v4 pickle load in {t1-t0:.2f}s)")

        # Convert v4 dense arrays to sparse for consistency
        if 'T_np' in data and 'O_np' in data:
            cfg = data['config']
            nactions = cfg['nacts']
            T_np = data['T_np']
            O_np = data['O_np']

            # Create sparse matrices from dense
            T_csr_list = [sparse.csr_matrix(T_np[a]) for a in range(nactions)]
            O_csr_list = [sparse.csr_matrix(O_np[a]) for a in range(nactions)]
            data['T_csr_list'] = T_csr_list
            data['O_csr_list'] = O_csr_list
            data['sparse'] = True

            if fast_mode:
                data['T_transpose'] = None
                data['O_transpose'] = None
                # Convert dense T_np to sparse dict format (avoids 18+ GB memory allocation)
                # This iterates only over non-zeros in the sparse CSR matrices
                # Use Python int() to avoid numpy int32 overflow for large state spaces
                T_sparse = {}
                nstates = int(cfg['nstates'])
                nsq = nstates * nstates
                for a, T_csr in enumerate(T_csr_list):
                    coo = T_csr.tocoo()
                    for s, sp, val in zip(coo.row, coo.col, coo.data):
                        idx = a * nsq + int(s) * nstates + int(sp)
                        T_sparse[idx] = float(val)
                data['T'] = T_sparse
                data['T_is_sparse'] = True

                O_sparse = {}
                nobs = int(cfg['nobs'])
                nso = nstates * nobs
                for a, O_csr in enumerate(O_csr_list):
                    coo = O_csr.tocoo()
                    for s, o, val in zip(coo.row, coo.col, coo.data):
                        idx = a * nso + int(s) * nobs + int(o)
                        O_sparse[idx] = float(val)
                data['O'] = O_sparse

                print(f"  (v4->sparse in {time.time()-t1:.2f}s, total {time.time()-t0:.2f}s)")
                return data

        if not fast_mode:
            T = [0.0] * data['T_size']
            for idx, val in data['T_coo']:
                T[idx] = val
            data['T'] = T

            O = [0.0] * data['O_size']
            for idx, val in data['O_coo']:
                O[idx] = val
            data['O'] = O

            T_transpose = [0.0] * data['T_transpose_size']
            for idx, val in data['T_transpose_coo']:
                T_transpose[idx] = val
            data['T_transpose'] = T_transpose

            O_transpose = [0.0] * data['O_transpose_size']
            for idx, val in data['O_transpose_coo']:
                O_transpose[idx] = val
            data['O_transpose'] = O_transpose
            print(f"  (v4 COO->dense in {time.time()-t1:.2f}s, total {time.time()-t0:.2f}s)")

        return data

    # Try v3 (COO sparse) next
    cache_path = get_cache_path(bid, "data_v3")
    if os.path.exists(cache_path):
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct dense T and O from COO format
        T = [0.0] * data['T_size']
        for idx, val in data['T_coo']:
            T[idx] = val

        O = [0.0] * data['O_size']
        for idx, val in data['O_coo']:
            O[idx] = val

        data['T'] = T
        data['O'] = O

        print(f"  (v3 COO->dense in {time.time()-t0:.2f}s)")
        return data

    # Fall back to v1 (dense, if exists)
    cache_path = get_cache_path(bid, "data")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            # Convert numpy arrays to lists if needed
            if isinstance(data.get('T'), np.ndarray):
                data['T'] = data['T'].tolist()
            if isinstance(data.get('O'), np.ndarray):
                data['O'] = data['O'].tolist()
            if isinstance(data.get('R'), np.ndarray):
                data['R'] = data['R'].tolist()
            if isinstance(data.get('init_beliefs'), np.ndarray):
                data['init_beliefs'] = data['init_beliefs'].tolist()
            return data

    return None


def load_cached_qmdp(bid, horizon):
    """
    Load cached QMDP data. Returns None if cache doesn't exist or horizon is insufficient.
    """
    cache_path = get_cache_path(bid, f"qmdp_h{horizon}")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Try to find a cache with higher horizon
    for h in range(horizon, horizon + 50):
        alt_path = get_cache_path(bid, f"qmdp_h{h}")
        if os.path.exists(alt_path):
            with open(alt_path, 'rb') as f:
                data = pickle.load(f)
                # Truncate to requested horizon
                data['qmdp_Q'] = data['qmdp_Q'][:horizon+1]
                return data

    return None


def create_config_from_cache(cache_data, horizon, maxit, ie_min2, alpha,
                             replan_at_all_syncs=False, use_top_node_ti1=False,
                             top_threshold=0.5, enable_qpomdp_fallback=False):
    """
    Create a LabyrinthConfig-like object from cached data without file I/O.
    """
    class CachedConfig:
        pass

    cfg = cache_data['config']
    config = CachedConfig()

    # Copy all cached config values
    for key, value in cfg.items():
        setattr(config, key, value)

    # Set runtime parameters
    config.horizon = horizon
    config.maxit = maxit
    config.ie_min2 = ie_min2
    config.alpha = alpha
    config.replan_at_all_syncs = replan_at_all_syncs
    config.use_top_node_ti1 = use_top_node_ti1
    config.top_threshold = top_threshold
    config.enable_qpomdp_fallback = enable_qpomdp_fallback

    return config


def create_loader_from_cache(cache_data, config):
    """
    Create a LabyrinthLoader-like object from cached data.
    Supports both standard mode (5-tuple) and drilling mode (3-tuple).
    """
    # Check if this is drilling mode (no found flags)
    drilling_mode = cache_data['config'].get('drilling_mode', False)

    class CachedLoader:
        def __init__(self, is_drilling):
            self.c = config
            self.drilling_mode = is_drilling

        def state_to_tuple(self, s_idx):
            """Decode state. Returns 3-tuple for drilling, 5-tuple for standard."""
            if self.drilling_mode:
                return self._state_to_tuple_drilling(s_idx)
            else:
                return self._state_to_tuple_standard(s_idx)

        def _state_to_tuple_drilling(self, s_idx):
            """Decode drilling state: (u1, u2, t_idx) - NO found flags."""
            if s_idx == self.c.sink_state:
                return -1, -1, -1
            N = self.num_nodes
            T = self.num_targets
            t_idx = s_idx % T
            temp = s_idx // T
            u2 = temp % N
            u1 = temp // N
            return u1, u2, t_idx

        def _state_to_tuple_standard(self, s_idx):
            """Decode standard state: (u1, u2, t_idx, found1, found2)."""
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

        def tuple_to_state(self, *args):
            """Encode state. Accepts 3 args for drilling, 5 for standard."""
            if self.drilling_mode:
                u1, u2, t_idx = args
                return self._tuple_to_state_drilling(u1, u2, t_idx)
            else:
                u1, u2, t_idx, found1, found2 = args
                return self._tuple_to_state_standard(u1, u2, t_idx, found1, found2)

        def _tuple_to_state_drilling(self, u1, u2, t_idx):
            """Encode drilling state: (u1, u2, t_idx) - NO found flags."""
            N = self.num_nodes
            T = self.num_targets
            return u1 * (N * T) + u2 * T + t_idx

        def _tuple_to_state_standard(self, u1, u2, t_idx, found1, found2):
            """Encode standard state: (u1, u2, t_idx, found1, found2)."""
            N = self.num_nodes
            T = self.num_targets
            return u1 * (N * T * 4) + u2 * (T * 4) + t_idx * 4 + found1 * 2 + found2

    loader = CachedLoader(drilling_mode)
    loader_cfg = cache_data['loader']

    loader.num_nodes = loader_cfg['num_nodes']
    loader.num_targets = loader_cfg['num_targets']
    loader.targets = loader_cfg['targets']
    loader.start_node = loader_cfg['start_node']
    loader.edges_list = loader_cfg['edges_list']
    loader.valid_joint_actions_per_state = cache_data['valid_actions_per_state']
    # Load single-agent action masks if available (for decentralized planning)
    loader.valid_actions_per_position = cache_data.get('valid_actions_per_position')

    return loader


def load_cached_decpomdp(bid, horizon, config, cache_data, qmdp_data=None,
                         maxit=200, ie_min2=3, alpha=0.1):
    """
    Create a DecPOMDP solver using cached data.
    Skips expensive file parsing and QMDP computation.
    """
    from RSSDA_state_approx import DecPOMDP, int_tuple

    cfg = cache_data['config']

    # Convert cached numpy arrays to lists (DecPOMDP expects lists)
    T = cache_data['T'].tolist()
    O = cache_data['O'].tolist()
    R = cache_data['R'].tolist()
    init_b = cache_data['init_beliefs'].tolist()
    valid_actions = cache_data['valid_actions_per_state']

    dec_pomdp = DecPOMDP(
        nagents=cfg['nagents'],
        nstates=cfg['nstates'],
        nactions=cfg['nacts'],
        nobs=cfg['nobs'],
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=cfg['nacts_factor'],
        nobs_factor=cfg['nobs_factor'],
        maxh=horizon,
        maxit=config.maxit,
        IEmin2=config.ie_min2,
        alpha=config.alpha,
        sync_trigger=cfg['state_trigger'],
        algorithm="approximate",
        TI1=True,
        TI2=True,
        TI3=True,
        TI4=False,
        TI5=True,
        score_limit=20,
        cen_threshold=0.6,
        sm_temperature=0.6,
        use_top_node_ti1=False,
        top_threshold=0.8,
        iter_limit=1000,
        rec_limit=2,
        heuristic_type="HYBRID",
        tail_heuristic_type="QMDP",
        hybrid_r=2,
        memory=2,
        valid_actions_per_state=valid_actions
    )

    # Inject cached QMDP if available
    if qmdp_data is not None and qmdp_data['max_horizon'] >= horizon:
        dec_pomdp.qmdp_Q = qmdp_data['qmdp_Q'][:horizon+1]
        print(f"Loaded cached QMDP (h={horizon})")

    return dec_pomdp


def precompute_pdict(bid, cache_data=None):
    """
    Precompute and cache pdict format for original decPOMDP.py.

    This converts sparse COO format to the pdict format expected by decPOMDP.py:
    - transitions[act*nstates + s] = (indices_array, values_array)
    - obs[act*nstates + snew] = (indices_array, values_array)

    This is expensive but only needs to be done once per benchmark.
    """
    from array import array

    if cache_data is None:
        cache_data = load_cached_labyrinth(bid)

    if cache_data is None:
        print(f"Error: No cache data for labyrinth {bid}. Run precompute_labyrinth first.")
        return None

    print(f"Precomputing pdict format for labyrinth {bid}...")
    t0 = time.time()

    cfg = cache_data['config']
    nactions = cfg['nacts']
    nstates = cfg['nstates']
    nobs = cfg['nobs']
    nsq = cfg['nsq']
    nso = cfg['nso']

    # Build pdict for transitions from COO data
    # Group non-zero entries by (act, s_from)
    print("  Building T_pdict from sparse data...")
    t1 = time.time()

    T_groups = {}  # (act, s) -> list of (snew, prob)
    for idx, val in cache_data['T_coo']:
        if val <= 0:
            continue
        act = idx // nsq
        remainder = idx % nsq
        s = remainder // nstates
        snew = remainder % nstates
        key = act * nstates + s
        if key not in T_groups:
            T_groups[key] = []
        T_groups[key].append((snew, val))

    # Convert to pdict format: list of (indices_array, values_array)
    T_pdict = []
    for act in range(nactions):
        for s in range(nstates):
            key = act * nstates + s
            if key in T_groups:
                pairs = T_groups[key]
                indices = array('i', [p[0] for p in pairs])
                values = array('d', [p[1] for p in pairs])
            else:
                indices = array('i', [])
                values = array('d', [])
            T_pdict.append((indices, values))

    print(f"    T_pdict built in {time.time()-t1:.2f}s ({len(T_groups)} non-empty entries)")

    # Build pdict for observations from COO data
    # Group non-zero entries by (act, snew)
    print("  Building O_pdict from sparse data...")
    t2 = time.time()

    O_groups = {}  # (act, snew) -> list of (o, prob)
    for idx, val in cache_data['O_coo']:
        if val <= 0:
            continue
        act = idx // nso
        remainder = idx % nso
        snew = remainder // nobs
        o = remainder % nobs
        key = act * nstates + snew
        if key not in O_groups:
            O_groups[key] = []
        O_groups[key].append((o, val))

    # Convert to pdict format
    O_pdict = []
    for act in range(nactions):
        for snew in range(nstates):
            key = act * nstates + snew
            if key in O_groups:
                pairs = O_groups[key]
                indices = array('i', [p[0] for p in pairs])
                values = array('d', [p[1] for p in pairs])
            else:
                indices = array('i', [])
                values = array('d', [])
            O_pdict.append((indices, values))

    print(f"    O_pdict built in {time.time()-t2:.2f}s ({len(O_groups)} non-empty entries)")

    pdict_data = {
        'T_pdict': T_pdict,
        'O_pdict': O_pdict,
        'nactions': nactions,
        'nstates': nstates,
        'nobs': nobs,
    }

    cache_path = get_cache_path(bid, "pdict")
    with open(cache_path, 'wb') as f:
        pickle.dump(pdict_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cached pdict for {bid}: {size_mb:.1f} MB in {elapsed:.2f}s")

    return pdict_data


def load_cached_pdict(bid):
    """
    Load cached pdict format for decPOMDP.py. Returns None if cache doesn't exist.
    """
    cache_path = get_cache_path(bid, "pdict")
    if not os.path.exists(cache_path):
        return None

    t0 = time.time()
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  (pdict cache loaded in {time.time()-t0:.2f}s)")
    return data


def precompute_all(bid, max_horizon, decentralized=False, centralized=False):
    """Precompute all caches for a labyrinth.

    Args:
        bid: Benchmark ID
        max_horizon: Maximum horizon for QMDP precomputation
        decentralized: If True, use decentralized mode data file
        centralized: If True, use centralized mode data file
    """
    data = precompute_labyrinth(bid, decentralized=decentralized, centralized=centralized)

    # Determine the cache key that was used
    if centralized:
        cache_bid = f"{bid}_centralized"
    elif not decentralized:
        # Check if semi-decentralized data file was used
        import os
        semi_dec_file = f"labyrinth_benchmarks/labyrinth_{bid}_semi_decentralized.data"
        if os.path.exists(semi_dec_file):
            cache_bid = f"{bid}_semi_decentralized"
        else:
            cache_bid = bid
    else:
        cache_bid = bid

    precompute_qmdp(cache_bid, max_horizon, data)
    precompute_pdict(cache_bid, data)
    print(f"\nPrecomputation complete for labyrinth {cache_bid}, horizon {max_horizon}")


# ============================================================
# Noisy Labyrinth Cache Functions
# ============================================================

def precompute_noisy_labyrinth(bid, detection_prob=0.85):
    """
    Precompute and cache drilling labyrinth data.
    Expects the noisy .data file to exist at labyrinth_benchmarks/noisy/labyrinth_{bid}_noisy_{prob}.data

    Drilling mode uses simplified state encoding: s = u1*(N*T) + u2*T + t_idx (NO found flags)
    """
    prob_int = int(detection_prob * 100)
    filename = f"labyrinth_benchmarks/noisy/labyrinth_{bid}_noisy_{prob_int}.data"

    if not os.path.exists(filename):
        print(f"Error: Noisy labyrinth file not found: {filename}")
        print(f"Run: python labyrinth_noisy_generator.py {bid} {detection_prob}")
        return None

    print(f"Precomputing noisy labyrinth {bid} (detection_prob={detection_prob})...")
    t0 = time.time()

    # Parse metadata from file
    nstates = None
    act_per_agent = None
    obs_per_agent = None

    # First pass: get dimensions
    with open(filename, 'r') as f:
        for line in f:
            d = line.split()
            if not d:
                continue
            if d[0] == "states:":
                nstates = int(d[1])
            elif d[0] == "actions:":
                act_per_agent = int(d[1])
            elif d[0] == "observations:":
                obs_per_agent = int(d[1])

    if nstates is None or act_per_agent is None or obs_per_agent is None:
        print(f"Error: Could not parse dimensions from {filename}")
        return None

    nacts = act_per_agent ** 2
    nobs = obs_per_agent ** 2
    nsq = nstates * nstates
    nso = nstates * nobs
    sink_state = nstates - 1

    # Derive num_nodes from obs_per_agent (obs = pos * 2 + sensor)
    num_nodes = obs_per_agent // 2
    num_targets = num_nodes - 1
    start_node = 0

    print(f"  Noisy dimensions: {nstates} states, {nacts} actions, {nobs} obs")
    print(f"  Nodes: {num_nodes}, Targets: {num_targets}")

    # Initialize arrays
    T = [0.0] * (nacts * nsq)
    O = [0.0] * (nacts * nso)
    R = [-1.0] * (nacts * nstates)

    # Second pass: parse T, O, R
    edges = set()
    with open(filename, 'r') as f:
        for line in f:
            d = line.split()
            if not d:
                continue

            row_type = d[0]

            if row_type == "T":
                # T <a1> <a2> <s_from> <s_to> <prob>
                a1, a2 = int(d[1]), int(d[2])
                s_from, s_to = int(d[3]), int(d[4])
                prob = float(d[5])

                ja = a1 + act_per_agent * a2
                T[ja * nsq + s_from * nstates + s_to] = prob

                # Extract edges from movement
                if prob > 0 and s_from != sink_state and s_to != sink_state:
                    # Decode states using drilling mode encoding: s = u1*(N*T) + u2*T + t_idx
                    # NO found flags in drilling mode
                    t_idx_from = s_from % num_targets
                    temp_from = s_from // num_targets
                    u2_from = temp_from % num_nodes
                    u1_from = temp_from // num_nodes

                    t_idx_to = s_to % num_targets
                    temp_to = s_to // num_targets
                    u2_to = temp_to % num_nodes
                    u1_to = temp_to // num_nodes

                    if u1_from != u1_to:
                        edges.add((u1_from, u1_to))
                    if u2_from != u2_to:
                        edges.add((u2_from, u2_to))

            elif row_type == "O":
                # O <a1> <a2> <s_end> <o1> <o2> <prob>
                a1, a2 = int(d[1]), int(d[2])
                s_end = int(d[3])
                o1, o2 = int(d[4]), int(d[5])
                prob = float(d[6])

                ja = a1 + act_per_agent * a2
                o = o1 + obs_per_agent * o2
                O[ja * nso + s_end * nobs + o] = prob

            elif row_type == "R":
                # R <a1> <a2> <s> * <reward>
                a1, a2 = int(d[1]), int(d[2])
                s = int(d[3])
                reward = float(d[5])

                ja = a1 + act_per_agent * a2
                R[ja * nstates + s] = reward

    print(f"  Edges: {len(edges)}")

    # Convert to COO sparse format
    T_indices = [(i, v) for i, v in enumerate(T) if v > 0]
    O_indices = [(i, v) for i, v in enumerate(O) if v > 0]

    print(f"  T sparsity: {len(T_indices)}/{len(T)} ({100*len(T_indices)/len(T):.3f}%)")
    print(f"  O sparsity: {len(O_indices)}/{len(O)} ({100*len(O_indices)/len(O):.3f}%)")

    # Pre-compute transposed arrays
    print("  Pre-computing transposed arrays...")
    t1 = time.time()

    T_transpose_coo = []
    for act in range(nacts):
        for s1 in range(nstates):
            for s2 in range(nstates):
                idx = act * nsq + s2 * nstates + s1
                val = T[idx]
                if val > 0:
                    new_idx = act * nsq + s1 * nstates + s2
                    T_transpose_coo.append((new_idx, val))

    O_transpose_coo = []
    for act in range(nacts):
        for snew in range(nstates):
            for o in range(nobs):
                idx = act * nso + snew * nobs + o
                val = O[idx]
                if val > 0:
                    new_idx = act * nobs * nstates + o * nstates + snew
                    O_transpose_coo.append((new_idx, val))

    print(f"    Transposed arrays computed in {time.time()-t1:.2f}s")

    # Pre-compute SPARSE CSR matrices (same format as standard cache v5)
    print("  Pre-computing SPARSE CSR matrices...")
    t2 = time.time()

    # Build sparse CSR matrices for T: one per action, shape (nstates, nstates)
    T_csr_list = []
    for a in range(nacts):
        rows, cols, data = [], [], []
        for idx, val in T_indices:
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

    # Build sparse CSR matrices for O: one per action, shape (nstates, nobs)
    O_csr_list = []
    for a in range(nacts):
        rows, cols, data = [], [], []
        for idx, val in O_indices:
            a_idx = idx // nso
            if a_idx == a:
                rem = idx % nso
                s = rem // nobs
                o = rem % nobs
                rows.append(s)
                cols.append(o)
                data.append(val)
        O_csr = sparse.csr_matrix((data, (rows, cols)), shape=(nstates, nobs), dtype=np.float64)
        O_csr_list.append(O_csr)

    # R is small, keep as dense numpy array
    R_np = np.array(R, dtype=np.float64).reshape(nacts, nstates)

    # Calculate memory savings
    total_nnz_T = sum(m.nnz for m in T_csr_list)
    total_nnz_O = sum(m.nnz for m in O_csr_list)
    sparse_T_bytes = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in T_csr_list)
    sparse_O_bytes = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in O_csr_list)

    print(f"    Sparse matrices created in {time.time()-t2:.2f}s")
    print(f"    T: {total_nnz_T} non-zeros, {sparse_T_bytes/(1024**2):.2f} MiB")
    print(f"    O: {total_nnz_O} non-zeros, {sparse_O_bytes/(1024**2):.2f} MiB")

    # Generate initial belief (uniform over target locations)
    # Drilling mode encoding: s = u1*(N*T) + u2*T + t_idx (NO found flags)
    init_beliefs = [0.0] * nstates
    prob_per_target = 1.0 / num_targets
    for t_idx in range(num_targets):
        s = start_node * (num_nodes * num_targets) + start_node * num_targets + t_idx
        if s < nstates:
            init_beliefs[s] = prob_per_target

    # Build valid actions per state (for drilling: DRILL is valid everywhere)
    valid_actions_per_state = {}
    for s in range(nstates):
        if s == sink_state:
            valid_actions_per_state[s] = [0]  # Only WAIT in sink
        else:
            # All actions valid (including DRILL)
            valid_actions_per_state[s] = list(range(nacts))

    # Store cache data (v5 sparse format, same as standard cache)
    cache_data = {
        'config': {
            'bid': bid,
            'nagents': 2,
            'nstates': nstates,
            'nacts': nacts,
            'nobs': nobs,
            'act_per_agent': act_per_agent,
            'obs_per_agent': obs_per_agent,
            'nacts_factor': [act_per_agent, act_per_agent],
            'nobs_factor': [obs_per_agent, obs_per_agent],
            'nsq': nsq,
            'nso': nso,
            'sink_state': sink_state,
            'state_trigger': [],  # Will be populated when loading
            'detection_prob': detection_prob,
            'noisy': True,
            'drilling_mode': True,  # New drilling labyrinth format (no found flags)
        },
        'loader': {
            'num_nodes': num_nodes,
            'num_targets': num_targets,
            'targets': [i for i in range(num_nodes) if i != start_node],
            'start_node': start_node,
            'edges_list': list(edges),
        },
        'T_coo': T_indices,
        'O_coo': O_indices,
        'T_transpose_coo': T_transpose_coo,
        'O_transpose_coo': O_transpose_coo,
        'T_size': len(T),
        'O_size': len(O),
        'T_transpose_size': nacts * nsq,
        'O_transpose_size': nacts * nobs * nstates,
        'T_csr_list': T_csr_list,  # SPARSE CSR matrices (v5 format)
        'O_csr_list': O_csr_list,  # SPARSE CSR matrices (v5 format)
        'R_np': R_np,
        'R': R,
        'init_beliefs': init_beliefs,
        'valid_actions_per_state': valid_actions_per_state,
        'sparse': True,  # Flag indicating v5 sparse format
    }

    cache_path = get_cache_path(f"{bid}_noisy_{prob_int}", "data_v5")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cached noisy labyrinth {bid}: {size_mb:.1f} MB in {elapsed:.2f}s")

    return cache_data


def load_cached_noisy_labyrinth(bid, detection_prob=0.85, fast_mode=True):
    """
    Load cached drilling labyrinth data. Returns None if cache doesn't exist.
    Supports both v4 (dense) and v5 (sparse) formats for backwards compatibility.
    """
    prob_int = int(detection_prob * 100)

    # Try v5 (sparse) format first
    cache_path_v5 = get_cache_path(f"{bid}_noisy_{prob_int}", "data_v5")
    cache_path_v4 = get_cache_path(f"{bid}_noisy_{prob_int}", "data_v4")

    if os.path.exists(cache_path_v5):
        cache_path = cache_path_v5
        is_v5 = True
    elif os.path.exists(cache_path_v4):
        cache_path = cache_path_v4
        is_v5 = False
    else:
        print(f"Noisy cache not found: {cache_path_v5}")
        return None

    t0 = time.time()
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    t1 = time.time()
    print(f"  (noisy pickle load in {t1-t0:.2f}s)")

    # v5 sparse format (same as standard cache)
    if is_v5 or data.get('sparse', False):
        # Sparse format: use CSR matrices, but also create sparse dicts for step_environment
        data['T_transpose'] = None
        data['O_transpose'] = None
        # Convert COO to sparse dict (same as standard cache)
        data['T'] = {idx: val for idx, val in data['T_coo']}
        data['O'] = {idx: val for idx, val in data['O_coo']}
        data['T_is_sparse'] = True  # Flag to indicate sparse format
        print(f"  (v5 sparse load in {time.time()-t1:.2f}s, total {time.time()-t0:.2f}s)")
    elif fast_mode:
        # v4 dense format with fast mode
        data['T_transpose'] = None
        data['O_transpose'] = None
        data['T'] = data['T_np'].ravel()
        data['O'] = data['O_np'].ravel()
        print(f"  (fast mode in {time.time()-t1:.2f}s, total {time.time()-t0:.2f}s)")
    else:
        # v4 dense format without fast mode (COO reconstruction)
        T = [0.0] * data['T_size']
        for idx, val in data['T_coo']:
            T[idx] = val
        data['T'] = T

        O = [0.0] * data['O_size']
        for idx, val in data['O_coo']:
            O[idx] = val
        data['O'] = O

        T_transpose = [0.0] * data['T_transpose_size']
        for idx, val in data['T_transpose_coo']:
            T_transpose[idx] = val
        data['T_transpose'] = T_transpose

        O_transpose = [0.0] * data['O_transpose_size']
        for idx, val in data['O_transpose_coo']:
            O_transpose[idx] = val
        data['O_transpose'] = O_transpose

        print(f"  (noisy COO->dense in {time.time()-t1:.2f}s, total {time.time()-t0:.2f}s)")

    return data


def precompute_noisy_all(bid, detection_prob=0.85):
    """Generate drilling labyrinth files and cache."""
    from labyrinth_noisy_generator import generate_noisy_labyrinth

    # Generate drilling .data file
    filename, generator = generate_noisy_labyrinth(bid, detection_prob)

    # Cache the drilling labyrinth
    cache_data = precompute_noisy_labyrinth(bid, detection_prob)

    print(f"\nDrilling precomputation complete for labyrinth {bid}, detection_prob={detection_prob}")
    return cache_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python labyrinth_cache.py precompute <bid> <horizon>       - Precompute standard caches")
        print("  python labyrinth_cache.py precompute_noisy <bid> [prob]    - Precompute noisy caches")
        print("  python labyrinth_cache.py list                             - List cached labyrinths")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "precompute":
        if len(sys.argv) < 4:
            print("Usage: python labyrinth_cache.py precompute <bid> <horizon>")
            sys.exit(1)
        bid = sys.argv[2]
        horizon = int(sys.argv[3])
        precompute_all(bid, horizon)

    elif cmd == "precompute_noisy":
        if len(sys.argv) < 3:
            print("Usage: python labyrinth_cache.py precompute_noisy <bid> [detection_prob]")
            sys.exit(1)
        bid = sys.argv[2]
        detection_prob = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        precompute_noisy_all(bid, detection_prob)

    elif cmd == "list":
        if not os.path.exists(CACHE_DIR):
            print("No cache directory found.")
            sys.exit(0)
        files = os.listdir(CACHE_DIR)
        if not files:
            print("No cached files.")
        else:
            print("Cached files:")
            for f in sorted(files):
                path = os.path.join(CACHE_DIR, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  {f}: {size_mb:.1f} MB")
    else:
        print(f"Unknown command: {cmd}")
