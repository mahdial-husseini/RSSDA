"""
Mars Rover Domain Driver for Semi-Decentralized Multi-Agent Planning

This module implements the Mars rover coordination benchmark domain for evaluating
the RSSDA algorithm. Two rovers operate on a 4x4 grid, collecting rock samples
and transmitting data with partial observability and communication constraints.
Requires mars.data

Domain Description:
    - Two rovers navigate a 4x4 grid (16 positions each, 256 joint states)
    - Each rover can: move (4 directions), sample rock, or transmit data
    - Observations are noisy and depend on terrain features
    - Goal: Maximize scientific value through coordinated exploration

State Space:
    - Joint state encodes both rover positions: s = pos1 * 16 + pos2
    - 256 total states (16 x 16 grid positions)

Synchronization Triggers:
    - Configurable trigger sets for different communication scenarios:
      * TRIG_CENTRALIZED: Always synchronized (baseline)
      * TRIG_SEMI: Right-band triggers (rovers in columns 2-3)
      * TRIG_DECENTRAL: Never synchronized (fully decentralized)

Usage:
    python sdec_mars_approx.py <horizon> [maxit] [ie_min2] [alpha]

    Arguments:
        horizon     Planning horizon
        maxit       Maximum A* iterations (default: 200)
        ie_min2     Internal expansion minimum depth (default: 3)
        alpha       Pruning threshold (default: 0.2)

Reference:
    Based on the Mars rover domain from Amato et al., "Scalable Planning and
    Learning for Multiagent POMDPs"

Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

import sys
import os
import time
import random
import bisect
import math
import numpy as np
from array import array

# Add parent directories to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)  # Parent of benchmarks/
sys.path.insert(0, _root_dir)  # For RSSDA.py
sys.path.insert(0, os.path.join(_root_dir, 'baselines'))  # For decPOMDP.py

try:
    from RSSDA import SDecPOMDP, SDecPOMDPModel, RSSDAConfig, int_tuple, MemoryLimitExceeded
except ImportError:
    print("Warning: Could not import SDecPOMDP solver. Running in test mode.")
    SDecPOMDP = None
    SDecPOMDPModel = None
    int_tuple = tuple

# Import original decPOMDP for fully decentralized RSMAA mode
try:
    from decPOMDP import DecPOMDP as OriginalDecPOMDP, MemoryLimitExceeded as DecPOMDPMemoryLimitExceeded
except ImportError:
    print("Warning: Could not import original decPOMDP. decentralized_RSMAA mode may not work.")
    OriginalDecPOMDP = None
    DecPOMDPMemoryLimitExceeded = None

# ============================================================================
#                           USER CONFIGURATION
# ============================================================================
# Modify the parameters below to control solver behavior.
# ============================================================================

# --- Problem Settings ---
TRIGGER_MODE = "semi"           # "centralized", "semi", "decentralized", or "decentralized_RSMAA"
                                # decentralized_RSMAA uses original decPOMDP.py (RS-MAA* algorithm)

# additional semi-decentralization COM_MODE options
# partial communication (beacons aligned with survyes): [80, 81, 84, 85, 88, 89, 92, 93] + [160, 161, 162, 163, 168, 169, 170, 171]
# partial communication (beacons aligned with drills): [0, 2, 4, 6, 8, 10, 12, 14] + [240, 241, 242, 243, 244, 245, 246, 247]
# partial communication (LOS ridge envelopes): list(range(0, 32)) + list(range(64, 96))
# partial communication (co-located): list(range(0, 15))  + list(range(80, 95)) + list(range(160, 175)) + list(range(240, 255))
# mars_right_band_triggers()
# mars_chebyshev1_triggers()

def mars_right_band_triggers():
    S = []
    for y1 in range(4):
        for x1 in (2,3):
            for y2 in range(4):
                for x2 in (2,3):
                    sid = ((y1*4 + x1) * 16) + (y2*4 + x2)
                    S.append(sid)
    return sorted(S)

def mars_chebyshev1_triggers():
    S = []
    for y1 in range(4):
        for x1 in range(4):
            for y2 in range(4):
                for x2 in range(4):
                    if max(abs(x1-x2), abs(y1-y2)) <= 1:
                        sid = ((y1*4 + x1) * 16) + (y2*4 + x2)
                        S.append(sid)
    return sorted(set(S))

COM_MODE = mars_right_band_triggers()

# --- Core Algorithm ---
ALGORITHM = "exact"             # "exact" or "approximate" (enables TI approximations)

# --- Heuristic Type ---
# Controls upper-bound heuristic for A* search guidance.
#   "QMDP"   - Loose/fast: ignores observations (dot product of belief and state values)
#   "POMDP"  - Tight/exact: full centralized POMDP value function
#   "HYBRID" - Runs exact POMDP for first HYBRID_R steps, then QMDP
# Rule of thumb: Use "POMDP" for exact algorithm, "QMDP" or "HYBRID" for approximate.
HEURISTIC_TYPE = "HYBRID"
HYBRID_R = 1                    # Steps of exact POMDP before switching to QMDP (HYBRID mode only)

# --- Decentralized Heuristic Search ---
# When computing decentralized component heuristics, we run a bounded A* search.
# These parameters control early termination of that inner search.
MAXIT = 200                     # Max A* iterations for decentralized heuristic computation
ALPHA = 0.2                     # Early termination: stop if value <= upper - alpha*|upper|
IE_MIN2 = 3                     # Min depth of information-sharing stages for decentralized heuristic

# --- Approximation Techniques (TI Flags) ---
# Enable these for faster but approximate solutions. Requires ALGORITHM = "approximate".
TI1 = False  # Interleaving Planning/Execution: prune branches via consensus voting
TI2 = False  # Progress-based Pruning: limit per-entity exploration budget
TI3 = False  # Tail Approximation: use heuristics for final REC_LIMIT stages
TI4 = False  # Memory-Bounded Clustering: merge clusters with same recent observations

# --- TI1: Interleaving Parameters ---
# Consensus voting among top nodes to detect centralized stages early.
SCORE_LIMIT = 20                # Number of top policy nodes to sample for voting
CEN_THRESHOLD = 0.6             # Weighted vote threshold to force centralization
SM_TEMPERATURE = 0.6            # Softmax temperature for node weights (lower = focus on best)

# --- TI2: Progress Pruning ---
# Total iteration budget; per-entity budget B = ITER_LIMIT / (nagents + 1).
# Prunes nodes when exploration exceeds entity's fair share.
ITER_LIMIT = 1000

# --- TI3: Tail Approximation ---
# When remaining horizon <= REC_LIMIT, use heuristic value instead of exact expansion.
REC_LIMIT = 2
TAIL_HEURISTIC_TYPE = "HYBRID"

# --- TI4: Finite Memory Clustering ---
# Clusters with identical last MEMORY observations are merged.
MEMORY = 2

# ============================================================================
#                        END USER CONFIGURATION
# ============================================================================

# ==========================================
# Configuration & Constants
# ==========================================

class MarsConfig:
    def __init__(self):
        # CLI Argument Parsing (uses USER CONFIGURATION as defaults)
        # Usage: python sdec_mars_approx.py <horizon> [maxit] [IEmin2] [alpha]
        self.horizon = int(sys.argv[1])
        self.maxit = int(sys.argv[2]) if len(sys.argv) > 2 else MAXIT
        self.ie_min2 = int(sys.argv[3]) if len(sys.argv) > 3 else IE_MIN2
        self.alpha = float(sys.argv[4]) if len(sys.argv) > 4 else ALPHA

        # Replanning Settings
        # replan_at_all_syncs: If True, replan at every sync trigger (even for complete policies)
        #                      If False, only replan if policy is partial (current behavior)
        self.replan_at_all_syncs = False

        # Mars Problem Constants
        self.nagents = 2
        self.nstates = 256
        self.act_per_agent = 6
        self.nacts = self.act_per_agent ** self.nagents
        self.nacts_factor = [self.act_per_agent, self.act_per_agent]

        self.obs_per_agent = 8
        self.nobs = self.obs_per_agent ** self.nagents
        self.nobs_factor = [self.obs_per_agent, self.obs_per_agent]

        self.nsq = self.nstates ** 2
        self.nso = self.nstates * self.nobs

        # Trigger Definitions
        self.TRIG_CENTRALIZED = range(0, self.nstates)
        self.TRIG_SEMI = COM_MODE
        self.TRIG_DECENTRAL = []

        # Set trigger based on TRIGGER_MODE
        if TRIGGER_MODE == "centralized":
            self.state_trigger = list(self.TRIG_CENTRALIZED)
        elif TRIGGER_MODE == "semi":
            self.state_trigger = self.TRIG_SEMI
        else:  # decentralized
            self.state_trigger = self.TRIG_DECENTRAL

class MarsProblemLoader:
    """Parses mars.data and structures matrices."""
    def __init__(self, config):
        self.c = config
        self.transit = [0.0] * (self.c.nsq * self.c.nacts)
        self.obs = [0.0] * (self.c.nso * self.c.nacts)
        self.reward = [0.0] * (self.c.nstates * self.c.nacts)
        self.init_beliefs = [0.0] * self.c.nstates
        
        # Standard Mars Start: Both agents at grid 0 (State 0)
        self.init_beliefs[0] = 1.0

    def load_data(self, filename="mars.data"):
        print(f"Loading {filename}...")
        # Use script directory for data file path
        filepath = os.path.join(_script_dir, filename)
        try:
            with open(filepath, "r") as data:
                for line in data:
                    d = line.split()
                    if not d: continue
                    
                    if d[0][0] == "T":
                        # T act1 act2 start_s end_s prob
                        act = int(d[1]) + self.c.act_per_agent * int(d[2])
                        s = int(d[4])
                        snew = int(d[6])
                        self.transit[act * self.c.nsq + s * self.c.nstates + snew] = float(d[8])
                        
                    elif d[0][0] == "O":
                        # O act1 act2 end_s obs1 obs2 prob
                        act = int(d[1]) + self.c.act_per_agent * int(d[2])
                        s = int(d[4])
                        o = int(d[6]) + self.c.obs_per_agent * int(d[7])
                        self.obs[act * self.c.nso + s * self.c.nobs + o] = float(d[9])
                        
                    elif d[0][0] == "R":
                        # R act1 act2 start_s reward
                        act = int(d[1]) + self.c.act_per_agent * int(d[2])
                        s = int(d[4])
                        self.reward[act * self.c.nstates + s] = float(d[10])
                        
            return self.transit, self.obs, self.reward, self.init_beliefs
        except FileNotFoundError:
            print(f"Error: {filename} not found. Please ensure it is in the same directory.")
            sys.exit(1)

# ==========================================
# Simulation Helper
# ==========================================

def step_environment(config, transit, obs_matrix, current_state, joint_action):
    """
    Simulates one step of the Mars environment using the matrices.
    Returns: (next_state, joint_observation)
    """
    # 1. Transition: Sample s' from T(s, a, s')
    start_idx = joint_action * config.nsq + current_state * config.nstates
    
    # Create CDF for sampling next state
    probs = []
    candidates = []
    cum_p = 0.0
    
    for s_next in range(config.nstates):
        p = transit[start_idx + s_next]
        if p > 0:
            cum_p += p
            probs.append(cum_p)
            candidates.append(s_next)
    
    if not candidates:
        next_state = current_state
    else:
        rand_val = random.random()
        idx = bisect.bisect_left(probs, rand_val)
        if idx < len(candidates):
            next_state = candidates[idx]
        else:
            next_state = candidates[-1]

    # 2. Observation: Sample o from O(a, s', o)
    start_idx_o = joint_action * config.nso + next_state * config.nobs
    
    probs_o = []
    candidates_o = []
    cum_p_o = 0.0
    
    for o in range(config.nobs):
        p = obs_matrix[start_idx_o + o]
        if p > 0:
            cum_p_o += p
            probs_o.append(cum_p_o)
            candidates_o.append(o)
            
    if not candidates_o:
        joint_obs = 0 
    else:
        rand_val = random.random()
        idx = bisect.bisect_left(probs_o, rand_val)
        if idx < len(candidates_o):
            joint_obs = candidates_o[idx]
        else:
            joint_obs = candidates_o[-1]
            
    return next_state, joint_obs

# ==========================================
# Execution Engine
# ==========================================

def run_mars_interleaved(config):
    start_time_total = time.time()
    
    # 1. Setup Problem
    loader = MarsProblemLoader(config)
    T, O, R, init_b = loader.load_data()

    # 2. Construct Trigger Mask
    trigger_mask = np.zeros((config.nacts, config.nstates), dtype=bool)
    
    for s in config.state_trigger:
        trigger_mask[:, s] = True  # Set entire column to True
    
    # 2. Initialize Model
    model = SDecPOMDPModel(
        nagents=config.nagents,
        nstates=config.nstates,
        nactions=config.nacts,
        nobs=config.nobs,
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=config.nacts_factor,
        nobs_factor=config.nobs_factor,
        sync_states=config.state_trigger,
        sync_actions=[],
        sync_observations=[]
    )

    # 3. Initialize Solver Config (uses top-level USER CONFIGURATION constants)
    solver_config = RSSDAConfig(
        maxh=config.horizon,
        maxit=config.maxit,
        IEmin2=config.ie_min2,
        alpha=config.alpha,
        algorithm=ALGORITHM,
        TI1=TI1,
        TI2=TI2,
        TI3=TI3,
        TI4=TI4,
        score_limit=SCORE_LIMIT,
        iter_limit=ITER_LIMIT,
        rec_limit=REC_LIMIT,
        cen_threshold=CEN_THRESHOLD,
        sm_temperature=SM_TEMPERATURE,
        heuristic_type=HEURISTIC_TYPE,
        tail_heuristic_type=TAIL_HEURISTIC_TYPE,
        hybrid_r=HYBRID_R,
        memory=MEMORY
    )

    # 4. Initialize Solver
    sdec_pomdp = SDecPOMDP(model=model, config=solver_config)

    print(f"Algorithm hyperparameters: algorithm: {sdec_pomdp.algorithm}, TI1: {sdec_pomdp.TI1}, TI2: {sdec_pomdp.TI2}, TI3: {sdec_pomdp.TI3}, TI4: {sdec_pomdp.TI4}, "
          f"iter_limit: {sdec_pomdp.iter_limit}, rec_limit: {sdec_pomdp.rec_limit}, "
          f"heuristic_type: {sdec_pomdp.heuristic_type}, tail_heuristic_type: {sdec_pomdp.tail_heuristic_type}, "
          f"maxit: {sdec_pomdp.maxit}, ie_min2: {sdec_pomdp.IEmin2}, memory: {sdec_pomdp.memory}, ")

    # 3. Initialize Simulation State
    true_state = 0
    print(f"Mars Initial State: {true_state}")
    
    current_horizon = config.horizon
    current_belief_idx = sdec_pomdp.dist_dict[int_tuple(sdec_pomdp.init_beliefs)]
    
    total_reward = 0
    step_global = 0
    termination_flag = False

    while not termination_flag:
        print(f"\n--- Planning Phase (Horizon: {current_horizon}, Start Belief: {current_belief_idx}) ---")
        
        # Reset solver structures that accumulate per run
        sdec_pomdp.cluster_dict.clear() 
        
        t0 = time.time()

        # RSSDA_state_approx returns: val, policy, clustering, cent_vector
        try:
            val, policy, clustering, cent_vector, cen_dists_map, clustering_cen = sdec_pomdp.multi_agent_astar(
                current_horizon, init_beliefs=current_belief_idx
            )
        except MemoryLimitExceeded as e:
            print(f"Result: MO")
            print(f"Memory limit exceeded: {e}")
            return "MO"

        plan_time = time.time() - t0
        print(f"Planning Time: {plan_time:.4f}s | Exp. Value: {val:.6f} | Policy: {policy}" )

        if val == -math.inf or policy is None:
            print("Planner failed (Time out or no solution). Check hyperparameters (e.g. iter_limit)")
            break

        # ---------------------------------------------------------
        # EXECUTION LOGIC: Strict Semi-Decentralized Interleaving
        # ---------------------------------------------------------
        
        steps_to_execute = len(policy)

        # Receding Horizon Logic (Only if Approximate):
        # Stop execution after encountering a synchronization point BEYOND the initial state.
        # Note: If we planned from a sync state, cent_vector[0] may be True, but that's the
        # state we just planned from, not a future replan point.
        if sdec_pomdp.algorithm == "approximate" and sdec_pomdp.TI1:
            try:
                # Look for sync points AFTER the first stage (skip initial sync if present)
                next_sync_idx = None
                for i in range(1, len(cent_vector)):
                    if cent_vector[i]:
                        next_sync_idx = i
                        break

                if next_sync_idx is not None:
                    limit = next_sync_idx + 1

                    if config.replan_at_all_syncs:
                        # Full replanning mode: Always replan at next sync
                        steps_to_execute = limit
                        print(f"RHC (Full): Stopping execution after Step {steps_to_execute} (Sync at stage {next_sync_idx+1}) to replan.")
                    else:
                        # Partial policy only mode: Only replan if policy is incomplete
                        if limit < steps_to_execute:
                            steps_to_execute = limit
                            print(f"RHC (Partial): Stopping execution after Step {steps_to_execute} (Sync at stage {next_sync_idx+1}) to replan.")
            except (ValueError, IndexError):
                pass # No syncs found, execute partial/full plan as is
            
        # Sanity check
        steps_to_execute = min(steps_to_execute, current_horizon)

        # Reset local history indices for execution of this segment
        # RSSDA assumes every plan starts at root, so history indices reset to 0
        current_oh = [0] * config.nagents

        # Execution Loop
        for step_local in range(steps_to_execute):
            step_global += 1

            # 1. Determine Centralization Status based on CURRENT BELIEF
            is_centralized_step = False
            c_ptr = -1

            if step_local < len(cen_dists_map):
                dists_at_step = cen_dists_map[step_local]
                if current_belief_idx in dists_at_step:
                    is_centralized_step = True
                    c_ptr = dists_at_step.index(current_belief_idx)

            # 2. Get Joint Action
            policy_mismatch = False
            try:
                if not is_centralized_step:
                    # DECENTRALIZED CASE
                    act1 = policy[step_local][0][0][current_oh[0]]
                    act2 = policy[step_local][0][1][current_oh[1]]
                    joint_act = act1 + (act2 * config.act_per_agent)
                else:
                    # CENTRALIZED CASE
                    # Use c_ptr to find specific action for this belief
                    joint_act = policy[step_local][1][c_ptr][0]
                    act1 = joint_act % config.act_per_agent
                    act2 = joint_act // config.act_per_agent

            except Exception as e:
                print(f"Error parsing policy at step {step_local}: {e}")
                policy_mismatch = True

            # Handle policy mismatch:
            if policy_mismatch:
                termination_flag = True
                break

            # 3. Simulation Step
            step_reward = R[joint_act * config.nstates + true_state]
            total_reward += step_reward

            next_state, joint_obs = step_environment(config, T, O, true_state, joint_act)

            o1 = joint_obs % config.obs_per_agent
            o2 = joint_obs // config.obs_per_agent

            # 4. Belief Update
            sparse_transitions = sdec_pomdp.get_terminal(current_belief_idx, joint_act)
            
            next_belief_idx = -1
            
            # Linear scan to find the matching observation
            for o, p, d in sparse_transitions:
                if o == joint_obs:
                    next_belief_idx = d
                    break
            
            if next_belief_idx != -1:
                current_belief_idx = next_belief_idx
            else:
                print(f"Warning: Belief update failed for obs {joint_obs}. Keeping belief constant.")

            sync_tag = "[SYNC]" if is_centralized_step else "[DEC]"
            print(f"  Step {step_global} {sync_tag}: S:{true_state} -> A:({act1},{act2}) -> R:{step_reward:.2f} -> S':{next_state} -> O:({o1},{o2})")

            # 5. Update Local History Indices (For next step)
            # Must happen unless we are at the very end of the execution block
            if step_local < steps_to_execute - 1:
                try:
                    if is_centralized_step:
                        # Use clustering_cen map for centralized observation history updates
                        if len(clustering_cen) > step_local and len(clustering_cen[step_local]) > 0:
                            # Check if observation is in the clustering map
                            try:
                                new_oh_0 = clustering_cen[step_local][0][c_ptr][o1]
                                new_oh_1 = clustering_cen[step_local][1][c_ptr][o2]
                                current_oh[0] = new_oh_0
                                current_oh[1] = new_oh_1
                            except (IndexError, KeyError):
                                print(f"  Warning: Obs ({o1},{o2}) not in clustering_cen[{step_local}][agent][{c_ptr}]. "
                                      f"Policy may be incomplete (TI1 early termination?). Resetting to root.")
                                current_oh = [0] * config.nagents
                        else:
                            # Next step is fully centralized root, reset or ignore
                            current_oh = [0] * config.nagents
                    else:
                        # Standard Decentralized Map
                        next_oh0 = clustering[step_local][0][current_oh[0]][o1]
                        next_oh1 = clustering[step_local][1][current_oh[1]][o2]

                        # Check if we are incorrectly flagging a valid Sync as a crash
                        if next_oh0 == -1 or next_oh1 == -1:
                            # Look ahead: Is the NEXT belief centralized?
                            # We check the map for step_local + 1
                            next_step_cen_dists = cen_dists_map[step_local + 1] if (step_local + 1) < len(cen_dists_map) else []

                            if current_belief_idx in next_step_cen_dists:
                                # VALID SYNC: The decentralized branch ended because we synchronized.
                                # We can safely set current_oh to dummy values because the
                                # next step will use the Centralized Policy (based on belief ID),
                                # not the Decentralized History.
                                current_oh = [0, 0] # Reset or dummy, doesn't matter for next Cent step
                            else:
                                print(f"  Warning: Encountered unclustered obs ({o1},{o2})...")
                                break
                        else:
                            current_oh[0] = next_oh0
                            current_oh[1] = next_oh1

                except (IndexError, TypeError) as e:
                    print(f"  Error updating history indices: {e}")
                    break

            true_state = next_state

            # Dynamic Sync Detection: Check if we landed in a sync trigger state during execution
            # If replan_at_all_syncs is True, break to replan from this sync state
            # Only break if we're not at the last step (otherwise we'd replan anyway)
            if config.replan_at_all_syncs and step_local < steps_to_execute - 1:
                if next_state in config.state_trigger:
                    print(f"  [Dynamic Sync Detected] Landed in sync state {next_state}. Breaking to replan from sync state.")
                    # Update steps_to_execute to reflect how many steps we actually executed
                    steps_to_execute = step_local + 1
                    break

        # Reduce remaining horizon
        current_horizon -= steps_to_execute
        
        if current_horizon <= 0:
            termination_flag = True
        else:
            sdec_pomdp.maxh = current_horizon

    print("\n=== Simulation Complete ===")
    print(f"Total Cumulative Reward: {total_reward}")
    print(f"Total Time: {time.time() - start_time_total:.4f}s")

# ==========================================
# Decentralized Mode (using original decPOMDP.py / RS-MAA*)
# ==========================================

def run_mars_decentralized_rsmaa(config, verbose=True):
    """
    Run the Mars problem using the original decPOMDP.py algorithm (RS-MAA*).
    This is the fully decentralized mode with no sync triggers.
    """
    if OriginalDecPOMDP is None:
        print("Original decPOMDP solver not available. Exiting.")
        return 0

    start_time = time.time()

    # 1. Load Problem Data
    loader = MarsProblemLoader(config)
    T, O, R, init_b = loader.load_data()

    # 2. Convert to pdict format required by decPOMDP.py
    # decPOMDP.py expects flat lists: transitions[act*nstates + s] = (indices, probs)
    T_pdict = []
    for act in range(config.nacts):
        for s in range(config.nstates):
            indices = []
            values = []
            for snew in range(config.nstates):
                val = T[act * config.nsq + s * config.nstates + snew]
                if val > 0:
                    indices.append(snew)
                    values.append(val)
            T_pdict.append((array('i', indices), array('d', values)))

    # obs[act*nstates + snew] = (obs_indices, probs)
    O_pdict = []
    for act in range(config.nacts):
        for snew in range(config.nstates):
            indices = []
            values = []
            for o in range(config.nobs):
                val = O[act * config.nso + snew * config.nobs + o]
                if val > 0:
                    indices.append(o)
                    values.append(val)
            O_pdict.append((array('i', indices), array('d', values)))

    if verbose:
        print(f"Mars Problem | States: {config.nstates} | Actions: {config.nacts} | Obs: {config.nobs}")

    # 3. Create original DecPOMDP solver (replicates exact optimal performance)
    dec_pomdp = OriginalDecPOMDP(
        nagents=config.nagents,
        nstates=config.nstates,
        nactions=config.nacts,
        nobs=config.nobs,
        transitions=T_pdict,
        obs=O_pdict,
        rewards=list(R),
        init_beliefs=list(init_b),
        nacts_factor=config.nacts_factor,
        nobs_factor=config.nobs_factor,
        maxh=config.horizon,
        cluster_type="lossless",
        maxit=config.maxit,
        q_depth=config.ie_min2,
        alpha=config.alpha,
        iter_limit="inf",
        maxrec="inf",
        memory=None,
        heuristic=None,
        rec_type=None,
        p_threshold_cluster=0,
        p_threshold_expand=0,
        policyvalfound=-math.inf,
        output=verbose
    )
    # Set required attributes not handled by constructor
    dec_pomdp.decentralized = False
    dec_pomdp.onesided = False

    print(f"\n--- Planning Phase (H: {config.horizon}) ---")

    # 4. Run the solver
    t0 = time.time()
    try:
        val, _, _ = dec_pomdp.multi_agent_astar(config.horizon)
        elapsed = time.time() - t0

        print(f"Planning Time: {elapsed:.4f}s | Exp. Value: {val:.4f}")
        print(f"Expected Value (decentralized RS-MAA*): {val:.5f}")
        print(f"Total Time: {time.time() - start_time:.4f}s")

        return val
    except DecPOMDPMemoryLimitExceeded as e:
        elapsed = time.time() - t0
        print(f"Result: MO")
        print(f"Memory limit exceeded: {e}")
        print(f"Planning Time: {elapsed:.4f}s")
        print(f"Total Time: {time.time() - start_time:.4f}s")
        return "MO"


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sdec_mars_approx.py <horizon> [maxit] [IEmin2] [alpha]")
        sys.exit(1)

    config = MarsConfig()

    if TRIGGER_MODE == "decentralized_RSMAA":
        print("Running in DECENTRALIZED mode (RS-MAA* via decPOMDP.py)")
        run_mars_decentralized_rsmaa(config)
    else:
        run_mars_interleaved(config)