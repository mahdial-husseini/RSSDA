"""
Maritime Medical Evacuation (MEDEVAC) Domain for Semi-Decentralized Planning

This module implements a maritime medical evacuation scenario where a helicopter
and a ship must coordinate to rescue a patient and transport them to a hospital.
The domain demonstrates semi-decentralized planning with asymmetric agent dynamics.

Domain Description:
    A helicopter and a ship operate on a 4x4 grid. They must:
    1. Navigate to the patient location
    2. Jointly pick up the patient (requires coordination)
    3. Transport to the hospital
    4. Jointly drop off the patient

    The helicopter moves faster (95% success) than the ship (85% success),
    creating situations where one agent arrives before the other.

State Space:
    - Helicopter position: 4x4 grid (16 positions)
    - Ship position: 4x4 grid (16 positions)
    - Carry flag: 0 (no patient) or 1 (carrying patient)
    - Total: 16 * 16 * 2 = 512 states

Actions (per agent):
    - WAIT: Stay in place
    - ADVANCE: Move toward current target (patient or hospital)
    - EXCHANGE: Attempt pickup (at patient) or dropoff (at hospital)

Observations (per agent):
    - at-target: Agent is at the current target location
    - not-at-target: Agent is elsewhere

Rewards:
    - Step cost: -0.3 per time step
    - Successful pickup: +5.0
    - Successful dropoff: +12.0
    - Failed solo exchange: -6.0
    - Wrong exchange location: -1.0

Synchronization Triggers:
    Semi-decentralized triggers are designed for states where coordination
    is most critical (e.g., when ship is delayed at patient location).

Usage:
    python maritimemedevac.py --table    # Run experiments across horizons/triggers

Reference:
    Based on Al-Husseini, M. "Semi-Decentralized Planning for Multi-Agent Systems"
    Technical Appendix A.3
    
Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

import sys
import os
import time
import pandas as pd
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

# Import original decPOMDP for fully decentralized mode
try:
    from decPOMDP import DecPOMDP as OriginalDecPOMDP, MemoryLimitExceeded as DecPOMDPMemoryLimitExceeded
except ImportError:
    print("Warning: Could not import original decPOMDP. Decentralized mode may not work.")
    OriginalDecPOMDP = None
    DecPOMDPMemoryLimitExceeded = None

# ============================================================================
#                           USER CONFIGURATION
# ============================================================================
# Modify the parameters below to control solver behavior.
# ============================================================================

# --- Experiment Settings ---
HORIZONS = (1, 2, 3, 4, 5, 6, 7)   # Planning horizons to evaluate

# --- Regimes to Run ---
# Toggle which planning regimes to include in the experiment.
RUN_DECENTRALIZED = True       # Fully decentralized (no synchronization); config provides more options for solver source
RUN_CENTRALIZED = True         # Fully centralized (always synchronized)
RUN_SEMI_DECENTRALIZED = True   # Semi-decentralized (state-triggered sync)

# --- Core Algorithm ---
ALGORITHM = "exact"             # "exact" or "approximate" (enables TI approximations)

# --- Heuristic Type ---
# Controls upper-bound heuristic for A* search guidance.
#   "QMDP"   - Loose/fast: ignores observations (dot product of belief and state values)
#   "POMDP"  - Tight/exact: full centralized POMDP value function
#   "HYBRID" - Runs exact POMDP for first HYBRID_R steps, then QMDP
# Rule of thumb: Use "POMDP" for exact algorithm, "QMDP" or "HYBRID" for approximate.
HEURISTIC_TYPE = "POMDP"
HYBRID_R = 2                    # Steps of exact POMDP before switching to QMDP (HYBRID mode only)

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
TI4 = False  # Max Clustering: cluster based on L1 distance between beliefs, weighted by probability mass

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
REC_LIMIT = 1
TAIL_HEURISTIC_TYPE = "POMDP"

# --- TI4: Max Clustering ---
# Cluster into MAX_Clusters based on combination of L1 distance between resulting beliefs and probability mass
MAX_CLUSTERS = 2

# ============================================================================
#                        END USER CONFIGURATION
# ============================================================================

G = 4
nagents = 2
a_per_agent = 3
o_per_agent = 2
nstates = (G * G) * (G * G) * 2
nactions = a_per_agent ** nagents
nobs = o_per_agent ** nagents

patient = (1,1)
hospital = (3,3)
helo_start = (0,1)
ship_start  = (1,0)

p_move_helo = 0.95
p_move_ship  = 0.85
p_pickup  = 0.95
p_dropoff = 0.95

step_cost = -0.3
wrong_exchange_cost = -1.0
pickup_reward  = 5.0
dropoff_reward = 12.0
pickup_mismatch_penalty  = -6.0
dropoff_mismatch_penalty = -6.0

def pos_to_idx(x, y): return y * G + x
def id_to_pos(idx): return (idx % G, idx // G)
def state_id(px, py, bx, by, carry):
    return carry * (G * G * G * G) + pos_to_idx(px, py) * (G * G) + pos_to_idx(bx, by)
def id_to_state(s):
    carry = 1 if s >= (G * G * G * G) else 0
    rem = s - carry * (G * G * G * G)
    ppos = rem // (G * G); bpos = rem % (G * G)
    px,py = id_to_pos(ppos); bx, by = id_to_pos(bpos)
    return (px, py, bx, by, carry)
def in_grid(x, y): return 0 <= x < G and 0 <= y < G

def next_step_toward(x, y, tx, ty, prefer='H'):
    if (x, y)==(tx, ty): return (x, y)
    if prefer=='H':
        if x < tx: return (x + 1, y)
        if y < ty: return (x, y + 1)
    else:
        if y < ty: return (x, y + 1)
        if x < tx: return (x + 1, y)
    return (x, y)

def advance_helo(x, y, carry):
    tx, ty = (patient if carry == 0 else hospital)
    nx, ny = next_step_toward(x, y, tx, ty, 'H')
    p_succ = p_move_helo if (nx, ny)!=(x, y) else 1.0
    return (nx, ny, p_succ)
def advance_ship(x, y, carry):
    tx, ty = (patient if carry == 0 else hospital)
    nx, ny = next_step_toward(x, y, tx, ty, 'V')
    p_succ = p_move_ship if (nx, ny)!=(x, y) else 1.0
    return (nx,ny, p_succ)

def at_patient(x, y): return (x, y) == patient
def at_hospital(x, y): return (x, y) == hospital
def at_target_for_agent(x, y, carry):
    t = patient if carry == 0 else hospital
    return 1 if (x, y) == t else 0

def build_problem():
    nsq = nstates * nstates
    nso = nstates * nobs
    transit = [0.0] * (nsq * nactions)
    obs = [0.0] * (nso * nactions)
    reward = [0.0] * (nstates * nactions)

    for a0 in range(a_per_agent):
        for a1 in range(a_per_agent):
            act = a0 + a_per_agent*a1
            for s in range(nstates):
                px, py, bx, by, carry = id_to_state(s)

                r = step_cost
                helo_at_pat  = at_patient(px, py)
                boat_at_pat   = at_patient(bx, by)
                helo_at_hosp = at_hospital(px, py)
                boat_at_hosp  = at_hospital(bx, by)

                if a0 == 2 and not (helo_at_pat or helo_at_hosp): r += wrong_exchange_cost
                if a1 == 2 and not (boat_at_pat  or boat_at_hosp ): r += wrong_exchange_cost

                if carry == 0:
                    if helo_at_pat and boat_at_pat and a0 == 2 and a1 == 2:
                        r += p_pickup * pickup_reward
                    if (a0 == 2 and helo_at_pat and not (a1 == 2 and boat_at_pat)) or \
                       (a1 == 2 and boat_at_pat  and not (a0 == 2 and helo_at_pat)):
                        r += pickup_mismatch_penalty
                else:
                    if helo_at_hosp and boat_at_hosp and a0 == 2 and a1 == 2:
                        r += p_dropoff * dropoff_reward
                    if (a0 == 2 and helo_at_hosp and not (a1 == 2 and boat_at_hosp)) or \
                       (a1 == 2 and boat_at_hosp  and not (a0 == 2 and helo_at_hosp)):
                        r += dropoff_mismatch_penalty

                reward[act * nstates + s] = r

                if a0 == 1:
                    nx_p, ny_p, p_succ_p = advance_helo(px, py, carry)
                else:
                    nx_p, ny_p, p_succ_p = px, py, 1.0
                if a1 == 1:
                    nx_b, ny_b, p_succ_b = advance_ship(bx, by, carry)
                else:
                    nx_b, ny_b, p_succ_b = bx, by, 1.0

                cases = [
                    (nx_p, ny_p, nx_b, ny_b, p_succ_p * p_succ_b),
                    (nx_p, ny_p, bx, by, p_succ_p * (1.0 - p_succ_b)),
                    (px, py, nx_b, ny_b, (1.0 - p_succ_p) * p_succ_b),
                    (px, py, bx, by, (1.0 - p_succ_p) * (1.0 - p_succ_b)),
                ]
                if a0 == 2:
                    cases = [(px, py, xb, yb, p if (xp == px and yp == py) else 0.0) for (xp, yp, xb, yb, p) in cases]
                if a1 == 2:
                    cases = [(xp, yp, bx, by, p if (xb == bx and yb == by) else 0.0) for (xp, yp, xb, yb, p) in cases]

                for (px2, py2, bx2, by2, p_case) in cases:
                    if p_case == 0.0: continue
                    if carry == 0 and helo_at_pat and boat_at_pat and a0 == 2 and a1 == 2:
                        s_succ = state_id(px2, py2, bx2, by2, 1)
                        s_fail = state_id(px2, py2, bx2, by2, 0)
                        transit[act * nstates * nstates + s * nstates + s_succ] += p_case * p_pickup
                        transit[act * nstates * nstates + s * nstates + s_fail] += p_case * (1.0 - p_pickup)
                    elif carry == 1 and helo_at_hosp and boat_at_hosp and a0 == 2 and a1 == 2:
                        s_succ = state_id(px2, py2, bx2, by2, 0)
                        s_fail = state_id(px2, py2, bx2, by2, 1)
                        transit[act * nstates * nstates + s * nstates + s_succ] += p_case * p_dropoff
                        transit[act * nstates * nstates + s * nstates + s_fail] += p_case * (1.0 - p_dropoff)
                    else:
                        s2 = state_id(px2, py2, bx2, by2, carry)
                        transit[act * nstates * nstates + s * nstates + s2] += p_case

            for s2 in range(nstates):
                px2, py2, bx2, by2, carry2 = id_to_state(s2)
                o0 = at_target_for_agent(px2, py2, carry2)
                o1 = at_target_for_agent(bx2, by2, carry2)
                o = o0 + o_per_agent*o1
                obs[act * (nstates * nobs) + s2 * nobs + o] = 1.0

    init_beliefs = [0.0]*nstates
    init_beliefs[state_id(helo_start[0], helo_start[1], ship_start[0], ship_start[1], 0)] = 1.0
    return transit, obs, reward, init_beliefs

def triggers_none(): return []
def triggers_semi():
    S = set()
    S.add(state_id(1, 1, 1, 0, 0))   # patient-ship-late
    S.add(state_id(3, 2, 3, 3, 1))   # hospital XOR: non-arrived north (helo north, ship at hosp)
    S.add(state_id(3, 3, 3, 2, 1))   # hospital XOR: non-arrived north (ship north, helo at hosp)
    return sorted(S)
def triggers_full(): return list(range(nstates))

def to_sparse_format(transit_dense, obs_dense, nstates, nactions, nobs):
    """Convert dense transition and observation matrices to sparse format for SDecPOMDP."""
    nsq = nstates * nstates
    nso = nstates * nobs

    # Convert transitions: transit_sparse[act*nstates + s] = (indices, probs)
    transit_sparse = []
    for act in range(nactions):
        for s in range(nstates):
            indices = array("i", [])
            probs = array("d", [])
            for snew in range(nstates):
                p = transit_dense[act * nsq + s * nstates + snew]
                if p > 0:
                    indices.append(snew)
                    probs.append(p)
            transit_sparse.append((indices, probs))

    # Convert observations: obs_sparse[act*nstates + s] = (indices, probs)
    obs_sparse = []
    for act in range(nactions):
        for s in range(nstates):
            indices = array("i", [])
            probs = array("d", [])
            for o in range(nobs):
                p = obs_dense[act * nso + s * nobs + o]
                if p > 0:
                    indices.append(o)
                    probs.append(p)
            obs_sparse.append((indices, probs))

    return transit_sparse, obs_sparse

# Uses USER CONFIGURATION constants for defaults
def run(horizons=HORIZONS, maxit=MAXIT, alpha=ALPHA):
    transit, obs, reward, init_beliefs = build_problem()
    # Convert to sparse format for SDecPOMDP (approximate solver)
    transit_sparse, obs_sparse = to_sparse_format(transit, obs, nstates, nactions, nobs)

    # Build regimes list based on USER CONFIGURATION flags
    regimes = []
    if RUN_DECENTRALIZED:
        regimes.append(("Decentralized (RSMAA)", triggers_none()))
    if RUN_CENTRALIZED:
        regimes.append(("Centralized", triggers_full()))
    if RUN_SEMI_DECENTRALIZED:
        regimes.append(("Semi-Decentralized", triggers_semi()))
    rows = []
    for label, state_trigger in regimes:
        for H in horizons:
            # 1. Start Timer
            t0 = time.time()

            if label == "Decentralized (App RSMAA)": # (replicates exact optimal performance)
                dec = OriginalDecPOMDP(
                    nagents=2,
                    nstates=nstates,
                    nactions=nactions,
                    nobs=nobs,
                    transitions=transit_sparse,
                    obs=obs_sparse,
                    rewards=reward,
                    init_beliefs=init_beliefs,
                    nacts_factor=[a_per_agent] * 2,
                    nobs_factor=[o_per_agent] * 2,
                    maxh=H,
                    cluster_type="lossless",
                    maxit=maxit,
                    q_depth=IE_MIN2,
                    alpha=alpha,
                    iter_limit="inf",
                    maxrec="inf",
                    memory=None,
                    heuristic=None,
                    rec_type=None,
                    p_threshold_cluster=0,
                    p_threshold_expand=0,
                    policyvalfound=None,
                    output=False
                )
                try:
                    val, _, _ = dec.multi_agent_astar(H)
                except DecPOMDPMemoryLimitExceeded as e:
                    print(f"Memory limit exceeded for {label} H={H}: {e}")
                    val = "MO"
            else:
                # Initialize Model
                model = SDecPOMDPModel(
                    nagents=2,
                    nstates=nstates,
                    nactions=nactions,
                    nobs=nobs,
                    transitions=transit,
                    obs=obs,
                    rewards=reward,
                    init_beliefs=init_beliefs,
                    nacts_factor=[a_per_agent] * 2,
                    nobs_factor=[o_per_agent] * 2,
                    sync_states=state_trigger,
                    sync_actions=[],
                    sync_observations=[]
                )

                # Initialize Solver Config (uses top-level USER CONFIGURATION constants)
                solver_config = RSSDAConfig(
                    maxh=H,
                    maxit=maxit,
                    IEmin2=IE_MIN2,
                    alpha=alpha,
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
                    max_clusters=MAX_CLUSTERS
                )

                # Initialize Solver
                dec = SDecPOMDP(model=model, config=solver_config)

                try:
                    val,_,_, _, _, _ = dec.multi_agent_astar(H)
                except MemoryLimitExceeded as e:
                    print(f"Memory limit exceeded for {label} H={H}: {e}")
                    val = "MO"

            # 2. Stop Timer
            dt = time.time() - t0

            # Optional: Suppress output if supported by your solver classes
            if hasattr(dec, 'output'):
                dec.output = False

            rows.append((H, label, val if val == "MO" else float(val), float(dt)))

    df = pd.DataFrame(rows, columns=["H", "Regime", "Optimal Value", "Solve Time (s)"])
    return df

if __name__ == "__main__":
    df = run()
    
    # Create a multi-level pivot table to show Value and Time side-by-side
    pivot_df = df.pivot(index="H", columns="Regime", values=["Optimal Value", "Solve Time (s)"])
        
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(pivot_df.to_string())