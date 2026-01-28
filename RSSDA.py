"""
Exact/Approximate Recursive Small-Step Semi-Decentralized A* (RS-SDA*)
Communication may be synchronized conditioned on states, joint actions, or joint observations. May be further extended to beliefs. 

This module implements the RsSSDA* algorithm (exact and approximate) for solving semi-decentralized POMDPs (SDec-POMDPs)
with state/joint action/joint observation-based communication triggers. The algorithm handles systems where agents 
operate decentralized by default but synchronize intermittently. The current version is designed for two-agent systems, but can be extended.

Algorithm Overview:
    RS-SDA* performs A* search over the space of partial policies, expanding one agent's
    action assignment at a time. The key innovation is tracking two parallel probability
    flows: decentralized (agents act independently) and centralized (agents coordinate).
    State-based triggers partition the belief at each stage based on which states require
    synchronization.

Key Components:
    - Policy: Represents partial/complete policies with decentralized and centralized
      action mappings, observation history clustering, and belief distributions.
    - SDecPOMDPModel: Encapsulates the SDec-POMDP model (transitions, observations, rewards).
    - SDecPOMDP: Main solver class implementing the A* search with heuristic computation.

Optional Approximation Techniques (TI1-TI4):
    - TI1: Early termination via weighted majority voting on centralization decisions; effective for stochastic observations and in centralized settings
    - TI2: Progress-based pruning to limit search depth per policy
    - TI3: Recursive horizon limiting with tail approximation [QMDP/HYBRID/POMDP]
    - TI4: Finite-memory clustering (window-based observation histories); effective for stochastic observations/complex belief dynamics

Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

from dataclasses import dataclass, field
from heapq import heappush, heappop, nsmallest, heapify
from itertools import count
import math
import sys
import numpy as np
from numba import jit
from typing import List, Tuple, Dict, Optional, Union, Any
import psutil

# ==========================================
# Type Definitions & Constants
# ==========================================

# Primitives matching paper notation
BeliefID = int      # Index into self.dists
StateID = int       # Index into nstates
ActionID = int      # Single agent action index
JointActionID = int # Joint action index
ObsID = int         # Observation index
Prob = float        # Probability [0, 1]

# Policy Structures
# Structure: [Stage][Decentralized=0/Centralized=1]
# Dec: [AgentIdx][HistoryIdx] -> ActionID
# Cen: [ClusterIdx] -> [JointActionID]
DecentralizedPol = List[List[int]] 
CentralizedPol = List[List[int]]
StagePolicy = List[Union[List[DecentralizedPol], CentralizedPol]]
FullPolicy = List[StagePolicy]

# Constants
EPSILON = 1e-12    # Probability threshold for numerical stability (near-zero checks)
TOLERANCE = 1e-8   # Value comparison tolerance for A* search pruning
HASH_FACTOR = 10**12 + 39  # Factor for hashing probability distributions

class MemoryLimitExceeded(Exception):
    """Raised when memory usage exceeds the configured limit."""
    pass

def get_memory_usage_gb() -> float:
    """Returns current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def fast_dynamics_sparse(belief, T_csr_list, O_csr_list, nactions, nstates, nobs):
    """
    Optimized sparse dynamics using flatnonzero to skip zero-probability states.
    """
    p_obs_all = np.zeros((nactions, nobs), dtype=np.float64)
    joint_unnorm_all = np.zeros((nactions, nobs, nstates), dtype=np.float64)

    # 1. Compute Next State Distribution
    for a in range(nactions):
        # belief (dense) @ T (sparse) -> next_states (dense)
        next_states = belief @ T_csr_list[a]

        # 2. OPTIMIZATION: Get indices of non-zero states immediately
        active_states = np.flatnonzero(next_states > EPSILON)

        if active_states.size == 0:
            continue

        # 3. Iterate only active states
        O_a = O_csr_list[a]

        for s_prime in active_states:
            val_ns = next_states[s_prime]

            start = O_a.indptr[s_prime]
            end = O_a.indptr[s_prime + 1]

            if start == end:
                continue

            for idx in range(start, end):
                o = O_a.indices[idx]
                prob = val_ns * O_a.data[idx]
                joint_unnorm_all[a, o, s_prime] = prob
                p_obs_all[a, o] += prob

    return p_obs_all, joint_unnorm_all

@jit(nopython=True, cache=True)
def fast_dynamics(belief, T, O, nactions, nstates, nobs):
    """
    Optimized dense dynamic calculations.
    """
    # PRE-ALLOCATE OUTPUTS
    next_states_all = np.zeros((nactions, nstates), dtype=np.float64)
    p_obs_all = np.zeros((nactions, nobs), dtype=np.float64)
    joint_unnorm_all = np.zeros((nactions, nobs, nstates), dtype=np.float64)

    # 1. COMPUTE NEXT STATE DISTRIBUTION
    for a in range(nactions):
        for s in range(nstates):
            val = belief[s]
            if val > 1e-12:  # Numba JIT requires literal constant
                for s_prime in range(nstates):
                    next_states_all[a, s_prime] += val * T[a, s, s_prime]

    # 2. COMPUTE OBSERVATION PROBABILITIES & JOINT BELIEFS
    for a in range(nactions):
        for s_prime in range(nstates):
            val_ns = next_states_all[a, s_prime]
            if val_ns > 0:
                for o in range(nobs):
                    prob = val_ns * O[a, s_prime, o]
                    joint_unnorm_all[a, o, s_prime] = prob
                    p_obs_all[a, o] += prob

    return p_obs_all, joint_unnorm_all

def int_tuple(plist: Union[List[float], np.ndarray, None]) -> Tuple[int, ...]:
    if plist is None:
        return tuple()
    # Create a sparse tuple representation (index, value) ignoring zeros
    return tuple([x * HASH_FACTOR + int(plist[x] * HASH_FACTOR) for x in range(len(plist)) if plist[x] > 0])

def cumprod(lens):     # cumprod takes an array [l_0, l_1, , l_{n-1}] and returns the array of cumulative products
    ll = len(lens)     # [1, l_0, l_0l_1, , l_0l_1l_{n-2}], as well as the full product l_0l_1l_{n-1} separately.
    div = [1]*ll
    for idx in range(ll-1):
        div[idx+1] = div[idx] * lens[idx]
    return(div, div[ll-1]*lens[ll-1])

def product(list1, list2, mult):                     # given lists L1, L2, computes the list of indices x + My 
    return [x+mult*y for y in list2 for x in list1]  # corresponding to pairs (x, y), where x in L1, y in L2.
    
def lists_product(lists, mults, nlists):             # given a list L of lists L_0, , L_{n-1}, computes the list of indices 
    list1 = lists[0]                                 # x_0 + M_1 x_1 +  M_{n-1} x_{n-1} corresponding to tuples (x_0, x_1, , x_{n-1})
    for idx in range(1, nlists):                     # where x_0 in L_0, x_1 in L_1, ., L_{n-1}. 
        if lists[idx] != [-1]: 
            m = mults[idx]
            l_curr = lists[idx]
            list1 = [x + m*y for y in l_curr for x in list1]
            
    return list1
    
def lists_product2(list1idx, list1, rlens, mults, nlists):  # Computes a list_product where only one list is not a range.
    return lists_product([list1 if idx == list1idx else range(rlens[idx]) for idx in range(nlists)], mults, nlists)

class Policy:
    
    def __init__(self, policy: FullPolicy, ncluster: List[List[int]], dists: List[List[BeliefID]], prob: List[List[Prob]], 
                 clustering: List[Tuple], values: List[float] = [], heuristics: List[float] = [math.inf], depth: Optional[int] = None, 
                 final_cidx_value: float = 0.0, dists_cen: List[List[BeliefID]] = [[-1]], prob_cen: List[List[Prob]] = [[1.0]], 
                 dec_split: List[float] = [1.0], clustering_cen: List[Tuple] = [], step_cen_values: List[float] = [], suffixes: List[Any] = []):
        
        self.policy = policy         
        self.ncluster = ncluster
        self.dists = dists
        self.prob = prob
        self.clustering = clustering
        self.clustering_cen = clustering_cen
        self.values = values
        self.heuristics = heuristics
        self.depth = depth
        self.dists_cen = dists_cen
        self.prob_cen = prob_cen
        self.dec_split = dec_split
        self.final_cidx_value = final_cidx_value
        self.step_cen_values = step_cen_values 
        self.suffixes = suffixes

    def policy_copy(self, idx: int, aidx: int) -> 'Policy': 
        policy = self.policy.copy()                         
        policy[idx] = policy[idx].copy()                     
        policy[idx][0] = policy[idx][0].copy()               
        policy[idx][0][aidx] = policy[idx][0][aidx].copy()  
        heuristics = self.heuristics.copy()
        return Policy(policy, self.ncluster, self.dists, self.prob, self.clustering, self.values, heuristics, self.depth, self.final_cidx_value, self.dists_cen, self.prob_cen, self.dec_split, self.clustering_cen, self.step_cen_values, self.suffixes)
    
    def policy_copy_cent(self, idx: int, jaidx: int) -> 'Policy': 
        policy = self.policy.copy()                                
        policy[idx] = policy[idx].copy()                           
        policy[idx][1] = policy[idx][1].copy()                               
        policy[idx][1][jaidx] = policy[idx][1][jaidx].copy()             
        heuristics = self.heuristics.copy()
        return Policy(policy, self.ncluster, self.dists, self.prob, self.clustering, self.values, heuristics, self.depth, self.final_cidx_value, self.dists_cen, self.prob_cen, self.dec_split, self.clustering_cen, self.step_cen_values, self.suffixes)
        
    def policy_copy_laststage(self, idx: int, aidx: int) -> 'Policy':  
        policy = self.policy.copy()                                    
        policy[idx] = policy[idx].copy()                      
        policy[idx][0] = policy[idx][0].copy()               
        policy[idx][0][aidx] = policy[idx][0][aidx].copy()                     
        return Policy(policy, self.ncluster, self.dists, self.prob, self.clustering, self.values, [], self.depth, self.final_cidx_value, self.dists_cen, self.prob_cen, self.dec_split, self.clustering_cen, self.step_cen_values, self.suffixes)
    
    def policy_copy_laststage_cent(self, idx: int) -> 'Policy':  
        policy = self.policy.copy()                          
        policy[idx] = policy[idx].copy()                     
        policy[idx][1] = policy[idx][1].copy()               
        heuristics = self.heuristics.copy()
        return Policy(policy, self.ncluster, self.dists, self.prob, self.clustering, self.values, heuristics, self.depth, self.final_cidx_value, self.dists_cen, self.prob_cen, self.dec_split, self.clustering_cen, self.step_cen_values, self.suffixes)
    
    def cluster_copy(self) -> 'Policy':  
        return Policy(self.policy.copy(), self.ncluster.copy(), self.dists.copy(), self.prob.copy(),
                      self.clustering.copy(), self.values.copy(), self.heuristics, self.depth, self.final_cidx_value, self.dists_cen.copy(), self.prob_cen.copy(), self.dec_split.copy(), self.clustering_cen.copy(), self.step_cen_values.copy(), self.suffixes.copy())

@dataclass
class RSSDAConfig:
    """
    Default configuration hyperparameters for the RSSDA solver. May be overwritten by benchmark/drivers.
    """
    # Search Horizon
    maxh: int
    
    # Heuristic & Search Control
    IEmin2: int = 3                 # Depth (d) of information-sharing stages for decentralized heuristic computation
    maxit: int = 200                # [early heuristic terminal technique] Max iterations per stage expansion
    alpha: float = 0.2              # [early heuristic terminal technique] Threshhold for dynamically abandoning heuristics early
    heuristic_type: str = "HYBRID"  # "QMDP", "POMDP", or "HYBRID"
    
    # Approximation Flags (The "TI" Tiers)
    algorithm: str = "exact"
    TI1: bool = False    # Interleaving Planning/Execution
    TI2: bool = False    # Progress-based Pruning
    TI3: bool = False    # Tail Value Approximation
    TI4: bool = False    # Memory-Bounded Clustering
    
    # TI1 Settings (Interleaving)
    score_limit: int = 20
    cen_threshold: float = 0.6
    sm_temperature: float = 0.6
    adaptive_check: int = 10
    
    # TI2 Settings (Pruning)
    # iter_limit: Total iteration budget. Per-entity budget B = iter_limit / (nagents + 1).
    # IMPORTANT: Set iter_limit such that B >= max expected clusters per entity per stage.
    # If an entity has more clusters than B, some may not be explored before pruning.
    iter_limit: int = 1000
    
    # TI3 Settings (Tail Approximation)
    rec_limit: int = 2
    tail_heuristic_type: Optional[str] = None  # Defaults to heuristic_type if None
    hybrid_r: int = 0
    
    # TI4 Settings (Clustering)
    memory: int = 2

    # Resource Limits
    memory_limit_gb: Optional[float] = 16.0  # Memory limit in GB; None = no limit
    memory_check_interval: int = 100          # Check memory every N iterations

    # Misc
    output: bool = False
    
    def __post_init__(self):
        if self.tail_heuristic_type is None:
            self.tail_heuristic_type = self.heuristic_type
        if self.memory is not None and self.memory < 1:
            raise ValueError(f"TI4 memory must be >= 1, got {self.memory}")

@dataclass
class TriggerProfile:
    """
    Encapsulates the Generalized Trigger Function Phi(C).

    Attributes:
        sync_actions: Set of joint action indices that trigger synchronization
        sync_observations: Set of joint observation indices that trigger synchronization
        state_mask: Boolean mask where True indicates a sync state
    """
    sync_actions: set = field(default_factory=set)
    sync_observations: set = field(default_factory=set)
    state_mask: Optional[np.ndarray] = None

class SDecPOMDPModel:
    """
    Encapsulates the static definition of the Dec-POMDP problem.
    Handles loading from raw inputs or cached sparse/dense structures.
    """
    def __init__(self, nagents, nstates, nactions, nobs, 
                 transitions=None, obs=None, rewards=None, init_beliefs=None,
                 nacts_factor=None, nobs_factor=None, 
                 cached_data=None, sync_states=None, 
                 sync_actions=None, sync_observations=None):
        
        self.nagents = nagents
        self.nstates = nstates
        self.nactions = nactions
        self.nobs = nobs
        self.nacts_factor = nacts_factor
        self.nobs_factor = nobs_factor
        self.init_beliefs = init_beliefs
        
        # Sync trigger logic (Model property)
        self.sync_states = sync_states if sync_states is not None else []
        self.sink_state = nstates - 1
        state_mask = np.zeros(nstates, dtype=bool)
        if self.sync_states:
            state_mask[self.sync_states] = True
        state_mask[self.sink_state] = True

        # 2. Create Unified Profile
        self.trigger_profile = TriggerProfile(
            sync_actions=set(sync_actions) if sync_actions else set(),
            sync_observations=set(sync_observations) if sync_observations else set(),
            state_mask=state_mask
        )

        # --- Data Loading Logic ---
        self.use_sparse = False
        self.T = None
        self.O = None
        self.T_csr_list = None
        self.O_csr_list = None
        self.RA = None

        if cached_data is not None:
            self._load_from_cache(cached_data)
        else:
            self._load_from_raw(transitions, obs, rewards)

    def _load_from_cache(self, cached_data: Dict[str, Any]) -> None:
        # Check for sparse matrices (v5 cache format)
        if 'T_csr_list' in cached_data and cached_data.get('sparse', False):
            self.T_csr_list = cached_data['T_csr_list']
            self.O_csr_list = cached_data['O_csr_list']
            self.use_sparse = True
        elif 'T_np' in cached_data:
            self.T = cached_data['T_np']
            self.O = cached_data['O_np']
            self.use_sparse = False
        else:
            raise ValueError("Cache data missing both sparse and dense arrays")
        
        self.RA = cached_data['R_np'].astype(np.float64)

    def _load_from_raw(self, transitions: List, obs: Union[List, Dict], rewards: List) -> None:
        # Normalize obs to sparse dict format
        if not isinstance(obs, dict):
            obs = {i: v for i, v in enumerate(obs) if v > 0}
        
        # Build dense arrays (standard initialization)
        self.T = np.array(transitions, dtype=np.float64).reshape(self.nactions, self.nstates, self.nstates)
        
        obs_size = self.nactions * self.nstates * self.nobs
        obs_dense = [obs.get(i, 0.0) for i in range(obs_size)]
        self.O = np.array(obs_dense, dtype=np.float64).reshape(self.nactions, self.nstates, self.nobs)
        
        self.RA = np.array(rewards, dtype=np.float64).reshape(self.nactions, self.nstates)
        self.use_sparse = False

class SDecPOMDP:
    def __init__(self, model: SDecPOMDPModel, config: RSSDAConfig, qmdp_data=None):
        
        # --- 1. Model Adoption ---
        self.model = model
        self.nagents = model.nagents
        self.nstates = model.nstates
        self.nactions = model.nactions
        self.nobs = model.nobs
        self.nacts_factor = model.nacts_factor
        self.nobs_factor = model.nobs_factor
        self.init_beliefs = model.init_beliefs
        self.sync_states = model.sync_states
        self.sink_state = model.sink_state
        self.trigger_profile = model.trigger_profile

        # Performance shortcuts
        self.T = model.T
        self.O = model.O
        self.T_csr_list = model.T_csr_list
        self.O_csr_list = model.O_csr_list
        self.RA = model.RA
        self.use_sparse = model.use_sparse

        # --- 2. Solver Configuration ---
        self.config = config
        self.maxh = config.maxh
        self.maxit = config.maxit
        self.IEmin2 = config.IEmin2
        self.alpha = config.alpha
        self.output = config.output
        
        # Approximation Settings
        self.algorithm = config.algorithm
        self.TI1 = config.TI1
        self.TI2 = config.TI2
        self.TI3 = config.TI3
        self.TI4 = config.TI4
        self.score_limit = config.score_limit
        self.iter_limit = config.iter_limit
        self.rec_limit = config.rec_limit
        self.cen_threshold = config.cen_threshold
        self.sm_temperature = config.sm_temperature
        self.memory = config.memory
        self.adaptive_check = config.adaptive_check
        self.hybrid_r = config.hybrid_r
        self.heuristic_type = config.heuristic_type
        self.tail_heuristic_type = config.tail_heuristic_type

        # Resource limits
        self.memory_limit_gb = config.memory_limit_gb
        self.memory_check_interval = config.memory_check_interval
        self._last_reported_mem_gb = 0  # Track last reported memory level for 1GB increment logging

        self.init_call = True

        # --- 3. Pre-computation (Counters & Factors) ---
        self.nsq = self.nstates ** 2
        self.nso = self.nstates * self.nobs
        self.maxa = max(self.nacts_factor)
        
        self.a_prod = [1]*self.nagents
        self.o_prod = [1]*self.nagents
        for idx in range(self.nagents-1):
            self.a_prod[idx+1] = self.a_prod[idx] * self.nacts_factor[idx]
            self.o_prod[idx+1] = self.o_prod[idx] * self.nobs_factor[idx]

        # Build ctrs mapping
        self.ctrs = {1: [1]}
        for a in range(self.nagents):
            for ctr_fix in list(self.ctrs.keys()):
                self.ctrs[ctr_fix] = [c * self.maxa + acta
                                      for c in self.ctrs[ctr_fix]
                                      for acta in range(self.nacts_factor[a])]
            for ctr_fix in self.ctrs[1]:
                self.ctrs[ctr_fix] = [ctr_fix]

        # Build counter-to-ja lookup
        self.ctr_to_ja = {}
        for full_ctr in self.ctrs[1]:
            actions = []
            temp = full_ctr
            while temp > 1:
                actions.append(temp % self.maxa)
                temp //= self.maxa
            actions.reverse()
            ja = sum(actions[i] * self.a_prod[i] for i in range(self.nagents))
            self.ctr_to_ja[full_ctr] = ja

        # --- 4. Caches & Heuristics ---
        self.dec_heuristic = dict()
        self.cen_heuristic = dict()
        self.newstatedist_dict = dict()
        self.terminal_dict = dict()
        self.cluster_dict = dict()
        self.belief_split_cache = {}
        self.cen_V = {}
        self.cen_V_hybrid = {}
        self.cen_Q = {}
        self.qmdp_cache = {}
        self.clusterctr_dict = {}
        self.terminalMDP_dict = {}
        self._terminal_batched = set()
        self._os_by_oa_cache = {}

        # Initial Belief Setup
        self.dist_dict = {int_tuple(self.init_beliefs): 0}
        self.dists = [self.init_beliefs]
        self.dists_sparse = {}
        self.reward_list = (self.RA @ self.init_beliefs).tolist()

        # Action Masks
        self.valid_actions_per_state = model.valid_actions_per_state if hasattr(model, 'valid_actions_per_state') else None
        self.use_action_masks = self.valid_actions_per_state is not None
        self.valid_actions_per_position = model.valid_actions_per_position if hasattr(model, 'valid_actions_per_position') else None
        self.use_position_action_masks = self.valid_actions_per_position is not None
        self._valid_actions_cache = {}

        # --- 5. Dynamics Dispatch Setup ---
        if self.use_sparse:
            self.T_repr = self.T_csr_list
            self.O_repr = self.O_csr_list
            self.dynamics_fn = fast_dynamics_sparse
        else:
            self.T_repr = self.T
            self.O_repr = self.O
            self.dynamics_fn = fast_dynamics

        # --- 6. QMDP Initialization ---
        if qmdp_data is not None:
            self.qmdp_Q = qmdp_data['qmdp_Q'][:self.maxh + 1]
        else:
            self.qmdp_Q = None
            self._solve_qmdp()

    # === ACTION MASK OPTIMIZATION METHODS ===
    def get_valid_actions_for_belief(self, dist_id: BeliefID) -> Union[range, List[JointActionID]]:
        """
        Returns list of valid joint actions for a given belief distribution.

        For sparse beliefs, this is the union of valid actions across all
        states with non-zero probability. Results are cached for efficiency.
        Falls back to all actions if action masks are not provided.
        """
        if not self.use_action_masks:
            return range(self.nactions)

        # Check cache first
        cached = self._valid_actions_cache.get(dist_id)
        if cached is not None:
            return cached

        belief = self.dists[dist_id]
        valid_set = set()

        for s, prob in enumerate(belief):
            if prob > EPSILON:
                valid_set.update(self.valid_actions_per_state.get(s, range(self.nactions)))

        # Convert to sorted list for consistent ordering
        valid_list = sorted(valid_set)
        if not valid_list:
            valid_list = list(range(self.nactions))  # Fallback to all actions

        self._valid_actions_cache[dist_id] = valid_list
        return valid_list

    @staticmethod
    def _wkey(w: float) -> int:
        return int((10 ** 12 + 39) * w)

    # ---------- Centralized DP heuristic ----------
    # Compute V_rh(b) where b is referenced by dist_id.
    def cen_dp_V(self, rh: int, dist_id: BeliefID) -> float:
        if rh <= 0:
            return 0.0
        key = (rh, dist_id)
        v = self.cen_V.get(key)
        if v is not None:
            return v
        # Maximize over valid joint actions (action mask optimization)
        best = -math.inf
        valid_actions = self.get_valid_actions_for_belief(dist_id)
        for ja in valid_actions:
            q = self.cen_dp_Q(rh, dist_id, ja)
            if q > best:
                best = q
        self.cen_V[key] = best
        # Also populate the heuristic map on the minimal key used elsewhere
        self.cen_heuristic[(rh, dist_id, 1)] = best
        return best

    # Compute Q_rh(b, a) = R(b,a) + sum_o P(o|b,a) V_{rh-1}(b_o')
    def cen_dp_Q(self, rh: int, dist_id: BeliefID, ja: JointActionID) -> float:
        key = (rh, dist_id, ja)
        q = self.cen_Q.get(key)
        if q is not None:
            return q

        r = self.reward_list[dist_id * self.nactions + ja]
        
        # Short-circuit: no future value at horizon 1
        if rh <= 1:
            self.cen_Q[key] = r
            return r

        # Only compute belief updates for h > 1
        sparse_transitions = self.get_terminal(dist_id, ja)
        exp = 0.0
        rh_1 = rh - 1

        for _, p_o, d_next in sparse_transitions:
            exp += p_o * self.cen_dp_V(rh_1, d_next)

        q = r + exp
        self.cen_Q[key] = q

        short_ctr = 2 + ja
        self.cen_heuristic[(rh, dist_id, short_ctr)] = q
        return q

    def cen_dp_V_hybrid(self, rh: int, dist_id: BeliefID, r_depth: int) -> float:
        # Wrapper for V that calls the hybrid Q
        if rh <= 0: return 0.0

        # Check cache to avoid redundant computation
        key = (rh, dist_id, r_depth)
        v = self.cen_V_hybrid.get(key)
        if v is not None:
            return v

        best = -math.inf
        # Action mask optimization: only iterate over valid actions
        valid_actions = self.get_valid_actions_for_belief(dist_id)
        for ja in valid_actions:
            # Call the hybrid Q function
            val = self.cen_dp_Q_hybrid(rh, dist_id, ja, r_depth)
            if val > best:
                best = val

        self.cen_V_hybrid[key] = best
        return best

    def _get_terminal_batched(self, dist: BeliefID, act: JointActionID) -> List[Tuple[ObsID, Prob, BeliefID]]:
        # 1. Prepare Inputs
        b = self.dists[dist]

        # 2. Call Optimized Function (Dispatched)
        p_obs_all, joint_unnorm = self.dynamics_fn(
            b, self.T_repr, self.O_repr, self.nactions, self.nstates, self.nobs
        )

        # 3. Process results into Dictionary
        for a in range(self.nactions):
            p_obs = p_obs_all[a]
            sparse_transitions = []
            
            valid_obs_indices = np.flatnonzero(p_obs > EPSILON)
            
            for o in valid_obs_indices:
                unnorm_posterior = joint_unnorm[a, o]
                _, did = self.get_init(unnorm_posterior)
                sparse_transitions.append((o, p_obs[o], did))
            
            self.terminal_dict[(dist, a)] = sparse_transitions
            
        return self.terminal_dict[(dist, act)]

    # r_depth: The number of steps we are allowed to perform full POMDP branching
    def cen_dp_Q_hybrid(self, rh: int, dist_id: BeliefID, ja: JointActionID, r_depth: int) -> float:
        # 1. Base Case: If we have exhausted our "POMDP budget" (r_depth == 0),
        #    we stop branching and return the QMDP value for the remaining horizon.
        if r_depth <= 0:
            belief = self.dists[dist_id]
            
            # self.qmdp_Q is shape (maxh+1, nactions, nstates)
            # We want the value for the remaining horizon 'rh'
            q_values_h = self.qmdp_Q[rh] 
            
            # QMDP Value V(b) = max_a Sum_s b(s) Q(s,a)
            # But here we are computing Q(b, ja), so we just need the specific action ja
            return np.dot(belief, q_values_h[ja])

        # 2. Standard Memoization Check
        # We need to include r_depth in the key so we don't mix hybrid vs full values
        key = (rh, dist_id, ja, r_depth)
        q = self.cen_Q.get(key)
        if q is not None:
            return q

        r_val = self.reward_list[dist_id * self.nactions + ja]
        
        if rh <= 1:
            self.cen_Q[key] = r_val
            return r_val

        # 3. Recursive Step (Standard QPOMDP logic)
        sparse_transitions = self.get_terminal(dist_id, ja)
        exp = 0.0
        rh_1 = rh - 1
        
        # Decrement the depth budget for the next step
        next_r = r_depth - 1

        for _, p_o, d_next in sparse_transitions:          
            # Recursively call V, which calls Q_hybrid
            exp += p_o * self.cen_dp_V_hybrid(rh_1, d_next, next_r)

        q = r_val + exp
        self.cen_Q[key] = q
        return q

    def exact_central_Q_sbt(self, rh: int, dist_id: BeliefID, ja: JointActionID, extra_horizon: int = 0) -> float:
        key = ("partQ", rh, dist_id, ja, extra_horizon)
        cached = self.cen_Q.get(key)
        if cached is not None:
            return cached

        r = self.reward_list[dist_id * self.nactions + ja]
        sparse_transitions = self.get_terminal(dist_id, ja)
        exp = 0.0
        rh_1 = rh - 1
        
        for _, p_o, d_next in sparse_transitions:
            c_id, d_id, p_dec = self.belief_split_by_id(d_next)

            # Determine if we're in the "tail" region where we use approximate heuristics
            # Both centralized and decentralized components use the same condition for consistency
            is_tail = self.TI3 and (rh_1 + extra_horizon <= self.rec_limit)

            v_c = 0.0
            if c_id != -1:
                if is_tail:
                    v_c = self.get_tail_centralized_value(rh_1 + extra_horizon, c_id)
                else:
                    v_c = self.get_core_centralized_value(rh_1 + extra_horizon, c_id)

            v_d = 0.0
            if p_dec > 0.0 and d_id != -1:
                if is_tail:
                    v_d = self.get_tail_centralized_value(rh_1 + extra_horizon, d_id)
                else:
                    v_d = self.get_core_centralized_value(rh_1 + extra_horizon, d_id)

            exp += p_o * ((1.0 - p_dec) * v_c + p_dec * v_d)

        q = r + exp
        self.cen_Q[key] = q
        return q

    def _solve_qmdp(self) -> None:
        """
        Solves the underlying MDP using vectorized Value Iteration.
        Populates self.qmdp_Q[h][a][s].
        """
        self.qmdp_Q = np.zeros((self.maxh + 1, self.nactions, self.nstates))

        if self.output:
            print(f"Pre-computing Q-MDP Q-values ({'sparse' if self.use_sparse else 'dense'})...", end=" ")

        for h in range(1, self.maxh + 1):
            # V(s') = max_a Q(s', a)
            v_prev = np.max(self.qmdp_Q[h-1], axis=0)

            # Expected future value: Sum(T(s,a,s') * V(s'))
            if self.use_sparse:
                for a in range(self.nactions):
                    # CSR @ vector
                    self.qmdp_Q[h, a, :] = self.RA[a, :] + self.T_repr[a] @ v_prev
            else:
                # einsum: 'asr,r->as'
                future_val = np.einsum('asr,r->as', self.T_repr, v_prev)
                self.qmdp_Q[h] = self.RA + future_val

        if self.output:
            print("Done.")

    def get_terminalMDP(self, init: BeliefID, h: int, ctr_fix: int = 1) -> float:
        """
        Equivalent to SDecPOMDP.get_terminalMDP for the MDP heuristic case.
        Returns: max_{ja extending ctr_fix} E_b[Q_MDP(s, ja, h)]

        Args:
            init: Belief distribution index
            h: Horizon (remaining horizon, should be >= 1)
            ctr_fix: Partial action counter (1 = maximize over all actions)

        Returns:
            MDP heuristic value
        """
        # Check cache first (matching SDecPOMDP's caching strategy)
        if ctr_fix == 1:
            cache_key = (init, h)
        else:
            cache_key = (init, h, ctr_fix)

        cached = self.terminalMDP_dict.get(cache_key)
        if cached is not None:
            return cached

        # Compute the value
        belief = np.asarray(self.dists[init], dtype=np.float64)

        best = -np.inf
        for full_ctr in self.ctrs[ctr_fix]:
            ja = self.ctr_to_ja[full_ctr]
            val = np.dot(belief, self.qmdp_Q[h, ja])
            if val > best:
                best = val

        # Cache and return
        self.terminalMDP_dict[cache_key] = best
        return best

    # ---------- TI2 (Fixed: Scoped Budgeting) ----------
    def check_progress_pruning(self, pi_c: Policy, idx: int, ctr: int, aidx: int, cidx: bool, policyidx: int) -> bool:
        """
        Calculates the semi-decentralized progress score (prog) and prunes if the
        global counter (ctr) exceeds the allowed progress for this policy depth.

        MODIFIED FORMULA:
            prog = σ * L + k * B + max(c/|C|, p) * B

        The entity progress term is now normalized:
            effective_fraction = max(c / |C|, p)
            entity_progress = effective_fraction * B

        This ensures:
        - Entity progress is always in [0, B] regardless of cluster count
        - Higher probability p always helps (never penalizes)
        - Progress scales correctly when |C| > B

        Symbol definitions:
        - σ (sigma): Completed stages
        - L: iter_limit (total budget)
        - k: Completed entities in current stage (Centralized=0, Agent0=1, ...)
        - B: Per-entity budget = L / (n + 1)
        - c: Completed clusters for current entity
        - |C|: Total clusters for current entity
        - p: Cumulative probability of completed clusters

        NOTE: iter_limit should be set such that B = L/(n+1) >= max expected
        clusters per entity. If total_clusters > B, some clusters may not be fully
        explored before pruning occurs (a warning will be logged).
        """

        current_stage = idx - 1
        L = self.iter_limit
        n = self.nagents
        n_entities = n + 1  # n Agents + 1 Centralized Component
        B = L / n_entities  # Per-entity budget

        progress = current_stage * L

        if cidx:
            k = 0
        else:
            k = 1 + aidx

        progress += k * B
        c = policyidx
        p = 0.0
        total_clusters = 0

        if cidx:
            # --- Centralized Component Logic ---
            if current_stage < len(pi_c.dists_cen):
                total_clusters = len(pi_c.dists_cen[current_stage])
                # Sum probability of all fixed clusters (0 to c-1)
                # prob_cen is a simple list [P(b0), P(b1), ...]
                if current_stage < len(pi_c.prob_cen):
                    limit = min(c, len(pi_c.prob_cen[current_stage]))
                    for i in range(limit):
                        p += pi_c.prob_cen[current_stage][i]
        elif aidx < n:
            # --- Decentralized Agent Logic ---
            if current_stage < len(pi_c.ncluster):
                total_clusters = pi_c.ncluster[current_stage][aidx]
                if current_stage < len(pi_c.prob) and c > 0:
                    divs, _ = cumprod(pi_c.ncluster[current_stage])
                    relevant_joint_indices = lists_product2(
                        aidx,
                        range(c),
                        pi_c.ncluster[current_stage],
                        divs,
                        self.nagents
                    )

                    for joint_idx in relevant_joint_indices:
                        if joint_idx < len(pi_c.prob[current_stage]):
                            p += pi_c.prob[current_stage][joint_idx]

        # 5. Compute Entity Progress (Normalized)
        if total_clusters > 0:
            if total_clusters > B and c == 0:
                entity_type = "centralized" if cidx else f"agent {aidx}"
                print(f"[TI2 WARNING] Stage {current_stage} {entity_type} has {total_clusters} clusters "
                      f"but per-entity budget is {B:.0f}. Consider increasing iter_limit.")

            # Normalize progress: use max of cluster fraction and probability fraction
            cluster_fraction = c / total_clusters
            effective_fraction = max(cluster_fraction, p)
            entity_progress = effective_fraction * B
        else:
            # No clusters for this entity
            entity_progress = 0

        progress += entity_progress

        if ctr > progress:
            return True

        return False
    
    # ---------- TI1 ----------
    def get_horizon_centralization_scores(self, q_heap: List, top_n: int, threshold: float, temperature: float) -> Tuple[List[bool], List[float]]:
        """
        Analyzes the top N nodes using Weighted Majority Voting.

        The horizon length that possesses the highest total Softmax probability mass
        is selected as the target horizon.

        Handles tri-state centralization vectors where:
            - True (1.0):  Stage is centralized
            - False (0.0): Stage is decentralized
            - None:        Stage is incomplete (excluded from voting for that stage)

        For each stage, only nodes with complete (non-None) values contribute to the
        weighted average. This prevents incomplete stages from being conflated with
        decentralized stages.
        """

        # 1. Efficiently peek at top N nodes
        top_k_tuples = nsmallest(top_n, q_heap)

        if not top_k_tuples:
            return [], []

        # 2. Extract Data (Value, Vector, Length)
        # Store these in parallel lists for efficient numpy masking later
        vals = []
        vecs = []
        lengths = []

        for tup in top_k_tuples:
            real_val = -tup[0]  # Flip negative heap value back to positive
            piv = tup[2]
            vec = self.centralization_vector(piv)

            vals.append(real_val)
            vecs.append(vec)
            lengths.append(len(vec))

        vals = np.array(vals)
        lengths = np.array(lengths)

        finite_mask = np.isfinite(vals)
        if not np.any(finite_mask):
            return [], []  # No valid values to analyze

        vals = vals[finite_mask]
        lengths = lengths[finite_mask]
        vecs = [vecs[i] for i in range(len(finite_mask)) if finite_mask[i]]

        # 3. GLOBAL Softmax Weighting
        # Compute weights for ALL candidates immediately.
        # This allows high-value nodes to dominate the selection process.

        # Shift values for numerical stability
        shift_vals = (vals - np.max(vals)) / temperature
        exp_vals = np.exp(shift_vals)
        weights = exp_vals / np.sum(exp_vals)

        # 4. Determine Best Horizon by Probability Mass
        # Sum the weights for every unique length found.
        unique_lengths = np.unique(lengths)
        best_horizon = -1
        max_mass = -1.0

        for l in unique_lengths:
            # Sum weights where length == l
            mass = np.sum(weights[lengths == l])
            if mass > max_mass:
                max_mass = mass
                best_horizon = l

        if best_horizon == -1:
            return [], []

        # 5. Filter Candidates by Best Horizon
        # Create a boolean mask for the winning length
        mask_horizon = (lengths == best_horizon)

        # Filter vectors and weights for the surviving subset
        filtered_vecs = [vecs[i] for i in range(len(vecs)) if mask_horizon[i]]
        filtered_weights = weights[mask_horizon]

        # Re-normalize weights so they sum to 1.0 within this new group
        weight_sum = np.sum(filtered_weights)
        if weight_sum <= 0:
            return [], []
        filtered_weights = filtered_weights / weight_sum

        # 6. Step-Wise Aggregation with None Handling
        # For each stage, compute weighted average only over nodes with complete (non-None) values.
        # This prevents incomplete stages from being treated as decentralized (0.0).
        step_scores = []
        for stage_idx in range(best_horizon):
            stage_vals = []
            stage_weights = []

            for node_idx, vec in enumerate(filtered_vecs):
                val = vec[stage_idx] if stage_idx < len(vec) else None
                if val is not None:
                    # Convert True/False to 1.0/0.0
                    stage_vals.append(1.0 if val else 0.0)
                    stage_weights.append(filtered_weights[node_idx])

            if stage_weights:
                # Re-normalize weights for this stage (only among nodes with complete values)
                stage_weights = np.array(stage_weights)
                stage_weights = stage_weights / np.sum(stage_weights)
                stage_score = np.dot(stage_weights, stage_vals)
            else:
                # No complete values for this stage - cannot determine centralization
                stage_score = 0.0

            step_scores.append(stage_score)

        # 7. Thresholding
        boolean_mask = [score > threshold for score in step_scores]

        return boolean_mask, step_scores

    # ---------- TI1 ----------
    def is_stage_complete(self, pi_c: Policy, stage_idx: int) -> bool:
        """
        Checks if a specific stage is fully expanded (complete) in the given policy.

        Returns True if the stage exists and is complete, False otherwise.
        A stage is complete when both its decentralized and centralized components
        (if applicable) are fully expanded.
        """
        if stage_idx >= len(pi_c.policy):
            return False

        # Check Decentralized Policy Expansion
        if stage_idx < len(pi_c.ncluster):
            for a in range(self.nagents):
                if len(pi_c.policy[stage_idx][0][a]) < pi_c.ncluster[stage_idx][a] or \
                   any(x == -2 for x in pi_c.policy[stage_idx][0][a]):
                    return False
        else:
            return False

        # Check Centralized Policy Expansion
        current_split = pi_c.dec_split[stage_idx] if stage_idx < len(pi_c.dec_split) else 1.0

        # If there is centralized mass, check if centralized policies are defined
        if current_split < 1.0 - EPSILON:
            n_cen_needed = len(pi_c.dists_cen[stage_idx]) if stage_idx < len(pi_c.dists_cen) else 0
            if len(pi_c.policy[stage_idx][1]) < n_cen_needed or \
               any(not vec for vec in pi_c.policy[stage_idx][1]):
                return False

        return True

    # ---------- TI1 ----------
    def centralization_vector(self, pi_c: Policy) -> List[Optional[bool]]:
        """
        Generates a tri-state vector indicating centralization status for each stage.

        Returns a list where each element is:
            - True:  Stage is fully expanded AND fully centralized (dec_split < EPSILON)
            - False: Stage is fully expanded AND decentralized (dec_split >= EPSILON)
            - None:  Stage is incomplete (not yet fully expanded) - should be excluded from voting
        """
        stages = len(pi_c.policy)
        cen_vec = []

        for i in range(stages):
            # --- 1. Check if Stage is Fully Expanded ---
            is_expanded = True

            # Check Decentralized Policy Expansion
            if i < len(pi_c.ncluster):
                for a in range(self.nagents):
                    # Check for length and presence of placeholder '-2' (node created but not yet solved)
                    if len(pi_c.policy[i][0][a]) < pi_c.ncluster[i][a] or \
                       any(x == -2 for x in pi_c.policy[i][0][a]):
                        is_expanded = False
                        break
            else:
                is_expanded = False

            # Check Centralized Policy Expansion
            if is_expanded:
                current_split = pi_c.dec_split[i] if i < len(pi_c.dec_split) else 1.0

                # If there is centralized mass, check if centralized policies are defined
                if current_split < 1.0 - EPSILON:
                    n_cen_needed = len(pi_c.dists_cen[i]) if i < len(pi_c.dists_cen) else 0

                    # Check if number of histories matches and if they contain actions
                    if len(pi_c.policy[i][1]) < n_cen_needed or \
                       any(not vec for vec in pi_c.policy[i][1]):
                        is_expanded = False

            if not is_expanded:
                # Incomplete stage - use None to distinguish from decentralized (False)
                cen_vec.append(None)
            else:
                # --- 2. Check Centralization Status ---
                # Check the split *entering* this stage.
                # If dec_split[i] is ~0.0, the system is wholly in the centralized component.

                split_val = pi_c.dec_split[i] if i < len(pi_c.dec_split) else 1.0
                is_centralized = bool(split_val < EPSILON) 

                # Returns True if the stage is fully centralized (no decentralized mass)
                cen_vec.append(is_centralized)

        return cen_vec

    # ---------- approximate heuristic relaxation (TI3/4) ----------
    def _get_centralized_value_internal(self, rh: int, init_belief_idx: BeliefID, heuristic_type: str, hybrid_r: int = 0) -> float:
        """
        Internal method to compute centralized value using specified heuristic type.

        Args:
            rh: Remaining horizon
            init_belief_idx: Index of belief distribution
            heuristic_type: "QMDP", "POMDP", "HYBRID", or other
            hybrid_r: Hybrid duration for hybrid heuristics (if applicable)
        """
        if rh <= 0: return 0.0

        if heuristic_type == "QMDP":
            # Q-MDP: Dot product of belief probabilities and State Values
            # Belief is self.dists[dist_id] (list or array)

            cache_key = (rh, init_belief_idx)
            if cache_key in self.qmdp_cache:
                return self.qmdp_cache[cache_key]
            
            belief = np.asarray(self.dists[init_belief_idx], dtype=float)
            q_values_h = self.qmdp_Q[rh]
            action_values = np.dot(q_values_h, belief)
            val = np.max(action_values)
            self.qmdp_cache[cache_key] = val

            return val

        # 3. Exact Centralized POMDP
        elif heuristic_type == "POMDP":
            return self.cen_dp_V(rh, init_belief_idx)

        elif heuristic_type == "HYBRID":
            return self.cen_dp_V_hybrid(rh, init_belief_idx, hybrid_r)

    def get_core_centralized_value(self, rh: int, init_belief_idx: BeliefID) -> float:
        """
        Compute centralized value for CORE algorithm heuristics.
        Uses self.heuristic_type setting.
        """
        return self._get_centralized_value_internal(rh, init_belief_idx, self.heuristic_type, self.hybrid_r)

    def get_tail_centralized_value(self, rh: int, init_belief_idx: BeliefID) -> float:
        """
        Compute centralized value for TI3 TAIL approximation.
        Uses self.tail_heuristic_type setting.
        """
        return self._get_centralized_value_internal(rh, init_belief_idx, self.tail_heuristic_type, self.hybrid_r)

    # ---------- approximate heuristic relaxation (TI3) ----------
    def compute_tail_value(self, pi_c: Policy, extra_horizon: int, dists_dec: Optional[List] = None, probs_dec: Optional[List] = None, dists_cen: Optional[List] = None, probs_cen: Optional[List] = None, split: Optional[float] = None) -> float:
        if extra_horizon <= 0: return 0.0
        val = 0.0
        
        if split is None:
            split = pi_c.dec_split[-1] if pi_c.dec_split else 1.0
        
        # 1. Decentralized
        if split > EPSILON:
            dists = dists_dec if dists_dec is not None else (pi_c.dists[-1] if pi_c.dists else [])
            probs = probs_dec if probs_dec is not None else (pi_c.prob[-1] if pi_c.prob else [])

            comp_val = 0.0
            for p, d in zip(probs, dists):
                if d != -1 and p > 0:
                    comp_val += p * self.get_tail_centralized_value(extra_horizon, d)
            val += comp_val * split

        # 2. Centralized
        if split < 1.0 - EPSILON:
            dists_c = dists_cen if dists_cen is not None else (pi_c.dists_cen[-1] if pi_c.dists_cen else [])
            probs_c = probs_cen if probs_cen is not None else (pi_c.prob_cen[-1] if pi_c.prob_cen else [])

            comp_val = 0.0
            for p, d in zip(probs_c, dists_c):
                if d != -1 and p > 0:
                    comp_val += p * self.get_tail_centralized_value(extra_horizon, d)
            val += comp_val * (1.0 - split)
            
        return val

    def window_clustering(self, pi_new: Policy, nOhs_dec: int, nOhs_cen: int, div_dec: List[int], aidx: int, dists_terminal: List[BeliefID], probs_terminal: List[Prob], nOhs_dec_parent: int) -> Tuple[int, List[List[int]], List[List[int]]]:
        dist_dicta = {} 
        cluster_newa = 0
        clustering_newa_dec = [[0]*self.nobs_factor[aidx] for _ in range(pi_new.ncluster[-1][aidx])]
        clustering_newa_cen = [[0]*self.nobs_factor[aidx] for _ in range(nOhs_cen)]
        
        new_stage_suffixes = {} 
        current_stage_idx = len(pi_new.ncluster)

        def get_parent_suffix(cluster_idx):
            # Check if parent stage exists in suffixes
            if current_stage_idx - 1 < 0 or current_stage_idx - 1 >= len(pi_new.suffixes):
                return ()
            try:
                return pi_new.suffixes[current_stage_idx - 1][aidx].get(cluster_idx, ())
            except (IndexError, KeyError):
                return ()

        local_validity = set() # Stores (oha, oa) pairs that have > 0 probability
        
        # 1. Scan Decentralized Parents
        limit_dec = min(nOhs_dec_parent * self.nobs, len(probs_terminal))

        for idx in range(limit_dec):
            if probs_terminal[idx] > EPSILON:
                # Decode Index -> Joint OH and Joint Obs
                oh = idx // self.nobs
                o = idx % self.nobs
                
                # Decode -> Local Cluster (oha) and Local Obs (oa)
                oha = (oh // div_dec[aidx]) % pi_new.ncluster[-1][aidx]
                oa = (o // self.o_prod[aidx]) % self.nobs_factor[aidx]
                
                local_validity.add((oha, oa))

        # --- 1. Process Decentralized Parents (Lookup Optimized) ---
        if nOhs_dec > 0:
            for oha in range(pi_new.ncluster[-1][aidx]):
                parent_suffix = get_parent_suffix(oha)
                
                for oa in range(self.nobs_factor[aidx]):
                    if (oha, oa) not in local_validity:
                        clustering_newa_dec[oha][oa] = -1
                        continue

                    new_suffix = parent_suffix + (oa,)
                    if self.memory is not None:
                        new_suffix = new_suffix[-self.memory:]
                    
                    if new_suffix not in dist_dicta:
                        dist_dicta[new_suffix] = cluster_newa
                        new_stage_suffixes[cluster_newa] = new_suffix
                        cluster_newa += 1
                    clustering_newa_dec[oha][oa] = dist_dicta[new_suffix]

        # --- 2. Process Centralized Parents ---        
        cen_offset = nOhs_dec_parent * self.nobs
        if nOhs_cen > 0:
            for j in range(nOhs_cen):
                for oa in range(self.nobs_factor[aidx]):
                    total_prob = 0
                    start_idx = cen_offset + j * self.nobs
                    
                    # Iterate all joint obs 'o' and check consistency (Vectorized-like)
                    # Since |O| is small (64), this inner loop is cheap (64 iters).
                    for o in range(self.nobs):
                        if (o // self.o_prod[aidx]) % self.nobs_factor[aidx] == oa:
                            idx_check = start_idx + o
                            if idx_check < len(probs_terminal) and probs_terminal[idx_check] > EPSILON:
                                total_prob = 1
                                break

                    if total_prob == 0:
                        clustering_newa_cen[j][oa] = -1
                    else:
                        new_suffix = (oa,) 
                        if self.memory is not None:
                            new_suffix = new_suffix[-self.memory:]

                        if new_suffix not in dist_dicta:
                            dist_dicta[new_suffix] = cluster_newa
                            new_stage_suffixes[cluster_newa] = new_suffix
                            cluster_newa += 1
                        clustering_newa_cen[j][oa] = dist_dicta[new_suffix]

        while len(pi_new.suffixes) <= current_stage_idx:
            pi_new.suffixes.append([{} for _ in range(self.nagents)])

        pi_new.suffixes[current_stage_idx][aidx] = new_stage_suffixes

        return cluster_newa, clustering_newa_dec, clustering_newa_cen

    def belief_split_by_id(self, dist_id: BeliefID) -> Tuple[BeliefID, BeliefID, float]:
        # Fast path: if no sync triggers, everything is decentralized
        if not self.sync_states:
            return (-1, dist_id, 1.0)

        cached = self.belief_split_cache.get(dist_id)
        if cached is not None:
            return cached

        # 1. Lazy Load Sparse Representation
        sparse_belief = self.dists_sparse.get(dist_id)
        if sparse_belief is None:
            # Build sparse representation from dense source of truth
            # This pays the O(N) cost exactly once per unique belief
            dense_belief = self.dists[dist_id]
            sparse_belief = [(s, p) for s, p in enumerate(dense_belief) if p > EPSILON]
            self.dists_sparse[dist_id] = sparse_belief

        # 2. Sparse Calculation of Mass
        prob_cen = 0.0
        prob_dec = 0.0

        mask = self.trigger_profile.state_mask
        
        # Iterate only over non-zero states (O(k) instead of O(N))
        for s, p in sparse_belief:
            # O(1) lookup using boolean mask
            if mask[s]:
                prob_cen += p
            else:
                prob_dec += p

        # 3. Fast Paths (Avoid constructing new arrays)
        if prob_dec <= EPSILON:
            out = (dist_id, -1, 0.0)     # all mass is centralized
            self.belief_split_cache[dist_id] = out
            return out
        
        if prob_cen <= EPSILON:
            out = (-1, dist_id, 1.0)     # all mass is decentralized
            self.belief_split_cache[dist_id] = out
            return out

        # 4. Partial Split Construction
        supp_c = np.zeros(self.nstates, dtype=np.float64)
        supp_d = np.zeros(self.nstates, dtype=np.float64)
        
        # Fill only the non-zero indices
        for s, p in sparse_belief:
            if mask[s]:
                supp_c[s] = p
            else:
                supp_d[s] = p

        _, c_id = self.get_init(supp_c)
        _, d_id = self.get_init(supp_d)
        out = (c_id, d_id, prob_dec)
        self.belief_split_cache[dist_id] = out
        return out

    def compute_clusterctr(self, cdict_i: Tuple) -> int:  # Compute_clusterctr: computes a number from which the clustering structure
        ci = 0                                            # Can be recovered, by converting the tuple into a number.
        factor = 1                               
        for a in range(self.nagents):
            factor1 = len(cdict_i[a]) * self.nobs_factor[a] + 1
            for x in cdict_i[a]:
                for y in x:
                    ci += factor*(y+1)
                    factor *= factor1
        return ci
        
    # Save heuristic values where more actions are fixed, based on the optimal policy where less actions are fixed
    def save_heuristic(self, rh: int, init: BeliefID, ctr: int, pi_heuristic: FullPolicy, aidx: int, dec: bool = True, extra_horizon: int = 0) -> None:
        ctr_heuristic = ctr
        pi0 = [x[0] for x in pi_heuristic[0][0]]
        for a in range(aidx, self.nagents):           
            ctr_heuristic = ctr_heuristic * self.maxa + pi0[a]
            if dec: 
                self.dec_heuristic[(rh, init, ctr_heuristic, (), extra_horizon)] = self.dec_heuristic[(rh, init, ctr, (), extra_horizon)]
            else:
                self.cen_heuristic[(rh + extra_horizon, init, ctr_heuristic)] = self.cen_heuristic[(rh + extra_horizon, init, ctr)]

    def save_heuristic2(self, rh: int, init: BeliefID, ctr: int, cdict_tup: Tuple, pi_heuristic: FullPolicy, aidx: int, policy_idx: int, extra_horizon: int = 0) -> None:
        ctr_heuristic = ctr
        idx = 0 
        pi_idx = pi_heuristic[idx][0][aidx][policy_idx:]
        for a in range(aidx+1, self.nagents):
            pi_idx.extend(pi_heuristic[idx][0][a])
        for p in pi_idx:
            ctr_heuristic = ctr_heuristic * self.maxa + p
            self.dec_heuristic[(rh, init, ctr_heuristic, cdict_tup, extra_horizon)] = self.dec_heuristic[(rh, init, ctr, cdict_tup, extra_horizon)]

    def compute_heuristic_init(self, rh: int, init: BeliefID, dec: bool = True, extra_horizon: int = 0) -> None:
        # 1. Centralized Case (Always fast)
        if not dec:
            val = self.get_core_centralized_value(rh + extra_horizon, init)
            self.cen_heuristic[(rh + extra_horizon, init, 1)] = val
            return

        # 2. Approximate Mode: Use QBG (Fast, Admissible, Looser)
        if self.algorithm == "approximate":
            h_total = rh + extra_horizon
            best_value = -math.inf

            valid_actions = self.get_valid_actions_for_belief(init)

            for ja in valid_actions:
                r = self.reward_list[init * self.nactions + ja]
                sparse_transitions = self.get_terminal(init, ja)

                future_val = 0.0
                for _, prob, next_belief_id in sparse_transitions:
                    qmdp_val = self._get_centralized_value_internal(h_total - 1, next_belief_id, "QMDP")
                    future_val += prob * qmdp_val

                if r + future_val > best_value:
                    best_value = r + future_val

            self.dec_heuristic[(rh, init, 1, (), extra_horizon)] = best_value
            return

        # 3. Exact Mode: Use Nested Search (Slow, Admissible, Tighter)
        # We keep the original logic here to ensure minimal node expansion for exact proofs.
        self.dec_heuristic[(rh, init, 1, (), extra_horizon)], pi_heuristic, _, _, _, _ = \
            self.multi_agent_astar(rh, init_beliefs = init, maxit = self.maxit, extra_horizon = extra_horizon)
        
        if pi_heuristic:
            self.save_heuristic(rh, init, 1, pi_heuristic, 0, dec=True, extra_horizon=extra_horizon)
    
    def compute_heuristic1(self, pi_c: Policy, oh: int, depth: int, rh: int, init: BeliefID, ctr: int, aidx: int, extra_horizon: int = 0) -> None:
        policy_oh = self.shorten_policy(pi_c, oh, depth, rh, extra_horizon=extra_horizon)
        
        upper_val = self.dec_heuristic.get((rh, init, ctr//self.maxa, (), extra_horizon))
        
        self.dec_heuristic[(rh, init, ctr, (), extra_horizon)], pi_heuristic, _, _, _, _ = \
            self.multi_agent_astar(rh, policy_oh, maxit = self.maxit, 
                                   upper = upper_val, 
                                   extra_horizon = extra_horizon)

        if pi_heuristic and aidx < self.nagents:
            self.save_heuristic(rh, init, ctr, pi_heuristic, aidx, dec=True, extra_horizon=extra_horizon)
        
    def compute_heuristic2(self, pi_c: Policy, oh: int, depth: int, rh: int, init: BeliefID, ctr: int, cdict_tup: Tuple, aidx: Optional[int] = None, extra_horizon: int = 0) -> None:
        policy_oh = self.shorten_policy(pi_c, oh, depth, rh, extra_horizon=extra_horizon)
        if aidx is None:
            aidx = 0
            while aidx < self.nagents and sum(1 for x in policy_oh.policy[-1][0][aidx] if x != -2) == policy_oh.ncluster[-1][aidx]: aidx += 1
        if aidx < self.nagents: 
            policy_idx = sum(1 for x in policy_oh.policy[-1][0][aidx] if x != -2)
            
        upper_h = self.dec_heuristic.get((rh, init, ctr//self.maxa, cdict_tup, extra_horizon))
        if upper_h is None: upper_h = self.dec_heuristic.get((rh, init, ctr//self.maxa, cdict_tup[:-1], extra_horizon))

        self.dec_heuristic[(rh, init, ctr, cdict_tup, extra_horizon)], pi_heuristic, _, _, _, _ = \
           self.multi_agent_astar(rh, policy_oh, maxit = self.maxit, upper = upper_h, extra_horizon = extra_horizon)

        if pi_heuristic and aidx < self.nagents:
            self.save_heuristic2(rh, init, ctr, cdict_tup, pi_heuristic, aidx, policy_idx, extra_horizon=extra_horizon)


    def get_init(self, probs: Union[List[float], np.ndarray]) -> Tuple[float, BeliefID]:
        if isinstance(probs, np.ndarray):
            if probs.dtype == np.float64:
                a = probs
            else:
                a = probs.astype(np.float64)
        else:
            a = np.asarray(probs, dtype=np.float64)

        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        dsum = float(a.sum())
        if dsum < EPSILON:
            return 0.0, -1
        a = a / dsum

        # SPARSE HASHING: Only hash non-zero entries
        nonzero_idx = np.nonzero(a > EPSILON)[0]
        
        # Create tuple of (state_idx, rounded_prob) pairs for non-zero entries
        dist_key = tuple((int(idx), int(a[idx] * HASH_FACTOR)) for idx in nonzero_idx)

        d = self.dist_dict.get(dist_key)
        if d is None:
            d = len(self.dists)
            self.dist_dict[dist_key] = d
            self.dists.append(a.copy()) 
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                exp_r = self.RA @ a
            
            exp_r = np.nan_to_num(exp_r, nan=0.0, posinf=0.0, neginf=0.0)
            self.reward_list.extend(exp_r.tolist())
        return dsum, d
    
    def evaluate_policy_sbt(self, pi_c: Policy, terminal_dec: bool = False, probs_terminal: Optional[List[Prob]] = None, dists_terminal: Optional[List[BeliefID]] = None, probs_terminal_cen: Optional[List[Prob]] = None, dists_terminal_cen: Optional[List[BeliefID]] = None, rh: Optional[int] = None, h: Optional[int] = None, extra_horizon: Optional[int] = None, dec_split: Optional[float] = None) -> float:
        if h is None: h = len(pi_c.policy)
        policyval = 0

        if terminal_dec: 
            prob_dec = pi_c.dec_split[-1] if pi_c.dec_split else 1.0
            # Decentralized
            if prob_dec > 0.0:
                if probs_terminal is None or dists_terminal is None:
                    idx_term = len(pi_c.prob) - 1 if pi_c.prob else -1
                    if idx_term >= 0:
                        probs_terminal = pi_c.prob[idx_term]
                        dists_terminal = pi_c.dists[idx_term]
                    else:
                        probs_terminal, dists_terminal = [], []
                for prob, init in zip(probs_terminal, dists_terminal):
                    if init != -1 and prob != 0.0:
                        heuristic = self.dec_heuristic.get((rh, init, 1, (), extra_horizon))
                        if heuristic is None:
                            self.compute_heuristic_init(rh, init, extra_horizon=extra_horizon)
                            heuristic = self.dec_heuristic[(rh, init, 1, (), extra_horizon)]

                        policyval += heuristic * prob * prob_dec
            
            # Centralized
            if prob_dec < 1.0:
                if probs_terminal_cen is None or dists_terminal_cen is None: 
                    idx_cen = len(pi_c.prob_cen) - 1 if pi_c.prob_cen else -1
                    if idx_cen >= 0:
                        probs_terminal_cen = pi_c.prob_cen[idx_cen]
                        dists_terminal_cen = pi_c.dists_cen[idx_cen]
                    else:
                        probs_terminal_cen, dists_terminal_cen = [], []
                for prob, init in zip(probs_terminal_cen, dists_terminal_cen):
                    if init != -1 and prob != 0.0:
                        heuristic = self.cen_heuristic.get((rh + extra_horizon, init, 1))
                        if heuristic is None:
                            self.compute_heuristic_init(rh, init, dec = False, extra_horizon=extra_horizon)
                            heuristic = self.cen_heuristic[(rh + extra_horizon, init, 1)]
                        policyval += heuristic * prob * (1.0 - prob_dec)
            
        if h == 0: 
            if not terminal_dec and self.algorithm == "approximate" and self.TI3 and extra_horizon:
                policyval += self.compute_tail_value(pi_c, extra_horizon)
            return policyval

        lp = len(pi_c.values)
        if lp < h:
            if lp == 0: pvalidx = 0
            else: pvalidx = pi_c.values[lp-1]
            for idx in range(lp, h):
                has_placeholder = False
                for a in range(self.nagents):
                    if any(x == -2 for x in pi_c.policy[idx][0][a]):
                        has_placeholder = True
                        break
                
                if has_placeholder:
                    break
                decent_acts_filtered = [[max(0, x) for x in pi_c.policy[idx][0][a]] for a in range(self.nagents)]
                decent_acts = [] if any(len(v) == 0 for v in decent_acts_filtered) else lists_product(decent_acts_filtered, self.a_prod, self.nagents)
                cent_acts = [item for subvec in pi_c.policy[idx][1] for item in subvec]
                
                if len(decent_acts) > 0: assert len(decent_acts) == len(pi_c.prob[idx]) == len(pi_c.dists[idx]), "Mismatch in decentralized shapes"
                if len(cent_acts) > 0: assert len(cent_acts) == len(pi_c.prob_cen[idx]) == len(pi_c.dists_cen[idx]), "Mismatch in centralized shapes"

                dec_mass = (pi_c.dec_split[idx] if idx < len(pi_c.dec_split) else (pi_c.dec_split[-1] if pi_c.dec_split else 1.0))
                cen_mass = 1.0 - dec_mass

                if len(decent_acts) > 0 and dec_mass > 0.0:
                    for ctr, (prob, init) in enumerate(zip(pi_c.prob[idx], pi_c.dists[idx])):
                        if init == -1 or prob == 0.0:
                            continue
                        pvalidx += self.reward_list[init*self.nactions + decent_acts[ctr]] * prob * dec_mass
                if len(cent_acts) > 0 and cen_mass > 0.0:
                    for ctr, (prob, init) in enumerate(zip(pi_c.prob_cen[idx], pi_c.dists_cen[idx])):
                        if init == -1 or prob == 0.0:
                            continue
                        pvalidx += self.reward_list[init*self.nactions + cent_acts[ctr]] * prob * cen_mass

                if idx < len(pi_c.step_cen_values):
                    pvalidx += pi_c.step_cen_values[idx]

                pi_c.values.append(pvalidx) 
        
        policyval += pi_c.values[h-1]

        if not terminal_dec and self.algorithm == "approximate" and self.TI3 and extra_horizon:
            if probs_terminal is not None:
                 policyval += self.compute_tail_value(pi_c, extra_horizon, 
                                                      dists_dec=dists_terminal, probs_dec=probs_terminal,
                                                      dists_cen=dists_terminal_cen, probs_cen=probs_terminal_cen,
                                                      split=dec_split)
            else:
                 policyval += self.compute_tail_value(pi_c, extra_horizon)

        return policyval
    
    def get_newstatedist(self, dist: BeliefID, act: JointActionID) -> np.ndarray:
        newstatedist = self.newstatedist_dict.get((dist, act))
        return newstatedist if newstatedist is not None else self.compute_newstatedist(dist, act)

    def compute_newstatedist(self, dist: BeliefID, act: JointActionID) -> np.ndarray:
        probs = self.dists[dist]

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Works for both sparse (CSR matrices) and dense (3D numpy array) representations
            newstatedist = probs @ self.T_repr[act]

        newstatedist = np.nan_to_num(newstatedist, nan=0.0, posinf=0.0, neginf=0.0)
        self.newstatedist_dict[(dist, act)] = newstatedist
        return newstatedist
        
    def get_terminal(self, dist: BeliefID, act: JointActionID) -> List[Tuple[ObsID, Prob, BeliefID]]:
        terminal = self.terminal_dict.get((dist, act))
        return terminal if terminal is not None else self._get_terminal_batched(dist, act)
 
    def _terminal_probabilities_fully_dec(self, pi_c: Policy) -> Tuple[List[BeliefID], List[BeliefID], List[Prob], List[Prob], float, int, int]:
        """
        Fast path for terminal_probabilities_sbt when sync_states/sync_actions/sync_observations are all empty.
        Mimics SDecPOMDP's simple terminal_probabilities - no splitting logic needed.
        """
        nOhs_dec = len(pi_c.dists[-1])
        dec_acts = lists_product(pi_c.policy[-1][0], self.a_prod, self.nagents)

        len_dec = nOhs_dec * self.nobs
        terminal_dists = [-1] * len_dec
        terminal_probs = [0.0] * len_dec

        # Simple loop like SDecPOMDP's terminal_probabilities
        for oh in range(nOhs_dec):
            if pi_c.dists[-1][oh] != -1 and pi_c.prob[-1][oh] > 0:
                # get_terminal returns sparse transitions: list of (obs_idx, prob, next_dist_id)
                sparse_transitions = self.get_terminal(pi_c.dists[-1][oh], dec_acts[oh])
                prob_oh = pi_c.prob[-1][oh]

                # Process sparse results - only store non-zero entries
                for obs, p_o, next_d in sparse_transitions:
                    idx = oh * self.nobs + obs
                    terminal_dists[idx] = next_d
                    terminal_probs[idx] = p_o * prob_oh

        # Return format expected by callers: all mass is decentralized
        # dists_dec_list, dists_cen_list, prob_dec_list, prob_cen_list, dec_split, nOhs_dec, nOhs_cen
        empty_cen = [-1] * len_dec
        zero_probs = [0.0] * len_dec
        return terminal_dists, empty_cen, terminal_probs, zero_probs, 1.0, nOhs_dec, 0

    def terminal_probabilities_sbt(self, pi_c: Policy) -> Tuple[List[BeliefID], List[BeliefID], List[Prob], List[Prob], float, int, int]:
        # FAST PATH: If NO triggers exist at all, use fully dec path
        if not self.sync_states and not self.trigger_profile.sync_actions and not self.trigger_profile.sync_observations:
            return self._terminal_probabilities_fully_dec(pi_c)

        # Use UNCOMPRESSED arrays for decentralized parents
        nOhs_dec = len(pi_c.dists[-1])
        nOhs_cen = len(pi_c.dists_cen[-1])

        dec_acts = lists_product(pi_c.policy[-1][0], self.a_prod, self.nagents)
        cent_acts = pi_c.policy[-1][1]

        len_dec = nOhs_dec * self.nobs
        len_cen = nOhs_cen * self.nobs

        # Allocate arrays (Initialize to -1 / 0.0)
        terminal_dists_dec = [-1] * len_dec
        terminal_probs_dec = [0.0] * len_dec
        terminal_dists_cen = [-1] * len_cen
        terminal_probs_cen = [0.0] * len_cen

        # Split tracking arrays
        dec_dec_split = [0.0] * len_dec
        dec_cen_split = [0.0] * len_dec
        cen_dec_split = [0.0] * len_cen
        cen_cen_split = [0.0] * len_cen

        terminal_dists_dec_dec = [-1] * len_dec
        terminal_dists_dec_cen = [-1] * len_dec
        terminal_dists_cen_dec = [-1] * len_cen
        terminal_dists_cen_cen = [-1] * len_cen

        terminal_probs_dec_dec = [0.0] * len_dec
        terminal_probs_dec_cen = [0.0] * len_dec
        terminal_probs_cen_dec = [0.0] * len_cen
        terminal_probs_cen_cen = [0.0] * len_cen

        prob_dec_dec_total = 0.0
        prob_dec_cen_total = 0.0
        prob_cen_dec_total = 0.0
        prob_cen_cen_total = 0.0

        # === PROCESS DECENTRALIZED PARENTS ===
        for oh in range(nOhs_dec):
            if pi_c.dists[-1][oh] == -1 or pi_c.prob[-1][oh] == 0:
                continue

            act = dec_acts[oh]
            prob_oh = pi_c.prob[-1][oh]

            # 1. ACTION TRIGGER CHECK
            force_sync_action = act in self.trigger_profile.sync_actions

            sparse_transitions = self.get_terminal(pi_c.dists[-1][oh], act)

            for o, p, d_next in sparse_transitions:
                idx = oh * self.nobs + o
                p_term = p * prob_oh

                # 2. OBSERVATION TRIGGER CHECK
                force_sync_obs = force_sync_action or (o in self.trigger_profile.sync_observations)

                if force_sync_obs:
                    # --- PATH A: TOTAL SYNCHRONIZATION (via action/observation trigger) ---
                    # Write to dec arrays (idx is based on dec parent), split indicates 100% cen
                    terminal_dists_dec[idx] = d_next
                    terminal_probs_dec[idx] = p_term

                    # 100% Centralized split (no mass stays decentralized)
                    dec_dec_split[idx] = 0.0
                    dec_cen_split[idx] = 1.0

                    # Mapping: Dec -> Cen (Total)
                    terminal_dists_dec_cen[idx] = d_next

                    prob_dec_cen_total += p_term
                else:
                    # --- PATH B: STATE-BASED / PARTIAL SYNCHRONIZATION ---
                    terminal_dists_dec[idx] = d_next
                    terminal_probs_dec[idx] = p_term

                    if p_term > EPSILON:
                        c_probs_idx, d_probs_idx, prob_dec = self.belief_split_by_id(d_next)

                        terminal_dists_dec_dec[idx] = d_probs_idx
                        terminal_dists_dec_cen[idx] = c_probs_idx
                        dec_dec_split[idx] = prob_dec
                        dec_cen_split[idx] = 1.0 - prob_dec

                        prob_dec_dec_total += p_term * prob_dec
                        prob_dec_cen_total += p_term * (1.0 - prob_dec)

        # Normalize Decentralized Parents
        if prob_dec_dec_total > 0:
            inv_total = 1.0 / prob_dec_dec_total
            for idx in range(len_dec):
                if terminal_probs_dec[idx] > 0 and dec_dec_split[idx] > 0:
                    terminal_probs_dec_dec[idx] = dec_dec_split[idx] * terminal_probs_dec[idx] * inv_total

        if prob_dec_cen_total > 0:
            inv_total = 1.0 / prob_dec_cen_total
            for idx in range(len_dec):
                # Both Path A (total sync) and Path B (partial sync) use terminal_probs_dec
                # dec_cen_split indicates fraction going to centralized (1.0 for Path A, variable for Path B)
                if terminal_probs_dec[idx] > 0 and dec_cen_split[idx] > 0:
                    terminal_probs_dec_cen[idx] = dec_cen_split[idx] * terminal_probs_dec[idx] * inv_total

        # === PROCESS CENTRALIZED PARENTS ===
        for oh in range(nOhs_cen):
            if pi_c.dists_cen[-1][oh] == -1 or pi_c.prob_cen[-1][oh] == 0:
                continue

            act = cent_acts[oh][0]
            prob_oh = pi_c.prob_cen[-1][oh]

            # 1. ACTION TRIGGER CHECK
            force_sync_action = act in self.trigger_profile.sync_actions

            sparse_transitions = self.get_terminal(pi_c.dists_cen[-1][oh], act)

            for o, p, d_next in sparse_transitions:
                idx = oh * self.nobs + o
                p_term = p * prob_oh

                # 2. OBSERVATION TRIGGER CHECK
                force_sync_obs = force_sync_action or (o in self.trigger_profile.sync_observations)

                if force_sync_obs:
                    # Force stay Centralized
                    terminal_dists_cen[idx] = d_next
                    terminal_probs_cen[idx] = p_term

                    cen_dec_split[idx] = 0.0
                    cen_cen_split[idx] = 1.0

                    terminal_dists_cen_cen[idx] = d_next
                    prob_cen_cen_total += p_term
                else:
                    # Standard State Split
                    terminal_dists_cen[idx] = d_next
                    terminal_probs_cen[idx] = p_term

                    if p_term > EPSILON:
                        c_probs_idx, d_probs_idx, prob_dec = self.belief_split_by_id(d_next)

                        terminal_dists_cen_dec[idx] = d_probs_idx
                        terminal_dists_cen_cen[idx] = c_probs_idx
                        cen_dec_split[idx] = prob_dec
                        cen_cen_split[idx] = 1.0 - prob_dec

                        prob_cen_dec_total += p_term * prob_dec
                        prob_cen_cen_total += p_term * (1.0 - prob_dec)

        # Normalize Centralized Parents
        if prob_cen_dec_total > 0:
            inv_total = 1.0 / prob_cen_dec_total
            for idx in range(len_cen):
                if terminal_probs_cen[idx] > 0 and cen_dec_split[idx] > 0:
                    terminal_probs_cen_dec[idx] = cen_dec_split[idx] * terminal_probs_cen[idx] * inv_total

        if prob_cen_cen_total > 0:
            inv_total = 1.0 / prob_cen_cen_total
            for idx in range(len_cen):
                 # Check both paths (Forced vs Split)
                 # In centralized parent case, both end up in terminal_probs_cen anyway
                 if terminal_probs_cen[idx] > 0:
                     # If forced, cen_cen_split is 1.0. If split, it's 1-p_dec.
                     terminal_probs_cen_cen[idx] = cen_cen_split[idx] * terminal_probs_cen[idx] * inv_total

        # Compute adjustment factors
        adj_factor_dec_dec = pi_c.dec_split[-1] * prob_dec_dec_total
        adj_factor_dec_cen = pi_c.dec_split[-1] * prob_dec_cen_total
        adj_factor_cen_dec = (1 - pi_c.dec_split[-1]) * prob_cen_dec_total
        adj_factor_cen_cen = (1 - pi_c.dec_split[-1]) * prob_cen_cen_total

        total_dec = adj_factor_dec_dec + adj_factor_cen_dec
        total_cen = adj_factor_dec_cen + adj_factor_cen_cen

        # Build output arrays
        prob_dec_list = [0.0] * (len_dec + len_cen)
        prob_cen_list = [0.0] * (len_dec + len_cen)
        dists_dec_list = [-1] * (len_dec + len_cen)
        dists_cen_list = [-1] * (len_dec + len_cen)

        # From decentralized parents
        if total_dec > 0 and adj_factor_dec_dec > 0:
            factor = adj_factor_dec_dec / total_dec
            for i in range(len_dec):
                prob_dec_list[i] = terminal_probs_dec_dec[i] * factor
        if total_cen > 0 and adj_factor_dec_cen > 0:
            factor = adj_factor_dec_cen / total_cen
            for i in range(len_dec):
                prob_cen_list[i] = terminal_probs_dec_cen[i] * factor
        for i in range(len_dec):
            dists_dec_list[i] = terminal_dists_dec_dec[i]
            dists_cen_list[i] = terminal_dists_dec_cen[i]

        # From centralized parents
        if total_dec > 0 and adj_factor_cen_dec > 0:
            factor = adj_factor_cen_dec / total_dec
            for i in range(len_cen):
                prob_dec_list[len_dec + i] = terminal_probs_cen_dec[i] * factor
        if total_cen > 0 and adj_factor_cen_cen > 0:
            factor = adj_factor_cen_cen / total_cen
            for i in range(len_cen):
                prob_cen_list[len_dec + i] = terminal_probs_cen_cen[i] * factor
        for i in range(len_cen):
            dists_dec_list[len_dec + i] = terminal_dists_cen_dec[i]
            dists_cen_list[len_dec + i] = terminal_dists_cen_cen[i]

        dec_split = adj_factor_dec_dec + adj_factor_cen_dec

        return dists_dec_list, dists_cen_list, prob_dec_list, prob_cen_list, dec_split, nOhs_dec, nOhs_cen
           
    def convert_probabilities(self, probs_new_s: np.ndarray, nOhs_new: int) -> Tuple[List[Prob], List[BeliefID]]:
        probs_new = [0.0]*nOhs_new
        dists_new = [-1]*nOhs_new
        for oh in range(nOhs_new):
            ohl = probs_new_s[oh*self.nstates:(oh+1)*self.nstates]
            probs_new[oh], dists_new[oh] = self.get_init(ohl)
        return probs_new, dists_new
    
    def cluster_policy_dec(self, pi_c: Policy, dists_terminal: List[BeliefID], probs_terminal: List[Prob], nOhs_dec_parent: int, nOhs_cen_parent: int) -> Policy:
        """
        Cluster policy for decentralized component.
        
        dists_terminal and probs_terminal are UNCOMPRESSED arrays:
        - First nOhs_dec_parent * nobs entries: from decentralized parents
        - Next nOhs_cen_parent * nobs entries: from centralized parents that transition to decentralized
        
        Uses direct indexing like RSMAA (no compression mapping needed).
        """
        pi_new = pi_c.cluster_copy()
        cluster_tuple = tuple(pi_c.ncluster[-1])
        probs_tuple = int_tuple(probs_terminal)
        dists_tuple = tuple(dists_terminal)
        new = self.cluster_dict.get((cluster_tuple, probs_tuple, dists_tuple))

        if new is not None:                   
            cluster_new, clustering_new, probs_new, dists_new, clustering_cen_new = new
            pi_new.ncluster.append(cluster_new)
            pi_new.clustering.append(clustering_new)
            pi_new.prob.append(probs_new)
            pi_new.dists.append(dists_new)
            pi_new.clustering_cen.append(clustering_cen_new)
            return pi_new
          
        cluster_new = []
        clustering_new = []
        div_dec, nOhs_dec_potential = cumprod(pi_new.ncluster[-1])
        
        clustering_new_cen_tmp = []

        for a in range(self.nagents):
            if self.algorithm == "approximate" and self.TI4:
                c_new_a, cl_new_dec_a, cl_new_cen_a = self.window_clustering(pi_new, nOhs_dec_potential, nOhs_cen_parent, div_dec, a, dists_terminal, probs_terminal, nOhs_dec_parent)
            else:
                c_new_a, cl_new_dec_a, cl_new_cen_a = self.lossless_clustering(pi_new, nOhs_dec_potential, nOhs_cen_parent, div_dec, a, dists_terminal, probs_terminal, nOhs_dec_parent)
            cluster_new.append(c_new_a)
            clustering_new.append(cl_new_dec_a)
            clustering_new_cen_tmp.append(cl_new_cen_a)

        clustering_new = tuple([tuple([tuple(x) for x in y]) for y in clustering_new])
        pi_new.ncluster.append(cluster_new)
        pi_new.clustering.append(clustering_new)
        clustering_new_cen_tuple = tuple([tuple([tuple(x) for x in y]) for y in clustering_new_cen_tmp])
        pi_new.clustering_cen.append(clustering_new_cen_tuple)

        cluster_newprod, nOhs_new = cumprod(cluster_new)
        probs_new_s = [0.0]*(nOhs_new*self.nstates)

        # Process decentralized parents using DIRECT indexing
        if nOhs_dec_potential > 0:
            oh_newlist_dec = None
            oholist_dec = None
            for a in range(self.nagents):
                oh_newlista = []
                oholista = []
                for oha in range(pi_new.ncluster[-2][a]):
                    for oa in range(self.nobs_factor[a]):
                        oh_newa = pi_new.clustering[-1][a][oha][oa]
                        if oh_newa != -1:
                            oh_newlista.append(oh_newa)
                            oholista.append(oha*div_dec[a]*self.nobs + oa*self.o_prod[a])
                if a == 0:
                    oh_newlist_dec = oh_newlista
                    oholist_dec = oholista
                else:
                    oh_newlist_dec = product(oh_newlist_dec, oh_newlista, cluster_newprod[a])
                    oholist_dec = product(oholist_dec, oholista, 1)

            # Direct indexing - no mapping needed since dists_terminal is uncompressed
            for oho, oh_new in zip(oholist_dec, oh_newlist_dec):
                if oho < len(dists_terminal) and dists_terminal[oho] != -1 and probs_terminal[oho] > 0.0:
                    dist = self.dists[dists_terminal[oho]]
                    base = oh_new * self.nstates
                    w = probs_terminal[oho]
                    for snew in range(self.nstates):
                        probs_new_s[base + snew] += w * dist[snew]

        # Process centralized parents that transition to decentralized
        # These start at offset nOhs_dec_parent * nobs in dists_terminal
        cen_offset = nOhs_dec_parent * self.nobs
        
        if nOhs_cen_parent > 0:
            def get_local_obs(o, a):
                return (o // self.o_prod[a]) % self.nobs_factor[a]

            for j in range(nOhs_cen_parent):
                for o in range(self.nobs):
                    idx_flat = cen_offset + j * self.nobs + o
                    if idx_flat >= len(dists_terminal) or dists_terminal[idx_flat] == -1 or probs_terminal[idx_flat] == 0.0:
                        continue
                    cluster_ids = [0] * self.nagents
                    valid = True
                    for a in range(self.nagents):
                        oa = get_local_obs(o, a)
                        cid = clustering_new_cen_tmp[a][j][oa]
                        if cid == -1:
                            valid = False
                            break
                        cluster_ids[a] = cid
                    if not valid:
                        continue
                    oh_new = 0
                    for a in range(self.nagents):
                        oh_new += cluster_ids[a] * cluster_newprod[a]
                    dist = self.dists[dists_terminal[idx_flat]]
                    base = oh_new * self.nstates
                    w = probs_terminal[idx_flat]
                    for snew in range(self.nstates):
                        probs_new_s[base + snew] += w * dist[snew]

        probs_new, dists_new = self.convert_probabilities(probs_new_s, nOhs_new)
        pi_new.prob.append(probs_new)
        pi_new.dists.append(dists_new)
        
        self.cluster_dict[(cluster_tuple, probs_tuple, dists_tuple)] = (cluster_new, clustering_new, probs_new, dists_new, clustering_new_cen_tuple)
        return pi_new
    
    def lossless_clustering(self, pi_new: Policy, nOhs_dec: int, nOhs_cen: int, div_dec: List[int], aidx: int, dists_terminal: List[BeliefID], probs_terminal: List[Prob], nOhs_dec_parent: int) -> Tuple[int, List[List[int]], List[List[int]]]:
        """
        Compute lossless clustering for agent aidx.

        dists_terminal and probs_terminal are UNCOMPRESSED:
        - First nOhs_dec_parent * nobs entries: from decentralized parents
        - Next nOhs_cen * nobs entries: from centralized parents

        Uses direct indexing (like RSMAA) - no parent_map needed.
        """
        nobs_a = self.nobs_factor[aidx]
        os_by_oa = self._os_by_oa_cache.get(aidx)
        if os_by_oa is None:
            os_by_oa = [lists_product2(aidx, [oa], self.nobs_factor, self.o_prod, self.nagents) for oa in range(nobs_a)]
            self._os_by_oa_cache[aidx] = os_by_oa

        clustering_newa_dec = [[0]*nobs_a for _ in range(pi_new.ncluster[-1][aidx])]
        clustering_newa_cen = [[0]*nobs_a for _ in range(nOhs_cen)]
        cluster_newa = 0
        dist_dicta = {}

        # 1. Decentralized Parents
        if nOhs_dec > 0:
            for oha in range(pi_new.ncluster[-1][aidx]):
                ohs_dec = lists_product2(aidx, [oha], pi_new.ncluster[-1], div_dec, self.nagents)

                for oa in range(nobs_a):
                    acc = {}
                    total = 0.0
                    os_list = os_by_oa[oa]

                    for oh in ohs_dec:
                        base = oh * self.nobs
                        for o in os_list:
                            idx = base + o
                            if idx >= len(probs_terminal):
                                continue
                            p = probs_terminal[idx]
                            if p == 0.0:
                                continue
                            dist_id = dists_terminal[idx]
                            if dist_id == -1:
                                continue
                            acc[dist_id] = acc.get(dist_id, 0.0) + p
                            total += p

                    if total <= 0.0:
                        clustering_newa_dec[oha][oa] = -1
                    else:
                        pairs = sorted((d, (acc[d]/total)) for d in acc)
                        sig = tuple((int(d), self._wkey(w)) for d, w in pairs)
                        cid = dist_dicta.get(sig)
                        if cid is None:
                            cid = cluster_newa
                            cluster_newa += 1
                            dist_dicta[sig] = cid
                        clustering_newa_dec[oha][oa] = cid

        # 2. Centralized Parents
        cen_offset = nOhs_dec_parent * self.nobs

        if nOhs_cen > 0:
            for j in range(nOhs_cen):
                parent_base = cen_offset + j * self.nobs
                for oa in range(nobs_a):
                    acc = {}
                    total = 0.0
                    os_list = os_by_oa[oa]

                    for o in os_list:
                        idx = parent_base + o
                        if idx >= len(probs_terminal):
                            continue
                        p = probs_terminal[idx]
                        if p == 0.0:
                            continue
                        dist_id = dists_terminal[idx]
                        if dist_id == -1:
                            continue
                        acc[dist_id] = acc.get(dist_id, 0.0) + p
                        total += p

                    if total <= 0.0:
                        clustering_newa_cen[j][oa] = -1
                    else:
                        pairs = sorted((d, (acc[d]/total)) for d in acc)
                        sig = tuple((int(d), self._wkey(w)) for d, w in pairs)
                        cid = dist_dicta.get(sig)
                        if cid is None:
                            cid = cluster_newa
                            cluster_newa += 1
                            dist_dicta[sig] = cid
                        clustering_newa_cen[j][oa] = cid

        return cluster_newa, clustering_newa_dec, clustering_newa_cen
    
    def compute_short_ctr(self, pi_c: Policy, oh: int, depth: int, div: List[int], idx: int) -> int:
        short_ctr = 1
        ohs_a = [[(oh//div[a])%pi_c.ncluster[depth][a]] for a in range(self.nagents)]

        for a in range(self.nagents):
            val = pi_c.policy[depth][0][a][ohs_a[a][0]]
            short_ctr = short_ctr * self.maxa + val
        
        # Compute all observation histories reachable from oh
        for j in range(depth+1, idx):
            ohs_newa = [[pi_c.clustering[j-1][a][ohj][o] for ohj in ohs_a[a] for o in range(self.nobs_factor[a])] 
                    for a in range(self.nagents)]
            ohs_newa = [list(set(x)) for x in ohs_newa]
            for a in range(self.nagents): 
                ohs_newa[a].sort()
                if ohs_newa[a][0] == -1:
                    ohs_newa[a] = ohs_newa[a][1:]
                    
            ohs_a = ohs_newa
            for a in range(self.nagents):
                lp = sum(x != -1 and x != -2 for x in pi_c.policy[j][0][a])
                for ohja in ohs_a[a]:
                    if lp > ohja:
                        val = pi_c.policy[j][0][a][ohja]
                        short_ctr = short_ctr * self.maxa + val
                    else: 
                        return short_ctr

        return short_ctr

    def shorten_cluster(self, pi_c: Policy, oh: int, depth: int, return_ctr: bool = True) -> Union[Tuple[int, ...], Tuple[List, List, List]]:
        rh = len(pi_c.policy) - depth

        if return_ctr:
            clustering_tup = tuple(pi_c.clustering[depth:])
            cached_val = self.clusterctr_dict.get((rh, oh, clustering_tup))
            if cached_val is not None:
                return cached_val
            
        div, _ = cumprod(pi_c.ncluster[depth])
        cluster_short = [[1]*self.nagents]
        cluster_map = [[{(oh//div[a])%pi_c.ncluster[depth][a]: 0} for a in range(self.nagents)]]
        cluster_map_inv = [[[(oh//div[a])%pi_c.ncluster[depth][a]] for a in range(self.nagents)]]
        clustering_short = []

        for idx in range(rh-1):
            cluster_new = []
            clustering_new = []
            cluster_map.append([dict() for _ in range(self.nagents)])
            cluster_map_inv.append([])
            
            for a in range(self.nagents):
                new_clusters = [pi_c.clustering[idx+depth][a][oh_a][oa] for oh_a in cluster_map_inv[idx][a] for oa in range(self.nobs_factor[a])]
                new_clusters = list(set(new_clusters))
                new_clusters.sort()
                if new_clusters[0] == -1:
                    new_clusters = new_clusters[1:]
                cluster_newa = len(new_clusters)
                cluster_map_inv[idx+1].append(new_clusters)
                for c in range(cluster_newa):
                    cluster_map[idx+1][a][new_clusters[c]] = c
                clustering_dicta = [[0]*self.nobs_factor[a] for _ in range(cluster_short[idx][a])] 
                c_range = range(cluster_short[idx][a])
                for j in c_range:
                    oh_a = cluster_map_inv[idx][a][j]
                    for oa in range(self.nobs_factor[a]):
                        oh_new = pi_c.clustering[idx+depth][a][oh_a][oa]
                        if oh_new == -1: clustering_dicta[j][oa] = -1
                        else: clustering_dicta[j][oa] = cluster_map[idx+1][a][oh_new]
                cluster_new.append(cluster_newa)
                clustering_new.append(clustering_dicta)
            
            cluster_short.append(cluster_new)
            clustering_short.append(clustering_new)
        
        if return_ctr:
            clusterctr = [self.compute_clusterctr(clustering_short[idx]) for idx in range(rh-1)]
            result = tuple(clusterctr)
            self.clusterctr_dict[(rh, oh, clustering_tup)] = result
            return result
                
        clustering_short = [tuple([tuple([tuple(x) for x in y]) for y in c]) for c in clustering_short]
        return cluster_short, clustering_short, cluster_map_inv

    def shorten_policy(self, pi_c: Policy, oh: int, depth: int, full_length: int, extra_horizon: int = 0) -> Policy:
        rh = len(pi_c.policy) - depth
        pi_short = []
        cluster_short, clustering_short, cluster_map_inv = self.shorten_cluster(pi_c, oh, depth, False)

        probs_short = [[1]]
        dists_short = [[pi_c.dists[depth][oh]]]

        step_cen_values = []

        values_short = []
        running_value = 0.0

        valid_transition_calc = True

        for idx in range(rh):
            pi_idx = [[[pi_c.policy[idx+depth][0][a][cluster_map_inv[idx][a][j]] for j in range(cluster_short[idx][a]) if cluster_map_inv[idx][a][j] < len(pi_c.policy[idx+depth][0][a])] for a in range(self.nagents)], []]
            pi_short.append(pi_idx)

            has_placeholders = False
            for a in range(self.nagents):
                if any(x == -2 for x in pi_idx[0][a]):
                    has_placeholders = True
                    break

            if idx == rh - 1:
                step_cen_values.append(0.0)
                break

            if has_placeholders or not valid_transition_calc:
                valid_transition_calc = False
                step_cen_values.append(0.0)
                probs_short.append([])
                dists_short.append([])
                continue

            div, nOhs = cumprod(cluster_short[idx])
            cluster_newprod, nOhs_new = cumprod(cluster_short[idx + 1])
            probs_new_s = np.zeros(nOhs_new * self.nstates, dtype=np.float64)

            try:
                acts = lists_product(pi_idx[0], self.a_prod, self.nagents)
            except (IndexError, ValueError, TypeError):
                valid_transition_calc = False
                step_cen_values.append(0.0)
                probs_short.append([])
                dists_short.append([])
                continue

            step_dec_reward = 0.0
            current_probs = probs_short[-1]
            current_dists = dists_short[-1]

            for oh_idx in range(nOhs):
                d = current_dists[oh_idx]
                p = current_probs[oh_idx]
                if d != -1 and p > EPSILON:
                    step_dec_reward += self.reward_list[d * self.nactions + acts[oh_idx]] * p

            oh_newas = [[] for a in range(self.nagents)]
            olistas = [[] for a in range(self.nagents)]
            for a in range(self.nagents):
                for oha in range(cluster_short[idx][a]):
                    oh_newas[a].append([c for c in clustering_short[idx][a][oha] if c != -1])
                    olistas[a].append([oa for oa in range(self.nobs_factor[a]) if clustering_short[idx][a][oha][oa] != -1])

            for oh in range(nOhs):
                act = acts[oh]
                ohas = [(oh//div[a])%cluster_short[idx][a] for a in range(self.nagents)]
                oh_newlist = lists_product([oh_newas[a][ohas[a]] for a in range(self.nagents)], cluster_newprod, self.nagents)
                olist = lists_product([olistas[a][ohas[a]] for a in range(self.nagents)], self.o_prod, self.nagents)

                sums = self.get_newstatedist(dists_short[-1][oh], act)
                snew_indices = np.nonzero(sums)[0]
                p_oh = probs_short[-1][oh]

                for snew in snew_indices:
                    val_s = sums[snew] * p_oh

                    for oh_new, o in zip(oh_newlist, olist):
                        obs_prob = self.O[act, snew, o] if not self.use_sparse else self.O_csr_list[act][snew, o]
                        if obs_prob > 0: probs_new_s[oh_new*self.nstates+snew] += obs_prob * val_s

            stage_cen_loss = 0.0
            remaining_horizon = full_length - (idx + 1)

            probs_new = [0.0] * nOhs_new
            dists_new = [-1] * nOhs_new

            for i in range(nOhs_new):
                start_idx = i * self.nstates
                end_idx = start_idx + self.nstates

                # Extract the dense probability vector for this OH
                ohl = probs_new_s[start_idx:end_idx]

                # 1. Single expensive call to get_init
                dsum, dist_id = self.get_init(ohl)

                probs_new[i] = dsum
                dists_new[i] = dist_id

                # 2. Perform centralization check using the result immediately
                if dsum > EPSILON:
                    c_id, _, p_dec = self.belief_split_by_id(dist_id)

                    if p_dec < 1.0 - EPSILON:
                        val_cen = 0.0
                        if c_id != -1:
                            val_cen = self.get_core_centralized_value(remaining_horizon + extra_horizon, c_id)

                        loss = dsum * (1.0 - p_dec) * val_cen
                        stage_cen_loss += loss

            step_cen_values.append(stage_cen_loss)

            # Update running value with both components
            running_value += step_dec_reward + stage_cen_loss
            values_short.append(running_value)

            probs_new, dists_new = self.convert_probabilities(probs_new_s, nOhs_new)
            probs_short.append(probs_new)
            dists_short.append(dists_new)

        proper_dists_cen = [[-1] for _ in range(len(pi_short))]
        proper_probs_cen = [[1.0] for _ in range(len(pi_short))]

        # Pass pre-calculated values to Policy
        pi_c_short = Policy(pi_short, cluster_short, dists_short, probs_short, clustering_short, values=values_short, dists_cen = proper_dists_cen, prob_cen = proper_probs_cen, dec_split = [1.0] * len(pi_short), clustering_cen = [], step_cen_values = step_cen_values)

        idx = len(pi_c_short.ncluster)
        pi_c_short.heuristics = [math.inf]*idx
        if idx > 1:
            pi_c_short.depth = min(idx-1, self.IEmin2)

            depth_short = idx - 1
            div, nOhs = cumprod(pi_c_short.ncluster[depth_short])

            # Use the pre-calculated value directly
            policyval = pi_c_short.values[depth_short-1] if depth_short > 0 else 0.0

            rh_rem = full_length - depth_short
            for oh in range(nOhs):
                init = pi_c_short.dists[depth_short][oh]
                if init != -1:
                    short_ctr = 1
                    aidx = 0
                    for a in range(self.nagents):
                        oha = (oh//div[a])%pi_c_short.ncluster[depth_short][a]
                        if oha < len(pi_c_short.policy[depth_short][0][a]):
                            val = pi_c_short.policy[depth_short][0][a][oha]
                            if val == -2: break
                            short_ctr = short_ctr * self.maxa + val
                            aidx += 1
                        else: break

                    if (rh_rem, init, short_ctr, (), extra_horizon) not in self.dec_heuristic:
                        if short_ctr == 1: self.compute_heuristic_init(rh_rem, init, extra_horizon=extra_horizon)
                        else: self.compute_heuristic1(pi_c_short, oh, depth_short, rh_rem, init, short_ctr, aidx, extra_horizon=extra_horizon)

                    policyval += pi_c_short.prob[depth_short][oh] * self.dec_heuristic[(rh_rem, init, short_ctr, (), extra_horizon)]
            pi_c_short.heuristics[depth_short] = policyval

            for depth_it in range(pi_c_short.depth, idx-1):
                div, nOhs = cumprod(pi_c_short.ncluster[depth_it])

                # Use the pre-calculated value directly
                policyval = pi_c_short.values[depth_it-1] if depth_it > 0 else 0.0

                rh_rem = full_length - depth_it

                for oh in range(nOhs):
                    init = pi_c_short.dists[depth_it][oh]
                    if init != -1:
                        short_ctr = self.compute_short_ctr(pi_c_short, oh, depth_it, div, idx)
                        ctup = self.shorten_cluster(pi_c_short, oh, depth_it)

                        if (rh_rem, init, short_ctr, ctup, extra_horizon) not in self.dec_heuristic:
                            if (rh_rem, init, short_ctr, ctup[:-1], extra_horizon) in self.dec_heuristic:
                                self.dec_heuristic[(rh_rem, init, short_ctr, ctup, extra_horizon)] = self.dec_heuristic[(rh_rem, init, short_ctr, ctup[:-1], extra_horizon)]
                            else: self.compute_heuristic2(pi_c_short, oh, depth_it, rh_rem, init, short_ctr, ctup, aidx=None, extra_horizon=extra_horizon)

                        policyval += pi_c_short.prob[depth_it][oh] * self.dec_heuristic[(rh_rem, init, short_ctr, ctup, extra_horizon)]
                pi_c_short.heuristics[depth_it] = policyval
        return pi_c_short

    def _compute_qmdp_action_value(self, act: ActionID, aidx: int, oha: int, len_ohna: int, act_allohna: List[JointActionID], dist_arr: List[BeliefID], prob_arr: List[Prob], extra_horizon: int) -> float:
        """
        Helper to compute QMDP Q-value for a given action at the last stage.
        Used by both action optimization and fixed action evaluation.
        """
        current_val = 0.0
        h_total = 1 + extra_horizon
        for ohna in range(len_ohna):
            idx_flat = ohna + oha * len_ohna
            if idx_flat < len(dist_arr) and idx_flat < len(prob_arr) and dist_arr[idx_flat] != -1 and prob_arr[idx_flat] > EPSILON:
                joint_act = act_allohna[ohna] + act * self.a_prod[aidx]
                dist_id = dist_arr[idx_flat]
                prob = prob_arr[idx_flat]
                belief = np.asarray(self.dists[dist_id], dtype=np.float64)
                q_vals = self.qmdp_Q[h_total, joint_act]
                current_val += np.dot(belief, q_vals) * prob
        return current_val

    def multi_agent_astar(self, h: int, init_policy: Optional[Policy] = None, init_beliefs: BeliefID = 0, maxit: Optional[int] = None, upper: Optional[float] = None, extra_horizon: int = 0) -> Tuple[float, Optional[FullPolicy], Optional[List], List, Dict, List]:       
        if h == 0:
            return 0.0, [], [], [], [], []

        if init_policy is None:
            self.init_call = True  # Top-level call: reset for reuse
            self._last_reported_mem_gb = 0  # Reset memory tracking for new solve
            c_probs_idx, d_probs_idx, prob_dec = self.belief_split_by_id(init_beliefs)
            init_policy_cen = [] if prob_dec > 1.0 - EPSILON else [[]]
            dec_ph = -2 if prob_dec > EPSILON else -1
            init_suffixes = [[{0: ()} for _ in range(self.nagents)]]
            init_policy = Policy(policy = [[[[dec_ph] for _ in range(self.nagents)], init_policy_cen]], ncluster = [[1 for _ in range(self.nagents)]],
                                    dists = [[d_probs_idx]], prob = [[1]], clustering = [], values = [], heuristics = [math.inf], prob_cen = [[1.0]], dists_cen = [[c_probs_idx]], dec_split = [prob_dec], suffixes = init_suffixes)
        
        init_call = self.init_call
        self.init_call = False

        if self.algorithm == "approximate" and self.TI3 and extra_horizon == 0 and (not init_call) and h > self.rec_limit:
            hnew = min(self.rec_limit, h)
            extra_horizon = h - hnew
            h = hnew
        
        if self.algorithm == "approximate" and self.TI3 and init_call and h > self.rec_limit:
            init_policy.depth = 0
            init_policy.heuristics[0] = self.get_terminalMDP(0, h)

        if upper is not None: bound = upper - self.alpha*max(abs(upper), 1) 
        policyval = min(init_policy.heuristics)
        unique = count()

        q = [(-policyval, -next(unique), init_policy)]
        policyvalfound = -math.inf

        ctr = 0 # Total iterations
        ctr2 = 0 # Iterations within finite heuristic (used for maxit termination)

        while True:
            if len(q) == 0:
                # Queue empty - no valid policy found
                # Return default values instead of crashing
                return -math.inf, [], [], [], {}, []
            value, _, pi_c = heappop(q)
            value = -value

            idx = len(pi_c.policy)
            aidx = 0
            cidx = any(len(vec) == 0 for vec in pi_c.policy[-1][1])

            while not cidx and aidx < self.nagents and not any(x == -2 for x in pi_c.policy[-1][0][aidx]):
                aidx += 1

            policyidx = 0
            if cidx:
                # Find index of first empty centralized cluster
                policyidx = next((i for i, vec in enumerate(pi_c.policy[idx-1][1]) if not vec), 0)
            elif aidx < self.nagents:
                policyidx = pi_c.policy[idx-1][0][aidx].index(-2)

            if self.algorithm == "approximate" and self.TI2 and init_call:
                if self.check_progress_pruning(pi_c, idx, ctr, aidx, cidx, policyidx):
                    continue

            ctr += 1
            if value != math.inf: ctr2 += 1

            # Progress logging for debugging (controlled by self.output flag)
            if self.output and (ctr < 1000 or (ctr % 100 == 0 and ctr < 10000) or ctr % 10000 == 0) and init_call:
                print(f"[A*] iter={ctr} value={value:.8f} best={policyvalfound:.8f} depth={pi_c.depth}")
                sys.stdout.flush()

            # Memory limit check (only on main call, every N iterations)
            if init_call and self.memory_limit_gb is not None and ctr % self.memory_check_interval == 0:
                mem_usage = get_memory_usage_gb()
                # Log when memory increases by 1GB or more since last report
                mem_gb_floor = int(mem_usage)
                last_gb_floor = int(self._last_reported_mem_gb)
                if mem_gb_floor > last_gb_floor:
                    print(f"[Memory] {mem_usage:.2f}GB used at iteration {ctr}")
                    sys.stdout.flush()
                    self._last_reported_mem_gb = mem_usage
                if mem_usage > self.memory_limit_gb:
                    raise MemoryLimitExceeded(f"Memory usage {mem_usage:.2f}GB exceeds limit {self.memory_limit_gb}GB at iteration {ctr}")

            if ctr2 == maxit or (upper is not None and value <= bound):
                rval = min(upper, value, min(pi_c.heuristics)) if upper is not None else min(value, min(pi_c.heuristics))
                return rval, None, None, [], [], []

            if aidx == self.nagents: # expand stage
                pi_c = pi_c.cluster_copy()
                if idx == h:
                    if len(pi_c.dists) == len(pi_c.policy):
                        dists_dec, dists_cen, probs_dec, probs_cen, dec_split, _, _ = self.terminal_probabilities_sbt(pi_c)
                        value = self.evaluate_policy_sbt(pi_c, extra_horizon = extra_horizon,
                                                         probs_terminal=probs_dec, dists_terminal=dists_dec,
                                                         probs_terminal_cen=probs_cen, dists_terminal_cen=dists_cen,
                                                         dec_split=dec_split)
                    else:
                        value = self.evaluate_policy_sbt(pi_c, extra_horizon = extra_horizon)

                    cent_vec = self.centralization_vector(pi_c) if init_call else []

                    # VALIDATION: Assert consistency between policy and clustering
                    for step_idx in range(len(pi_c.policy) - 1):  # Skip last step (no clustering needed)
                        dec_policy = pi_c.policy[step_idx][0]
                        has_valid_dec_actions = any(a >= 0 for agent in dec_policy for a in agent)
                        has_clustering = step_idx < len(pi_c.clustering) and len(pi_c.clustering[step_idx]) >= self.nagents
                        assert not (has_valid_dec_actions and not has_clustering), f"Inconsistent policy at step {step_idx}: valid dec actions but no clustering"

                    return value, pi_c.policy, pi_c.clustering, cent_vec, pi_c.dists_cen, pi_c.clustering_cen

                dists_dec_list, dists_cen_list, prob_dec_list, prob_cen_list, dec_split, nOhs_dec_parent, nOhs_cen_parent = self.terminal_probabilities_sbt(pi_c)
                pi_c.dec_split.append(dec_split)

                # Cluster decentralized dists/probs (merged parents)
                if dec_split > EPSILON:
                    pi_c = self.cluster_policy_dec(pi_c, dists_dec_list, prob_dec_list, nOhs_dec_parent, nOhs_cen_parent)
                else:
                    pi_c.prob.append([1.0])
                    pi_c.dists.append([-1])
                    prev_ncluster = pi_c.ncluster[-1]
                    pi_c.ncluster.append([0] * self.nagents)
                    # Create minimal valid clustering structure
                    minimal_clustering = []
                    for a in range(self.nagents):
                        agent_clustering = []
                        num_clusters = max(1, prev_ncluster[a])
                        for _ in range(num_clusters):
                            agent_clustering.append(tuple([-1] * self.nobs_factor[a]))

                        # Single history cluster with all observations invalid (-1)
                        minimal_clustering.append(agent_clustering)
                    
                    pi_c.clustering.append(tuple([tuple(c) for c in minimal_clustering]))
                    pi_c.clustering_cen.append(tuple([tuple(c) for c in minimal_clustering]))

                # Cluster centralized mass (group identical posterior dists)
                if dec_split < 1.0 - EPSILON:
                    groups = {}
                    for prob, dist in zip(prob_cen_list, dists_cen_list):
                        groups.setdefault(dist, []).append((prob))

                    new_cent_probs, new_cent_dists = [], []
                    for dist, items in groups.items():
                        if dist != -1:
                            s = sum(prob for prob in items)
                            new_cent_probs.append(s)
                            new_cent_dists.append(dist)

                    pi_c.prob_cen.append(new_cent_probs)
                    pi_c.dists_cen.append(new_cent_dists)
                else:
                    pi_c.prob_cen.append([1.0])
                    pi_c.dists_cen.append([-1])

                # Heuristic updates
                policyval_new = self.evaluate_policy_sbt(pi_c, terminal_dec = True, probs_terminal = None, dists_terminal = None, rh = max(0, h - idx), extra_horizon = extra_horizon)
                pi_c.heuristics.append(policyval_new)

                if pi_c.depth is None: 
                    pi_c.depth = idx

                while len(pi_c.heuristics) > pi_c.depth + 1 and (pi_c.depth < self.IEmin2 or (self.TI3 and pi_c.depth <= idx - self.rec_limit) or pi_c.heuristics[pi_c.depth] + TOLERANCE > min(pi_c.heuristics[(pi_c.depth+1):])):
                    pi_c.depth += 1

                cen_ct = len(pi_c.prob_cen[-1]) if pi_c.dec_split[-1] < 1.0 - EPSILON else 0
                dec_ph = -2 if pi_c.dec_split[-1] > EPSILON else -1
                cidx = (cen_ct > 0)
                pi_c.policy.append([[[dec_ph] * max(1, pi_c.ncluster[-1][a]) for a in range(self.nagents)], [[] for _ in range(cen_ct)]])
                
                idx += 1
                aidx = 0
                policyidx = 0      

            cidx = any(len(vec) == 0 for vec in pi_c.policy[idx-1][1])            

            if idx == h and cidx:
                new_pi_c = pi_c.policy_copy_laststage_cent(idx-1)
                policyval = self.evaluate_policy_sbt(new_pi_c, h = h-1, extra_horizon = extra_horizon)
                depth_cen = len(pi_c.prob_cen) - 1
                if depth_cen >= 0:
                    probs_cen_arr = pi_c.prob_cen[depth_cen]
                    dists_cen_arr = pi_c.dists_cen[depth_cen]
                    num = min(len(new_pi_c.policy[-1][1]), len(probs_cen_arr), len(dists_cen_arr))
                    for oha in range(num):
                        init = dists_cen_arr[oha]
                        if init == -1 or probs_cen_arr[oha] < EPSILON:
                            new_pi_c.policy[idx-1][1][oha] = [-1]  # Invalid/skip action for unreachable slots
                            continue
                        w = probs_cen_arr[oha]
                        
                        valid_actions = self.get_valid_actions_for_belief(init) # Action mask optimization: only evaluate valid actions
                        if self.algorithm == "approximate" and self.TI3 and extra_horizon > 0:
                            h_total = 1 + extra_horizon
                            belief = np.asarray(self.dists[init], dtype=np.float64)
                            val_act_max = -math.inf
                            joa_select = valid_actions[0] if valid_actions else 0
                            for joa in valid_actions:
                                q_vals = self.qmdp_Q[h_total, joa]
                                val = np.dot(belief, q_vals) * w
                                if val > val_act_max:
                                    val_act_max = val
                                    joa_select = joa
                        else:
                            base = init * self.nactions
                            val_act_max = -math.inf
                            joa_select = valid_actions[0] if valid_actions else 0
                            for joa in valid_actions:
                                val = self.reward_list[base + joa] * w
                                if val > val_act_max:
                                    val_act_max = val
                                    joa_select = joa
            
                        if val_act_max == -math.inf: # Guard against empty valid_actions (val_act_max remains -inf)
                            val_act_max = 0.0
                        new_pi_c.policy[idx-1][1][oha] = [joa_select]
                        policyval += val_act_max * (1 - pi_c.dec_split[-1])
                
                if not any(-2 in x for x in new_pi_c.policy[-1][0]):
                    new_pi_c.heuristics = [policyval]
                    if policyval > policyvalfound:
                        policyvalfound = policyval
                        q = [q_elt for q_elt in q if -q_elt[0] + TOLERANCE >= policyvalfound]
                        heapify(q)
                    if abs(policyval - value) < EPSILON:
                        cent_vec = self.centralization_vector(new_pi_c) if init_call else []
                        return policyval, new_pi_c.policy, new_pi_c.clustering, cent_vec, new_pi_c.dists_cen, new_pi_c.clustering_cen
                    if policyval + TOLERANCE >= policyvalfound:
                        heappush(q, (-policyval, -next(unique), new_pi_c))
                else:
                    new_pi_c.final_cidx_value = policyval
                    policyval = min(new_pi_c.heuristics)
                    if policyval + TOLERANCE >= policyvalfound:
                        heappush(q, (-policyval, -next(unique), new_pi_c))

            elif idx == h and aidx == self.nagents - 1:
                new_pi_c = pi_c.policy_copy_laststage(idx-1, aidx)
                policyval = new_pi_c.final_cidx_value if len(new_pi_c.policy[-1][1]) else self.evaluate_policy_sbt(new_pi_c, h = h-1) 
                additive_val = 0
                act_allohna = [x for x in pi_c.policy[-1][0][0] if x != -1]

                for a in range(1, self.nagents - 1):
                    act_allohna = product(act_allohna, [x for x in pi_c.policy[-1][0][a] if x != -1], self.a_prod[a])
                len_ohna = len(act_allohna)
                depth_dec = len(pi_c.prob) - 1
                                
                # Iterate directly over the policy vector slots
                for i, action_val in enumerate(new_pi_c.policy[idx-1][0][aidx]):
                    oha = i 
                    if action_val == -1:
                        continue # Skip invalid slots

                    if depth_dec < 0:
                        continue

                    # Setup rewards/weights (Same logic as before)
                    prob_arr = pi_c.prob[depth_dec]
                    dist_arr = pi_c.dists[depth_dec]
                    reward_idxs = []
                    weights = []

                    valid_acts_for_slot = set()

                    for ohna in range(len_ohna):
                        idx_flat = ohna + oha*len_ohna
                        if idx_flat < len(dist_arr) and idx_flat < len(prob_arr) and dist_arr[idx_flat] != -1 and prob_arr[idx_flat] > EPSILON:
                            reward_idxs.append(dist_arr[idx_flat]*self.nactions + act_allohna[ohna])
                            weights.append(prob_arr[idx_flat])
                            
                            # Extract valid single-agent actions from valid joint actions
                            if self.use_action_masks:
                                valid_jas = self.get_valid_actions_for_belief(dist_arr[idx_flat])
                                for ja in valid_jas:
                                    s_act = (ja // self.a_prod[aidx]) % self.nacts_factor[aidx]
                                    valid_acts_for_slot.add(s_act)
                    
                    if not reward_idxs:
                        # Fallback for empty rewards (should be rare if prob > 0)
                        if action_val == -2:
                            new_pi_c.policy[idx-1][0][aidx][i] = -1  # Mark as invalid/unreachable
                        continue
                    
                    if not valid_acts_for_slot:
                        valid_acts_for_slot = range(self.nacts_factor[aidx])
                    else:
                        valid_acts_for_slot = sorted(list(valid_acts_for_slot))

                    use_qmdp = self.algorithm == "approximate" and self.TI3 and extra_horizon > 0

                    if action_val == -2:
                        # Case A: Slot is empty (-2). Solve for optimal action.
                        val_acts = []
                        for act in valid_acts_for_slot:
                            if use_qmdp:
                                current_val = self._compute_qmdp_action_value(act, aidx, oha, len_ohna, act_allohna, dist_arr, prob_arr, extra_horizon)
                            else:
                                current_val = sum(self.reward_list[ridx + act*self.a_prod[aidx]] * w for ridx, w in zip(reward_idxs, weights))
                            val_acts.append(current_val)

                        # Guard against empty val_acts (degenerate case)
                        if not val_acts:
                            val_act_max = 0.0
                            best_act = -1  # Mark as invalid
                        else:
                            val_act_max = max(val_acts)
                            best_act_idx = val_acts.index(val_act_max)
                            best_act = valid_acts_for_slot[best_act_idx]
                        new_pi_c.policy[idx-1][0][aidx][i] = best_act
                        additive_val += val_act_max * pi_c.dec_split[-1]
                    else:
                        # Case B: Slot is already fixed. Calculate its value contribution.
                        act = action_val
                        if use_qmdp:
                            current_val = self._compute_qmdp_action_value(act, aidx, oha, len_ohna, act_allohna, dist_arr, prob_arr, extra_horizon)
                        else:
                            current_val = sum(self.reward_list[ridx + act*self.a_prod[aidx]] * w for ridx, w in zip(reward_idxs, weights))
                        additive_val += current_val * pi_c.dec_split[-1]
                
                policyval += additive_val
                new_pi_c.heuristics = [policyval]
                if policyval > policyvalfound:
                    policyvalfound = policyval
                    q = [q_elt for q_elt in q if -q_elt[0] + TOLERANCE >= policyvalfound]
                    heapify(q)

                if abs(policyval - value) < EPSILON:
                    cent_vec = self.centralization_vector(new_pi_c) if init_call else []
                    return policyval, new_pi_c.policy, new_pi_c.clustering, cent_vec, new_pi_c.dists_cen, new_pi_c.clustering_cen
                if policyval + TOLERANCE >= policyvalfound:
                    heappush(q, (-policyval, -next(unique), new_pi_c))
            else: 
                if cidx:
                    if policyidx == -1 or policyidx >= len(pi_c.policy[idx-1][1]):
                        # Invalid policyidx - this shouldn't happen; log warning and skip
                        if self.output:
                            print(f"[WARNING] Invalid centralized policyidx={policyidx}, skipping node")
                        continue
                    
                    depth = idx - 1
                    rh_exact = h - depth
                    rh_total = rh_exact + extra_horizon

                    c_ctr = policyidx
                    init = pi_c.dists_cen[depth][c_ctr]
                    p_oh = pi_c.prob_cen[depth][c_ctr]

                    # Case 1: Zero probability or invalid belief
                    if init == -1 or p_oh < EPSILON:
                        new_pi_c = pi_c.policy_copy_cent(idx-1, policyidx)
                        # Append -1 to mark as invalid/unreachable
                        new_pi_c.policy[idx-1][1][policyidx].append(-1)
                        
                        # Heuristic doesn't change
                        policyval = min(value, min(pi_c.heuristics[pi_c.depth:])) if pi_c.depth is not None else math.inf
                        
                        if policyval + TOLERANCE >= policyvalfound:
                            heappush(q, (-policyval, -next(unique), new_pi_c))
                    
                    # Case 2: Valid belief, perform branching
                    else:
                        base_v = self.get_core_centralized_value(rh_total, init)
                        valid_actions = self.get_valid_actions_for_belief(init)

                        # Fallback if no valid actions (prevents dropping node)
                        if not valid_actions:
                            valid_actions = range(self.nactions)

                        # Pre-calculate the weight for heuristic updates
                        # (1 - dec_split) is the centralized mass at this depth
                        cen_weight = 1.0 - pi_c.dec_split[depth]
                        
                        for ja in valid_actions:
                            # 1. Compute Q-value (The expensive part, unavoidable)
                            q_val = self.exact_central_Q_sbt(rh_exact, init, ja, extra_horizon=extra_horizon)

                            # 2. Calculate potential heuristic value WITHOUT copying policy yet
                            potential_val = math.inf
                            
                            if pi_c.depth is not None:
                                # Calculate the change in value
                                delta = p_oh * (q_val - base_v)
                                weighted_delta = cen_weight * delta                               
                                min_h = math.inf
                                
                                # Check the updated range
                                for d in range(pi_c.depth, idx):
                                    h_new = pi_c.heuristics[d] + weighted_delta
                                    if h_new < min_h:
                                        min_h = h_new
                                
                                # Check the remaining range (if any exist and are valid)
                                # Usually heuristics list length is idx at this point.
                                for d in range(idx, len(pi_c.heuristics)):
                                    if pi_c.heuristics[d] < min_h:
                                        min_h = pi_c.heuristics[d]

                                potential_val = min(value, min_h)
                                
                                # Numerical stability check
                                if abs(potential_val - value) < EPSILON:
                                    potential_val = value
                            else:
                                # If depth is None, we haven't established a heuristic baseline yet
                                potential_val = value

                            # 3. PRUNING: Check if this branch is dead before allocation
                            if potential_val + TOLERANCE < policyvalfound:
                                continue 

                            # 4. Allocation: Node survived, now we copy
                            new_pi_c = pi_c.policy_copy_cent(idx-1, policyidx)
                            new_pi_c.policy[idx-1][1][policyidx].append(ja)

                            # Apply the heuristic update to the object
                            if pi_c.depth is not None and abs(weighted_delta) > EPSILON:
                                for d in range(pi_c.depth, idx):
                                    new_pi_c.heuristics[d] += weighted_delta
                            
                            heappush(q, (-potential_val, -next(unique), new_pi_c))
                            
                else: 
                    lena = self.nacts_factor[aidx]
                    valid_branch_acts = set()
                    
                    if self.use_action_masks:
                        depth = idx - 1
                        div, _ = cumprod(pi_c.ncluster[depth])
                        # Get all Joint OHs that map to this local cluster
                        Oh_change = lists_product2(aidx, [policyidx], pi_c.ncluster[depth], div, self.nagents)
                        
                        for oh in Oh_change:
                            init = pi_c.dists[depth][oh]
                            # Check if belief is valid and has probability mass
                            if init != -1 and pi_c.prob[depth][oh] > EPSILON:
                                valid_jas = self.get_valid_actions_for_belief(init)
                                for ja in valid_jas:
                                    # Decode joint action to single agent action
                                    # ja = ... + act * a_prod[aidx] + ...
                                    s_act = (ja // self.a_prod[aidx]) % lena
                                    valid_branch_acts.add(s_act)
                    
                    if not valid_branch_acts:
                        valid_branch_acts = set(range(lena))
                    
                    sorted_branch_acts = sorted(list(valid_branch_acts))
                    new_pi_cs = {}

                    for p in sorted_branch_acts:
                        pol_copy = pi_c.policy_copy(idx-1, aidx)
                        new_pi_cs[p] = pol_copy
                        new_pi_cs[p].policy[idx-1][0][aidx][policyidx] = p

                    if pi_c.depth is not None:
                        policydelta = [0]*(idx - pi_c.depth)
                        policydelta_new = {p: [0.0]*(idx - pi_c.depth) for p in sorted_branch_acts}

                        depth = idx - 1
                        rh = h - depth #  h - depth
                        d = depth - pi_c.depth

                        div, _ = cumprod(pi_c.ncluster[depth])
                        Oh_change = lists_product2(aidx, [policyidx], pi_c.ncluster[depth], div, self.nagents)
                        oh_as_set = {policyidx}
                        
                        if pi_c.depth != 0:
                            for oh in Oh_change:
                                init = pi_c.dists[depth][oh]
                                if init != -1:
                                    p_oh = pi_c.prob[depth][oh]
                                    short_ctr = 1
                                    skip_oh = False
                                    for a in range(aidx):
                                        action = pi_c.policy[depth][0][a][(oh//div[a])%pi_c.ncluster[depth][a]]
                                        if action < 0:
                                            skip_oh = True
                                            break
                                        short_ctr = short_ctr * self.maxa + action
                                    if skip_oh:
                                        continue
                                    heuristic = self.dec_heuristic.get((rh, init, short_ctr, (), extra_horizon))
                                    if heuristic is None:
                                        if short_ctr == 1:
                                            self.compute_heuristic_init(rh, init, extra_horizon=extra_horizon)
                                        else:
                                            self.compute_heuristic1(pi_c, oh, depth, rh, init, short_ctr, aidx, extra_horizon=extra_horizon)
                                        heuristic = self.dec_heuristic[(rh, init, short_ctr, (), extra_horizon)]
                                    policydelta[d] += p_oh * heuristic

                                    for p in sorted_branch_acts:
                                        short_ctr_new = short_ctr * self.maxa + p
                                        heuristic = self.dec_heuristic.get((rh, init, short_ctr_new, (), extra_horizon))
                                        if heuristic is None:
                                            self.compute_heuristic1(new_pi_cs[p], oh, depth, rh, init, short_ctr_new, aidx + 1, extra_horizon=extra_horizon)
                                            heuristic = self.dec_heuristic[(rh, init, short_ctr_new, (), extra_horizon)]
                                        policydelta_new[p][d] += p_oh * heuristic  
                        else:
                            short_ctr = 1
                            skip_depth = False
                            for a in range(aidx):
                                action = pi_c.policy[0][0][a][0]
                                if action < 0:
                                    skip_depth = True
                                    break
                                short_ctr = short_ctr * self.maxa + action

                            if not skip_depth:
                                heuristic = self.get_terminalMDP(0, rh, short_ctr)
                                policydelta[d] += heuristic

                                for p in sorted_branch_acts:
                                    short_ctr_new = short_ctr * self.maxa + p
                                    heuristic = self.get_terminalMDP(0, rh, short_ctr_new)
                                    policydelta_new[p][d] += heuristic

                        for depth in range(idx-2, pi_c.depth-1, -1): 
                            oh_asnew = set()
                            for oh_p in range(pi_c.ncluster[depth][aidx]):
                                for o in range(self.nobs_factor[aidx]):
                                    if pi_c.clustering[depth][aidx][oh_p][o] in oh_as_set:
                                        oh_asnew.add(oh_p)
                                        break
                            oh_as_set = oh_asnew
                            oh_as = list(oh_as_set)
                            
                            rh =  h - depth
                            d = depth - pi_c.depth 
                            div, _ = cumprod(pi_c.ncluster[depth])
                            Oh_change = lists_product2(aidx, oh_as, pi_c.ncluster[depth], div, self.nagents)
                            
                            for oh in Oh_change:
                                init = pi_c.dists[depth][oh]
                                if init != -1: 
                                    p_oh = pi_c.prob[depth][oh]
                                    short_ctr = self.compute_short_ctr(pi_c, oh, depth, div, idx)
                                    ctup = self.shorten_cluster(pi_c, oh, depth, return_ctr = True)
                                    key = (rh, init, short_ctr, ctup, extra_horizon)
                                    heuristic = self.dec_heuristic.get(key)
                                    if heuristic is None:
                                        prefix_key = (rh, init, short_ctr, ctup[:-1], extra_horizon)
                                        prefix_val = self.dec_heuristic.get(prefix_key)
                                        if prefix_val is not None:
                                            self.dec_heuristic[key] = prefix_val
                                            heuristic = prefix_val
                                        else:
                                            self.compute_heuristic2(pi_c, oh, depth, rh, init, short_ctr, ctup, aidx, extra_horizon=extra_horizon)
                                            heuristic = self.dec_heuristic[key]
                                    policydelta[d] += p_oh * heuristic

                                    for p in sorted_branch_acts:
                                        short_ctr_new = short_ctr * self.maxa + p
                                        heuristic = self.dec_heuristic.get((rh, init, short_ctr_new, ctup, extra_horizon))
                                        if heuristic is None:
                                            self.compute_heuristic2(new_pi_cs[p], oh, depth, rh, init, short_ctr_new, ctup, aidx, extra_horizon=extra_horizon)
                                            heuristic = self.dec_heuristic[(rh, init, short_ctr_new, ctup, extra_horizon)]
                                        policydelta_new[p][d] += p_oh * heuristic

                    for p in sorted_branch_acts:
                        if pi_c.depth is not None:
                            for depth_it in range(pi_c.depth, idx):
                                d = depth_it - pi_c.depth
                                w = (pi_c.dec_split[depth_it] if depth_it < len(pi_c.dec_split) else (pi_c.dec_split[-1] if pi_c.dec_split else 1.0))
                                new_pi_cs[p].heuristics[depth_it] = pi_c.heuristics[depth_it] + (policydelta_new[p][d] - policydelta[d]) * w

                            policyval = min(value, min(new_pi_cs[p].heuristics[new_pi_cs[p].depth:]))
                            if abs(policyval - value) < EPSILON: policyval = value
                        else: policyval = math.inf
                        
                        if pi_c.depth is None and aidx == self.nagents - 1 and policyidx == pi_c.ncluster[idx-1][aidx]-1:
                            joint_act = sum([new_pi_cs[p].policy[0][0][a][0]*self.a_prod[a] for a in range(self.nagents)])
                            sparse_transitions = self.get_terminal(new_pi_cs[p].dists[0][0], joint_act)
                            
                            # Reconstruct dense arrays for evaluate_policy_sbt (evaluate_policy_sbt still expects dense arrays for history lookup)
                            terminal_dists = [-1] * self.nobs
                            terminal_probs = [0.0] * self.nobs
                            
                            for o, prob, d_next in sparse_transitions:
                                terminal_dists[o] = d_next
                                terminal_probs[o] = prob

                            if new_pi_cs[p].policy[0][1] != []:
                                sparse_transitions_cen = self.get_terminal(new_pi_cs[p].dists_cen[0][0], new_pi_cs[p].policy[0][1][0][0])
                                # Reconstruct dense arrays for centralized terminal
                                terminal_dists_cen = [-1] * self.nobs
                                terminal_probs_cen = [0.0] * self.nobs
                                for o, prob, d_next in sparse_transitions_cen:
                                    terminal_dists_cen[o] = d_next
                                    terminal_probs_cen[o] = prob
                            else:
                                terminal_dists_cen = [-1] * self.nobs
                                terminal_probs_cen = [0.0] * self.nobs
                            new_pi_cs[p].values = new_pi_cs[p].values.copy()

                            policyval = self.evaluate_policy_sbt(new_pi_cs[p], terminal_dec = True, probs_terminal = terminal_probs, dists_terminal = terminal_dists, 
                                                                probs_terminal_cen = terminal_probs_cen, dists_terminal_cen = terminal_dists_cen, rh = h-idx, extra_horizon = extra_horizon) 

                        if policyval + TOLERANCE >= policyvalfound:
                            heappush(q, (-policyval, -next(unique), new_pi_cs[p]))



            # TI1: Adaptive check frequency based on iter_limit
            ti1_check_freq = max(self.adaptive_check, int(self.iter_limit / 100)) if self.iter_limit != math.inf else 30
            
            if self.algorithm == "approximate" and self.TI1 and init_call and (ctr > max(self.nacts_factor) + 1 and ctr % ti1_check_freq == 0):

                # 1. Get Consensus from the "Crowd" (Top N nodes)
                active_mask = self.get_horizon_centralization_scores(q, top_n=self.score_limit, threshold=self.cen_threshold, temperature=self.sm_temperature)
                
                # 2. Check if a trigger exists (we usually ignore stage 0 as it is the root)
                if active_mask[0] and any(active_mask[0][1:]):
                    
                    # Identify stages that MUST be centralized according to voting
                    forced_central_stages = [i for i, must_cen in enumerate(active_mask[0]) if must_cen and i > 0]
                    
                    if forced_central_stages:
                        # --- STEP A: PRUNING ---
                        # Remove nodes that explicitly violate the consensus.
                        # This clears the way for the best Centralized node to rise to the top.
                        new_q = []
                        pruned_count = 0
                        
                        for item in q:
                            policy_obj = item[2]
                            pol_vec = self.centralization_vector(policy_obj)
                            
                            keep_node = True
                            for stage_idx in forced_central_stages:
                                # Only prune if the stage is fully expanded and decentralized. Keep incomplete (None) stages to allow them to evolve
                                if stage_idx < len(pol_vec) and pol_vec[stage_idx] is not None:
                                    if pol_vec[stage_idx] == False: 
                                        keep_node = False
                                        break
                            
                            if keep_node:
                                new_q.append(item)
                            else:
                                pruned_count += 1
                        
                        # Rebuild the heap if we pruned anything
                        if len(new_q) > 0:
                            heapify(new_q)
                            q = new_q
                            if self.output:
                                print(f"TI1: Pruned {pruned_count} decentralized nodes. Queue size: {len(q)}")
                        
                        # --- STEP B: CHECK TOP NODE FOR EXIT ---
                        # Look at the winner (Top of Heap). If the winner complies with the trigger stage, Exit. 
                        # This preserves the Interleaving architecture but guarantees the prefix is optimal.
                        
                        if q:
                            top_val, _, top_policy = q[0]
                            trigger_stage = forced_central_stages[0] # Focus on the earliest trigger
                            
                            # Check if the top policy is fully expanded at the trigger stage
                            if self.is_stage_complete(top_policy, trigger_stage):
                                top_vec = self.centralization_vector(top_policy)
                                
                                # Verify it actually centralized (it should, given the pruning)
                                if trigger_stage < len(top_vec) and top_vec[trigger_stage]:
                                    if self.output:
                                        print(f"TI1: Top node ready at stage {trigger_stage}. Interleaving execution.")
                                    return -top_val, top_policy.policy, top_policy.clustering, top_vec, top_policy.dists_cen, top_policy.clustering_cen