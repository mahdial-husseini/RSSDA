"""
Fire Fighting Domain (3 Houses) for Semi-Decentralized Multi-Agent Planning

This module implements the fire fighting benchmark domain for evaluating
the RSSDA algorithm. Two fire fighters coordinate to extinguish fires
across three houses with partial observability. Requires fireFighting233.data.

Domain Description:
    - Three houses that may be on fire (8 fire configurations)
    - Two agents that can move between houses
    - Each agent starts at a designated position
    - Agents observe flames/no-flames at their current location
    - Goal: Extinguish all fires with coordinated action

State Space:
    - Fire status of 3 houses: 2^3 = 8 fire configurations
    - Agent 1 position: 4 locations (h1, h2, h3, start)
    - Agent 2 position: 4 locations (h1, h2, h3, start)
    - Total: 8 * 4 * 4 = 128 base states (432 with additional encoding)

Actions (per agent):
    - go1: Move to house 1
    - go2: Move to house 2
    - go3: Move to house 3

Observations (per agent):
    - flames: Agent observes fire at current location
    - noFlames: Agent observes no fire at current location

Synchronization Triggers:
    Action-based triggers: Agents synchronize when both choose the same
    destination (go1+go1, go2+go2, go3+go3).

Usage:
    python sdec_fireFight3houses.py <horizon> [maxit] [IEmin2] [alpha]

Reference:
    Oliehoek et al., "Optimal and Approximate Q-value Functions for
    Decentralized POMDPs"

Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

import sys
import os
import time
import math
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
TRIGGER_MODE = "semi"   # "centralized", "semi", "decentralized", or "decentralized_RSMAA"
                                       # decentralized_RSMAA uses original decPOMDP.py (RS-MAA* algorithm)

# --- Core Algorithm ---
ALGORITHM = "exact"             # "exact" or "approximate" (enables TI approximations)

# --- Heuristic Type ---
# Controls upper-bound heuristic for A* search guidance.
#   "QMDP"   - Loose/fast: ignores observations (dot product of belief and state values)
#   "POMDP"  - Tight/exact: full centralized POMDP value function
#   "HYBRID" - Runs exact POMDP for first HYBRID_R steps, then QMDP
# Rule of thumb: Use "POMDP" for exact algorithm, "QMDP" or "HYBRID" for approximate.
HEURISTIC_TYPE = "POMDP"
TAIL_HEURISTIC_TYPE = "QMDP"
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
REC_LIMIT = 2

# --- TI4: Max Clustering ---
# Cluster into MAX_Clusters based on combination of L1 distance between resulting beliefs and probability mass
MAX_CLUSTERS = 2

# ============================================================================
#                        END USER CONFIGURATION
# ============================================================================


# ==========================================
# Problem Configuration
# ==========================================

class FireFightConfig:
    """Holds configuration for the Fire Fighting problem execution."""

    # Problem constants
    NAGENTS = 2
    NSTATES = 432
    ACT_PER_AGENT = 3
    OBS_PER_AGENT = 2

    # Trigger definitions (action-based)
    # Format: [list of joint action indices, list of individual action pairs]
    TRIG_CENTRALIZED = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]]
    TRIG_SEMI = [[0, 4, 8], [[0, 0], [1, 1], [2, 2]]]  # Same destination = sync
    TRIG_DECENTRAL = [[], []]

    def __init__(self):
        # CLI argument parsing (uses USER CONFIGURATION as defaults)
        self.horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        self.maxit = int(sys.argv[2]) if len(sys.argv) > 2 else MAXIT
        self.ie_min2 = int(sys.argv[3]) if len(sys.argv) > 3 else IE_MIN2
        self.alpha = float(sys.argv[4]) if len(sys.argv) > 4 else ALPHA

        # Derived constants
        self.nacts = self.ACT_PER_AGENT ** self.NAGENTS
        self.nobs = self.OBS_PER_AGENT ** self.NAGENTS
        self.nacts_factor = [self.ACT_PER_AGENT] * self.NAGENTS
        self.nobs_factor = [self.OBS_PER_AGENT] * self.NAGENTS
        self.nsq = self.NSTATES ** 2
        self.nso = self.NSTATES * self.nobs

        # Set trigger based on TRIGGER_MODE
        if TRIGGER_MODE == "centralized":
            self.active_trigger = self.TRIG_CENTRALIZED
        elif TRIGGER_MODE == "semi":
            self.active_trigger = self.TRIG_SEMI
        else:  # decentralized
            self.active_trigger = self.TRIG_DECENTRAL


class FireFightProblemLoader:
    """Parses fireFighting233.data and structures matrices."""

    def __init__(self, config):
        self.config = config

        # Action and observation dictionaries
        self.act_dict = {"go1": 0, "go2": 1, "go3": 2}
        self.obs_dict = {"flames": 0, "noFlames": 1}

        # Build state dictionary from state string
        states_str = "f0_f0_f0_h1_h1 f0_f0_f0_h1_h2 f0_f0_f0_h1_h3 f0_f0_f0_h1_start f0_f0_f0_h2_h1 f0_f0_f0_h2_h2 f0_f0_f0_h2_h3 f0_f0_f0_h2_start f0_f0_f0_h3_h1 f0_f0_f0_h3_h2 f0_f0_f0_h3_h3 f0_f0_f0_h3_start f0_f0_f0_start_h1 f0_f0_f0_start_h2 f0_f0_f0_start_h3 f0_f0_f0_start_start f0_f0_f1_h1_h1 f0_f0_f1_h1_h2 f0_f0_f1_h1_h3 f0_f0_f1_h1_start f0_f0_f1_h2_h1 f0_f0_f1_h2_h2 f0_f0_f1_h2_h3 f0_f0_f1_h2_start f0_f0_f1_h3_h1 f0_f0_f1_h3_h2 f0_f0_f1_h3_h3 f0_f0_f1_h3_start f0_f0_f1_start_h1 f0_f0_f1_start_h2 f0_f0_f1_start_h3 f0_f0_f1_start_start f0_f0_f2_h1_h1 f0_f0_f2_h1_h2 f0_f0_f2_h1_h3 f0_f0_f2_h1_start f0_f0_f2_h2_h1 f0_f0_f2_h2_h2 f0_f0_f2_h2_h3 f0_f0_f2_h2_start f0_f0_f2_h3_h1 f0_f0_f2_h3_h2 f0_f0_f2_h3_h3 f0_f0_f2_h3_start f0_f0_f2_start_h1 f0_f0_f2_start_h2 f0_f0_f2_start_h3 f0_f0_f2_start_start f0_f1_f0_h1_h1 f0_f1_f0_h1_h2 f0_f1_f0_h1_h3 f0_f1_f0_h1_start f0_f1_f0_h2_h1 f0_f1_f0_h2_h2 f0_f1_f0_h2_h3 f0_f1_f0_h2_start f0_f1_f0_h3_h1 f0_f1_f0_h3_h2 f0_f1_f0_h3_h3 f0_f1_f0_h3_start f0_f1_f0_start_h1 f0_f1_f0_start_h2 f0_f1_f0_start_h3 f0_f1_f0_start_start f0_f1_f1_h1_h1 f0_f1_f1_h1_h2 f0_f1_f1_h1_h3 f0_f1_f1_h1_start f0_f1_f1_h2_h1 f0_f1_f1_h2_h2 f0_f1_f1_h2_h3 f0_f1_f1_h2_start f0_f1_f1_h3_h1 f0_f1_f1_h3_h2 f0_f1_f1_h3_h3 f0_f1_f1_h3_start f0_f1_f1_start_h1 f0_f1_f1_start_h2 f0_f1_f1_start_h3 f0_f1_f1_start_start f0_f1_f2_h1_h1 f0_f1_f2_h1_h2 f0_f1_f2_h1_h3 f0_f1_f2_h1_start f0_f1_f2_h2_h1 f0_f1_f2_h2_h2 f0_f1_f2_h2_h3 f0_f1_f2_h2_start f0_f1_f2_h3_h1 f0_f1_f2_h3_h2 f0_f1_f2_h3_h3 f0_f1_f2_h3_start f0_f1_f2_start_h1 f0_f1_f2_start_h2 f0_f1_f2_start_h3 f0_f1_f2_start_start f0_f2_f0_h1_h1 f0_f2_f0_h1_h2 f0_f2_f0_h1_h3 f0_f2_f0_h1_start f0_f2_f0_h2_h1 f0_f2_f0_h2_h2 f0_f2_f0_h2_h3 f0_f2_f0_h2_start f0_f2_f0_h3_h1 f0_f2_f0_h3_h2 f0_f2_f0_h3_h3 f0_f2_f0_h3_start f0_f2_f0_start_h1 f0_f2_f0_start_h2 f0_f2_f0_start_h3 f0_f2_f0_start_start f0_f2_f1_h1_h1 f0_f2_f1_h1_h2 f0_f2_f1_h1_h3 f0_f2_f1_h1_start f0_f2_f1_h2_h1 f0_f2_f1_h2_h2 f0_f2_f1_h2_h3 f0_f2_f1_h2_start f0_f2_f1_h3_h1 f0_f2_f1_h3_h2 f0_f2_f1_h3_h3 f0_f2_f1_h3_start f0_f2_f1_start_h1 f0_f2_f1_start_h2 f0_f2_f1_start_h3 f0_f2_f1_start_start f0_f2_f2_h1_h1 f0_f2_f2_h1_h2 f0_f2_f2_h1_h3 f0_f2_f2_h1_start f0_f2_f2_h2_h1 f0_f2_f2_h2_h2 f0_f2_f2_h2_h3 f0_f2_f2_h2_start f0_f2_f2_h3_h1 f0_f2_f2_h3_h2 f0_f2_f2_h3_h3 f0_f2_f2_h3_start f0_f2_f2_start_h1 f0_f2_f2_start_h2 f0_f2_f2_start_h3 f0_f2_f2_start_start f1_f0_f0_h1_h1 f1_f0_f0_h1_h2 f1_f0_f0_h1_h3 f1_f0_f0_h1_start f1_f0_f0_h2_h1 f1_f0_f0_h2_h2 f1_f0_f0_h2_h3 f1_f0_f0_h2_start f1_f0_f0_h3_h1 f1_f0_f0_h3_h2 f1_f0_f0_h3_h3 f1_f0_f0_h3_start f1_f0_f0_start_h1 f1_f0_f0_start_h2 f1_f0_f0_start_h3 f1_f0_f0_start_start f1_f0_f1_h1_h1 f1_f0_f1_h1_h2 f1_f0_f1_h1_h3 f1_f0_f1_h1_start f1_f0_f1_h2_h1 f1_f0_f1_h2_h2 f1_f0_f1_h2_h3 f1_f0_f1_h2_start f1_f0_f1_h3_h1 f1_f0_f1_h3_h2 f1_f0_f1_h3_h3 f1_f0_f1_h3_start f1_f0_f1_start_h1 f1_f0_f1_start_h2 f1_f0_f1_start_h3 f1_f0_f1_start_start f1_f0_f2_h1_h1 f1_f0_f2_h1_h2 f1_f0_f2_h1_h3 f1_f0_f2_h1_start f1_f0_f2_h2_h1 f1_f0_f2_h2_h2 f1_f0_f2_h2_h3 f1_f0_f2_h2_start f1_f0_f2_h3_h1 f1_f0_f2_h3_h2 f1_f0_f2_h3_h3 f1_f0_f2_h3_start f1_f0_f2_start_h1 f1_f0_f2_start_h2 f1_f0_f2_start_h3 f1_f0_f2_start_start f1_f1_f0_h1_h1 f1_f1_f0_h1_h2 f1_f1_f0_h1_h3 f1_f1_f0_h1_start f1_f1_f0_h2_h1 f1_f1_f0_h2_h2 f1_f1_f0_h2_h3 f1_f1_f0_h2_start f1_f1_f0_h3_h1 f1_f1_f0_h3_h2 f1_f1_f0_h3_h3 f1_f1_f0_h3_start f1_f1_f0_start_h1 f1_f1_f0_start_h2 f1_f1_f0_start_h3 f1_f1_f0_start_start f1_f1_f1_h1_h1 f1_f1_f1_h1_h2 f1_f1_f1_h1_h3 f1_f1_f1_h1_start f1_f1_f1_h2_h1 f1_f1_f1_h2_h2 f1_f1_f1_h2_h3 f1_f1_f1_h2_start f1_f1_f1_h3_h1 f1_f1_f1_h3_h2 f1_f1_f1_h3_h3 f1_f1_f1_h3_start f1_f1_f1_start_h1 f1_f1_f1_start_h2 f1_f1_f1_start_h3 f1_f1_f1_start_start f1_f1_f2_h1_h1 f1_f1_f2_h1_h2 f1_f1_f2_h1_h3 f1_f1_f2_h1_start f1_f1_f2_h2_h1 f1_f1_f2_h2_h2 f1_f1_f2_h2_h3 f1_f1_f2_h2_start f1_f1_f2_h3_h1 f1_f1_f2_h3_h2 f1_f1_f2_h3_h3 f1_f1_f2_h3_start f1_f1_f2_start_h1 f1_f1_f2_start_h2 f1_f1_f2_start_h3 f1_f1_f2_start_start f1_f2_f0_h1_h1 f1_f2_f0_h1_h2 f1_f2_f0_h1_h3 f1_f2_f0_h1_start f1_f2_f0_h2_h1 f1_f2_f0_h2_h2 f1_f2_f0_h2_h3 f1_f2_f0_h2_start f1_f2_f0_h3_h1 f1_f2_f0_h3_h2 f1_f2_f0_h3_h3 f1_f2_f0_h3_start f1_f2_f0_start_h1 f1_f2_f0_start_h2 f1_f2_f0_start_h3 f1_f2_f0_start_start f1_f2_f1_h1_h1 f1_f2_f1_h1_h2 f1_f2_f1_h1_h3 f1_f2_f1_h1_start f1_f2_f1_h2_h1 f1_f2_f1_h2_h2 f1_f2_f1_h2_h3 f1_f2_f1_h2_start f1_f2_f1_h3_h1 f1_f2_f1_h3_h2 f1_f2_f1_h3_h3 f1_f2_f1_h3_start f1_f2_f1_start_h1 f1_f2_f1_start_h2 f1_f2_f1_start_h3 f1_f2_f1_start_start f1_f2_f2_h1_h1 f1_f2_f2_h1_h2 f1_f2_f2_h1_h3 f1_f2_f2_h1_start f1_f2_f2_h2_h1 f1_f2_f2_h2_h2 f1_f2_f2_h2_h3 f1_f2_f2_h2_start f1_f2_f2_h3_h1 f1_f2_f2_h3_h2 f1_f2_f2_h3_h3 f1_f2_f2_h3_start f1_f2_f2_start_h1 f1_f2_f2_start_h2 f1_f2_f2_start_h3 f1_f2_f2_start_start f2_f0_f0_h1_h1 f2_f0_f0_h1_h2 f2_f0_f0_h1_h3 f2_f0_f0_h1_start f2_f0_f0_h2_h1 f2_f0_f0_h2_h2 f2_f0_f0_h2_h3 f2_f0_f0_h2_start f2_f0_f0_h3_h1 f2_f0_f0_h3_h2 f2_f0_f0_h3_h3 f2_f0_f0_h3_start f2_f0_f0_start_h1 f2_f0_f0_start_h2 f2_f0_f0_start_h3 f2_f0_f0_start_start f2_f0_f1_h1_h1 f2_f0_f1_h1_h2 f2_f0_f1_h1_h3 f2_f0_f1_h1_start f2_f0_f1_h2_h1 f2_f0_f1_h2_h2 f2_f0_f1_h2_h3 f2_f0_f1_h2_start f2_f0_f1_h3_h1 f2_f0_f1_h3_h2 f2_f0_f1_h3_h3 f2_f0_f1_h3_start f2_f0_f1_start_h1 f2_f0_f1_start_h2 f2_f0_f1_start_h3 f2_f0_f1_start_start f2_f0_f2_h1_h1 f2_f0_f2_h1_h2 f2_f0_f2_h1_h3 f2_f0_f2_h1_start f2_f0_f2_h2_h1 f2_f0_f2_h2_h2 f2_f0_f2_h2_h3 f2_f0_f2_h2_start f2_f0_f2_h3_h1 f2_f0_f2_h3_h2 f2_f0_f2_h3_h3 f2_f0_f2_h3_start f2_f0_f2_start_h1 f2_f0_f2_start_h2 f2_f0_f2_start_h3 f2_f0_f2_start_start f2_f1_f0_h1_h1 f2_f1_f0_h1_h2 f2_f1_f0_h1_h3 f2_f1_f0_h1_start f2_f1_f0_h2_h1 f2_f1_f0_h2_h2 f2_f1_f0_h2_h3 f2_f1_f0_h2_start f2_f1_f0_h3_h1 f2_f1_f0_h3_h2 f2_f1_f0_h3_h3 f2_f1_f0_h3_start f2_f1_f0_start_h1 f2_f1_f0_start_h2 f2_f1_f0_start_h3 f2_f1_f0_start_start f2_f1_f1_h1_h1 f2_f1_f1_h1_h2 f2_f1_f1_h1_h3 f2_f1_f1_h1_start f2_f1_f1_h2_h1 f2_f1_f1_h2_h2 f2_f1_f1_h2_h3 f2_f1_f1_h2_start f2_f1_f1_h3_h1 f2_f1_f1_h3_h2 f2_f1_f1_h3_h3 f2_f1_f1_h3_start f2_f1_f1_start_h1 f2_f1_f1_start_h2 f2_f1_f1_start_h3 f2_f1_f1_start_start f2_f1_f2_h1_h1 f2_f1_f2_h1_h2 f2_f1_f2_h1_h3 f2_f1_f2_h1_start f2_f1_f2_h2_h1 f2_f1_f2_h2_h2 f2_f1_f2_h2_h3 f2_f1_f2_h2_start f2_f1_f2_h3_h1 f2_f1_f2_h3_h2 f2_f1_f2_h3_h3 f2_f1_f2_h3_start f2_f1_f2_start_h1 f2_f1_f2_start_h2 f2_f1_f2_start_h3 f2_f1_f2_start_start f2_f2_f0_h1_h1 f2_f2_f0_h1_h2 f2_f2_f0_h1_h3 f2_f2_f0_h1_start f2_f2_f0_h2_h1 f2_f2_f0_h2_h2 f2_f2_f0_h2_h3 f2_f2_f0_h2_start f2_f2_f0_h3_h1 f2_f2_f0_h3_h2 f2_f2_f0_h3_h3 f2_f2_f0_h3_start f2_f2_f0_start_h1 f2_f2_f0_start_h2 f2_f2_f0_start_h3 f2_f2_f0_start_start f2_f2_f1_h1_h1 f2_f2_f1_h1_h2 f2_f2_f1_h1_h3 f2_f2_f1_h1_start f2_f2_f1_h2_h1 f2_f2_f1_h2_h2 f2_f2_f1_h2_h3 f2_f2_f1_h2_start f2_f2_f1_h3_h1 f2_f2_f1_h3_h2 f2_f2_f1_h3_h3 f2_f2_f1_h3_start f2_f2_f1_start_h1 f2_f2_f1_start_h2 f2_f2_f1_start_h3 f2_f2_f1_start_start f2_f2_f2_h1_h1 f2_f2_f2_h1_h2 f2_f2_f2_h1_h3 f2_f2_f2_h1_start f2_f2_f2_h2_h1 f2_f2_f2_h2_h2 f2_f2_f2_h2_h3 f2_f2_f2_h2_start f2_f2_f2_h3_h1 f2_f2_f2_h3_h2 f2_f2_f2_h3_h3 f2_f2_f2_h3_start f2_f2_f2_start_h1 f2_f2_f2_start_h2 f2_f2_f2_start_h3 f2_f2_f2_start_start"
        self.state_list = states_str.split()
        self.state_dict = {s: i for i, s in enumerate(self.state_list)}

        # Initialize matrices
        self.transit = [0.0] * (config.nsq * config.nacts)
        self.obs = [0.0] * (config.nso * config.nacts)
        self.reward = [0.0] * (config.NSTATES * config.nacts)
        self.init_beliefs = [0.0] * config.NSTATES

        # Initial belief: uniform over start_start states
        for i, state_name in enumerate(self.state_list):
            if "start_start" in state_name:
                self.init_beliefs[i] = 1.0 / 27  # 27 fire configurations with start_start

    def load_data(self, filename="fireFighting233.data"):
        """Load transition, observation, and reward data from file."""
        c = self.config

        # Use script directory for data file path
        filepath = os.path.join(_script_dir, filename)
        with open(filepath, "r") as data:
            for line in data:
                d = line.split()
                if not d:
                    continue

                if d[0][0] == "T":
                    # T act1 act2 : start_state : end_state : prob
                    act = self.act_dict[d[1]] + c.ACT_PER_AGENT * self.act_dict[d[2]]
                    s = self.state_dict[d[4]]
                    snew = self.state_dict[d[6]]
                    self.transit[act * c.nsq + s * c.NSTATES + snew] = float(d[8])

                elif d[0][0] == "O":
                    # O : state : obs1 obs2 : prob
                    s = self.state_dict[d[3]]
                    o = self.obs_dict[d[5]] + c.OBS_PER_AGENT * self.obs_dict[d[6]]
                    p = float(d[8])
                    # Observation is action-independent
                    for act in range(c.nacts):
                        self.obs[act * c.nso + s * c.nobs + o] = p

                elif d[0][0] == "R":
                    # R : : : end_state : : : : reward
                    snew = self.state_dict[d[5]]
                    rw = float(d[9])
                    # Reward depends on reaching state snew
                    for act in range(c.nacts):
                        for s in range(c.NSTATES):
                            self.reward[act * c.NSTATES + s] += rw * self.transit[act * c.nsq + s * c.NSTATES + snew]

        return self.transit, self.obs, self.reward, self.init_beliefs


# ==========================================
# Decentralized Mode (using original decPOMDP.py / RS-MAA*)
# ==========================================

def run_firefight_decentralized_rsmaa(config, verbose=True):
    """
    Run the Fire Fighting problem using the original decPOMDP.py algorithm (RS-MAA*).
    This is the fully decentralized mode with no sync triggers.
    """
    if OriginalDecPOMDP is None:
        print("Original decPOMDP solver not available. Exiting.")
        return 0

    time_start = time.time()

    # 1. Load Problem Data
    loader = FireFightProblemLoader(config)
    T, O, R, init_b = loader.load_data()

    time_parse = time.time()

    # 2. Convert to pdict format required by decPOMDP.py
    # decPOMDP.py expects flat lists: transitions[act*nstates + s] = (indices, probs)
    T_pdict = []
    for act in range(config.nacts):
        for s in range(config.NSTATES):
            indices = []
            values = []
            for snew in range(config.NSTATES):
                val = T[act * config.nsq + s * config.NSTATES + snew]
                if val > 0:
                    indices.append(snew)
                    values.append(val)
            T_pdict.append((array('i', indices), array('d', values)))

    # obs[act*nstates + snew] = (obs_indices, probs)
    O_pdict = []
    for act in range(config.nacts):
        for snew in range(config.NSTATES):
            indices = []
            values = []
            for o in range(config.nobs):
                val = O[act * config.nso + snew * config.nobs + o]
                if val > 0:
                    indices.append(o)
                    values.append(val)
            O_pdict.append((array('i', indices), array('d', values)))

    if verbose:
        print(f"FireFight Problem | States: {config.NSTATES} | Actions: {config.nacts} | Obs: {config.nobs}")
        print(f"Time (parsing): {time_parse - time_start:.3f}s")

    # 3. Create original DecPOMDP solver (replicates exact optimal performance)
    dec_pomdp = OriginalDecPOMDP(
        nagents=config.NAGENTS,
        nstates=config.NSTATES,
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
    time_solve_start = time.time()
    try:
        val, _, _ = dec_pomdp.multi_agent_astar(config.horizon)
        time_solve_end = time.time()

        print(f"Result: {val}")
        print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
        print(f"Time (total): {time_solve_end - time_start:.3f}s")

        return val
    except DecPOMDPMemoryLimitExceeded as e:
        time_solve_end = time.time()
        print(f"Result: MO")
        print(f"Memory limit exceeded: {e}")
        print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
        print(f"Time (total): {time_solve_end - time_start:.3f}s")
        return "MO"


# ==========================================
# Main Execution (RSSDA)
# ==========================================

def run_firefight_rssda(config):
    """Run the Fire Fighting problem using RSSDA."""
    time_start = time.time()

    # 1. Load Problem Data
    loader = FireFightProblemLoader(config)
    T, O, R, init_b = loader.load_data()

    time_parse = time.time()

    # 2. Extract action triggers (joint action indices)
    action_triggers = config.active_trigger[0]

    # 3. Initialize Model
    model = SDecPOMDPModel(
        nagents=config.NAGENTS,
        nstates=config.NSTATES,
        nactions=config.nacts,
        nobs=config.nobs,
        transitions=T,
        obs=O,
        rewards=R,
        init_beliefs=init_b,
        nacts_factor=config.nacts_factor,
        nobs_factor=config.nobs_factor,
        sync_states=[],
        sync_actions=action_triggers,
        sync_observations=[]
    )

    # 4. Initialize Solver Config (uses top-level USER CONFIGURATION constants)
    solver_config = RSSDAConfig(
        maxh=config.horizon,
        maxit=config.maxit,
        IEmin2=config.ie_min2,
        alpha=config.alpha,
        algorithm=ALGORITHM,
        heuristic_type=HEURISTIC_TYPE,
        tail_heuristic_type=TAIL_HEURISTIC_TYPE,
        TI1=TI1,
        TI2=TI2,
        TI3=TI3,
        TI4=TI4,
        score_limit=SCORE_LIMIT,
        cen_threshold=CEN_THRESHOLD,
        sm_temperature=SM_TEMPERATURE,
        iter_limit=ITER_LIMIT,
        rec_limit=REC_LIMIT,
        hybrid_r=HYBRID_R,
        max_clusters=MAX_CLUSTERS
    )

    # 5. Initialize Solver
    sdec_pomdp = SDecPOMDP(model=model, config=solver_config)

    # 6. Solve
    time_solve_start = time.time()
    try:
        result = sdec_pomdp.multi_agent_astar(config.horizon)
        time_solve_end = time.time()

        # 7. Output Results
        print(f"Result: {result[0]}")
        print(f"Time (parsing): {time_parse - time_start:.3f}s")
        print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
        print(f"Time (total): {time_solve_end - time_start:.3f}s")

        return result[0]
    except MemoryLimitExceeded as e:
        time_solve_end = time.time()
        print(f"Result: MO")
        print(f"Memory limit exceeded: {e}")
        print(f"Time (parsing): {time_parse - time_start:.3f}s")
        print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
        print(f"Time (total): {time_solve_end - time_start:.3f}s")
        return "MO"


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    config = FireFightConfig()

    if TRIGGER_MODE == "decentralized_RSMAA":
        print("Running in DECENTRALIZED mode (RS-MAA* via decPOMDP.py)")
        run_firefight_decentralized_rsmaa(config)
    else:
        run_firefight_rssda(config)
