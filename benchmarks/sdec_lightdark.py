"""
LightDark1D Domain for POMDP Planning

This module implements a discrete adaptation of the classic LightDark1D POMDP
problem for evaluating the RSSDA algorithm. An agent navigates a 1D space
to localize and commit to the goal at the origin, with observation noise that
depends on position - minimal noise at the "light" region (y=5) and increasing
noise as distance from light increases.

Problem Formulation (following JuliaPOMDP/POMDPModels.jl):
    The agent starts with uncertain position (initially around y=2) and must
    reach the goal at y=0 with high confidence before committing. The key
    insight is that observation noise σ(y) = |y - 5|/√2 + ε, meaning the agent
    must often move AWAY from the goal toward y=5 to reduce uncertainty, then
    return to commit at y=0.

State Space:
    - Discrete positions: -GRID_MIN to +GRID_MAX (centered around origin)
    - Terminal state (after COMMIT action)

Actions:
    - LEFT (-1): Move one step toward negative y
    - COMMIT (0): Commit to current position as goal location
    - RIGHT (+1): Move one step toward positive y

Observations:
    - Noisy position measurement
    - Noise follows: σ(y) = |y - LIGHT_POSITION|/√2 + MIN_NOISE
    - Discretized to integer position observations

Rewards:
    - COMMIT within ±1 of goal (y=0): +10
    - COMMIT elsewhere: -10
    - Movement: -movement_cost (default 0)
    - Terminal state: 0

Reference:
    Platt et al., "Belief space planning assuming maximum likelihood observations"
    Robotics: Science and Systems, 2010
    https://groups.csail.mit.edu/robotics-center/public_papers/Platt10.pdf

    JuliaPOMDP LightDark implementation:
    https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/LightDark.jl

Author: [Mahdi Al-Husseini]
License: MIT  (https://opensource.org/license/mit/)
"""

import sys
import os
import time
import math
import numpy as np
from array import array
from scipy import stats

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

try:
    from decPOMDP import DecPOMDP as OriginalDecPOMDP, MemoryLimitExceeded as DecPOMDPMemoryLimitExceeded
except ImportError:
    print("Warning: Could not import original decPOMDP. decpomdp mode may not work.")
    OriginalDecPOMDP = None
    DecPOMDPMemoryLimitExceeded = None

# ============================================================================
#                           PROBLEM PARAMETERS
# ============================================================================
# These follow the original LightDark1D formulation from JuliaPOMDP
# ============================================================================

# --- Grid Discretization ---
GRID_MIN = -5                   # Minimum position (y)
GRID_MAX = 10                   # Maximum position (y)
STEP_SIZE = 1                   # Movement step size (matching original)

# --- Light Region ---
LIGHT_POSITION = 5              # Position of minimum noise (original: y=5)
MIN_NOISE = 0.01                # Minimum noise at light position (original: 0.01)
SQRT2 = math.sqrt(2)            # σ(y) = |y - 5|/√2 + 0.01

# --- Goal ---
GOAL_POSITION = 0               # Goal is at origin (y=0)
GOAL_RADIUS = 1                 # Commit is correct if |y - goal| < radius

# --- Rewards (original formulation) ---
CORRECT_REWARD = 10.0           # Reward for committing within goal radius
INCORRECT_PENALTY = -10.0       # Penalty for committing outside goal radius
MOVEMENT_COST = 0.0             # Cost per movement action (original: 0)

# --- Initial Distribution ---
INIT_MEAN = 2.0                 # Mean of initial belief (original: 2)
INIT_STD = 3.0                  # Std dev of initial belief (original: 3)

# --- Discount ---
DISCOUNT = 0.9                  # Discount factor (original: 0.9)

# ============================================================================
#                           SOLVER CONFIGURATION
# ============================================================================

# --- Algorithm Mode ---
TRIGGER_MODE = "centralized"    # Single agent, so mode doesn't affect behavior
ALGORITHM = "approximate"             # "exact" or "approximate"

# --- Heuristic Type ---
HEURISTIC_TYPE = "HYBRID"        # "QMDP", "POMDP", or "HYBRID"
TAIL_HEURISTIC_TYPE = "HYBRID"
HYBRID_R = 1

# --- Search Parameters ---
MAXIT = 200
ALPHA = 0.2
IE_MIN2 = 3

# --- Approximation Techniques ---
TI1 = False
TI2 = True
TI3 = True
TI4 = True
SCORE_LIMIT = 20
CEN_THRESHOLD = 0.6
SM_TEMPERATURE = 0.6
ITER_LIMIT = 1000
REC_LIMIT = 1
MAX_CLUSTERS = 3

# ============================================================================
#                        END CONFIGURATION
# ============================================================================


class LightDarkConfig:
    """Configuration for the discrete LightDark1D problem."""

    def __init__(self):
        # CLI argument parsing
        self.horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        self.maxit = int(sys.argv[2]) if len(sys.argv) > 2 else MAXIT
        self.ie_min2 = int(sys.argv[3]) if len(sys.argv) > 3 else IE_MIN2
        self.alpha = float(sys.argv[4]) if len(sys.argv) > 4 else ALPHA

        # Single agent
        self.nagents = 1

        # Discrete positions: GRID_MIN to GRID_MAX inclusive
        self.grid_min = GRID_MIN
        self.grid_max = GRID_MAX
        self.grid_size = GRID_MAX - GRID_MIN + 1  # Number of position states

        # State space: positions + terminal state
        self.nstates = self.grid_size + 1
        self.terminal_state = self.grid_size  # Last state is terminal

        # Actions: LEFT=-1, COMMIT=0, RIGHT=+1 (mapped to indices 0, 1, 2)
        # Index 0 = LEFT, Index 1 = COMMIT, Index 2 = RIGHT
        self.nactions = 3
        self.nacts_factor = [self.nactions]
        self.ACTION_LEFT = 0
        self.ACTION_COMMIT = 1
        self.ACTION_RIGHT = 2

        # Observations: discretized positions + terminal observation
        self.nobs = self.grid_size + 1
        self.nobs_factor = [self.nobs]
        self.terminal_obs = self.grid_size

        self.nsq = self.nstates ** 2
        self.nso = self.nstates * self.nobs

        # Triggers (for single agent, doesn't affect behavior)
        self.state_trigger = list(range(self.nstates))

    def pos_to_state(self, y):
        """Convert continuous position y to state index."""
        # Clamp to grid bounds
        y_clamped = max(self.grid_min, min(self.grid_max, y))
        # Round to nearest integer position
        y_int = round(y_clamped)
        return y_int - self.grid_min

    def state_to_pos(self, state):
        """Convert state index to position y."""
        if state == self.terminal_state:
            return None
        return state + self.grid_min


class LightDarkProblemFactory:
    """Generates T, O, R matrices for the discrete LightDark1D problem."""

    def __init__(self, config):
        self.c = config

        # Matrices
        self.transit = [0.0] * (config.nsq * config.nactions)
        self.obs = [0.0] * (config.nso * config.nactions)
        self.reward = [0.0] * (config.nstates * config.nactions)

        # Initial belief: discretized Gaussian N(INIT_MEAN, INIT_STD)
        self.init_beliefs = self._build_initial_belief()

    def _build_initial_belief(self):
        """Build initial belief as discretized Gaussian."""
        beliefs = [0.0] * self.c.nstates

        # Use scipy for proper discretization of normal distribution
        dist = stats.norm(loc=INIT_MEAN, scale=INIT_STD)

        for state in range(self.c.grid_size):
            y = self.c.state_to_pos(state)
            # Probability mass in bin [y-0.5, y+0.5]
            prob = dist.cdf(y + 0.5) - dist.cdf(y - 0.5)
            beliefs[state] = prob

        # Normalize (handles truncation at grid boundaries)
        total = sum(beliefs)
        if total > 0:
            beliefs = [p / total for p in beliefs]

        # Terminal state has 0 probability initially
        beliefs[self.c.terminal_state] = 0.0

        return beliefs

    def _sigma(self, y):
        """
        Observation noise standard deviation at position y.

        Following original: σ(y) = |y - 5|/√2 + 0.01
        """
        return abs(y - LIGHT_POSITION) / SQRT2 + MIN_NOISE

    def _get_obs_distribution(self, y):
        """
        Get discretized observation distribution for position y.

        Observations are drawn from N(y, σ(y)) and discretized to grid positions.
        """
        if y is None:  # Terminal state
            probs = [0.0] * self.c.nobs
            probs[self.c.terminal_obs] = 1.0
            return probs

        sigma = self._sigma(y)
        dist = stats.norm(loc=y, scale=sigma)

        probs = [0.0] * self.c.nobs
        for obs_state in range(self.c.grid_size):
            obs_y = self.c.state_to_pos(obs_state)
            # Probability mass in bin [obs_y - 0.5, obs_y + 0.5]
            prob = dist.cdf(obs_y + 0.5) - dist.cdf(obs_y - 0.5)
            probs[obs_state] = prob

        # Normalize (handles probability mass outside grid)
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        return probs

    def _apply_action(self, state, action):
        """
        Apply action to state and return new state.

        Actions: LEFT=0, COMMIT=1, RIGHT=2
        """
        if state == self.c.terminal_state:
            return self.c.terminal_state

        y = self.c.state_to_pos(state)

        if action == self.c.ACTION_COMMIT:
            return self.c.terminal_state
        elif action == self.c.ACTION_LEFT:
            new_y = y - STEP_SIZE
        else:  # ACTION_RIGHT
            new_y = y + STEP_SIZE

        # Clamp to grid
        new_y = max(self.c.grid_min, min(self.c.grid_max, new_y))
        return self.c.pos_to_state(new_y)

    def _get_reward(self, state, action):
        """
        Compute reward R(s, a).

        Following original:
        - Terminal state: 0
        - COMMIT action: +10 if |y| < 1, -10 otherwise
        - Movement: -movement_cost (default 0)
        """
        if state == self.c.terminal_state:
            return 0.0

        y = self.c.state_to_pos(state)

        if action == self.c.ACTION_COMMIT:
            # Correct if within GOAL_RADIUS of GOAL_POSITION
            if abs(y - GOAL_POSITION) < GOAL_RADIUS:
                return CORRECT_REWARD
            else:
                return INCORRECT_PENALTY
        else:
            # Movement cost (0 by default in original)
            return -MOVEMENT_COST

    def generate(self):
        """Construct and return the T, O, R matrices."""
        c = self.c

        for action in range(c.nactions):
            for state in range(c.nstates):
                # Reward
                self.reward[action * c.nstates + state] = self._get_reward(state, action)

                # Transition (deterministic)
                new_state = self._apply_action(state, action)
                self.transit[action * c.nsq + state * c.nstates + new_state] = 1.0

            # Observations (depend on resulting state, not action)
            for new_state in range(c.nstates):
                y = c.state_to_pos(new_state)
                obs_dist = self._get_obs_distribution(y)
                for obs in range(c.nobs):
                    if obs_dist[obs] > 0:
                        self.obs[action * c.nso + new_state * c.nobs + obs] = obs_dist[obs]

        return (self.transit, self.obs, self.reward, self.init_beliefs,
                c.nacts_factor, c.nobs_factor)


# ==========================================
# Solver Interfaces
# ==========================================

def run_lightdark_rssda(config, verbose=True):
    """Run LightDark using RSSDA solver."""
    time_start = time.time()

    factory = LightDarkProblemFactory(config)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()

    time_gen = time.time()

    if verbose:
        print(f"LightDark1D Problem (Discrete)")
        print(f"  Grid: [{config.grid_min}, {config.grid_max}] ({config.grid_size} positions)")
        print(f"  States: {config.nstates} | Actions: {config.nactions} | Obs: {config.nobs}")
        print(f"  Light position: {LIGHT_POSITION} | Goal: {GOAL_POSITION} (radius {GOAL_RADIUS})")
        print(f"  Initial belief: N({INIT_MEAN}, {INIT_STD})")
        print(f"  Noise model: σ(y) = |y - {LIGHT_POSITION}|/√2 + {MIN_NOISE}")
        print(f"  Rewards: correct={CORRECT_REWARD}, incorrect={INCORRECT_PENALTY}")
        print(f"Time (generation): {time_gen - time_start:.3f}s")

    model = SDecPOMDPModel(
        nagents=config.nagents,
        nstates=config.nstates,
        nactions=config.nactions,
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

    solver_config = RSSDAConfig(
        maxh=config.horizon,
        maxit=config.maxit,
        IEmin2=config.ie_min2,
        alpha=config.alpha,
        algorithm=ALGORITHM,
        heuristic_type=HEURISTIC_TYPE,
        tail_heuristic_type=TAIL_HEURISTIC_TYPE,
        TI1=TI1, TI2=TI2, TI3=TI3, TI4=TI4,
        score_limit=SCORE_LIMIT,
        cen_threshold=CEN_THRESHOLD,
        sm_temperature=SM_TEMPERATURE,
        iter_limit=ITER_LIMIT,
        rec_limit=REC_LIMIT,
        hybrid_r=HYBRID_R,
        max_clusters=MAX_CLUSTERS
    )

    sdec_pomdp = SDecPOMDP(model=model, config=solver_config)

    if verbose:
        print(f"\n--- Planning (H={config.horizon}) ---")

    time_solve_start = time.time()
    try:
        result = sdec_pomdp.multi_agent_astar(config.horizon)
        time_solve_end = time.time()

        val = result[0]
        policy = result[1]

        if verbose:
            print(f"Value: {val:.4f}")
            print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
            print(f"Time (total): {time_solve_end - time_start:.3f}s")

            if "--policy" in sys.argv:
                print_policy(policy, config)

        return val, policy, result[2] if len(result) > 2 else None

    except MemoryLimitExceeded as e:
        time_solve_end = time.time()
        if verbose:
            print(f"Result: MO (Memory limit exceeded)")
            print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
        return "MO", None, None


def run_lightdark_decpomdp(config, verbose=True):
    """Run LightDark using original decPOMDP solver."""
    if OriginalDecPOMDP is None:
        print("decPOMDP solver not available.")
        return "NA", None, None

    time_start = time.time()

    factory = LightDarkProblemFactory(config)
    T, O, R, init_b, nacts_fac, nobs_fac = factory.generate()

    # Convert to pdict format
    T_pdict = []
    for act in range(config.nactions):
        for s in range(config.nstates):
            indices, values = [], []
            for snew in range(config.nstates):
                val = T[act * config.nsq + s * config.nstates + snew]
                if val > 0:
                    indices.append(snew)
                    values.append(val)
            T_pdict.append((array('i', indices), array('d', values)))

    O_pdict = []
    for act in range(config.nactions):
        for snew in range(config.nstates):
            indices, values = [], []
            for o in range(config.nobs):
                val = O[act * config.nso + snew * config.nobs + o]
                if val > 0:
                    indices.append(o)
                    values.append(val)
            O_pdict.append((array('i', indices), array('d', values)))

    time_gen = time.time()

    if verbose:
        print(f"LightDark1D Problem (decPOMDP solver)")
        print(f"  Grid: [{config.grid_min}, {config.grid_max}] ({config.grid_size} positions)")
        print(f"  States: {config.nstates} | Actions: {config.nactions} | Obs: {config.nobs}")
        print(f"Time (generation): {time_gen - time_start:.3f}s")

    dec_pomdp = OriginalDecPOMDP(
        nagents=config.nagents,
        nstates=config.nstates,
        nactions=config.nactions,
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
    dec_pomdp.decentralized = False
    dec_pomdp.onesided = False

    if verbose:
        print(f"\n--- Planning (H={config.horizon}) ---")

    time_solve_start = time.time()
    try:
        val, policy, clustering = dec_pomdp.multi_agent_astar(config.horizon)
        time_solve_end = time.time()

        if verbose:
            print(f"Value: {val:.4f}")
            print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
            print(f"Time (total): {time_solve_end - time_start:.3f}s")

        return val, policy, clustering

    except DecPOMDPMemoryLimitExceeded as e:
        time_solve_end = time.time()
        if verbose:
            print(f"Result: MO (Memory limit exceeded)")
            print(f"Time (solving): {time_solve_end - time_solve_start:.3f}s")
        return "MO", None, None


def print_policy(policy, config):
    """Print policy in human-readable format."""
    action_names = ["LEFT", "COMMIT", "RIGHT"]
    print("\n--- Policy ---")
    for stage, stage_policy in enumerate(policy):
        print(f"Stage {stage + 1}:")
        if stage_policy and stage_policy[0] and stage_policy[0][0]:
            for oh_idx, act in enumerate(stage_policy[0][0]):
                if act >= 0:
                    print(f"  OH[{oh_idx}]: {action_names[act]}")


# ==========================================
# Simulation
# ==========================================

def simulate(config, policy, clustering, num_episodes=1000, verbose=False):
    """
    Monte Carlo simulation to estimate expected value.
    """
    factory = LightDarkProblemFactory(config)
    T, O, R, init_b, _, _ = factory.generate()

    action_names = ["LEFT", "COMMIT", "RIGHT"]
    rewards = []

    for ep in range(num_episodes):
        # Sample initial state
        state = np.random.choice(config.nstates, p=init_b)
        episode_reward = 0.0
        oh_idx = 0
        discount = 1.0

        for step in range(config.horizon):
            if state == config.terminal_state:
                break

            # Get action from policy
            action = 1  # Default to COMMIT
            if step < len(policy) and policy[step][0] and policy[step][0][0]:
                if oh_idx < len(policy[step][0][0]):
                    action = policy[step][0][0][oh_idx]
                    if action < 0:
                        action = 1

            # Reward
            r = R[action * config.nstates + state]
            episode_reward += discount * r
            discount *= DISCOUNT

            # Transition
            new_state = factory._apply_action(state, action)

            # Observation
            y = config.state_to_pos(new_state)
            obs_dist = factory._get_obs_distribution(y)
            obs = np.random.choice(config.nobs, p=obs_dist)

            if verbose and ep == 0:
                pos = config.state_to_pos(state)
                new_pos = config.state_to_pos(new_state) if new_state != config.terminal_state else "TERM"
                print(f"  Step {step+1}: y={pos} a={action_names[action]} y'={new_pos} o={obs} r={r:.1f}")

            # Update observation history
            if step < len(clustering) and clustering[step] and clustering[step][0]:
                if oh_idx < len(clustering[step][0]):
                    cluster_map = clustering[step][0][oh_idx]
                    if obs < len(cluster_map) and cluster_map[obs] >= 0:
                        oh_idx = cluster_map[obs]

            state = new_state

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


# ==========================================
# Benchmark Mode
# ==========================================

def run_benchmark(horizons=None, solver="rssda"):
    """Run benchmark experiments across multiple horizons."""
    if horizons is None:
        horizons = [5, 6, 7, 8, 9, 10]

    print("=" * 60)
    print("LightDark1D Benchmark")
    print("=" * 60)
    print(f"Grid: [{GRID_MIN}, {GRID_MAX}] | Light: {LIGHT_POSITION} | Goal: {GOAL_POSITION}")
    print(f"Solver: {solver.upper()}")
    print("-" * 60)
    print(f"{'H':>4} | {'Value':>12} | {'Time (s)':>10} | {'Status':>8}")
    print("-" * 60)

    results = []
    for h in horizons:
        config = LightDarkConfig()
        config.horizon = h

        t0 = time.time()
        if solver == "rssda":
            val, _, _ = run_lightdark_rssda(config, verbose=False)
        else:
            val, _, _ = run_lightdark_decpomdp(config, verbose=False)
        elapsed = time.time() - t0

        status = "OK" if val != "MO" else "MO"
        val_str = f"{val:.4f}" if val != "MO" else "MO"
        print(f"{h:>4} | {val_str:>12} | {elapsed:>10.3f} | {status:>8}")

        results.append({"horizon": h, "value": val, "time": elapsed, "status": status})

    print("=" * 60)
    return results


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("LightDark1D POMDP Benchmark")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python sdec_lightdark.py <horizon> [options]")
        print()
        print("Options:")
        print("  --policy        Print policy structure")
        print("  --decpomdp      Use original decPOMDP solver")
        print("  --sim <N>       Simulate N episodes after solving")
        print("  --benchmark     Run benchmark across horizons 5-10")
        print()
        print("Examples:")
        print("  python sdec_lightdark.py 8              # Solve with horizon 8")
        print("  python sdec_lightdark.py 8 --sim 1000   # Solve and simulate")
        print("  python sdec_lightdark.py --benchmark    # Run full benchmark")
        print()
        print(f"Problem: Grid [{GRID_MIN}, {GRID_MAX}], Light at {LIGHT_POSITION}, Goal at {GOAL_POSITION}")
        sys.exit(0)

    if "--benchmark" in sys.argv:
        solver = "decpomdp" if "--decpomdp" in sys.argv else "rssda"
        run_benchmark(solver=solver)
    else:
        config = LightDarkConfig()

        if "--decpomdp" in sys.argv:
            val, policy, clustering = run_lightdark_decpomdp(config)
        else:
            val, policy, clustering = run_lightdark_rssda(config)

        if "--sim" in sys.argv and policy is not None and clustering is not None:
            sim_idx = sys.argv.index("--sim")
            n_eps = int(sys.argv[sim_idx + 1]) if sim_idx + 1 < len(sys.argv) else 1000
            print(f"\nSimulating {n_eps} episodes...")
            mean_r, std_r = simulate(config, policy, clustering, n_eps, verbose=True)
            print(f"Simulated value: {mean_r:.4f} ± {std_r:.4f}")
