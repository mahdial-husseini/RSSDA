import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("horizon", type=int, help="Horizon (h) of the problem.")
    parser.add_argument("--cluster_type", default="lossless", help="Clustering type. One of lossless, only_possible, state_beliefs, one_cluster, none, \
             finite_memory_nocluster, finite_memory_only_possible, finite_memory_cluster")
    parser.add_argument("--maxit", default=200, help="Maximum number of iterations for computing heuristics.")
    parser.add_argument("--q_depth", default=3, help="Lowest horizon for which heuristics keep being computed.")
    parser.add_argument("--alpha", default=0.2, help="Threshold for heuristic decrease before terminating heuristic computation.")
    parser.add_argument("--iter_limit", default="inf", required=False,
        help="Maximum number of iterations per horizon in main computation (setting this to less than infinity gives up guarantees).") # default is "inf"
    parser.add_argument("--maxrec", default="inf", required=False, help="Determines the maximal recursion depth.") # default is "inf"
    parser.add_argument("--memory", type=int, default=None, required=False,
        help="Window size for sliding window clustering.") # default is None
    parser.add_argument("--heuristic", default="MDP", required=False, 
        help="One of MDP or POMDP; determines the type of precomputed heuristic.")    
    parser.add_argument("--rec_type", default="MDP", required=False, 
        help="One of max_reward, MDP, rec_state, recursive; determines the type of terminal heuristic.")
    
    parser.add_argument("--agents", type=int, default=2, required=False, help="Number of agents.")
    parser.add_argument("--policyvalfound", type=float, default=None, required=False, help="Value of best solution found.")
    parser.add_argument("--p_threshold_cluster", type=float, default=0, required=False, help="Probability for probability-based clustering.")
    parser.add_argument("--p_threshold_expand", type=float, default=0, required=False, help="Probability for limited expanding.")
    parser.add_argument("--output", action='store_true', default=True, required=False, help="Print progress output.")
    
    parser.add_argument("--random", action='store_true', default=False, required=False, help="Compute the expected reward of the random policy.")
    parser = parser.parse_args()
    
    if parser.heuristic is not None:
        parser.maxit = 0
        parser.q_depth = "inf"
        parser.alpha = 0
    
    return parser