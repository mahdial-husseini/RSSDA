from heapq import heappush, heappop, heapify     # heapq, used as a Priority queue
from itertools import count             # counter, used as tie-breaker in the Priority queue
import math, sys, os, time        # resource for os only
from array import array                 # memory efficient arrays of ints or doubles
import psutil

# Memory tracking utilities
class MemoryLimitExceeded(Exception):
    """Raised when memory usage exceeds the configured limit."""
    pass

def get_memory_usage_gb():
    """Returns current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def int_tuple(plist, factor):        # int_tuple computes a tuple of integers from a (sparse) list of reals to use as dictionary key
    return tuple([x*factor + int(plist[x] * factor) for x in range(len(plist)) if plist[x] > 0])
    
def int_tuple_pdict(pdict, factor):  # int_tuple_dict computes a tuple of integers from a probability dictionary
    return tuple([x*factor + int(y * factor) for x, y in zip(*pdict) if y>0])

def val_tuple(vlist, factor):        # val_tuple computes a tuple of integers from a (dense) list of reals
    return tuple([int(x * factor) for x in vlist])

def cumprod(lens):           # cumprod takes an array [l_0, l_1, ..., l_{n-1}] and returns the array of cumulative products
    ll = len(lens)           # [1, l_0, l_0l_1, ..., l_0l_1...l_{n-2}], as well as the full product l_0l_1...l_{n-1} separately.
    div = [1]*ll
    for idx in range(ll-1):
        div[idx+1] = div[idx] * lens[idx]
    return(div, div[ll-1]*lens[ll-1])

def product(list1, list2, mult):                     # Given lists L1, L2, computes the list of indices x + My 
    return [x+mult*y for y in list2 for x in list1]  # corresponding to pairs (x, y), where x in L1, y in L2.
    
def lists_product(lists, mults, nlists):             # Given a list L of lists L_0, ..., L_{n-1}, computes the list of indices 
    list1 = lists[0]                                 # x_0 + M_1 x_1 + ... M_{n-1} x_{n-1} corresponding to tuples (x_0, x_1, ..., x_{n-1})
    for idx in range(1, nlists):                     # where x_0 in L_0, x_1 in L_1, ...., L_{n-1}. 
        list1 = [x+mults[idx]*y for y in lists[idx] for x in list1]
    return list1
    
def lists_product2(list1idx, list1, rlens, mults, nlists):  # Computes a list product where only one list is not a range.
    return lists_product([list1 if idx == list1idx else range(rlens[idx]) for idx in range(nlists)], mults, nlists)
    
def pdict(plist):    # Computes a memory efficient representation of a sparse list of reals
    supp = [s for s in range(len(plist)) if plist[s] > 0]
    return array("i", supp), array("d", [plist[s] for s in supp])
    
class Policy:
    
    def __init__(self, policy, ncluster, dists, prob = None, clustering = None, 
                       values = None, heuristics = None, depth = None):
          # policy: list of list of lists, giving the actions corresponding to each (clustered) observation history (oh).
          #   Indexing: policy[stage][agent][oh].
          # ncluster: list of lists, giving the number of clustered observation histories. Indexing: ncluster[stage][agent]. 
          # dists: list of lists, giving the index of a distribution over states for each joint oh. Indexing: dists[stage][oh].
          # prob: list of lists, giving the probability of each joint oh. Indexing: prob[stage][oh].
          # clustering: shows which observation history is clustered to which, by giving for each old oh and each observation
          #   the number of the new oh, where -1 denotes that an observation has probability 0. Indexing: clustering[stage][agent][oh][o].
          # values: cumulative policy value for the first (i+1) stages. Indexing: values[stage].
          # heuristics: list of heuristics, one for each depth. Indexing: heuristics[stage].
          # depth: lowest depth for which heuristics are still computed.
          
        if prob == None: 
            prob = [[1]]
        if clustering == None: 
            clustering = []
        if values == None: 
            values = []
        if heuristics == None: 
            heuristics = [math.inf]
        
        self.policy = policy         
        self.ncluster = ncluster
        self.dists = dists
        self.prob = prob
        self.clustering = clustering
        self.values = values
        self.heuristics = heuristics
        self.depth = depth
    
    def policy_copy(self, idx, aidx):                        # Makes a copy of a (partial) policy, so that it can be altered.
        policy = self.policy.copy()                          # Only copies policy and heuristics, since the other elements of a policy
        policy[idx] = policy[idx].copy()                     # are only changed/expanded when going to the next stage, and can hence be
        policy[idx][aidx] = policy[idx][aidx].copy()         # shared between different partial policies of the same stage.
        heuristics = self.heuristics.copy()
        return Policy(policy, self.ncluster, self.dists, self.prob, self.clustering, self.values, heuristics, self.depth)
        
    def policy_copy_laststage(self, idx, aidx):              # Makes a copy of a partial policy and prepares the determination of
        policy = self.policy.copy()                          # the action for all ohs for a given stage and agent at once. 
        policy[idx] = policy[idx].copy()                     # Used for the last stage of the last agent.
        policy[idx][aidx] = [0]*self.ncluster[idx][aidx]
        return Policy(policy, self.ncluster, self.dists, self.prob, self.clustering, self.values, [], self.depth)
    
                                                             # Makes a copy of the clustering structure
    def cluster_copy(self):                                  # Used when expanding the clustering structure at the end of a stage. 
        return Policy(self.policy, self.ncluster.copy(), self.dists.copy(), self.prob.copy(), 
                      self.clustering.copy(), self.values.copy(), self.heuristics, self.depth)


class DecPOMDP:
    
    def __init__(self, nagents, nstates, nactions, nobs, transitions, obs, rewards, 
                    init_beliefs, nacts_factor, nobs_factor, maxh,
                    cluster_type, maxit, q_depth, alpha, iter_limit, maxrec, memory, heuristic, rec_type, 
                    p_threshold_cluster, p_threshold_expand, policyvalfound, output, **kwargs):
          # DecPOMDP parameters:
          # nagents: integer n, number of agents
          # nstates: integer |S|; the states are numbered 0 up to |S|-1
          # nactions: total number of actions |A|
          # nobs: total number of observations
          # transitions: for each action a in A, and each state s in S, a probability distribution 
          #   over the states in S, represented as one list of length |A| * |S| * |S|
          # obs: for each action a in A, and each new state s in S, a probability distribution 
          #   over the observations in obs_list, represented as one list of length |A| * |S| * |O|
          # rewards: reward of action a in state s, represented as one list of length |A| * |S|
          # init_beliefs: initial belief state, common to all agents
          # nacts_factor: list, giving the number of actions for each agent
          # nobs_factor: list, giving the number of observations for each agent
          # maxh: maximum horizon for which the DecPOMDP is solved
          
          # Solving parameters:
          # cluster_type: whether (lossless) clustering is used, or another type of (lossy) clustering
          #   possibilities: "lossless", "only_possible", "state_beliefs", "one_cluster", "none",  
          #   "finite_memory_nocluster", "finite_memory_only_possible", "finite_memory_cluster"
          # maxit: maximum number of iterations for computing heuristics (M in the paper)
          # q_depth: lowest horizon for which heuristics keep being computed (d in the paper)
          # alpha: threshold for heuristic decrease before terminating heuristic computation
          # iter_limit: maximum number of iterations per horizon in main computation (L in the paper)
          #     (setting this to less than infinity gives up guarantees)
          # maxrec: determines the maximal recursion depth (r in the paper)
          # memory: window size for sliding window memory clustering (k in the paper)
          # heuristic: determines which precomputed heuristic is used
          #   possibilities: None, "MDP", "POMDP"
          # rec_type: determines the type of terminal heuristic
          #   possibilities: None, "max_reward", "MDP", "rec_state", "recursive"
          # p_threshold_cluster: probability threshold used in finite memory clustering
          # p_threshold_expand: probability threshold used for adding all successors to priority queue
          # policyvalfound: best value of a policy found so far
          # output: determines whether extensive output should be printed
          
        assert (cluster_type in ["lossless", "only_possible", "state_beliefs", "one_cluster", "none",  
             "finite_memory_nocluster", "finite_memory_only_possible", "finite_memory_cluster"]), "invalid clustering type"
        assert (heuristic in [None, "MDP", "POMDP"]), "invalid heuristic type"
        assert (rec_type in [None, "max_reward", "MDP", "rec_state", "recursive"]), "invalid rec_type"
          
        self.nagents = nagents
        self.nstates = nstates 
        self.nactions = nactions
        self.transitions = transitions
        self.nobs = nobs
        self.obs = obs
        self.rewards = array('d',rewards)
        self.init_beliefs = pdict(init_beliefs)
        self.nacts_factor = nacts_factor
        self.nobs_factor = nobs_factor
        self.maxh = maxh
        
        self.maxa = max(nacts_factor)
        self.a_prod = [1]*(self.nagents+1)
        self.o_prod = [1]*(self.nagents+1)
        for idx in range(self.nagents):
            self.a_prod[idx+1] = self.a_prod[idx] * self.nacts_factor[idx]
            self.o_prod[idx+1] = self.o_prod[idx] * self.nobs_factor[idx]
            
        self.cluster_type = cluster_type
        self.SH = True  # save extra heuristics             
        self.maxit = int(maxit) if maxit not in ["inf", math.inf] else math.inf
        self.q_depth = int(q_depth) if q_depth not in ["inf", math.inf] else math.inf
        self.alpha = float(alpha) if alpha not in ["inf", math.inf] else math.inf
        self.iter_limit = int(iter_limit) if iter_limit not in ["inf", math.inf] else math.inf
        self.maxrec = int(maxrec) if maxrec not in ["inf", math.inf] else math.inf
        self.memory = int(memory) if memory is not None else None
        
        self.heuristic = heuristic
        self.MDP = (heuristic == "MDP")
        self.POMDP = (heuristic == "POMDP")
        if self.POMDP:   #POMDP heuristic works only with a lossless clustering type
            assert self.cluster_type in ["lossless", "only_possible", "none"], \
                "POMDP heuristic works only with a lossless clustering type"
                
        self.MDP_heuristic = None      # stores the MDP heuristic
        self.maxMDP = None             # stores the maximum MDP value accross all states
        self.POMDP_heuristic = None    # stores the POMDP heuristic (value function of belief MDP)
        self.bl_POMDP = None           # stores belief lengths used for indexing POMDP heuristic
        self.maxr = max(rewards)       # max reward, used for the maximum reward heuristic

        self.rec_type = rec_type
        self.maxrrec = (rec_type == "max_reward")
        self.rec = (rec_type != "max_reward")
        self.MDPrec = (rec_type == "MDP")
        self.recrec = (rec_type == "recursive")
        self.recstate = (rec_type == "rec_state")

        self.p_threshold_cluster = p_threshold_cluster
        self.p_threshold_expand = p_threshold_expand
        self.policyvalfound = policyvalfound
        self.output = output
        
        self.factor = 10**12+39 
           # factor used to multiply probabilities with and round to integers for use in dictionary keys
           # plus 39 to make it prime (to avoid problems with rounding errors with probabilities which are an exact multiple of 10^{-12}).   
        self.init_call = True  # used to determine whether the call to MAA* is the initial call or not (e.g. for computing heuristics or output)
        self.zero = True       # used to determine whether a heuristic should be computed for depth 0, if applicable

        # Memory tracking (optional, extracted from kwargs)
        self.memory_limit_gb = kwargs.get('memory_limit_gb', 16.0)  # Default 16GB limit
        self.memory_check_interval = kwargs.get('memory_check_interval', 100)  # Check every 100 iterations
        self._last_reported_mem_gb = 0  # Track last reported memory level for 1GB increment logging

          # dec_heuristic: dictionary, saving computed heuristic values indexed by
          #   remaining horizon rh, initial distribution, fixed policy, clustering structure
          # newstatedist_dict: dictionary storing computed new state distributions corresponding to a given state distribution and action
          # terminal_dict: dictionary storing computed terminal probabilities corresponding to a given state distribution and action
          # cluster_dict: dictionary storing computed clustering structures for given terminal probabilities
          # clusterctr_dict: dictionary storing clusterctrs corresponding to clustering structures
          # clusteridx_dict: dictionary storing indexes of clusterctrs
          # prevcluster: list storing for each clusteridx the index of the clustering structure in the previous stage
          # terminalMDP_dict: dictionary storing terminal reward MDP heuristics
          # reward_list: list storing the reward  corresponding to a given state distribution and action 
          # dist_dict: giving the index of the distribution given the (rounded) distribution
          # dists: giving the floating point distribution corresponding to an index
        self.dec_heuristic = dict()
        self.newstatedist_dict = dict()
        self.terminal_dict = dict()
        self.cluster_dict = dict()
        self.clusterctr_dict = dict()
        self.clusteridx_dict = {(): 0}
        self.prevcluster = [None]
        self.terminalMDP_dict = dict()
        self.reward_list = array('d',[sum([rewards[act*nstates+s] * init_beliefs[s] for s in range(nstates)]) for act in range(nactions)])
        self.dist_dict = {int_tuple(init_beliefs, self.factor): 0}
        self.dists = [self.init_beliefs]        

          # ctrs: lists of ctrs that extend a given ctr
        self.ctrs = {1:[1]}
        self.ctrs_partial = [[1]]
        for a in range(self.nagents):
            for ctr_fix in self.ctrs:
                self.ctrs[ctr_fix] = [c*self.maxa+acta for c in self.ctrs[ctr_fix] for acta in range(self.nacts_factor[a])] 
            for ctr_fix in self.ctrs[1]:
                self.ctrs[ctr_fix] = [ctr_fix]
            self.ctrs_partial.append(self.ctrs[1].copy())

        self.transitions_transpose = [(array("i", []), array("d", [])) for _ in range(nstates*nactions)]
        self.obs_transpose = [(array("i", []), array("d", [])) for _ in range(nactions*nobs)]
        for act in range(nactions):
            for s in range(nstates):
                for idx, snew in enumerate(transitions[act*nstates+s][0]):
                    self.transitions_transpose[act*nstates+snew][0].append(s)
                    self.transitions_transpose[act*nstates+snew][1].append(transitions[act*nstates+s][1][idx])
                for idx, o in enumerate(obs[act*nstates+s][0]):
                    self.obs_transpose[act*nobs+o][0].append(s)
                    self.obs_transpose[act*nobs+o][1].append(obs[act*nstates+s][1][idx])
                    
          # supports of next state distributions in transition function
        self.trans_supps = [self.compute_supports(act) for act in range(self.nactions)]
        
        self.memlimit = 16000000 * (1000 if sys.platform == "darwin" else 1)
        self.memdiv = 1000 * (1000 if sys.platform == "darwin" else 1)

        

    def compute_MDP_heuristic(self, h):
          # computes the MDP heuristic over horizon h, using value iteration
        
        values = []  # final stage
        validx = array('d',[0]*(2*self.maxa**self.nagents*self.nstates))
        for act in range(self.nactions):  # best reward over horizon 1 when the action is fixed,
            a = self.nagents              # is simply the reward corresponding to that action
            policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
            for s in range(self.nstates):
                validx[policyctr*self.nstates+s] = self.rewards[act*self.nstates+s]
                            
        for a in range(self.nagents-1, -1, -1):
            for act in range(self.a_prod[a]):      # maximize over the actions of the agents, one by one
                policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
                for s in range(self.nstates):
                    validx[policyctr*self.nstates+s] = max([validx[(policyctr*self.maxa+acta)*self.nstates+s] for acta in range(self.nacts_factor[a])])

        values.append(validx)
        val1 = validx
        for idx in range(1, h):
            validx = array('d',[0]*(2*self.maxa**self.nagents*self.nstates))
            a = self.nagents
            for act in range(self.nactions):  # compute the value over horizon idx when the action is fixed
                policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
                for s in range(self.nstates):
                    validx[policyctr*self.nstates+s] = self.rewards[act*self.nstates+s] + \
                          sum([val1[self.nstates+snew]*self.transitions[act*self.nstates+s][1][idx] 
                            for idx, snew in enumerate(self.transitions[act*self.nstates+s][0])])
                                
            for a in range(self.nagents-1, -1, -1):
                for act in range(self.a_prod[a]):  # maximize over the actions of the agents, one by one
                    policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
                    for s in range(self.nstates):
                        validx[policyctr*self.nstates+s] = max([validx[(policyctr*self.maxa+acta)*self.nstates+s] for acta in range(self.nacts_factor[a])])
                        
            values.append(validx)
            val1 = validx
        
        self.MDP_heuristic = values
        self.maxMDP = [max(validx) for validx in self.MDP_heuristic] 
    
        
    def compute_beliefs(self, h, init = 0):
          # computes all reachable beliefs within h stage from belief init
        beliefs = [[init]]
        belief1 = [init]
        for idx in range(1, h):
            beliefidx = []
            for b in belief1:
                for act in range(self.nactions):
                    terminal_dists, _ = self.get_terminal(b, act)
                    beliefidx.extend(terminal_dists)
            beliefidx = set(beliefidx)
            beliefidx.discard(-1)
            beliefs.append(beliefidx)
            belief1 = beliefidx
        
        return beliefs
 
 
    def compute_POMDP_heuristic(self, h):
          # computes a POMDP-like heuristic for all reachable beliefs, using value iteration on the belief MDP
        beliefs = self.compute_beliefs(h)  # compute all reachable beliefs up to horizon h
        self.bl_POMDP = [max(belief_list)+1 for belief_list in beliefs][::-1]
        
        values = [] # first stage
        bl = self.bl_POMDP[0]
        validx = array('d',[0]*(2*self.maxa**self.nagents*bl))
        a = self.nagents                      # best reward in horizon 1 for a fixed action
        for act in range(self.nactions):      # is just the expected reward for that action
            policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
            for b in beliefs[h-1]: validx[policyctr*bl+b] = self.reward_list[b*self.nactions + act]
        
        for a in range(self.nagents-1, -1, -1):
            for act in range(self.a_prod[a]):      # maximize over the actions of the agents, one by one
                policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
                for b in beliefs[h-1]: 
                    validx[policyctr*bl+b] = max([validx[(policyctr*self.maxa+acta)*bl+b] for acta in range(self.nacts_factor[a])])

        values.append(validx)
        val1 = validx
        for idx in range(h-2, -1, -1):
            bl = self.bl_POMDP[h-1-idx]
            bl1 = self.bl_POMDP[h-2-idx]
            validx = array('d',[0]*(2*self.maxa**self.nagents*bl))
            a = self.nagents
            for act in range(self.nactions):
                policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
                for b in beliefs[idx]:  # calculate optimal reward for a fixed action act
                    terminal_dists, terminal_probs = self.get_terminal(b, act)
                    heuristic = self.reward_list[b*self.nactions + act]
                    if self.decentralized or self.onesided:
                        terminal_ctrs = [[0]*(2*self.maxa**self.nagents) for o in range(self.nobs)]
                        for o in range(self.nobs):
                            for ctr_o in self.ctrs[1]:   #calculate reward for each action for each belief after receiving additional observation
                                terminal_ctrs[o][ctr_o] = val1[ctr_o * bl1 + terminal_dists[o]] * terminal_probs[o]
                    
                    heuristic += sum([val1[bl1 + terminal_dists[o]] * terminal_probs[o] for o in range(self.nobs)])
                    validx[policyctr*bl+b] = heuristic
    
            for a in range(self.nagents-1, -1, -1):
                for act in range(self.a_prod[a]):      # maximize over the actions of the agents, one by one
                    policyctr = self.maxa**a + sum([(act//self.a_prod[i])%self.nacts_factor[i] * self.maxa**(a-1-i) for i in range(a)])
                    for b in beliefs[idx]: 
                        validx[policyctr*bl+b] = max([validx[(policyctr*self.maxa+acta)*bl+b] for acta in range(self.nacts_factor[a])])

            values.append(validx)
            val1 = validx
        
        self.POMDP_heuristic = values


    def compute_clusterctr(self, cdict_i):       # compute_clusterctr: computes a number from which the clustering structure
        ci = 0                                   # can be recovered, by converting the tuple into a number.
        factor = 1                               
        for a in range(self.nagents):
            factor1 = len(cdict_i[a]) * self.nobs_factor[a] + 1    # largest possible number of new clusters, +1 to account for the -1
            for x in cdict_i[a]:                                   # (used for denoting that an (oh, o) pair has probability 0)
                for y in x:                                        # Note that len(cdict_i[a]) is known from the previous clusterctr.
                    ci += factor*(y+1)
                    factor *= factor1
        return ci
        
          # save_heuristic and save_heuristic2: save heuristic values where more actions
          # are fixed, based on the optimal policy where less actions are fixed
    def save_heuristic(self, rh, init, ctr, pi_heuristic, cdict_heuristic, aidx, h_MDP):
        ctr_heuristic = ctr
        pi0 = [x[0] for x in pi_heuristic[0]]
        for a in range(aidx, self.nagents):
            ctr_heuristic = ctr_heuristic * self.maxa + pi0[a]
            self.dec_heuristic[(rh, init, ctr_heuristic, 0, h_MDP)] = \
               self.dec_heuristic[(rh, init, ctr, 0, h_MDP)]

    def save_heuristic2(self, rh, init, ctr, cdict_tup, pi_heuristic, cdict_heuristic, aidx, policy_idx, h_MDP, idx):
        ctr_heuristic = ctr
        pi_idx = pi_heuristic[idx][aidx][policy_idx:]  # compute list of (optimal) actions
        for a in range(aidx+1, self.nagents):          # that were not fixed 
            pi_idx.extend(pi_heuristic[idx][a])
        for p in pi_idx:
            ctr_heuristic = ctr_heuristic * self.maxa + p
            self.dec_heuristic[(rh, init, ctr_heuristic, cdict_tup, h_MDP)] = \
               self.dec_heuristic[(rh, init, ctr, cdict_tup, h_MDP)]
    
    def compute_heuristic_init(self, rh, init, h_MDP):
          # compute and save the heuristic (rh, init, 1, ())
        if self.MDP:
            states, probs = self.dists[init]
            self.dec_heuristic[(rh, init, 1, 0, 0)] = sum([self.MDP_heuristic[rh-1][self.nstates+s]*probs[idx] for idx, s in enumerate(states)])
        elif self.POMDP:
            self.dec_heuristic[(rh, init, 1, 0, 0)] = self.POMDP_heuristic[rh-1][self.bl_POMDP[rh-1]+init]
        else:
            self.dec_heuristic[(rh, init, 1, 0, h_MDP)], pi_heuristic, cdict_heuristic = \
               self.multi_agent_astar(rh, init_beliefs = init, maxit = self.maxit, h_MDP = h_MDP)
            if self.SH and pi_heuristic != None:
                self.save_heuristic(rh, init, 1, pi_heuristic, cdict_heuristic, 0, h_MDP)
                
    def compute_heuristic_ctr(self, rh, init, ctr):
          # compute and save the heuristic (rh, init, ctr, 0). Used for recstate heuristic
        n_ctr = self.nagents # determine how many actions are assigned for ctr
        while ctr < self.maxa**n_ctr: n_ctr -= 1
        policy_init = Policy([[[(ctr//self.maxa**(n_ctr-a-1))%self.maxa] if a < n_ctr else [] for a in range(self.nagents)]], [[1 for a in range(self.nagents)]], [[init]])
        self.dec_heuristic[(rh, init, ctr, 0, 0)], _, _ = self.multi_agent_astar(rh, policy_init, maxit = self.maxit)
    
    def compute_heuristic1(self, pi_c, oh, depth, rh, init, ctr, aidx, h_MDP):
          # compute and save the heuristic (rh, init, ctr, 0)
        if self.MDP:
            states, probs = self.dists[init]
            self.dec_heuristic[(rh, init, ctr, 0, 0)] = sum([self.MDP_heuristic[rh-1][ctr*self.nstates+s]*probs[idx] for idx, s in enumerate(states)])
        elif self.POMDP:
            self.dec_heuristic[(rh, init, ctr, 0, 0)] = self.POMDP_heuristic[rh-1][ctr*self.bl_POMDP[rh-1]+init]
        else:
            policy_oh = self.shorten_policy(pi_c, oh, depth, rh, h_MDP)
            self.dec_heuristic[(rh, init, ctr, 0, h_MDP)], pi_heuristic, cdict_heuristic = \
                self.multi_agent_astar(rh, policy_oh, maxit = self.maxit, \
                     upper = self.dec_heuristic.get((rh, init, ctr//self.maxa, 0, h_MDP)), h_MDP = h_MDP)
            if self.SH and pi_heuristic != None and aidx < self.nagents:
                self.save_heuristic(rh, init, ctr, pi_heuristic, cdict_heuristic, aidx, h_MDP)
        
    def compute_heuristic2(self, pi_c, oh, depth, rh, init, ctr, cdict_tup, h_MDP, aidx = None): 
          # compute and save the heuristic (rh, init, ctr, cdict_tup)
        policy_oh = self.shorten_policy(pi_c, oh, depth, rh, h_MDP)
        if self.SH and aidx == None:
            aidx = 0  # compute for how many agents pi_c assigned all actions in the current stage 
            while aidx < self.nagents and len(policy_oh.policy[-1][aidx]) == policy_oh.ncluster[-1][aidx]: aidx += 1
        if self.SH and aidx < self.nagents: 
            policy_idx = len(policy_oh.policy[-1][aidx])
        upper_h = self.dec_heuristic.get((rh, init, ctr//self.maxa, cdict_tup, h_MDP))
        if upper_h == None: upper_h = self.dec_heuristic.get((rh, init, ctr//self.maxa, self.prevcluster[cdict_tup], h_MDP))
        self.dec_heuristic[(rh, init, ctr, cdict_tup, h_MDP)], pi_heuristic, cdict_heuristic = \
           self.multi_agent_astar(rh, policy_oh, maxit = self.maxit, upper = upper_h, h_MDP = h_MDP)
        if self.SH and pi_heuristic != None and aidx < self.nagents:
            self.save_heuristic2(rh, init, ctr, cdict_tup, pi_heuristic, cdict_heuristic, aidx, policy_idx, h_MDP, len(policy_oh.policy)-1)
        
    def get_init(self, probs):            # computes index of a probability distribution over states,
        dsum = sum(probs)                 # as well as a normalization factor/total probability
        if dsum == 0:                     # this (oh, o) pair is not possible
            return 0, -1                  # (i.e. has probability 0), so assign -1                           
        if dsum != 1: probs = [x/dsum for x in probs]
        dist = int_tuple(probs, self.factor)
        d = self.dist_dict.get(dist)
        if d == None:
            if (not self.init_call) and self.POMDP:
                  # this can only happen due to rounding errors 
                  # when using the POMDP heuristic, we precompute all possible distributions
                minerr = 1e-9
                minerridx = -1
                for idx, dist1 in enumerate(self.dists):
                    err = 0
                    for s_idx, s in enumerate(dist1[0]):
                        err += abs(probs[s] - dist1[1][s_idx])
                        if err > minerr: break
                    if err < minerr:
                        minerr = err
                        minerridx = idx
                if minerr > 1e-12:
                    print("distribution not found, replacement error:", minerr)
                return dsum, minerridx
                    
            d = len(self.dists)
            self.dist_dict[dist] = d
            pdict_probs = pdict(probs)
            self.dists.append(pdict_probs)
            self.reward_list.extend([sum([self.rewards[act*self.nstates+s] * probs[s] for s in pdict_probs[0]]) 
                 for act in range(self.nactions)])
        return dsum, d
        
    def get_init_dict(self, pdict):
        states, probs = pdict
        dsum = sum(probs)
        if dsum == 0: return 0, -1 
        if dsum != 1: probs = array('d', [x/dsum for x in probs])
        pdict_probs = (states, probs)
        dist = int_tuple_pdict(pdict_probs, self.factor)
        d = self.dist_dict.get(dist)
        if d == None:
            d = len(self.dists)
            self.dist_dict[dist] = d
            self.dists.append(pdict_probs)
            self.reward_list.extend([sum([self.rewards[act*self.nstates+s] * probs[idx] for idx, s in enumerate(pdict_probs[0])]) 
                 for act in range(self.nactions)])
        return dsum, d
    
    
    def evaluate_policy(self, pi_c, terminal_dec = False, probs_terminal = None, dists_terminal = None, rh = None, h = None, h_MDP = None):
          # evaluates the policy pi_c up to horizon h, using the recursive heuristic for the remaining horizon rh
          # based on terminal probabilities given in probs_terminal and new initial distributions given in dists_terminal
        if h == None: h = len(pi_c.policy)
        policyval = 0
        if terminal_dec:
            if probs_terminal == None:
                probs_terminal, dists_terminal = pi_c.prob[h], pi_c.dists[h]
            for prob, init in zip(probs_terminal, dists_terminal):    # if init == -1, the (oh, o) combination
                if init != -1:                                        # has probability 0, so value is irrelevant
                    heuristic = self.dec_heuristic.get((rh, init, 1, 0, h_MDP))
                    if heuristic == None:
                        self.compute_heuristic_init(rh, init, h_MDP)
                        heuristic = self.dec_heuristic[(rh, init, 1, 0, h_MDP)]
                    policyval += heuristic * prob
                    
        elif h_MDP != None and h_MDP > 0 and self.rec:
            dists_terminal, probs_terminal = self.terminal_probabilities(pi_c)
            for prob, init in zip(probs_terminal, dists_terminal):   
                if prob != 0:                 
                    policyval += prob * self.get_terminalMDP(init, h_MDP)
        
        if h == 0: return policyval
        lp = len(pi_c.values)
        if lp < h:
            if lp == 0: pvalidx = 0
            else: pvalidx = pi_c.values[lp-1]
            for idx in range(lp, h):
                acts = lists_product(pi_c.policy[idx], self.a_prod, self.nagents)
                pvalidx += sum([self.reward_list[pi_c.dists[idx][oh]*self.nactions+acts[oh]]*pi_c.prob[idx][oh] for oh in range(len(acts))])
                pi_c.values.append(pvalidx)

        policyval += pi_c.values[h-1]
        return policyval
    
    
    def get_terminalMDP(self, init, h_MDP, ctr_fix = 1):
        if self.recrec:   # recursive terminal reward heuristics
            terminal_val = self.dec_heuristic.get((h_MDP, init, ctr_fix, 0, 0))
            if terminal_val == None: 
                if ctr_fix == 1: self.compute_heuristic_init(h_MDP, init, 0)
                else: self.compute_heuristic_ctr(h_MDP, init, ctr_fix)
                terminal_val = self.dec_heuristic[(h_MDP, init, ctr_fix, 0, 0)]
            return terminal_val
        if ctr_fix == 1: terminalMDP = self.terminalMDP_dict.get((init, h_MDP))
        else: terminalMDP = self.terminalMDP_dict.get((init, h_MDP, ctr_fix))
        return terminalMDP if terminalMDP is not None else self.compute_terminalMDP(init, h_MDP, ctr_fix)
    
    
    def compute_terminalMDP(self, dist, h_MDP, ctr_fix):
        states, probs_list = self.dists[dist]
        if self.recstate:
            terminalMDP = -math.inf
            probs = {states[idx]:probs_list[idx] for idx in range(len(states))}
            supp = list(states)    # process states by decreasing probability
            supp.sort(key = lambda s: -probs[s])
            terminal_ctrs = [0]*(2*self.maxa**self.nagents)
            inits = {s:self.get_init_dict(([s], [1.0]))[1] for s in supp}
            for ctr in self.ctrs[ctr_fix]:   # first compute MDP heuristic to avoid computing expensive heuristic if not necessary
                terminal_ctrs[ctr] = sum([self.MDP_heuristic[h_MDP-1][ctr*self.nstates+s]*probs[s] for s in supp])
            ctr_list = self.ctrs[ctr_fix].copy()   # process possible actions in order of MDP value
            ctr_list.sort(key = lambda ctr : -terminal_ctrs[ctr])
            for ctr in ctr_list:
                terminal_ctr = terminal_ctrs[ctr]
                done = {s:False for s in supp}
                for s in supp:  # first process all heuristics that have already been computed
                    if terminal_ctr < terminalMDP: break  # this action cannot be the best
                    terminal_val_s = self.dec_heuristic.get((h_MDP, inits[s], ctr, 0, 0))
                    if terminal_val_s != None:  # replace MDP heuristic with terminal reward heuristic
                        terminal_ctr += probs[s] * (terminal_val_s - self.MDP_heuristic[h_MDP-1][ctr*self.nstates+s])
                        done[s] = True
                for s in supp:
                    if done[s]: continue   # compute and process the remaining heuristics
                    elif terminal_ctr < terminalMDP: break # this action cannot be the best
                    self.compute_heuristic_ctr(h_MDP, inits[s], ctr)
                    terminal_val_s = self.dec_heuristic[(h_MDP, inits[s], ctr, 0, 0)]
                    terminal_ctr += probs[s] * (terminal_val_s - self.MDP_heuristic[h_MDP-1][ctr*self.nstates+s])
                terminalMDP = max(terminalMDP, terminal_ctr)
        else: # terminal MDP heuristic 
            terminalMDP = max([sum([self.MDP_heuristic[h_MDP-1][ctr*self.nstates+s]*probs_list[idx] 
                                        for idx, s in enumerate(states)]) for ctr in self.ctrs[ctr_fix]])
        if ctr_fix == 1: self.terminalMDP_dict[(dist, h_MDP)] = terminalMDP
        else: self.terminalMDP_dict[(dist, h_MDP, ctr_fix)] = terminalMDP
        return terminalMDP
    
    def get_newstatedist(self, dist, act):                         # get new distribution over states,
        newstatedist = self.newstatedist_dict.get((dist, act))     # given old distribution over states and action taken
        return newstatedist if newstatedist is not None else self.compute_newstatedist(dist, act)
        
    def compute_newstatedist(self, dist, act):                     # compute new distribution over states,
        states, probs = self.dists[dist]                           # given old distribution over states and action taken
        len_states = len(states)
        supp_list, snew_supp = self.trans_supps[act]
        supp_s_idx_list = []
        supp_p_idx_list = []
        for supp in supp_list:                                     # compute intersection of support of dist with 
            p_idx = 0                                              # the states from which a particular state can be reached
            supp_s_idx = []
            supp_p_idx = []
            for s_idx, s in enumerate(supp):                       # both lists are sorted, so iterate over the two lists
                while p_idx < len_states and states[p_idx] < s: 
                    p_idx += 1
                if p_idx == len_states: break
                if states[p_idx] == s:
                    supp_s_idx.append(s_idx)
                    supp_p_idx.append(p_idx)
            supp_s_idx_list.append(supp_s_idx)
            supp_p_idx_list.append(supp_p_idx)   

        newstatedist = array('d', [0.0]*self.nstates)
        for snew in range(self.nstates):
            supp_no = snew_supp[snew]
            s_idcs, p_idcs = supp_s_idx_list[supp_no], supp_p_idx_list[supp_no]
            newstatedist[snew] = sum([probs[p_idx] * self.transitions_transpose[act*self.nstates+snew][1][s_idx] 
                    for p_idx, s_idx in zip(p_idcs, s_idcs)]) 
        newstatedist = pdict(newstatedist)
        self.newstatedist_dict[(dist, act)] = newstatedist
        return newstatedist
        
    def compute_supports(self, act):          # compute for each action and each new state s'
        supp_list = []                        # from which old states s it holds that P(s' | a, s) > 0
        supp_dict = dict()
        snew_supp = [-1]*self.nstates
        for snew in range(self.nstates):
            supp = self.transitions_transpose[act*self.nstates+snew][0]
            supp_tup = tuple(supp)
            supp_no = supp_dict.get(supp_tup)
            if supp_no == None:
                supp_no = len(supp_list)
                supp_list.append(supp)
                supp_dict[supp_tup] = supp_no
            snew_supp[snew] = supp_no
        return (supp_list, snew_supp)
        
        
    def get_terminal(self, dist, act):                             # get new distribution over states and observations,
        terminal = self.terminal_dict.get((dist, act))             # given old distribution over states and action taken
        return terminal if terminal is not None else self.compute_terminal(dist, act)
        
    def compute_terminal(self, dist, act):                         # compute new distribution over states and observations,
        states, newstatedist = self.get_newstatedist(dist, act)    # given old distribution over states and action taken
        terminal_dists = array('i', [0]*self.nobs)
        terminal_probs = array('d', [0]*self.nobs)
        for o in range(self.nobs):
            ohl = [0.0]*self.nstates 
            o_idx = 0
            o_len = len(self.obs_transpose[act*self.nobs+o][0])     # find and process intersection of new state dist support
            for s_idx, snew in enumerate(states):                   # and observation function support
                while o_idx < o_len and self.obs_transpose[act*self.nobs+o][0][o_idx] < snew: 
                    o_idx += 1
                if o_idx == o_len: break
                if self.obs_transpose[act*self.nobs+o][0][o_idx] == snew:
                    ohl[snew] = newstatedist[s_idx] * self.obs_transpose[act*self.nobs+o][1][o_idx]
            terminal_probs[o], terminal_dists[o] = self.get_init(ohl)
        self.terminal_dict[(dist, act)] = (terminal_dists, terminal_probs)
        return (terminal_dists, terminal_probs)
     
    def terminal_probabilities(self, pi_c):                                 # computes distributions over observation histories,
        acts = lists_product(pi_c.policy[-1], self.a_prod, self.nagents)    # new observations and states
        nOhs = len(acts)
        len1 = nOhs*self.nobs
        terminal_dists = [-1]*len1                 # index of conditional distribution P(s | oh, o)
        terminal_probs = [0.0]*len1                # marginal probability P(oh, o)
        for oh in range(nOhs):
            if pi_c.prob[-1][oh] != 0:
                terminal_dists_oh, terminal_probs_oh = self.get_terminal(pi_c.dists[-1][oh], acts[oh]) 
                terminal_dists[oh*self.nobs:(oh+1)*self.nobs] = terminal_dists_oh
                terminal_probs[oh*self.nobs:(oh+1)*self.nobs] = [x * pi_c.prob[-1][oh] for x in terminal_probs_oh]

        return terminal_dists, terminal_probs

    def lossless_clustering(self, pi_new, nOhs, div, aidx, dists_terminal, probs_terminal):
           # computes all conditional distributions P(oh, o, s | oh[aidx], o[aidx])
           # and clusters observation histories based on these distributions
        dist_dicta = dict()                                    
        len_ohna = nOhs//pi_new.ncluster[-1][aidx]
        len_ona = self.nobs//self.nobs_factor[aidx] 
        len_t = len_ohna * len_ona
        cluster_newa = 0
        clustering_newa = [[0]*self.nobs_factor[aidx] for i in range(pi_new.ncluster[-1][aidx])]
        for oha in range(pi_new.ncluster[-1][aidx]):
            ohs = lists_product2(aidx, [oha], pi_new.ncluster[-1], div, self.nagents)
            for oa in range(self.nobs_factor[aidx]):
                cond_dist = [0]*len_t     # index of conditional distribution P(s | oh, o)
                cond_prob = [0]*len_t     # conditional probability P(oh, o | oh[aidx], o[aidx])
                os = lists_product2(aidx, [oa], self.nobs_factor, self.o_prod, self.nagents)
                   # compute conditional probability P(oh, o | oh[aidx], o[aidx]), and index of corresponding distribution over states
                for ohna, oh in enumerate(ohs):
                    for ona, o in enumerate(os):
                        cond_dist[ohna*len_ona+ona] = dists_terminal[oh*self.nobs+o]
                        cond_prob[ohna*len_ona+ona] = probs_terminal[oh*self.nobs+o]
                            
                total_prob = sum(cond_prob)
                if total_prob > 0:
                    cond_prob = [x/total_prob for x in cond_prob]   # create tuple containing conditional distribution
                    probsint = int_tuple(cond_prob, self.factor)    
                    cond_dist = tuple(cond_dist)                   
                    d = dist_dicta.get((probsint, cond_dist))    # check whether there is already a cluster
                    if d == None:                                # corresponding to this conditional distribution
                        d = cluster_newa                         # if not, create a new cluster
                        cluster_newa += 1
                        dist_dicta[(probsint, cond_dist)] = d
                    clustering_newa[oha][oa] = d
                else: clustering_newa[oha][oa] = -1
                 
        return cluster_newa, clustering_newa
     
    def state_beliefs_clustering(self, pi_new, nOhs, div, aidx, dists_terminal, probs_terminal):     
           # computes all conditional distributions P(s | oh[aidx], o[aidx])
           # and clusters observation histories based on these distributions. NB: not lossless!
        dist_dicta = dict()                                    
        cluster_newa = 0
        clustering_newa = [[0]*self.nobs_factor[aidx] for i in range(pi_new.ncluster[-1][aidx])]
        for oha in range(pi_new.ncluster[-1][aidx]):
            ohs = lists_product2(aidx, [oha], pi_new.ncluster[-1], div, self.nagents)
            for oa in range(self.nobs_factor[aidx]):
                state_dist = [0]*self.nstates
                os = lists_product2(aidx, [oa], self.nobs_factor, self.o_prod, self.nagents)
                
                for oh in ohs:
                    for o in os:
                        states, prob = self.dists[dists_terminal[oh*self.nobs+o]]
                        for idx, s in enumerate(states):
                            state_dist[s] += probs_terminal[oh*self.nobs+o] * prob[idx]
                            
                total_prob = sum(state_dist)
                if total_prob > 0:
                    state_dist = [x/total_prob for x in state_dist]
                    probsint = int_tuple(state_dist, self.factor)
                    d = dist_dicta.get(probsint)
                    if d == None:
                        d = cluster_newa
                        cluster_newa += 1
                        dist_dicta[probsint] = d
                    clustering_newa[oha][oa] = d
                else: clustering_newa[oha][oa] = -1
                   
        return cluster_newa, clustering_newa
        
    def possible_oh_clustering(self, pi_new, nOhs, div, aidx, probs_terminal):    
          # no clustering except removing new observation histories (oh, o) with probability 0 (i.e. setting them to -1)
        cluster_newa = 0                                                       
        clustering_newa = [[0]*self.nobs_factor[aidx] for i in range(pi_new.ncluster[-1][aidx])]
        for oha in range(pi_new.ncluster[-1][aidx]):                     
            ohs = lists_product2(aidx, [oha], pi_new.ncluster[-1], div, self.nagents)
            for oa in range(self.nobs_factor[aidx]):
                possible = False
                os = lists_product2(aidx, [oa], self.nobs_factor, self.o_prod, self.nagents)
                for oh in ohs:        # check whether (oha, oa) is possible by checking whether there is a joint OH (oh, o)
                    for o in os:      # with (oh[a], o[a]) = (oha, oa) with positive probability
                        if probs_terminal[oh*self.nobs+o] > 0: 
                            possible = True; break
                    if possible: break
                if possible: 
                    clustering_newa[oha][oa] = cluster_newa
                    cluster_newa += 1
                else: clustering_newa[oha][oa] = -1
        
        return cluster_newa, clustering_newa
             
    def no_clustering(self, pi_new, aidx):       # no clustering, so each (oh[a], o[a]) pair gets its own cluster     
        return pi_new.ncluster[-1][aidx] * self.nobs_factor[aidx], \
          [list(range(self.nobs_factor[aidx]*oha, self.nobs_factor[aidx]*(oha+1))) for oha in range(pi_new.ncluster[-1][aidx])]
    
    def one_cluster(self, pi_new, aidx):         # everything in one cluster
        return 1, [[0]*self.nobs_factor[aidx] for oha in range(pi_new.ncluster[-1][aidx])]
        
    def finite_memory_nocluster(self, pi_new, aidx):  # sliding window memory without additional clustering
        memory = self.memory
        stage = len(pi_new.policy)
        if stage <= memory:
            return self.no_clustering(pi_new, aidx)
        factor = pi_new.ncluster[-1][aidx] // self.nobs_factor[aidx]
        return pi_new.ncluster[-1][aidx], \
          [list(range(self.nobs_factor[aidx]*(oha%factor), self.nobs_factor[aidx]*(oha%factor+1))) for oha in range(pi_new.ncluster[-1][aidx])]
    
    def finite_memory_possibleoh(self, pi_new, nOhs, div, aidx, dists_terminal, probs_terminal):
          # sliding window memory and removing new observation histories (oh, o) with probability 0 (i.e. setting them to -1)
        memory = self.memory
        stage = len(pi_new.policy)
                                # no observations are forgotten, so it coincides with clustering where 
        if stage <= memory:     # only observation histories (oh, o) with probability 0 have no cluster
            return self.possible_oh_clustering(pi_new, nOhs, div, aidx, probs_terminal)

        cluster_newa = 0                                                       
        clustering_newa = [[0]*self.nobs_factor[aidx] for i in range(pi_new.ncluster[-1][aidx])]
            
        d_oa = [-1 for oha in range(pi_new.ncluster[-1][aidx])]
        cluster_dicta = dict()
        
        if memory > 1:  # calculate which sequence of observation histories belongs to which most recent clustered oh
            for oha in range(pi_new.ncluster[-memory][aidx]):
                  # check for each old cluster what happens when observing a particular sequence of observations
                for omem in range(self.nobs_factor[aidx]**(memory-1)):
                    oha_new = oha
                    for idx in range(memory-1):  # calculate new cluster after observing observations in omem
                        oi = (omem//self.nobs_factor[aidx]**idx)%self.nobs_factor[aidx]
                        oha_new = pi_new.clustering[-memory+idx+1][aidx][oha_new][oi]
                        if oha_new == -1: break
                    if oha_new != -1:
                        if d_oa[oha_new] == -1:
                            d_oa[oha_new] = omem  
                        else: assert d_oa[oha_new] == omem   # each cluster can only correspond to one oh suffix

        for oha in range(pi_new.ncluster[-1][aidx]):                     
            ohs = lists_product2(aidx, [oha], pi_new.ncluster[-1], div, self.nagents)
            for oa in range(self.nobs_factor[aidx]):
                possible = False
                os = lists_product2(aidx, [oa], self.nobs_factor, self.o_prod, self.nagents)
                for oh in ohs:      # check whether (oha, oa) is possible by checking whether there is a joint OH (oh, o)
                    for o in os:    # with (oh[a], o[a]) = (oha, oa) with positive probability
                        if probs_terminal[oh*self.nobs+o] > 0: 
                            possible = True; break
                    if possible: break
                if possible: 
                    reduced_oha = d_oa[oha]
                    d = cluster_dicta.get((reduced_oha, oa))
                    if d == None:
                        d = cluster_newa
                        cluster_newa += 1
                        cluster_dicta[(reduced_oha, oa)] = d
                    clustering_newa[oha][oa] = d
                else: clustering_newa[oha][oa] = -1
        return cluster_newa, clustering_newa
     
       
    def finite_memory_cluster(self, pi_new, nOhs, div, aidx, dists_terminal, probs_terminal, rh):
        memory = self.memory
        stage = len(pi_new.ncluster)    # number of stages (time steps) for which action is specified
        lowest = min(stage-1, memory)   # number of observations remembered at this point
        if stage > memory:              # number of observations that will be forgotten
            lim = min(lowest, rh)       # which is at most lowest and at most the number of remaining stages
        else: lim = min(lowest, max(rh-(memory-stage+1), 0)) #next memory-stage+1 stages no observations are forgotten
    
        cluster_newa = 0                                                       
        clustering_newa = [[0]*self.nobs_factor[aidx] for i in range(pi_new.ncluster[-1][aidx])]
        
          # stores which observation histories of length idx are clustered in cluster oha of the current agent aidx
        d_oa = [[[] for oha in range(pi_new.ncluster[-1][aidx])]] 
          # stores to which cluster the observation history omem belongs (inverse of d_oa) of the current agent aidx
        d_omem = [[-1 for omem in range(self.nobs_factor[aidx]**idx)] for idx in range(lowest, -1, -1)]
        cluster_dicta = dict()
        
        oha_map_ag = [None for ag in range(self.nagents)]  # stores to which new cluster old clusters belong after forgetting one observation
        len_ag = [None for ag in range(self.nagents)]      # stores length of oha_map_ag[ag]
        if stage > memory:
              # throw away oldest observation of other agents
            for ag in range(self.nagents):
                if ag != aidx:
                      # stores which observation histories of length memory are clustered in cluster oha of agent ag
                    d_oa_ag = [[] for oha in range(pi_new.ncluster[-1][ag])] 
                    for omem in range(self.nobs_factor[ag]**lowest):
                        # check for each old cluster what happens when observing a particular sequence of observations
                        for oha in range(pi_new.ncluster[-(lowest+1)][ag]):
                            oha_new = oha
                            for idx in range(lowest):  # calculate new cluster after observing observations in omem
                                oi = (omem//self.nobs_factor[ag]**idx)%self.nobs_factor[ag]
                                oha_new = pi_new.clustering[-lowest+idx][ag][oha_new][oi]
                                if oha_new == -1: break
                            if oha_new != -1:
                                d_oa_ag[oha_new].append(omem)
                    
                    prev_len = pi_new.ncluster[-1][ag]
                      # stores which observation histories of length memory-1 are clustered in cluster oha of agent ag
                    d_oa_temp = [[omem//self.nobs_factor[ag] for omem in d_oa_ag[oha]] for oha in range(prev_len)]
                      # stores to which cluster observation histories of length memory-1 belong
                    d_omem_temp = [-1 for omem in range(self.nobs_factor[ag]**(lowest-1))]
                      # merge clusters which have shortened observation histories of length memory-1 in common
                    for oha in range(prev_len): 
                          # find clusters corresponding to shortened observation histories of length memory-1 in oha
                        new_ohas = set([d_omem_temp[omem] for omem in d_oa_temp[oha]])
                        new_ohas.discard(-1)
                        for noha in new_ohas: # merge clusters: add further observation histories of length memory-1
                            d_oa_temp[oha].extend(d_oa_temp[noha])
                            d_oa_temp[noha] = [] 
                        d_oa_temp[oha] = list(set(d_oa_temp[oha]))
                          # merge cluster: assign merged cluster to all observation histories of length memory-1
                        for omem in d_oa_temp[oha]:
                            d_omem_temp[omem] = oha
                    
                      # renumber clusters to make the cluster numbers consecutive
                    new_ohas = list(set(d_omem_temp))
                    new_ohas.sort()
                    if new_ohas[0] == -1: new_ohas = new_ohas[1:]
                    for idx, new_oha in enumerate(new_ohas):
                        for omem in d_oa_temp[new_oha]:
                            d_omem_temp[omem] = idx
                      # to compute the new cluster, we map the cluster to an oh, 
                      # forget one observation from the oh and map back to a (new) cluster
                    oha_map_ag[ag] = [d_omem_temp[d_oa_ag[oha][0]//self.nobs_factor[ag]] for oha in range(prev_len)]
                    len_ag[ag] = len(new_ohas)
        else:  # if stage<=memory we do not forget observations, so the map is just the identity
            for ag in range(self.nagents):
                if ag != aidx:   
                    oha_map_ag[ag] = [oha for oha in range(pi_new.ncluster[-1][ag])]
                    len_ag[ag] = pi_new.ncluster[-1][ag]

        
            # calculate which sequence of observation histories belongs to which most recent clustered oh
        for omem in range(self.nobs_factor[aidx]**lowest):
            # check for each old cluster what happens when observing a particular sequence of observations
            for oha in range(pi_new.ncluster[-(lowest+1)][aidx]):
                oha_new = oha
                for idx in range(lowest):  # calculate new cluster after observing observations in omem
                    oi = (omem//self.nobs_factor[aidx]**idx)%self.nobs_factor[aidx]
                    oha_new = pi_new.clustering[-lowest+idx][aidx][oha_new][oi]
                    if oha_new == -1: break
                if oha_new != -1:
                    d_oa[0][oha_new].append(omem)
                    assert d_omem[0][omem] in [-1, oha_new]  # each omem should belong to only one cluster
                    d_omem[0][omem] = oha_new
                    

        len_ag_prod = [1]*(self.nagents+1)
        len_ohas_old = [1]*(self.nagents+1)
        for ag in range(self.nagents):
            if ag != aidx:
                len_ag_prod[ag+1] = len_ag_prod[ag] * len_ag[ag]
                len_ohas_old[ag+1] = len_ohas_old[ag] * pi_new.ncluster[-1][ag]
            else: 
                len_ag_prod[ag+1] = len_ag_prod[ag]
                len_ohas_old[ag+1] = len_ohas_old[ag]
        len_ohna = len_ag_prod[-1]
        len_ona = self.nobs//self.nobs_factor[aidx] 
        len_t = len_ohna * len_ona
        
          # calculate conditional distribution over clusters of other agents and states
        cond_probs = [[] for _ in range(lowest+1)]
        cond_dists = [[] for _ in range(lowest+1)]
        for oha in range(pi_new.ncluster[-1][aidx]):
            ohs = lists_product2(aidx, [oha], pi_new.ncluster[-1], div, self.nagents)
            for oa in range(self.nobs_factor[aidx]):
                cond_sdist = [[0]*self.nstates for idx in range(len_t)]
                cond_dist = [0]*len_t
                cond_prob = [0]*len_t
                os = lists_product2(aidx, [oa], self.nobs_factor, self.o_prod, self.nagents)
                   # compute conditional probability P(oh, o | oh[aidx], o[aidx]), and index of corresponding distribution over states
                for ohna, oh in enumerate(ohs):
                    idx_new = 0
                    for ag in range(self.nagents):
                        if ag != aidx:
                            ohna_oldag = (ohna//len_ohas_old[ag]) % pi_new.ncluster[-1][ag]
                            ohna_newag = oha_map_ag[ag][ohna_oldag]
                            idx_new += len_ag_prod[ag] * ohna_newag
                      
                      # compute conditional distribution over the states
                    for ona, o in enumerate(os):
                        pcur = probs_terminal[oh*self.nobs+o]
                        if pcur != 0:
                            idx_new_ona = idx_new * len_ona + ona
                            cond_prob[idx_new] += pcur
                            states, sprobs = self.dists[dists_terminal[oh*self.nobs+o]]
                            for idx, s in enumerate(states):
                                cond_sdist[idx_new][s] += sprobs[idx]*pcur
                                
                for idx_new in range(len_t):
                    if cond_prob[idx_new] != 0:
                        _, cond_dist[idx_new] = self.get_init(cond_sdist[idx_new])
                    else: cond_dist[idx_new] = -1
                        
                cond_probs[0].append(cond_prob)
                cond_dists[0].append(cond_dist)
                
        oha_map = []
              # compute conditional distributions over clusters of other agents and beliefs after throwing away observations
        for omem_len in range(lowest, lowest-lim, -1):
              # compute new ohas after throwing away observations of agent aidx
            prev_idx = lowest - omem_len
            cur_idx = lowest + 1 - omem_len
            prev_len = len(d_oa[prev_idx])
              
              # forget one observation from each omem corresponding to a cluster oha
            d_oa_temp = [[omem//self.nobs_factor[aidx] for omem in d_oa[prev_idx][oha]] for oha in range(prev_len)]
              # merge clusters which have shortened observation histories in common
            for oha in range(prev_len):
                new_ohas = set([d_omem[cur_idx][omem] for omem in d_oa_temp[oha]])
                new_ohas.discard(-1)
                for noha in new_ohas:
                    d_oa_temp[oha].extend(d_oa_temp[noha])
                    d_oa_temp[noha] = []
                d_oa_temp[oha] = list(set(d_oa_temp[oha]))
                for omem in d_oa_temp[oha]:
                    d_omem[cur_idx][omem] = oha
            
              # renumber clusters to make the cluster numbers consecutive
            new_ohas = list(set(d_omem[cur_idx]))
            new_ohas.sort()
            if new_ohas[0] == -1: new_ohas = new_ohas[1:]
            for idx, new_oha in enumerate(new_ohas):
                for omem in d_oa_temp[new_oha]:
                    d_omem[cur_idx][omem] = idx
            d_oa.append([omemlist for omemlist in d_oa_temp if len(omemlist) > 0])
            oha_map.append([d_omem[cur_idx][d_oa[prev_idx][oha][0]//self.nobs_factor[aidx]] for oha in range(prev_len)])  
            cur_len = len(new_ohas)
            
              # calculate new conditional distributions
            for noha in range(cur_len):
                for oa in range(self.nobs_factor[aidx]):
                      # calculate new belief after throwing away one observation
                    cond_dist = [0]*len_t
                    cond_prob = [0]*len_t
                    ohas = [oha for oha in range(prev_len) if oha_map[prev_idx][oha] == noha]
                    for cond_idx in range(len_t):
                        new_statedist = [0]*self.nstates
                        prob = 0
                        for oha in ohas:
                            oha_oa = oha*self.nobs_factor[aidx]+oa
                            pcur = cond_probs[prev_idx][oha_oa][cond_idx]
                            if pcur != 0:
                                prob += pcur
                                states, sprobs = self.dists[cond_dists[prev_idx][oha_oa][cond_idx]]
                                for idx, s in enumerate(states):
                                    new_statedist[s] += sprobs[idx]*pcur
                        cond_prob[cond_idx] = prob
                        if prob != 0:
                            _, cond_dist[cond_idx] = self.get_init(new_statedist)   
                        else: cond_dist[cond_idx] = -1
                    
                    cond_probs[cur_idx].append(cond_prob)
                    cond_dists[cur_idx].append(cond_dist)
                    
           # keep unscaled probabilities for probability-based clustering
        if self.p_threshold_cluster != 0:
            cond_probs_old_zero = [x.copy() for x in cond_probs[0]]

        for idx in range(lim+1):
            for p_idx in range(len(cond_probs[idx])):
                total_prob = sum(cond_probs[idx][p_idx])
                if total_prob == 0:
                    cond_probs[idx][p_idx] = -1
                    continue
                cond_prob = [x/total_prob for x in cond_probs[idx][p_idx]]
                cond_probs[idx][p_idx] = int_tuple(cond_prob, self.factor)
        cond_dists = [[tuple(y) for y in x] for x in cond_dists]
          
           # construct cluster keys: tuples describing the conditional distributions 
           # over states and clusters of other agents (after forgetting some number of observations)
        cluster_keys = [None for _ in range(pi_new.ncluster[-1][aidx]*self.nobs_factor[aidx])]
        for oha in range(pi_new.ncluster[-1][aidx]):                     
            for oa in range(self.nobs_factor[aidx]):
                cluster_key = []
                noha = oha
                p_idx = cond_probs[0][noha*self.nobs_factor[aidx]+oa]
                if p_idx != -1: 
                    if stage<=memory:
                        d_idx = cond_dists[0][noha*self.nobs_factor[aidx]+oa]
                        cluster_key.append(p_idx)
                        cluster_key.append(d_idx)
                    for idx in range(1, lim+1):
                        noha = oha_map[idx-1][noha]
                        p_idx = cond_probs[idx][noha*self.nobs_factor[aidx]+oa]
                        d_idx = cond_dists[idx][noha*self.nobs_factor[aidx]+oa]
                        cluster_key.append(p_idx)
                        cluster_key.append(d_idx)
                    cluster_keys[oha*self.nobs_factor[aidx]+oa] = tuple(cluster_key)
        
           # check if all ohs can be clustered, i.e. whether all cluster_keys are the same or -1
        cluster_key_set = set(cluster_keys)
        cluster_key_set.discard(-1)
        if len(cluster_key_set) == 1:
            for oha in range(pi_new.ncluster[-1][aidx]):                     
                for oa in range(self.nobs_factor[aidx]):
                    cluster_key = []
                    noha = oha
                    p_idx = cond_probs[0][noha*self.nobs_factor[aidx]+oa]
                    if p_idx != -1: # cluster 0 for possible ohs, cluster -1 otherwise
                        clustering_newa[oha][oa] = 0
                    else: clustering_newa[oha][oa] = -1

            return 1, clustering_newa
        
          # otherwise, check per new observation whether clustering is possible
        cluster_temp = 0
        clustering_temp = [[-1]*self.nobs_factor[aidx] for i in range(pi_new.ncluster[-1][aidx])]

        for oa in range(self.nobs_factor[aidx]):
              # check for increasing lengths of the fixed suffiex whether clustering is possible
            for omem_len in range(lowest):
                rlen = lowest-omem_len
                fix_len = self.nobs_factor[aidx]**omem_len
                factor = self.nobs_factor[aidx]**rlen
                  # loop over all fixed suffixes
                for omem_fix in range(fix_len):
                    cluster = True      # stores whether this suffix forms a cluster
                    clustered = False   # stores whether these suffixed already clustered in larger cluster
                    key = None
                    ohas = set()
                       # loop over all observation histories with omem_fix suffix
                    for omem in range(factor):
                          # find current cluster corresponding to oh
                        omem_full = omem_fix*factor + omem
                        oha = d_omem[0][omem_full]
                        if oha == -1: continue
                        
                        if clustering_temp[oha][oa] != -1:
                            clustered = True
                            break # already part of a larger cluster
                        key_oha = cluster_keys[oha*self.nobs_factor[aidx]+oa]
                        if key_oha == None:
                            continue
                        if key == None:        # store first key (which is not None)
                            key = key_oha
                        elif key != key_oha:   # we can only cluster if all keys 
                            cluster = False    # (unequal to None) are equal
                            break
                        ohas.add(oha)
                      
                      # cluster based on probability-based clustering   
                    if (not cluster) and (not clustered) and self.p_threshold_cluster != 0:
                        ohas = set([d_omem[0][omem_fix*factor + omem] for omem in range(factor)]); ohas.discard(-1)
                        p_total = sum([sum(cond_probs_old_zero[oha*self.nobs_factor[aidx]+oa]) for oha in ohas])
                        if p_total < self.p_threshold_cluster: 
                            cluster = True
                        
                    if cluster and not clustered:
                        for oha in ohas:
                            clustering_temp[oha][oa] = cluster_temp
                        cluster_temp += 1
           
           # renumber clusters using clustering_temp       
        for oha in range(pi_new.ncluster[-1][aidx]):                     
            for oa in range(self.nobs_factor[aidx]):
                p_idx = cond_probs[0][oha*self.nobs_factor[aidx]+oa]
                if p_idx != -1: 
                    cluster_key = clustering_temp[oha][oa]
                    if cluster_key == -1:
                        d = cluster_newa
                        cluster_newa += 1
                    else:
                        d = cluster_dicta.get(cluster_key)
                        if d == None:
                            d = cluster_newa
                            cluster_newa += 1
                            cluster_dicta[cluster_key] = d
                    clustering_newa[oha][oa] = d
                else: clustering_newa[oha][oa] = -1
        
        return cluster_newa, clustering_newa
            
    
    def convert_probabilities(self, probs_new_s, nOhs_new):    # converts a full distribution p(oh, s) into the marginal p(oh) and 
        probs_new = [0.0]*nOhs_new                             # indices of the conditional distributions p(s | oh)
        dists_new = [-1]*nOhs_new
        for oh in range(nOhs_new):
            ohl = probs_new_s[oh*self.nstates:(oh+1)*self.nstates]
            probs_new[oh], dists_new[oh] = self.get_init(ohl)
        return array("d", probs_new), array("i", dists_new)
    
    def cluster_policy(self, pi_c, dists_terminal, probs_terminal, rh):
          # computes the clustering structure for the next horizon
        pi_new = pi_c.cluster_copy()
        
        cluster_type = self.cluster_type
        if cluster_type == "finite_memory_cluster" and len(pi_c.policy)+rh <= self.memory+1:
            cluster_type = "lossless" # if the full horizon is small enough, then use lossless clustering
        
          # reuse previously computed clustering if available
        cluster_tuple = tuple(pi_c.ncluster[-1])
        probs_tuple = int_tuple(probs_terminal, self.factor)
        dists_tuple = tuple(dists_terminal)
        if cluster_type == "finite_memory_cluster":   # prior clustering matters for finite_memory_cluster
            dict_tuple = (tuple(pi_c.clustering[-(self.memory+1):]), probs_tuple, dists_tuple, min(rh, self.memory))
        else: dict_tuple = (cluster_tuple, probs_tuple, dists_tuple)
        new = self.cluster_dict.get(dict_tuple)
        if new != None:                   
            cluster_new, clustering_new, probs_new, dists_new = new
            pi_new.ncluster.append(cluster_new)
            pi_new.clustering.append(clustering_new)
            pi_new.prob.append(probs_new)
            pi_new.dists.append(dists_new)
            return pi_new
              
        cluster_new = []
        clustering_new = []
        div, nOhs = cumprod(pi_new.ncluster[-1])
        for a in range(self.nagents):     # compute clustering structure for each agent
            if cluster_type == "lossless": 
                cluster_newa, clustering_newa = self.lossless_clustering(pi_new, nOhs, div, a, dists_terminal, probs_terminal)
            elif cluster_type == "state_beliefs": 
                cluster_newa, clustering_newa = self.state_beliefs_clustering(pi_new, nOhs, div, a, dists_terminal, probs_terminal)
            elif cluster_type == "only_possible": 
                cluster_newa, clustering_newa = self.possible_oh_clustering(pi_new, nOhs, div, a, probs_terminal)
            elif cluster_type == "one_cluster":
                 cluster_newa, clustering_newa = self.one_cluster(pi_new, a)
            elif cluster_type == "finite_memory_nocluster":
                cluster_newa, clustering_newa = self.finite_memory_nocluster(pi_new, a)
            elif cluster_type == "finite_memory_only_possible":
                cluster_newa, clustering_newa = self.finite_memory_possibleoh(pi_new, nOhs, div, a, dists_terminal, probs_terminal)
            elif cluster_type == "finite_memory_cluster":
                cluster_newa, clustering_newa = self.finite_memory_cluster(pi_new, nOhs, div, a, dists_terminal, probs_terminal, rh)
            elif cluster_type == "none": 
                cluster_newa, clustering_newa = self.no_clustering(pi_new, a)
            else: assert False, "invalid clustering type"
            cluster_new.append(cluster_newa)
            clustering_new.append(clustering_newa)
        clustering_new = tuple([tuple([tuple(x) for x in y]) for y in clustering_new])
        pi_new.ncluster.append(cluster_new)
        pi_new.clustering.append(clustering_new)
        
          # compute probabilities of new observation histories
        cluster_newprod, nOhs_new = cumprod(cluster_new)
        probs_new_s = [0.0]*(nOhs_new*self.nstates)
        
        for a in range(self.nagents):
            oh_newlista = []
            oholista = []
            for oha in range(pi_new.ncluster[-2][a]):
                for oa in range(self.nobs_factor[a]):
                    oh_newa = pi_new.clustering[-1][a][oha][oa]
                    if oh_newa != -1:
                        oh_newlista.append(oh_newa)
                        oholista.append(oha*div[a]*self.nobs + oa*self.o_prod[a])
            
            oh_newlist = oh_newlista if a == 0 else product(oh_newlist, oh_newlista, cluster_newprod[a])
            oholist = oholista if a == 0 else product(oholist, oholista, 1)
            
        for oho, oh_new in zip(oholist, oh_newlist):
            if probs_terminal[oho] != -1:
                states, dist = self.dists[dists_terminal[oho]]
                for idx, snew in enumerate(states):
                    probs_new_s[oh_new*self.nstates+snew] += probs_terminal[oho] * dist[idx]

        probs_new, dists_new  = self.convert_probabilities(probs_new_s, nOhs_new)
        pi_new.prob.append(probs_new)
        pi_new.dists.append(dists_new)
        
        self.cluster_dict[dict_tuple] = (cluster_new, clustering_new, probs_new, dists_new)
        return pi_new
    
    
    def compute_short_ctr(self, pi_c, oh, depth, div, idx):   # computes a policyctr of the policy pi_c shortened to start at the
        short_ctr = 1                                         # observation history oh (which is of length depth)
        ohs_a = [[(oh//div[a])%pi_c.ncluster[depth][a]] for a in range(self.nagents)]
        for a in range(self.nagents):
            short_ctr = short_ctr * self.maxa + pi_c.policy[depth][a][ohs_a[a][0]]
        
          # compute all observation histories reachable from oh
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
                lp = len(pi_c.policy[j][a])
                for ohja in ohs_a[a]:
                    if lp > ohja:
                        short_ctr = short_ctr * self.maxa + pi_c.policy[j][a][ohja]
                    else: return short_ctr    # we have reached the last specified action
        
        return short_ctr
            
    def shorten_cluster(self, pi_c, oh, depth, return_ctr = True):    # computes the clustering structure of the policy pi_c shortened 
        rh = len(pi_c.policy) - depth                                 # to start at the observation history oh (which is of length depth)
        if return_ctr:
            clustering_tup = tuple(pi_c.clustering[depth:])
            clusterctr_idx = self.clusterctr_dict.get((rh, oh, clustering_tup))
            if clusterctr_idx != None: return clusterctr_idx
        
        div, _ = cumprod(pi_c.ncluster[depth])
        cluster_short = [[1]*self.nagents]
        cluster_map     = [[{(oh//div[a])%pi_c.ncluster[depth][a]: 0} for a in range(self.nagents)]]
        cluster_map_inv = [[[(oh//div[a])%pi_c.ncluster[depth][a]] for a in range(self.nagents)]]
        clustering_short = []

        for idx in range(rh-1):
            cluster_new = []
            clustering_new = []
            cluster_map.append([dict() for a in range(self.nagents)])
            cluster_map_inv.append([])
            
               # compute all observation histories reachable from oh, and renumber them to be consecutive again
            for a in range(self.nagents):
                new_clusters = [pi_c.clustering[idx+depth][a][oh_a][oa] 
                                  for oh_a in cluster_map_inv[idx][a] for oa in range(self.nobs_factor[a])]
                new_clusters = list(set(new_clusters))
                new_clusters.sort()                         # sort to ensure that the clusters are assigned in the order
                if new_clusters[0] == -1:                   # consistent with the order in which actions where assigned in pi_c
                    new_clusters = new_clusters[1:]         # (i.e. to make sure that in the shortened policy, there are no gaps in
                cluster_newa = len(new_clusters)            #  the clustered ohs to which actions were assigned)
                cluster_map_inv[idx+1].append(new_clusters)
                for c in range(cluster_newa):
                    cluster_map[idx+1][a][new_clusters[c]] = c
                clustering_dicta = [[0]*self.nobs_factor[a] for i in range(cluster_short[idx][a])]
                for j in range(cluster_short[idx][a]):
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
            clusterctr = tuple([self.compute_clusterctr(clustering_short[idx]) for idx in range(rh-1)])
            clusterctr_idx = self.clusteridx_dict.get(clusterctr)
            if clusterctr_idx == None:
                clusterctr_idx = len(self.clusteridx_dict)
                self.clusteridx_dict[clusterctr] = clusterctr_idx
                self.prevcluster.append(self.clusteridx_dict[clusterctr[:-1]])
            self.clusterctr_dict[(rh, oh, clustering_tup)] = clusterctr_idx
            return clusterctr_idx
        
        clustering_short = [tuple([tuple([tuple(x) for x in y]) for y in c]) for c in clustering_short]
        return cluster_short, clustering_short, cluster_map_inv
    
    
    def shorten_policy(self, pi_c, oh, depth, full_length, h_MDP):     # computes a shortened policy: the policy pi_c shortened to
        rh = len(pi_c.policy) - depth                                       # observation history oh (which is of length depth)
        pi_short = []
        cluster_short, clustering_short, cluster_map_inv = self.shorten_cluster(pi_c, oh, depth, False)
        total_prob = pi_c.prob[depth][oh]
        probs_short = [[1]]
        dists_short = [[pi_c.dists[depth][oh]]]
                
        for idx in range(rh):
            pi_idx = [[pi_c.policy[idx+depth][a][cluster_map_inv[idx][a][j]] for j in range(cluster_short[idx][a]) 
               if cluster_map_inv[idx][a][j] < len(pi_c.policy[idx+depth][a])]  for a in range(self.nagents)]
            pi_short.append(pi_idx)  
            
            if idx == rh-1: break
      
               # compute probabilities and indexes of state distributions of shortened policy
            div, nOhs = cumprod(cluster_short[idx])
            cluster_newprod, nOhs_new = cumprod(cluster_short[idx+1])
            probs_new_s = [0.0]*(nOhs_new*self.nstates)
            acts = lists_product(pi_idx, self.a_prod, self.nagents)
            
            oh_newas = [[] for a in range(self.nagents)]     # new clusters that have positive probability
            olistas = [[] for a in range(self.nagents)]      # local observations that have positive probability,
            for a in range(self.nagents):                    # corresponding to the clusters in oh_newas
                for oha in range(cluster_short[idx][a]):
                    oh_newas[a].append([c for c in clustering_short[idx][a][oha] if c != -1])
                    olistas[a].append([oa for oa in range(self.nobs_factor[a]) if clustering_short[idx][a][oha][oa] != -1])
            
            for oh in range(nOhs):
                act = acts[oh]
                ohas = [(oh//div[a])%cluster_short[idx][a] for a in range(self.nagents)]
                oh_newlist = lists_product([oh_newas[a][ohas[a]] for a in range(self.nagents)], cluster_newprod, self.nagents)
                olist = lists_product([olistas[a][ohas[a]] for a in range(self.nagents)], self.o_prod, self.nagents)
                    
                states, sums = self.get_newstatedist(dists_short[-1][oh], act)
                for s_idx, snew in enumerate(states):
                    obs_dense = [0.0]*self.nobs
                    for o_idx, o in enumerate(self.obs[act*self.nstates+snew][0]):
                        obs_dense[o] = self.obs[act*self.nstates+snew][1][o_idx]
                    for oh_new, o in zip(oh_newlist, olist):     
                        probs_new_s[oh_new*self.nstates+snew] += obs_dense[o] * sums[s_idx] * probs_short[-1][oh]

            probs_new, dists_new = self.convert_probabilities(probs_new_s, nOhs_new)
            probs_short.append(probs_new)
            dists_short.append(dists_new)
        
        pi_c_short = Policy(pi_short, cluster_short, dists_short, probs_short, clustering_short, [], [], None) 
        
          # compute heuristics for the shortened policy
        idx = len(pi_c_short.ncluster)
        pi_c_short.heuristics = [math.inf]*idx
        if idx > 1:
            if (not self.rec) and full_length > self.maxrec:
                full_length = min(max(self.maxrec, len(pi_c_short.policy)), full_length)         
            if self.rec and h_MDP == 0 and full_length > self.maxrec:
                hnew = min(max(self.maxrec, len(pi_c_short.policy)), full_length)
                h_MDP = full_length - hnew
                full_length = hnew
            
            pi_c_short.depth = min(idx-1, self.q_depth)
            
            depth = idx - 1
            div, nOhs = cumprod(pi_c_short.ncluster[depth])
            policyval = self.evaluate_policy(pi_c_short, h = depth)
            rh = full_length - depth
            
            for oh in range(nOhs):
                init = pi_c_short.dists[depth][oh]
                if init != -1: 
                    p_oh = pi_c_short.prob[depth][oh]
                      
                    short_ctr = 1
                    aidx = 0
                    for a in range(self.nagents):
                        oha = (oh//div[a])%pi_c_short.ncluster[depth][a]
                        if len(pi_c_short.policy[depth][a]) > oha:
                            short_ctr = short_ctr * self.maxa + pi_c_short.policy[depth][a][oha]
                            aidx += 1
                        else: break
                    
                    if (rh, init, short_ctr, 0, h_MDP) not in self.dec_heuristic:
                        if short_ctr == 1: self.compute_heuristic_init(rh, init, h_MDP)
                        else: self.compute_heuristic1(pi_c_short, oh, depth, rh, init, short_ctr, aidx, h_MDP) 
    
                    policyval += p_oh * self.dec_heuristic[(rh, init, short_ctr, 0, h_MDP)]
            pi_c_short.heuristics[depth] = policyval
            
            for depth in range(pi_c_short.depth, idx-1):  
                div, nOhs = cumprod(pi_c_short.ncluster[depth])
                policyval = self.evaluate_policy(pi_c_short, h = depth)
                rh = full_length - depth
                
                for oh in range(nOhs):
                    init = pi_c_short.dists[depth][oh]
                    if init != -1: 
                        p_oh = pi_c_short.prob[depth][oh]
                        short_ctr = self.compute_short_ctr(pi_c_short, oh, depth, div, idx)
                        ctup = self.shorten_cluster(pi_c_short, oh, depth)
                        if (rh, init, short_ctr, ctup, h_MDP) not in self.dec_heuristic:
                            if (rh, init, short_ctr, self.prevcluster[ctup], h_MDP) in self.dec_heuristic:
                                self.dec_heuristic[(rh, init, short_ctr, ctup, h_MDP)] = \
                                   self.dec_heuristic[(rh, init, short_ctr, self.prevcluster[ctup], h_MDP)]
                            else: self.compute_heuristic2(pi_c_short, oh, depth, rh, init, short_ctr, ctup, h_MDP)
   
                        policyval += p_oh * self.dec_heuristic[(rh, init, short_ctr, ctup, h_MDP)]
                pi_c_short.heuristics[depth] = policyval
        return pi_c_short
    
                 
    def multi_agent_astar(self, h, init_policy = None, init_beliefs = None, maxit = None, upper = None, h_MDP = 0):
          # computes a policy with horizon h, or a heuristic (in case maxit is set)
        if self.init_call:  # precompute heuristics
            time0 = time.time()
            if self.MDP or (self.maxrec<self.maxh and self.rec_type != "max_reward"): 
                self.compute_MDP_heuristic(h)
                states_init, probs_init = self.init_beliefs
                print(sum([self.MDP_heuristic[h-1][self.nstates+s]*probs_init[idx] for idx, s in enumerate(states_init)]))
                sys.stdout.flush()
            if self.POMDP: self.compute_POMDP_heuristic(h)
            time1 = time.time()-time0
            if time1 > 1e-4:
                print("heuristic computation time:", time1)
                sys.stdout.flush()

        if h == 0: return 0, [], []
        if init_policy == None: 
            init_policy = Policy([[[] for a in range(self.nagents)]], [[1 for a in range(self.nagents)]], 
                                 [[init_beliefs if init_beliefs is not None else 0]])
        
        init_call = self.init_call
        self.init_call = False
        if init_call:
            self._last_reported_mem_gb = 0  # Reset memory tracking for new solve

        h_maxr = 0
        if self.maxrrec and (not init_call) and h > self.maxrec:
            hnew = min(self.maxrec, h)
            h_maxr = h - hnew
            h = hnew
        if self.rec and h_MDP == 0 and (not init_call) and h > self.maxrec:
            hnew = min(self.maxrec, h)
            h_MDP = h - hnew
            h = hnew
        
        if init_call and h > self.maxrec and self.zero and self.rec_type != "max_reward":
            init_policy.depth = 0
            init_policy.heuristics[0] = self.get_terminalMDP(0, h)
                    
        if upper != None: 
            bound = upper - self.alpha*max(abs(upper), 1) - h_maxr*self.maxr
        policyval = min(init_policy.heuristics)
        
        unique = count()
        q = [(-policyval, -next(unique), init_policy)]
        policyvalfound = -math.inf
        if init_call and self.policyvalfound != None: policyvalfound = self.policyvalfound
        
        ctr = 0
        ctr2 = 0
        preval = math.inf
        prectr = 0

        while True:
            value, _, pi_c = heappop(q)
            value = -value
            
            idx = len(pi_c.policy)
            aidx = 0   # compute for which agent an action should be added
            while aidx < self.nagents and len(pi_c.policy[idx-1][aidx]) == pi_c.ncluster[idx-1][aidx]:
                aidx += 1
                
            if init_call and self.iter_limit != math.inf:
                progress = self.iter_limit * (idx-1 + aidx/self.nagents)

                if aidx != self.nagents: 
                    cluster_prods, _ = cumprod(pi_c.ncluster[idx-1])
                    cluster_prod = cluster_prods[aidx]
                    ohs = lists_product2(aidx, range(len(pi_c.policy[idx-1][aidx])), pi_c.ncluster[idx-1], cluster_prods, self.nagents)
                    progress += 1/self.nagents * (self.iter_limit - self.nagents*pi_c.ncluster[idx-1][aidx]) * sum([pi_c.prob[idx-1][oh] for oh in ohs])
                    progress += len(pi_c.policy[idx-1][aidx])

                if ctr > progress: 
                    # print(f"[PRUNE] Stage {idx-1} | Exp: {ctr} > Prog:x {progress:.2f}")
                    continue
            
            # if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > self.memlimit: exit()
            
            ctr += 1
            if value != math.inf: ctr2 += 1
            if init_call and self.output and (ctr < 1000 or (ctr%100 == 0 and ctr<10000) or ctr%10000 == 0):
                prectr = ctr
                preval = value
                # print("{0:.8f}".format(value), unique, ctr, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//self.memdiv)
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

            if ctr2 == maxit or (upper != None and value <= bound):       # check if heuristic computation should terminate early
                rval = min(value, min(pi_c.heuristics))
                if h_maxr>0 and self.maxrrec: rval += h_maxr*self.maxr
                if upper != None: rval = min(upper, rval)
                return rval, None, None
            
            if aidx == self.nagents:
                if idx == h:
                    if ctr == 1: 
                        pi_c.values = pi_c.values.copy()
                        value = self.evaluate_policy(pi_c, h_MDP = h_MDP)
                    if h_maxr>0 and self.maxrrec: value += h_maxr*self.maxr
                    return value, pi_c.policy, pi_c.clustering

                terminal_dists, terminal_probs = self.terminal_probabilities(pi_c)
                pi_c = self.cluster_policy(pi_c, terminal_dists, terminal_probs, h-idx)
                policyval_new = self.evaluate_policy(pi_c, terminal_dec = True, \
                        probs_terminal = None, dists_terminal = None, rh = h-idx, h_MDP = h_MDP)
                pi_c.heuristics.append(policyval_new)

                if pi_c.depth == None: 
                    pi_c.depth = idx

                while len(pi_c.heuristics) > pi_c.depth+1 and \
                   (pi_c.depth < self.q_depth or pi_c.depth <= idx - self.maxrec or pi_c.heuristics[pi_c.depth] + 1e-8 > min(pi_c.heuristics[(pi_c.depth+1):])): 
                    pi_c.depth += 1
                
                pi_c.policy.append([[] for a in range(self.nagents)])
                
                idx += 1
                aidx = 0
                policyidx = 0
            else: policyidx = len(pi_c.policy[idx-1][aidx])
            
            if idx == h and aidx == self.nagents - 1:
                  # only the policy of the last agent in the last period needs to be determined
                  # compute the best response for each oh of the last agent
                new_pi_c = pi_c.policy_copy_laststage(idx-1, aidx)            
                policyval = self.evaluate_policy(new_pi_c, h = h-1)

                if self.nagents > 1: act_allohna = pi_c.policy[-1][0]
                else: act_allohna = [0]
                for a in range(1, self.nagents-1):
                    act_allohna = product(act_allohna, pi_c.policy[-1][a], self.a_prod[a])
                len_ohna = len(act_allohna)
                if h_MDP>0 and self.rec:
                    ctr_allohna = [self.maxa**self.nagents + sum([(act//self.a_prod[a])%self.nacts_factor[a] * self.maxa**(self.nagents-1-a) 
                                     for a in range(self.nagents-1)]) for act in act_allohna]
                                     
                for oha in range(pi_c.ncluster[-1][aidx]):
                    reward_idxs = [pi_c.dists[h-1][ohna + oha*len_ohna]*self.nactions + act_allohna[ohna] for ohna in range(len_ohna)]                        
                    val_acts = [sum([self.reward_list[reward_idxs[ohna]+act*self.a_prod[aidx]] * pi_c.prob[h-1][ohna + oha*len_ohna] 
                                  for ohna in range(len_ohna)]) for act in range(self.nacts_factor[aidx])] 
                       # terminal reward heuristics
                    if h_MDP>0 and self.rec:
                        val_acts_curmax = -math.inf
                        p_total = sum([pi_c.prob[h-1][ohna + oha*len_ohna] for ohna in range(len_ohna)])
                        acts = list(range(self.nacts_factor[aidx]))
                        acts.sort(key = lambda idx : -val_acts[idx])  # process actions by decreasing immediate reward
                          # process observation histories by decreasing probability
                        ohna_sorted = [ohna for ohna in range(len_ohna) if pi_c.prob[h-1][ohna + oha*len_ohna] != 0]
                        ohna_sorted.sort(key = lambda ohna: -pi_c.prob[h-1][ohna + oha*len_ohna])

                        for act in acts:
                            if val_acts_curmax != -math.inf:
                                  # first compute an MDP heuristic to see if this action has a chance of being the best
                                val_acts[act] += p_total*self.maxMDP[h_MDP-1]
                                if val_acts[act] < val_acts_curmax: continue
                                MDP_ohnas = dict()
                                for ohna in ohna_sorted:
                                    prob_ohna = pi_c.prob[h-1][ohna + oha*len_ohna]
                                    init_ohna = pi_c.dists[h-1][ohna + oha*len_ohna]
                                    states_ohna, dist_ohna = self.dists[pi_c.dists[h-1][ohna + oha*len_ohna]]
                                    ctr = ctr_allohna[ohna] + act
                                    act_full = act_allohna[ohna] + act * self.a_prod[aidx]
                                    MDP_ohnas[ohna] = (sum([self.MDP_heuristic[h_MDP][ctr*self.nstates+s]*dist_ohna[idx] for idx, s in enumerate(states_ohna)]) \
                                        - self.reward_list[init_ohna*self.nactions + act_full])                    
                                    val_acts[act] += prob_ohna * (MDP_ohnas[ohna] - self.maxMDP[h_MDP-1]) 
                                    if val_acts[act] < val_acts_curmax: break
                                if val_acts[act] < val_acts_curmax: continue
                            
                            acts_all = [act_allohna[ohna]+act*self.a_prod[aidx] for ohna in range(len_ohna)] 

                            for ohna in ohna_sorted:                                
                                prob_ohna = pi_c.prob[h-1][ohna + oha*len_ohna] 
                                dist_ohna = pi_c.dists[h-1][ohna + oha*len_ohna] 
                                act_all = acts_all[ohna]
                                
                                terminal_dists, terminal_probs = self.get_terminal(dist_ohna, act_all)
                                for prob, init in zip(terminal_probs, terminal_dists):   
                                    if prob != 0:               
                                        val_acts[act] += prob_ohna * prob * self.get_terminalMDP(init, h_MDP)
                            
                                if val_acts_curmax != -math.inf:
                                    val_acts[act] -= prob_ohna * MDP_ohnas[ohna]
                                    if val_acts[act] < val_acts_curmax:
                                        break # this action can never beat the current maximum

                            val_acts_curmax = max(val_acts_curmax, val_acts[act])
                    
                    val_act_max = max(val_acts)

                    new_pi_c.policy[idx-1][aidx][oha] = val_acts.index(val_act_max) 
                    policyval += val_act_max
                new_pi_c.heuristics = [policyval]

                if abs(policyval-value) < 1e-12:    #policyval equals heuristic, up to rounding error; so policy is optimal
                    if h_maxr>0 and self.maxrrec: policyval += h_maxr*self.maxr
                    return policyval, new_pi_c.policy, new_pi_c.clustering
                if policyval + 1e-8 >= policyvalfound:
                    heappush(q, (-policyval, -next(unique), new_pi_c))
                
                if policyvalfound < policyval:
                    policyvalfound = max(policyvalfound, policyval)
                    q = [q_elt for q_elt in q if -q_elt[0] + 1e-8 >= policyvalfound]
                    heapify(q)
                
            
            else: 
                p_total = 1
                new_pi_cs = []
                lena = self.nacts_factor[aidx]
                for p in range(lena):
                    new_pi_cs.append(pi_c.policy_copy(idx-1, aidx))
                    new_pi_cs[p].policy[idx-1][aidx].append(p)
                    
                if pi_c.depth != None:
                    policydelta = [0]*(idx - pi_c.depth)
                    policydelta_new = [[0]*(idx - pi_c.depth) for i in range(lena)]
                    
                    depth = idx - 1
                    oh_as_set = [policyidx]      
                    rh = h - depth
                    d = depth - pi_c.depth
                    div, _ = cumprod(pi_c.ncluster[depth])
                    
                    if pi_c.depth != 0:
                        Oh_change = lists_product2(aidx, [policyidx], pi_c.ncluster[depth], div, self.nagents)
                        if self.p_threshold_expand != 0:
                            p_total = sum([pi_c.prob[depth][oh] for oh in Oh_change])
                        
                        for oh in Oh_change:
                            init = pi_c.dists[depth][oh] 
                            if init != -1: 
                                p_oh = pi_c.prob[depth][oh]
    
                                short_ctr = 1
                                for a in range(aidx):
                                    short_ctr = short_ctr * self.maxa + pi_c.policy[depth][a][(oh//div[a])%pi_c.ncluster[depth][a]]
                                
                                heuristic = self.dec_heuristic.get((rh, init, short_ctr, 0, h_MDP))
                                if heuristic == None:
                                    if short_ctr == 1: self.compute_heuristic_init(rh, init, h_MDP)
                                    else: self.compute_heuristic1(pi_c, oh, depth, rh, init, short_ctr, aidx, h_MDP)
                                    heuristic = self.dec_heuristic[(rh, init, short_ctr, 0, h_MDP)]
                                policydelta[d] += p_oh * heuristic
                                
                                for p in range(lena):
                                    short_ctr_new = short_ctr * self.maxa + p
                                    heuristic = self.dec_heuristic.get((rh, init, short_ctr_new, 0, h_MDP))
                                    if heuristic == None:
                                        self.compute_heuristic1(new_pi_cs[p], oh, depth, rh, init, short_ctr_new, aidx+1, h_MDP)
                                        heuristic = self.dec_heuristic[(rh, init, short_ctr_new, 0, h_MDP)]
                                    policydelta_new[p][d] += p_oh * heuristic
                    else:
                        short_ctr = 1
                        for a in range(aidx):
                            short_ctr = short_ctr * self.maxa + pi_c.policy[0][a][0]
                        
                        heuristic = self.get_terminalMDP(0, h, short_ctr)
                        policydelta[d] += heuristic
                        
                        for p in range(lena):
                            short_ctr_new = short_ctr * self.maxa + p
                            heuristic = self.get_terminalMDP(0, h, short_ctr_new)
                            policydelta_new[p][d] += heuristic
                                

                    for depth in range(idx-2, pi_c.depth-1, -1): 
                        
                        oh_asnew = set()                                         # compute all ohs at stage depth from which the oh for
                        if len(oh_as_set) == pi_c.ncluster[depth+1][aidx]:       # which an action is added currently, can be reached
                            oh_asnew = set(range(pi_c.ncluster[depth][aidx]))                                                        
                        else:
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
                                ctup = self.shorten_cluster(pi_c, oh, depth)
                                heuristic = self.dec_heuristic.get((rh, init, short_ctr, ctup, h_MDP))
                                if heuristic == None:
                                    self.dec_heuristic[(rh, init, short_ctr, ctup, h_MDP)] = \
                                      self.dec_heuristic[(rh, init, short_ctr, self.prevcluster[ctup], h_MDP)]
                                    heuristic = self.dec_heuristic[(rh, init, short_ctr, ctup, h_MDP)] 
                                policydelta[d] += p_oh * heuristic
                                
                                for p in range(lena):
                                    short_ctr_new = short_ctr * self.maxa + p
                                    heuristic = self.dec_heuristic.get((rh, init, short_ctr_new, ctup, h_MDP))
                                    if heuristic == None:
                                        self.compute_heuristic2(new_pi_cs[p], oh, depth, rh, init, short_ctr_new, ctup, h_MDP, aidx)
                                        heuristic = self.dec_heuristic[(rh, init, short_ctr_new, ctup, h_MDP)]
                                    policydelta_new[p][d] += p_oh * heuristic
                                    
                expand_all = (self.p_threshold_expand == 0) or (p_total  > self.p_threshold_expand)
                best = (-math.inf, None)
                for p in range(lena):
                    if pi_c.depth != None:
                        for depth in range(pi_c.depth, idx): 
                            d = depth - pi_c.depth                               
                            new_pi_cs[p].heuristics[depth] = pi_c.heuristics[depth] + policydelta_new[p][d] - policydelta[d]
                        policyval = min(value, min(new_pi_cs[p].heuristics[new_pi_cs[p].depth:]))
                        if abs(policyval-value)<1e-12: policyval = value    # prevent rounding errors from leading to problems
                    else: policyval = math.inf
                    
                    if pi_c.depth == None and policyidx == pi_c.ncluster[idx-1][aidx]-1 and aidx == self.nagents - 1:
                        terminal_dists, terminal_probs = self.get_terminal(new_pi_cs[p].dists[0][0], 
                              sum([new_pi_cs[p].policy[0][a][0]*self.a_prod[a] for a in range(self.nagents)]))
                        new_pi_cs[p].values = new_pi_cs[p].values.copy()
                        policyval = self.evaluate_policy(new_pi_cs[p], terminal_dec = True, \
                                      probs_terminal = terminal_probs, dists_terminal = terminal_dists, rh = h-idx, h_MDP = h_MDP) 
                    
                    if expand_all and policyval + 1e-8 >= policyvalfound:
                        heappush(q, (-policyval, -next(unique), new_pi_cs[p]))
                    elif policyval > best[0]:
                        best = (policyval, new_pi_cs[p])
                
                if not expand_all:
                    heappush(q, (-best[0], -next(unique), best[1]))

            # optional debugging sequence
            # if self.output and (ctr < 100 or (ctr%100 == 0 and ctr<10000) or ctr%10000 == 0) and init_call:
            #     qv = q.copy()
            #     for _ in range(len(qv)): 
            #         valv, _, pi_debug = heappop(qv)
            #         print("> openNode value: ", -valv, pi_debug.heuristics, policyvalfound, pi_debug.depth, pi_debug.policy, pi_debug.values)
            #     sys.stdout.flush()
            #     time.sleep(1)