
def cumprod(lens):           # cumprod takes an array [l_0, l_1, ..., l_{n-1}] and returns the array of cumulative products
    ll = len(lens)           # [1, l_0, l_0l_1, ..., l_0l_1...l_{n-2}], as well as the full product l_0l_1...l_{n-1} separately.
    div = [1]*ll
    for idx in range(ll-1):
        div[idx+1] = div[idx] * lens[idx]
    return(div, div[ll-1]*lens[ll-1])

class DecPOMDPverify:
    
    def __init__(self, nagents, nstates, nactions, nobs, transitions, 
                    obs, rewards, init_beliefs, nacts_factor, nobs_factor):
          # nagents: integer k, number of agents
          # nstates: integer n; the states are numbered 0 up to n-1
          # nactions: total number of actions
          # nobs: total number of observations
          # transitions: for each action a in A, and each state s in S, a probability distribution 
          #   over the states in S, represented as one list of length |A| * |S| * |S|
          # obs: for each action a in A, and each new state s in S, a probability distribution 
          #   over the observations in obs_list, represented as one list of length |A| * |S| * |O|
          # rewards: reward of action a in state s, represented as one list of length |A| * |S|
          # init_beliefs: initial belief state, common to all agents
          # nacts_factor: list, giving the number of actions for each agent
          # nobs_factor: list, giving the number of observations for each agent
          
        self.nagents = nagents
        self.nstates = nstates 
        self.nactions = nactions
        self.transitions = transitions
        self.nobs = nobs
        self.obs = obs
        self.rewards = rewards
        self.init_beliefs = init_beliefs
        self.nacts_factor = nacts_factor
        self.nobs_factor = nobs_factor
        
        self.nsq = nstates ** 2
        self.nso = nstates * nobs
        self.a_prod = [1]*(self.nagents+1)
        self.o_prod = [1]*(self.nagents+1)
        for idx in range(self.nagents):
            self.a_prod[idx+1] = self.a_prod[idx] * self.nacts_factor[idx]
            self.o_prod[idx+1] = self.o_prod[idx] * self.nobs_factor[idx]
            

    def evaluate_policy(self, pi, pi_cluster):
        h = len(pi)
        reward = 0
        probs = self.init_beliefs
        ncluster = [[1]*self.nagents] + [[max([max(c) for c in cluster_a])+1 for cluster_a in cluster_h] for cluster_h in pi_cluster] 
        for idx in range(h):
            div, prod = cumprod(ncluster[idx])
            if idx < h-1:
                newdiv, newprod = cumprod(ncluster[idx+1]) 
                newprobs = [0]*(self.nstates * newprod)
            for idx_old in range(self.nstates * prod):
                if probs[idx_old] != 0:
                    idx_old_a = [idx_old//(self.nstates * div[a])%ncluster[idx][a] for a in range(self.nagents)]
                    act = sum([pi[idx][a][idx_old_a[a]] * self.a_prod[a] for a in range(self.nagents)])
                    scur = idx_old % self.nstates
                    reward += probs[idx_old] * self.rewards[act * self.nstates + scur]
                    if idx < h-1:
                        for s_idx, snew in enumerate(self.transitions[act * self.nstates + scur][0]):
                            p_snew = self.transitions[act * self.nstates + scur][1][s_idx]
                            for o_idx, o in enumerate(self.obs[act * self.nstates + snew][0]):
                                p_obs = self.obs[act * self.nstates + snew][1][o_idx]
                                newcluster_idx = 0
                                for a in range(self.nagents):
                                    nc = pi_cluster[idx][a][idx_old_a[a]][(o//self.o_prod[a])%self.nobs_factor[a]]
                                    if nc == -1:
                                        nc = 0 # instead arbitrarily assign cluster 0
                                    newcluster_idx += nc * newdiv[a]
                                newprobs[newcluster_idx * self.nstates + snew] += probs[idx_old] * p_snew * p_obs
                
            if idx < h-1: 
                probs = newprobs
                assert abs(sum(newprobs)-1) < 1e-12, newprobs
        return reward
    
    
