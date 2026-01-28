
class DecPOMDPrandom:
    
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
        
    def evaluate_random_policy(self, h):
        reward = 0
        probs = self.init_beliefs
        for idx in range(h):
            newprobs = [0]*self.nstates
            for sold in range(self.nstates):
                if probs[sold] != 0:
                    for act in range(self.nactions):
                        reward += probs[sold] * self.rewards[act * self.nstates + sold] / self.nactions
                        if idx < h-1:
                            for s_idx, snew in enumerate(self.transitions[act * self.nstates + sold][0]):
                                newprobs[snew] += probs[sold] * self.transitions[act * self.nstates + sold][1][s_idx] / self.nactions 
                                
            if idx < h-1: 
                probs = newprobs
                assert abs(sum(newprobs)-1) < 1e-12, newprobs
        return reward
    
    
