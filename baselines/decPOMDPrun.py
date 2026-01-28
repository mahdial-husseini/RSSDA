from decPOMDP import DecPOMDP
from decPOMDPverify import DecPOMDPverify
from decPOMDPrandom import DecPOMDPrandom
import time

def run(nagents, nstates, nacts, nobs, transit, obs, reward, init_beliefs,
    nacts_factor, nobs_factor, horizon, parser, time1):

    if parser.random:
        decRandom = DecPOMDPrandom(nagents, nstates, nacts, nobs,
               transit, obs, reward, init_beliefs, nacts_factor, nobs_factor)
        print(decRandom.evaluate_random_policy(horizon))
        exit()
    
    time0 = time.time()
    dec = DecPOMDP(nagents, nstates, nacts, nobs,
        transit, obs, reward, init_beliefs,
        nacts_factor, nobs_factor, horizon, **vars(parser))
    
    value, pi, pi_cluster = dec.multi_agent_astar(horizon)
    time_toc = time.time()
    
    decVerify = DecPOMDPverify(nagents, nstates, nacts, nobs,
        transit, obs, reward, init_beliefs, nacts_factor, nobs_factor)
    valueVerify = decVerify.evaluate_policy(pi, pi_cluster)
    if abs(valueVerify-value) > 1e-6:
        print("warning: value from verifier (", valueVerify, "), and solver (", value, "), differ substantially",sep='')
    print((valueVerify, pi, pi_cluster))
    print("time (excluding parse)", time_toc-time0)
    print("time (total)", time_toc-time1)