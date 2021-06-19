import numpy as np
import math

def _log_birth_density(times, phi, tCutoff=None, condition_on_n=True):
    'log probability of coalescence times given Yule process with splitting rate phi'

    n = len(times) + 1 #number of samples is 1 more than number of coalescence events
    if tCutoff is None:
        T = times[-1] #tmrca
    else:
        T = min(tCutoff, times[-1]) #furthest time in past we are interested in is the min of the cutoff time and tmrca
    times = T - times #switch to forward in time perspective from T
    times = np.sort(times[times>0]) #remove older coalescence events (we'll assume panmixia beyond tCutoff, so this will cancel out in importance sampling)
    n0 = n - len(times) #number of lineages at time T

    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    # probability of each coalescence time
    for i,t in enumerate(times): #for each coalescence event i at time t
        k = n0 + i #number of lineages before the event
        logp += np.log(k * phi) - k * phi *  (t - prevt) #log probability of waiting time  t-prevt (waiting times are exponentially distributed with rate k*phi)
        prevt = t #update time

    #add probability of no coalescence to present time (from most recent coalescence to present, with n samples and rate n*phi)
    logp += - n * phi * (T - prevt)

    #condition on having n samples from n0 in time T
    if condition_on_n:
        logp -= np.log(math.comb(n - 1, n - n0) * (1 - np.exp(-phi * T))**(n - n0)) - phi * n0 * T # see page 234 of https://www.pitt.edu/~super7/19011-20001/19531.pdf for two different expressions

    return logp

