import numpy as np
import math

def _log_birth_density(times, phi, tCutoff=1e4, condition_on_n=True):

    """
    log probability of coalescence times given Yule process with splitting rate phi
    """

    n = len(times) + 1 #number of samples is 1 more than number of coalescence events
    T = min(tCutoff, times[-1]) #furthest time in past we are interested in is the min of the cutoff time and the oldest coalesence event
    times = T - times #switch to forward in time perspective from T
    times = np.sort(times[times>0]) #remove older coalescence events (we'll assume panmixia beyond tCutoff, so this will cancel out in importance sampling)
    n0 = n-len(times) #number of lineages at time T
 iEpoch = int(np.digitize(np.array([t]),epochs)[0]-1) #epoch 
    t1 = epochs[iEpoch] #time at which the previous epoch ended
    Lambda = intensityMemos[iEpoch] #intensity up to end of previous epoch
    Lambda += 1/(2*N[iEpoch]) * (t-t1) #add intensity for time in current epoch
    return Lambda

def _coal_intensity_memos(epochs, N):

    """
    coalescence intensity up to the end of each epoch
    """

    Lambda = np.zeros(len(epochs))
    for ie in range(1,len(epochs)):
        t0 = epochs[ie-1] #start time
        t1 = epochs[ie] #end time
        Lambda[ie] = (t1-t0) #elapsed time
        Lambda[ie] *= 1/(2*N[ie-1]) #multiply by coalescence intensity
        Lambda[ie] += Lambda[ie-1] #add previous intensity
    return Lambda

