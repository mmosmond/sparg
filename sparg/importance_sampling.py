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

    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    # probability of each coalescence time
    for i,t in enumerate(times): #for each coalescence event i at time t
        k = n0+i #number of lineages before the event
        logp += np.log(k*phi) - k * phi *  (t - prevt) #log probability of waiting time  t-prevt (waiting times are exponentially distributed with rate k*phi)
        prevt = t #update time

    #add probability of no coalescence to present time (from most recent coalescence to present, with n samples and rate n*phi)
    logp += - n * phi * (T - prevt)

    #condition on having n samples from n0 in time T
    if condition_on_n:
        logp -= np.log(math.comb(n-1,n-n0)*(1-np.exp(-phi*T))**(n-n0)) - phi * n0 * T # see page 234 of https://www.pitt.edu/~super7/19011-20001/19531.pdf for two different expressions

    return logp

def _log_coal_density(times, Nes, epochs=None, tCutoff=None):
    
    """
    log probability of coalescent times under standard neutral/panmictic coalescent
    """

    if Nes is None:
        print('must supply Ne to get probability of coalescence times')
        return

    if epochs is None and len(Nes) == 1:
        epochs = [0, max(times)] #one big epoch
        Nes = [Nes[0], Nes[0]] #repeat the effective population size so same length as epochs 

    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    prevLambda = 0 #initialize coalescent intensity
    n = len(times) + 1 #number of samples
    if tCutoff is not None:
        times = times[times < tCutoff] #ignore old times
    myIntensityMemos = _coal_intensity_memos(epochs, Nes) #intensities up to end of each epoch

    # probability of each coalescence time
    for i,t in enumerate(times): #for each coalescence event i at time t
        k = n-i #number of lineages remaining
        kchoose2 = k * (k - 1) / 2 #binomial coefficient
        Lambda = _coal_intensity_using_memos(t, epochs, myIntensityMemos, Nes) #coalescent intensity up to time t
        ie = np.digitize(np.array([t]), epochs) #epoch at the time of coalescence
        logpk = np.log(kchoose2 * 1 / (2 * Nes[ie])) - kchoose2 * (Lambda - prevLambda) #log probability (waiting times are time-inhomogeneous exponentially distributed)
        logp += logpk
        prevt = t
        prevLambda = Lambda

    # now add the probability of lineages not coalescing by tCutoff
    if k > 1 and tCutoff is not None: #if we have more than one lineage remaining
        k -= 1 #after the last coalescence event we have one less sample
        kchoose2 = k * (k - 1) / 2 #binomial coefficient
        Lambda = _coal_intensity_using_memos(tCutoff, epochs, myIntensityMemos, Nes) #coalescent intensity up to time tCutoff
        logPk = - kchoose2 * (Lambda - prevLambda) #log probability of no coalescence
        logp += logPk

    return logp[0]

def _coal_intensity_using_memos(t, epochs, intensityMemos, Nes):
    
    """
    add coal intensity up to time t
    """

    iEpoch = int(np.digitize(np.array([t]), epochs)[0] - 1) #epoch 
    t1 = epochs[iEpoch] #time at which the previous epoch ended
    Lambda = intensityMemos[iEpoch] #intensity up to end of previous epoch
    Lambda += 1 / (2 * Nes[iEpoch]) * (t - t1) #add intensity for time in current epoch
    return Lambda

def _coal_intensity_memos(epochs, Nes):

    """
    coalescence intensity up to the end of each epoch
    """

    Lambda = np.zeros(len(epochs))
    for ie in range(1, len(epochs)):
        t0 = epochs[ie - 1] #start time
        t1 = epochs[ie] #end time
        Lambda[ie] = (t1 - t0) #elapsed time
        Lambda[ie] *= 1 / (2 * Nes[ie - 1]) #multiply by coalescence intensity
        Lambda[ie] += Lambda[ie - 1] #add previous intensity

    return Lambda

