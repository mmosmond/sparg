from scipy.optimize import minimize
from sparg.utils import _sigma_phi, _logsumexp, _lognormpdf
from sparg.importance_sample import _log_birth_density
import time
import numpy as np

def estimate(locations, shared_times, samples, x0, bnds=None, important=True, coal_times=None, logpcoals=None, n=None, tCutoff=None, tsplits=[], quiet=False, method='L-BFGS-B', options=None, scale_phi=1, remove_missing=False):
    
    """
    find maximum likelihood dispersal given locations and tree info (coal_times, logpcoals, shared_times, samples)'
    """

    M = len(samples[0]) #number of trees per locus 
    if n == None:
        n = M #number of trees to use per locus

    if bnds is None:
        bnds = tuple([(None,None) for _ in x0]) #make the same length as x0

    f = _sum_mc(locations, shared_times, samples, coal_times=coal_times, logpcoals=logpcoals, n=n, important=important, tCutoff=tCutoff, tsplits=tsplits, scale_phi=scale_phi, remove_missing=remove_missing) #negative composite log likelihood ratio

    if not quiet:
        print('searching for maximum likelihood parameters...')
        t0 = time.time()

    m = minimize(f, x0=x0, bounds=bnds, method=method, options=options) #find MLE

    if not quiet:
        print('the max is ', m.x)
        print('finding the max took', time.time()-t0, 'seconds')

    return m

def _sum_mc(locations, shared_times, samples, coal_times=None, logpcoals=None, n=None, important=True, tCutoff=None, tsplits=[], scale_phi=1, remove_missing=False):
    
    """
    sum monte carlo estimates of log likelihood ratios across loci
    """

    M = len(samples[0]) #unmber of trees per locus
    if n == None:
        n = M #number of trees to use per locus
    elif n > M:
        print('must have n<=M: cant use more trees than were sampled')
        return 

    def sumf(x):
        Sigma, phi = _sigma_phi(x, tsplits, important) 
        if important:
            phi = phi / scale_phi # scaled so that estimated phi is scale_phi times true value, to put on same scale as dispersal
        g = 0
        nloci = len(samples)
        for i in range(nloci):
            coal_timesi = None
            if coal_times is not None:
                coal_timesi = coal_times[i][0:n]
            logpcoalsi = None
            if logpcoals is not None:
                logpcoalsi = logpcoals[i][0:n]
            g -= _mc(
                     locations=locations,
                     shared_times=shared_times[i][0:n], #use n samples at each locus
                     samples=samples[i][0:n],
                     coal_times=coal_timesi,
                     logpcoals=logpcoalsi,
                     Sigma=Sigma,
                     phi=phi, 
                     tCutoff=tCutoff,
                     important=important,
                     tsplits=tsplits, 
                     remove_missing=remove_missing
                    )
        return g

    return sumf

def _mc(locations, shared_times, samples, Sigma, phi=None, coal_times=None, logpcoals=None, tCutoff=None, important=True, tsplits=[], remove_missing=False):
    
    """
    estimate log likelihood ratio of the locations given parameters (Sigma,phi) vs data given standard coalescent with Monte Carlo
    """

    M = len(samples) #number of samples of branch lengths
    LLRs = np.zeros(M) #log likelihood ratios

    if coal_times is None and logpcoals is None:
        coal_times, logpcoals = [None for _ in samples], [None for _ in samples] 

    for i, (shared_time, sample, coal_time, logpcoal) in enumerate(zip(shared_times, samples, coal_times, logpcoals)):

        LLRs[i] = _loglikelihood_ratio(locations, shared_time, sample, Sigma, phi, coal_time, logpcoal, tCutoff, important, tsplits, remove_missing)

    LLRhat = _logsumexp(LLRs) - np.log(M) #average over trees at this locus

    return LLRhat #monte carlo estimate of log likelihood ratio

def _loglikelihood_ratio(locations, shared_times, samples, Sigma, phi=None, coal_times=None, logpcoals=None, tCutoff=None, important=True, tsplits=[], remove_missing=False):
        
    """ 
    log likelihood of locations given parameters and tree summaries
    """

    LLR = 0
    for shared_time, sample in zip(shared_times, samples): #looping over subtrees
        if len(shared_time) > 1: #need at least two samples in subtree to mean center and still have info on dispersal rate
            LLR += _location_loglikelihood(locations[sample], shared_time, Sigma, tsplits, remove_missing) #log likelihood of locations given shared evolutionary times, dispersal matrix, and MRCA location
    if important:
        LLR += _log_birth_density(coal_times, phi, tCutoff) #log probability of coalescence times given pure birth process with rate phi
        LLR -= logpcoals #log probability density of coalescence times under standard coalescent with varying population size and cutoff

    return LLR

def _location_loglikelihood(locations, shared_time, Sigma, tsplits=[], remove_missing=False):
    
    """
    log likelihood of locations given mean location and dispersal covariance 
    """

    if remove_missing:
        # remove samples without location data
        not_missing = np.argwhere(np.isfinite(locations).any(axis=1)).flatten()
        if len(not_missing) < len(locations):
            #print('removing', len(locations) - len(not_missing), 'samples with missing locations from dispersal estimate')
            locations = locations[not_missing]
            shared_time = shared_time[not_missing][:,not_missing]

    n = len(locations) #number of samples
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[0:-1]; #matrix for mean centering and dropping one sample

    # mean center locations
    locs = np.matmul(Tmat, locations) #mean center locations and drop last sample (bc lost a degree of freedom when taking mean); see Lee & Coop 2017 Genetics eq A.16; note numpy broadcasts across the columns in locs
    x = np.transpose(locs).flatten() #write locations as vector (all x locations first, then all y, etc)
    mean = np.zeros(len(x)) #mean of mvn

    # if we don't allow dispersal to vary across time
    if tsplits == []:

        stime = np.matmul(Tmat, np.matmul(shared_time, np.transpose(Tmat))) #mean center shared times
        cov = np.kron(Sigma[0], stime) #covariance of mvn

    else:

        # times in each epoch
        tsplits = [0] + tsplits #append 0 to front of list of split times
        Ts = [tsplits[i + 1] - tsplits[i] for i in range(len(tsplits) - 1)] #amount of time in all but the most distant epoch
        T = shared_time[0][0] #tmrca
        Ts.append(T - tsplits[-1]) #add time in the most distant epoch

        #shared times in each epoch
        stimes = [] #place to store shared time matrix in each epoch
        stimes.append(np.minimum(shared_time, Ts[-1])) #shared times in most distant epoch
        for i in range(len(tsplits) - 1):
            stimes.append(np.minimum(np.maximum(shared_time - (T - tsplits[-1 - i]), 0), Ts[-2 - i])) #shared times in each most recent epochs (note this means the theta produced will go in chronological order, from most distant to most recent)

        # mean center shared times in each epoch (note this makes each row and column sum to zero)    
        for i,stime in enumerate(stimes):
            stimes[i] = np.matmul(Tmat, np.matmul(stime, np.transpose(Tmat))) #mean center covariance matrix (see eq A.18 in Lee & Coop 2017 Genetics)

        # sum up covariance matrices
        cov = np.zeros((len(x), len(x))) #covariance matrix of mvn
        for i,stime in enumerate(stimes):
            cov += np.kron(Sigma[i], stime) #covariance matrix for epoch i

    if np.linalg.matrix_rank(cov) == len(cov):
        return _lognormpdf(x, mean, cov)
    else:
        print('singular matrix')
        return 

def mle(locations, shared_time):
    
    """
    MLE dispersal rate
    """

    n = len(locations) #number of samples
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[0:-1]; #matrix for mean centering and dropping one sample
    locs = np.matmul(Tmat, locations) #mean center locations and drop last sample (bc lost a degree of freedom when taking mean)
    stime = np.matmul(Tmat, np.matmul(shared_time, np.transpose(Tmat))) #mean center shared times
    Tinv = np.linalg.pinv(np.array(stime)) #inverse of shared time matrix

    return np.matmul(np.matmul(np.transpose(locs), Tinv), locs) / (n - 1) #mle dispersal rate 

