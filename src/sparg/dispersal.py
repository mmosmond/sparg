from scipy.optimize import minimize
from sparg.utils import _sigma_phi, _logsumexp, _lognormpdf
from sparg.importance_sample import _log_birth_density
import time
import numpy as np

def mle(locations, shared_time):    
    """Maximum likelihood estimate of dispersal rate given sample locations and shared times between sample lineages.

    Parameters:
        locations (array-like): Locations of samples. An n x d array, where n is the number of samples and d is the number of spatial dimensions.
        shared_time (array-like): Shared times between sample lineages. An n x n array, where n is the number of samples (arranged in same order as locations). 
    
    Returns:
        Maximum likelihood dispersal rate as a d x d covariance matrix.
    """

    n = len(locations) #number of samples
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[0:-1]; #matrix for mean centering and dropping one sample
    locs = np.matmul(Tmat, locations) #mean center locations and drop last sample (bc lost a degree of freedom when taking mean)
    stime = np.matmul(Tmat, np.matmul(shared_time, np.transpose(Tmat))) #mean center shared times
    Tinv = np.linalg.pinv(np.array(stime)) #inverse of shared time matrix

    return np.matmul(np.matmul(np.transpose(locs), Tinv), locs) / (n - 1) #mle dispersal rate 

def estimate(locations, shared_times, samples, x0, bnds=None, important=True, coal_times=None, logpcoals=None, scale_phi=1, tCutoff=None, tsplits=[], method='L-BFGS-B', options=None, n=None):
    """Numerically estimate maximum likelihood dispersal rate (and possibly branching rate) given sample locations and processed trees.

    Parameters:
        locations (array-like): Locations of samples. An n x d array, where n is the number of samples and d is the number of spatial dimensions.
        shared_times (array-like): Shared times between sample lineages. An l x m x s_i x n_j x n_j array, where l is the number of loci, m is the number of sampled trees per locus, s_i is the number of subtrees of tree i, and n_j is the number of samples in subtree j
        samples (array-like): The placement of sample nodes in shared_times, to connect the shared times with the sample locations. Same shape as shared_times.   
        x0 (array-like): Initial guess of parameters. An e * d * (1 + d) / 2 + i array, where e is the number of epochs, d is the number of spatial dimensions, and i is 1 if importance sampling and 0 otherwise. Currently support only d=1 and d=2. Examples: if d=1 and i=0, then x0 = [sigma_1, sigma_2, ..., sigma_e], where sigma_i is a guess of the dispersal rate (standard deviation of Gaussian dispersal kernel) in epoch i. If i=1 then x0 = [sigma_1, sigma_2, ..., sigma_e, phi], where phi is a guess of the branching rate of a Yule model. If d=2 then each sigma_i is replaced with a sigma_i_x, sigma_i_y, and rho_i, representing the standard deviations in each dimension and the correlation between them.
        bnds (tuple): Lower and upper bounds on each parameter being esimated. Often required to keep standard deviations and branching rates positive and correlations between -1 and 1.
        important (boolean): Whether to use importance sampling. 
        coal_times (array-like): Coalescence times. An l x m x (n - 1) array, where l is the number of loci, m is the number of sampled trees per locus, and n is the number of samples. Used to compute the probability of the trees under a Yule model when importance sampling.
        logpcoals (array-like): Log probability of coalescence times under the standard neutral coalescent. An l x m x 1 array, where l is the number of loci and m is the number of sampled trees per locus. Compared to the probability of the trees under a Yule model when importance sampling. 
        scale_phi (float): Scaling factor for branching parameter, phi. We search for the phi * scale_phi that maximizes the likelihood. Scaling phi so that it has a similar variance as the dispersal paramters can speed up the numerical search.
        tCutoff (float): Time in the past to chop the trees into subtrees, effectively ignoring deeper times.
        tsplits (array-like): Split times between epochs. Note that there will be len(tsplits) + 1 epochs.
        method (string): Numerical optimizer method to pass to scipy's minimize. See scipy docs for more info.
        options: Additional options to pass to scipy's minmize. See scipy docs for more info.
        n (int): Number of tree samples to use at each locus. Useful if you have processed M samples at each locus but want to see the effect of using <M.
    
    Returns:
        Dictionary output of scipy's minimize, including the estimated parameters (x).
    """

    M = len(samples[0]) #number of trees per locus 
    if n == None:
        n = M #number of trees to use per locus

    if bnds is None:
        bnds = tuple([(None,None) for _ in x0]) #make the same length as x0

    f = _sum_mc(locations, shared_times, samples, important=important, coal_times=coal_times, logpcoals=logpcoals, scale_phi=scale_phi, tCutoff=tCutoff, tsplits=tsplits, n=n) #negative composite log likelihood ratio

    print('searching for maximum likelihood parameters...')
    t0 = time.time()

    m = minimize(f, x0=x0, bounds=bnds, method=method, options=options) #find MLE

    print('the max is ', m.x)
    print('finding the max took', time.time()-t0, 'seconds')

    return m

def _sum_mc(locations, shared_times, samples, important=True, coal_times=None, logpcoals=None, scale_phi=1, tCutoff=None, tsplits=[], n=None):
    
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
                     Sigma=Sigma,
                     important=important,
                     coal_times=coal_timesi,
                     logpcoals=logpcoalsi,
                     phi=phi, 
                     tCutoff=tCutoff,
                     tsplits=tsplits
                    )
        return g

    return sumf

def _mc(locations, shared_times, samples, Sigma, important=True, coal_times=None, logpcoals=None, phi=None, tCutoff=None, tsplits=[]):
    
    """
    estimate log likelihood ratio of the locations given parameters (Sigma,phi) vs data given standard coalescent with Monte Carlo
    """

    M = len(samples) #number of samples of branch lengths
    LLRs = np.zeros(M) #log likelihood ratios

    if coal_times is None and logpcoals is None:
        coal_times, logpcoals = [None for _ in samples], [None for _ in samples] 

    for i, (shared_time, sample, coal_time, logpcoal) in enumerate(zip(shared_times, samples, coal_times, logpcoals)):

        LLRs[i] = _loglikelihood_ratio(locations, shared_time, sample, Sigma, important=important, coal_times=coal_time, logpcoals=logpcoal, phi=phi, tCutoff=tCutoff, tsplits=tsplits)

    LLRhat = _logsumexp(LLRs) - np.log(M) #average over trees at this locus

    return LLRhat #monte carlo estimate of log likelihood ratio

def _loglikelihood_ratio(locations, shared_times, samples, Sigma, important=True, coal_times=None, logpcoals=None, phi=None, tCutoff=None, tsplits=[]):
        
    """ 
    log likelihood of locations given parameters and tree summaries
    """

    LLR = 0
    for shared_time, sample in zip(shared_times, samples): #looping over subtrees
        if len(shared_time) > 1: #need at least two samples in subtree to mean center and still have info on dispersal rate
            LLR += _location_loglikelihood(locations[sample], shared_time, Sigma, tsplits) #log likelihood of locations given shared evolutionary times, dispersal matrix, and MRCA location
    if important:
        LLR += _log_birth_density(coal_times, phi, tCutoff) #log probability of coalescence times given pure birth process with rate phi
        LLR -= logpcoals #log probability density of coalescence times under standard coalescent with varying population size and cutoff

    return LLR

def _location_loglikelihood(locations, shared_time, Sigma, tsplits=[]):
    
    """
    log likelihood of locations given mean location and dispersal covariance 
    """

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


