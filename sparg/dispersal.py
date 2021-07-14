from scipy.optimize import minimize
from sparg.utils import _sigma_phi, _logsumexp, _lognormpdf
from sparg.importance_sampling import _log_birth_density

def dispersal(locations, coal_times, pcoals, shared_times, samples, x0=[0,0], bnds=(None,None), n=None, important=True, tCutoff=1e4, tsplits=[], quiet=False, method='L-BFGS-B', options=None, scale_phi=1, remove_missing=False):
    
    """
    find maximum likelihood dispersal given locations and tree info (coal_times, pcoals, shared_times, samples)'
    """

    M = len(pcoals[0]) #number of trees per locus 
    if n == None:
        n = M #number of trees to use per locus

    f = _sum_mc(locations, coal_times, pcoals, shared_times, samples, n=n, important=important, tCutoff=tCutoff, tsplits=tsplits, scale_phi=scale_phi, remove_missing=remove_missing) #negative composite log likelihood ratio

    if not quiet:
        print('searching for maximum likelihood parameters...')
        t0 = time.time()

    m = minimize( f, x0 = x0, bounds = bnds, method=method, options=options) #find MLE

    if not quiet:
        print('the max is ', m.x)
        print('finding the max took', time.time()-t0, 'seconds')

    return m

def _sum_mc(locations, coal_times, pcoals, shared_times, samples, n = None, important = True, tCutoff = 1e6, tsplits=[], scale_phi=1, remove_missing=False):
    
    """
    sum monte carlo estimates of log likelihood ratios across loci
    """

    M = len(pcoals[0]) #unmber of trees per locus
    if n == None:
        n = M #number of trees to use per locus
    elif n > M:
        print('must have n<=M: cant use more trees than were sampled')
        exit

    def sumf(x):
        g = 0
        nloci = len(pcoals)
        for i in range(nloci):
            g -= _mc(shared_times[i][0:n], #use n samples at each locus
                    samples[i][0:n],
                    pcoals[i][0:n],
                    coal_times[i][0:n],
                    _sigma_phi(x, tsplits, important)[1]/scale_phi, #phi (prescaled so that estimated phi is scale_phi times true value, to put on same scale as dispersal)
                    _sigma_phi(x, tsplits, important)[0], #sigma
                    locations,
                    tCutoff,
                    important,
                    tsplits, remove_missing)
        return g

    return sumf

def _mc(shared_times, samples, pcoals, coal_times, phi, Sigma, locations, tCutoff=1e4, important=True, tsplits=[], remove_missing=False):
    
    """
    estimate log likelihood ratio of the locations given parameters (Sigma,phi) vs data given standard coalescent with Monte Carlo
    """

    M = len(pcoals) #number of samples of branch lengths
    LLRs = np.zeros(M) #log likelihood ratios
    for i,shared_time in enumerate(shared_times):

        LLRs[i] = _loglikelihood_ratio(shared_time, samples[i], pcoals[i], coal_times[i], phi, Sigma, locations, tCutoff, important, tsplits, remove_missing)

    LLRhat = _logsumexp(LLRs) - np.log(M) #average over trees

    return LLRhat #monte carlo estimate of log likelihood ratio

def _loglikelihood_ratio(shared_times, samples, pcoal, coal_times, phi, Sigma, locations, tCutoff=1e4, important=True, tsplits=[], remove_missing=False):
        
    """ 
    log likelihood of locations given parameters and tree summaries
    """

    LLR = 0
    for i,shared_time in enumerate(shared_times):
        if len(shared_time) > 1: #need at least two samples in subtree to mean center and still have info on dispersal rate
            LLR += _location_loglikelihood(locations[samples[i]], shared_time, Sigma, tsplits, remove_missing) #log likelihood of locations given shared evolutionary times, dispersal matrix, and MRCA location
    if important:
        LLR += _log_birth_density(coal_times, phi, tCutoff) #log probability of coalescence times given pure birth process with rate phi
        LLR -= pcoal #log probability density of coalescence times under standard coalescent with varying population size and cutoff

    return LLR

def _location_loglikelihood(locations, shared_time, Sigma, tsplits=[], remove_missing=False):
    
    """
    log likelihood of locations given shared times and dispersal matrix
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
        #print('singular matrix')
        return 0 

