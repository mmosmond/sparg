import numpy as np
from scipy.optimize import minimize
from sparg.utils import _get_focal_index, _lognormpdf, _logsumexp, _sigma_phi
from sparg.importance_sample import _log_birth_density

def locate(nodes, times, locations, shared_times, samples, dispersal_rate=None, x0=None, bnds=((None,None),(None,None)), important=True, coal_times=None, logpcoals=None, tCutoff=None, tsplits=[], method='L-BFGS-B', weight=False, keep=None, BLUP=False, phi=None):
    """Find maximum likelihood locations of ancestors using sample locations, processed trees, and dispersal rate.

    Parameters:
        nodes (array-like): Sample nodes we wish to find ancestors of. An array of length nn, where nn is the number of nodes we are locating ancestors of.
        times (array-like): Times in the past that we wish to locate the ancestors. An array of length t, where t is the number of times we locate ancestors at.
        locations (array-like): Locations of samples. An n x d array, where n is the number of samples and d is the number of spatial dimensions.
        shared_times (array-like): Shared times between sample lineages. An l x m x s_i x n_j x n_j array, where l is the number of loci, m is the number of sampled trees per locus, s_i is the number of subtrees of tree i, and n_j is the number of samples in subtree j
        samples (array-like): The placement of sample nodes in shared_times, to connect the shared times with the sample locations. Same shape as shared_times.   
        dispersal_rate (array-like): Dispersal (and branching) rates to use for each locus. An l x e * d * (1 + d) / 2 + i array, where l is the number of loci, e the number of epochs, d the number of spatial dimensions, and i=1 if importance sampling and 0 otherwise. Currently support only d=1 and d=2. Examples: if l=1, d=1, and i=0, then dispersal_rate = [[sigma_1, sigma_2, ..., sigma_e]], where sigma_i is the dispersal rate (standard deviation of Gaussian dispersal kernel) in epoch i. If i=1 then dispersal_rate = [[sigma_1, sigma_2, ..., sigma_e, phi]], where phi is the branching rate. If d=2 then each sigma_i is replaced with a sigma_i_x, sigma_i_y, and rho_i, representing the standard deviations in each dimension and the correlation between them. We will often want to use the same, genome-wide, dispersal (and branching) rate at all loci, and so if this is dispersal_rate_global then we can just set dispersal_rate = [dispersal_rate_global for _ in len(samples)].
        x0 (array-like): Initial guess of locations. Currently the same for all ancestors being located, so very geographically distinct ancestors may want to use separate instances. An array of length d, where d is the number of spatial dimensions. 
        bnds (tuple): Lower and upper bounds on each location. A d x 2 tuple. 
        important (boolean): Whether to use importance sampling. 
        coal_times (array-like): Coalescence times. An l x m x (n - 1) array, where l is the number of loci, m is the number of sampled trees per locus, and n is the number of samples. Used to compute the probability of the trees under a Yule model when importance sampling.
        logpcoals (array-like): Log probability of coalescence times under the standard neutral coalescent. An l x m x 1 array, where l is the number of loci and m is the number of sampled trees per locus. Compared to the probability of the trees under a Yule model when importance sampling. 
        tCutoff (float): Time in the past to chop the trees into subtrees, effectively ignoring deeper times.
        tsplits (array-like): Split times between epochs. Note that there will be len(tsplits) + 1 epochs.
        method (string): Numerical optimizer method to pass to scipy's minimize. See scipy docs for more info.
        weight (boolean): Whether to return inverse variance weighting of locations (to later calculate weighted average locations over loci)
        keep (array-like): What sample locations to use in locating ancestors. keep=None uses all samples. 
        BLUP (boolean): Whether to calculate the (importance sampled) average of MLE locations over tree samples, rather than numerically find the location that maximizes the (importance sampled) average likelihood.
        phi (float): Branching rate in Yule process, for importance sampling. Typically supplied as part of dispersal_rate, but if calculating BLUPs with importance sampling then only need phi. 

    Returns:
        mle_locations (numpy array): Estimated ancestor locations. A l x t x nn array, where l is the number of loci, t is the number of times, and nn is the number of nodes we find ancestors of.
        weights (numpy array): Inverse variance weights for each location estimate.
    """

    # loop over loci
    mle_locations = []
    weights = []
    for i, (shared_times_i, samples_i) in enumerate(zip(shared_times, samples)):
        if important:
            coal_times_i = coal_times[i]
            logpcoals_i = logpcoals[i]
        else:
            coal_times_i, logpcoals_i = None, None 

        # if we only have one sampled tree at a locus then we can find the MLE analytically
        if len(samples_i) == 1:
            BLUP = True
            #print('only one tree at this locus so calculating MLE locations analytically')
    
        # load dispersal and branching rate at this locus
        if BLUP and tsplits == []:
            SIGMA = None
        else:
            [SIGMA, phi] =  _sigma_phi(dispersal_rate[i], tsplits, important) 
        
        # loop over times
        mle_locations_t = [] #locations for all nodes at this time
        weights_t = [] #weights of locations (measure of uncertainty)
        for time in times:
    
            # loop over nodes
            mle_locations_n = [] #locations for all nodes at particular time at this locus
            weights_n = []
            for node in nodes:
                
                # get location of node i at time t at this locus
                f = _ancestor_location_meanloglikelihood(node, time, locations, shared_times_i, samples_i, SIGMA=SIGMA, important=important, coal_times=coal_times_i, logpcoals=logpcoals_i, phi=phi, tCutoff=tCutoff, tsplits=tsplits, keep=keep, BLUP=BLUP) #mean log likelihood or mean MLE (if BLUP)
    
                if f is None:
                    mle_locations_n.append(np.ones(len(x0)) * np.nan)
                    #print('locating ancestor of %d at time %d failed at a locus' %(focal_node, time))
    
                else:
    
                    # if we want the best linear unbiased predictor (BLUP), ie the importance sampled MLE 
                    if BLUP:
                        mle_locations_n.append(f) #just add mean MLE
    
                        if weight:
                            print('weights not implemented for BLUPs, if you want weights set BLUP=False')
    
                    # find the true MLE by finding the max of the average log likelihood across importance samples
                    else:
    
                        g = lambda x: -f(x) #flip bc we look for min
                        gmin = minimize(g, x0=x0, bounds=bnds, method=method) #find MLE
                        # if numerical search worked
                        if gmin.success:
                            mle_locations_n.append(gmin.x) #append mle location for this node
                        else:
                            mle_locations_n.append(np.ones(len(x0)) * np.nan) #some nans to fill up the matrix but not influence results
                            #print('locating ancestor of %d at time %d failed at a locus' %(focal_node, time))
    
                        # we may want to weight locations over loci depending on how certain the estimate is
                        if weight:
                            if method=='L-BFGS-B':
                                if gmin.success:
                                    if gmin.x[0] > bnds[0][0] and gmin.x[0] < bnds[0][1] and gmin.x[1] > bnds[1][0] and gmin.x[1] < bnds[1][1]: # if did not hit bounds
                                        cov = gmin['hess_inv'].todense() #inverse hessian (negative fisher info) is estimate of covariance matrix
                                        weights_n.append(cov) #append covariance matrix as weights
                                    else:
                                        weights_n.append(np.zeros((len(x0)-1, len(x0)-1))) #dont use this estimate if hit bounds                
                                else:
                                    weights_n.append(np.zeros((len(x0)-1, len(x0)-1))) #dont use this estimate if numerical minimizer failed                
                            else:
                                weights_n.append(1) #weights all equally if don't have the hessian
                                print('choose method=L-BFGS-B if you want weights')
    
            mle_locations_t.append(mle_locations_n) #append locations across all nodes for particular time at particular locus
            weights_t.append(weights_n) #append weights across all nodes for particular time at particular locus

        mle_locations.append(mle_locations_t) #append locations acrocss all nodes and all times for particular locus
        weights.append(weights_t) #append weights acrocss all nodes and all times for particular locus

    if weight:
        return np.array(mle_locations), np.array(weights)
    else:
        return np.array(mle_locations) 

def _ancestor_location_meanloglikelihood(node, time, locations, shared_times, samples, SIGMA=None, important=True, coal_times=None, logpcoals=None, phi=None, tCutoff=None, tsplits=[], keep=None, BLUP=False):
    """
    mean log likelihood of ancestor location (over sampled trees), or mean MLE location if BLUP
    """

    fs = []
    ws = []
    #loop over trees (importance samples) at a locus
    for tree,sample in enumerate(samples): 
        
        # get shared times with ancestor and sort sample locations
        n,m = _get_focal_index(node, sample) #subtree and sample index of focal_node
        sts = shared_times[tree][n] #shared times between all sample lineages in subtree
        atimes = _atimes(sts, time, m) #get shared times between sample lineages and ancestor lineage
        if keep is None:
            alltimes = _alltimes(sts, atimes, keep=keep)
            locs = locations[sample[n]] #locations of samples
        else:
            ixs = np.where(np.in1d(sample[n], keep))[0] #what indices are we keeping location info for
            alltimes = _alltimes(sts, atimes, keep=ixs) #combine into one covariance matrix (and filter to keep)
            sample_keep = [sample[n][ix] for ix in ixs] #keep these samples
            locs = locations[sample_keep] #locations of kept samples
       
        # append ancestor mle or likelihood
        if BLUP:
           
            ahat = _ahatavar(alltimes, locs, tsplits, SIGMA, ahat_only=BLUP) #the mle only
            fs.append(ahat)

        else:

            try:
                ahat, avar = _ahatavar(alltimes, locs, tsplits, SIGMA, ahat_only=BLUP) #mle and covariance
                if np.linalg.matrix_rank(avar) == len(avar):
                    fs.append(lambda x: _lognormpdf(x, ahat, avar, relative=False)) #log likelihood
                else:
                    #print('singular matrix')
                    fs.append(lambda x: 0) #uninformative function (not a function of x so should not affect optima finder)
            except:
                fs.append(lambda x: 0) #this is in case there is problem with inverting matrices in get_ahatavar  
 
        # importance weights
        if important:
            ws.append(_log_birth_density(coal_times[tree], phi, tCutoff) - logpcoals[tree]) #log importance weight
        else:
            ws.append(0) #equal (log) weights for all

    totw = _logsumexp(ws) #log total weight

    # if we want the Best Linear Unbiased Predictor (BLUP), ie the importance sampled MLE location
    if BLUP:
        return sum([mle*np.exp(weight-totw) for mle,weight in zip(fs, ws)]) #average MLE

    # if we want the full importance sampled log likelihood
    else:
        return lambda x: _logsumexp([f(x) + ws[i] for i,f in enumerate(fs)]) - totw #average log likelihood

def _ahatavar(alltimes, locations, tsplits=[], SIGMA=None, ahat_only=False):

    """
    Calculate mean and covariance in ancestor location using shared times (alltimes) and sample locations (locations)
    """
    
    if np.any(alltimes < 0):
        print('error: cant have negative shared times')
        return

    if len(alltimes) < 2:
        print('error: need at least one sample to locate ancestor')
        return

    tmrca = alltimes[-1,-1] #time to mrca
    
    no_descendants = alltimes[0,0] > 0 and all(alltimes[0,1:] == 0)  #ancestor has no direct descendants?
    #if no_descendants:
    #    print('note: no direct descendants in this tree') 

    # special case of only one sample, which means we can't mean center
    if len(alltimes) == 2:
        
        if SIGMA is None:
            print('error: gotta supply SIGMA: cant estimate from a tree with one lineage!')
            return
        
        ahat = locations[0] #best guess for ancestor location is just where lone sample is

        if ahat_only:
            return ahat
        
        # variance accumulated going up from the sample
        ts = [0] + [t for t in tsplits if t < tmrca] + [tmrca] #start and end times of relevant epochs
        cts = [ts[i+1] - ts[i] for i in range(len(ts)-1)] #time spent in each epoch
        avar = sum([ct * SIGMA[-i-1] for i,ct in enumerate(cts)])
        
        # if the ancestor has no direct descendants then we need to add some extra variance
        if no_descendants:
        
            # variance accumalated going down to the ancestor
            ta = tmrca - alltimes[0,0] #time the ancestor was alive
            # going in reversed order this time
            ts = [tmrca] + [t for t in tsplits[::-1] if t < tmrca and t > ta] + [ta] #start and end times of relevant epochs
            cts = [ts[i] - ts[i+1] for i in range(len(ts)-1)] #time spent in each epoch
            epoch_ix = len([t for t in tsplits if t < tmrca]) #which epoch is tmrca in
            avar += sum([ct * SIGMA[::-1][epoch_ix - i] for i,ct in enumerate(cts)]) 

    # if more than one sample we can mean center to get correct variance in all cases
    elif len(alltimes) > 2:

        # mean centering tools
        n = len(locations) #number of samples
        Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[:-1] #mean centering matrix
        x = np.array([[1] + [-1/n] * n]) #vector to mean center ancestors variance

        # mean center the locations
        locationsc = np.matmul(Tmat, locations)
                
        # if only one epoch the relevant covariance matrices are
        if len(tsplits) == 0:
        
            # split into component matrices
            cov = alltimes
            d = 1
            Sigma11 = cov[:1,:1] 
            Sigma12 = cov[:1,1:]
            Sigma21 = cov[1:,:1] 
            Sigma22 = cov[1:,1:]
        
        # if more than one epoch the relevant covariance matrices are
        else:
            
            if SIGMA is None:
                print('error: gotta supply SIGMA if more than 1 epoch cause cant solve for MLE')
                return
            if len(SIGMA) < len(tsplits)+1:
                print('error: need as many SIGMAs as epochs, which is one more than the number of tsplits')
                return

            # calculate time spent in each epoch
            split_times = [0] + [t for t in tsplits if t < tmrca] + [tmrca] #ignore times deeper than tmrca and append end points
            ts = [split_times[i+1] - split_times[i] for i in range(len(split_times)-1)] #amount of time in each epoch
            nzeros = len(SIGMA) - len(ts) #number of epochs missing from tree
            ts = ts + [0] * nzeros #add zeros for missing epochs

            # add up covariance in each epoch (we go in chronolgical order, from most distant to most recent -- make sure SIGMA the same)
            covs = []
            ct = 0
            for i in range(len(ts)):
                rstime = alltimes - ct #subtract off time already moved down the tree, this is the remaining shared time
                rstime = np.maximum(rstime, 0) #make negative values 0
                rstime = np.minimum(rstime, ts[-i-1]) #can only share as much time as length of epoch
                covs.append(np.kron(rstime, SIGMA[i]))
                ct += ts[-i-1] #add time moved down tree

            # split into component matrices
            cov = np.sum(covs, axis=0)
            d = len(SIGMA[0]) #number of spatial dimensions
            SIGMA = [1] #dummy variable for below, since SIGMA already incorporated into cov
            Sigma11 = cov[:d,:d]
            Sigma12 = cov[:d,d:]
            Sigma21 = cov[d:,:d]
            Sigma22 = cov[d:,d:]
            
            # mean centering tools (update to new shape)
            Tmat = np.kron( Tmat, np.identity(d)); #mean centering matrix
            x = np.kron( x, np.identity(d)) #vector to mean center ancestors variance

        # mean center the covariance matrices
        Sigma11c = np.matmul(np.matmul(x, cov), x.transpose()) #mean centered ancestor variance
        Sigma21c = np.matmul(Tmat, Sigma21) - np.matmul( np.matmul(Tmat, Sigma22), np.kron(np.ones(n).reshape(-1,1), np.identity(d)))/n #mean centered covariance between ancestor and samples
        Sigma22c = np.matmul(Tmat, np.matmul(Sigma22, np.transpose(Tmat))) #mean centered covariance of samples

        # invert matrix of (mean centered) shared times
        try:
            Sigma22c_pinv = np.linalg.pinv(Sigma22c) #take generalized inverse
        except:
            print('error: SVD failed')
            return 

        # mle ancestor location
        ahat = np.mean(locations, axis=0) + np.matmul(np.matmul(Sigma21c.transpose(), Sigma22c_pinv), np.kron(locationsc, np.ones(d).reshape(-1,1)))[0]

        if ahat_only:
            return ahat

        # get MLE dispersal rate if nothing supplied
        if SIGMA is None: #if not given, and only one epoch, then estimate from this tree alone
            SIGMA = [np.matmul(np.matmul(np.transpose(locationsc), Sigma22c_pinv), locationsc) / (n-1)]

        # covariance in ancestor location
        avar = (Sigma11c - np.matmul( np.matmul(Sigma21c.transpose(), Sigma22c_pinv), Sigma21c)) * SIGMA[0] 
           
    return ahat, avar

def _alltimes(stimes, atimes, keep=None):

    """
    combine shared times between sample lineages (stimes) and shared times of samples with ancestral lineage (atimes)
    """

    # filter out samples we don't want to use for locating
    if keep is not None:
        stimes = stimes[keep][:,keep]
        atimes = atimes[np.append(keep,-1)] #be sure to keep the last entry (shared time of ancestral lineage with itself)
    
    # join stimes and atimes into one matrix
    n = len(atimes)
    alltimes = np.zeros((n,n))
    alltimes[0,0] = atimes[-1] #element in first row and column
    alltimes[0,1:] = atimes[:-1] #remainder of first row
    alltimes[1:,0] = atimes[:-1] #remainder of first column
    alltimes[1:,1:] = stimes #remainder of matrix
    
    # trim off excess on the tree
    alltimes = alltimes - np.min(alltimes)

    return alltimes

def _atimes(stimes, time, focal_index):

    """
    Calculate shared times of sample lineages with ancestor
    
    Parameters
    ----------
    stimes : ndarray 
        matrix of shared times between sample lineages
    time : non-negative real number
        time at which the ancestor existed
    focal_index : integer in [0, len(`stimes`))
        index of sample whose ancestor we're interested in
        
    Returns
    -------
    atimes : ndarray
        Shared times of each sample lineage (in same order as `stimes`) and ancestor with ancestor
        
    Notes
    -----
    Can give negative values if `time` > TMRCA, which should be dealt with downstream.
    
    Examples
    --------
    >>> stimes = np.array([[10,0],[0,10]])
    >>> _atimes(stimes, 7.5, 0)
    array([2.5, 0., 2.5])
    
    >>> stimes = np.array([[10,0],[0,10]])
    >>> _atimes(stimes, 12.5, 0)
    array([-2.5, -2.5, -2.5])
    """
    
    tmrca = stimes[focal_index, focal_index] #time to most recent common ancestor of sample lineages
    taa = tmrca - time #shared time of ancestor with itself 

    atimes = [] 
    for t in stimes[focal_index]:
        atimes.append(min(t, taa)) # shared times between ancestor and each sample lineage

    atimes.append(taa) #add shared time with itself
        
    return np.array(atimes)


