import numpy as np
from scipy.optimize import minimize
from sparg.utils import _get_focal_index, _lognormpdf, _logsumexp, _sigma_phi
from sparg.importance_sampling import _log_birth_density

def locate(nodes, times, treefiles, mlefiles, locations, tCutoff=1e4, importance=True, x0=[0,0], bnds=((None,None),(None,None)), method='L-BFGS-B', weight=True, keep=None, tsplits=[], BLUP=False):

    """
    locate nodes=nodes at times=times using trees (treefiles), mle dispersal (mlefiles), and sample locations (locations)
    """

    # loop over loci
    mle_locations = []
    weights = []
    for treefile,mlefile in zip(treefiles,mlefiles):

        # load processed trees at this locus
        processed_trees = np.load(treefile, allow_pickle=True)
        coal_times = processed_trees['coal_times']
        pcoals = processed_trees['pcoals']
        shared_times = processed_trees['shared_times']
        samples = processed_trees['samples']

        # if we only have one sampled tree at a locus then we can find the MLE analytically
        if len(pcoals[0]) == 1:
            BLUP = True
            print('only one tree at this locus so calculating MLE locations analytically')

        # load mle dispersal and branching rate at this locus
        x = np.load(mlefile, allow_pickle=True).item().x
        [SIGMA, phi] =  _sigma_phi(x, tsplits, importance) 
        
        # loop over times
        mle_locations_t = [] #locations for all nodes at all loci at this time
        weights_t = [] #weights of locations (measure of uncertainty)
        for time in times:

            # loop over nodes
            mle_locations_i = [] #locations for all nodes at particular time at this locus
            weights_i = []
            for focal_node in nodes:
                
                # get location of node i at time t at locus
                f = _ancestor_location_meanloglikelihood(node, time, coal_times, pcoals, shared_times, samples, locations, keep=keep, phi=phi, SIGMA=SIGMA, locus=0, tCutoff=tCutoff, importance=importance, tsplits=tsplits, BLUP=BLUP) #mean log likelihood or mean MLE (if BLUP)

                if f is None:
                    mle_locations_i.append(np.ones(len(x0)) * np.nan)
                    #print('locating ancestor of %d at time %d failed at a locus' %(focal_node, time))

                else:

                    # if we want the best linear unbiased predictor (BLUP), ie the importance sampled MLE 
                    if BLUP:
                   
                        mle_locations_i.append(f) #just add mean MLE
    
                        if weight:
                            print('weights not implemented for BLUPs, if you want weights set BLUP=False')
    
		    # find the true MLE by finding the max of the average log likelihood across importance samples
                    else:
    
                        g = lambda x: -f(x) #flip bc we look for min
                        gmin = minimize(g, x0 = x0, bounds = bnds, method=method) #find MLE
                        # if numerical search worked
                        if gmin.success:
                            mle_locations_i.append(gmin.x) #append mle location for this node
                        else:
                            mle_locations_i.append(np.ones(len(x0)) * np.nan) #some nans to fill up the matrix but not influence results
                            #print('locating ancestor of %d at time %d failed at a locus' %(focal_node, time))
    
                        # we may want to weight locations over loci depending on how certain the estimate is
                        if weight:
                            if method=='L-BFGS-B':
                                if gmin.success:
                                    if gmin.x[0] > bnds[0][0] and gmin.x[0] < bnds[0][1] and gmin.x[1] > bnds[1][0] and gmin.x[1] < bnds[1][1]: # if did not hit bounds
                                        cov = gmin['hess_inv'].todense() #inverse hessian (negative fisher info) is estimate of covariance matrix
                                        weights_i.append(cov) #append covariance matrix as weights
                                    else:
                                        weights_i.append(np.zeros((len(x0)-1, len(x0)-1))) #dont use this estimate if hit bounds                
                                else:
                                    weights_i.append(np.zeros((len(x0)-1, len(x0)-1))) #dont use this estimate if numerical minimizer failed                
                            else:
                                weights_i.append(1) #weights all equally if don't have the hessian
                                print('choose method=L-BFGS-B if you want weights')

            mle_locations_t.append(mle_locations_i) #append locations across all nodes for particular time at particular locus
            weights_t.append(weights_i) #append weights across all nodes for particular time at particular locus
        mle_locations.append(mle_locations_t) #append locations acrocss all nodes and all times for particular locus
        weights.append(weights_t) #append weights acrocss all nodes and all times for particular locus

    #print(mle_locations)

    if weight:
        return np.array(mle_locations), np.array(weights)
    else:
        return np.array(mle_locations) 

def _ancestor_location_meanloglikelihood(node, time, coal_times, pcoals, shared_times, samples, locations, keep=None, phi=1, SIGMA=None, locus=0, tCutoff=1e4, importance=True, tsplits=[], BLUP=False):
    """
    log likelihood of ancestor location (ancestor of node=node at time=time) averaged over sampled trees, calculated using the coalescent times (coal_times), probability of the coalescent times (pcoals), shared evolutionary times (shared_times), and locations (locations). samples connects the tips of the trees with locations.
    """

    fs = []
    ws = []
    #loop over trees (importance samples) at a locus
    for tree,sample in enumerate(samples[locus]): 
        
        # get shared times and sample locations
        n,m = _get_focal_index(node, sample) #subtree and sample index of focal_node
        sts = shared_times[locus][tree][n] #shared times between all sample lineages in subtree
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
        if importance:
            ws.append(_log_birth_density(coal_times[locus][tree], phi, tCutoff) - pcoals[locus][tree]) #log importance weight
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


