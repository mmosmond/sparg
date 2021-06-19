import numpy as np
from scipy.optimize import minimize
from sparg.importance import _log_birth_density
from sparg.utils import _lognormpdf, _logsumexp, _get_focal_index

def locate(treefiles, mlefiles, nodes, times, locations, keep=None, tCutoff=None, tsplits=[], importance=True, weight=True, BLUP=False, x0=[0,0], bnds=((None,None),(None,None)), method='L-BFGS-B'):
    """
    locate ancestors

    treefiles: files containing information about local trees (coalescent times, probability of the coalescent times under a Yule prcess, shared evolutionary times, and subtree structure)
    mlefiles: files containing dispersal rate and Yule branching rate
    nodes: the samples we want to find the ancestors of
    times: the times to locate the ancestors
    locations: locations of all sample nodes
    keep: list of samples we use in inference (if None use all sample nodes)
    tCutoff: time to cut tree off to ignore deeper relationships (if None go back to MRCA of all samples)
    tsplits: split times between epochs
    importance: whether or not to use importance sampling over branch length estimates
    weight: whether to calculate and output weights (for inverse variance weighting of locations across loci)
    BLUP: whether to analytically calculate the average of MLE location across branch length estimates (otherwise find the max of the average log likelihood) 
    x0: starting point for maximum finder
    bnds: bounds for parameters in maximum finder
    method: method of scipy's minimizer
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
        SIGMA = []
        for i in range(len(tsplits) + 1):
            SIGMA.append(np.array([[x[3*i]**2, x[3*i]*x[3*i+1]*x[3*i+2]], [x[3*i]*x[3*i+1]*x[3*i+2], x[3*i+1]**2]])) #append matrices for each epoch (chronological order)
        if importance:
            phi = x[-1]/scale_phi
        else:
            phi = 1 #irrelevant and arbitary value if not importance sampling
        
        # loop over times
        mle_locations_t = [] #locations for all nodes at all loci at this time
        weights_t = [] #weights of locations (measure of uncertainty)
        for time in times:

            # loop over nodes
            mle_locations_i = [] #locations of ancestors of all nodes at particular time at this locus
            weights_i = []
            for focal_node in nodes:
                
                # get location of node i at time t at locus
                f = _ancestor_location_meanloglikelihood(coal_times, pcoals, shared_times, samples, locations, focal_node, keep=keep, phi=phi, SIGMA=SIGMA, time=time, tCutoff=tCutoff, importance=importance, tsplits=tsplits, BLUP=BLUP) #mean log likelihood or mean MLE (if BLUP)

                # if there is an error then f will be None 
                if f is None:
                    mle_locations_i.append(np.ones(len(x0)) * np.nan)
                    #print('locating ancestor of %d at time %d failed at a locus' %(focal_node, time))
                
                # otherwise
                else:

                    # if we want the best linear unbiased predictor (BLUP), ie the average MLE 
                    if BLUP:
                   
                        mle_locations_i.append(f) #just add mean MLE
    
                        if weight:
                            print('weights not implemented for BLUPs, if you want weights set BLUP=False')
    
		    # otherwise find the max of the average log likelihood
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
                            if method == 'L-BFGS-B':
                                if gmin.success:
                                    if gmin.x[0] > bnds[0][0] and gmin.x[0] < bnds[0][1] and gmin.x[1] > bnds[1][0] and gmin.x[1] < bnds[1][1]: # if did not hit bounds 
                                        cov = gmin['hess_inv'].todense() #inverse hessian (negative fisher info) is estimate of covariance matrix
                                        weights_i.append(cov) #append covariance matrix as weights
                                    else:
                                        weights_i.append(np.zeros((len(x0)-1, len(x0)-1))) #dont use this estimate if hit bounds                
                                else:
                                    weights_i.append(np.zeros((len(x0)-1, len(x0)-1))) #dont use this estimate if numerical minimizer failed                
                            else:
                                weights_i.append(1) #weight all estimates equally if don't have the hessian (which we only get from some minimize methods)
                                print('choose method=L-BFGS-B if you want weights')

            mle_locations_t.append(mle_locations_i) #append locations across all nodes for particular time at particular locus
            weights_t.append(weights_i) #append weights across all nodes for particular time at particular locus
        mle_locations.append(mle_locations_t) #append locations acrocss all nodes and all times for particular locus
        weights.append(weights_t) #append weights acrocss all nodes and all times for particular locus

    if weight:
        return np.array(mle_locations), np.array(weights)
    else:
        return np.array(mle_locations) 

def _ancestor_location_meanloglikelihood(coal_times, pcoals, shared_times, samples, locations, focal_node, time, locus=0, keep=None, phi=1, SIGMA=None, tCutoff=None, tsplits=[], importance=True, BLUP=False):
    """
    mean log likelihood of ancestor location

    coal_times: coalescent times
    pcoals: probability of coalescent times
    shared_times: shared evolutionary times in subtrees
    samples: samples in each subtree
    locations: locations of all samples
    focal_node: node we are finding the ancestor of
    time: generations ago to find ancestor
    locus: which locus we are finding ancestor at (defaults to 0 because often supply only 1)
    keep: which sample locations to use (if None use all)
    phi: branching rate in Yule process (for importance sampling)
    SIGMA: dispersal rate
    tCutoff: time to go back to (ignore deeper times; None means go back to MRCA)
    tsplits: split times between epochs
    importance: whether to importance sample or not
    BLUP: whether to calculate mean MLE, else mean log likelihood
    """

    fs = []
    ws = []
    #loop over trees (importance samples) at a locus
    for tree,sample in enumerate(samples[locus]): 
        
        n,m = _get_focal_index(focal_node, sample) #subtree and sample index of focal_node
        sts = shared_times[locus][tree][n] #shared times between all samples in subtree
        sts_focal_focal = sts[m,m] - time #shared time of ancestor with itself
        
        # calculate shared times with ancestor
        Atimes = [] 

        # if we're not excluding any samples
        if keep is None:
            for st in sts[m]:
                Atimes.append(min(st,sts_focal_focal)) # shared times between ancestor and each sample
            locs = locations[sample[n]] #locations of samples

        # if we are excluding some samples
        else:
            ixs = np.where(np.in1d(sample[n], keep))[0] #what indices are we keeping location info for
            for st in sts[m][ixs]:
                Atimes.append(min(st,sts_focal_focal)) # shared times between focal and kept samples
            sts = sts[ixs][:,ixs] #keep these shared_times among samples
            sample_keep = [sample[n][ix] for ix in ixs] #keep these samples
            locs = locations[sample_keep] #locations of kept samples
        
        Atimes.append(sts_focal_focal) #add shared time with itself
 
        # log likelihoods (or MLEs) across importance samples 
        fs.append(_ancestor_location_loglikelihood(sts, locs, SIGMA=SIGMA, Atimes=Atimes, tsplits=tsplits, MLE_only=BLUP)) 

        # importance weights
        if importance:
            ws.append(_log_birth_density(coal_times[locus][tree], phi, tCutoff) - pcoals[locus][tree]) #log importance weight
        else:
            ws.append(0) #equal (log) weights for all

    # make sure at least one importance sample worked
    if len(fs) == 0:
        #print('WE HAVE NO IDEA WHERE THIS ANCESTOR IS')
        return None

    else:
        
        totw = _logsumexp(ws) #log total weight

        # if we want the Best Linear Unbiased Predictor (BLUP), ie the importance sampled MLE location
        if BLUP:
            return sum([mle*np.exp(weight-totw) for mle,weight in zip(fs, ws)]) #average MLE

        # if we want the full importance sampled log likelihood
        else:
            return lambda x: _logsumexp([f(x) + ws[i] for i,f in enumerate(fs)]) - totw #average log likelihood

def _ancestor_location_loglikelihood(shared_time, locs, Atimes=None, focal_node=None, t=None, SIGMA=None, tsplits=[], MLE_only=False):
    """
    log likelihood location of ancestor of focal_node t generations ago

    shared_time: shared times between samples in subtree
    locs: locations of samples in subtree
    Atimes: shared times between ancestor and sample lineages (if None must supply focal_node and t)
    focal_node: node to find ancestor of 
    t: time to find ancestor at
    SIGMA: dispersal rate
    tsplits: split times between epochs
    MLE_only: whether to just give the MLE location, else the log likelihood
    """

    tplus = 0 #time beyond the mrca 
    tmrca = shared_time[0][0] #time to mrca
    
    # special case of only one sample (Brownian motion is just linear increase in variance around this sample location)
    if len(shared_time) == 1:
        #print('locating ancestor with only 1 sample location')
        Ahat = locs[0] #best guess for ancestor location is just where lone sample is
        if MLE_only:
            return Ahat
        else:
            if t == None:
                t = -Atimes[0] #time we want the ancestor location (-(tmrca - t) = t)
                shared_time[0,0] = t #tmrca=0 but want to go back to t
                Atimes = np.array([0,0]) #ancestor back at t so no shared time with itself of sample lineages
                #print('locating ancestor at time %d, if want another time supply t' %t)
            ts = [0] + [i for i in tsplits if i<t] + [t] #start and end times of relevant epochs
            cts = [ts[i+1] - ts[i] for i in range(len(ts)-1)] #time spent in each epoch
            Avar = sum([ct * SIGMA[i] for i,ct in enumerate(cts)]) #variance increases like t*SIGMA in each epoch, and sum together               
            if np.linalg.matrix_rank(Avar) == len(Avar):
                 return lambda x: _lognormpdf(x, Ahat, Avar, relative=False) #normal distribution with mean and variance as above
            else:
                 print('singular matrix with single sample!')
                 return

    # shared times between ancestor and all sample lineages
    if Atimes is None: #if not passing shared times between ancestor and sample lineages directly, calculate
        tAA = tmrca - t #time to mrca from ancestor (ie shared time with itself)
        Atimes = [] #empty vector for shared times with other nodes
        for st in shared_time[focal_node]: #for each time the focal node shares with all other nodes
            Atimes.append(np.min([tAA, st])) #the shared time with the ancestor of the focal node is the smaller of tAA and the time shared with the focal node

    else: #if we do pass the times directly, separate the shared times
        tAA = Atimes[-1] #shared time of ancestor with itself
        Atimes = Atimes[:-1] #shared times with samples    

    # dealing with ancestors beyond the mrca
    if tAA < 0:  
        #print('locating ancestor beyond the mrca')
        tplus = -tAA #extra time beyond the mrca we need to deal with
        Atimes = [0 for _ in Atimes] #make shared times with all samples zero
        tAA = 0 #shared time with self zero

    n,d = locs.shape #number of samples and spatial dimensions
    
    # combine shared times into one matrix (will need it below)     
    stime = np.zeros((n+1,n+1))
    stime[0,0] = tAA #first diagonal element
    stime[0,1:] = Atimes #remainder of first row
    stime[1:,0] = Atimes #remainder of first column
    stime[1:,1:] = shared_time #remainder of matrix

    # split (back) into component matrices
    Sigma11 = stime[:1,:1] #[[tAA]]
    Sigma12 = stime[:1,1:] #Atimes
    Sigma21 = stime[1:,:1] #Sigma12.reshape(-1,1)
    Sigma22 = stime[1:,1:] #shared_time
    
    # mean centering tools
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[:-1] #mean centering matrix
    x = np.array([[1] + [-1/n] * n]) #vector to mean center ancestors variance

    # mean center the locations
    locationsc = np.matmul(Tmat, locs)
    
    # if just one epoch
    if tsplits == []:
        
        # mean center the covariance matrices
        Sigma11c = np.matmul(np.matmul(x, stime) , x.transpose()) #mean centered ancestor variance
        Sigma21c = np.matmul(Tmat, Sigma21) - np.matmul( np.matmul(Tmat, Sigma22), np.ones(n).reshape(-1,1))/n #mean centered covariance between ancestor and samples
        Sigma22c = np.matmul(Tmat, np.matmul(Sigma22, np.transpose(Tmat))) #mean centered covariance of samples

        # mle ancestor location
        try:
            Sigma22c_pinv = np.linalg.pinv(Sigma22c) #take generalized inverse (need it again for variance)
        except:
            print('SVD failed')
            return lambda x: 0 #if inverse fails return non-informative function
        Ahat = np.mean(locs, axis=0) + np.matmul( np.matmul(Sigma21c.transpose(), Sigma22c_pinv), locationsc)
        Ahat = Ahat[0]

        # if just want MLE
        if MLE_only:
            return Ahat

        # if we want full (log) likelihood        
        else:

            # get MLE dispersal rate if nothing supplied
            if SIGMA is None: #if not given, and only one epoch, then estimate with mle
                Tinv = np.linalg.pinv(np.array(Sigma22c)) #inverse of centered shared time matrix
                SIGMA = [np.matmul(np.matmul(np.transpose(locationsc), Tinv), locationsc) / (n-1)] #mle
                
            # covariance matrix in ancestor location
            Avar = (Sigma11c - np.matmul( np.matmul(Sigma21c.transpose(), Sigma22c_pinv), Sigma21c)) * SIGMA[0] + tplus * SIGMA[0]
        
    # if more than one epoch
    else:
        
        if SIGMA is None:
            print('gotta supply SIGMA if more than 1 epoch cause cant solve for MLE')
            return
        if len(SIGMA) != len(tsplits)+1 or SIGMA[0].shape != (d,d):
            print('need as many SIGMAs as epochs, which is one more than the number of tsplits')
            return
       
        # length of each epoch
        split_times = [0] + [t for t in tsplits if t < tmrca] + [tmrca] #ignore times deeper than tmrca and append end points
        Ts = [split_times[i+1] - split_times[i] for i in range(len(split_times)-1)] #amount of time in each epoch
        nzeros = len(SIGMA) - len(Ts) #number of epochs missing from tree
        Ts = Ts + [0] * nzeros #add zeros for missing epochs
    
        # covariance in each epoch (we go in chronolgical order, from most distant to most recent -- make sure SIGMA the same)
        covs = []
        cumulative_time = 0
        for i in range(len(Ts)):
            rstime = stime - cumulative_time #subtract off time already moved down the tree, this is the remaining shared time
            rstime = np.maximum(rstime, 0) #make negative values 0
            rstime = np.minimum(rstime, Ts[-i-1]) #can only share as much time as length of epoch
            covs.append(np.kron(rstime, SIGMA[i]))
            cumulative_time += Ts[-i-1] #add time moved down tree

        extra_cov = 0
        if tplus > 0:
            #covariance beyond the tmrca 
            extra_time = [cumulative_time] + [t for t in tsplits if t>cumulative_time] + [tmrca + tplus] #split times beyond trmca
            extra_time = [extra_time[i+1] - extra_time[i] for i in range(len(extra_time)-1)] #cumulative times in each epoch
            extra_cov = sum([extra_time[-i-1] * SIGMA[-i-1] for i in range(len(extra_time))]) #covariance summed over epochs

        # split into component matrices
        cov = np.sum(covs, axis=0)
        Sigma11 = cov[:d,:d] 
        Sigma12 = cov[:d,d:] 
        Sigma21 = cov[d:,:d] 
        Sigma22 = cov[d:,d:]
        
        # mean centering tools (update to new shape)
        Tmat = np.kron( Tmat, np.identity(d)); #mean centering matrix
        x = np.kron( x, np.identity(d)) #vector to mean center ancestors variance
        
        # mean center the covariance matrices
        Sigma11c = np.matmul(np.matmul(x, cov) , x.transpose()) #mean centered ancestor variance
        Sigma21c = np.matmul(Tmat, Sigma21) - np.matmul( np.matmul(Tmat, Sigma22), np.kron(np.ones(n).reshape(-1,1), np.identity(d)))/n #mean centered covariance between ancestor and samples
        Sigma22c = np.matmul(Tmat, np.matmul(Sigma22, np.transpose(Tmat))) #mean centered covariance of samples
        
        # mle ancestor location
        try:
            Sigma22c_pinv = np.linalg.pinv(Sigma22c) #take generalized inverse (need it again for variance)
        except:
            print('SVD failed')
            return lambda x: 0 #if inverse fails return non-informative function
        Ahat = np.mean(locs, axis=0) + np.matmul( np.matmul(Sigma21c.transpose(), Sigma22c_pinv), np.kron(locationsc, np.ones(d).reshape(-1,1)))[0] #this seems pretty silly but works

        # if we only want MLE
        if MLE_only:
            return Ahat
        
        # if we want full (log) likelihood
        else:

            # covariance in ancestor location
            Avar = (Sigma11c - np.matmul( np.matmul(Sigma21c.transpose(), Sigma22c_pinv), Sigma21c)) + extra_cov

    #protection in case entries of Avar so small they become zero
    if np.linalg.matrix_rank(Avar) == len(Avar):
        return lambda x: _lognormpdf(x, Ahat, Avar, relative=False)
    else:
        #print('singular matrix')
        return lambda x: 0 #uninformative function (not a function of x so should not affect optima finder)

