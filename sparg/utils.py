import numpy as np
import scipy.sparse as sp

def filter_samples(samples, shared_times, keep_ix):
    
    """
    filter to keep only specified samples and shared_times
    """

    samples_keep = []
    shared_times_keep = []
    for j,locus in enumerate(samples): #loop over loci
      samples_keep_j = []
      shared_times_keep_j = []
      for k, tree in enumerate(locus): #loop over trees
        samples_keep_k = []
        shared_times_keep_k = []
        for l,subtree in enumerate(tree): #loop over subtrees
          ixs = np.where(np.in1d(subtree, keep_ix))[0] #what indices do we want to keep
          samples_keep_k.append([subtree[ix] for ix in ixs]) #keep these samples
          shared_times_keep_k.append((shared_times[j][k][l])[ixs][:,ixs]) #keep these shared_times
        samples_keep_j.append(samples_keep_k)
        shared_times_keep_j.append(shared_times_keep_k)
      samples_keep.append(samples_keep_j)
      shared_times_keep.append(shared_times_keep_j)

    return samples_keep, shared_times_keep

def _get_focal_index(focal_node, listoflists):

    """
    get the subtree and index within that subtree for focal_node (listoflists here is list of samples for each subtree)
    """

    for i,j in enumerate(listoflists):
        if focal_node in j:
            n = i
            for k,l in enumerate(j):
                if focal_node == l:
                    m = k
    return n,m

def _lognormpdf(x, mu, S, relative=True):

    """
    Calculate log probability density of x, when x ~ N(mu,S)
    """

    # log of coefficient in front of exponential (times -2)
    nx = len(S)
    if relative == False:
        norm_coeff = nx * math.log(2 * math.pi) + np.linalg.slogdet(S)[1] 
    else:
        norm_coeff = np.linalg.slogdet(S)[1] #just care about relative likelihood so drop the constant

    # term in exponential (times -2)
    err = x - mu #difference between mean and data
    if sp.issparse(S):
        numerator = spln.spsolve(S, err).T.dot(err) #use faster sparse methods if possible
    else:
        numerator = np.linalg.solve(S, err).T.dot(err) #just a fancy way of calculating err.T * S^-1  * err

    return -0.5 * (norm_coeff + numerator) #add the two terms together

def _logsumexp(a):

    """
    take the log of a sum of exponentials without losing information
    """

    a_max = np.max(a) #max element in list a
    tmp = np.exp(a - a_max) #now subtract off the max from each a before taking exponential (ie divide sum of exponentials by exp(a_max))
    s = np.sum(tmp) #and sum those up
    out = np.log(s) #and take log
    out += a_max  #and then add max element back on (ie multiply sum by exp(a_max), ie add log(exp(a_max)) to logged sum)

    return out

def _sigma_phi(x, tsplits=[], important=True):
    
    """
    convert list of parameters being estimated into covariance matrix and birth rate
    """

    d = int((len(x) - 1)/(len(tsplits)+1)) #number of dispersal parameters we're estimating
    
    Sigma = [] #make as list
    for i in range(len(tsplits) + 1):
        if d == 3: #three dispesal parameters means we are working in 2D (two SDs and a corelation) so we need to make 2x2 covariance matrix
            Sigma.append(np.array([[x[d*i]**2, x[d*i]*x[d*i+1]*x[d*i+2]], [x[d*i]*x[d*i+1]*x[d*i+2], x[d*i+1]**2]])) #append covariance matrices for each epoch
        if d == 1:
            Sigma.append(np.array([[x[d*i]**2]])) #covariance matrix is just variance 
       # to do: write this more generally for any dimension

    if important:
        phi = x[d*(len(tsplits) + 1)]
    else:
        phi = 1 #if not importance sampling this value is irrelevant, so we just supply an arbitrary value in case nothing supplied by user

    return [Sigma,phi]


