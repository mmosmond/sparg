import numpy as np
import math

def _lognormpdf(x, mu, S, relative=True):
    """ 
    Calculate log probability density of x, when x ~ N(mu,sigma) 
   
    relative: whether to drop normalizing constant, else get true log probability  
    """

    # log of coefficient in front of exponential (times -2)
    nx = len(S)
    if relative:
        norm_coeff = np.linalg.slogdet(S)[1] #if just care about relative likelihood then drop the constant
    else:
        norm_coeff = nx * math.log(2 * math.pi) + np.linalg.slogdet(S)[1] 

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

def _get_focal_index(focal_node, listoflists):
    """
    get the subtree and index within that subtree for given focal node
    """

    # loop over list of sample in each subtree
    for i,j in enumerate(listoflists):
        # if focal_node in the subtree
        if focal_node in j:
            n = i #index of subtree
            # loop over samples in subtree
            for k,l in enumerate(j):
                # once find focal
                if focal_node == l:
                    m = k #index of focal in subtree

    return n,m

