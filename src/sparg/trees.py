from tqdm.auto import tqdm
from sparg.importance_sample import _log_coal_density
import dendropy
import numpy as np
import os
import tskit
from io import StringIO

def choose_loci(ts, which, mode='tree'):
    """Get tree indices and genomic intervals at loci of interest from tree sequence.

    Parameters:
        ts (tskit tree sequence): Tree sequence.
        which (array-like): List of tree indices (if mode='tree') or sites (if mode='site') we are interested in.
        mode (string): Whether to interpret `which` as tree indices or sites. 

    Returns:
        which_trees (numpy array): List of tree indices.
        intervals (numpy array): Genomic interval of each tree in 'which_trees'.
    """

    breakpoints = ts.breakpoints(as_array=True) #breakpoints between trees
    
    # loci indices
    if mode == 'site':
        which_trees = []
        for i in which:
            which_trees.append(np.argmax(breakpoints[1:] >= i)) #much faster to just use breakpoints
        which_trees = np.unique(which_trees) #in case >1 bp fall in same tree
    elif mode == 'tree':
        which_trees = np.unique([i for i in which if i < ts.num_trees])
 
    # genomic intervals of each chosen locus
    intervals=[]
    for i in which_trees:
        intervals.append(breakpoints[i:i+2]) #again, much faster to just use breakpoints

    return which_trees, np.array(intervals)

def get_dendropy_trees(treefile, which_trees=None, from_ts=False):
    """Get dendropy trees from file of newick trees or tree sequence

    Parameters:
        treefile: Name of file of newick trees (if from_ts=False) or tskit tree sequence (if from_ts=True).
        which_trees (array-like): List of tree indices to convert.
        from_ts (boolean): Whether we are getting the trees from a tree sequence, rather than from newick files. 

    Returns:
        trees (list): List (over loci/tree indices) of lists (over samples of trees per locus) of dendropy trees.  
    """

    if which_trees is not None:
        progress_bar = tqdm(total=len(which_trees))

    trees = []
    if not from_ts: #if taking from newick files
        with open(treefile, mode='r') as file:
            for i,line in enumerate(file): 
                if which_trees is None: #if using all samples 
                    if i>0: #just skip header
                        tree = StringIO(line.split()[4]) #convert Newick tree to string
                        trees.append(dendropy.Tree.get(file=tree, schema='newick')) #append to list of dendropy trees
                elif i-1 in which_trees: #else if taking a selection of trees, only take those in selection (note we skip the header here)
                    tree = StringIO(line.split()[4]) #convert Newick tree to string
                    trees.append(dendropy.Tree.get(file=tree, schema='newick'))
                    progress_bar.update()

    else: #if taking from tree sequence
        n = treefile.num_samples
        node_labels = { i: i for i in range(n) }
        if which_trees is None: #if using all samples 
            for tree in treefile.trees(): #for each tree in tree sequence
                newick = tree.newick(node_labels=node_labels) #get newick representation
                trees.append(dendropy.Tree.get(data=newick, schema='newick')) #append to list of dendropy trees
        else:
            for i,tree in enumerate(treefile.trees()):
                if i > max(which_trees):
                    break
                if i in which_trees:
                    newick = tree.newick(node_labels=node_labels) #get newick representation
                    trees.append(dendropy.Tree.get(data=newick, schema='newick')) #append to list of dendropy trees
                    progress_bar.update()
        trees = [[tree] for tree in trees] #reshape so first dimn is loci and second is samples of trees at a locus (here just one sample per locus)
            
    if which_trees is not None:
        progress_bar.close()

    return trees

def process(trees, Nes=None, epochs=None, tCutoff=None, important=True):
    """Process dendropy trees for downstream calculations.

    Parameters:
        trees: List (over loci) of list (over tree samples) of dendropy trees.
        Nes (array-like): Effective population sizes in each epoch (with last entry repeated).
        epochs (array-like): Start and end times of epochs.
        tCutoff (float): When to chop tree off into subtrees, effectively ignoring older times.
        important (boolean): Whether we are going to want to importance sample. If not then we do not need to compute everything.

    Returns:
        shared_times (list): Shared times between sample lineages. An l x m x s_i x n_j x n_j array, where l is the number of loci, m is the number of sampled trees per locus, s_i is the number of subtrees of tree i, and n_j is the number of samples in subtree j
        samples (list): The placement of sample nodes in shared_times, to connect the shared times with the sample locations. Same shape as shared_times.   
        coal_times (list): Coalescence times. An l x m x (n - 1) array, where l is the number of loci, m is the number of sampled trees per locus, and n is the number of samples. Used to compute the probability of the trees under a Yule model when importance sampling.
        logpcoals (list): Log probability of coalescence times under the standard neutral coalescent. An l x m x 1 array, where l is the number of loci and m is the number of sampled trees per locus. Compared to the probability of the trees under a Yule model when importance sampling. 
    """

    shared_times = []
    samples = []
    coal_times = []
    logpcoals = []

    progress_bar = tqdm(total=len(trees))
    for locus in trees:

        shared_times_i = []
        samples_i = []
        coal_times_i = []
        logpcoals_i = []
        for tree in locus:

            sts = _shared_times(tree, tCutoff) #shared times and samples for each subtree
            shared_times_i.append(sts[0])
            samples_i.append(sts[1])

            if important:
                cts = _coal_times(tree) #coalescence times
                coal_times_i.append(cts)
                logpcoals_i.append(_log_coal_density(cts, Nes, epochs, tCutoff)) #probability of coalescence times in neutral coalescent

        shared_times.append(shared_times_i)
        samples.append(samples_i)
        coal_times.append(coal_times_i)
        logpcoals.append(logpcoals_i)

        progress_bar.update()
    progress_bar.close()

    if important:
        return shared_times, samples, coal_times, logpcoals 

    else:
        return shared_times, samples

def _shared_times(tree, tCutoff=None):
    
    """
    get shared evolutionary times between sampled lineages from dendropy tree
    """

    pdm = tree.phylogenetic_distance_matrix() #denropy method to get time between samples
    taxa = np.array([i.taxon for i in tree.leaf_nodes()]) #taxa representing each sample
    n = len(taxa) #number of samples
    tmrcas = np.zeros((n,n)) #matrix to store time to mrcas
    for i in range(n):
        for j in range(i):
            tmrcas[i,j] = pdm(taxa[i],taxa[j])/2 #time to mrca is 1/2 of time between samples
            tmrcas[j,i] = tmrcas[i,j] #symmetric

    tmrca = np.max(tmrcas) #time to most recent common ancestor of all samples

    # if we're not chopping the tree into subtrees
    if tCutoff is None or tCutoff > tmrca:
        times = [tmrca - tmrcas] #shared times
        samples = [range(n)] #samples

    # if we do want to chop the tree
    else:
        shared_time = tCutoff - tmrcas #shared time since tCutoff
    
        # get shared times and samples in each subtrees
        i = 0 #start with first sample
        withi = shared_time[i] >= 0 #true if share time with i
        timesi = shared_time[withi][:, withi] #shared times with i
        timesi = timesi - np.min(timesi) #trim off lineage from MRCA to tCutoff
        times = [timesi] #start list with shared times in subtree with i
        samples = [np.where(withi)] #samples in this subtree
        taken = withi #samples already in a subtree
        while sum(taken) < n: #while some samples not yet in a subtree
            i = np.argmax(taken == False) #choose next sample not yet in a subtree
            withi = shared_time[i] >= 0 #true if share time with i
            timesi = shared_time[withi][:, withi] #shared times of subtree with i
            timesi = timesi - np.min(timesi) #trim
            times.append(timesi) #append        
            samples.append(np.where(withi)) #samples in this subtree
            taken = np.array([i[0] or i[1] for i in zip(taken, withi)]) #samples already in a subtree

    return times, [[int(i.label) for i in taxa[j]] for j in samples]

def _coal_times(tree):
    
    """
    get coalescence times in ascending order from dendropy tree
    """

    return np.array(tree.internal_node_ages(ultrametricity_precision=False))

