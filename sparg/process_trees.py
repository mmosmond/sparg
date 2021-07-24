from tqdm.auto import tqdm
from sparg.importance_sampling import _log_coal_density
import dendropy
import numpy as np
import os
import tskit

def choose_loci(ts, which=[0], mode='site'):

    """
    get tree indices and genomic intervals at loci of interest from treesequence
    """

    breakpoints = ts.breakpoints(as_array=True) #breakpoints between trees
    
    # loci indices
    print('choosing loci...')
    
    if mode == 'site':
        #which_loci = [ts.at(i).index if i<ts.sequence_length else ts.num_trees-1 for i in which] #index/locus for each chosen site (note if site number too high we use the last tree)
        which_loci = []
        for i in which:
            which_loci.append(np.argmax(breakpoints[1:] >= i)) #much faster to just use breakpoints
        which_loci = np.unique(which_loci) #in case >1 bp fall in same tree
    elif mode == 'tree':
        which_loci = np.unique([i for i in which if i<ts.num_trees])
       
    print('number of loci: ', len(which_loci))
 
    # genomic intervals of each chosen locus
    print('getting intervals...')
    intervals=[]
    #for i,tree in enumerate(ts.trees()):
    #    if i > which_loci[-1]:
    #        break
    #    elif i in which_loci:
    #        intervals.append([tree.interval[0],tree.interval[1]])
    for i in which_loci:
        intervals.append(breakpoints[i:i+2]) #again, much faster to just use breakpoints
    intervals - np.array(intervals)

    return which_loci, intervals

def process_trees_relate(which_loci, intervals, tCutoff=1e4, important=True, M=1, infile=None, outfile='temp', PATH_TO_RELATE='', u=1.25e-8, coalfile=None):

    """
    process trees, using Relate to sample branch lengths M times
    """
   
    # sample M trees at each chosen locus (if not already done) and save as dendropy trees
    print('getting trees...')
    trees = []
    progress_bar = tqdm(total=len(which_loci))
    for i,interval in enumerate(intervals):
        first_bp = int(interval[0])
        last_bp = int(interval[1])-1
        fname = outfile + '%d' %which_loci[i]
        _sample_times(PATH_TO_RELATE, infile, coalfile, u = u, M = M, first_bp = first_bp, last_bp = last_bp, outfile = fname) #sample branch times at tree, only run once
        trees.append(_get_dendropy_trees(fname+'.newick')) #load trees from relate
        progress_bar.update()
    progress_bar.close()

    # get demography 
    if coalfile == None:
        epochs = np.array([0,tCutoff]) #irrelevant if not importance sampling
        N = np.array([1,1]) #irrelevant if not importance sampling
    else:
        epochs, N = _get_epochs(coalfile)

    # process trees
    print('processing trees...')
    coal_times, pcoals, shared_times, samples = _process_trees(trees, epochs, N, tCutoff) #process trees

    return coal_times, pcoals, shared_times, samples 

def process_trees(ts, which_trees, from_ts=False, tCutoff=None, important=True, epochs=None, Nes=None):
    
    """
    convert trees (either newick files or tree sequence) into dendropy trees and process 
    """

    print('converting to dendropy trees...')
    trees = _get_dendropy_trees(ts, which_trees=which_trees, from_ts=from_ts) #first convert to denropy

    print('processing...')
    return _process_trees(trees=trees, epochs=epochs, Nes=Nes, tCutoff=tCutoff, important=important) #then process


def _sample_times(PATH_TO_RELATE, infile, coalfile, u = 1.25e-8, M = 10, first_bp = 1, last_bp = 1, outfile = None):

    """
    sample branch lengths at bp M times with Relate
    """

    if outfile == None:
        outfile = infile + '_sub' #name of output file
    
    #note that format only available with v1.1.* or greater
    if first_bp != None:
        script = '%s/scripts/SampleBranchLengths/SampleBranchLengths.sh \
                 -i %s \
                 --coal %s \
                 -m %.10f \
                 --num_samples %d \
                 --first_bp %d \
                 --last_bp %d \
                 --format n \
                 -o %s' %(PATH_TO_RELATE, infile, coalfile, u, M, first_bp, last_bp, outfile)
    else:
        script = '%s/scripts/SampleBranchLengths/SampleBranchLengths.sh \
                 -i %s \
                 --coal %s \
                 -m %.10f \
                 --num_samples %d \
                 -o %s' %(PATH_TO_RELATE, infile, coalfile, u, M, outfile)
    
    os.system(script) #run this on the command line, no need to return anything

def _get_dendropy_trees(treefile, which_trees=None, from_ts=False):

    """
    get dendropy trees from file of newick trees
    """

    if which_trees is not None:
        progress_bar = tqdm(total=len(which_trees))

    trees = []
    if not from_ts: #if taking from newick files
        with open(treefile, mode='r') as file:
            for i,line in enumerate(file): 
                if which_trees is None: #if using all trees 
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
        if which_trees is None: #if using all trees
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

def _get_epochs(coalfile):

    """
    get epoch times and population size in each from coal file (produced by Relate)
    """

    epochs = np.genfromtxt(coalfile,skip_header=1,skip_footer=1) #time at which each epoch starts (and the final one ends)
    N = 0.5/np.genfromtxt(coalfile,skip_header=2)[2:-1] #effective population size during each epoch (note that the coalescent rate becomes 0 after all trees have coalesced, and so Ne goes to infinity)
    N = np.array(list(N)+[N[-1]]) #add the final size once more to make same length as epochs

    return epochs,N

def _process_trees(trees, Nes=None, epochs=None, tCutoff=None, important=True):

    """
    function to get summaries of trees that dont depend on parameters
    """

    coal_times = []
    pcoals = []
    shared_times = []
    samples = []

    progress_bar = tqdm(total=len(trees))
    for locus in trees:
        coal_times_i = []
        pcoals_i = []
        shared_times_i = []
        samples_i = []
        for tree in locus:
            if important:
                cts = _coal_times(tree) #coalescence times
                coal_times_i.append(cts)
                pcoals_i.append(_log_coal_density(cts, Nes, epochs, tCutoff)) #probability of coalescence times in neutral coalescent
            sts = _shared_times(tree, tCutoff) #shared times and samples for each subtree
            shared_times_i.append(sts[0])
            samples_i.append(sts[1])
        coal_times.append(coal_times_i)
        pcoals.append(pcoals_i)
        shared_times.append(shared_times_i)
        samples.append(samples_i)
        progress_bar.update()
    progress_bar.close()

    if important:
        return coal_times, pcoals, shared_times, samples

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

