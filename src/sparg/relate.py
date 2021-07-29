from tqdm.auto import tqdm
from sparg.trees import get_dendropy_trees
import os
import numpy as np

def get_epochs(coalfile):
    """Get epochs and population sizes in each from coal file (produced by Relate).

    Parameters:
        coalfile (string): Name of .coal file.

    Returns:
        epochs (numpy array): Start and end times of each epoch.
        Nes (np array): Effective population sizes in each epoch (with last entry repeated).
    """

    epochs = np.genfromtxt(coalfile,skip_header=1,skip_footer=1) #time at which each epoch starts (and the final one ends)
    Nes = 0.5/np.genfromtxt(coalfile,skip_header=2)[2:-1] #effective population size during each epoch (note that the coalescent rate becomes 0 after all trees have coalesced, and so Ne goes to infinity)
    Nes = np.array(list(Nes)+[Nes[-1]]) #add the final size once more to make same length as epochs

    return epochs, Nes

def sample(which_trees, intervals, PATH_TO_RELATE, infile, coalfile, U, M=10):
    """Use Relate to resample trees (branch lengths) and then convert to dendropy trees

    Parameters:
        which_trees (array-like): Indices of trees we are resampling.
        intervals (array-like): Genomic intervals [start, stop] of trees we are resampling.
        PATH_TO_RELATE (string): Path to Relate on your machine.
        infile (string): Name of anc/mut files (without .anc or .mut) that we will use for resampling. Also determines name of output files created in process.
        coalfile (string): Name of .coal file that we will use for resampling.
        U (float): Mutation rate, per basepair per generation.
        M (int): Number of times to resample. Note that we will often only use the resamples (and not the original trees), and so this is the number of sampled trees for importance sampling.

    Returns:
        trees: List of dendropy trees, size l x M where l is the number of loci and M is the number of resamples
    """
   
    # sample M trees at each chosen locus (if not already done) and save as dendropy trees
    trees = []
    progress_bar = tqdm(total = len(which_trees))
    for tree,interval in zip(which_trees, intervals):
        
        first_bp = int(interval[0]) #start point
        last_bp = int(interval[1]) - 1 #end point
        outfile = infile + '_tree%d' %tree #outfile name

        try:
            trees.append(get_dendropy_trees(outfile + '.newick')) #load trees from relate
        except:
            _sample_times(PATH_TO_RELATE=PATH_TO_RELATE, infile=infile, outfile=outfile, coalfile=coalfile, first_bp=first_bp, last_bp=last_bp, U=U, M=M) #sample branch lengths M times
            trees.append(get_dendropy_trees(outfile + '.newick')) #load trees from relate
        
        progress_bar.update()
    progress_bar.close()

    return trees 

def _sample_times(PATH_TO_RELATE, infile, outfile, coalfile, first_bp, last_bp, U, M=10):

    """
    sample branch lengths of tree M times with Relate
    """

    script = '%s/scripts/SampleBranchLengths/SampleBranchLengths.sh \
             -i %s \
             --coal %s \
             -m %.10f \
             --num_samples %d \
             --first_bp %d \
             --last_bp %d \
             --format n \
             -o %s' %(PATH_TO_RELATE, infile, coalfile, U, M, first_bp, last_bp, outfile)
    
    os.system(script) #run this on the command line, no need to return anything

