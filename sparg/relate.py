from tqdm.auto import tqdm
from sparg.trees import get_dendropy_trees
import os
import numpy as np

def sample(which_trees, intervals, PATH_TO_RELATE, infile, coalfile, U, M=10):

    """
    get resampled trees from Relate
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

def get_epochs(coalfile):

    """
    get epoch times and population size in each from coal file (produced by Relate)
    """

    epochs = np.genfromtxt(coalfile,skip_header=1,skip_footer=1) #time at which each epoch starts (and the final one ends)
    Nes = 0.5/np.genfromtxt(coalfile,skip_header=2)[2:-1] #effective population size during each epoch (note that the coalescent rate becomes 0 after all trees have coalesced, and so Ne goes to infinity)
    Nes = np.array(list(Nes)+[Nes[-1]]) #add the final size once more to make same length as epochs

    return epochs, Nes
