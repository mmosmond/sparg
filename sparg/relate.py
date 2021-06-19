import os

def _sample_times(PATH_TO_RELATE, infile, coalfile, u, M, first_bp, last_bp, outfile):
    'sample branch lengths of tree at given genome location (first_bp, last_bp) M times using anc/mut files (infile) and population size estimates (coalfile) and mutation rate u'
    'creates a newick file (outfile)'
    'see https://myersgroup.github.io/relate/modules.html#SampleBranchLengths'

    #note that format only available with Relate v1.1.* or greater
    script = '%s/scripts/SampleBranchLengths/SampleBranchLengths.sh \
             -i %s \
             --coal %s \
             -m %.10f \
             --num_samples %d \
             --first_bp %d \
             --last_bp %d \
             --format n \
             -o %s' %(PATH_TO_RELATE, infile, coalfile, u, M, first_bp, last_bp, outfile)
    
    os.system(script)

def _get_epochs(coalfile):
    'get epoch times and population size in each from coalfile'

    epochs = np.genfromtxt(coalfile, skip_header=1, skip_footer=1) #time at which each epoch starts (and the final one ends)
    N = 0.5/np.genfromtxt(coalfile, skip_header=2)[2:-1] #effective population size during each epoch (note that the coalescent rate becomes 0 after all trees have coalesced, and so Ne goes to infinity)
    N = np.array(list(N) + [N[-1]]) #add the final size once more to make same length as epochs

    return epochs, N

def _shared_times(tree, tCutoff=1e6):
    'create subtrees from cutting denropy tree off at tCutoff and get shared evolutionary times between sampled lineages within each subtree'

    # shared times for full tree
    pdm = tree.phylogenetic_distance_matrix() #denropy method to get time between samples
    taxa = np.array([i.taxon for i in tree.leaf_nodes()]) #taxa representing each sample
    n = len(taxa) #number of samples
    tmrcas = np.zeros((n,n)) #matrix to store time to mrcas
    for i in range(n):
        for j in range(i):
            tmrcas[i,j] = pdm(taxa[i],taxa[j])/2 #time to mrca is 1/2 of time between samples
            tmrcas[j,i] = tmrcas[i,j] #symmetric
    tmrca = np.max(tmrcas) #time to most recent common ancestor of all samples
    shared_time = min(tCutoff, tmrca) - tmrcas #shared time since tCutoff (or tmrca, if less)

    # shared times for subtrees
    i = 0 #start with first sample
    withi = shared_time[i]>=0 #true if share time with i
    timesi = shared_time[withi][:,withi] #shared times
    timesi = timesi - np.min(timesi) #trim off lineage from mrca to tcutoff
    times = [timesi] #start list with shared times of subtree with i
    samples = [np.where(withi)] #samples in this subtree
    taken = withi #samples already in a subtree
    while sum(taken) < n: #while some samples not yet in a subtree
        i = np.argmax(taken == False) #choose next sample not yet in a subtree
        withi = shared_time[i]>=0 #true if share time with i
        timesi = shared_time[withi][:,withi] #shared times of subtree with i
        timesi = timesi - np.min(timesi) #trim
        times.append(timesi) #append        
        samples.append(np.where(withi)) #samples in this subtree
        taken = np.array([i[0] or i[1] for i in zip(taken,withi)]) #samples already in a subtree

    samples = [[int(i.label) for i in taxa[j]] for j in samples] #samples in each subtree

    return times, samples 

def _coal_times(tree):
    'get coalescence times in ascending order from dendropy tree'

    return np.array(tree.internal_node_ages(ultrametricity_precision=False))

def _coal_intensity_memos(epochs, N):
    'coalescence intensity up to the end of each epoch'

    Lambda = np.zeros(len(epochs))
    for ie in range(1,len(epochs)):
        t0 = epochs[ie-1] #start time
        t1 = epochs[ie] #end time
        Lambda[ie] = (t1-t0) #elapsed time
        Lambda[ie] *= 1/(2*N[ie-1]) #multiply by coalescence intensity
        Lambda[ie] += Lambda[ie-1] #add previous intensity
    return Lambda

def _coal_intensity_using_memos(t, epochs, intensityMemos, N):
    'add coal intensity up to time t'

    iEpoch = int(np.digitize(np.array([t]),epochs)[0]-1) #epoch 
    t1 = epochs[iEpoch] #time at which the previous epoch ended
    Lambda = intensityMemos[iEpoch] #intensity up to end of previous epoch
    Lambda += 1/(2*N[iEpoch]) * (t-t1) #add intensity for time in current epoch
    return Lambda

def _log_coal_density(times, epochs, N, tCutoff=1e6):
    'log probability of coalescent times under standard neutral/panmictic coalescent'

    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    prevLambda = 0 #initialize coalescent intensity
    n = len(times) + 1 #number of samples
    times = times[times < tCutoff] #ignore old times
    times = times[times >= 0] #make sure everything non-negative!
    myIntensityMemos = _coal_intensity_memos(epochs,N) #intensities up to end of each epoch

    # probability of each coalescence time
    for i,t in enumerate(times): #for each coalescence event i at time t
        k = n-i #number of lineages remaining
        kchoose2 = k*(k-1)/2 #binomial coefficient
        Lambda = _coal_intensity_using_memos(t,epochs,myIntensityMemos,N) #coalescent intensity up to time t
        ie = np.digitize(np.array([t]), epochs) #epoch at the time of coalescence
        logpk = np.log(kchoose2 * 1/(2*N[ie])) - kchoose2 * (Lambda - prevLambda) #log probability (waiting times are time-inhomogeneous exponentially distributed)
        logp += logpk
        prevt = t
        prevLambda = Lambda

    # now add the probability of lineages not coalescing by tCutoff
    k -= 1 #after the last coalescence event we have one less sample
    kchoose2 = k*(k-1)/2 #binomial coefficient
    Lambda = _coal_intensity_using_memos(tCutoff,epochs,myIntensityMemos,N) #coalescent intensity up to time tCutoff
    logPk = - kchoose2 * (Lambda - prevLambda) #log probability of no coalescence
    logp += logPk

    return logp[0]

def _process_trees(trees, epochs, N, tCutoff=1e6):
    'function to get summaries of trees that dont depend on parameters'

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
            cts = _coal_times(tree) #coalescence times
            coal_times_i.append(cts)
            pcoals_i.append(_log_coal_density(cts, epochs, N, tCutoff)) #probability of coalescence times in neutral coalescent
            sts = _shared_times(tree, tCutoff) #shared times and samples for each subtree
            shared_times_i.append(sts[0])
            samples_i.append(sts[1])
        coal_times.append(coal_times_i)
        pcoals.append(pcoals_i)
        shared_times.append(shared_times_i)
        samples.append(samples_i)
        progress_bar.update()
    progress_bar.close(); del progress_bar

    return coal_times, pcoals, shared_times, samples

def _get_dendropy_trees(treefile, which_trees=None, fromts=False):
    'get dendropy trees from file of newick trees produced by relate'

    if which_trees != None:
        progress_bar = tqdm(total=len(which_trees))
    trees = []
    if fromts == False: #if taking from newick files
        with open(treefile, mode='r') as file:
            for i,line in enumerate(file): 
                if which_trees == None: #if using all trees 
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
        if which_trees == None: #if using all trees
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
            
    if which_trees != None:
        progress_bar.close()
    return trees

def _filter_samples_times(samples, shared_times, keep_ix):
    '''filter to keep only specified samples and shared_times'''
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

def _log_birth_density(times, phi, tCutoff=1e6, condition_on_n=True):
    'log probability of coalescence times given Yule process with splitting rate phi'

    n = len(times) + 1 #number of samples is 1 more than number of coalescence events
    T = min(tCutoff, times[-1]) #furthest time in past we are interested in is the min of the cutoff time and the oldest coalesence event
    times = T - times #switch to forward in time perspective from T
    times = np.sort(times[times>0]) #remove older coalescence events (we'll assume panmixia beyond tCutoff, so this will cancel out in importance sampling)
    n0 = n-len(times) #number of lineages at time T

    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    # probability of each coalescence time
    for i,t in enumerate(times): #for each coalescence event i at time t
        k = n0+i #number of lineages before the event
        logp += np.log(k*phi) - k * phi *  (t - prevt) #log probability of waiting time  t-prevt (waiting times are exponentially distributed with rate k*phi)
        prevt = t #update time

    #add probability of no coalescence to present time (from most recent coalescence to present, with n samples and rate n*phi)
    logp += - n * phi * (T - prevt)

    #condition on having n samples from n0 in time T
    if condition_on_n:
        logp -= np.log(math.comb(n-1,n-n0)*(1-np.exp(-phi*T))**(n-n0)) - phi * n0 * T # see page 234 of https://www.pitt.edu/~super7/19011-20001/19531.pdf for two different expressions

    return logp

def _lognormpdf(x, mu, S, relative=True):
    """ Calculate log probability density of x, when x ~ N(mu,sigma) """

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

def _location_loglikelihood(locations, shared_time, Sigma, tsplits=[], remove_missing=False):
    'log likelihood of locations given shared times and dispersal matrix'

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
    locs = np.matmul(Tmat,locations) #mean center locations and drop last sample (bc lost a degree of freedom when taking mean); see Lee & Coop 2017 Genetics eq A.16; note numpy broadcasts across the columns in locs
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
        return 0 #CHECK: while not a function of SIGMA, this give a likelihood of 1 with this subtree and whatever SIGMA is being evaluated -- so we'd need this subtree to give a 1 (ie be singular) for all SIGMA to avoid bias

def _loglikelihood_ratio(shared_times, samples, pcoal, coal_times, phi, Sigma, locations, tCutoff=1e6, important=True, tsplits=[], remove_missing=False):
        'log likelihood of locations given parameters and tree summaries'

        LLR = 0
        for i,shared_time in enumerate(shared_times):
            if len(shared_time) > 2: #need at least two samples in subtree to mean center
                LLR += _location_loglikelihood(locations[samples[i]], shared_time, Sigma, tsplits, remove_missing) #log likelihood of locations given shared evolutionary times, dispersal matrix, and MRCA location
        if important:
            LLR += _log_birth_density(coal_times, phi, tCutoff) #log probability of coalescence times given pure birth process with rate phi
            LLR -= pcoal #log probability density of coalescence times under standard coalescent with varying population size and cutoff

        return LLR

def _logsumexp(a):
    'take the log of a sum of exponentials without losing information'

    a_max = np.max(a) #max element in list a
    tmp = np.exp(a - a_max) #now subtract off the max from each a before taking exponential (ie divide sum of exponentials by exp(a_max))
    s = np.sum(tmp) #and sum those up
    out = np.log(s) #and take log
    out += a_max  #and then add max element back on (ie multiply sum by exp(a_max), ie add log(exp(a_max)) to logged sum)

    return out

def _mc(shared_times, samples, pcoals, coal_times, phi, Sigma, locations, tCutoff = 1e6, important=True, tsplits=[], remove_missing=False):
    'estimate log likelihood ratio of the locations given parameters (A,Sigma,phi) vs data given standard coalescent with Monte Carlo'

    M = len(pcoals) #number of samples of branch lengths
    LLRs = np.zeros(M) #log likelihood ratios
    for i,shared_time in enumerate(shared_times):

        LLRs[i] = _loglikelihood_ratio(shared_time, samples[i], pcoals[i], coal_times[i], phi, Sigma, locations, tCutoff, important, tsplits, remove_missing)

    LLRhat = _logsumexp(LLRs) - np.log(M) #average over trees

    return LLRhat #monte carlo estimate of log likelihood ratio

def _sigma_phi(x, tsplits=[], important=True):
    'set up parameters to be estimated'

    Sigma = [] #make as list
    for i in range(len(tsplits) + 1):
        Sigma.append(np.array([[x[3*i]**2, x[3*i]*x[3*i+1]*x[3*i+2]], [x[3*i]*x[3*i+1]*x[3*i+2], x[3*i+1]**2]])) #append covariance matrices (created from standard deviations and correlation) for each epoch
    if important:
        phi = x[3*(len(tsplits) + 1)]
    else:
        phi = 1 #if not importance sampling this value is irrelevant, so we just supply an arbitrary value in case nothing supplied by user

    return [Sigma,phi]

def _sum_mc(locations, coal_times, pcoals, shared_times, samples, n = None, important = True, tCutoff = 1e6, tsplits=[], scale_phi=1, remove_missing=False):
    'sum monte carlo estimates of log likelihood ratios across loci'

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

def find_mle(locations, coal_times, pcoals, shared_times, samples, x0, bnds, n=None, important=True, tCutoff=1e6, tsplits=[], quiet=False, method='SLSQP', options=None, scale_phi=1, remove_missing=False):
    'find maximum likelihood parameters from treefile'

    M = len(pcoals[0]) #number of trees per locus 
    if n == None:
        n = M #number of trees to use per locus

    f = _sum_mc(locations, coal_times, pcoals, shared_times, samples, n=n, important=important, tCutoff=tCutoff, tsplits=tsplits, scale_phi=scale_phi, remove_missing=remove_missing) #negative composite log likelihood ratio

    if quiet==False:
        print('searching for maximum likelihood parameters...')
        t0 = time.time()

    m = minimize( f, x0 = x0, bounds = bnds, method=method, options=options) #find MLE

    if quiet==False:
        print('the max is ', m.x)
        print('finding the max took', time.time()-t0, 'seconds')

    return m

def recap_mutate_sample(treefile, RBP, Ne, U, k, outfile, vcf=True, output=False, W=50, d=100, keep_unary=False, recap=True, mutate=True, dump_simplified=False, dump_recapped=False, dump_mutated=True):

    # load
    ts = pyslim.load(treefile) #load tree sequence
    
    # sample present-day individuals
    alive = ts.individuals_alive_at(0) #individuals alive at sampling time (exlcudes any remembered ancients)
    alive_locs = ts.individual_locations[alive] #locations of all living individuals
    alive = alive[((alive_locs[:, 0] - W/2)**2 + (alive_locs[:, 1] - W/2)**2)**0.5 < d] #individuals that are alive and have euclidean distance from center less than d
    inds = np.random.choice(alive, k, replace=False) #select k individuals randomly from current generation
    ind_locs =  alive_locs[inds] #get locations of samples
    np.savetxt(outfile+'.locs', ind_locs) #save locations associated with sample too

    # get nodes (haploid genomes) of present-day samples
    ind_nodes = [] #empty vector for nodes
    for ind in inds: #for each individual
        for node in ts.individual(ind).nodes:
            ind_nodes.append(node) #append each genome
    
    # simplify 
    ts = ts.simplify(ind_nodes, keep_unary=keep_unary) #option to keep info on unary nodes too
    if dump_simplified:
        ts.dump(outfile+'_simplified_true.trees')
 
    # recap
    if recap:
        ts = ts.recapitate(recombination_rate=RBP, Ne=Ne) #recapitate using equil census popn size as effective size
        if dump_recapped:    
            ts.dump(outfile+'_recapped_true.trees') #save for later use
    
    # mutate
    if mutate:
        ts = pyslim.SlimTreeSequence(msprime.mutate(ts, rate=U)) # layer on neutral mutations
        if dump_mutated:
            ts.dump(outfile+'_true.trees')
        if vcf:
            # write the SNP data for this sample to vcf for use with Relate later
            with open(outfile+'.vcf', "w") as vcf_file:
                ts.write_vcf(vcf_file, individuals = range(k))

    if output:	
        return ts, ind_locs

def infer_ts(datadir, filename, path_to_relate, RBP, L, U, k, true_trees_file=None, outfile=None, popsize=True, output=False, threshold=0, num_iter=1, nthreads=1, memory=5):

    fname = datadir+filename #vcf file, will also name haps/sample, map, and non-biallelic files similarly

    # convert vcf to haps/sample
    script="%s/bin/RelateFileFormats \
                --mode ConvertFromVcf \
                --haps %s.haps \
                --sample %s.sample \
                --chr 1 \
                -i %s" %(path_to_relate, fname, fname, fname)
    os.system(script)

    # make uniform recombination map
    r = (1 - (1 - 2 * RBP)**L)/2 #recombination distance from one end of chromosome to other
    cm = 50 * np.log(1/(1-2*r)) #length in centiMorgans
    cr = cm/L * 1e6 #cM per million bases
    script = "pos COMBINED_rate Genetic_Map \n0 %f 0 \n%d %f %f" %(cr, L, cr, cm)
    os.system("echo '" + script + "' > %s.map" %fname)

    # remove non-biallelic SNPs (given msprime assumes infinite sites, this must be SNPs that are fixed or lost in the sample?? -- throws error without most of time)
    script="%s/bin/RelateFileFormats \
         --mode RemoveNonBiallelicSNPs \
         --haps %s.haps \
         -o %s_biallelic" %(path_to_relate, fname, fname)
    os.system(script)

    # estimate effective pop size from pi
    if true_trees_file == None:
        true_trees_file = fname+'_true.trees' #for backwards compatibility
    ts_true = msprime.load(true_trees_file)
    Ne = round(ts_true.diversity() / (4 * U))

    # clear directory (eg failed past attempts)
    script="rm -rf %s" %(filename)
    os.system(script)

    # infer tree with Relate (using correct mutation rate and recombination map, and estimated (diploid) effective population size)
    script="%s/bin/Relate \
                --mode All \
                -m %.10f \
                -N %f \
                --haps %s_biallelic.haps \
                --sample %s.sample \
                --map %s.map \
                --memory %d \
                -o %s" %(path_to_relate, U, 2*Ne, fname, fname, fname, memory, filename)
    os.system(script)

    # the above needed to be output in the working directory, so now we move it to the data folder
    script="mv %s.* %s" %(filename, datadir)
    os.system(script)
    
    # clear temp files
    script="rm -rf %s" %(filename)
    #script="rm -rf chunk_0" #for relate 110
    os.system(script)

    if outfile == None:
        outfile = fname #for backwards compatibility
 
    if popsize == False: #if not inferring pop size

        # convert Relate format into tree sequence
        script="%s/bin/RelateFileFormats \
                    --mode ConvertToTreeSequence \
                    -i %s \
                    -o %s " %(path_to_relate, fname, outfile)
        os.system(script)

    # if we are inferring popsize
    else:

        # make a file of population labels
        os.system('echo "sample population group sex" > %s.poplabels' %fname)
        for i in range(k):
            os.system('echo "%d 1 1 NA" >> %s.poplabels' %(i,fname)) #change NA to 1 if using haploids

        # then run the relate script
        script = "%s/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
              -i %s \
              -o %s \
              -m %.10f \
              --poplabels %s.poplabels \
              --years_per_gen 1 \
              --threshold %f \
              --num_iter %d \
              --threads %d" %(path_to_relate, fname, outfile, U, fname, threshold, num_iter, nthreads)
        os.system(script)

        # convert Relate format into tree sequence
        script="%s/bin/RelateFileFormats \
                    --mode ConvertToTreeSequence \
                    -i %s \
                    -o %s " %(path_to_relate, outfile, outfile)
        os.system(script)

    if output:
      ts_inf = msprime.load("%s.trees" %outfile)
      return ts_inf

def choose_loci(ts, which=[0], mode='site'):

    breakpoints = ts.breakpoints(as_array=True)
    
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
       
    print('number of loci: ',len(which_loci))
 
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

def process_trees(which_loci, intervals=None, tCutoff=51**2*2*4, important=False, M=1, infile=None, outfile='temp', PATH_TO_RELATE='', u=1.25e-8, coalfile=None, ts=None):
   
    if important: 
        # sample M trees at each chosen locus (if not already done) and save as dendropy trees
        print('getting trees...')
        trees = []
        progress_bar = tqdm(total=len(which_loci))
        for i,interval in enumerate(intervals):
            first_bp = int(interval[0])
            last_bp = int(interval[1])-1
            fname = outfile + '%d' %which_loci[i]

            # try using existing sample of trees (careful)
            #try:
            #    trees.append(_get_dendropy_trees(fname+'.newick')) #load trees from relate
            #    print('using existing samples of branch lengths -- hope all parameters are unchanged!')
            #except:
            #    _sample_times(PATH_TO_RELATE, infile, coalfile, u = u, M = M, first_bp = first_bp, last_bp = last_bp, outfile = fname) #sample branch times at tree, only run once
            #    trees.append(_get_dendropy_trees(fname+'.newick') #load trees from relate

            # enforce sampling new trees (in case something has changed, like M)
            _sample_times(PATH_TO_RELATE, infile, coalfile, u = u, M = M, first_bp = first_bp, last_bp = last_bp, outfile = fname) #sample branch times at tree, only run once
            trees.append(_get_dendropy_trees(fname+'.newick')) #load trees from relate

            progress_bar.update()
        progress_bar.close()
        #trees = [item for sublist in trees for item in sublist] #flatten

    # if not importance sampling, ie true trees
    else:
        print('getting trees...')
        trees = _get_dendropy_trees(ts, which_loci, fromts=True) #load as dendropy trees for downstream analyses
        trees = [[tree] for tree in trees] #repackage by locus    

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

def mle_dispersal(locations, shared_time):
    'maximum likelihood dispersal rate'
    
    not_missing = np.argwhere(np.isfinite(locations).any(axis=1)).flatten()
    if len(not_missing) < len(locations):
        print('removing', len(locations) - len(not_missing), 'samples with missing locations from dispersal estimate')
        locations = locations[not_missing]
        shared_time = shared_time[not_missing][:,not_missing]

    n = len(locations) #number of samples
     
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[:-1]; #matrix for mean centering and dropping one sample
    locs = np.matmul(Tmat, locations) #mean center locations and drop last sample (bc lost a degree of freedom when taking mean); see Lee & Coop 2017 Genetics eq A.16; note numpy broadcasts across the columns in locs
    stime = np.matmul(Tmat, np.matmul(shared_time, np.transpose(Tmat))) #mean center the shared time matrix
    
    Tinv = np.linalg.pinv(np.array(stime)) #inverse of shared time matrix
    
    SigmaHat = np.matmul(np.matmul(np.transpose(locs), Tinv), locs) / (n-1) #mle dispersal rate 
    
    return SigmaHat

def get_mles(ts, locations, x0, bnds=None, which_sites=[0], tCutoff=50**2*2*4, important=False, M=1, infile='temp', outfile='temp', PATH_TO_RELATE='', u=1.25e-8, get_mles=True, coalfile=None, try_analytic=False, method='SLSQP', tsplits=[], scale_phi=1):

    # loci indices
    print('choosing loci...')
    which_loci = [ts.at(i).index if i<ts.sequence_length else ts.num_trees-1 for i in which_sites] #index/locus for each chosen site (note if site number too high we use the last tree)

    # if importance sampling
    if important:
        # genomic intervals of each chosen locus
        intervals=[]
        for i,tree in enumerate(ts.trees()):
            if i>which_loci[-1]:
                break
            elif i in which_loci:
                intervals.append([tree.interval[0],tree.interval[1]])

        # sample M trees at each chosen locus (if not already done) and save as dendropy trees
        print('getting trees...')
        trees = []
        progress_bar = tqdm(total=len(which_loci))
        for i,interval in enumerate(intervals):
            first_bp = int(interval[0])
            last_bp = int(interval[1])-1
            fname = outfile + '_%d' %which_loci[i]
            try:
                trees.append(_get_dendropy_trees(fname+'.newick')) #load trees from relate
            except:
                _sample_times(PATH_TO_RELATE, infile, coalfile, u = u, M = M, first_bp = first_bp, last_bp = last_bp, outfile = fname) #sample branch times at tree, only run once
                trees.append(_get_dendropy_trees(fname+'.newick')) #load trees from relate
            progress_bar.update()
        progress_bar.close()
        #trees = [item for sublist in trees for item in sublist] #flatten

    # if not importance sampling, ie true trees
    else:
        print('getting trees...')
        trees = _get_dendropy_trees(ts, which_loci, fromts=True) #load as dendropy trees for downstream analyses
        trees = [[tree] for tree in trees] #format to have two axes, one for locus, and another for samples of trees at a locus (here just one)

    # get demography 
    if coalfile == None:
        epochs = np.array([0,tCutoff]) #irrelevant if not importance sampling
        N = np.array([1,1]) #irrelevant if not importance sampling
    else:
        epochs, N = _get_epochs(coalfile)

    # process trees
    print('processing trees...')
    coal_times, pcoals, shared_times, samples = _process_trees(trees, epochs, N, tCutoff) #process trees

    # find max composite likelihood estimate over all sites
    print('getting max composite likelihood estimate...')
    mcle = find_mle(locations, coal_times, pcoals, shared_times, samples, x0=x0, bnds=bnds, important=important, tCutoff=tCutoff, method=method, tsplits=tsplits, quiet=False, scale_phi=scale_phi)

    # mles at each site
    if get_mles:
        mles = np.empty((len(which_sites),len(x0)))
        print('getting mles at each locus...')
        progress_bar = tqdm(total=len(which_sites))
        for i in range(len(which_sites)):
            #print(len(shared_times[i]))
            # decide whether to get analytically or numerically
            if try_analytic and M == 1 and len(shared_times[i]) == 1 and len(tsplits) == 0: #can get analytic MLE if just one tree at a locus (but very memory intensive to invert matrices)
                print('trying analytically')
                sigmaHat = mle_dispersal(locations[samples[i][0]], shared_times[i][0])
                mles[i,0] = sigmaHat[0,0]**(1/2)
                mles[i,1] = sigmaHat[1,1]**(1/2)
                mles[i,2] = sigmaHat[0,1] / (mles[i,0]*mles[i,1])
                mles[i,3] = x0[3]
            else: #otherwise got to get numerically
                mle = find_mle(locations, [coal_times[i]], [pcoals[i]], [shared_times[i]], [samples[i]], x0=x0, bnds=bnds, important=important, tCutoff=tCutoff, quiet=True, method=method, tsplits=tsplits, scale_phi=scale_phi)
                mles[i] = mle.x
            progress_bar.update()
        progress_bar.close()

        return mles, mcle
    else:
        return mcle

def get_shared_times(ts):

    k = ts.num_samples
    W = np.identity(k) # each node, i in [0,1,...,k] given row vector of weights with 1 in column i and 0's elsewhere 
    def f(x): return (x.reshape(-1,1) * x).flatten() # matrix with 1's where branch above a node with value x contributes to shared time between samples 
    
    return ts.general_stat(
        W, f, k**2, mode='branch', windows='trees', polarised=True, strict=False
    ).reshape(ts.num_trees, k, k)

def get_true_ancestral_locations(treeseq, loci, focal_nodes, times, progress_bar=False):
        
    # get all ancient nodes and locations
    ancient_nodes = []
    ancient_locations = []
    for time in times:

        # get all ancient nodes at given time
        ancient_nodes_time = []
        ancient_locations_time = []
        for ind in treeseq.individuals_alive_at(time): # for all individuals alive at this previous time

            # get all genomes
            for genome in treeseq.individual(ind).nodes:
                ancient_nodes_time.append(genome)
                # get all locations
                ancient_locations_time.append(treeseq.individual(ind).location) #append location of individual the same number of times as there are nodes

        # add list of ancient nodes at time t to larger list
        ancient_nodes.append(ancient_nodes_time)
        ancient_locations.append(ancient_locations_time)

    # find locations of ancestors for all samples
    if progress_bar:
        prog_bar = tqdm(total=len(loci)*len(times))
    
    # loop through loci
    ancestors = []
    for locus,tree in enumerate(treeseq.trees()):
        if locus > max(loci):
            break #stop if can
        elif locus in loci:
    
            # loop through times
            ancestors_locus = []
            for ancient_nodes_time, ancient_locations_time in zip(ancient_nodes,ancient_locations):
                
                # loop through ancient nodes
                ancestors_locus_time = np.empty((len(focal_nodes),3)) #set up list of ancestors
                for i,ancient_node in enumerate(ancient_nodes_time):

                    # loop through descendants
                    for leaf in tree.leaves(ancient_node):
                        if leaf in focal_nodes: #just focal samples, as leaves() returns ancient_node if no leaves
                            ancestors_locus_time[leaf] = ancient_locations_time[i] #save ancestral location of leaf
                
                ancestors_locus.append(ancestors_locus_time)
                
                if progress_bar:
                    prog_bar.update()
    
            ancestors.append(ancestors_locus)
    
            if progress_bar:
                prog_bar.close()
    
    ancestors = np.array(ancestors).swapaxes(1,2) #numpy array with order: locus, node, time
    return ancestors


def get_mle_ancestral_locations(loci, focal_nodes, times, shared_times, locations, SIGMAs=None, covariance=False):
    'get the mle location, and covariance around it, of focal_nodes ancestor time generations ago'
    'shared_time is matrix of shared times between samples'
    'locations is matrix of locations of samples'
    'SIGMA is estimate of dispersal rate (covariance matrix)'

    mles = np.empty((len(loci),len(focal_nodes),len(times),2))
    covs = np.empty((len(loci),len(focal_nodes),len(times),2,2))
    
    for i,locus in enumerate(loci):
        shared_time = shared_times[locus]
    
        for j,focal_node in enumerate(focal_nodes):
            
            tAA = tmrca - time #time to mrca from ancestor (ie shared time with itself)
            Atimes = [] #empty vector for shared times with other nodes
            for st in shared_time[focal_node]: #for each time the focal node shares with all other nodes
                Atimes.append(np.min([tAA, st])) #the shared time with the ancestor of the focal node is the smaller of tAA and the time shared with the focal node
            Atimes = np.array(Atimes)

            # combine ancestor times with shared times among samples
            n,m = shared_time.shape #number of samples
            stime = np.zeros((n+1,m+1))
            stime[0,0] = tAA #first diagonal element
            stime[0,1:] = Atimes #remainder of first row
            stime[1:,0] = Atimes #remainder of first column
            stime[1:,1:] = shared_time #remainder of matrix

            # retrieve submatrices from combined times
            Sigma11 = stime[:1,:1]
            Sigma12 = stime[:1,1:]
            Sigma21 = stime[1:,:1]
            Sigma22 = stime[1:,1:]

            # mean centering
            Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[:-1] #mean centering matrix

            Sigma22c = np.matmul(Tmat, np.matmul(Sigma22, np.transpose(Tmat))) #mean centered covariance matrix of samples
            if len(Sigma22c) == 0:
                Sigma22c = np.zeros((1,1)) #in case there is only one sample, which is removed by mean centering, just set shared time with self to be zero

            locationsc = np.matmul(Tmat, locations) #mean centered locations
            if len(locationsc) == 0:
                locationsc = np.zeros((1,m)) #again, if we dropped the only sample just set to zero

            Sigma21c = np.matmul(Tmat, Sigma21) - np.matmul(np.matmul(Tmat, Sigma22), np.ones(n).reshape(-1,1)/n) #mean center ancestors shared time with samples
            if len(Sigma21c) == 0:
                Sigma21c = np.zeros((1,1)) #in case there is only one sample, which is removed by mean centering, just set shared time with ancestor to be zero

            # MLE ancestor location and covariance around it
            Sigma22c_pinv = np.linalg.pinv(Sigma22c)
            mles[i,j,t] = np.mean(locations, axis=0) + np.matmul( np.matmul(Sigma21c.transpose(), Sigma22c_pinv), locationsc)
            if covariance:
                x = np.array([[1] + [-1/n] * n]) #vector to mean center ancestors shared time with itself
                Sigma11c = np.matmul(np.matmul(x, stime) , x.transpose()) # mean center ancestors shared time with itself
                covs[i,j,t] = (Sigma11c - np.matmul(np.matmul(Sigma21c.transpose(), Sigma22c_pinv), Sigma21c)) * SIGMA

    if covariance:            
        return mles, covs
    else:
        return mles

def _get_focal_index(focal_node, listoflists):
    'get the subtree and index within that subtree for given focal node'
    for i,j in enumerate(listoflists):
        if focal_node in j:
            n = i
            for k,l in enumerate(j):
                if focal_node == l:
                    m = k
    return n,m

def _ancestor_location_loglikelihood(shared_time, locs, focal_node=None, t=None, SIGMA=None, Atimes=None, tsplits=[], MLE_only=False):
    'log likelihood location of ancestor of focal_node t generations ago'

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

def _ancestor_location_meanloglikelihood(coal_times, pcoals, shared_times, samples, locations, focal_node, keep=None, phi=1, SIGMA=None, locus=0, time=0, tCutoff=1e6, importance=True, tsplits=[], BLUP=False):
    'log likelihoods of ancestor locations of focal node at locus for M samples of tree'

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

def locate(treefiles, mlefiles, nodes, locations, times=[0], tCutoff=1e4, importance=True, x0=[0,0], bnds=((None,None),(None,None)), method='L-BFGS-B', weight = True, scale_phi=1, keep=None, tsplits=[], BLUP=False):
    'locate nodes at times times using locations of keep samples'

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
        SIGMA = [] #make as list
        for i in range(len(tsplits) + 1):
            SIGMA.append(np.array([[x[3*i]**2, x[3*i]*x[3*i+1]*x[3*i+2]], [x[3*i]*x[3*i+1]*x[3*i+2], x[3*i+1]**2]])) #append matrices for each epoch (chronological order)

        # if we're importance sampling we need the branching rate 
        if importance:
            phi = x[-1]/scale_phi
        else:
            phi = 1 #irrelevant and arbitary value
        
        # loop over times
        mle_locations_t = [] #locations for all nodes at all loci at this time
        weights_t = [] #weights of locations (measure of uncertainty)
        for time in times:

            # loop over nodes
            mle_locations_i = [] #locations for all nodes at particular time at this locus
            weights_i = []
            for focal_node in nodes:
                
                # get location of node i at time t at locus
                f = _ancestor_location_meanloglikelihood(coal_times, pcoals, shared_times, samples, locations, focal_node, keep=keep, phi=phi, SIGMA=SIGMA, locus=0, time=time, tCutoff=tCutoff, importance=importance, tsplits=tsplits, BLUP=BLUP) #mean log likelihood or mean MLE (if BLUP)

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

    if weight:
        return np.array(mle_locations), np.array(weights)
    else:
        return np.array(mle_locations) 
