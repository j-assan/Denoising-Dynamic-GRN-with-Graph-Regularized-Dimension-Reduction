import sys; sys.path += ['/cluster/home/jassan/projects/Beeline']
import pandas as pd
import numpy as np
import pyttb as ttb
import scanpy as sc
import os
import pickle
from experiment import generate_laplacian
from cp_als import cp_als
# from BLEval.computeJaccard import computePairwiseJaccard

def computePairwiseJacc(inDict):
    """
    A helper function to compute all pairwise Jaccard similarity indices
    of predicted top-k edges for a given set of datasets (obtained from
    the same reference network). Here k is the number of edges in the
    reference network (excluding self loops). 
    
    

    :param inDict:  A dictionary contaninig top-k predicted edges  for each dataset. Here, keys are the dataset name and the values are the set of top-k edges.
    :type inDict: dict
    :returns:
        A dataframe containing pairwise Jaccard similarity index values
    """
    jaccDF = {key:{key1:{} for key1 in inDict.keys()} for key in inDict.keys()}
    for key_i in inDict.keys():
        for key_j in inDict.keys():
            num = len(inDict[key_i].intersection(inDict[key_j]))
            den = len(inDict[key_i].union(inDict[key_j]))
            if den != 0:
                jaccDF[key_i][key_j] = num/den
            else:
                jaccDF[key_i][key_j] = 0
    return pd.DataFrame(jaccDF)


INPUT_PATH = "/cluster/scratch/jassan/planarian/"
# # Load the adata
# adata = sc.read_h5ad(os.path.join(INPUT_PATH,"planarians-timecourse.h5ad"))
# lengths = np.sort(list(set(adata.obs["Lengthmm"].tolist())))
# np.savetxt(os.path.join(INPUT_PATH, 'lenghts'), lengths)
def jacc():
    k = 300
    lengths = np.loadtxt(os.path.join(INPUT_PATH, 'lengths.txt'))
    jacc = np.zeros((len(lengths), 5, 5))
    mode = 'bin'
    for i, l in enumerate(lengths):
        sets = {}
        fname = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-{mode}/network_{l.round(3)}.csv')
        df = pd.read_csv(fname, sep='\t').loc[0:k-1]
        assert df['importance'].loc[k-1].round(6) > 0
        df = df.drop('importance', axis=1)
        sets['100'] = set(df['TF'] + "|" + df['target'])
        fracs = ['80', '60', '40', '20']
        for frac in fracs:
            fname = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-{mode}-frac{frac}/network_{l.round(3)}.csv')
            df = pd.read_csv(fname, sep='\t').loc[0:k-1]
            assert df['importance'].loc[k-1].round(6) > 0
            df = df.drop('importance', axis=1)
            sets[frac] = set(df['TF'] + "|" + df['target'])
        jacc[i] = computePairwiseJacc(sets)
    np.save(os.path.join(INPUT_PATH, f'{mode}_jacc.npy'), jacc)
    np.savetxt(os.path.join(INPUT_PATH, f'{mode}_jacc.txt'), jacc.mean(axis=0))


def denoise(i, mode='bin', seed=297192305631970044400581332683131391363):
    fracs = ['', '-frac80', '-frac60', '-frac40', '-frac20']
    frac = fracs[i]
    seeds = np.random.SeedSequence(seed).spawn(10)
    if mode == 'base':
        seed = seeds[i]
    elif mode == 'bin':
        seed = seeds[5+i]
    INPUT_PATH = "/cluster/scratch/jassan/planarian/"
    # Load the adata
    adata = sc.read_h5ad(os.path.join(INPUT_PATH,"planarians-timecourse.h5ad"))
    TF_file = pd.read_csv('TF-list2.csv')
    idx = np.where( TF_file['TF.group'] != '-')[0]
    TFs = TF_file['gene'][idx].tolist()
    adata_TF = adata[:, adata.var.index.isin(TFs)]
    TF_names = adata_TF.var.index.tolist()

    adata_subset = adata[ adata.obs['broad_cell_type'].isin(['Epidermal','Neoblast']), : ]
    sc.pp.highly_variable_genes( adata_subset, n_top_genes=500)
    adata_to_use = adata_subset[ : , adata_subset.var.index.isin(TFs) + adata_subset.var['highly_variable'] == True  ]
    genes_to_use = adata_to_use.var.index.tolist()
    lengths = np.sort(list(set(adata.obs["Lengthmm"].tolist())))

    # Precompute index mappings
    tf_index = {tf: i for i, tf in enumerate(TF_names)}
    gene_index = {g: j for j, g in enumerate(genes_to_use)}

    # Allocate array
    T = len(lengths)
    matrices = np.zeros((T, len(TF_names), len(genes_to_use)))

    for i in range(T):
        print(i)

        out_file = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-{mode}{frac}/network_{lengths[i].round(3)}.csv')
        network = pd.read_csv(out_file, sep='\t')

        # Filter for TFs and targets we care about
        network = network[network['TF'].isin(TF_names) & network['target'].isin(genes_to_use)]

        # Replace with indices
        row_idx = network['TF'].map(tf_index)
        col_idx = network['target'].map(gene_index)
        vals = network['importance'].values
        # vals = network['partialcor'].values

        # Efficient assignment
        matrices[i, row_idx, col_idx] = vals

    Xten = ttb.tensor(matrices)
    Lmat = generate_laplacian(T, 1.0)
    M, res, output = cp_als(Xten, Lmat, reg_lam=0.1, rank=round(2*len(TF_names)), maxiters=1000, seed=seed)
    with open(f'/cluster/scratch/jassan/planarian/cp{mode}{frac}.pkl', 'wb') as file:
        pickle.dump({'M': M, 'res': res, 'output': output}, file)
    Xhat = M.double()

    if not os.path.exists(os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp-{mode}{frac}')):
        os.mkdir(os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp-{mode}{frac}'))
    for i, l in enumerate(lengths):
        out_file = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp-{mode}{frac}/network_{l.round(3)}.csv')
        multi_index = pd.MultiIndex.from_product([TF_names, genes_to_use], names=['TF', 'target'])
        network = pd.DataFrame(index=multi_index)
        for tf in TF_names:
            for target in genes_to_use:
                network.at[(tf, target), 'importance'] = Xhat[i, tf_index[tf],gene_index[target]]
        network = network.reset_index()
        network = network.sort_values('importance', ascending=False)
        network.to_csv(out_file, index = False, sep = '\t')

def denoisePCA(i, mode='bin'):
    fracs = ['', '-frac80', '-frac60', '-frac40', '-frac20']
    frac = fracs[i]
    INPUT_PATH = "/cluster/scratch/jassan/planarian/"
    # Load the adata
    adata = sc.read_h5ad(os.path.join(INPUT_PATH,"planarians-timecourse.h5ad"))
    TF_file = pd.read_csv('TF-list2.csv')
    idx = np.where( TF_file['TF.group'] != '-')[0]
    TFs = TF_file['gene'][idx].tolist()
    adata_TF = adata[:, adata.var.index.isin(TFs)]
    TF_names = adata_TF.var.index.tolist()

    adata_subset = adata[ adata.obs['broad_cell_type'].isin(['Epidermal','Neoblast']), : ]
    sc.pp.highly_variable_genes( adata_subset, n_top_genes=500)
    adata_to_use = adata_subset[ : , adata_subset.var.index.isin(TFs) + adata_subset.var['highly_variable'] == True  ]
    genes_to_use = adata_to_use.var.index.tolist()
    lengths = np.sort(list(set(adata.obs["Lengthmm"].tolist())))

    # Precompute index mappings
    tf_index = {tf: i for i, tf in enumerate(TF_names)}
    gene_index = {g: j for j, g in enumerate(genes_to_use)}

    # Allocate array
    T = len(lengths)
    matrices = np.zeros((T, len(TF_names), len(genes_to_use)))

    for i in range(T):
        print(i)

        out_file = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-{mode}{frac}/network_{lengths[i].round(3)}.csv')
        network = pd.read_csv(out_file, sep='\t')

        # Filter for TFs and targets we care about
        network = network[network['TF'].isin(TF_names) & network['target'].isin(genes_to_use)]

        # Replace with indices
        row_idx = network['TF'].map(tf_index)
        col_idx = network['target'].map(gene_index)
        vals = network['importance'].values
        # vals = network['partialcor'].values

        # Efficient assignment
        matrices[i, row_idx, col_idx] = vals

    X = matrices.reshape(T, -1)
    print(X.mean(axis=1))
    Lmat = generate_laplacian(T, 1.0)
    M, res, output = cp_als(Xten, Lmat, reg_lam=0.1, rank=round(2*len(TF_names)), maxiters=1000, seed=seed)
    with open(f'/cluster/scratch/jassan/planarian/cp{mode}{frac}.pkl', 'wb') as file:
        pickle.dump({'M': M, 'res': res, 'output': output}, file)
    Xhat = M.double()

    if not os.path.exists(os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp-{mode}{frac}')):
        os.mkdir(os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp-{mode}{frac}'))
    for i, l in enumerate(lengths):
        out_file = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp-{mode}{frac}/network_{l.round(3)}.csv')
        multi_index = pd.MultiIndex.from_product([TF_names, genes_to_use], names=['TF', 'target'])
        network = pd.DataFrame(index=multi_index)
        for tf in TF_names:
            for target in genes_to_use:
                network.at[(tf, target), 'importance'] = Xhat[i, tf_index[tf],gene_index[target]]
        network = network.reset_index()
        network = network.sort_values('importance', ascending=False)
        network.to_csv(out_file, index = False, sep = '\t')


def from_pickle():
    with open('cp.pkl', 'rb') as file:
        d = pickle.load(file)
    Xhat = d['M'].double()
    INPUT_PATH = "/cluster/scratch/jassan/planarian/"
    # Load the adata
    adata = sc.read_h5ad(os.path.join(INPUT_PATH,"planarians-timecourse.h5ad"))
    TF_file = pd.read_csv('TF-list2.csv')
    idx = np.where( TF_file['TF.group'] != '-')[0]
    TFs = TF_file['gene'][idx].tolist()
    adata_TF = adata[:, adata.var.index.isin(TFs)]
    TF_names = adata_TF.var.index.tolist()

    adata_subset = adata[ adata.obs['broad_cell_type'].isin(['Epidermal','Neoblast']), : ]
    sc.pp.highly_variable_genes( adata_subset, n_top_genes=500)
    adata_to_use = adata_subset[ : , adata_subset.var.index.isin(TFs) + adata_subset.var['highly_variable'] == True  ]
    genes_to_use = adata_to_use.var.index.tolist()
    lengths = np.sort(list(set(adata.obs["Lengthmm"].tolist())))

    # Precompute index mappings
    tf_index = {tf: i for i, tf in enumerate(TF_names)}
    gene_index = {g: j for j, g in enumerate(genes_to_use)}

    # Allocate array
    T = len(lengths)
    if not os.path.exists(os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp')):
        os.mkdir(os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp'))
    for i, l in enumerate(lengths):
        if i < 30:
            continue
        print(i)
        out_file = os.path.join(INPUT_PATH, f'results/GRNs/neoblast-epidermal-cp/network_{l.round(3)}.csv')
        multi_index = pd.MultiIndex.from_product([TF_names, genes_to_use], names=['TF', 'target'])
        network = pd.DataFrame(index=multi_index)
        for tf in TF_names:
            for target in genes_to_use:
                network.at[(tf, target), 'importance'] = Xhat[i, tf_index[tf],gene_index[target]]
        network = network.reset_index()
        network = network.sort_values('importance', ascending=False)
        network.to_csv(out_file, index = False, sep = '\t')
# i = int(sys.argv[1])
# denoise(i)
jacc()