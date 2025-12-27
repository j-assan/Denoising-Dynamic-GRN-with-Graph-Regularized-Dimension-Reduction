import os

import pandas as pd
import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Inital setting for plot size
rcParams["figure.figsize"] = (10,8)
import arboreto
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from joblib import Parallel, delayed
from dask_jobqueue.slurm import SLURMCluster
from dask.distributed import Client
cluster = SLURMCluster(cores=1, memory='1GB',
                        job_extra_directives=[f'--mem-per-cpu=512MB', '--time=03:00:00', '--output=./dask_%j.out'],
                        job_directives_skip=['--mem'])
cluster.scale(25)

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


def worker(task_id,i, frac=None, bin_len=False, seed=None):
    """
    task_id: id of job if run in parallel
    i:  i-th worm
    frac: fraction of cells that should be used
    bin_len: 
            False if only cells from worm i should be used
            True if cells of other worms close in length should be used
    """
    
    client = Client(cluster)
    print(f"Task {task_id} started")
    if bin_len:
        delta = 2*np.mean(np.diff(lengths))
        idx = np.where( abs(lengths - lengths[i])<delta )[0]
        ex_matrix = adata_to_use[ adata_to_use.obs['Lengthmm'].isin(lengths[idx]) ,: ].X.todense()
    else:
        ex_matrix = adata_to_use[ adata_to_use.obs['Lengthmm'].isin([lengths[i]]) ,: ].X.todense()
    print(ex_matrix.shape)
    df = pd.DataFrame(ex_matrix, columns=genes_to_use)
    if frac:
        df = df.sample(frac=frac, random_state=seed, axis=0)
        print(df.shape)
    # local_cluster = LocalCluster()
    # custom_client = Client(local_cluster)  
    adj_matrix = grnboost2(df, tf_names=TF_names, seed=seed, client_or_address=client, verbose=False)
    out_file = './network_'+str(lengths[i].round(3))+'.csv'
    print(out_file)
    adj_matrix.to_csv(out_file, sep='\t')#, index=False, header=False)
    client.close()
    print(f"Task {task_id} finished")

seed = 235027571498096000902223140007522080777
n = len(lengths)
seeds = np.random.SeedSequence(seed).spawn(n)

for i in np.arange(0,n):
    # ex_matrix = adata[ adata.obs['Lengthmm'] == lengths[i] , adata.var['gene_ddv6'].isin(genes_to_use) ]
    # ex_matrix = ex_matrix[ex_matrix.obs['broad_cell_type'].isin(['Epidermal','Neoblast']) ,:].X.todense()
    print(i)
    worker(0,i, seed=np.random.RandomState(seeds[i].generate_state(5)), bin_len=True, frac=0.2)

cluster.close()

# Parallel(n_jobs=4)(delayed(worker)(i, i+2) for i in range(8))