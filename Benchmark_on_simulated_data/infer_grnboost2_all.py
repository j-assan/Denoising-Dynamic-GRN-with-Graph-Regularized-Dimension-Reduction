# Script pour lancer tous les benchmarks de grnboost2
import sys; sys.path += ['./_scripts']
import time as timer
import numpy as np
import os
from arboreto.algo import grnboost2
import pandas as pd
from dask_jobqueue.slurm import SLURMCluster
from dask.distributed import Client, get_client
cluster = SLURMCluster(cores=1, memory='1GB',
                        job_extra_directives=[f'--mem-per-cpu=512MB', '--time=00:30:00', '--output=/cluster/scratch/jassan/cardamom/out/slurm_dask_%j.txt'],
                        job_directives_skip=['--mem'])
cluster.scale(24)
client = Client(cluster)
# seed = 23421299121297147538894723705775650942 #all cells
# seed = 261918155584570515984208129952250978199 #all1000 cells
# seed = 1657940068327669607231610943067 #all500 cells
seed = 293748801234679977005156722160476958500 # t20 100 cells
# seed = 83154407552656485910961776648750840960 # t20 500 cells
seeds = np.random.SeedSequence(seed).spawn(9)
# Number of runs
N = 10

def denoise(matrix):
    return np.average(matrix, axis=0)



def run_experiment(network_dir: str, seed=None):
    for r in range(N):
        path = '/cluster/scratch/jassan/cardamom' # '/cluster/home/jassan/projects/cardamom/results_article/Benchmark_on_simulated_data' 
        fname = os.path.join(path, network_dir, 'Data/data_t20_{}.txt'.format(r+1))
        data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
        time_labels = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]

        n_genes = data.shape[0]

        df = pd.DataFrame(data.T, columns=[str(i) for i in range(n_genes)]).sample(frac=0.2)
        rs = np.random.RandomState(seed.generate_state(5))
        result = grnboost2(df, client_or_address=client, seed=rs, verbose=True)
        score = np.zeros((n_genes, n_genes))
        score[[int(tf) for tf in result['TF']], [int(tar) for tar in result['target']]] = result['importance']

        os.makedirs(os.path.join('/cluster/scratch/jassan/cardamom/', network_dir, 'GRNBOOST2'), exist_ok=True)
        np.save(os.path.join('/cluster/scratch/jassan/cardamom', network_dir, 'GRNBOOST2/score_t20_{}'.format(r+1)), score)
        # score = denoise(matrix)
        
        # np.save(os.path.join('/cluster/scratch/jassan/cardamom', network_dir, 'avg/score_{}'.format(r+1)), score)


# Inference for Cycle
run_experiment('CN5', seed=seeds[0])

# Inference for Trifurcation
run_experiment('FN8', seed=seeds[1])

# Inference for Network4
run_experiment('FN4', seed=seeds[2])

# Inference for Bifurcation
run_experiment('BN8', seed=seeds[3])

# Inference for tree-like networks
for i, n in enumerate([5, 10, 20,50,100]):
    net_dir = f'Trees{n}'
    run_experiment(net_dir, seed=seeds[4+i])

client.close()
cluster.close