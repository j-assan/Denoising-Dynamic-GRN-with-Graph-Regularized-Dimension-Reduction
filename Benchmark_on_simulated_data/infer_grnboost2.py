# Script pour lancer tous les benchmarks de grnboost2
import sys; sys.path += ['./_scripts']
import time as timer
import numpy as np
import os
from arboreto.algo import grnboost2
import pandas as pd
from dask_jobqueue.slurm import SLURMCluster
from dask.distributed import Client, get_client
# cluster = SLURMCluster(cores=1, memory='1GB',
#                         job_extra_directives=[f'--mem-per-cpu=450MB', '--time=00:30:00', '--output=/cluster/scratch/jassan/cardamom/out/slurm_dask_%j.txt'],
#                         job_directives_skip=['--mem'])
# cluster.scale(16)
cluster = SLURMCluster(
    cores=1,
    memory='1GB',
    n_workers=24,  # Request 16 workers initially
    job_extra_directives=[f'--mem-per-cpu=450MB', '--time=00:30:00', '--output=/cluster/scratch/jassan/cardamom/out/slurm_dask_%j.txt'],
    job_directives_skip=['--mem']
)

client = Client(cluster)
# seed = 282687200569470771573722947675378760940 # for max
# seed = 252116985052517366521215423717352012263 # for avg
# seed = 117612549922011834707575521903248636130 # for denoise
# seed = 130618266573576803824571963652002962995 # for denoise1000
# seed = 114571913703133810183335840278885049174 # for denoise500
# seed = 164581622617734100467744384755360816401 # for denoise_bin500
# seed = 281770256205476037251097559086803755564 # for denoise_bin
# seed = 29407188230625394854980811797713624466 # for denoise_bin1000
# seed = 232031585474786604728396572878153641299 # for denoise t20 bin100
seed = 99461213116388222668100160972571492492 # for denoise t20 bin500
seeds = np.random.SeedSequence(seed).spawn(9)
# Number of runs
N = 10

def denoise(matrix):
    return np.average(matrix, axis=0)

def grnboost2_wrapper(data, seed_seq):
    client = get_client()
    rs = np.random.RandomState(seed_seq.generate_state(5))
    result = grnboost2(data, client_or_address=client, seed=rs, verbose=True)
    return result


def run_experiment(network_dir: str, seed=None):
    for r in range(N):
        path = '/cluster/scratch/jassan/cardamom'
        # path = '/cluster/home/jassan/projects/cardamom/results_article/Benchmark_on_simulated_data'
        fname = os.path.join(path, network_dir, 'Data/data_t20_{}.txt'.format(r+1))
        data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
        time_labels = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]

        n_genes = data.shape[0]
        multiindex = pd.MultiIndex.from_arrays([time_labels], names=['time'])
        times = np.unique(time_labels)
        dfs = []
        for k in range(1, len(times)-1):
            dfs.append(pd.DataFrame(data.T[[t in times[k-1:k+2] for t in time_labels]], columns=[str(i) for i in range(n_genes)]).sample(frac=0.2))
        matrix = np.zeros((len(times), n_genes, n_genes))
        
        L = client.map(grnboost2_wrapper, dfs, seed.spawn(len(times)))

        for i, result in enumerate(client.gather(L)):
            matrix[i, [int(tf) for tf in result['TF']], [int(tar) for tar in result['target']]] = result['importance']

        # os.makedirs(os.path.join('/cluster/scratch/jassan/cardamom/', network_dir, 'denoise'), exist_ok=True)
        np.save(os.path.join('/cluster/scratch/jassan/cardamom', network_dir, 'denoise/matrix_t20_bin_{}'.format(r+1)), matrix)
        # score = denoise(matrix)
        
        # np.save(os.path.join('/cluster/scratch/jassan/cardamom', network_dir, 'avg/score_{}'.format(r+1)), score)


# # Inference for Cycle
# run_experiment('CN5', seed=seeds[0])

# # Inference for Trifurcation
# run_experiment('FN8', seed=seeds[1])

# Inference for Network4
run_experiment('FN4', seed=seeds[2])

# Inference for Bifurcation
# run_experiment('BN8', seed=seeds[3])

# Inference for tree-like networks
for i, n in enumerate([5, 10, 20,50,100]):
    net_dir = f'Trees{n}'
    run_experiment(net_dir, seed=seeds[4+i])

client.close()
cluster.close