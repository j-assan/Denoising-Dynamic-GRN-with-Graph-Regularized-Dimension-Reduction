import sys; sys.path += ['./_scripts']
import numpy as np
from GraphPCA import GraphPCA, construct_filtering_matrix
from experiment import generate_laplacian
import os
import math
import json
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve
from sklearn.metrics import precision_recall_curve, auc
from sweetsweep import parameter_sweep_parallel, parameter_sweep, get_num_exp
from utils import get_auc, cut_array, ancestor
import pyttb as ttb
from cp_als import cp_als
import pickle

networks = ['CN5', 'BN8', 'FN4', 'FN8', 'Trees5', 'Trees10', 'Trees20', 'Trees50', 'Trees100']
ngenes = np.array([5, 8, 4, 8, 5, 10, 20, 50, 100])
ngenes += 1
nedges = ngenes ** 2
i = int(sys.argv[1])


def avg_and_max(path):
    N = 10
    mode = '_t20_bin500'
    data = '500_t20'
    if not os.path.exists(os.path.join(path, 'binAVG')):
        os.mkdir(os.path.join(path, 'binAVG'))
    if not os.path.exists(os.path.join(path, 'binMAX')):
        os.mkdir(os.path.join(path, 'binMAX'))
    for r in range(N):
        network = np.load(os.path.join(path, 'denoise', f'matrix{mode}_{r+1}.npy'))
        network_mean = np.mean(network, axis=0)
        network_max = np.max(network, axis=0)
        np.save(os.path.join(os.path.join(path, 'binAVG'),f'score{data}_{r+1}'), network_mean)
        np.save(os.path.join(os.path.join(path, 'binMAX'),f'score{data}_{r+1}'), network_max)

   

def mainPCA(args, exp_dir):
    N = 10
    T = 10
    auprcD_mean = []
    auprcD_max = []
    auprcU_mean = []
    auprcU_max = []
    aurocD_mean = []
    aurocD_max = []
    aurocU_mean = []
    aurocU_max = []
    mode = '_bin1000'
    for r in range(N):
        network = np.load(os.path.join(ancestor(exp_dir, 2), 'denoise', f'matrix{mode}_{r+1}.npy'))

        Lmat = generate_laplacian(T, args['gamma'])
        Kmat = construct_filtering_matrix(Lmat, args['reg'])
        d, p = network[0].shape
        Xmat = network.reshape(T, d*p)
        Xmat_mean = Xmat.mean(axis=0, keepdims=True)
        Xmat = Xmat - Xmat_mean
        result = GraphPCA(Xmat, Kmat, args['dim'], method='LinOp')
        Zmat = result['Zmat']         # Projected time series (T x proj_dim)
        Wmat = result['Wmat']         # Spatial EOF-like basis vectors (d*p x proj_dim)
        I_L = np.eye(T)+args['reg']*Lmat
        Xhat = solve(I_L, Xmat @ Wmat @ Wmat.T)
        Xhat = Xhat + Xmat_mean
        Xhat = Xhat.reshape(T, d, p)
        network_mean = np.mean(Xhat, axis=0)
        network_max = np.max(Xhat, axis=0)
        np.save(os.path.join(exp_dir,f'score_mean_{r+1}'), network_mean)
        np.save(os.path.join(exp_dir,f'score_{r+1}'), network_max)
        np.savetxt(os.path.join(exp_dir, f"Zmat_{r+1}.txt"), Zmat)
        np.savetxt(os.path.join(exp_dir, f"Wmatsum_{r+1}.txt"), Wmat.reshape(d, p, args['dim']).sum(axis=1))
        np.savetxt(os.path.join(exp_dir, f"EV_{r+1}.txt"), result["eigenvalues"])
        if mode == '_bin' or mode == '':
            network_name = os.path.basename(ancestor(exp_dir, 2))
            net_dir = f'/cluster/home/jassan/projects/cardamom/results_article/Benchmark_on_simulated_data/{network_name}'
        else:
            net_dir = ancestor(exp_dir, 2)
        auprcD_mean.append(get_auc(network_mean, r, net_dir, undirected=False,t='t20'))
        auprcD_max.append(get_auc(network_max, r, net_dir, undirected=False,t='t20'))
        auprcU_mean.append(get_auc(network_mean, r, net_dir, undirected=True,t='t20'))
        auprcU_max.append(get_auc(network_max, r, net_dir, undirected=True,t='t20'))
        aurocD_mean.append(get_auc(network_mean, r, net_dir, undirected=False, pr=False,t='t20'))
        aurocD_max.append(get_auc(network_max, r, net_dir, undirected=False, pr=False,t='t20'))
        aurocU_mean.append(get_auc(network_mean, r, net_dir, undirected=True, pr=False,t='t20'))
        aurocU_max.append(get_auc(network_max, r, net_dir, undirected=True, pr=False,t='t20'))
    return {'auprcD_mean': np.mean(auprcD_mean), 'auprcD_max': np.mean(auprcD_max), 'auprcU_mean': np.mean(auprcU_mean), 'auprcU_max': np.mean(auprcU_max), 
            'aurocD_mean': np.mean(aurocD_mean), 'aurocD_max': np.mean(aurocD_max), 'aurocU_mean': np.mean(aurocU_mean), 'aurocU_max': np.mean(aurocU_max)}

def mainCP(args, exp_dir, seed):
    N = 10
    T = 20
    auprcD_mean = []
    auprcD_max = []
    auprcU_mean = []
    auprcU_max = []
    aurocD_mean = []
    aurocD_max = []
    aurocU_mean = []
    aurocU_max = []
    for r in range(N):
        mode = '_t20_bin'
        network = np.load(os.path.join(ancestor(exp_dir, 2), 'denoise', f'matrix{mode}_{r+1}.npy'))

        Lmat = generate_laplacian(T, args['gamma'])
        M, _, output = cp_als(ttb.tensor(network), Lmat, args['reg'], rank=args['dim'], maxiters=1000, seed=seed)
        fit = output['fit']
        np.savetxt(os.path.join(exp_dir, f"A_{r+1}.txt"), M.factor_matrices[0])
        np.savetxt(os.path.join(exp_dir, f"B_{r+1}.txt"), M.factor_matrices[1])
        np.savetxt(os.path.join(exp_dir, f"C_{r+1}.txt"), M.factor_matrices[2])
        np.savetxt(os.path.join(exp_dir, f"weights_{r+1}.txt"), M.weights)
        Xhat = M.double()
        network_mean = np.mean(Xhat, axis=0)
        network_max = np.max(Xhat, axis=0)
        np.save(os.path.join(exp_dir,f'score_mean_{r+1}'), network_mean)
        np.save(os.path.join(exp_dir,f'score_{r+1}'), network_max)
        if mode == '_bin' or mode == '':
            network_name = os.path.basename(ancestor(exp_dir, 2))
            net_dir = f'/cluster/home/jassan/projects/cardamom/results_article/Benchmark_on_simulated_data/{network_name}'
        else:
            net_dir = ancestor(exp_dir, 2)
        auprcD_mean.append(get_auc(network_mean, r, net_dir, undirected=False,t='t20'))
        auprcD_max.append(get_auc(network_max, r, net_dir, undirected=False, t='t20'))
        auprcU_mean.append(get_auc(network_mean, r, net_dir, undirected=True, t='t20'))
        auprcU_max.append(get_auc(network_max, r, net_dir, undirected=True, t='t20'))
        aurocD_mean.append(get_auc(network_mean, r, net_dir, undirected=False, pr=False, t='t20'))
        aurocD_max.append(get_auc(network_max, r, net_dir, undirected=False, pr=False, t='t20'))
        aurocU_mean.append(get_auc(network_mean, r, net_dir, undirected=True, pr=False, t='t20'))
        aurocU_max.append(get_auc(network_max, r, net_dir, undirected=True, pr=False, t='t20'))
    return {'auprcD_mean': np.mean(auprcD_mean), 'auprcD_max': np.mean(auprcD_max), 'auprcU_mean': np.mean(auprcU_mean), 'auprcU_max': np.mean(auprcU_max), 
            'aurocD_mean': np.mean(aurocD_mean), 'aurocD_max': np.mean(aurocD_max), 'aurocU_mean': np.mean(aurocU_mean), 'aurocU_max': np.mean(aurocU_max),
            'fit': fit}

# if '--input' in sys.argv:
#     param_sweep, args = argparser(sys.argv[sys.argv.index('--input')+1])
# else:
#     param_sweep, args = argparser()

seed = None
seeds = None
if seed:
    num_exp = get_num_exp(param_sweep)
    seeds = np.random.default_rng(seed).spawn(num_exp)

## ------------------------------------
# Process argv parameters

# Don't print anything before this line
if '--get-num-exp' in sys.argv:
    print(get_num_exp(param_sweep))
    exit()

only_exp_id = None
direct_seed = None
if '--only-exp-id' in sys.argv:
    only_exp_id = int(sys.argv[sys.argv.index('--only-exp-id')+1])

exp_ids = None
if '--exp-ids' in sys.argv:
     exp_ids = [int(x) for x in (sys.argv[sys.argv.index('--exp-ids')+1]).split(',')]

## ------------------------------------


  # Default output dir


start_index = 0
if '--start-index' in sys.argv:
     start_index = int(sys.argv[sys.argv.index('--start-index')+1])


# Name of the image to save
# image_filename = 
# Name of the csv file to save (one row per experiment)
csv_filename = "results.csv"

# Save the param_sweep file

# Add parameters for the viewer if you need (see README.md)
# params["viewer_filePattern"] = image_filename
# params["viewer_cropLBRT"] = [0, 0, 0, 0]


def tuple_encoder(d):
    encoded = {}
    for k, v in d.items():
        dict_item = v
        if isinstance(v, list):
               dict_item = []
               for i in range(len(v)):
                    list_item = v[i]
                    if isinstance(v[i], tuple):
                         list_item = str(v[i])
                    dict_item.append(list_item)
        encoded[k] = dict_item
    return encoded


def experiment_wrapperPCA(exp_id, param_dict, exp_dir, seeds=seeds, start_index=start_index):
    print("Experiment #%d:"%exp_id, param_dict)
    seed = seeds[exp_id-start_index] if seeds else None
    return mainPCA(param_dict, exp_dir)


clean = True
def runPCA(i):
    network_dir = networks[i]
    reg = [0.1, 1, 10, 50, 100, 150, 200]
    dim = [2, 5, 7, 10, 15, 20, 24]
    dim = cut_array(dim, nedges[i])
    gamma = [0, 0.1, 1]
    param_sweep = {'reg': reg, 'dim': dim, 'gamma': gamma}
    params = param_sweep.copy()
    params["viewer_resultsCSV"] = csv_filename
    my_sweep_dir = f"/cluster/scratch/jassan/cardamom/{network_dir}/sweepbin1000PCA" 
    if not os.path.exists(my_sweep_dir):
        os.mkdir(my_sweep_dir)
    if clean:
        for item in os.listdir(my_sweep_dir):
            item_path = os.path.join(my_sweep_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove files
    json.dump(tuple_encoder(params), open(os.path.join(my_sweep_dir, "sweep.txt"), "w"))
    parameter_sweep(
            param_sweep,
            experiment_wrapperPCA,
            my_sweep_dir,
            start_index=start_index,
            result_csv_filename=csv_filename,
            only_exp_id=only_exp_id
        )

def runCP(i, start_index=start_index):
    # seed = 207286176469376554019802248553751085913 # for bincp
    # seed = 307687580091076395142013805660162838809 # for bincp2 
    # seed = 327928293948337889553771953469861646695 # bincp3
    # seed = 64400853149123397133765573403877076551 # for bin1000cp
    # seed = 200915378629404803173581205859412506946 # for binCP_ratio
    # seed = 73793927316691499349446774576375913536 # for bin500CPt20
    seed = 274962682044732398085549393870981455480 # for binCPt20
    seed_seq = np.random.SeedSequence(seed).spawn(9)[i]
    network_dir = networks[i]
    reg = [0.1, 1, 10, 50, 100, 150]
    dim = [round(x*ngenes[i]) for x in [1/3, 2/3, 1, 4/3, 5/3]]
    gamma = [0, 0.1, 1]
    param_sweep = {'reg': reg, 'dim': dim, 'gamma': gamma}
    seeds = seed_seq.spawn(get_num_exp(param_sweep))

    def experiment_wrapperCP(exp_id, param_dict, exp_dir, seeds=seeds, start_index=start_index):
        print("Experiment #%d:"%exp_id, param_dict)
        seed = seeds[exp_id-start_index] if seeds else None
        return mainCP(param_dict, exp_dir, seed)

    params = param_sweep.copy()
    params["viewer_resultsCSV"] = csv_filename
    my_sweep_dir = f"/cluster/scratch/jassan/cardamom/{network_dir}/sweepbinCPt20" 
    if not os.path.exists(my_sweep_dir):
        os.mkdir(my_sweep_dir)
    if clean:
        for item in os.listdir(my_sweep_dir):
            item_path = os.path.join(my_sweep_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove files
    json.dump(tuple_encoder(params), open(os.path.join(my_sweep_dir, "sweep.txt"), "w"))
    parameter_sweep(
            param_sweep,
            experiment_wrapperCP,
            my_sweep_dir,
            start_index=start_index,
            result_csv_filename=csv_filename,
            only_exp_id=only_exp_id
        )

# for network_dir in networks:
#    avg_and_max(f"/cluster/scratch/jassan/cardamom/{network_dir}")

runPCA(i)
