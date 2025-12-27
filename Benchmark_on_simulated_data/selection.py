#%% 
import sys; sys.path += ['./_scripts']
import numpy as np
import pandas as pd
import os
from utils import get_auc, cut_array, ancestor
import matplotlib.pyplot as plt
import shutil

networks = ['CN5', 'BN8', 'FN4', 'FN8', 'Trees5', 'Trees10', 'Trees20', 'Trees50', 'Trees100']
ngenes = np.array([5, 8, 4, 8, 5, 10, 20, 50, 100]) + 1
def selection(mode = 'bin500PCA', alg='binPCA', exclude=[]):
    random = [0.3125,0.24,0.203125,0.234375,0.2,0.1,0.05,0.02,0.01]
    dfs = []
    smalldfs = []
    nets = networks.copy()
    for i in exclude:
        nets.pop(i)
    for i, network in enumerate(networks):
        if i in exclude:
            continue
        path = f'/cluster/scratch/jassan/cardamom/{network}/sweep{mode}'
        df = pd.read_csv(os.path.join(path, 'results.csv')).drop(['src_exp_id'], axis=1)
        if alg == 'CP' or alg == 'binCP':
            dim = [round(x*ngenes[i]) for x in [1/3, 2/3, 1, 4/3, 5/3]]
            mapping = dict(zip(dim, [1/3, 2/3, 1, 4/3, 5/3]))
            df['r'] = df['dim'].map(mapping)
            df = df.set_index(['reg','r', 'gamma'])
        else:
            df = df.set_index(['reg','dim', 'gamma'])
        df['auprcU_max'] = df['auprcU_max'] / random[i]
        df['auprcU_mean'] = df['auprcU_mean'] / random[i]
        df['auprcD_max'] = df['auprcD_max'] / random[i]
        df['auprcD_mean'] = df['auprcD_mean'] / random[i]
        smalldfs.append(df['auprcD_max'])
        # dfs.append(df.loc[df.index.duplicated(keep='last')])
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=1, keys=nets)
    combined_smalldf = pd.concat(smalldfs, axis=1, keys=nets)
    print(combined_smalldf.max(), combined_smalldf.idxmax())
    average_df = combined_smalldf.T.mean().T
    # print(average_df.max(), average_df.idxmax())
    # print(average_df.loc[average_df.idxmax()])
    # print(combined_df.loc[average_df.idxmax(), pd.IndexSlice[:, ['exp_id', 'auprcD_max']]])
    # print(dfs[0]['exp_id'].loc[average_df.idxmax()], average_df.idxmax())
    print(dfs[0]['exp_id'].loc[average_df.idxmax()], average_df.max(),  average_df.idxmax())
    return dfs[0]['exp_id'].loc[average_df.idxmax()], average_df.max(),  average_df.idxmax()


def change_auc(path):
    for exp_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, exp_dir)):
            continue
        auprc_max = []
        auprc_mean = []
        for r in range(10):
            network_mean = np.load(os.path.join(path, exp_dir,f'score_mean_{r+1}.npy'))
            network_max = np.load(os.path.join(path, exp_dir,f'score_{r+1}.npy'))
            auprc_mean.append(get_auc(network_mean, r, ancestor(os.path.join(path, exp_dir), 2)))
            auprc_max.append(get_auc(network_max, r, ancestor(os.path.join(path, exp_dir), 2)))
        print({'auprc_mean': np.mean(auprc_mean), 'auprc_max': np.mean(auprc_max)})

def get_dirname(mode, i, alg='CP'):
    dfs = []
    network = networks[i]
    path = f'/cluster/scratch/jassan/cardamom/{network}/sweep{mode}'
    
    if alg == 'CP' or alg == 'binCP':
        df = pd.read_csv(os.path.join(path, 'results.csv')).drop(['src_exp_id'], axis=1)
        dim = [round(x*ngenes[i]) for x in [1/3, 2/3, 1, 4/3, 5/3]]
        mapping = dict(zip(dim, [1/3, 2/3, 1, 4/3, 5/3]))
        rev_map = dict(zip([1/3, 2/3, 1, 4/3, 5/3], dim))
        df['r'] = df['dim'].map(mapping)
        df = df.set_index(['reg','r', 'gamma'])
        sel = selection(mode, alg = alg, exclude=[i])
        dir_name = os.path.join(path, "exp_{:02d}__reg{:g}_dim{:g}_gamma{:g}".format(sel[0], sel[2][0], df['dim'].loc[sel[2]], sel[2][2]))
    else:
        sel = selection(mode, alg = alg, exclude=[i])
        dir_name = os.path.join(path, "exp_{:03d}__reg{:g}_dim{:g}_gamma{:g}".format(sel[0], sel[2][0], sel[2][1], sel[2][2]))
    return dir_name

def best_dirname(mode, i, alg='CP'):
    random = [0.3125,0.24,0.203125,0.234375,0.2,0.1,0.05,0.02,0.01]
    dfs = []
    network = networks[i]
    path = f'/cluster/scratch/jassan/cardamom/{network}/sweep{mode}'
    
    if alg == 'CP' or alg == 'binCP':
        df = pd.read_csv(os.path.join(path, 'results.csv')).drop(['src_exp_id'], axis=1)
        df = df.set_index(['exp_id', 'reg','dim', 'gamma'])
        exp_id, reg, dim, gamma = df['auprcD_max'].idxmax()
        print(df['auprcD_max'].loc[exp_id, reg, dim, gamma] / random[i])
        dir_name = os.path.join(path, "exp_{:02d}__reg{:g}_dim{:g}_gamma{:g}".format(exp_id, reg, dim, gamma))
    else:
        sel = selection(mode, alg = alg, exclude=[i])
        dir_name = os.path.join(path, "exp_{:03d}__reg{:g}_dim{:g}_gamma{:g}".format(sel[0], sel[2][0], sel[2][1], sel[2][2]))
    return dir_name

def analyzeCP(i):
    mode = 'bin500CPt20'
    dir_name = best_dirname(mode, i)
    
    weights = np.loadtxt(os.path.join(dir_name, 'weights_1.txt'))
    A = np.loadtxt(os.path.join(dir_name, 'A_1.txt'))
    B = np.loadtxt(os.path.join(dir_name, 'B_1.txt'))
    C = np.loadtxt(os.path.join(dir_name, 'C_1.txt'))
    fig, ax = plt.subplots(10, 3, figsize=(6,7))
    for j in range(min(10, A.shape[1])):
        ax[j,0].plot(A[:,j]*weights[j])
        ax[j,1].bar(range(len(B[:,j])), B[:,j])
        ax[j,2].bar(range(len(C[:,j])), C[:,j])
    for a in ax.flatten():
        a.tick_params(axis='both', which='major', labelsize=10)
        a.tick_params(axis='both', which='minor', labelsize=10)
    fig.tight_layout(pad=0.1)
    fig.savefig(f'plots/plot{i}.png')

def move():
    name = 'bin500t20PCA'
    alg = 'binPCA'
    data = '500_t20'
    for i in range(9):
        dir_name = get_dirname(name, i, alg=alg)
        target_dir = os.path.join(ancestor(dir_name, 2), alg)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for i in range(1,11):
            shutil.copy(os.path.join(dir_name, f'score_{i}.npy'), os.path.join(target_dir, f'score{data}_{i}.npy'))

# change_auc('/cluster/scratch/jassan/cardamom/Trees5/sweepbinCP')
# for i in range(8):
#     analyzeCP(i)
move()
# selection(mode='bint20PCA', alg='binPCA')
# analyzeCP(0)
# %%
