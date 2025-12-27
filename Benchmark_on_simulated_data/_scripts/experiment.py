#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pyttb as ttb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import solve
import datatensor
from pickle import dump

# Inital setting for plot size
rcParams["figure.figsize"] = (10,8)
# from GraphCP import CP_ALS
from cp_als import cp_als
from GraphPCA import GraphPCA, construct_filtering_matrix




#%%
def plot_time_components(A, proj_dim, reg_lam, exp_dir, max=10, reference=False):
    t = np.arange(A.shape[0])
    plt.figure()
    plt.rcParams.update({'font.size': 14})  # or any number you like

    for i in range(min(max, proj_dim)):
        plt.plot(A[:,i],label='Component-'+str(i))
    plt.legend()
    if reference:
        plt.savefig(os.path.join(exp_dir, f"reference.png"))
        plt.close()
    else:
        plt.savefig(os.path.join(exp_dir, f"time_components_k_{proj_dim}_lam_{reg_lam}.png"))
        plt.close()
    return 0

def matrix_plot(A, xticks, yticks, name, title, exp_dir):
    fig = plt.figure(figsize=(18,9))
    ax = plt.gca()
    vmin = np.nanmin(A[np.nonzero(A)])
    vmax = np.nanmax(A)
    cmap = plt.cm.viridis
    cmap.set_under('white')
    ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), z in np.ndenumerate(A):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    ax.set_ylabel("regularization_parameter")
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel("decomposition rank")
    ax.set_title(title)
    np.savetxt(os.path.join(exp_dir, f"{name}.txt"), A)
    fig.savefig(os.path.join(exp_dir, f"{name}.png"))
    plt.close()

def cp_experiment(Xten, init, Lapmat, reg_lam, proj_dim, normXstar, Xstar, exp_dir):
    M, _, output = cp_als(Xten, Lapmat, reg_lam, rank=proj_dim, maxiters=1000, init=init)
    fit = 1 - np.sqrt(normXstar**2 + M.norm()**2 - 2*M.innerprod(Xstar))/normXstar
    fit_noisy = output['fit']
    rel_error = 1 - fit_noisy
    np.savetxt(os.path.join(exp_dir, f"A_k_{proj_dim}_lam_{reg_lam}.txt"), M.factor_matrices[0])
    np.savetxt(os.path.join(exp_dir, f"B_k_{proj_dim}_lam_{reg_lam}.txt"), M.factor_matrices[1])
    np.savetxt(os.path.join(exp_dir, f"C_k_{proj_dim}_lam_{reg_lam}.txt"), M.factor_matrices[2])
    np.savetxt(os.path.join(exp_dir, f"weights_k_{proj_dim}_lam_{reg_lam}.txt"), M.weights)
    plot_time_components(M.factor_matrices[0], proj_dim, reg_lam, exp_dir)
    return fit, fit_noisy, rel_error, M
    

def graphpca_experiment(Xten, Lapmat, reg_lam, proj_dim, normXstar, Xstar, exp_dir):
    # Step 1: Flatten matrices -> Xmat (T x (d*p))
    T, d, p = Xten.shape
    Xmat = Xten.data.reshape(T, d * p)
    Xmat_mean = Xmat.mean(axis=0, keepdims=True)
    # Xmat = Xmat - Xmat_mean
    
    # Step 2: Construct filtering matrix K = (I + λL)^(-1)
    Kmat = construct_filtering_matrix(Lapmat, reg_lam)
    
    # Step 3: Run Graph-PCA
    result = GraphPCA(Xmat, Kmat, proj_dim, method='LinOp')
    
    # Step 4: Extract results
    Zmat = result['Zmat']         # Projected time series (T x proj_dim)
    Wmat = result['Wmat']         # Spatial EOF-like basis vectors (d*p x proj_dim)
    # eigenvalues = result['eigenvalues']
    
    # # Optional: reshape Wmat back to d x p x proj_dim if you want to interpret spatial modes
    # Wmat_reshaped = Wmat.reshape(d, p, proj_dim)

    I_L = np.eye(T)+reg_lam*Lapmat
    # XV = Xmat @ Wmat
    # UT = solve(I_L, XV)
    # # Xhat = solve(Wmat.T, result['Zmat'].T @ I_L.T).T
    # Xhat = UT @ Wmat.T
    Xhat = solve(I_L, Xmat @ Wmat @ Wmat.T)
    # Xhat += Xmat_mean
    normXten = Xten.norm()
    Xhat = ttb.tensor(Xhat.reshape(T, d, p))
    fit = 1 - np.sqrt(normXstar**2 + Xhat.norm()**2 - 2*Xstar.innerprod(Xhat))/normXstar
    rel_error = np.sqrt(normXten**2 + Xhat.norm()**2 - 2*Xten.innerprod(Xhat))/normXten
    fit_noisy = 1 - rel_error
    np.savetxt(os.path.join(exp_dir, f"Zmat_k_{proj_dim}_lam_{reg_lam}.txt"), Zmat)
    np.savetxt(os.path.join(exp_dir, f"Wmatsum_k_{proj_dim}_lam_{reg_lam}.txt"), Wmat.reshape(d, p, proj_dim).sum(axis=1))
    np.savetxt(os.path.join(exp_dir, f"EV_k_{proj_dim}_lam_{reg_lam}.txt"), result["eigenvalues"])
    plot_time_components(Zmat, proj_dim, reg_lam, exp_dir)
    return fit, fit_noisy, rel_error

def experiment(args, exp_dir, seed=None):
    rng = np.random.default_rng(seed)
    X0 = args['X0']
    one_X0 = args.pop('one_X0')
    if type(X0) in [np.ndarray, list]:
        X0 = np.array(X0)

    elif one_X0:
        X0 = datatensor.X0_gen(X0[0], X0[1], rng)
        args['G'] = datatensor.X0_gen(X0.shape[0], X0.shape[1], rng)

    args['X0'] = X0
    args['rng'] = rng
    reg_lam = args.pop('reg_lam')
    method =args.pop('method')
    minrank = args.pop('minrank')
    maxrank = args.pop('maxrank')
    k = args.pop('n_ranks')
    gamma = args.pop('gamma')
    Xten, Xstar, K = datatensor.generate_tensor(**args)
    with open(os.path.join(exp_dir, "input_data.pkl"), "wb") as file:
        dump({"Xten": Xten, "Xstar": Xstar, "K": K}, file)
    plot_time_components(K.factor_matrices[0], K.factor_matrices[0].shape[1], 0, exp_dir, max=100, reference=True)
    normXstar = Xstar.norm()
    Lapmat = generate_laplacian(args['T'], gamma)
    results = {}
    metrics = {"fit_noisy": r"$1-||X-\hat{X}||/||X||$", "fit": r"$1-||X^*-\hat{X}||/||X^*||$", "rel_error": r"$||X-\hat{X}||/||X||$"}

    if method == "pyttbGraphCP": inits = [0] * k
    fit = np.zeros((len(reg_lam), k))
    fit_noisy = np.zeros((len(reg_lam), k))
    rel_error = np.zeros((len(reg_lam), k))
    best_rank_ind = np.zeros(len(reg_lam), dtype=int)
    ranks = np.linspace(minrank, maxrank, k, dtype=int)
    best_reg_ind = np.zeros(len(ranks), dtype=int)
    for i, lam in enumerate(reg_lam):
        for j, proj_dim in enumerate(ranks):
            try:
                if method == "GraphPCA":
                    fit[i,j], fit_noisy[i,j], rel_error[i,j] = graphpca_experiment(Xten, Lapmat, lam, proj_dim, normXstar, Xstar, exp_dir)
                elif method == "pyttbGraphCP":
                    fit[i,j], fit_noisy[i,j], rel_error[i,j], M = cp_experiment(Xten, "random", Lapmat, lam, proj_dim, normXstar, Xstar, exp_dir)
                if i == len(reg_lam)-1:
                    best_reg_ind[j] = np.argmax(fit[:,j])
            except Exception as e:
                print(f"Skipped lam: {lam} proj_dim: {proj_dim} because of exception: {str(e)}")
                continue
        best_rank_ind[i] = np.argmax(fit[i,:])

        # best_rank_ind[i] = np.asarray(diff < threshold).nonzero()[0][0] if np.any(diff < threshold) else -1
    for k, v in metrics.items():
        matrix_plot(eval(k), ranks, reg_lam, k, v, exp_dir)
        for i, lam in enumerate(reg_lam):
            results[f'{k}_lam_{lam}'] = eval(k)[i, best_rank_ind[i]]
            results[f'bestrank_lam_{lam}'] = ranks[best_rank_ind[i]]
        for i, r in enumerate(ranks):
            results[f'{k}_rank_{r}'] = eval(k)[best_reg_ind[i], i]
            results[f'bestreg_rank_{r}'] = reg_lam[best_reg_ind[i]]
        results[f'best_{k}'] = np.nanmax(eval(k))
    

    return  results

#%%

# m, n, p = (100, 2, 2)
def generate_laplacian(T, gamma=0):
    K = np.zeros((T, T))
    for t in range(T):
        for s in range(max(0, t-2), min(T, t+3)):
            if s != t:
                K[t, s] = np.exp(-gamma*(t-s)**2)
    D = np.diag(K.sum(axis=1))
    Lapmat = D - K
    return Lapmat




#%%

    
# %%
