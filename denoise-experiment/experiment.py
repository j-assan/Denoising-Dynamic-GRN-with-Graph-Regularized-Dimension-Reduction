#!/usr/bin/env python
# coding: utf-8


import sys; sys.path += ['../src']
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
from GraphCP import cp_als
from GraphPCA import GraphPCA, construct_filtering_matrix




#%%
def plot_time_components(A, proj_dim, reg_lam, exp_dir, max=10, reference=False):
    """
    Helper function that saves a plot of the time components:
         A[:,i] for the CP decomposition X = [|A, B, C|] or Z[:,i] with Z = K @ X @ W for PCA 
    """

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

def matrix_plot_old(A, xticks, yticks, name, title, exp_dir):
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    vmin = np.nanmin(A[np.nonzero(A)])
    vmax = np.nanmax(A)
    cmap = plt.cm.viridis
    cmap.set_under('white')
    ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), z in np.ndenumerate(A):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', fontsize=10)
    ax.set_ylabel("regularization_parameter")
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel("decomposition rank")
    ax.set_title(title)
    fig.savefig(os.path.join(exp_dir, f"old_{name}.png"))
    plt.close()

def matrix_plot(A, xticks, yticks, name, title, exp_dir):
    """
    Save a color plot of the fit 1 - norm(Xstar - Xhat)/norm(Xstar) as a function of the decomposition rank
    and regularization parameter as in Figure 4.2
    """

    fig = plt.figure(figsize=(6,2.5))
    ax = plt.gca()
    vmin = np.nanmin(A[np.nonzero(A)])
    vmax = np.nanmax(A)
    cmap = plt.cm.viridis
    cmap.set_under('white')
    im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
    # for (i, j), z in np.ndenumerate(A):
    #     ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', fontsize=11)
    ax.set_ylabel("regularization parameter", fontsize=11)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel("decomposition rank", fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_title(title, fontsize=11)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    np.savetxt(os.path.join(exp_dir, f"{name}.txt"), A)
    fig.savefig(os.path.join(exp_dir, f"{name}.png"))
    plt.close()

def cp_experiment(Xten, init, Lapmat, reg_lam, proj_dim, normXstar, Xstar, exp_dir, seed=None):
    """
    Wrapper to call the cp_als function to calculate graph regularized cp decomposition
    """

    M, _, output = cp_als(Xten, Lapmat, reg_lam, rank=proj_dim, maxiters=1000, init=init, seed=seed)
    fit = 1 - np.sqrt(np.round(normXstar**2 + M.norm()**2 - 2*M.innerprod(Xstar), 10))/normXstar
    fit_noisy = output['fit']
    rel_error = 1 - fit_noisy
    with open(os.path.join(exp_dir, f'Kten_k_{proj_dim}_lam_{reg_lam}.pkl'), 'wb') as file:
        dump(M, file)
    # plot_time_components(M.factor_matrices[0], proj_dim, reg_lam, exp_dir)
    return fit, fit_noisy, rel_error, M
    

def graphpca_experiment(Xten, Lapmat, reg_lam, proj_dim, normXstar, Xstar, exp_dir):
    """
    Wrapper to call the GraphPCA function to calculate graph regularized PCA
    """

    # Step 1: Flatten matrices -> Xmat (T x (d*p))
    T, d, p = Xten.shape
    Xmat = Xten.data.reshape(T, d * p)
    Xmat_mean = Xmat.mean(axis=0, keepdims=True)
    Xmat = Xmat - Xmat_mean
    
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
    Xhat += Xmat_mean
    normXten = Xten.norm()
    Xhat = ttb.tensor(Xhat.reshape(T, d, p))
    fit = 1 - np.sqrt(np.round(normXstar**2 + Xhat.norm()**2 - 2*Xstar.innerprod(Xhat), 10))/normXstar
    rel_error = np.sqrt(np.round(normXten**2 + Xhat.norm()**2 - 2*Xten.innerprod(Xhat), 10))/normXten
    fit_noisy = 1 - rel_error
    with open(os.path.join(exp_dir, f'pca_k_{proj_dim}_lam_{reg_lam}.pkl'), 'wb') as file:
        dump({'Zmat': Zmat, 'Wmat': Wmat.reshape(d,p, proj_dim).sum(axis=1), 'EV': result["eigenvalues"]}, file)
    # plot_time_components(Zmat, proj_dim, reg_lam, exp_dir)
    return fit, fit_noisy, rel_error

def experiment(args, exp_dir, seed=None):
    """
    Main function to run the 'denoise experiment'
    """

    rngs = np.random.default_rng(seed).spawn(5)
    reg_lam = args.pop('reg_lam')
    method =args.pop('method')
    minrank = args.pop('minrank')
    maxrank = args.pop('maxrank')
    k = args.pop('n_ranks')
    gamma = args.pop('gamma')
    one_X0 = args.pop('one_X0')

    all_fit = []
    all_fit_noisy = []
    all_rel_error = []
    all_baselines = []
    for run_i in range(5):
        rng = np.random.default_rng(rngs[run_i])
        X0 = args['X0']
        if type(X0) in [np.ndarray, list]:
            X0 = np.array(X0)

        elif one_X0:
            X0 = datatensor.X0_gen(X0[0], X0[1], rng)
            args['G'] = datatensor.X0_gen(X0.shape[0], X0.shape[1], rng)

        args['X0'] = X0
        args['rng'] = rng
        Xten, Xstar, K = datatensor.generate_tensor(**args)
        with open(os.path.join(exp_dir, "input_data.pkl"), "wb") as file:
            dump({"Xten": Xten, "Xstar": Xstar, "K": K}, file)
        plot_time_components(K.factor_matrices[0], K.factor_matrices[0].shape[1], 0, exp_dir, max=100, reference=True)
        normXstar = Xstar.norm()
        baseline = 1 - (Xten-Xstar).norm() / normXstar
        Lapmat = generate_laplacian(args['T'], gamma)
        results = {}
        metrics = {"fit_noisy": r"$1-||X-\hat{X}||/||X||$", "fit": r"$1-||X^*-\hat{X}||/||X^*||$", "rel_error": r"$||X-\hat{X}||/||X||$"}

        if method == "GraphCP": inits = [0] * k
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
                    elif method == "GraphCP":
                        fit[i,j], fit_noisy[i,j], rel_error[i,j], M = cp_experiment(Xten, "random", Lapmat, lam, proj_dim, normXstar, Xstar, exp_dir, rng)
                    if i == len(reg_lam)-1:
                        best_reg_ind[j] = np.argmax(fit[:,j])
                except Exception as e:
                    print(f"Skipped lam: {lam} proj_dim: {proj_dim} because of exception: {str(e)}")
                    continue
            best_rank_ind[i] = np.argmax(fit[i,:])
        all_fit.append(fit)
        all_fit_noisy.append(fit_noisy)
        all_rel_error.append(rel_error)
        all_baselines.append(baseline)
    np.save(os.path.join(exp_dir, 'all_fit.npy'), all_fit)
    all_fit = np.sum(all_fit, axis=0) / 5
    np.savetxt(os.path.join(exp_dir, 'avg_fit.txt'), all_fit)
    all_fit_noisy = np.sum(all_fit_noisy, axis=0) / 5
    all_rel_error = np.sum(all_rel_error, axis=0) / 5
    all_best_reg_ind = np.argmax(all_fit, axis=0)
    all_best_rank_ind = np.argmax(all_fit, axis=1)
    all_baselines = np.sum(all_baselines) / 5
    results['baseline'] = baseline
    results['avg_baseline'] = all_baselines
    results['avg_improvement'] = np.nanmax(all_fit) / all_baselines
    results['improvement'] = np.nanmax(fit) / baseline



            # best_rank_ind[i] = np.asarray(diff < threshold).nonzero()[0][0] if np.any(diff < threshold) else -1
    matrix_plot(fit, ranks, reg_lam, 'fit', r"$1-||X^*-\hat{X}||/||X^*||$", exp_dir)
    for k, v in metrics.items():
        matrix_plot_old(eval(k), ranks, reg_lam, k, v, exp_dir)
        for i, lam in enumerate(reg_lam):
            results[f'{k}_lam_{lam}'] = eval(k)[i, best_rank_ind[i]]
            results[f'bestrank_lam_{lam}'] = ranks[best_rank_ind[i]]
            results[f'avg_{k}_lam_{lam}'] = eval('all_'+k)[i, all_best_rank_ind[i]]
            results[f'avg_bestrank_lam_{lam}'] = ranks[all_best_rank_ind[i]]
        for i, r in enumerate(ranks):
            results[f'avg_{k}_rank_{r}'] = eval('all_'+k)[all_best_reg_ind[i], i]
            results[f'avg_bestreg_rank_{r}'] = reg_lam[all_best_reg_ind[i]]
        results[f'best_{k}'] = np.nanmax(eval(k))
        results[f'avg_best_{k}'] = np.nanmax(eval('all_'+k))
        
    

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
