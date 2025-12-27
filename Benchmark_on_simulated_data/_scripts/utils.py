# Some utility functions for bencharking Harissa
import numpy as np
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def roc(score, inter):
    """
    Compute a receiver operating characteristic (ROC) curve.
    Here score and inter are arrays of shape (G,G) where:
    * score[i,j] is the estimated score of interaction i -> j
    * inter[i,j] = 1 if i -> j is present and 0 otherwise.
    """
    G = inter.shape[0]
    s, v = [], []
    for i in range(G):
        for j in range(1,G):
            if i != j:
                s.append(score[i,j])
                v.append(inter[i,j])
    s = abs(np.array(s))
    v = abs(np.array(v))
    x, y, t = roc_curve(v,s)
    return x, y

def auroc(score, inter):
    """
    Area under ROC curve (see function `roc`).
    """
    x, y = roc(score, inter)
    return auc(x,y)

def plot_roc(score, inter, file=None):
    """
    Plot a ROC curve (see function `roc`).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs
    fig = plt.figure(figsize=(5,5), dpi=100)
    grid = gs.GridSpec(1,1)
    ax = fig.add_subplot(grid[0,0])
    x, y = roc(score, inter)
    ax.plot([0,1], [0,1], color='gray', ls='--')
    ax.plot(x, y, color='dodgerblue')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    if file is None: file = 'roc.pdf'
    fig.savefig(file, dpi=100, bbox_inches='tight', frameon=False)

def pr(score, inter):
    """
    Compute a precision recall (PR) curve.
    Here score and inter are arrays of shape (G,G) where:
    * score[i,j] is the estimated score of interaction i -> j
    * inter[i,j] = 1 if i -> j is present and 0 otherwise.
    """
    G = inter.shape[0]
    s, v = [], []
    for i in range(G):
        for j in range(1,G):
            if i != j:
                s.append(score[i,j])
                v.append(inter[i,j])
    s = abs(np.array(s))
    v = abs(np.array(v))
    y, x, t = precision_recall_curve(v,s)
    return x, y

def aupr(score, inter):
    """
    Area under PR curve (see function `pr`).
    """
    x, y = pr(score, inter)
    return auc(x,y)

def plot_pr(score, inter, file=None):
    """
    Plot a PR curve (see function `pr`).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs
    fig = plt.figure(figsize=(5,5), dpi=100)
    grid = gs.GridSpec(1,1)
    ax = fig.add_subplot(grid[0,0])
    x, y = pr(score, inter)
    G, G = inter.shape
    v = []
    for i in range(G):
        for j in range(1,G):
            if i != j: v.append(inter[i,j])
    v = abs(np.array(v))
    b = np.mean(v)
    ax.plot([0,1], [b,b], color='gray', ls='--')
    ax.plot(x, y, color='dodgerblue')
    ax.set_xlim(0,1)
    ax.set_ylim(0)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if file is None: file = 'pr.pdf'
    fig.savefig(file, dpi=100, bbox_inches='tight', frameon=False)

def ancestor(path, level):
    for i in range(level):
        path = os.path.dirname(path)
    return path

def cut_array(arr, x):
    # Find the index where elements exceed x
    for i in range(len(arr)):
        if arr[i] >= x:
            return arr[:i]  # Return the sliced array up to that index
    return arr  # Return the whole array if all elements are <= x

def get_auc(score, r, net_dir, undirected=False, pr=True, t=''):
    inter = abs(np.load(os.path.join(net_dir, f'True/inter_{t}_{r+1}.npy')))
    G = inter.shape[0]
    if undirected:
        edges = [(i,j) for i in range(G) for j in range(i+1,G)]
        y0 = np.array([max(inter[i,j],inter[j,i]) for (i,j) in edges])
        y1 = np.array([max(score[i,j],score[j,i]) for (i,j) in edges])
    else:
        edges = [(i,j) for i in range(G) for j in set(range(1,G))-{i}]
        y0 = np.array([inter[i,j] for (i,j) in edges])
        y1 = np.array([score[i,j] for (i,j) in edges])
        
    if pr:
        precision, recall, thresholds = precision_recall_curve(y0, y1)
        result = auc(recall, precision)
    else:
        x, y, t = roc_curve(y0, y1)
        result = auc(x,y)
    return result
    
# Tests
if __name__ == '__main__':
    score = np.array([[0,0.7,0.4],[0,0,0.8],[0,0,0]])
    inter = np.array([[0,1,0],[0,0,1],[0,0,0]])
    print(score)
    print(inter)
    # ROC curve
    x, y = roc(score, inter)
    print(auroc(score, inter))
    plot_roc(score, inter)
    # PR curve
    x, y = pr(score, inter)
    print(aupr(score, inter))
    plot_pr(score, inter)