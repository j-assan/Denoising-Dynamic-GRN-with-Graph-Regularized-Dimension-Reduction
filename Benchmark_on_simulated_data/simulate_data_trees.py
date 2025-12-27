# Generate data for tree-like networks
import sys; sys.path += ['../']
import numpy as np
from harissa import Tree
import os

def run(r, seed):
    np.random.seed(seed)

    # Number of cells
    C = 10000

    # Time points
    
    t = [0, 3, 6, 9, 12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96] # np.linspace(0, 25, 10, dtype='int')
    print(f't = {t}')
    k = np.linspace(0, C, len(t)+1, dtype='int')
    time = np.zeros(C, dtype='int')
    for i in range(len(t)): time[k[i]:k[i+1]] = t[i]

    # Number of genes
    for G in [5, 10, 20, 50, 100]:
        print(f'Tree-like networks with {G} genes:')
        # Prepare data
        data = np.zeros((C+1,G+2), dtype='int')
        data[0][1:] = np.arange(G+1)
        data[1:,0] = time # Time points
        data[1:,1] = 1 * (time > 0) # Stimulus


        print(f'Run {r+1}...')
        
        # Initialize the model
        model = Tree(G)
        model.d[0] = 1
        model.d[1] = 0.2
        model.d /= 4

        # Save true network topology
        inter = 1 * (abs(model.inter) > 0)
        path = '/cluster/scratch/jassan/cardamom/'
        fname = path + f'Trees{G}/True/inter_t20_{r+1}'
        if not os.path.exists(os.path.join(path, f'Trees{G}/True')):
            os.makedirs(os.path.join(path, f'Trees{G}/True'))
        np.save(fname, inter)

        # Generate data
        for k in range(C):
            # print(f'* Cell {k+1}')
            sim = model.simulate(time[k], burnin=5)
            data[k+1,2:] = np.random.poisson(sim.m[0])

        # Save data for use with PIDC
        if not os.path.exists(os.path.join(path, f'Trees{G}/Data')):
            os.mkdir(os.path.join(path, f'Trees{G}/Data'))
        fname = path + f'Trees{G}/Data/data_t20_{r+1}.txt'
        np.savetxt(fname, data.T, fmt='%d', delimiter='\t')

N = int(sys.argv[1])
r = int(sys.argv[2])
# seed_seq = np.random.SeedSequence(306787110102538151287748727388389858240) # 10000
seed_seq = np.random.SeedSequence(181902957406028987077335988959406083412) # 10000 20 tp
seeds = seed_seq.spawn(N)
seed = seeds[r].generate_state(5)
run(r, seed)

