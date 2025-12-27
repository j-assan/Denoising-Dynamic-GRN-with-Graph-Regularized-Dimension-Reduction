# Generate data for the 4-gene network of figure 3
import sys; sys.path += ['../']
import numpy as np
from harissa import NetworkModel


def run(r, seed):
    np.random.seed(seed)
    
    # Number of cells
    C = 10000

    # Time points
    t = [0, 3, 6, 9, 12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96] # np.linspace(0, 25, 10, dtype='int')
    k = np.linspace(0, C, len(t) + 1, dtype='int')
    print(f't = {t}')
    time = np.zeros(C, dtype='int')
    for i in range(len(t)): time[k[i]:k[i+1]] = t[i]

    # Number of genes
    G = 8

    # Prepare data
    data = np.zeros((C+1,G+2), dtype='int')
    data[0][1:] = np.arange(G+1)
    data[1:,0] = time # Time points
    data[1:,1] = 100 * (time > 0) # Stimulus

    print(f'Run {r+1}...')
    
    # Initialize the model
    model = NetworkModel(G)
    model.d[0] = 0.4
    model.d[1] = 0.08

    model.basal[1:] = [-5, -5, -5, -5, -5, -5, -5, -5]
    model.inter[0, 1] = 10
    model.inter[1, 2] = 10
    model.inter[2, 3] = 10
    model.inter[3, 4] = 10
    model.inter[3, 5] = 10
    model.inter[3, 6] = 10
    model.inter[4, 1] = -10
    model.inter[5, 1] = -10
    model.inter[6, 1] = -10
    model.inter[4, 4] = 10
    model.inter[5, 5] = 10
    model.inter[6, 6] = 10
    model.inter[4, 8] = -10
    model.inter[4, 7] = -10
    model.inter[6, 7] = 10
    model.inter[7, 6] = 10
    model.inter[8, 8] = 10

    # Save true network topology
    inter = 1 * (abs(model.inter) > 0)
    path = '/cluster/scratch/jassan/cardamom/'
    np.save(path + f'FN8/True/inter_t20_{r+1}', inter)

    # Generate data
    for k in range(C):
        # print(f'* Cell {k+1}')
        sim = model.simulate(time[k], burnin=5)
        data[k+1,2:] = np.random.poisson(sim.m[-1])

    # Save data for use with PIDC
    path = '/cluster/scratch/jassan/cardamom/'
    fname = path + f'FN8/Data/data_t20_{r+1}.txt'
    np.savetxt(fname, data.T, fmt='%d', delimiter='\t')

N = int(sys.argv[1])
r = int(sys.argv[2])
# seed for the simulation with 20 time points:
seed_seq = np.random.SeedSequence(110547844434648552605459073395773322289) 
 # seed for the simulation with 1000 cells per time point: 54399050814365466361179279051255698430) # for 1000
seeds = seed_seq.spawn(N)
seed = seeds[r].generate_state(5)
run(r, seed)
