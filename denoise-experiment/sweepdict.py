import numpy as np
import json

def results():
	reg_lam = [0.0, 0.1, 1, 10]
	metrics = {"fit_noisy": r"$1-||X-\hat{X}||/||X||$", "fit": r"$1-||X^*-\hat{X}||/||X^*||$", "rel_error": r"$||X-\hat{X}||/||X||$"}
	ranks = np.linspace(1, 10, 10, dtype=int)
	names = []
	for k, v in metrics.items():
			for i, lam in enumerate(reg_lam):
				names.append(f'{k}_lam_{lam}')
				names.append(f'bestrank_lam_{lam}')
			for i, r in enumerate(ranks):
				names.append(f'{k}_rank_{r}')
				names.append(f'bestreg_rank_{r}')
				
	print(json.dumps(names))

def files(reg_lam = [0.0, 0.1, 1, 10], maxrank=20, n_ranks=20):
	names = []
	metrics = {"fit_noisy": r"$1-||X-\hat{X}||/||X||$", "fit": r"$1-||X^*-\hat{X}||/||X^*||$", "rel_error": r"$||X-\hat{X}||/||X||$"}
	for m in metrics.keys():
		names.append(f"{m}.png")
	names.append("reference.png")
	ranks = np.linspace(1, maxrank,n_ranks, dtype=int)
	for lam in reg_lam:
		for proj_dim in ranks:
			names.append(f"time_components_k_{proj_dim}_lam_{lam}.png")
	return names
