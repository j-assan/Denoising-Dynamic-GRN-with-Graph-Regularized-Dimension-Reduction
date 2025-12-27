import numpy as np
import pyttb as ttb
from sympy import symbols, lambdify
from scipy.linalg import svd

def time_dependency(func_names, size, sel_mode=None, p=0.5, rng=None):
	"""Create the functions f_i(t) to get a datatensor X[t,selection(i)] = noise(X0[selection(i)] * f_i(t))
	
	Parameters
	----------
	func_names : list of str
		expression of the function f
	size : tuple of int
		size of the non time dimensions of the tensor
	sel_mode : str, optional
		options are "allow_overlap" or "no_overlap". If not specified each function will
		be applied to every entry
	p : float, list of floats
		probability for each entry to depend on function f_i
	"""
	rng = np.random.default_rng(rng)
	time = []
	func_names = [f.strip() for f in func_names.split("+")]
	selections = np.zeros((len(func_names), size[0], size[1]))
	x = symbols('x')
	if isinstance(func_names, str): 
		time.append(lambdify(x, func_names))
		selections[0] = rng.choice([0,1], size, p)
	elif len(func_names) == 1 or sel_mode == 'allow_overlap':
		for i,f in  enumerate(func_names):
			time.append(lambdify(x, f))
			selections[i] = rng.choice([0,1], size, p)
	elif sel_mode == 'no_overlap':
		selection = rng.choice(range(len(func_names)), size, p)
		for i,f in  enumerate(func_names):
			time.append(lambdify(x, f))
			selections[i] = (selection==i)
	else:
		for i, f in  enumerate(func_names):
			time.append(lambdify(x, f))
		selections += 1
	return time, selections

def generate_tensor(X0,
					T,
					func_names,
					sel_mode = None,
					p = 0.5,
					noise = "gaussian",
					num_trajectories = 10,
					num_time_steps = 10,
					dt_EM = 0.1,
					dt = 0.1,
					G = None,
					X0_dist = None,
					destroyed_samples = False,
					shuffle = False,
					var = 0.1,
					rank = 1,
					rng=None):
	rng = np.random.default_rng(rng)
	m, n = X0
	time_dependencies, _ = time_dependency(func_names, X0, sel_mode, p, rng)
	factor_mat, weights =  generate_network_factor_matrices(X0[0], X0[1], rank, reps=len(time_dependencies), rng=rng)
	weights = weights / sum(weights**2)
	factor_one = np.zeros((T, rank*len(time_dependencies)))
	for i, f in enumerate(time_dependencies):
		fmax = np.max(f(np.arange(T)))
		for j, r in enumerate(range(rank)):
			factor_one[:,i*rank + j] = f(np.arange(T)) / fmax
	factor_mat.insert(0, factor_one)
	K = ttb.ktensor(factor_mat, weights=weights)
	K.weights = K.weights / K.norm() * np.sqrt(T)
	assert abs(K.norm()**2 / T - 1) < 1e-6,  f"time average X(t)={K.norm()**2 / T} not 1"
	Xstar = K.full()
	X = ttb.tensor(np.zeros((T, m, n)))
	for t in range(T):
		X[t,:,:] = Xstar.data[t,:,:] + rng.normal(0, np.sqrt(var/(m*n)), size=(m,n))
	assert var == 0.0 or abs(np.sum((X.data-Xstar.data)**2) / T - var) < var*0.1, f"avg||eps(t)||_2^2 = {np.sum((X.data-Xstar.data)**2) / T}, var = {var}"
	K.arrange()
	K.fixsigns()
	return X, Xstar, K
	

def X0_gen(m, n, rng):
	return rng.standard_normal((m,n))



def generate_network_factor_matrices(m, n, rank, reps=1, rng=None):
	assert rank <= min(m, n)
	rng = np.random.default_rng(rng)
	# TODO
	factor_matrices = [np.zeros((m, rank*reps)), np.zeros((n, rank*reps))]
	weights = np.zeros(rank*reps)
	for i in range(reps):
		A = rng.normal(0,1, size=(m,n))
		U, d, Vh = svd(A, full_matrices=False)
		factor_matrices[0][:,rank*i:rank*(i+1)] = U[:,:rank]
		factor_matrices[1][:,rank*i:rank*(i+1)] = Vh.T[:,:rank]
		weights[rank*i:rank*(i+1)] = d[:rank]
	return factor_matrices, weights

		

