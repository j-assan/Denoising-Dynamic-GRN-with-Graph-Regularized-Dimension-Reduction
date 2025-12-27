# Denoising-Dynamic-GRN-with-Graph-Regularized-Dimension-Reduction

## Installation
1. clone the repository `git clone https://github.com/j-assan/Denoising-Dynamic-GRN-with-Graph-Regularized-Dimension-Reduction.git`
2. create virtual environment `python -m venv venv`
3. activate virtual environment `source venv/bin/activate`
4. update pip `pip install --upgrade pip`
5. install requirements `pip install -r requirements.txt`
if there are troubles with arboreto (GRNboost2) and dask, install the github version of arboreto:
```pip uninstall arboreto
git clone https://github.com/aertslab/arboreto.git
cd arboreto
python setup.py install
``` 

## Structure
There are 4 folders
- `src` contains the code for running GrapCP and GraphPCA
- one folder for each of the experiments described in the main text

###
The scripts `simulate_data_network.py` are used to simulate the data with 1000 cells per time point and with 20 time points. For 100 cells per time point and 10 time points, the original data by Ventre et al. (2023) are used.

The file `infer_grnboost2_all.py` is used to infer GRNs using cells from all time points. The file `infer_grnboost2.py` is used to infer GRNs using cells from each time point plus the two adjacent time points.

The file `denoise.py` is used to get GRNs denoised using GraphCP or GraphPCA and to get static GRNs from the dynamic GRNs by taking the average or maximum score for each interaction.

 The file `selection.py` has the function `selection` to find the combination of parameters with the best results. The file `figure2.py` is used to plot the results.

### Denoise experiment
Get the number of experiments to run (should be 1440)
```python run.py --input "input.in" --get-num-exp```
For i in 0:1439 run:
```python run.py --input "input.in" --outdir "test" --only-exp-id i```

### Stability experiment
To get GRNs using GRNboost2 run:
```python grninference.py```
The script defines and calls a function called `worker` specify if fraction of cells to be used and if cells of different worms should be binned together in this function. To reproduce the data in the text use the following seeds:
```
base: seed = 22323032525405440762759776405901633071
base-frac80: seed = 191387900439387131375866557119140424685
base-frac60: seed = 278258364927959352370261598130162581930
base-frac40: seed = 231639717000064068799980312866726940474
bin: seed = 285768097169853429834721162892017338400
bin-frac80: seed = 6011114618774008554049020720225778711
bin-frac60: seed = 170901267290282036139122595725023552659
bin-frac40: seed = 41648746375714764341902171417861875966
bin-frac20: seed = 235027571498096000902223140007522080777
```
