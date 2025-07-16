import numpy as np
import neuroboros as nb
from scipy.stats import zscore,pearsonr
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from hyperalignment.procrustes import procrustes
from hyperalignment.ridge import ridge
from hyperalignment.local_template import compute_template
from hyperalignment.searchlight import searchlight_procrustes, searchlight_ridge
import pandas as pd
import seaborn as sns

dset = nb.Life()
sids = dset.subjects
print(sids)

training_runs = [1,2]
test_runs = [3,4]
lr = 'lr' # left + right hemisphere, 'l' for left hemisphere, 'r' for right hemisphere
ts1_train = dset.get_data(sids[0], 'life', [1, 2], lr)
ts1_test = dset.get_data(sids[0], 'life', [3, 4], lr)

ts2_train = dset.get_data(sids[1], 'life', [1, 2], lr)
ts2_test = dset.get_data(sids[1], 'life', [3, 4], lr)
radius = 20
sls, dists = nb.sls(lr, radius, return_dists=True)
print(len(sls),sls[0].shape,sls[1].shape) 
# for onavg-ico32 we have 19341 vertices after medial wall masking (9675 left and 9666 right), searchlight 0 has 25 vertices, and 1 has 28 vertices

wRHA = searchlight_ridge(ts1_train, ts2_train, sls, dists, radius) 