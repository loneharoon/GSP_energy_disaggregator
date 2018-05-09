#!/usr/bin/env python2


# -*- coding: utf-8 -*-
"""
This implements GSP energy disaggregation method proposed in the paper "On a training-less solution for non-intrusive appliance load monitoring using graph signal processing"

Created on Thu Feb  1 15:42:41 2018

@author: haroonr
"""
from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import gsp_support as gsp
from collections import OrderedDict
from copy import deepcopy

#%%

print("This code has been tested in Python 2")
csvfile = "/Volumes/MacintoshHD2/Users/haroonr/Dropbox/GSP_disaggregator/demo_data.csv"
df = pd.read_csv(csvfile, index_col = "Time")
df.index = pd.to_datetime(df.index)

sigma = 40;
ri = 0.1 # obained empirically
T_Positive = 30;
T_Negative = -30;
# alpha define weight given to magnitude and beta define weight given to time
alpha = 0.5
beta  = 0.5
# this defines the number of times an appliance is set ON in one month
instancelimit = 25 # [normal value 25 for one month] 
#%% 
main_val = df.values
main_ind = df.index
data_vec =  main_val
delta_p = [round(data_vec[i+1] - data_vec[i],2) for i in range(0,len(data_vec)-1)]
event =  [i for i in range(0, len(delta_p)) if (delta_p[i] > T_Positive or delta_p[i] < T_Negative) ]
clusters = gsp.refined_clustering_block(event, delta_p, sigma, ri)
finalclusters, pairs = gsp.pair_clusters_appliance_wise(clusters, data_vec, delta_p, instancelimit)
appliance_pairs = gsp.feature_matching_module(pairs, delta_p, finalclusters, alpha, beta)
power_series = gsp.generate_appliance_powerseries(appliance_pairs, delta_p)

power_timeseries = gsp.create_appliance_timeseries_signature(power_series, main_ind)
gsp_result = pd.concat(power_timeseries, axis = 1)

#%%
