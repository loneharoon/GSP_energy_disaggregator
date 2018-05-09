#!/usr/bin/env python2


# -*- coding: utf-8 -*-
"""
In this I implement Bochao's GSP disaggregation method
Created on Thu Feb  1 15:42:41 2018

@author: haroonr
"""
from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
import pickle
sys.path.append('/Volumes/MacintoshHD2/Users/haroonr/Dropbox/UniOfStra/AD/disaggregation_codes/')
import accuracy_metrics_disagg as acmat
import matplotlib.pyplot as plt
import gsp_support as gsp
import AD_support as ads
from collections import OrderedDict
from copy import deepcopy
import standardize_column_names

#%%

print("Run this code in python 2")
dir = "/Volumes/MacintoshHD2/Users/haroonr/Detailed_datasets/REFITT/REFIT_selected/"
home = "House10.csv"
df = pd.read_csv(dir+home, index_col = "Time")
df.index = pd.to_datetime(df.index)
df_sub = deepcopy(df[:])
#% Resampling data
#TODO : Toggle switch and set sampling rate correctly
resample = True
if resample: 
  df_samp = df_sub.resample('10T',label='right',closed='right').mean()
  df_samp.drop('Issues',axis=1,inplace=True)
  standardize_column_names.rename_appliances(home,df_samp) # this renames columns
  #df_samp.rename(columns={'Aggregate':'use'},inplace=True) # renaming agg column
  print("*****RESAMPling DONE********")
  if home == "House16.csv":
      df_samp = df_samp[df_samp.index!= '2014-03-08'] # after resamping this day gets created 
else:
  df_samp = deepcopy(df_sub)
  df_samp.drop('Issues',axis=1,inplace=True)
  standardize_column_names.rename_appliances(home,df_samp) # this renames columns  

energy = df_samp.sum(axis=0)
high_energy_apps = energy.nlargest(7).keys() # CONTROL : selects few appliances
df_selected = df_samp[high_energy_apps]
#%
#TODO : TUNE ME
denoised = False
if denoised:
    # chaning aggregate column
    iams = high_energy_apps.difference(['use'])
    df_selected['use'] = df_selected[iams].sum(axis=1)
    print('**********DENOISED DATA*************8')
train_dset,test_dset = ads.get_selected_home_data(home,df_selected)
#%%
#TODO : tune us
sigma = 40;
ri = 0.1 # obained empirically
# good thresholds are as
# home10: 40 watts,
T_Positive = 30;
T_Negative = -30;
# alpha define weight given to magnitude and beta define weight given to time
alpha = 0.5
beta = 0.5
# this defines the number of times an appliance is set ON in one month
instancelimit = 25 # [normal value 25 for one month] for home 18, I set it to  15 (otherwise it predicts less no. of appliances), for remaining home it was set 25. and for home 10 (at 10 minutes sampling I set it to 20)
#%% if you want to run in monthly wise, then run next cell and skip this one
main = train_dset['use']
main_val = main.values
main_ind = main.index
data_vec =  main_val
delta_p = [round(data_vec[i+1] - data_vec[i],2) for i in range(0,len(data_vec)-1)]
event =  [i for i in range(0, len(delta_p)) if (delta_p[i] > T_Positive or delta_p[i] < T_Negative) ]
clusters = gsp.refined_clustering_block(event, delta_p, sigma, ri)
finalclusters, pairs = gsp.pair_clusters_appliance_wise(clusters, data_vec, delta_p, instancelimit)
appliance_pairs = gsp.feature_matching_module(pairs, delta_p, finalclusters, alpha, beta)
power_series = gsp.generate_appliance_powerseries(appliance_pairs, delta_p)
power_timeseries = gsp.create_appliance_timeseries_signature(power_series, main_ind)
gsp_result = pd.concat(power_timeseries, axis = 1)
mapped_names = gsp.map_appliance_names(train_dset, gsp_result)
gsp_result.rename(columns = mapped_names, inplace = True)
#gsp_result.plot(subplots=True)

#%% run gsp monthly basis
monthly_groups = test_dset.groupby(test_dset.index.month)
#monthly_gsp_res = OrderedDict()
monthly_gsp_res = []
gt = []
for k,v in monthly_groups:
    print('Month is {}'.format(k))
    dset = v
    main = dset['use']
    main_val = main.values
    if len(main_val) < 10: # when downsampling creates unnecessarily extra one day reading
        continue
    main_ind = main.index
    data_vec =  main_val
    delta_p = [round(data_vec[i+1] - data_vec[i],2) for i in range(0,len(data_vec)-1)]
    event =  [i for i in range(0, len(delta_p)) if (delta_p[i] > T_Positive or delta_p[i] < T_Negative) ]
    clusters = gsp.refined_clustering_block(event, delta_p, sigma, ri)
    finalclusters, pairs = gsp.pair_clusters_appliance_wise(clusters, data_vec, delta_p, instancelimit)
    appliance_pairs = gsp.feature_matching_module(pairs, delta_p, finalclusters, alpha, beta)
    power_series = gsp.generate_appliance_powerseries(appliance_pairs, delta_p)
    power_timeseries = gsp.create_appliance_timeseries_signature(power_series, main_ind)
    gsp_result = pd.concat(power_timeseries, axis = 1)
    mapped_names = gsp.map_appliance_names(dset, gsp_result)
    gsp_result = gsp_result.rename(columns = mapped_names, inplace = False)
    print ('Keep appliances present in training data only')
    apps = [v for k, v in mapped_names.items()]
    #monthly_gsp_res[k] =  gsp_result[apps] # drops remaining columns
    monthly_gsp_res.append(gsp_result[apps])
    gt.append(dset)
  #%% save results
save_dir = "/Volumes/MacintoshHD2/Users/haroonr/Detailed_datasets/REFITT/Intermediary_results/"
#TODO : TUNE ME
filename = save_dir + "noisy/gsp/selected_10min/" + home.split('.')[0]+'.pkl'
gsp_result = {}
gsp_result['decoded_power'] = pd.concat(monthly_gsp_res,axis=0)
gsp_result['actual_power'] = pd.concat(gt,axis=0)
gsp_result['train_power'] = train_dset
handle = open(filename,'wb')
pickle.dump(gsp_result,handle)
handle.close()      
    

#%%
fig,axes = plt.subplots(nrows=9,ncols=2,sharex=False,sharey=False,figsize=(12,15))
app =0
for ax in range(len(power_series)//2):
    axes[ax,0].plot(power_series[app].timestamp,power_series[app].power)
    app+=1
    axes[ax,1].plot(power_series[app].timestamp,power_series[app].power)
    app+=1
#fig.savefig("gsp.png")
#%%
def create_appliance_timeseries_signature(power_series,main_ind):
    '''This converts ordinary number indexexed power series into time indexed power series'''
    result = OrderedDict()
    for i in range(len(power_series)):
        #print (i)
        temp = power_series[i]
        if len(temp) < 1: # corner case found
            continue
        temp.index = temp.timestamp
        dummy = pd.Series(0,main_ind)
        dummy[main_ind[temp.index.values]] = temp.power.values
        result[i] = dummy
        
        
        
    return(result)



