#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Includes all supporting functions for gsp_disaggregator.py
Created on Fri Feb  2 12:09:27 2018

@author: haroonr
"""

from __future__ import division 
import numpy as np
import pandas as pd
from collections import OrderedDict
from copy import deepcopy
from collections import defaultdict
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import csv
from IPython.display import display
from math import sqrt
import os

#%%
def gspclustering_event2(event,delta_p,sigma):
 
  winL = 1000 # this define  number of observations in a window, the algorthm works in a sliding window manner
  Smstar = np.zeros((len(event),1))
  for k in range(0,int(np.floor(len(event)/winL))):
    r = []
    event_1 =  event[k*winL:((k+1)*winL)]
    # followed as such from the MATLAB code
    r.append(delta_p[event[0]])
    [r.append(delta_p[event_1[i]]) for i in range(0,len(event_1))]
    templen = winL + 1
    Sm = np.zeros((templen,1))
    Sm[0] = 1;

    Am = np.zeros((templen,templen))
    for i in range(0,templen):
      for j in range(0,templen):
         Am[i,j] = math.exp(-((r[i]-r[j])/sigma)**2);
         #Gaussian kernel weighting function
    Dm = np.zeros((templen,templen));
    # create diagonal matrix
    for i in range(templen):
      Dm[i,i] = np.sum(Am[:,i]);
    Lm = Dm - Am;
    Smstar[k*winL:(k+1)*winL] = np.matmul(np.linalg.pinv(Lm[1:templen,1:templen]), ((-Sm[0].T) * Lm[0,1:templen]).reshape(-1,1));
  # for remaining elements of the event list
  if (len(event)%winL > 0):
    r = []
    event_1 =  event[int(np.floor(len(event)/winL))*winL:]
    newlen = len(event_1) + 1
    r.append(delta_p[event[0]])
    [r.append(delta_p[event_1[i]]) for i in range(0,len(event_1))]
    Sm = np.zeros((newlen,1))
    Sm[0] = 1;
    Am = np.zeros((newlen,newlen))
    for i in range(newlen):
      for j in range(newlen):
          #print(i,j)
          #print('\n')
         Am[i,j] =  math.exp(-((r[i]-r[j])/sigma)**2);
         #Gaussian kernel weighting function
    Dm = np.zeros((newlen,newlen));
    for i in range(newlen):
      Dm[i,i] = np.sum(Am[:,i]);
    Lm = Dm - Am;
    Smstar_temp = np.matmul(np.linalg.pinv(Lm[1:newlen,1:newlen]), ((-Sm[0].T) * Lm[0,1:newlen]).reshape(-1,1));
    Smstar[(int(np.floor(len(event)/winL))*winL):len(event)] = Smstar_temp
  # 0.98 values has been obtained emparically
  cluster = [event[i] for i in range(len(Smstar)) if (Smstar[i] > 0.98)]
  return cluster

#%%
def johntable(clusters,precluster,delta_p,ri):
  import math
  for h in range(0,len(clusters)):  
    stds = np.std([delta_p[i] for i in clusters[h]],ddof=1)
    if(math.isnan(stds)):
      stds = 0
    means = np.mean([delta_p[i] for i in clusters[h]])
    if abs(stds/means) <= ri :
      precluster.append([i for i in clusters[h]])
  return precluster

#%%
def find_new_events(clusters,delta_p,ri):
  ''' This differs from johntable function in line containing divison statement'''
  import math
  newevents = []
  for h in range(0,len(clusters)):  
    stds = np.std([delta_p[i] for i in clusters[h]],ddof=1)
    if(math.isnan(stds)):
      stds = 0
    means = np.mean([delta_p[i] for i in clusters[h]])
    if abs(stds/means) > ri :
      newevents.append([i for i in clusters[h]])
  newevents = [subitem for item in newevents for subitem in item]
  return newevents

#%%
def feature_matching_module(pairs,DelP,Newcluster,alpha,beta):
    #alpha = 0.5
    #beta = 0.5
    appliance_pairs = OrderedDict()
    for i in range(len(pairs)):
      pos_cluster = sorted(Newcluster[pairs[i][0]])
      neg_cluster = sorted(Newcluster[pairs[i][1]])
      flag = 0
      state_pairs = []
      for j in range(len(pos_cluster)):
         if j==len(pos_cluster)-1:  # last positive element
             flag = 1 
             start_pos = pos_cluster[j]
         if flag:
             neg_set = [h for h in neg_cluster if (h > start_pos)]
         else:
             start_pos = pos_cluster[j]
             next_pos = pos_cluster[j+1]
             if (next_pos - start_pos) == 1:  #shows both are consecutive to one another, so skip
                 continue
             neg_set = [h for h in neg_cluster if (h > start_pos and h< next_pos)]
         if len(neg_set)==1:
             #pair the positive and neg edges
             pair= (start_pos,neg_set[0])
             state_pairs.append(pair)
         elif len(neg_set)==0: # no negative edge found
             #print("No negative edge found for positive edge: {}".format(start_pos))
             continue
         else:
             phi_m = [DelP[h]+DelP[start_pos] for h in neg_set]
             phi_t = [(h-start_pos) for h in neg_set]
             newlen= len(neg_set)
             Am = np.zeros((newlen,newlen))
             At = np.zeros((newlen,newlen))
             sigma = 1 # confirmed with Bochao
             for k in range(newlen):
                 for p in range(newlen):
                     Am[k,p] = np.exp(-((phi_m[k]-phi_m[p])/sigma)**2);
             for k in range(newlen):
                 for p in range(newlen):
                     At[k,p] = np.exp(-((phi_t[k]-phi_t[p])/sigma)**2);
             Dm = np.zeros((newlen,newlen));
             for z in range(newlen):
                 Dm[z,z] = np.sum(Am[:,z]);
             Lm = Dm - Am;
             Sm = np.zeros((newlen,1))
             Sm[0] = np.average(phi_m)
             Smstar = np.matmul(np.linalg.pinv(Lm[0:newlen,0:newlen]), ((-Sm[0].T) * Lm[0,0:newlen]).reshape(-1,1))
             Dt = np.zeros((newlen,newlen));
             for z in range(newlen):
                 Dt[z,z] = np.sum(At[:,z]);
             Lt = Dt - At;
             St = np.zeros((newlen,1))
             St[0] = np.median(phi_t)
             Ststar = np.matmul(np.linalg.pinv(Lt[0:newlen,0:newlen]), ((-St[0].T) * Lt[0,0:newlen]).reshape(-1,1))
             result_vec = []
             for f in range(Smstar.shape[0]):
                 #temp=alpha * Smstar[f][0] + beta  * Ststar[f][0]
                 temp = np.nansum([alpha * Smstar[f][0] , beta  * Ststar[f][0] ])
                 result_vec.append(temp)
             #print(i,j)
             best_pos = [a for a in range(len(result_vec)) if (result_vec[a] == min(result_vec))][0]
             pair = (start_pos,neg_set[best_pos])
             state_pairs.append(pair)
      appliance_pairs[i] = state_pairs
    return appliance_pairs

#%%
def generate_appliance_powerseries(appliance_pairs,DelP):
    ''' generates full power series of appliances'''
    print ("3 of 6> generates full power series of appliances")
    appliance_signatures = OrderedDict()
    power_series = OrderedDict()
    ctlf = OrderedDict()
    for i in range(len(appliance_pairs)):
        events = appliance_pairs[i]
        timeseq= []
        powerseq  = []
        for event in events:
            start= event[0]
            end = event[1]
            duration = end - start
            instance = []
            instance.append([DelP[start]])
            temp= np.repeat(np.nan,duration-1).tolist()
            instance.append(temp)
            instance.append([abs(DelP[end])])
            final = np.array([j for sub in instance for j in sub], dtype=float).flatten()
            timeval = range(start,end+1,1)
            #print (event)
            powerval = interpolate_values(final) if sum(np.isnan(final)) else final
            timeseq.append(timeval)
            powerseq.append(powerval)
        powerseq =  [j for sub in powerseq for j in sub]
        timeseq =  [j for sub in timeseq for j in sub]
        power_series[i] = pd.DataFrame({'timestamp':timeseq,'power':powerseq})
        appliance_signatures[i] = pd.DataFrame(powerseq)

    return power_series, appliance_signatures

#%%
def label_appliances(appliance_signatures, signature_database, threshold):
    print ("4 of 6> checking appliance power signatures matches")
    labeled_appliances = OrderedDict()
    dfw = pd.concat(appliance_signatures, axis = 1, ignore_index=True)
    dfw.drop(dfw.index[1], axis=1)
    #dfw.to_csv("signature_database.csv")

    dfr = pd.read_csv(signature_database, index_col=0)
    rowr, columnsr = dfr.shape
    roww, columnsw = dfw.shape
    print("        > found "+ str(columnsw) + " appliances. Verifying signature matching")
    for i in range(columnsw):
        for j in range(columnsr):
            last_idxr = dfr.iloc[:,j].last_valid_index()
            last_idxw = dfw.iloc[:,i].last_valid_index()
            D = FastDTW(dfw.iloc[:last_idxw,i].values, dfr.iloc[:last_idxr,j].values, 10)
            if D < threshold:
                print("          > found match " + str(i+1) + " with " + dfr.iloc[:0,j].name)
                labeled_appliances[i] = dfr.iloc[:0,j].name

    return labeled_appliances

#%%
def DTW(s1, s2):
    DTW={}
    
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

#%%
def FastDTW(s1, s2, w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

#%%
def write_csv_df(path, filename, df):
    # Give the filename you wish to save the file to
    pathfile = os.path.normpath(os.path.join(path,filename))
    
    # Use this function to search for any files which match your filename
    files_present = os.path.isfile(pathfile)
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
        df.to_csv(pathfile)
    else:
        overwrite = raw_input("WARNING: " + pathfile + " already exists! Overwrite <y/n>? \n ")
        if overwrite == 'y':
            df.to_csv(pathfile)
        elif overwrite == 'n':
            return
        else:
            print("Not a valid input. Data is NOT saved!\n")
    return

#%%
def calculate_energy_pct(dfd, dfc):
    # Plot total energy consumption of each appliance found

    fig = plt.figure()
    ax1 = fig.add_axes([0, .3, .5, .5], aspect=1)
    ax2 = fig.add_axes([.5, .3, .5, .5], aspect=1)
    fig.suptitle('Total energy consumption', fontsize = 14)

    cons1 = dfd[dfd.columns.values].sum().sort_values(ascending=False)
    cons2 = dfc[dfc.columns.values].sum().sort_values(ascending=False)

    ax1.pie(cons1.values, autopct='%1.1f%%', startangle=90)
    ax2.pie(cons2.values, autopct='%1.1f%%', startangle=90)
    first_legend = ax1.legend(dfd.columns, loc = 'lower center', bbox_to_anchor=(.5, -.4), fontsize = 8)
    second_legend = ax2.legend(dfc.columns, loc = 'lower center', bbox_to_anchor=(.5, -.4), fontsize = 8)
    ax1.set_title('Ground truth')
    ax2.set_title('Disaggregated')
    ax1.axis('equal')
    ax2.axis('equal')
    plt.tight_layout()
    plt.show()

#%%
def interpolate_values(A):
    ''' fills values between pairs of events'''
    if type(A) ==list :
        A= np.array(A)
    ok = ~np.isnan(A)
    xp = ok.nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)
    A = [round(i) for i in A]
    return A

#%%
def create_appliance_timeseries(power_series,main_ind):
    '''This converts ordinary number indexed power series into time indexed power series'''
    print ("5 of 6> creating appliance power timeseries")
    result = OrderedDict()
    for i in range(len(power_series)):
        temp = power_series[i]
        if len(temp) < 1: # corner case found
            continue
        temp.index = temp.timestamp
        dummy = pd.Series(0,main_ind)
        dummy = dummy.loc[~dummy.index.duplicated(keep='first')]
        dummy[main_ind[temp.index.values]] = temp.power.values
        result[i] = dummy
    return(result)

#%%
def refined_clustering_block(event,delta_p,sigma,ri):
    '''this section performs clustering as explained in Figure 1 (Flowchart) of the IEEE Acess paper'''
    sigmas = [sigma,sigma/2,sigma/4,sigma/8,sigma/14,sigma/32,sigma/64]
    Finalcluster = []
    for k in range(0,len(sigmas)):
        clusters = []     
        event = sorted(list(set(event)-set(clusters))) 
        while len(event):
            clus =  gspclustering_event2(event,delta_p,sigmas[k]);
            clusters.append(clus)
            event = sorted(list(set(event)-set(clus)))
        if k == len(sigmas)-1:
            Finalcluster = Finalcluster + clusters 
        else:
            jt = johntable(clusters,Finalcluster,delta_p,ri)
            Finalcluster = jt
            events_updated = find_new_events(clusters,delta_p,ri)
            events_updated = sorted(events_updated)
            event = events_updated
    if len(event) > 0:
      Finalcluster.append(event)
    return Finalcluster

#%%
def find_closest_pair(cluster_means,cluster_group): 
    ''' this identifies closest clusters wrt to mean and then merges those clusters into one'''
    distances = []   
    for i in range(len(cluster_means)-1):
        for j in range((i+1),len(cluster_means)):
            #print i,j
           distance = abs(cluster_means[i] - cluster_means[j])  
           distances.append((i,j,distance))
    merge_pair = min(distances, key = lambda h:h[2])
    # convert list to dict for simplicity
    cluster_dict = {}
    for i in range(len(cluster_group)): 
        cluster_dict[i] =  cluster_group[i]
    # merge cluster using above merge_pair and copy remaining as such
    tempcluster = []
    tempcluster.append(cluster_dict[merge_pair[0]] + cluster_dict[merge_pair[1]])
    del cluster_dict[merge_pair[0]]
    del cluster_dict[merge_pair[1]]
    for k,v in cluster_dict.items():
        tempcluster.append(v)
    return tempcluster

#%%
def pair_clusters_appliance_wise(Finalcluster, data_vec, delta_p, instancelimit):
    print ("2 of 6> pair clusters appliance wise")
    #% Here I count number of members of each cluster, their mean and standard deviation and store such stats in Table_1. Next, I sort 'Finalcluster' according to cluster means in decreasing order.
    Table_1 =  np.zeros((len(Finalcluster),4))
    for i in range(len(Finalcluster)):
      Table_1[i,0] = len(Finalcluster[i])
      Table_1[i,1] = np.mean([delta_p[j] for j in Finalcluster[i]])
      Table_1[i,2] = np.std([delta_p[j] for j in Finalcluster[i]],ddof=1)
      Table_1[i,3] =  abs(Table_1[i,2]/ Table_1[i,1])
    #% sorting module
    sort_means = np.argsort(Table_1[:,1]).tolist() # returns positions of sorted array
    sort_means.reverse() # gives decreasing order
    sorted_cluster =[]
    FinalTable = []
    for i in range(len(sort_means)):
      sorted_cluster.append(Finalcluster[sort_means[i]])
      FinalTable.append(Table_1[sort_means[i]].tolist())
    #%
    # Here I reduce number of clusters. I keep clusters with more than or equal 'instancelimit' members as such and in next cell I merge cluster with less than 5 members to clusters with more than 'instancelimit' members 
    # DelP seems redundant but lets move on
    DelP = [(data_vec[i+1]-data_vec[i]).round(2) for i in range(0,len(data_vec)-1)]
    Newcluster_1 = []
    Newtable = []
    #instancelimit = 20
    for i in range(0,len(FinalTable)):
      if (FinalTable[i][0] >= instancelimit):
        Newcluster_1.append(sorted_cluster[i])
        Newtable.append(FinalTable[i])
    Newcluster = Newcluster_1
    #% merge cluster with less than intancelimit members to clusters with more than 5 members 
    for i in range(0,len(FinalTable)):
      if(FinalTable[i][0] < instancelimit ):
        for j in range(len(sorted_cluster[i])):
          count =  []
          for k in range(len(Newcluster)):
            count.append(norm.pdf(DelP[sorted_cluster[i][j]],Newtable[k][1],Newtable[k][2]))
          asv = [h == max(count) for h in count]
          if sum(asv) == 1:
            johnIndex = count.index(max(count))
          elif DelP[sorted_cluster[i][j]] > 0:
              #print("case1",i,j)
            tablemeans = [r[1] for r in Newtable]
            tempelem = [r for r in tablemeans if r < DelP[sorted_cluster[i][j]]][0]
            johnIndex = tablemeans.index(tempelem)
          else:
              #print("case else",i,j)
            tablemeans = [r[1] for r in Newtable]
            tempelem = [r for r in tablemeans if r > DelP[sorted_cluster[i][j]]].pop()
            johnIndex = tablemeans.index(tempelem)
          Newcluster[johnIndex].append(sorted_cluster[i][j])
    # updating table means in new table
    Table_2 =  np.zeros((len(Newcluster),4))
    for i in range(len(Newcluster)):
      Table_2[i,0] = len(Newcluster[i])
      Table_2[i,1] = np.mean([delta_p[j] for j in Newcluster[i]])
      Table_2[i,2] = np.std([delta_p[j] for j in Newcluster[i]],ddof=1)
      Table_2[i,3] =  abs(Table_2[i,2]/ Table_2[i,1])
    Newtable = Table_2
    #%
    # Ideally, number of positive clusters should be equal to negative clusters. if one type is more than the other then we merge extra clusters until we get equal number of positive and negative clusters
    pos_clusters = neg_clusters = 0
    for i in range(Newtable.shape[0]):
        if Newtable[i][1] > 0:
            pos_clusters += 1
        else:
            neg_clusters += 1
    Newcluster_cp = deepcopy(Newcluster)
    # merge until we get equal number of positive and negative clusters
    while pos_clusters != neg_clusters:
        index_cluster = Newcluster_cp
        power_cluster = []
        for i in index_cluster:
            list_member = []
            for j in i:
                list_member.append(delta_p[j])
            power_cluster.append(list_member)
            
        clustermeans = [np.mean(i) for i in power_cluster]
        positive_cluster_chunk= []
        negative_cluster_chunk = []
        positive_cluster_means= []
        negative_cluster_means = []
        pos_clusters = neg_clusters = 0
        for j in range(len(clustermeans)):
           if clustermeans[j] > 0:
                pos_clusters += 1
                positive_cluster_chunk.append(index_cluster[j])
                positive_cluster_means.append(clustermeans[j])
           else:
                neg_clusters += 1
                negative_cluster_chunk.append(index_cluster[j])
                negative_cluster_means.append(clustermeans[j])
                
        if pos_clusters > neg_clusters:
            #print ('call positive')
             positive_cluster_chunk = find_closest_pair(positive_cluster_means, positive_cluster_chunk)
        elif neg_clusters > pos_clusters:
            #print ('call negative')
             negative_cluster_chunk = find_closest_pair(negative_cluster_means, negative_cluster_chunk)
        else:
            pass
        Newcluster_cp = positive_cluster_chunk + negative_cluster_chunk
    
    #%
    # Use Newcluster_cp for pairing. Basically here we combine one positive cluster with one negative cluster, which corresponds to ON and OFF instances of the same appliance
    clus_means = []
    for i in Newcluster_cp:
        list_member = []
        for j in i:
            list_member.append(delta_p[j])
        clus_means.append(np.mean(list_member))    
    pairs = []
    for i in range(len(clus_means)):
      if clus_means[i] > 0: # positive edge
        neg_edges = [ (abs(clus_means[i] + clus_means[j]),j) for j in range(i+1,len(clus_means)) if clus_means[j] < 0] # find all neg edges and their location in tuple form
        edge_mag = [j[0] for j in neg_edges] # 0 corresponds to list magnitude in the tuple
        match_loc = neg_edges[edge_mag.index(min(edge_mag))][1]
        pairs.append((i,match_loc))
    #%
    # while looking at pairs, we find that there are cases where more than one positive edge has paired with more than one negative edge. To solve this issue, we fill process again this pairing process. step 1: save this in default dic by negative edge wise step 2: see with which positive edge matches the negative edge matches the most
    #pairs_temp = deepcopy(pairs)
    dic_def = defaultdict(list)
    for value,key in pairs:
        dic_def[key].append(value)
    #%
    updated_pairs= []
    for neg_edge in dic_def.keys():
        #neg_edge= 35
        pos_edges = dic_def[neg_edge]
        if len(pos_edges) >1:
            candidates = [abs(clus_means[edge]+ clus_means[neg_edge]) for edge in pos_edges]
            good_pos_edge =  [el_pos for el_pos in range(len(candidates)) if candidates[el_pos] == min(candidates)][0]
            good_pair = (pos_edges[good_pos_edge],neg_edge)
        else:
            good_pair = (pos_edges[0],neg_edge)
        updated_pairs.append(good_pair)
    return Newcluster_cp,updated_pairs

#%%
# seems obselte one
def find_closest_pairs(start_cluster,end_cluster,cluster_means,required_reduction): 
    distances = []   
    for i in range(start_cluster, end_cluster):
        for j in range((i+1),end_cluster+1):
            #print i,j
           distance = abs(cluster_means[i] - cluster_means[j])  
           distances.append((i,j,distance))
    distances  = pd.DataFrame.from_records(distances)
    distances.columns = ['cluster_1','cluster_2','difference']
    distances.sort_values('difference',axis=0,inplace=True)
    return distances.head(required_reduction)
