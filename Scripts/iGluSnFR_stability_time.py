# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 14:08:50 2024

@author: ruo4037
"""

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
import pandas as pd
import numpy as np
import os
import scipy.stats as stats


path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Folders\Control_iglusnfr_stability_time'

files = sorted(os.listdir(path))


NAME = []
AMPS_SET1, AMPS_SET2 = [],[]
PPR_SET1, PPR_SET2 = [],[]
F0_SET1, F0_SET2 = [],[]
for file in range(len(files)):
    if 'Amp.xlsx' in files[file]:
        name = files[file].rsplit('_',7)[0]
        NAME.append(name)
        
        data = pd.read_excel(f'{path}/{files[file]}')       
        for i in range(data.shape[0]):
            amps = data.iloc[i,1:].astype(float)
            ppr = amps/amps[0].astype(float)
            
            if 'set1' in data.iloc[i,0]:
                AMPS_SET1.append(amps)
                PPR_SET1.append(ppr)
            elif 'set2' in data.iloc[i,0]:
                AMPS_SET2.append(amps)
                PPR_SET2.append(ppr)
    if 'traces_converted.xlsx' in files[file]:
        f0 = pd.read_excel(f'{path}/{files[file]}', sheet_name='F0')['Average'].values
        if 'set1' in files[file]:
            F0_SET1.append(f0)
        if 'set2' in files[file]:
            F0_SET2.append(f0)


##########################
######### PARAMS #########
##########################

# def viridis_colors(num):
#     clr = []
#     cmap = cm.rainbow(np.linspace(0.2,1,num))
#     for c in range(num):
#         rgba = cmap[c]
#         clr.append(colors.rgb2hex(rgba))
#     return clr

fibers = np.unique(NAME)
colors=['darkgreen','limegreen']
# palette = viridis_colors(len(fibers))

a1_set1 = np.array(AMPS_SET1)[:,0]
ppr_set1 = np.array(PPR_SET1)[:,5]
a1_set2 = np.array(AMPS_SET2)[:,0]
ppr_set2 = np.array(PPR_SET2)[:,5]
ppr_fold_change = np.array(PPR_SET2)/np.array(PPR_SET1)


x = np.arange(1,11)
fig, ax = plt.subplots(1,3,figsize=(8,3),tight_layout=True)
sns.boxplot(data=[np.array(AMPS_SET1)[:,0], np.array(AMPS_SET2)[:,0]], ax=ax[0], palette=colors, showmeans=True)
sns.boxplot(data=[np.array(PPR_SET1)[:,1], np.array(PPR_SET2)[:,1]], ax=ax[1], palette=colors, showmeans=True)
sns.boxplot(data=[np.ravel(F0_SET1), np.ravel(F0_SET2)], ax=ax[2], palette=colors, showmeans=True)
for i in range(len(np.unique(NAME))):
    fibers_a1_set1 = [a1_set1[j] for j in range(len(a1_set1)) if NAME[j] == np.unique(NAME)[i]]
    fibers_a1_set2 = [a1_set2[j] for j in range(len(a1_set2)) if NAME[j] == np.unique(NAME)[i]]
    fibers_ppr_set1 = [ppr_set1[j] for j in range(len(ppr_set1)) if NAME[j] == np.unique(NAME)[i]]
    fibers_ppr_set2 = [ppr_set2[j] for j in range(len(ppr_set2)) if NAME[j] == np.unique(NAME)[i]]
    # fibers_fold_change = [ppr_fold_change[j] for j in range(len(ppr_fold_change)) if NAME[j] == np.unique(NAME)[i]]
    ax[0].plot([0,1], [fibers_a1_set1, fibers_a1_set2], color='k', marker='o', alpha=0.5)
    ax[1].plot([0,1], [fibers_ppr_set1, fibers_ppr_set2], color='k', marker='o', alpha=0.5)
    # [ax.plot(x, fibers_fold_change[j], color=palette[i], marker='o') for j in range(len(fibers_fold_change))]

[ax[2].plot([0,1], [F0_SET1[i], F0_SET2[i]], color='k', marker='o') for i in range(len(F0_SET1))]

ax[0].set_ylabel('A1 (DF/F0)')
ax[1].set_ylabel('PPR2/1')
ax[2].set_ylabel('F0 (a.u.)')
ax[0].set_xticklabels(['Set1', 'Set2'])
ax[1].set_xticklabels(['Set1', 'Set2'])
ax[2].set_xticklabels(['Set1', 'Set2'])
ax[0].set_ylim(0)
ax[1].set_ylim(0)
ax[2].set_ylim(0)

mean_amps_set1 = np.mean(AMPS_SET1, axis=0)
mean_amps_set2 = np.mean(AMPS_SET2, axis=0)
sem_amps_set1 = stats.sem(AMPS_SET1, axis=0)
sem_amps_set2 = stats.sem(AMPS_SET2, axis=0)

cumsum_amps_set1 = np.cumsum(AMPS_SET1, axis=1)
cumsum_amps_set2 = np.cumsum(AMPS_SET2, axis=1)
mean_cumsum_amps_set1 = np.mean(cumsum_amps_set1, axis=0)
mean_cumsum_amps_set2 = np.mean(cumsum_amps_set2, axis=0)
sem_cumsum_amps_set1 = stats.sem(cumsum_amps_set1)
sem_cumsum_amps_set2 = stats.sem(cumsum_amps_set2)


mean_ppr_set1 = np.mean(PPR_SET1, axis=0)
mean_ppr_set2 = np.mean(PPR_SET2, axis=0)
sem_ppr_set1 = stats.sem(PPR_SET1, axis=0)
sem_ppr_set2 = stats.sem(PPR_SET2, axis=0)


# ax[2].plot(x, np.mean(cumsum_amps_set1, axis=0), color='b', marker='o')
# ax[2].plot(x, np.mean(cumsum_amps_set2, axis=0), color='orange', marker='o')
# ax[2].fill_between(x, mean_cumsum_amps_set1-sem_cumsum_amps_set1, mean_cumsum_amps_set1+sem_cumsum_amps_set1, color='b', alpha=0.2)
# ax[2].fill_between(x, mean_cumsum_amps_set2-sem_cumsum_amps_set2, mean_cumsum_amps_set2+sem_cumsum_amps_set2, color='orange', alpha=0.2)
# ax[2].set_ylabel('Cumulative amps')

for i in range(len(np.array(AMPS_SET1[0]))):
    T_amp, p_amp = stats.ttest_rel(np.array(AMPS_SET1)[:,i], np.array(AMPS_SET2)[:,i])
    T_ppr, p_ppr = stats.ttest_rel(np.array(PPR_SET1)[:,i], np.array(PPR_SET2)[:,i])
    print(p_amp)
    print(p_ppr)


##########################
### PROFILES STABILITY ###
##########################

fig_profiles, ax_profiles = plt.subplots(1, 2, figsize=(8,3), tight_layout=True)
for i in range(len(AMPS_SET1)):
    # plt.figure(figsize=(4,2))
    # plt.plot(x, AMPS_SET1[i], color='b', marker='o')
    # plt.plot(x, AMPS_SET2[i], color='orange', marker='o')
    
    ax_profiles[0].plot(x, AMPS_SET1[i], color=colors[0], alpha=0.2)
    ax_profiles[0].plot(x, AMPS_SET2[i], color=colors[1], alpha=0.2)
    ax_profiles[1].plot(x, PPR_SET1[i], color=colors[0], alpha=0.2)
    ax_profiles[1].plot(x, PPR_SET2[i], color=colors[1], alpha=0.2)
    
ax_profiles[0].plot(x, mean_amps_set1, color=colors[0], marker='o', label = 'Set1: t=0')
ax_profiles[0].plot(x, mean_amps_set2, color=colors[1], marker='o', label = 'Set2: t+8min')
ax_profiles[0].fill_between(x, mean_amps_set1-sem_amps_set1, mean_amps_set1+sem_amps_set1, color=colors[0], alpha=0.2)
ax_profiles[0].fill_between(x, mean_amps_set2-sem_amps_set2, mean_amps_set2+sem_amps_set2, color=colors[1], alpha=0.2)
ax_profiles[0].set_ylabel('Amplitude (DF/F0)')
ax_profiles[0].set_xlabel('Pulse number')
ax_profiles[0].set_ylim(0)
ax_profiles[0].legend()

ax_profiles[1].plot(x, mean_ppr_set1, color=colors[0], marker='o')
ax_profiles[1].plot(x, mean_ppr_set2, color=colors[1], marker='o')
ax_profiles[1].fill_between(x, mean_ppr_set1-sem_ppr_set1, mean_ppr_set1+sem_ppr_set1, color=colors[0], alpha=0.2)
ax_profiles[1].fill_between(x, mean_ppr_set2-sem_ppr_set2, mean_ppr_set2+sem_ppr_set2, color=colors[1], alpha=0.2)
ax_profiles[1].set_ylabel('PPR (An/A1)')
ax_profiles[1].set_xlabel('Pulse number')
ax_profiles[1].set_ylim(0)