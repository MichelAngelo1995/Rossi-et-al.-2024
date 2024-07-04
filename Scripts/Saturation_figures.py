# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:49:54 2023

@author: ruo4037
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


file = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\Saturation_data.xlsx'
file_Ca_trace = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\iGluSnFR_avg_traces_allCa_filtered_3sigma.xlsx"
file_Ca_amps = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx"

####################################
####### SATURATION PLOTS ###########
####################################

traces_25mM = pd.read_excel(file, sheet_name='2.5mMCa_traces')
traces_4mM = pd.read_excel(file, sheet_name='4mMCa_traces')
amps_25mM = pd.read_excel(file, sheet_name='2.5mMCa_amps')
amps_4mM = pd.read_excel(file, sheet_name='4mMCa_amps')

Fmax_25mM = amps_25mM['Fmax']
Fmax_4mM = amps_4mM['Fmax']
list_Fmax = [Fmax_25mM, Fmax_4mM]

norm_amps_25mM = amps_25mM.iloc[:,1:4].div(Fmax_25mM, axis=0)
norm_amps_4mM = amps_4mM.iloc[:,1:4].div(Fmax_4mM, axis=0)


colors = ['green', 'darkred']

fig_fmax, ax_fmax = plt.subplots(figsize=(4,4))
sns.boxplot(data=[Fmax_25mM,Fmax_4mM], ax=ax_fmax, palette=colors, showmeans=True)
s, p = stats.ttest_ind(Fmax_25mM,Fmax_4mM)
ax_fmax.set_ylabel('DF/F0 max')
ax_fmax.set_xticklabels(['2.5mM','4mM'])
ax_fmax.set_ylim(0,8)

# fig_amps, ax_amps = plt.subplots(1,2,tight_layout=True)
# sns.boxplot(data=norm_amps_25mM, ax=ax_amps[0], palette=[colors[0]]*3, showmeans=True)
# sns.boxplot(data=norm_amps_4mM, ax=ax_amps[1], palette=[colors[1]]*3, showmeans=True)
# ax_amps[0].axhline(y=0.8, color='b', ls='--')
# ax_amps[1].axhline(y=0.8, color='b', ls='--')
# ax_amps[0].set_xlim(0,1)


####################################
###### EXAMPLE NORM AMPS PLOT ######
####################################

def func(x,a,b,c):
    return a*np.log(b+x)+c

frequencies = ['20Hz', '50Hz']
for freq in frequencies:  
    sheets = [f'{freq}_2,5mM', f'{freq}_4mM']
    if freq == '20Hz':
        stims_start = np.arange(0.5, 1.0, 0.05)
        stims_stop = np.arange(0.51, 1.01, 0.05)
    elif freq == '50Hz':
        stims_start = np.arange(0.5, 0.7, 0.02)
        stims_stop = np.arange(0.51, 0.71, 0.02)
    
    plt.figure(figsize=(6,4))
    x = np.arange(1,11)
    for frame in range(len(sheets)):
        traces = pd.read_excel(file_Ca_trace, sheet_name=sheets[frame]).drop(['Unnamed: 0'], axis=1)
        amps = pd.read_excel(file_Ca_amps, sheet_name=sheets[frame]).drop(['ID'], axis=1).iloc[:,:10]
        
        EPISODES = []
        for col in range(traces.shape[1]):
            if 'time' in traces.columns[col]:
                time = traces.iloc[:,col].dropna()
            if 'avg' in traces.columns[col]:
                avg = savgol_filter(traces.iloc[:,col].dropna(), 9, 2)
                
                delay_start = 0.005
                delay_end = 0.01
                
                if '20190801_linescan1' in traces.columns[col]:
                    start_fit = np.ravel(np.where(time <= stims_start[0]+0.5-delay_start))[-1]
                    end_fit = np.ravel(np.where(time >= stims_stop[-1]+0.5+delay_end))[0]
                else:
                    start_fit = np.ravel(np.where(time <= stims_start[0]-delay_start))[-1]
                    end_fit = np.ravel(np.where(time >= stims_stop[-1]+delay_end))[0]
                
                # trace_for_fit = avg[start_fit:end_fit]
                # x = np.linspace(time[start_fit], time[end_fit], len(trace_for_fit))
                # popt, pcov = curve_fit(func, x, trace_for_fit, maxfev=5000)
                
                # plt.figure()
                # plt.plot(time,avg)
                # plt.plot(x, func(x, *popt)/list_Fmax[frame].mean(), color=colors[frame])
                
                NORM_PEAKS = []
                for i in range(len(stims_start)):
                    if '20190801_linescan1' in traces.columns[col]:
                        peak_start = np.ravel(np.where(time <= stims_start[i]+0.5))[-1]
                        peak_stop = np.ravel(np.where(time <= stims_stop[i]+0.5))[-1]
                    else:
                        peak_start = np.ravel(np.where(time <= stims_start[i]))[-1]
                        peak_stop = np.ravel(np.where(time <= stims_stop[i]))[-1]
                    
                    peak = avg[peak_start:peak_stop].max()
                    norm_peak = peak/list_Fmax[frame].mean()
                    NORM_PEAKS.append(norm_peak)
                EPISODES.append(NORM_PEAKS)
                
                # plt.plot(x, NORM_PEAKS, color=colors[frame], alpha=0.2)
                
        mean = np.mean(EPISODES, axis=0)
        std = np.std(EPISODES, axis=0)
        # sem = stats.sem(EPISODES, axis=0)
        
        norm_amps = amps/list_Fmax[frame].mean()
        mean_norm_amps = norm_amps.mean()
        sem_norm_amps = stats.sem(norm_amps)
        
        plt.plot(x, mean, color=colors[frame], marker='o', label=sheets[frame])
        plt.fill_between(x, mean+std, mean-std, color=colors[frame], alpha=0.3)
        # plt.fill_between(x, mean+sem, mean-sem, color=colors[frame])
        # plt.plot(x, mean_norm_amps, color='k', marker='o')
        # plt.fill_between(x, mean_norm_amps+sem_norm_amps, mean_norm_amps-sem_norm_amps, color=colors[frame])
        plt.title(freq)
        plt.ylabel('Norm. amplitude')
        plt.xlabel('Pulse number')
    
    plt.axhline(y=0.8, color='b', ls='--', label='0.8 Fmax')
    plt.ylim(0)
    plt.legend()