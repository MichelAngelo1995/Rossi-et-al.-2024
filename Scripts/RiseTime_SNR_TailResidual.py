# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:47:41 2023

@author: ruo4037
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats

Path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files'
file_amplitudes = 'GluSnFR_avg_variables_allCa_filtered_3sigma'
file_traces = 'iGluSnFR_avg_traces_allCa_filtered_3sigma'


data_amps_20Hz = pd.read_excel(f'{Path}\{file_amplitudes}.xlsx', sheet_name='20Hz_2,5mM')
data_amps_50Hz = pd.read_excel(f'{Path}\{file_amplitudes}.xlsx', sheet_name='50Hz_2,5mM')
data_amps = [data_amps_20Hz, data_amps_50Hz]

data_traces_20Hz = pd.read_excel(f'{Path}\{file_traces}.xlsx', sheet_name='20Hz_2,5mM').iloc[:,1:]
data_traces_50Hz = pd.read_excel(f'{Path}\{file_traces}.xlsx', sheet_name='50Hz_2,5mM').iloc[:,1:]
data_traces = [data_traces_20Hz, data_traces_50Hz]


SNR = []
RISE_TIME = []
ALL_NOISE = []
Fmax_25mM_20perc = 3.78*0.2
for frame in range(len(data_amps)):
    fig_tails, ax_tails = plt.subplots(tight_layout=True)
    
    amp1 = data_amps[frame]['AMP1']
    noise_std = data_amps[frame]['Baseline_std']
    for i in range(len(amp1)):
        snr = amp1[i]/noise_std[i] 
        SNR.append(snr)

    
    NAME, NOISE_MEAN, NOISE_STD = [],[],[]
    TAIL = []
    for col in range(data_traces[frame].shape[1]):
        if 'time' in data_traces[frame].columns[col]:
            time = data_traces[frame].iloc[:,col].dropna()
        if 'avg' in data_traces[frame].columns[col]:
            name = data_traces[frame].columns[col].rsplit('_',1)[0]
            average = savgol_filter(data_traces[frame].iloc[:,col].dropna(), 9, 2)
            
            if '20190801' in data_traces[frame].columns[col]:
                start = np.ravel(np.where(time >= 0.995))[0]
                stop = np.ravel(np.where(time >= 1.01))[0]
            else:
                start = np.ravel(np.where(time >= 0.495))[0]
                stop = np.ravel(np.where(time >= 0.51))[0]
            
            peak_idx = time[time[start:stop].index[average[start:stop].argmax()]]
            time_window = peak_idx - time[start]
            rise_time = ((time_window*0.9) - (time_window*0.1))*1000
            RISE_TIME.append(rise_time)
            
            # print(name)
            # print(rise_time)
            
            
            # plt.figure()
            # plt.plot(time, average, color='b')
            # plt.axvline(x=time[start], color='g', ls='--')
            # plt.axvline(x=peak_idx, color='g', ls='--')
            # plt.axhline(y=average[start:stop].max(), color='r', ls='--')
            # plt.title(name)
            
        
            
            if '20190801' in data_traces[frame].columns[col]:
                start_bsl = np.ravel(np.where(time >= 0.88))[0]
                stop_bsl = np.ravel(np.where(time <= 0.98))[-1]
                
                tail_start = np.arange(1.035, 1.535, 0.05)
                tail_stop = np.arange(1.045, 1.545, 0.05)
                
            else:
                start_bsl = np.ravel(np.where(time >= 0.38))[0]
                stop_bsl = np.ravel(np.where(time <= 0.48))[-1]
                
                if frame == 0:
                    tail_start = np.arange(0.535, 1.035, 0.05)
                    tail_stop = np.arange(0.545, 1.045, 0.05)
                if frame == 1:
                    tail_start = np.arange(0.512, 0.694, 0.02)
                    tail_stop = np.arange(0.517, 0.699, 0.02)
            
            noise_sd = np.std(average[start_bsl:stop_bsl])
            noise_mean = np.mean(average[start_bsl:stop_bsl])
            name = data_traces[frame].columns[col].rsplit('_',1)[0]
            NAME.append(name)
            NOISE_STD.append(noise_sd)
            NOISE_MEAN.append(noise_mean)
            ALL_NOISE.append(noise_mean)
            
            tail_mean = [np.mean(average[np.ravel(np.where(time <= tail_start[i]))[-1]:np.ravel(np.where(time <= tail_stop[i]))[-1]]) for i in range(len(tail_start))]
            z_score = (tail_mean-noise_mean)/noise_sd
            TAIL.append(tail_mean)
            
            
            # plt.figure()
            # plt.plot(time, average)
            # plt.axvline(x=time[start_bsl], color='r', ls='--')
            # plt.axvline(x=time[stop_bsl], color='r', ls='--')
            # plt.title(name)
            # [plt.axvline(x=tail_start[i], color='r', ls='--') for i in range(len(tail_start))]
            # [plt.axvline(x=tail_stop[i], color='r', ls='--') for i in range(len(tail_stop))]
            
    
    df_tail = pd.concat((pd.DataFrame(NOISE_MEAN, columns=['Noise_mean']),
                         pd.DataFrame(NOISE_STD, columns=['Noise_std']),
                         pd.DataFrame(TAIL, columns=[f'Tail{i+1}' for i in range(len(tail_start))])), axis=1)
    df_mean = df_tail.iloc[:,2:].mean()
    df_std = stats.sem(df_tail.iloc[:,2:])
    label = ['20Hz', '50Hz']
    print(df_tail.shape)
    
    ax_tails.plot(df_mean, color='k', marker='o')
    ax_tails.fill_between(np.arange(0, df_tail.iloc[:,2:].shape[1]), df_mean+df_std, df_mean-df_std, color='r', interpolate=True)
    ax_tails.axhline(y=Fmax_25mM_20perc, color='r', ls='--')
    ax_tails.set_title(f'{label[frame]}')
    ax_tails.set_ylabel('DF/F')
    ax_tails.set_ylim(0,1.2)
    print(df_tail.mean())
    
    # [ax_tails.plot(df_tail.iloc[i,2:], color='k', marker='o', alpha=0.2) for i in range(df_tail.shape[0])]
    
    # ax_tails.fill_between(np.arange(1, df_tail.iloc[:,2:].shape[1]+1), df_mean, color='k', interpolate=True)
    # ax_tails.fill_between(np.arange(1, df_tail.iloc[:,2:].shape[1]+1), df_mean, 3, where=df_mean >= 3, color='r', interpolate=True)
    
    
    # fig_noise, ax_noise = plt.subplots()
    # sns.histplot(data=NOISE_MEAN, ax=ax_noise, binwidth=0.002, stat='probability')
    # ax_noise.set_title(f'{label[frame]}')
    
    # df = pd.concat((pd.DataFrame(NAME, columns=['ID']), pd.DataFrame(NOISE, columns=['Noise'])), axis=1)
    # with pd.ExcelWriter(r'{}\Noise_STD_{}.xlsx'.format(Path, frame)) as writer:
    #     df.to_excel(writer)
    
fig, ax = plt.subplots(1,2,tight_layout=True)
sns.histplot(data=SNR, ax=ax[0], binwidth=1.5, stat='probability')
sns.histplot(data=RISE_TIME, ax=ax[1], binwidth=1, stat='probability')
ax[0].set_xlabel('SNR')
ax[0].set_xlim(0)
ax[1].set_xlabel('Rise time (ms)')
ax[1].set_xlim(0,15)

fig_all_noise, ax_all_noise = plt.subplots()
sns.histplot(data=ALL_NOISE, ax=ax_all_noise, binwidth=0.001, stat='probability')
ax_all_noise.set_ylim(0,.25)
ax_all_noise.set_xlim(-.12,.02)