# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:33:29 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
from sklearn.mixture import GaussianMixture
import scipy.stats as stats


file = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\iGluSnFR_Tau_averages_allCa.xlsx'
sheets = ['20Hz_2.5mM', '50Hz_2.5mM']

TAUS1 = []
TAUS10 = []
for i in range(len(sheets)):
    data = pd.read_excel(f'{file}', sheet_name=sheets[i]).drop(['ID'], axis=1)*1000
    TAU1 = []
    TAU10 = []
    EPISODES = []
    for j in range(data.shape[0]):
        TAUS = []
        for k in range(data.shape[1]):
            if data.iloc[j,k]<100:
                tau = data.iloc[j,k]
            else:
                tau = data.iloc[:,k].median()
            TAUS.append(tau)
        EPISODES.append(TAUS)
        
        if data['TAU1'][j] < 100:
            tau1 = data['TAU1'][j]
            TAU1.append(tau1)
            TAUS1.append(tau1)
        if data['TAU10'][j] < 100:
            tau10 = data['TAU10'][j]
            TAU10.append(tau10)
            TAUS10.append(tau10)
    
    mean = np.mean(EPISODES, axis=0)
    std = np.std(EPISODES, axis=0)
    sem = stats.sem(EPISODES)
    
    x = np.arange(1,11)
    plt.figure()
    plt.plot(x, mean, marker='o')
    plt.fill_between(x, mean+sem, mean-sem, alpha=0.2)
    plt.ylim(0,24)
    plt.title(sheets[i])
    plt.ylabel('Decay-time (ms)')
    plt.xlabel('Pulse number')
    print('n = ' + str(len(EPISODES)))
    
    df = pd.concat((pd.DataFrame(TAU1),
                    pd.DataFrame(TAU10)), axis=1)
    df.columns = ['TAU1', 'TAU10']
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, ax=ax, showmeans=True)
    ax.set_ylabel('Tau (ms)')
    ax.set_title(f'{sheets[i]}')
    
    s,p = stats.mannwhitneyu(TAU1,TAU10)
    print(s,p)


df_total = pd.concat((pd.DataFrame(TAUS1),
                      pd.DataFrame(TAUS10)), axis=1)
df_total.columns = ['TAU1', 'TAU10']

fig_total, ax_total = plt.subplots()
sns.boxplot(data=df_total, ax=ax_total, showmeans=True)
ax.set_ylabel('Tau (ms)')
s_tot, p_tot = stats.mannwhitneyu(TAUS1,TAUS10)
print(s_tot,p_tot)

# TAU = list(itertools.chain.from_iterable(TAU))
# x = np.array(tau1).reshape(-1,1)

# x_mean = np.mean(x)
# x_std = np.std(x)

# gm = GaussianMixture(n_components=1).fit(x)
# gm.means_
# gmm_x = np.linspace(0, max(x), 5000)
# gmm_y = np.exp(gm.score_samples(gmm_x.reshape(-1, 1)))

fig_hist, ax_hist = plt.subplots()
sns.histplot(data=TAUS1, ax=ax_hist, binwidth=1.5, stat='probability')
# ax.plot(gmm_x, gmm_y)
ax_hist.set_xlabel('A1 decay-time (ms)')
