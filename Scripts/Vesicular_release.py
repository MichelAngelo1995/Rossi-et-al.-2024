# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:10:43 2023

@author: ruo4037
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import curve_fit
from statannot import add_stat_annotation


path = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx"
q_fluo = 0.36

file = pd.read_excel(path, sheet_name='20Hz_Amps_no_fail_2,5mMCa')
file_raw_ppr = pd.read_excel(path, sheet_name='20Hz_2,5mM')
target = file['Target']
clusters = file['Cluster']
boutons = file.drop(['ID','Target','A1q 2.5mMCa','Cluster'], axis=1)
N = boutons/q_fluo
cumsum = np.cumsum(N, axis=1)

mean_PC = np.mean([cumsum.iloc[i,:] for i in range(cumsum.shape[0]) if target[i]=='PC'], axis=0)
mean_IN = np.mean([cumsum.iloc[i,:] for i in range(cumsum.shape[0]) if target[i]=='IN'], axis=0)
sem_PC = stats.sem([cumsum.iloc[i,:] for i in range(cumsum.shape[0]) if target[i]=='PC'], axis=0)
sem_IN = stats.sem([cumsum.iloc[i,:] for i in range(cumsum.shape[0]) if target[i]=='IN'], axis=0)

classes = [[cumsum.iloc[i,:] for i in range(len(cumsum)) if clusters[i] == j] for j in np.unique(clusters)]

x = np.arange(1,11)
# plt.figure()
# plt.plot(x, mean_PC, c='r', marker='o')
# plt.plot(x, mean_IN, c='b', marker='o')
# plt.fill_between(x, mean_PC+sem_PC, mean_PC-sem_PC, color='r', alpha=0.2)
# plt.fill_between(x, mean_IN+sem_IN, mean_IN-sem_IN, color='b', alpha=0.2)
# plt.ylim(0,25)
# plt.xlim(0,11)
# for i in range(cumsum.shape[0]):
#     # if target[i] == 'UN':
#     #     plt.plot(x, cumsum.iloc[i,:], c='k', alpha=0.3)
#     if target[i] == 'PC':
#         plt.plot(x, cumsum.iloc[i,:], c='r', alpha=0.3)
#     if target[i] == 'IN':
#         plt.plot(x, cumsum.iloc[i,:], c='b', alpha=0.3)


##### N PC vs IN with failures #######
###########################################

# N_PC = [N['AMP1'][i] for i in range(len(N['AMP1'])) if target[i] == 'PC']
# N_IN = [N['AMP1'][i] for i in range(len(N['AMP1'])) if target[i] == 'IN']
# df_N = pd.concat((pd.DataFrame(N_PC, columns=['N_PC']),
#                   pd.DataFrame(N_IN, columns=['N_IN'])), axis=1)

# fig_N, ax_N = plt.subplots()
# sns.boxplot(data=[df_N['N_PC'], df_N['N_IN']], ax=ax_N, palette=['r','b'], showmeans=True)
# # ax.set_ylim(0,100)


####### A1 failure rate PC vs IN ##########
###########################################

data = pd.read_excel(path, sheet_name='20Hz_2,5mM')
post_cell = data['Target']
fail = data['%Fail1']

fail_PC = [fail[i] for i in range(len(fail)) if post_cell[i] == 'PC']
fail_IN = [fail[i] for i in range(len(fail)) if post_cell[i] == 'IN']
df = pd.concat((pd.DataFrame(fail_PC, columns=['Fail_PC']),
                pd.DataFrame(fail_IN, columns=['Fail_IN'])), axis=1)

fig, ax = plt.subplots()
sns.boxplot(data=df, ax=ax, palette=['grey','magenta'], showmeans=True)
add_stat_annotation(ax, data=df, box_pairs=[('Fail_PC','Fail_IN')], test='Mann-Whitney', text_format='full', verbose=2)
ax.set_ylim(0,100)


##### Cumulative q classes #####
################################

def linereg(x, a, b):
    return a+b*x

def Colors(num):
    clr = []
    cmap = cm.YlGnBu(np.linspace(0.2,1,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr

clr = Colors(len(classes))
xbis = np.arange(0,10)
mean_q_classes = [np.mean(classes[i], axis=0) for i in range(len(classes))]
sem_q_classes = [stats.sem(classes[i], axis=0) for i in range(len(classes))]

plt.figure()
[plt.plot(xbis, mean_q_classes[i], color=clr[i], marker='o', label=f'C{i}') for i in range(len(mean_q_classes))]
[plt.fill_between(xbis, mean_q_classes[i]+sem_q_classes[i], mean_q_classes[i]-sem_q_classes[i], color=clr[i], alpha=0.2) for i in range(len(mean_q_classes))]
plt.ylabel('Cumulative quantal release')
plt.xlabel('Pulse number')
plt.legend()

for i in range(len(mean_q_classes)):
    pars, cov = curve_fit(linereg, xbis[4:], mean_q_classes[i][4:])
    plt.plot(xbis, linereg(xbis, *pars), color=clr[i], ls='--')
    
    print(f'RRP calss {i}: {pars[0]}')
    print(f'Slope calss {i}: {pars[1]}')
    print('----------')
    

##### Pie chart PC vs IN #######
################################

# proportions = [len(ppr_PC)/(len(ppr_PC)+len(ppr_IN)), len(ppr_IN)/(len(ppr_PC)+len(ppr_IN))]
# plt.figure()
# plt.pie(proportions, labels=df.columns, colors=['r','b'], startangle=90, autopct='%.1f%%')