# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:48:51 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
from statannot import add_stat_annotation
import scikit_posthocs as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy import signal
from scipy.interpolate import interp1d
from itertools import combinations
import dabest


Path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files'

File_clusters = 'PCA_HCPC_6clusters_2.5mM_20Hz_3sigma.xlsx'
File_avg_traces = 'iGluSnFR_avg_traces_allCa_filtered_3sigma.xlsx'
sheet = '20Hz_2,5mM'
freq = '20Hz'


def Colors(num):
    clr = []
    cmap = cm.YlGnBu(np.linspace(0.2,1,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr


def func_mono_exp(x, a, b, c, d):
    return a * np.exp(-(x-b)/c) + d


def Fit_single_trace(time, trace, start, end):
    '''
    Computes an exponential fit on a window of a dataset values.
    The dataset has to be an excel file with variables as columns.
    
    time : time variable.
    trace : the trace the fit must be applied on.
    x_start : first limit of the window.
    x_end : second limit of the window.

    Returns
    -------
    popt : array
        optimal values for the parameters
    idx_start : int
        the first index of the window.
    idx_stop : int
        the second index of the window.

    '''
    
    idx_start = np.ravel(np.where(time >= start))[0]
    idx_stop = np.ravel(np.where(time <= end))[-1]
    
    x = time[idx_start:idx_stop]
    y = trace[idx_start:idx_stop]
    x2 = np.array(np.squeeze(x))
    y2 = np.array(np.squeeze(y))  
    
    param_bounds=([-np.inf,0.,0.,-1000.],[np.inf,1.,10.,1000.])      # be careful ok for seconds. If millisec change param 2 and 3
    popt, pcov = curve_fit(func_mono_exp, x2, y2, bounds=param_bounds, maxfev=10000) 
        
    bleach = func_mono_exp(time, *popt)
    
    return bleach
    


data = pd.read_excel(f'{Path}/{File_clusters}', index_col=0).reset_index()
clusters = data.iloc[:,-2].astype(int)
labels = Counter(clusters).keys()
clr = Colors(len(labels))
q = data['q'].dropna()
# targets = data.iloc[:,-2]


ppr_idx1 = list(data.columns).index('PPR2/1')
ppr_idx2 = list(data.columns).index('PPR10/1')
ppr = pd.concat((pd.DataFrame(np.ones(data.shape[0])), data.iloc[:,ppr_idx1:ppr_idx2+1]), axis=1)


traces = pd.read_excel(f'{Path}/{File_avg_traces}', sheet_name=sheet).transpose()
traces_noFback = pd.read_excel(f'{Path}/{File_avg_traces}', sheet_name='20Hz_2,5mM_noFback').transpose()

if freq == '20Hz':
    new_time_exp = np.linspace(0., 1., num=800, endpoint=True)
    new_time_trace = np.linspace(0., 0.62, num=800, endpoint=True)
if freq == '50Hz':
    new_time = np.linspace(0., 0.49, num=490, endpoint=True)


AVG_TRACES, TIME_TRACES = [],[]
for idx in range(traces.shape[0]):
    if 'time' in traces.index[idx]:
        if '20190801' in traces.index[idx]:
            start = np.ravel(np.where(traces.iloc[idx,:] >= 0.9))[0]
            end = np.ravel(np.where(traces.iloc[idx,:] <= 1.7))[-1]
        else:
            if freq == '20Hz':
                start = np.ravel(np.where(traces.iloc[idx,:] >= 0.4))[0] 
                end = np.ravel(np.where(traces.iloc[idx,:] <= 1.2))[-1] 
            
            if freq == '50Hz':
                start = np.ravel(np.where(traces.iloc[idx,:] >= 0.4))[0]
                end = np.ravel(np.where(traces.iloc[idx,:] <= 0.9))[-1]
        
        time_exp = np.array(traces.iloc[idx,0:end])
        time_trace = np.array(traces.iloc[idx,start:end])
        time_trace = time_trace-time_trace[0]
        # TIME_TRACES.append(new_time)
        
    elif 'avg' in traces.index[idx]:
                
        trace_trace = traces.iloc[idx,start:end]
        trace_trace = signal.savgol_filter(trace_trace, 9, 2)
        f_trace = interp1d(time_trace, trace_trace)
        y_trace = f_trace(new_time_trace)
        AVG_TRACES.append(y_trace)


EXP = []
for idx in range(traces_noFback.shape[0]):
    if 'time' in traces_noFback.index[idx]:
        if '20190801' in traces_noFback.index[idx]:
            start = np.ravel(np.where(traces_noFback.iloc[idx,:] >= 0.9))[0]
            end = np.ravel(np.where(traces_noFback.iloc[idx,:] <= 1.7))[-1]
        else:
            if freq == '20Hz':
                start = np.ravel(np.where(traces_noFback.iloc[idx,:] >= 0.4))[0] 
                end = np.ravel(np.where(traces_noFback.iloc[idx,:] <= 1.2))[-1] 
            
            if freq == '50Hz':
                start = np.ravel(np.where(traces_noFback.iloc[idx,:] >= 0.4))[0]
                end = np.ravel(np.where(traces_noFback.iloc[idx,:] <= 0.9))[-1]
        
        time_exp = np.array(traces_noFback.iloc[idx,0:end])
        time_trace = np.array(traces_noFback.iloc[idx,start:end])
        time_trace = time_trace-time_trace[0]
        
    if 'avg' in traces_noFback.index[idx]:
        # TRACES WITH EXP FIT PHOTOBLEACHING
        trace_exp = traces_noFback.iloc[idx,0:end]
        trace_exp = signal.savgol_filter(trace_exp.dropna(), 9, 2)
        f_exp = interp1d(time_exp, trace_exp)
        y_exp = f_exp(new_time_exp)
        exp_bleach = Fit_single_trace(new_time_exp, y_exp, 0., 0.49)
        norm_exp = exp_bleach/max(exp_bleach)
        EXP.append(norm_exp)
        
        # plt.figure()
        # plt.plot(new_time_exp, y_exp, color='k')
        # plt.plot(new_time_exp, exp_bleach, color='r')
        # plt.title(f'{traces.index[idx]}')
    

# plt.figure()
# mean_traces = np.mean(AVG_TRACES, axis=0)
# std_traces = np.std(AVG_TRACES, axis=0)
# plt.plot(new_time_exp, mean_traces, 'r')
# plt.fill_between(new_time_exp, mean_traces+std_traces, mean_traces-std_traces, color='r', alpha=0.2)


fig, ax = plt.subplots(1,2, figsize=(8, 4), tight_layout=True)
fig_bleach, ax_bleach = plt.subplots(figsize=(6, 4), tight_layout=True)
x = np.arange(1, ppr.shape[1]+1, 1)

AMP1, AMP2 = [],[]
PPR2_1, PPR3_1, PPR3_2 = [],[],[]
FAIL1, FAIL2 = [],[]
Q = []

for i in range(len(labels)):
    amp1 = [data['AMP1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    amp2 = [data['AMP2'][j] for j in range(data.shape[0]) if clusters[j] == i]
    ppr2_1 = [data['PPR2/1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    ppr3_1 = [data['PPR3/1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    ppr3_2 = [data['AMP3'][j]/data['AMP2'][j] for j in range(data.shape[0]) if clusters[j] == i]
    fail1 = [data['%Fail1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    fail2 = [data['%Fail2'][j] for j in range(data.shape[0]) if clusters[j] == i]
    
    AMP1.append(amp1)
    AMP2.append(amp2)
    PPR2_1.append(ppr2_1)
    PPR3_1.append(ppr3_1)
    PPR3_2.append(ppr3_2)
    FAIL1.append(fail1)
    FAIL2.append(fail2)
    
    
    
    episodes = [ppr.iloc[j,:] for j in range(ppr.shape[0]) if clusters[j] == i]
    mean = np.mean(episodes, axis=0)
    # std = np.std(episodes, axis=0)
    std = stats.sem(episodes, axis=0)
    
    group_avg_traces = [AVG_TRACES[j] for j in range(len(AVG_TRACES)) if clusters[j] == i]
    mean_group_avg = np.mean(group_avg_traces, axis=0)
    # std_group_avg = np.std(group_avg_traces, axis=0)
    std_group_avg = stats.sem(group_avg_traces, axis=0)
    
    
    group_exp = [EXP[e] for e in range(len(EXP)) if clusters[e] == i]
    mean_group_exp = np.mean(group_exp, axis=0)
    std_group_exp = stats.sem(group_exp, axis=0)
    bleach_50 = 1-((1-min(mean_group_exp))/2)
    print(f'50% exp curve for C{i}: {bleach_50}')
    
    
    q_cluster = [q[j] for j in q.index if clusters[j] == i]
    Q.append(q_cluster)
    # df_q[f'C{i}'] = pd.Series(q_cluster)
    
    
    ax[0].plot(x, mean, color=clr[i], marker='o')
    ax[0].fill_between(x, mean+std, mean-std, color=clr[i], alpha=0.3)
    
    ax[1].plot(new_time_trace, mean_group_avg, color=clr[i])
    ax[1].fill_between(new_time_trace, mean_group_avg+std_group_avg, mean_group_avg-std_group_avg, color=clr[i], alpha=0.3)
    
    ax_bleach.plot(new_time_exp, mean_group_exp, color=clr[i])
    ax_bleach.fill_between(new_time_exp, mean_group_exp+std_group_exp, mean_group_exp-std_group_exp, color=clr[i], alpha=0.3)
    
    # sns.histplot(data=amp1, bins=10, binwidth=0.05, ax=ax[2], color=clr[i], stat='density', kde=True)
    
    normality_amp1 = stats.shapiro(amp1)
    normality_ppr = stats.shapiro(ppr2_1)
    normality_fail = stats.shapiro(fail1)
    print(f'NORMALITY P-VALUE CLUSTER{i}')
    print('----------------------------')
    print(f'Amp1: {normality_amp1[1]}')
    print(f'PPR: {normality_ppr[1]}')
    print(f'Fail: {normality_fail[1]}')
    print('----------------------------')
    
    # homogeneity_amp1 = stats.levene(amp1)
    # homogeneity_ppr = stats.shapiro(ppr2_1)
    # homogeneity_fail = stats.shapiro(fail1)
    # print('NORMALITY P-VALUE')
    # print(f'Amp1: {normality_amp1[1]}')
    # print(f'PPR: {normality_ppr[1]}')
    # print(f'Fail: {normality_fail1[1]}')

columns = sorted([f'C{i}' for i in labels])
df_amp1 = pd.DataFrame(AMP1, index=columns).T
df_amp2 = pd.DataFrame(AMP2, index=columns).T
df_ppr2_1 = pd.DataFrame(PPR2_1, index=columns).T
df_ppr3_1 = pd.DataFrame(PPR3_1, index=columns).T
df_ppr3_2 = pd.DataFrame(PPR3_2, index=columns).T
df_fail1 = pd.DataFrame(FAIL1, index=columns).T
df_fail2 = pd.DataFrame(FAIL2, index=columns).T
df_q = pd.DataFrame(Q, index=columns).T

    
ax[0].set_title('Profiles')
ax[0].set_ylabel('An/A1')
ax[0].set_xlabel('Number stim')
ax[1].set_title('Avg traces')
ax[1].set_ylabel('DF/F0')
ax[1].set_xlabel('Time (sec)')
ax_bleach.set_title('Bleaching')
ax_bleach.set_ylabel('Norm. fluorescence')
ax_bleach.set_xlabel('Time (sec)')
ax_bleach.set_ylim(0,1)

fig_pie, ax_pie = plt.subplots(figsize=(3,3))
proportions = [len(df_amp1.iloc[:,i].dropna()) for i in range(df_amp1.shape[1])]
profiles = [col for col in df_amp1.columns]
plt.pie(proportions, labels=profiles, colors=clr, startangle=90, autopct='%.1f%%')

pairs = list(combinations(df_ppr2_1.columns,2))
stat_test = 'Mann-Whitney'

#########################
#### TRACES FIGURES #####
#########################

fig_box, ax_box = plt.subplots(1,6, figsize=(14, 4), tight_layout=True)

### BOXPLOT AMP1+AMP2 ####

sns.boxplot(data=df_amp1, ax=ax_box[0], palette=clr, showmeans=True)
add_stat_annotation(ax_box[0], data=df_amp1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_amp1['C0'].dropna(), df_amp1['C1'].dropna(), df_amp1['C2'].dropna(), df_amp1['C3'].dropna(), df_amp1['C4'].dropna(), df_amp1['C5'].dropna())
# sp.posthoc_dunn([df_amp1['C0'], df_amp1['C1'].dropna(), df_amp1['C2'].dropna(), df_amp1['C3'].dropna(), df_amp1['C4'].dropna()], p_adjust='bonferroni')
ax_box[0].set_title('Amp peak1')
ax_box[0].set_ylabel('A1 (DF/F)')
ax_box[0].set_xticklabels(df_amp1.columns)

sns.boxplot(data=df_amp2, ax=ax_box[1], palette=clr, showmeans=True)
add_stat_annotation(ax_box[1], data=df_amp2, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_amp2['C0'].dropna(), df_amp2['C1'].dropna(), df_amp2['C2'].dropna(), df_amp2['C3'].dropna(), df_amp2['C4'].dropna(), df_amp2['C5'].dropna())
# sp.posthoc_dunn([df_amp2['C0'], df_amp2['C1'].dropna(), df_amp2['C2'].dropna(), df_amp2['C3'].dropna(), df_amp2['C4'].dropna()], p_adjust='bonferroni')
ax_box[1].set_title('Amp peak2')
ax_box[1].set_ylabel('A2 (DF/F)')
ax_box[1].set_xticklabels(df_amp2.columns)


# fig_comp_amps, ax_comp_amps = plt.subplots()
# sns.boxplot(data=[df_amp1['C0'], df_amp2['C0'],
#                   df_amp1['C1'], df_amp2['C1'],
#                   df_amp1['C2'], df_amp2['C2'],
#                   df_amp1['C3'], df_amp2['C3']], ax=ax_comp_amps, palette=[clr[0],clr[0],
#                                                                             clr[1],clr[1],
#                                                                             clr[2],clr[2],
#                                                                             clr[3],clr[3]], showmeans=True)
# ax_comp_amps.set_ylabel('Amplitude (dF/F)')
# ax_comp_amps.set_ylim(0)

# stats.mannwhitneyu(df_amp1['C0'], df_amp2['C0'])
# stats.mannwhitneyu(df_amp1['C1'].dropna(), df_amp2['C1'].dropna())
# stats.mannwhitneyu(df_amp1['C2'].dropna(), df_amp2['C2'].dropna())
# stats.mannwhitneyu(df_amp1['C3'].dropna(), df_amp2['C3'].dropna())



##### BOXPLOT PPR2/1 ####

sns.boxplot(data=df_ppr2_1, ax=ax_box[2], palette=clr, showmeans=True)
add_stat_annotation(ax_box[2], data=df_ppr2_1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_ppr2_1['C0'].dropna(), df_ppr2_1['C1'].dropna(), df_ppr2_1['C2'].dropna(), df_ppr2_1['C3'].dropna(), df_ppr2_1['C4'].dropna(), df_ppr2_1['C5'].dropna())
# sp.posthoc_dunn([df_ppr2_1['C0'], df_ppr2_1['C1'].dropna(), df_ppr2_1['C2'].dropna(), df_ppr2_1['C3'].dropna(), df_ppr2_1['C4'].dropna()], p_adjust='bonferroni')
ax_box[2].set_title('PPR2/1')
ax_box[2].set_ylabel('A2/A1')
ax_box[2].set_xticklabels(df_ppr2_1.columns)

sns.boxplot(data=df_ppr3_1, ax=ax_box[3], palette=clr, showmeans=True)
add_stat_annotation(ax_box[3], data=df_ppr3_1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_ppr3_1['C0'].dropna(), df_ppr3_1['C1'].dropna(), df_ppr3_1['C2'].dropna(), df_ppr3_1['C3'].dropna(), df_ppr3_1['C4'].dropna(), df_ppr3_1['C5'].dropna())
# sp.posthoc_dunn([df_ppr3_1['C0'], df_ppr3_1['C1'].dropna(), df_ppr3_1['C2'].dropna(), df_ppr3_1['C3'].dropna(), df_ppr3_1['C4'].dropna()], p_adjust='bonferroni')

ax_box[3].set_title('PPR3/1')
ax_box[3].set_ylabel('A3/A1')
ax_box[3].set_xticklabels(df_ppr3_1.columns)


##### BOXPLOT PPR3/2 ####

fig_ppr32, ax_ppr32 = plt.subplots(tight_layout=True)
sns.boxplot(data=df_ppr3_2, ax=ax_ppr32, palette=clr, showmeans=True)
add_stat_annotation(ax_ppr32, data=df_ppr3_2, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_ppr3_2['C0'], df_ppr3_2['C1'].dropna(), df_ppr3_2['C2'].dropna(), df_ppr3_2['C3'].dropna(), df_ppr3_2['C4'].dropna(), df_ppr3_2['C5'].dropna())
ax_ppr32.set_title('PPR3/2')
ax_ppr32.set_ylabel('A3/A2')
ax_ppr32.set_xticklabels(df_ppr3_2.columns)

## BOXPLOT FAIL1+FAIL2 ##

sns.boxplot(data=df_fail1, ax=ax_box[4], palette=clr, showmeans=True)
add_stat_annotation(ax_box[4], data=df_fail1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_fail1['C0'].dropna(), df_fail1['C1'].dropna(), df_fail1['C2'].dropna(), df_fail1['C3'].dropna(), df_fail1['C4'].dropna(), df_fail1['C5'].dropna())
# sp.posthoc_dunn([df_fail1['C0'], df_fail1['C1'].dropna(), df_fail1['C2'].dropna(), df_fail1['C3'].dropna(), df_fail1['C4'].dropna()], p_adjust='bonferroni')
ax_box[4].set_title('Psyn peak1')
ax_box[4].set_ylabel('A1 Psyn')
ax_box[4].set_xticklabels(df_fail1.columns)

sns.boxplot(data=df_fail2, ax=ax_box[5], palette=clr, showmeans=True)
add_stat_annotation(ax_box[5], data=df_fail2, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
stats.kruskal(df_fail2['C0'].dropna(), df_fail2['C1'].dropna(), df_fail2['C2'].dropna(), df_fail2['C3'].dropna(), df_fail2['C4'].dropna(), df_fail2['C5'].dropna())
# sp.posthoc_dunn([df_fail2['C0'], df_fail2['C1'].dropna(), df_fail2['C2'].dropna(), df_fail2['C3'].dropna(), df_fail2['C4'].dropna()], p_adjust='bonferroni')
ax_box[5].set_title('Psyn peak2')
ax_box[5].set_ylabel('A2 Psyn')
ax_box[5].set_xticklabels(df_fail2.columns)



# plt.figure()
# for item in range(data.shape[0]):
#     if '20220425_linescan1' in data.iloc[item,0]:
#         plt.plot(x, ppr.iloc[item,:], marker='o')
# plt.ylim(0,4)


# fig_comp_psucc, ax_comp_psucc = plt.subplots()
# sns.boxplot(data=[df_fail1['C0'], df_fail2['C0'],
#                   df_fail1['C1'], df_fail2['C1'],
#                   df_fail1['C2'], df_fail2['C2'],
#                   df_fail1['C3'], df_fail2['C3']], ax=ax_comp_psucc, palette=[clr[0],clr[0],
#                                                                               clr[1],clr[1],
#                                                                               clr[2],clr[2],
#                                                                               clr[3],clr[3]], showmeans=True)
# ax_comp_psucc.set_ylabel('P success')
# ax_comp_psucc.set_ylim(0,1.1)

# stats.mannwhitneyu(df_fail1['C0'], df_fail2['C0'])
# stats.mannwhitneyu(df_fail1['C1'].dropna(), df_fail2['C1'].dropna())
# stats.mannwhitneyu(df_fail1['C2'].dropna(), df_fail2['C2'].dropna())
# stats.mannwhitneyu(df_fail1['C3'].dropna(), df_fail2['C3'].dropna())

##### BOXPLOT Q CLUSTERS #####
print('------------------')
print('QUANTAL RELEASE')
print('------------------')
fig_box_q, ax_box_q = plt.subplots()
sns.boxplot(data=df_q, ax=ax_box_q, palette=clr, showmeans=True)
stats.kruskal(df_q['C0'].dropna(), df_q['C1'].dropna(), df_q['C2'].dropna(), df_q['C3'].dropna(), df_q['C4'].dropna(), df_q['C5'].dropna())
add_stat_annotation(ax_box_q, data=df_q, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
ax_box_q.set_ylabel('A1 quantal release')

## POLAR PLOT ##

# def _invert(x, limits):
#     """inverts a value x on a scale from
#     limits[0] to limits[1]"""
#     return limits[1] - (x - limits[0])

# def _scale_data(data, ranges):
#     """scales data[1:] to ranges[0],
#     inverts if the scale is reversed"""
#     for d, (y1, y2) in zip(data[1:], ranges[1:]):
#         assert (y1 <= d <= y2) or (y2 <= d <= y1)
#     x1, x2 = ranges[0]
#     d = data[0]
#     if x1 > x2:
#         d = _invert(d, (x1, x2))
#         x1, x2 = x2, x1
#     sdata = [d]
#     for d, (y1, y2) in zip(data[1:], ranges[1:]):
#         if y1 > y2:
#             d = _invert(d, (y1, y2))
#             y1, y2 = y2, y1
#         sdata.append((d-y1) / (y2-y1) 
#                      * (x2 - x1) + x1)
#     return sdata

# class ComplexRadar():
#     def __init__(self, fig, variables, ranges,
#                  n_ordinate_levels=6):
#         angles = np.arange(0, 360, 360./len(variables))

#         axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True,
#                 label = "axes{}".format(i)) 
#                 for i in range(len(variables))]
#         l, text = axes[0].set_thetagrids(angles, 
#                                          labels=variables)
#         [txt.set_rotation(angle-90) for txt, angle 
#              in zip(text, angles)]
#         for ax in axes[1:]:
#             ax.patch.set_visible(False)
#             ax.grid("off")
#             ax.xaxis.set_visible(False)
#         for i, ax in enumerate(axes):
#             grid = np.linspace(*ranges[i], 
#                                num=n_ordinate_levels)
#             gridlabel = ["{}".format(round(x,2)) 
#                          for x in grid]
#             if ranges[i][0] > ranges[i][1]:
#                 grid = grid[::-1] # hack to invert grid
#                           # gridlabels aren't reversed
#             gridlabel[0] = "" # clean up origin
#             ax.set_rgrids(grid, labels=gridlabel,
#                          angle=angles[i])
#             #ax.spines["polar"].set_visible(False)
#             ax.set_ylim(*ranges[i])
#         # variables for plotting
#         self.angle = np.deg2rad(np.r_[angles, angles[0]])
#         self.ranges = ranges
#         self.ax = axes[0]
#     def plot(self, data, *args, **kw):
#         sdata = _scale_data(data, self.ranges)
#         self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
#     def fill(self, data, *args, **kw):
#         sdata = _scale_data(data, self.ranges)
#         self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


# variables = ('A1(DF/F)', 'F1(%)', 'PPR')
# ranges = [(0., 1.), (0., 100), (0., 4.)]            

# fig1 = plt.figure(figsize=(6, 6))
# radar = ComplexRadar(fig1, variables, ranges)
# for i in range(len(clr)):
#     data = (np.mean(df_amp1)[i], np.mean(df_fail1)[i], np.mean(df_ppr2_1)[i])
#     radar.plot(data)
#     radar.fill(data, alpha=0.2)


#############################################
############## 3DPLOT PARAMETERS ############
#############################################

# ax = plt.axes(projection='3d')
# for i in range(len(clr)):
#     data = (np.mean(df_amp1)[i], np.mean(df_fail1)[i], np.mean(df_ppr2_1)[i])
#     x = [data[0],0,0]
#     y = [0,0,data[1]]
#     z = [0,data[2],0]
#     ax.plot(x, z, y, color=clr[i])
#     verts = [np.c_[x,z,y]]
#     ax.add_collection3d(Poly3DCollection(verts, facecolors=clr[i], alpha=0.2))
# ax.set_zlim(0,1)
# ax.set_xlabel('A1 DF/F')
# ax.set_ylabel('PPR2/1')
# ax.set_zlabel('P syn')