# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:22:57 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
import seaborn as sns
from statannot import add_stat_annotation
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
import uncertainties as unc
import uncertainties.unumpy as unp

Path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files'

file = 'GluSnFR_avg_variables_1.5_4mM_Ca_paired_filtered_3sigma.xlsx'
file_all_Ca = 'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'

freq = ['20Hz', '50Hz']

data_20Hz = pd.read_excel(f'{Path}/{file}', sheet_name=freq[0])
data_50Hz = pd.read_excel(f'{Path}/{file}', sheet_name=freq[1])
datasets = [data_20Hz, data_50Hz]
colors = ['b','k','r']


def linereg(x, a, b):
    return a+b*x

def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))



name = []
all_names = []

all_amps_15, all_amps_4 = [],[]
a1_15, a1_4 = [],[]
fail1_15, fail1_4 = [],[]
ppr_15_20Hz, ppr_4_20Hz = [],[]
ppr_15_50Hz, ppr_4_50Hz = [],[]

for dataset in range(len(datasets)):
    for i in range(datasets[dataset].shape[0]):
        if '1.5mM' in datasets[dataset].iloc[i,0]:
            all_names.append(datasets[dataset].iloc[i,0])
            a1_15.append(datasets[dataset].iloc[i,1])
            fail1_15.append(datasets[dataset]['%Fail1'][i])
            if dataset == 0:
                name.append(datasets[dataset].iloc[i,0])
                all_amps_15.append(datasets[dataset].iloc[i,2:11])
                ppr_15_20Hz.append(datasets[dataset].iloc[i,11])
            elif dataset == 1:
                ppr_15_50Hz.append(datasets[dataset].iloc[i,11])
            # df.loc[len(df.index)] = [datasets[dataset].iloc[i,0], datasets[dataset].iloc[i,1], i, datasets[dataset].iloc[i,11], i]
        elif '4mM' in datasets[dataset].iloc[i,0]:
            a1_4.append(datasets[dataset].iloc[i,1])
            fail1_4.append(datasets[dataset]['%Fail1'][i])
            if dataset == 0:
                all_amps_4.append(datasets[dataset].iloc[i,1:11])
                ppr_4_20Hz.append(datasets[dataset].iloc[i,11])
            elif dataset == 1:
                ppr_4_50Hz.append(datasets[dataset].iloc[i,11])
            # df['AMP1_4mM'].replace({df.iloc[i-df.shape[0],2]: datasets[dataset].iloc[i,1]}, inplace=True)
            # df['PPR_4mM'].replace({df.iloc[i-df.shape[0],4]: datasets[dataset].iloc[i,11]}, inplace=True)


df_a1 = pd.concat((pd.DataFrame(a1_15, columns=['A1_1.5mM']),
                   pd.DataFrame(a1_4, columns=['A1_4mM'])), axis=1)

df_fail = pd.concat((pd.DataFrame(fail1_15, columns=['%Fail1_1.5mM']),
                     pd.DataFrame(fail1_4, columns=['%Fail1_4mM'])), axis=1)

df_ppr = pd.concat((pd.DataFrame(ppr_15_20Hz),
                    pd.DataFrame(ppr_4_20Hz),
                    pd.DataFrame(ppr_15_50Hz),
                    pd.DataFrame(ppr_4_50Hz)), axis=1)
df_ppr.columns = ['PPR_15_20Hz','PPR_4_20Hz','PPR_15_50Hz','PPR_4_50Hz']


######## PAIRS 1.5/4mM Ca ############
######################################

fig, ax = plt.subplots(1,3, figsize=(10,3), tight_layout=True)
ax[1].set_ylim(0,100)
sns.boxplot(data=df_a1, ax=ax[0], palette=[colors[0], colors[2]], showmeans=True)
sns.boxplot(data=df_fail, ax=ax[1], palette=[colors[0], colors[2]], showmeans=True)
sns.boxplot(data=df_ppr, ax=ax[2], palette=[colors[0], colors[2], colors[0], colors[2]], showmeans=True)
add_stat_annotation(data=df_a1, ax=ax[0], box_pairs = [('A1_1.5mM', 'A1_4mM')], test = 'Wilcoxon', text_format='full')
print('-----------')
add_stat_annotation(data=df_fail, ax=ax[1], box_pairs = [('%Fail1_1.5mM', '%Fail1_4mM')], test = 'Wilcoxon', text_format='full')
print('-----------')
add_stat_annotation(data=df_ppr, ax=ax[2], box_pairs = [('PPR_15_20Hz', 'PPR_4_20Hz')], test = 'Wilcoxon', text_format='full')
print('-----------')
add_stat_annotation(data=df_ppr, ax=ax[2], box_pairs = [('PPR_15_50Hz', 'PPR_4_50Hz')], test = 'Wilcoxon', text_format='full')
print('-----------')
print('-----------')


# [ax[0].plot([0,1],[df_a1['A1_1.5mM'][i], df_a1['A1_4mM'][i]], color='k', marker='o') for i in range(df_ppr.shape[0])]
# [ax[1].plot([0,1],[df_ppr['PPR_15_20Hz'][i], df_ppr['PPR_4_20Hz'][i]], color='k', marker='o') for i in range(df_ppr.shape[0])]

# [ax[0].scatter(0, df_a1['A1_1.5mM'][i], color='b', alpha=0.2) for i in range(df_a1.shape[0])]
# [ax[0].scatter(1, df_a1['A1_4mM'][i], color='r', alpha=0.5) for i in range(df_a1.shape[0])]
# [ax[0].plot([0,1], [df_a1['A1_1.5mM'][i], df_a1['A1_4mM'][i]], color='k', alpha=0.5) for i in range(df_a1.shape[0])]
# ax[0].scatter(0, df_a1['A1_1.5mM'].mean(), color='b', marker='*', s=60)
# ax[0].scatter(1, df_a1['A1_4mM'].mean(), color='r', marker='*', s=60)
# ax[0].errorbar(0, df_a1['A1_1.5mM'].mean(), yerr = df_a1['A1_1.5mM'].std(), color='b', capsize=5)
# ax[0].errorbar(1, df_a1['A1_4mM'].mean(), yerr = df_a1['A1_4mM'].std(), color='r', capsize=5)

# [ax[1].scatter(0, df_ppr['PPR_15_20Hz'][i], color='b', alpha=0.2) for i in range(df_ppr['PPR_15_20Hz'].shape[0])]
# [ax[1].scatter(1, df_ppr['PPR_4_20Hz'][i], color='r', alpha=0.5) for i in range(df_ppr['PPR_4_20Hz'].shape[0])]
# [ax[1].plot([0,1], [df_ppr['PPR_15_20Hz'][i], df_ppr['PPR_4_20Hz'][i]], color='k', alpha=0.5) for i in range(df_ppr['PPR_15_20Hz'].shape[0])]
# ax[1].scatter(0, df_ppr['PPR_15_20Hz'].mean(), color='b', marker='*', s=60)
# ax[1].scatter(1, df_ppr['PPR_4_20Hz'].mean(), color='r', marker='*', s=60)
# ax[1].errorbar(0, df_ppr['PPR_15_20Hz'].mean(), yerr = df_ppr['PPR_15_20Hz'].std(), color='b', capsize=5)
# ax[1].errorbar(1, df_ppr['PPR_4_20Hz'].mean(), yerr = df_ppr['PPR_4_20Hz'].std(), color='r', capsize=5)

# [ax[1].scatter(2, df_ppr['PPR_15_50Hz'][i], color='b', alpha=0.2) for i in range(df_ppr['PPR_15_50Hz'].shape[0])]
# [ax[1].scatter(3, df_ppr['PPR_4_50Hz'][i], color='r', alpha=0.5) for i in range(df_ppr['PPR_4_50Hz'].shape[0])]
# [ax[1].plot([2,3], [df_ppr['PPR_15_50Hz'][i], df_ppr['PPR_4_50Hz'][i]], color='k', alpha=0.5) for i in range(df_ppr['PPR_15_50Hz'].shape[0])]
# ax[1].scatter(2, df_ppr['PPR_15_50Hz'].mean(), color='b', marker='*', s=60)
# ax[1].scatter(3, df_ppr['PPR_4_50Hz'].mean(), color='r', marker='*', s=60)
# ax[1].errorbar(2, df_ppr['PPR_15_50Hz'].mean(), yerr = df_ppr['PPR_15_50Hz'].std(), color='b', capsize=5)
# ax[1].errorbar(3, df_ppr['PPR_4_50Hz'].mean(), yerr = df_ppr['PPR_4_50Hz'].std(), color='r', capsize=5)


# sns.boxplot(data=df_a1, ax=ax[0], showmeans=True)
# sns.boxplot(data=[df_ppr['PPR_15_20Hz'],
#                   df_ppr['PPR_4_20Hz'],
#                   df_ppr['PPR_15_50Hz'].dropna(),
#                   df_ppr['PPR_4_50Hz'].dropna()], ax=ax[1], showmeans=True)


# j = 0

# for i in range(len(a1_15)):
#     if j < 10:
#         ax[0].plot([0,1], [a1_15[i], a1_4[i]], marker='o', alpha=0.4, ls='dotted', label=all_names[i])
#     elif 10 <= j < 20:
#         ax[0].plot([0,1], [a1_15[i], a1_4[i]], marker='o', label=all_names[i])
#     elif j == 30:
#         ax[0].plot([0,1], [a1_15[i], a1_4[i]], 'k', marker='o', label=all_names[i])
#     else:
#         ax[0].plot([0,1], [a1_15[i], a1_4[i]], marker='o', ls='dashdot', label=all_names[i])
#     j += 1

# k = 0
# for i in range(df_ppr['PPR_15_20Hz'].shape[0]):
#     if k < 10:
#         ax[1].plot([0,1], [df_ppr['PPR_15_20Hz'][i], df_ppr['PPR_4_20Hz'][i]], marker='o', alpha=0.4, ls='--', label=name[i])
#     elif k == 20:
#         ax[1].plot([0,1], [df_ppr['PPR_15_20Hz'][i], df_ppr['PPR_4_20Hz'][i]], 'k', marker='o', label=name[i])
#     else:
#         ax[1].plot([0,1], [df_ppr['PPR_15_20Hz'][i], df_ppr['PPR_4_20Hz'][i]], marker='o', label=name[i])
#     k += 1
    
# [ax[1].plot([2,3], [df_ppr['PPR_15_50Hz'][i], df_ppr['PPR_4_50Hz'][i]], 'k', marker='o', alpha=0.2) for i in range(df_ppr['PPR_15_50Hz'].shape[0])]

# ax[0].set_xticklabels(['1.5mM','4mM'], rotation=45, ha='right')
# ax[0].set_ylabel('A1 (DF/F)')
# ax[0].set_ylim(0)
# ax[0].legend(loc='upper left', prop={'size':6})
# ax[1].set_xticklabels(['PPR_1.5mM_20Hz','PPR_4mM_20Hz','PPR_1.5mM_50Hz','PPR_4mM_50Hz'], rotation=45, ha='right')
# ax[1].axhline(y=1, color='r', ls='--')
# ax[1].set_ylabel('PPR2/1')
# ax[1].set_ylim(0)
# ax[1].legend(loc='lower right', prop={'size':6})


# add_stat_annotation(data=df_ppr, ax=ax[1], box_pairs = [('PPR_15_20Hz', 'PPR_4_20Hz')], test = 'Wilcoxon', text_format='full')
# add_stat_annotation(data=df_ppr, ax=ax[1], box_pairs = [('PPR_15_50Hz', 'PPR_4_50Hz')], test = 'Wilcoxon', text_format='full')

##### A1 NO FAIL HISTOGRAM ############
#######################################

data_no_fail_a1_15 = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name='Amp1_no_fail_1,5mMCa')
data_no_fail_a1_25 = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name='All_Amp1_no_fail_2,5mMCa')
data_no_fail_a1_4 = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name='Amp1_no_fail_4mMCa')
data_no_fail = [data_no_fail_a1_15['AMP1'], data_no_fail_a1_25['AMP1'], data_no_fail_a1_4['AMP1']]
data_AmpFail_a1_15 = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name='Amp1_AmpFailures_1,5mMCa')
data_ = [data_no_fail[0], data_AmpFail_a1_15['AMP1']]

mean_no_fail = [data_no_fail[i].mean() for i in range(len(data_no_fail))]

fig_no_fail, ax_no_fail = plt.subplots()
for i in range(len(data_no_fail)):   
    sns.histplot(data=data_no_fail[i], binwidth=0.05, color=colors[i], ax=ax_no_fail, stat='probability', alpha=0.5)
ax_no_fail.legend(['1.5mM', '2.5mM', '4mM'])

fig_15, ax_15 = plt.subplots()
print('Histograms mode:')
for frame in range(len(data_)):
    sns.histplot(data=data_[frame], color=colors[:2][frame], binwidth=0.05, ax=ax_15, stat='density')
    gmm = GaussianMixture(n_components=1, covariance_type="full", tol=0.001)
    gmm_x = np.linspace(0, 1, 5000)
    gmm_ = gmm.fit(X=np.expand_dims(data_[frame], 1))
    print(gmm_.means_)
    gmm_y = np.exp(gmm_.score_samples(gmm_x.reshape(-1, 1)))
    ax_15.plot(gmm_x, gmm_y, color=colors[:2][frame], lw=2)
ax_15.legend(['Success', 'Failure'])

skewness_a1 = stats.skew(data_no_fail[0])
skewness_fail = stats.skew(data_[1])
KS_stat, KS_pvalue = stats.ks_2samp(data_[0], data_[1])
print('-----------')
print('-----------')


##### A1 N ###############
##########################

data_success_amps_15_pair = pd.read_excel(f'{Path}/{file}', sheet_name='Amps_no_fail_1,5mMCa')
data_no_fail_a1_4_pair = pd.read_excel(f'{Path}/{file}', sheet_name='Amp1_no_fail_4mMCa')

name_no_fail_20Hz = [data_success_amps_15_pair['ID'][i] for i in range(data_success_amps_15_pair.shape[0]) if '20Hz' in data_success_amps_15_pair['ID'][i]]
a1_no_fail_15mM_20Hz = [data_success_amps_15_pair['AMP1'][i] for i in range(data_success_amps_15_pair.shape[0]) if '20Hz' in data_success_amps_15_pair['ID'][i]]
a1_no_fail_4mM_20Hz = [data_no_fail_a1_4_pair['AMP1'][i] for i in range(data_no_fail_a1_4_pair.shape[0]) if '20Hz' in data_no_fail_a1_4_pair['ID'][i]]

N_4 = [data_no_fail_a1_4_pair['AMP1'][i]/data_success_amps_15_pair['AMP1'][i] for i in range(data_no_fail_a1_4_pair.shape[0])]
mean_N = np.mean(N_4)

N_25 = data_no_fail_a1_25['AMP1']/mean_no_fail[0]
df_q = pd.concat((N_25, pd.DataFrame(N_4)), axis=1)
df_q.columns = ['q 2.5mMCa', 'q 4mMCa']

fig_N, ax_N = plt.subplots()
sns.histplot(data=N_4, binwidth=0.5, color=colors[2], ax=ax_N, stat='probability')
ax_N.set_xlim(0)
ax_N.set_xlabel('A1 quantal release')

fig_N_box, ax_N_box = plt.subplots()
sns.boxplot(data=df_q, ax=ax_N_box, palette=['g','r'], showmeans=True)
add_stat_annotation(data=df_q, ax=ax_N_box, box_pairs = [('q 2.5mMCa', 'q 4mMCa')], test = 'Mann-Whitney', text_format='full')
ax_N_box.set_ylabel('A1 quantal release')
print('-----------')
print('-----------')

# with pd.ExcelWriter(f'{Path}\A1_noFail_2.5sigma.xlsx') as writer:
#     df_q.to_excel(writer)

##### CUMULATIVE PLOT #####
###########################

fig_cumulative_N, ax_cumulative_N = plt.subplots()
x_pulses = np.arange(0,10)
FOLD_4mM = []

amp_15_success = data_success_amps_15_pair.dropna().reset_index().iloc[:,2:]
fold_15mM = np.cumsum(pd.DataFrame([amp_15_success.iloc[:,i]/amp_15_success['AMP1'] for i in range(amp_15_success.shape[1])]).T, axis=1)

for i in range(len(name)):
    for j in range(len(name_no_fail_20Hz)):
        if name[i] == name_no_fail_20Hz[j]:
            amps_no_fail_4mM_20Hz = all_amps_4[i].replace(to_replace = all_amps_4[i][0], value = a1_no_fail_4mM_20Hz[j])
            fold_4mM = np.cumsum(amps_no_fail_4mM_20Hz/a1_no_fail_15mM_20Hz[j]).astype(float)
            FOLD_4mM.append(fold_4mM)
            

fold_15mM_mean = fold_15mM.mean()
fold_15mM_sem = fold_15mM.sem()

fold_4mM_mean = np.mean(FOLD_4mM, axis=0)
fold_4mM_sem = stats.sem(FOLD_4mM, axis=0)

ax_cumulative_N.plot(x_pulses, fold_15mM_mean, 'k', marker='o', label='1.5mM')
ax_cumulative_N.fill_between(x_pulses, fold_15mM_mean+fold_15mM_sem, fold_15mM_mean-fold_15mM_sem, color=colors[1], alpha=0.4)

ax_cumulative_N.plot(x_pulses, fold_4mM_mean, 'r', marker='o', label='4mM')
ax_cumulative_N.fill_between(x_pulses, fold_4mM_mean+fold_4mM_sem, fold_4mM_mean-fold_4mM_sem, color=colors[2], alpha=0.4)

ax_cumulative_N.set_ylabel('Cumulative quantal release')
ax_cumulative_N.set_xlabel('Pulse number')
ax_cumulative_N.set_ylim(0)
ax_cumulative_N.set_title('20Hz')


pars_15, cov_15 = curve_fit(linereg, x_pulses[4:], fold_15mM_mean[4:])
slope_15 = pars_15[1]/50
ax_cumulative_N.plot(x_pulses, linereg(x_pulses, *pars_15), color=colors[0], ls='--')

pars_4, cov_4 = curve_fit(linereg, x_pulses[4:], fold_4mM_mean[4:])
slope_4 = pars_4[1]/50
ax_cumulative_N.plot(x_pulses, linereg(x_pulses, *pars_4), color=colors[2], ls='--')

ax_cumulative_N.legend()

print('RRP size:')
print(f'1.5mM: {pars_15[0]}')
print(f'4mM: {pars_4[0]}')
print('Refilling rate:')
print(f'1.5mM: {slope_15}')
print(f'4mM: {slope_4}')
print('-----------')
print('-----------')


###### STP PROFILES #######
###########################

profiles_20Hz_15mM = [data_20Hz.iloc[i,1:11].astype(float)/data_20Hz.iloc[i,1] for i in range(data_20Hz.shape[0]) if '1.5mM' in data_20Hz.iloc[i,0]]
profiles_20Hz_4mM = [data_20Hz.iloc[i,1:11].astype(float)/data_20Hz.iloc[i,1] for i in range(data_20Hz.shape[0]) if '4mM' in data_20Hz.iloc[i,0]]
profiles_50Hz_15mM = [data_50Hz.iloc[i,1:11].astype(float)/data_50Hz.iloc[i,1] for i in range(data_50Hz.shape[0]) if '1.5mM' in data_50Hz.iloc[i,0]]
profiles_50Hz_4mM = [data_50Hz.iloc[i,1:11].astype(float)/data_50Hz.iloc[i,1] for i in range(data_50Hz.shape[0]) if '4mM' in data_50Hz.iloc[i,0]]
profiles = [profiles_20Hz_15mM, profiles_20Hz_4mM, profiles_50Hz_15mM, profiles_50Hz_4mM]


fig_norm_profile, ax_norm_profile = plt.subplots(1,2, sharey=True, tight_layout=True)
for i in range(len(profiles)):
    mean_profile = np.mean(profiles[i], axis=0)
    sem_profile = stats.sem(profiles[i])
    
    if i < 2:
        [ax_norm_profile[0].plot(x_pulses, profiles[i][j], color=colors[1], lw=2, alpha=0.1) for j in range(len(profiles[i]))]
        ax_norm_profile[0].plot(x_pulses, mean_profile, lw=2, marker='o')
        ax_norm_profile[0].fill_between(x_pulses, mean_profile + sem_profile, mean_profile - sem_profile, alpha=0.5)
    if i >= 2:
        [ax_norm_profile[1].plot(x_pulses, profiles[i][j], color=colors[1], lw=2, alpha=0.1) for j in range(len(profiles[i]))]
        ax_norm_profile[1].plot(x_pulses, mean_profile, lw=2, marker='o')
        ax_norm_profile[1].fill_between(x_pulses, mean_profile + sem_profile, mean_profile - sem_profile, alpha=0.5)
    
    
ax_norm_profile[0].set_title('20Hz')
ax_norm_profile[1].set_title('50Hz')
ax_norm_profile[0].set_xlabel('Pulse number')
ax_norm_profile[0].set_ylabel('PPR (An/A1)')
ax_norm_profile[1].set_xlabel('Pulse number')


##### SEGREGATION BASED ON PPR #######
######################################

# diff_ppr = [df_ppr['PPR_15_20Hz'][i] - df_ppr['PPR_4_20Hz'][i] for i in range(df_ppr['PPR_15_20Hz'].shape[0])]


# fig_seg, ax_seg = plt.subplots(3, 3, figsize=(8,5), sharey='col', sharex='col', tight_layout=True)
# Decreased_PPR_15, Increased_PPR_15 = [],[]
# Decreased_PPR_4, Increased_PPR_4 = [],[]

# for value in range(len(diff_ppr)):
#     if diff_ppr[value] > 0.05:
#         ax_seg[0,0].plot([0,1], [df_ppr['PPR_15_20Hz'][value], df_ppr['PPR_4_20Hz'][value]], marker='o', alpha=0.5)
#         ax_seg[0,1].plot([0,1], [a1_15[:21][value], a1_4[:21][value]], marker='o', alpha=0.5)
#         ax_seg[0,2].plot([0,1], [df_fail['%Fail1_1.5mM'][:21][value], df_fail['%Fail1_4mM'][:21][value]], marker='o', alpha=0.5)
#         ax_seg[0,0].set_ylabel('PPR <')
#         ax_seg[0,1].set_ylabel('A1')
#         ax_seg[0,2].set_ylabel('Failure rate (%)')
#         ax_seg[0,0].set_xticks([0,1])
#         ax_seg[0,0].set_xticklabels(['1.5mM','4mM'])
        
#         Decreased_PPR_15.append(df_ppr['PPR_15_20Hz'][value])
#         Decreased_PPR_4.append(df_ppr['PPR_4_20Hz'][value])
        
        
#     elif diff_ppr[value] < -0.05:
#         ax_seg[1,0].plot([0,1], [df_ppr['PPR_15_20Hz'][value], df_ppr['PPR_4_20Hz'][value]], marker='o', alpha=0.5)
#         ax_seg[1,1].plot([0,1], [a1_15[:21][value], a1_4[:21][value]], marker='o', alpha=0.5)
#         ax_seg[1,2].plot([0,1], [df_fail['%Fail1_1.5mM'][:21][value], df_fail['%Fail1_4mM'][:21][value]], marker='o', alpha=0.5)
#         ax_seg[1,0].set_ylabel('PPR >')
        
#         Increased_PPR_15.append(df_ppr['PPR_15_20Hz'][value])
#         Increased_PPR_4.append(df_ppr['PPR_4_20Hz'][value])

        
#     else:
#         ax_seg[2,0].plot([0,1], [df_ppr['PPR_15_20Hz'][value], df_ppr['PPR_4_20Hz'][value]], marker='o', alpha=0.5)
#         ax_seg[2,1].plot([0,1], [a1_15[:21][value], a1_4[:21][value]], marker='o', alpha=0.5)
#         ax_seg[2,2].plot([0,1], [df_fail['%Fail1_1.5mM'][:21][value], df_fail['%Fail1_4mM'][:21][value]], marker='o', alpha=0.5)
#         ax_seg[2,0].set_ylabel('PPR =')
        
# ax_seg[0,1].set_title('20Hz')



##### CORRELATIONS A1/PPR AND PSYN/PPR #######

x_amp = np.array(a1_15[:21]+a1_4[:21])
x_psyn = np.array(list(df_fail['%Fail1_1.5mM'][:21])+list(df_fail['%Fail1_4mM'][:21]))
x = [x_amp, x_psyn]

y = np.array(list(df_ppr['PPR_15_20Hz'])+list(df_ppr['PPR_4_20Hz']))

n = len(y)
color = 'darkgreen'


def func_reg(x, a, b):
    return a * x + b

def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


fig_correlations, ax_correlations = plt.subplots(1, 2, sharey=True, tight_layout=True)
for i in range(len(x)):
    popt, pcov = curve_fit(func_reg, x[i], y)
    # retrieve parameter values
    a = popt[0]
    b = popt[1]
    print('Optimal Values')
    print('a: ' + str(a))
    print('b: ' + str(b))
    
    # compute r^2
    r2 = 1.0-(sum((y-func_reg(x[i],a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
    print('R^2: ' + str(r2))
    
    # calculate parameter confidence interval
    a,b = unc.correlated_values(popt, pcov)
    print('Uncertainty')
    print('a: ' + str(a))
    print('b: ' + str(b))
    
    # pearson correlation coefficient and p value
    r, p = stats.pearsonr(x[i], y)
    print('Pearson test')
    print('r: ' + str(r))
    print('p: ' + str(p))
    
    # calculate regression confidence interval
    px = np.linspace(np.min(x[i]), np.max(x[i]), len(x[i]))
    py = a*px+b
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    
    lpb, upb = predband(px, x[i], y, popt, func_reg, conf=0.95)


    ax_correlations[i].plot(px, nom, c='black')
    ax_correlations[i].plot(px, nom - 1.96 * std, c='darkorange')
    ax_correlations[i].plot(px, nom + 1.96 * std, c='darkorange')
    ax_correlations[i].plot(px, lpb, 'k--')
    ax_correlations[i].plot(px, upb, 'k--')
    print('-----------')



ax_correlations[0].scatter(a1_15[:21], df_ppr['PPR_15_20Hz'], color=colors[1], alpha=0.5, label='1.5mM')
ax_correlations[0].scatter( a1_4[:21], df_ppr['PPR_4_20Hz'], color=colors[2], alpha=0.5, label='4mM')
ax_correlations[1].scatter(df_fail['%Fail1_1.5mM'][:21], df_ppr['PPR_15_20Hz'], color=colors[1], alpha=0.3, label='1.5mM')
ax_correlations[1].scatter(df_fail['%Fail1_4mM'][:21], df_ppr['PPR_4_20Hz'], color=colors[2], alpha=0.3, label='4mM')

ax_correlations[0].set_ylabel('PPR - 20Hz')
ax_correlations[0].set_xlabel('A1')
ax_correlations[1].set_xlabel('Failure rate for A1')
ax_correlations[0].set_ylim(0)
ax_correlations[0].set_xlim(0)
ax_correlations[1].set_ylim(0)
ax_correlations[1].set_xlim(0)
ax_correlations[0].legend()



##### BOOTSTRAP ON PPR #######

# list_data = [(Decreased_PPR_15,),
#              (Decreased_PPR_4,),
#              (Increased_PPR_15,),
#              (Increased_PPR_4,)]

# res = [stats.bootstrap(list_data[item], np.std, confidence_level=0.95) for item in range(len(list_data))]

# fig_bootstrap, ax_bootstrap = plt.subplots(2,2,tight_layout=True)
# ax_bootstrap[0,0].hist(res[0].bootstrap_distribution, bins=30, color=colors[1])
# ax_bootstrap[0,1].hist(res[1].bootstrap_distribution, bins=30, color=colors[2])
# ax_bootstrap[1,0].hist(res[2].bootstrap_distribution, bins=30, color=colors[1])
# ax_bootstrap[1,1].hist(res[3].bootstrap_distribution, bins=30, color=colors[2])
# ax_bootstrap[1,0].set_xlabel('Statistic value')
# ax_bootstrap[0,0].set_ylabel('Frequency (Decreasing PPR)')
# ax_bootstrap[1,0].set_ylabel('Frequency (Increasing PPR)')
# ax_bootstrap[0,0].set_title('1.5mM')
# ax_bootstrap[0,1].set_title('4mM')