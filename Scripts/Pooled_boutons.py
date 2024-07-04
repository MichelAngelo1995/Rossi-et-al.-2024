# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:20:19 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from statannot import add_stat_annotation
import scikit_posthocs as sp
import dabest




File = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx"
save_path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\PhD paper'
save = False


File_20Hz = pd.read_excel(File, sheet_name='20Hz_2,5mM')
File_50Hz = pd.read_excel(File, sheet_name='50Hz_2,5mM')
colors = ['grey','darkgreen','limegreen']
test_stat = 'Mann-Whitney'


x = np.arange(1,11)
amps_20Hz = File_20Hz.iloc[:,1:11]
amps_50Hz = File_50Hz.iloc[:,1:11]
amp1 = amps_20Hz['AMP1'].tolist() + amps_50Hz['AMP1'].tolist()
df_amps = pd.concat((pd.DataFrame(amp1),
                     amps_20Hz['AMP2'],
                     amps_50Hz['AMP2']), axis=1)
df_amps.columns = ['AMP1','AMP2_20Hz','AMP2_50Hz']
amp1_mean = np.mean(amp1)
amp1_std = np.std(amp1)

PPR_20Hz = [[amps_20Hz.iloc[i,j]/amps_20Hz.iloc[i,0] for j in range(amps_20Hz.shape[1])] for i in range(amps_20Hz.shape[0])]
PPR_50Hz = [[amps_50Hz.iloc[i,j]/amps_50Hz.iloc[i,0] for j in range(amps_50Hz.shape[1])] for i in range(amps_50Hz.shape[0])]
MEAN_20Hz = np.mean(PPR_20Hz, axis=0)
STD_20Hz = np.std(PPR_20Hz, axis=0)
SEM_20Hz = stats.sem(PPR_20Hz, axis=0)
MEAN_50Hz = np.mean(PPR_50Hz, axis=0)
STD_50Hz = np.std(PPR_50Hz, axis=0)
SEM_50Hz = stats.sem(PPR_50Hz, axis=0)

PPR3_2_20Hz = amps_20Hz['AMP3']/amps_20Hz['AMP2']
PPR3_2_50Hz = amps_50Hz['AMP3']/amps_50Hz['AMP2']
df_ppr3_2 = pd.concat((PPR3_2_20Hz, PPR3_2_50Hz), axis=1)
df_ppr3_2.columns = ['PPR3/2_20Hz','PPR3/2_50Hz']

PPR2_1_20Hz = [PPR_20Hz[i][1] for i in range(len(PPR_20Hz))]
PPR2_1_50Hz = [PPR_50Hz[i][1] for i in range(len(PPR_50Hz))]
df_ppr2_1 = pd.concat((pd.DataFrame(PPR2_1_20Hz), pd.DataFrame(PPR2_1_50Hz)), axis=1)
df_ppr2_1.columns = ['PPR2/1_20Hz', 'PPR2/1_50Hz']

Fail_20Hz = File_20Hz.iloc[:,20:23]
Fail_50Hz = File_50Hz.iloc[:,20:23]
all_fail1 = pd.concat((Fail_20Hz['%Fail1'], Fail_50Hz['%Fail1']), axis=0, ignore_index=True)
df_fail = pd.concat((all_fail1, Fail_20Hz['%Fail2'], Fail_50Hz['%Fail2']), axis=1)
df_fail.columns = ['%Fail1', '%Fail2_20Hz', '%Fail2_50Hz']


# TAU = File_20Hz['Tau1'].dropna().tolist() + File_50Hz['Tau1'].dropna().tolist()
# TAU_mean = np.mean(TAU)
# TAU_std = np.std(TAU)


##################################### FIGURES #################################################


### FIG POOLED BOUTONS PPR ###
fig_pool, ax_pool = plt.subplots(1,2, figsize=(8,4), sharey=True, tight_layout=True)
[ax_pool[0].plot(x, PPR_20Hz[i], 'k', alpha=0.05, lw=4) for i in range(len(PPR_20Hz))]
ax_pool[0].plot(x, np.mean(PPR_20Hz, axis=0), color=colors[1], lw=3, marker='o')
ax_pool[0].fill_between(x, MEAN_20Hz+SEM_20Hz, MEAN_20Hz-SEM_20Hz, color=colors[1], alpha=0.4)
ax_pool[0].axhline(1.0, color=colors[0], ls='--', lw=2)
ax_pool[0].set_ylabel('PPR An/A1')
ax_pool[0].set_xlabel('Pulse number')
ax_pool[0].set_title(f'PPR profiles 20Hz\nCa = 2.5mM\nn={len(PPR_20Hz)}')

[ax_pool[1].plot(x, PPR_50Hz[i], 'k', alpha=0.05, lw=4) for i in range(len(PPR_50Hz))]
ax_pool[1].plot(x, np.mean(PPR_50Hz, axis=0), color=colors[2], lw=3, marker='o')
ax_pool[1].fill_between(x, MEAN_50Hz+SEM_50Hz, MEAN_50Hz-SEM_50Hz, color=colors[2], alpha=0.4)
ax_pool[1].axhline(1.0, color=colors[0], ls='--', lw=2)
ax_pool[1].set_ylabel('PPR An/A1')
ax_pool[1].set_xlabel('Pulse number')
ax_pool[1].set_title(f'PPR profiles 50Hz\nCa = 2.5mM\nn={len(PPR_50Hz)}')

if save == True:
    plt.savefig(f'{save_path}\Pooled_boutons_profiles.pdf')




### FIG AMP1 ###
################

fig_amp1, ax_amp1 = plt.subplots(1,3, figsize=(8,3), tight_layout=True)
sns.boxplot(data=[amp1, amps_20Hz['AMP2'], amps_50Hz['AMP2']], ax=ax_amp1[0], palette=colors, showmeans=True)

cv_amp1 = stats.variation(amp1)
skewness_amp1 = stats.skew(amp1)

cv_amp2_20Hz = stats.variation(amps_20Hz['AMP2'])
skewness_amp2_20Hz = stats.skew(amps_20Hz['AMP2'])

cv_amp2_50Hz = stats.variation(amps_50Hz['AMP2'])
skewness_amp2_50Hz = stats.skew(amps_50Hz['AMP2'])

ax_amp1[1].bar([0,1,2], [cv_amp1, cv_amp2_20Hz, cv_amp2_50Hz], color=colors, tick_label=['All A1', 'A2-20Hz', 'A2-50Hz'])
ax_amp1[2].bar([0,1,2], [skewness_amp1, skewness_amp2_20Hz, skewness_amp2_50Hz], color=colors, tick_label=['All A1', 'A2-20Hz', 'A2-50Hz'])
ax_amp1[0].set_ylim(0)
ax_amp1[1].set_ylim(0,1)
ax_amp1[2].set_ylim(0,2)
ax_amp1[0].set_ylabel('Amplitude (DF/F0)')
ax_amp1[1].set_ylabel('CV')
ax_amp1[2].set_ylabel('Skewness')
ax_amp1[0].set_xticklabels(['All A1', 'A2-20Hz', 'A2-50Hz'])


stats.kruskal(amp1, amps_20Hz['AMP2'], amps_50Hz['AMP2'])
add_stat_annotation(ax_amp1[0], data=df_amps, box_pairs=[('AMP1', 'AMP2_20Hz')],
                    test=test_stat, text_format='full', verbose=2)
add_stat_annotation(ax_amp1[0], data=df_amps, box_pairs=[('AMP1', 'AMP2_50Hz')],
                    test=test_stat, text_format='full', verbose=2)
add_stat_annotation(ax_amp1[0], data=df_amps, box_pairs=[('AMP2_20Hz', 'AMP2_50Hz')],
                    test=test_stat, text_format='full', verbose=2)

# sns.histplot(data=amp1, binwidth=0.04, ax=ax_amp1, color=colors[0], stat='probability',  alpha=0.5)
# ax_amp1.set_xlabel('DF/F')

if save == True:
    plt.savefig(f'{save_path}\Amp1_histogram.pdf')



### FIG PPR2/1 ###
##################

fig_ppr, ax_ppr = plt.subplots(1,3, figsize=(8,3), tight_layout=True)
sns.boxplot(data=df_ppr2_1, ax=ax_ppr[0], palette=[colors[1], colors[2]], showmeans=True)

add_stat_annotation(ax_ppr[0], data=df_ppr2_1, box_pairs=[('PPR2/1_20Hz', 'PPR2/1_50Hz')],
                    test=test_stat, text_format='full', verbose=2)

cv_ppr_20Hz = stats.variation(df_ppr2_1['PPR2/1_20Hz'])
skewness_ppr_20Hz = stats.skew(df_ppr2_1['PPR2/1_20Hz'])

cv_ppr_50Hz = stats.variation(df_ppr2_1['PPR2/1_50Hz'].dropna())
skewness_ppr_50Hz = stats.skew(df_ppr2_1['PPR2/1_50Hz'].dropna())

ax_ppr[1].bar([0,1], [cv_ppr_20Hz, cv_ppr_50Hz], color=[colors[1], colors[2]], tick_label=['20Hz', '50Hz'])
ax_ppr[2].bar([0,1], [skewness_ppr_20Hz, skewness_ppr_50Hz], color=[colors[1], colors[2]], tick_label=['20Hz', '50Hz'])
ax_ppr[1].set_ylim(0,1)
ax_ppr[2].set_ylim(0,1)
ax_ppr[1].set_ylabel('CV')
ax_ppr[2].set_ylabel('Skewness')

ax_ppr[0].set_ylabel('A2/A1')
ax_ppr[0].set_xticklabels(['20Hz','50Hz'])
ax_ppr[0].set_ylim(0)

# ppr_eff_size = dabest.load(df_ppr2_1, idx=('PPR2/1_20Hz', 'PPR2/1_50Hz'), resamples=5000)
# ppr_eff_size.mean_diff.plot(ax=ax_ppr[1], custom_palette=['b', 'orange'])

if save == True:
    plt.savefig(f'{save_path}\PPR_comparison.pdf')
    

### FIG PPR3/2 ###
##################

# fig_ppr32, ax_ppr32 = plt.subplots(tight_layout=True)
# sns.boxplot(data=df_ppr3_2, ax=ax_ppr32, palette=['b', 'orange'], showmeans=True)


#### FIG TAU #####
##################
# fig_tau, ax_tau = plt.subplots()
# sns.histplot(data=TAU, bins=40, ax=ax_tau, color=colors[0], stat='proportion', alpha=0.5)
# ax_tau.set_xlabel('Tau (ms)')
# ax_tau.set_title('µ = {:.2f} ms ; std = {:.2f}'.format(TAU_mean, TAU_std))

# if save == True:
#     plt.savefig(f'{save_path}\Tau1_histogram.pdf')




### FIG PPR2/1 VS AMP1 ###
##########################

fig_ppr_amp, ax_ppr_amp = plt.subplots(tight_layout=True)

transformer = FunctionTransformer(np.log, validate=True)
model = LinearRegression()

x_20Hz = np.array(amps_50Hz['AMP1']).reshape(-1,1)
y_20Hz = np.array(PPR2_1_50Hz).reshape(-1,1)

x_20Hz_trans = transformer.fit_transform(x_20Hz)
reg = model.fit(x_20Hz_trans, y_20Hz)
r = np.sqrt(reg.score(x_20Hz_trans, y_20Hz))
pred = reg.predict(x_20Hz_trans)

# PAIRS = [[x,y] for x, y in zip(amps_50Hz['AMP1'], PPR2_1_50Hz)]
# PAIRS_sorted = sorted(PAIRS)
# X = [PAIRS_sorted[pair][0] for pair in range(len(PAIRS_sorted))]
# X = np.array(X).reshape(len(X),1)
# Y = [PAIRS_sorted[pair][1] for pair in range(len(PAIRS_sorted))]

# model = SVR()
# model.fit(X, Y)
# model.score(X, Y)
# prediction = model.predict(X)

ax_ppr_amp.scatter(x_20Hz, y_20Hz, color=colors[1], alpha=0.8)
ax_ppr_amp.scatter(amps_50Hz['AMP1'], PPR2_1_50Hz, color=colors[2], alpha=0.8)
ax_ppr_amp.plot(sorted(x_20Hz), sorted(pred, reverse=True), color='r', alpha=0.8)
ax_ppr_amp.axhline(y=1, color=colors[0], ls='--')

ax_ppr_amp.set_ylabel('PPR (A2/A1)')
ax_ppr_amp.set_xlabel('A1')
ax_ppr_amp.set_ylim(0)

if save == True:
    plt.savefig(f'{save_path}\PPR_against_Amp1.pdf')




### FIG %Failure AMP1 ###
#########################

fig_fail, ax_fail = plt.subplots(1,3, figsize=(8,3), tight_layout=True)
sns.boxplot(data=df_fail, ax=ax_fail[0], palette=colors, showmeans=True)
add_stat_annotation(ax_fail[0], data=df_fail, box_pairs=[('%Fail1','%Fail2_20Hz'),
                                                         ('%Fail1','%Fail2_50Hz'),
                                                         ('%Fail2_20Hz','%Fail2_50Hz')], test='Mann-Whitney', text_format='full', verbose=2)

cv_fail1 = stats.variation(df_fail['%Fail1'])
skewness_fail1 = stats.skew(df_fail['%Fail1'])

cv_fail2_20Hz = stats.variation(df_fail['%Fail2_20Hz'].dropna())
skewness_fail2_20Hz = stats.skew(df_fail['%Fail2_20Hz'].dropna())

cv_fail2_50Hz = stats.variation(df_fail['%Fail2_50Hz'].dropna())
skewness_fail2_50Hz = stats.skew(df_fail['%Fail2_50Hz'].dropna())

ax_fail[1].bar([0,1,2], [cv_fail1, cv_fail2_20Hz, cv_fail2_50Hz], color=colors, tick_label=['All A1', 'A2-20Hz', 'A2-50Hz'])
ax_fail[2].bar([0,1,2], [skewness_fail1, skewness_fail2_20Hz, skewness_fail2_50Hz], color=colors, tick_label=['All A1', 'A2-20Hz', 'A2-50Hz'])
ax_fail[1].set_ylim(0,1.5)
ax_fail[2].set_ylim(-0.5,2)
ax_fail[1].set_ylabel('CV')
ax_fail[2].set_ylabel('Skewness')

ax_fail[0].set_xticklabels(['All A1', 'A2-20Hz', 'A2-50Hz'])
ax_fail[0].set_ylabel('Failure rate (%)')
ax_fail[0].set_ylim(0,100)

stats.kruskal(df_fail['%Fail1'], df_fail['%Fail2_20Hz'].dropna(), df_fail['%Fail2_50Hz'].dropna())


### FIG FAIL1 AGAINST AMP1 ###
##############################

fig_fail1_amp1, ax_fail1_amp1 = plt.subplots()
x_fail = np.array(amp1).reshape(-1,1)
y_fail = np.array(all_fail1).reshape(-1,1)
reg = model.fit(x_fail, y_fail)
r = np.sqrt(reg.score(x_fail, y_fail))
pred = reg.predict(x_fail)
ax_fail1_amp1.scatter(amp1, all_fail1, color=colors[0], alpha=0.8)
ax_fail1_amp1.plot(x_fail, pred, color='r')
ax_fail1_amp1.set_title('A1 Psyn against A1')
ax_fail1_amp1.set_ylabel('A1 Psyn')
ax_fail1_amp1.set_xlabel('A1 (DF/F)')
ax_fail1_amp1.set_xlim(0)
ax_fail1_amp1.set_ylim(0)



### FIG FAIL1 AGAINST PPR ###
#############################

fig_fail1_ppr, ax_fail1_ppr = plt.subplots()
ax_fail1_ppr.scatter(df_ppr2_1['PPR2/1_20Hz'], Fail_20Hz['%Fail1'], color=colors[1], alpha=0.8)
ax_fail1_ppr.scatter(df_ppr2_1['PPR2/1_50Hz'].dropna(), Fail_50Hz['%Fail1'], color=colors[2], alpha=0.8)
ax_fail1_ppr.set_title('%Fail1 against PPR 20Hz')
ax_fail1_ppr.set_ylabel('%Fail1')
ax_fail1_ppr.set_xlabel('PPR2/1')
ax_fail1_ppr.legend(['20Hz','50Hz'])

if save == True:
    plt.savefig(f'{save_path}\Failures_amp1.pdf')