# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:56:23 2023

@author: ruo4037
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as stats
import uncertainties as unc
import uncertainties.unumpy as unp
import seaborn as sns

sheet = '20Hz_2,5mM'
file = pd.read_excel(r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx', sheet_name = sheet)
# file = pd.read_excel(r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\PhD paper\Pooled_data_for_correlation.xlsx', sheet_name='Sheet2')

x = file['F0']
variable = ['AMP1','AMP2','PPR2/1']
color = 'darkgreen'
fig_hist_F0, ax_hist_F0 = plt.subplots()
sns.histplot(data=x, binwidth=1, ax=ax_hist_F0, stat='probability', kde=True)

def func_reg(x, a, b):
    return a * x + b


for i in range(len(variable)):
    y = file[variable[i]]
    n = len(y)
    
    popt, pcov = curve_fit(func_reg, x, y)
    # retrieve parameter values
    a = popt[0]
    b = popt[1]
    print(variable[i])
    print('----------------')
    print('Optimal Values')
    print('a: ' + str(a))
    print('b: ' + str(b))
    
    # compute r^2
    r2 = 1.0-(sum((y-func_reg(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
    print('R^2: ' + str(r2))
    
    # calculate parameter confidence interval
    a,b = unc.correlated_values(popt, pcov)
    print('Uncertainty')
    print('a: ' + str(a))
    print('b: ' + str(b))
    
    # pearson correlation coefficient and p value
    r, p = stats.pearsonr(x,y)
    print('Pearson test')
    print('r: ' + str(r))
    print('p: ' + str(p))
    print('----------------')
    print('----------------')
    
    # plot data
    plt.figure()
    plt.scatter(x, y, s=40, color=color, alpha=0.5, label='Data')
    
    # calculate regression confidence interval
    px = np.linspace(np.min(x), np.max(x), len(x))
    py = a*px+b
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    
    confidence = 0.95
    def predband(x, xd, yd, p, func, conf=confidence):
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
    
    lpb, upb = predband(px, x, y, popt, func_reg, conf=confidence)
    
    # plot the regression
    plt.plot(px, nom, c='black', label='y=a x + b')
    
    # uncertainty lines (95% confidence)
    plt.plot(px, nom - 1.96 * std, c='darkorange',\
             label=f'{confidence*100}% Confidence Region')
    plt.plot(px, nom + 1.96 * std, c='darkorange')
    # prediction band (95% confidence)
    plt.plot(px, lpb, 'k--',label=f'{confidence*100}% Prediction Band')
    plt.plot(px, upb, 'k--')
    plt.xlabel('F0 (a.u.)')
    plt.ylabel(variable[i])
    plt.legend(loc='best')