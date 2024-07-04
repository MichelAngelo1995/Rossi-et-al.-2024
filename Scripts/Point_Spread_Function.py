# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:16:46 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from statannot import add_stat_annotation

File = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\PSF_2P_laser.xlsx"

label = ['Lateral', 'Axial']
colors = ['b','g']
for i in range(len(label)):
    file = pd.read_excel(File, sheet_name=i+1)
    x = file['X_Fit:_Gaussian']
    y = file['Fit:_Gaussian']
    
    baseline = np.mean(y[:20])
    y_norm = (y - baseline)/np.max(y - baseline)
    
    plt.figure()
    plt.plot(x, y_norm, color=colors[i])
    plt.xlabel('Distance (µm)')
    plt.ylabel('Norm. intensity')
    plt.title(f'{label[i]}')


file_measures = pd.read_excel(File, sheet_name = 'Measures')
lateral = file_measures['FWHM lateral exp (µm)'][:-4]
axial = file_measures['FWHM axial exp (µm)'][:-4]
df = pd.concat((pd.DataFrame(lateral), pd.DataFrame(axial)), axis=1)
df.columns = label
print('FWHM values:')
print('------------')
print(df)

fig, ax = plt.subplots(figsize=(3,4), tight_layout=True)
sns.barplot(data=df, ax=ax, palette=colors, capsize=0.2)
add_stat_annotation(ax, data=df, box_pairs=[('Lateral', 'Axial')],
                    test='Wilcoxon', text_format='full', verbose=2)
ax.set_ylabel('FWHM (µm)')