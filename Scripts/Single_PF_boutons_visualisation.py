# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:17:17 2024

@author: ruo4037
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.signal import savgol_filter

Path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Folders\Boutons_analysis'
files = sorted(os.listdir(Path))

freq = '20Hz'
calcium = '2.5mM'
parallel_fiber = '20210304_linescan1'


fig, ax = plt.subplots(1,2,figsize=(10,4))
x = np.arange(1,11)
for file in range(len(files)):
    if freq in files[file]:
        if calcium in files[file]:
            if parallel_fiber in files[file]:
               if 'traces_converted.xlsx' in files[file]: 
                    print(files[file])
                    data = pd.read_excel(f'{Path}/{files[file]}', sheet_name='Traces DF_F0')
                    trace = data['Average']
                    time = data['Time']  
                    episodes = data.drop(['Average','Time'], axis=1)
                    
                    fig_ep, ax_ep = plt.subplots()
                    [ax_ep.plot(time, savgol_filter(episodes.iloc[:,ep],9,2), color='k', alpha=0.3) for ep in range(episodes.shape[1])]
                    ax_ep.set_title(files[file])
                    ax_ep.set_ylabel('Amplitude (DF/F0)')
                    ax_ep.set_xlabel('Time (ms)')
                    
                    ax[0].plot(time, savgol_filter(trace,9,2))
                    ax[0].set_title('Average traces')
                    ax[0].set_ylabel('Amplitude (DF/F0)')
                    ax[0].set_xlabel('Time (ms)')
                    
               if 'Amp.xlsx' in files[file]:
                   amps = pd.read_excel(f'{Path}/{files[file]}')
                   ppr = amps.iloc[0,1:]/amps.iloc[0,1]
                   ax[1].plot(x, ppr, marker='o')
                   ax[1].set_ylabel('PPR (An/A1)')
                   ax[1].set_xlabel('Pulse number')
