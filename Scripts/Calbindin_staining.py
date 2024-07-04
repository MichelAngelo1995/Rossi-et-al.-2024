# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:10:21 2022

@author: theo.rossi
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

file = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\Calbindin_L7-tdTomato_PCs_counting.xlsx"

data = pd.read_excel(file, sheet_name='Merge')

sagittal = data.iloc[1,1:]
horizontal = data.iloc[-1,1:]

# stats.chisquare(sagittal)
# stats.chisquare(horizontal)

fig, ax = plt.subplots(1,2,figsize=(4,4),sharey=True,tight_layout=True)
ax[0].bar(0,sagittal,color='orange', tick_label=['Merge'])
ax[1].bar(0,horizontal,color='orange', tick_label=['Merge'])
ax[0].set_title('Sagittal slice')
ax[1].set_title('Horizontal slice')
ax[0].set_ylabel('% cells stained')
ax[0].set_ylim(0,100)


# colors = ['r','g']
# fig, ax = plt.subplots(1,2,figsize=(6,4),sharey=True,tight_layout=True)
# ax[0].bar([0,1],sagittal,color=colors, tick_label=['tdTomato','Calbindin'])
# ax[1].bar([0,1],horizontal,color=colors, tick_label=['tdTomato','Calbindin'])
# ax[0].set_title('Sagittal slice')
# ax[1].set_title('Horizontal slice')
# ax[0].set_ylabel('% cells stained')
# ax[0].set_ylim(0,100)