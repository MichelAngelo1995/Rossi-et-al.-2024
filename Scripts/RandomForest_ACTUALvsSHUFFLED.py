# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:28:32 2023

@author: ruo4037
"""

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statannot import add_stat_annotation
import pandas as pd
import numpy as np

file = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\Random_Forest_scores.xlsx'

data =  pd.read_excel(file).iloc[:,1:]
colors = ['b','grey']

fig, ax = plt.subplots()
sns.boxplot(data=data, ax=ax, palette=colors, showmeans=True)
add_stat_annotation(ax, data=data, box_pairs=[('Scores_actual','Scores_shuffled')],
                    test='t-test_ind', text_format='full', verbose=2)
ax.set_ylim(0,1)