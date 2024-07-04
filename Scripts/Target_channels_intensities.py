# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:07:47 2023

@author: ruo4037
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.stats as stats
import configparser as cp
import imageio.v3 as iio
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

image = iio.imread(uri=r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\20210308_linescan2_bouton2_ZX.png")[:,:,:2]
ini = r"C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\Scripts_Rossi_et_al_2024\Files\20210308_15_22_37_linescan2_zstack_XYTZ.ini"
coord_center = (19.6, 3.7)


config = cp.ConfigParser()
config.read(ini)

pixel_size = float(config.get('_', 'x.pixel.sz'))*1000000
x_distance_vector = np.arange(0, pixel_size*image.shape[1], pixel_size)
z_distance_vector = np.arange(0, pixel_size*image.shape[0], pixel_size)

fig_merge, ax_merge = plt.subplots(figsize=(16,5))
colors = ['RdPu',
          'Greens']

for i in range(image.shape[2]):
    t = threshold_otsu(image[:,:,i])
    if i == 0:
        binary_mask = image > t
    else:
        binary_mask = image > t+(t*0.2)
        
    fig, ax = plt.subplots(2, 1, figsize=(16,5), tight_layout=True)
    ax[0].imshow(image[:,:,i], cmap=colors[i], extent=[x_distance_vector[0], x_distance_vector[-1], z_distance_vector[-1], z_distance_vector[0]])
    ax[1].imshow(binary_mask[:,:,i], cmap=colors[i], extent=[x_distance_vector[0], x_distance_vector[-1], z_distance_vector[-1], z_distance_vector[0]])
    ax[1].set_ylabel('Z axis')
    
    ax_merge.imshow(binary_mask[:,:,i], cmap=colors[i], extent=[x_distance_vector[0], x_distance_vector[-1], z_distance_vector[-1], z_distance_vector[0]], alpha=0.7)
    ax_merge.set_ylabel('Z axis')

ax_merge.minorticks_on()
ax_merge.grid(which='major')
ax_merge.grid(which='minor', linestyle='--')


# #Radial plot
# radius = np.array([1,2,3,4,5])
# ax_merge.scatter(coord_center[0], coord_center[1], color='k', s=10)
# for i in range(len(radius)):
#     major_circ = Circle(coord_center, radius[i], facecolor='None', edgecolor='k', alpha=0.5)
#     minor_circ = Circle(coord_center, radius[i]-0.5, facecolor='None', edgecolor='k', ls='--', alpha=0.5)
#     ax_merge.add_patch(major_circ)
#     ax_merge.add_patch(minor_circ)
