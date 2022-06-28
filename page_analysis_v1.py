# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:28:07 2022

@author: rmish
"""

from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from math import sqrt
import skimage.draw as skdraw
import numpy as np

def circle_mask_generator(h,w,radius):  
    """Returns binary mask with a circle in the center for use with a DMD.
    h,w: height, width of mask.
    radius: radius of circle in the center of the mask."""
    rr,cc = skdraw.disk((h/2,w/2),radius,shape=[h,w])
    mask1 = np.zeros([h,w],dtype='uint8')
    mask1[rr,cc] = 255
    return mask1

image = io.imread('analysis_v1.jpg')
image_gray = color.rgb2gray(io.imread('analysis_v1.jpg'))
# plt.imshow(img)

h,w = np.shape(image_gray)
blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, min_sigma=10, max_sigma=30, threshold=.01)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()