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

import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
from skimage.filters import threshold_otsu
import pandas as pd
def circle_mask_generator(h,w,x,y,radius):  
    """Returns binary mask with a circle in the center for use with a DMD.
    h,w: height, width of mask.
    radius: radius of circle in the center of the mask."""
    rr,cc = skdraw.disk((x,y),radius,shape=[h,w])
    mask1 = np.zeros([h,w],dtype='bool')
    mask1[rr,cc] = 1
    return mask1


def connected_components(filename, sigma=1.0, t=0.5, connectivity=2):
    # load the image
    image = skimage.io.imread(filename)
    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image)
    # denoise the image with a Gaussian filter
    blurred_image = skimage.filters.gaussian(gray_image, sigma=sigma)
    # mask the image according to threshold
    binary_mask = blurred_image < t
    # perform connected component analysis
    labeled_image, count = skimage.measure.label(binary_mask,
                                                 connectivity=connectivity, return_num=True)
    return labeled_image, count


image = io.imread('analysis_v1.jpg')
image_gray = color.rgb2gray(io.imread('analysis_v1.jpg'))

thresh = threshold_otsu(image_gray)

blurred_image = skimage.filters.gaussian(image_gray, sigma=1)
binary_mask = blurred_image > thresh
labeled_image, count = skimage.measure.label(binary_mask,
                                             connectivity=2, return_num=True)

# plt.imshow(img)
fig, ax = plt.subplots(4,1)
ax[0].imshow(image)
ax[1].imshow(image_gray,cmap='Greys_r')
plt.axis("off")
plt.show()
colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0,image_alpha=0.5)
ax[2].imshow(colored_label_image)

ft = pd.DataFrame(skimage.measure.regionprops_table(labeled_image, intensity_image=image_gray,
                                                  properties=('centroid',
                                                 'intensity_max',
                                                 'intensity_min',
                                                 'intensity_mean',
                                                 'image_intensity',
                                                 'area')))


# fig, ax = plt.subplots()
ax[3].imshow(labeled_image)

ft_sorted=ft.sort_values(by=['centroid-1'],ignore_index=True)

#up to here is good. next, I think I should create a clustering algorithm 
#that finds the correct points, and makes a box mask around them.

x = ft_sorted['centroid-0']
y = ft_sorted['centroid-1']

xy = pd.concat([x,y],axis=1)

ax[3].scatter(y,x)
ts = np.arange(count)

for i, txt in enumerate(ts):
    ax[3].annotate(txt,(y[i],x[i]))
# plt.axis("off")
# plt.show()

plt.figure()
ims = ft_sorted['intensity_mean']
plt.bar(ts,ims)
plt.figure()
norm_ims = (ims-ims.min()*0.9)/(ims.max()-ims.min()*0.9)
plt.bar(ts,norm_ims)


groups = [[0],[1],[2],[3],[4,5],[6,7],[8,9],[11,10],[12],[13],[14]]
ratios = np.zeros(len(groups))
c=0
for i in groups:
    print(i)
    if len(i) == 1:
        ratios[c] = 1
    elif len(i) == 2:
        cleave = norm_ims[i[0]]/(norm_ims[i[0]]+norm_ims[i[1]])
        cleave1 = norm_ims[i[0]]/norm_ims[i[1]]
        cleave2 = norm_ims[i[1]]/norm_ims[i[0]]
        ratios[c] = cleave#np.min((cleave1,cleave2))
        
    c+=1

plt.figure()
z=np.vstack((groups,ratios))
plt.bar(np.arange(len(groups)),ratios)
