# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:28:07 2022

@author: rmish
"""

import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import numpy as np
import cv2
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
# import skimage.exposure
from skimage.filters import threshold_otsu
from skimage import exposure
import pandas as pd

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

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    # Calculate grayscale histogram
    # hist = cv2.calcHist([gray],[0],None,[65536],[0,65536])
    hist = skimage.exposure.histogram(gray)
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

image = io.imread('analysis_v1.jpg')
image_gray = color.rgb2gray(io.imread('analysis_v1.jpg'))
z = exposure.adjust_sigmoid(image_gray,cutoff=0.5)
cont_im = exposure.rescale_intensity(image_gray)
z1 = exposure.equalize_adapthist(image_gray)

thresh = threshold_otsu(z1)

blurred_image = skimage.filters.gaussian(z1, sigma=1)
binary_mask = blurred_image > thresh
labeled_image, count = skimage.measure.label(binary_mask,
                                             connectivity=2, return_num=True)

# plt.imshow(img)
fig, ax = plt.subplots(4,1)
ax[0].imshow(image)
ax[1].imshow(z1,cmap='Greys_r')
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
norm_ims = (ims-ims.min()*0.99)/(ims.max()-ims.min()*0.99)
plt.bar(ts,norm_ims)


groups = [[0],[1],[2],[3],[4,5],[6,7],[8,9],[10,11],[12],[13],[14]]
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

z=np.vstack((groups,ratios))
