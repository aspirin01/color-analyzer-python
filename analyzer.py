import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from glob import glob
import os
%matplotlib inline
from collections import Counter
from sklearn.cluster import KMeans
def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    total=0
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
        total+=perc[i]
    perc = dict(sorted(perc.items()))
    
    #for logging purposes
    print(perc)
    print(total)
    
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette
clt=KMeans(n_clusters=10)

img_mask = '/content/*.jpeg'
img_names = glob(img_mask)

for fn in img_names:
  img = cv.imread(fn)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  print(fn)
  dim = (500, 300)
  # resize image
  img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
  clt_1 = clt.fit(img.reshape(-1, 3))
  show_img_compar(img, palette_perc(clt_1))
