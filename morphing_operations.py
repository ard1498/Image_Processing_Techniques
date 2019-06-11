
#%%
import cv2, numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as Img
import math


#%%
def dilation(img,f):
    print(img.shape)
    m,n= img.shape
    new_img = np.zeros(((m-f+1),(n-f+1)), dtype = 'int') 
    for i in range(m-f+1):
        for j in range(n-f+1):
                x = img[i:i+f, j:j+f].sum()/(f**2)
                if x > 0:
                    new_img[i,j] = 255
                else:
                    new_img[i,j] = 0
    return new_img


#%%
def erosion(img,f):
    m,n = img.shape
    new_img = np.zeros(((m-f+1),(n-f+1)), dtype = 'int')
    for i in range(m-f+1):
        for j in range(n-f+1):
                x = img[i:i+f, j:j+f].sum()/(f**2)
                if x == 255:
                    new_img[i,j] = 255
    return new_img


#%%
img = cv2.resize(cv2.imread('binary.png',0),(200,200))
plt.imshow(img)
plt.show()
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
dilated_image = dilation(img.copy(),2)
ero_image = erosion(img.copy(),2)
plt.imshow(dilated_image)
plt.show()
plt.imshow(ero_image)
plt.show()

#%%
# opening :
open_img = erosion(img.copy(), 2)
print(open_img)
open_img = dilation(open_img, 2)
plt.imshow(open_img)
plt.show()

#%%
# closing
close_img = dilation(img.copy(), 2)
close_img = erosion(close_img.copy(), 2)
plt.imshow(close_img)
plt.show()

#%%
cv2.imshow('Closing Image', close_img)
cv2.imshow('Opening Image', open_img)
cv2.imshow('dilated_image.png',dilated_image)
cv2.imshow('erosion_image.png',ero_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


