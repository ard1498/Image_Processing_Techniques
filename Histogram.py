# Histogrm Equalisation

import cv2
import numpy as np
from collections import Counter
from pprint import pprint
import matplotlib.pyplot as plt

image = cv2.resize(cv2.imread("./IA1.jpg"),(400,200))
plt.imshow(image)
plt.show()

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

counter = Counter(image.ravel().copy())
# print(counter)
plt.bar(counter.keys(), counter.values())
plt.show()

dic = dict(counter)
Pixels = np.array([i for i in dic.values()])
total_pixels = Pixels.sum()
n = len(dic)


Keys = np.array([i for i in dic.keys()])

for i in range(256):
    l = list()
    if i not in dic: 
        l.append(0)
        dic[i] = 0
    else : 
        l.append(dic[i])
        
    l.append(float(dic[i]/total_pixels))
    
    if i == 0: 
        # print(l[-1])
        l.append(l[-1])
    else: 
        # print(dic[i-1][2])
        l.append(l[-1] + dic[i-1][2])
    
    l.append( int(l[-1]*(n-1)) )
    dic[i] = l

# pprint(dic)
# print(dic[:10])
new_dic = dict()
for i in range(len(dic)):
    if dic[i][-1] not in new_dic:
        new_dic[dic[i][-1]] = 0
    new_dic[dic[i][-1]] += dic[i][0]
# pprint(new_dic)

plt.bar(new_dic.keys(), new_dic.values())
plt.show()

new_image = image.copy()
for i in range(new_image.shape[0]):
    for j in range(new_image.shape[1]):
        for k in range(new_image.shape[2]):
            new_image[i, j, k] = dic[new_image[i, j, k]][-1]
        
new_image = np.array(new_image)
print('Enhanced Image')
plt.imshow(new_image.copy())
plt.show()

cv2.imshow('Image1', image.copy())
cv2.imshow('Enhanced Image', new_image.copy())
# cv2.imwrite('Desired_image.jpg',new_image.copy())
cv2.waitKey(0)
cv2.destroyAllWindows()


