# component labelling in an image
# Author : Anirudh Aggarwal

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint

class Label :

    def __init__(self):
        self.tree = {}

    def print_tree(self):
        pprint.pprint(self.tree)

    def show_image(self, image):
        cv2.imshow('Component Labelled Image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        plt.imshow(image,cmap='gray')
        plt.show()
    
    def unionSet(self, val1, val2):
        if val1 < val2:
            self.tree[val2] = val1
        else:
            self.tree[val1] = val2
        return

    def findparent(self, val1):
        if self.tree[val1] == val1:
            return val1
        return self.findparent(self.tree[val1])
    
    def tuned_Image(self, image):
        n = image.shape[0]
        m = image.shape[1]

        for i in range(n):
            for j in range(m):
                if image[i,j] == 0:
                    continue
                if i == 0:
                    continue
                elif j == 0:
                    continue
                else:
                    if image[i,j-1] != 0 and image[i-1,j] != 0:
                        self.unionSet(image[i,j-1],image[i-1,j])
                        image[i,j] = self.findparent(self.tree[max(image[i,j-1],image[i-1,j])])

        print('just before second tuning')
        #plt.imshow(image, cmap= 'gray')
        #plt.show()
        for i in range(n):
            for j in range(m):
                if image[i,j] == 0:
                    continue
                else:
                    image[i,j] = self.findparent(image[i,j])
        return image
    
    def getnooflabels(self,image):
        print(np.unique(image))
        return len(np.unique(image))
    
    def component_labelling(self,image):
        B = [0]
        n, m = image.shape[0],image.shape[1]
        color = 10
        for i in range(n):
            for j in range(m):
                if image[i,j] in B:
                    pass
                elif i >= 1 and image[i-1,j] not in B :
                    image[i,j] = image[i-1,j]
                elif i >= 1 and j < m-1  and image[i-1,j+1] not in B:
                    image[i,j] = image[i-1,j+1]
                    if j >= 1 and image[i-1,j-1] not in B:
                        self.unionSet(image[i-1,j-1] , image[i-1,j+1])
                    elif j >= 1 and image[i, j-1] not in B:
                        self.unionSet(image[i-1,j-1], image[i,j-1])
                elif i >= 1 and j >= 1 and image[i-1,j-1] not in B:
                    image[i,j] = image[i-1,j-1]
                elif j >= 1 and image[i,j-1] not in B:
                    image[i,j] = image[i,j-1]
                else:
                    image[i,j] = color
                    self.tree[color] = color
                    color += 5
        return image, color
    
def main():
    ch = input('enter (1) for image OR (2) for random Image generation:')
    image = ''
    if ch == '1':
        input_image = input('enter the image :')
        image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    if ch == '2':
        n = int(input('enter the row :'))
        m = int(input('enter the col :'))
        n_colors = int(input('enter no of shades of gray:'))
        image = np.random.randint(n_colors, size=(n,m))
    image_copy = np.array(image)
    print('Original Image:')
    pprint.pprint(image_copy)
    
    label_obj = Label()
    label_obj.show_image(image_copy)
    image_obtained, colors = label_obj.component_labelling(image_copy)
    print('no of colors used till now',(colors-10)//5)
    tuned_image = label_obj.tuned_Image(image_obtained)
    label_obj.show_image(tuned_image)
    # tuned_image2 = label_obj.tuned_Image(tuned_image)
    label_obj.show_image(tuned_image)
    print('Tuned Image:')
    pprint.pprint(tuned_image)
    print('enter the unique labels :' + str(label_obj.getnooflabels(tuned_image)))
    # label_obj.print_tree()
    return()

if __name__ == '__main__' :
    main()
