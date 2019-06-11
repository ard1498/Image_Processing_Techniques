# Connectivity of pixels
# Author: Anirudh Aggarwal


import numpy as np
import matplotlib.pyplot as plt
import cv2

class Connectivity:

    def getPlotFromXY(self, X, Y, n, m):
        for i in range(n):
            for j in range(m):
                plt.scatter(i,j)
        for i in range(len(X)):
            plt.plot(X[i],Y[i])
        plt.show()

    def getPixelsFromXY(self, X, Y):
        Pixel_List = []
        for i in range(len(X)):
            pixel1 = (X[i][0],Y[i][0])
            pixel2 = (X[i][1],Y[i][1])
            Pixel_List.append(pixel1)
            Pixel_List.append(pixel2)
        Pixel_List = set(Pixel_List)
        return Pixel_List

    def getImageFromPixels(self, image, Pixel_List):
        n = image.shape[0]
        m = image.shape[1]

        for i in range(n):
            for j in range(m):
                if (i,j) in Pixel_List:
                    image[i,j] = 0
                else:
                    image[i,j] = 1
        return image

    def get4connected(self, image, V):
        n = image.shape[0]
        m = image.shape[1]
        X = []
        Y = []

        for j in range(m):
            for i in range(n):
                if image[i,j] not in V:
                    continue

                if i+1 < n and image[i+1,j] in V:
                    X.append([i,i+1])
                    Y.append([j,j])

                if j+1 < m and image[i,j+1] in V:
                    X.append([i,i])
                    Y.append([j,j+1])

        self.getPlotFromXY(X, Y, n, m)
        Pixel_List = self.getPixelsFromXY(X,Y)
        return self.getImageFromPixels(image, Pixel_List)

    def getDconnected(self, image, V):
        n = image.shape[0]
        m = image.shape[1]
        X = []
        Y = []

        for j in range(m):
            for i in range(n):
                if image[i,j] not in V:
                    continue

                if j+1 < m:
                    if i+1 < n and image[i+1,j+1] in V:
                        X.append([i,i+1])
                        Y.append([j,j+1])
                    if i-1 >= 0 and image[i-1,j+1] in V:
                        X.append([i,i-1])
                        Y.append([j,j+1])

        self.getPlotFromXY(X, Y, n, m)
        Pixel_List = self.getPixelsFromXY(X,Y)
        return self.getImageFromPixels(image, Pixel_List)

    def get8connected(self, image, V):
        n = image.shape[0]
        m = image.shape[1]
        X = []
        Y = []

        for j in range(m):
            for i in range(n):
                if image[i,j] not in V:
                    continue

                if i+1 < n and image[i+1,j] in V:
                    X.append([i,i+1])
                    Y.append([j,j])

                if j+1 < m:
                    if image[i,j+1] in V:
                        X.append([i,i])
                        Y.append([j,j+1])
                    if i+1 < n and image[i+1,j+1] in V:
                        X.append([i,i+1])
                        Y.append([j,j+1])
                    if i-1 >= 0 and image[i-1,j+1] in V:
                        X.append([i,i-1])
                        Y.append([j,j+1])

        self.getPlotFromXY(X, Y, n, m)
        Pixel_List = self.getPixelsFromXY(X,Y)
        return self.getImageFromPixels(image, Pixel_List)

    def getMconnected(self, image, V):
        n = image.shape[0]
        m = image.shape[1]
        X = []
        Y = []

        for j in range(m):
            for i in range(n):
                if image[i,j] not in V:
                    continue
                # marking all N4(p) q's
                if i+1 < n and image[i+1,j] in V:
                    X.append([i,i+1])
                    Y.append([j,j])

                if j+1 < m and image[i,j+1] in V:
                    X.append([i,i])
                    Y.append([j,j+1])

                # now checking Nd(p) and N4(p) and N4(q)
                if j+1 < m:
                    if i+1 < n and image[i+1,j+1] in V:
                        if image[i+1,j] not in V and image[i,j+1] not in V:
                            X.append([i,i+1])
                            Y.append([j,j+1])
                    if i-1 >= 0 and image[i-1,j+1] in V:
                        if image[i-1,j] not in V and image[i,j+1] not in V:
                            X.append([i,i-1])
                            Y.append([j,j+1])

        self.getPlotFromXY(X, Y, n, m)
        Pixel_List = self.getPixelsFromXY(X,Y)
        return self.getImageFromPixels(image, Pixel_List)

def main():
    ch = int(input('enter (1) image or (2) for random image:'))
    if ch == 2:
        n = int(input("enter the number of rows or row pixels :"))
        m = int(input("enter the number of columns or column pixels :"))

        image = np.random.randint(2,size = (n,m))
    if ch == 1:
        img = input('enter the image:')
        image = cv2.resize(cv2.imread(img,cv2.IMREAD_GRAYSCALE), (32,32))
    # print(image)

    plt.imshow(image,cmap='gray')
    plt.show()

    c = Connectivity()
    image_N4 = np.array(image.copy())
    image_N8 = np.array(image.copy())
    image_ND = np.array(image.copy())
    image_NM = np.array(image.copy())

    image_N4 = c.get4connected(image_N4, [0])
    image_N8 = c.get8connected(image_N8, [0])
    image_ND = c.getDconnected(image_ND, [0])
    image_NM = c.getMconnected(image_NM, [0])

    plt.imshow(image_N4,cmap='gray')
    plt.show()
    plt.imshow(image_N8,cmap='gray')
    plt.show()
    plt.imshow(image_ND,cmap='gray')
    plt.show()
    plt.imshow(image_NM,cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
