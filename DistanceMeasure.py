# Different Distance Measures for a given pixel
# Author : Anirudh Aggarwal

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

#%%
class Distance:
    def Euclid(self, p, q):
        x,y = p[0],p[1]
        s,t = q[0],q[1]
        return ((x-s)**2 + (y-t)**2)**(0.5)
    
    def Manhattan(self, p, q):
        x,y = p[0],p[1]
        s,t = q[0],q[1]
        return abs(x-s) + abs(y-t)
    
    def ChessBoard(self, p, q):
        x,y = p[0],p[1]
        s,t = q[0],q[1]
        return max(abs(x-s),abs(y-t))
    
    def plotDistanceMeasure(self, image, Dist_dic):
        n = image.shape[0]
        m = image.shape[1]

        for k in Dist_dic.keys():
            for j in range(m):
                for i in range(n):
                    if (i,j) in Dist_dic[k]:
                        image[i,j] = 0
                    else:
                        image[i,j] = 1
            plt.imshow(image,cmap = 'gray')
            plt.show()

    def getDistance(self, image, pixel):
        n = image.shape[0]
        m = image.shape[1]
        x,y = pixel[0],pixel[1]

        dist_dic_Euclid = {}
        dist_dic_Man = {}
        dist_dic_CB = {}
        for j in range(m):
            for i in range(n):
                dist_E = self.Euclid([x,y],[i,j])
                if dist_E not in dist_dic_Euclid:
                    dist_dic_Euclid[dist_E] = []
                dist_dic_Euclid[dist_E].append((i,j))
                dist_M = self.Manhattan([x,y],[i,j])
                if dist_M not in dist_dic_Man:
                    dist_dic_Man[dist_M] = []
                dist_dic_Man[dist_M].append((i,j))
                dist_CB = self.ChessBoard([x,y],[i,j])
                if dist_CB not in dist_dic_CB:
                    dist_dic_CB[dist_CB] = []
                dist_dic_CB[dist_CB].append((i,j))
        
        return dist_dic_CB,dist_dic_Euclid,dist_dic_Man

#%%
def main():
    ch = int(input('enter choice (1) for image and (2) for general image'))
    if ch == 2:
        n = int(input("enter the no of rows :"))
        m = int(input("enter the no of cols :"))
        image = np.random.randint(2, size=(n,m))
    if ch == 1:
        img = input('enter the image:')
        image = cv2.resize(cv2.imread(img,cv2.IMREAD_GRAYSCALE),(16,16))
    plt.imshow(image,cmap = 'gray')
    plt.show()
    Dis_obj = Distance()

    x = int(input('enter the x coordinate of pixel:'))
    y = int(input('enter the y coordinate of pixel:'))

    image_Euclid = np.array(image)
    image_Euclid[x,y] = 0
    image_Man = np.array(image)
    image_Man[x,y] = 0
    image_CB = np.array(image)
    image_CB[x,y] = 0

    dist_dic_CB,dist_dic_Euclid,dist_dic_Man = Dis_obj.getDistance(image, [x,y])

    Dis_obj.plotDistanceMeasure(image_Euclid, dist_dic_Euclid)
    Dis_obj.plotDistanceMeasure(image_Man, dist_dic_Man)
    Dis_obj.plotDistanceMeasure(image_CB, dist_dic_CB)
#%%
if __name__ == '__main__':
    main()

#%%
