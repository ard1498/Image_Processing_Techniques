# shortest path and connected components code
# Author : Anirudh Aggarwal

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

class paths:
    def __init__(self):
        self.pathdic = {}
        self.shortPath = []

    def get_surround_pixels(self, ch, i, j, image, V):
        # 0 : N4 , 1 : ND, 2 : N8, 3 : M
        Pixel_list = []
        n,m = image.shape[0],image.shape[1]
        if ch == 0:
            # N4
            # row = [0,1,0,-1]
            # col = [1,0,-1,0]
            if image[i,j] not in V:
                return Pixel_list
            if i+1 < n and image[i+1,j] in V:
                Pixel_list.append([i+1,j])
            if j+1 < m and image[i,j+1] in V:
                Pixel_list.append([i,j+1])
            if j-1 >= 0 and image[i,j-1] in V :
                Pixel_list.append([i,j-1])
            if i-1 >= 0 and image[i-1,j] in V:
                Pixel_list.append([i-1,j])
        elif ch == 1:
            # Nd
            # row = [1,1,-1,1]
            # col = [1,-1,-1,1]
            if image[i,j] not in V:
                return Pixel_list
            if j+1 < m:
                if i+1 < n and image[i+1,j+1] in V:
                    Pixel_list.append([i+1,j+1])
                if i-1 >= 0 and image[i-1,j+1] in V:
                    Pixel_list.append([i-1,j+1])
            if j-1 >= 0:
                if i+1 < n and image[i+1,j-1] in V:
                    Pixel_list.append([i+1,j-1])
                if i-1 >= 0 and image[i-1,j+1] in V:
                    Pixel_list.append([i-1,j-1])
        elif ch == 2:
            # N8
            # row = [-1,0,1,-1,1,-1,0,1]
            # col = [-1,-1,-1,0,0,1,1,1]
            if image[i,j] not in V:
                return Pixel_list
            if i+1 < n and image[i+1,j] in V:
                Pixel_list.append([i+1,j])
            if j+1 < m :
                if image[i,j+1] in V:
                    Pixel_list.append([i,j+1])
                if i+1 < n and image[i+1,j+1] in V:
                    Pixel_list.append([i+1,j+1])
                if i-1 >= 0 and image[i-1,j+1] in V:
                    Pixel_list.append([i-1,j+1])
            if j-1 >= 0 :
                if image[i,j-1] in V :
                    Pixel_list.append([i,j-1])
                if i+1 < n and image[i+1,j-1] in V:
                    Pixel_list.append([i+1,j-1])
                if i-1 >= 0 and image[i-1,j+1] in V:
                    Pixel_list.append([i-1,j-1])
            if i-1 >= 0 and image[i-1,j] in V:
                Pixel_list.append([i-1,j])
        elif ch == 3:
            # M connected
            if image[i,j] not in V:
                return Pixel_list
            if i+1 < n and image[i+1,j] in V:
                Pixel_list.append([i+1,j])
            if j+1 < m and image[i,j+1] in V:
                Pixel_list.append([i,j+1])
            if j-1 >= 0 and image[i,j-1] in V :
                Pixel_list.append([i,j-1])
            if i-1 >= 0 and image[i-1,j] in V:
                Pixel_list.append([i-1,j])
            # now checking Nd(p) and N4(p) and N4(q)
            if j+1 < m:
                if i+1 < n and image[i+1,j+1] in V:
                    if image[i+1,j] not in V and image[i,j+1] not in V:
                        Pixel_list.append([i+1,j+1])
                if i-1 >= 0 and image[i-1,j+1] in V:
                    if image[i-1,j] not in V and image[i,j+1] not in V:
                        Pixel_list.append([i-1,j+1])
            if j-1 >= 0:
                if i+1 < n and image[i+1,j-1] in V:
                    if image[i+1,j] not in V and image[i,j-1] not in V:
                        Pixel_list.append([i+1,j-1])
                if i-1 >= 0 and image[i-1,j-1] in V:
                    if image[i-1,j] not in V and image[i,j-1] not in V:
                        Pixel_list.append([i-1,j-1])
        return Pixel_list
    
    def plotshortestpath(self, image, pathlist, From):
        for j in range(image.shape[1]):
            for i in range(image.shape[0]):
                if [i,j] in self.shortPath:
                    image[i,j] = 3
                elif [i,j] in pathlist:
                    image[i,j] = 6
                else:
                    image[i,j] = 9
        image[From[0],From[1]] = 0
        plt.grid()
        plt.imshow(image, cmap = 'gray')
        plt.show()
    
    def connectedPixels(self, image, pathlist, From):
        for j in range(image.shape[1]):
            for i in range(image.shape[0]):
                if [i,j] in pathlist:
                    image[i,j] = 10
                else:
                    image[i,j] = 5
        image[From[0],From[1]] = 0
        plt.imshow(image, cmap = 'gray')
        plt.show()

    def getpaths(self, image, From, To, ch):
        x1,y1 = From[0],From[1]
        print('From :',[x1,y1])
        V = [image[x1,y1]]
        x2,y2 = To[0],To[1]
        print('To :',[x2,y2])
        n = image.shape[0]
        m = image.shape[1]
        visited = [[False for j in range(m)] for i in range(n)]
    
        que = []
        que.append([x1,y1])
        shorted_dist,cur_path_len = math.inf,0
        Flag = False
        
        self.pathdic[(x1,y1)] = [None,None]
        path_list = [[x1,y1]]
        while(len(que) > 0):
            csize = len(que)
            cur_path_len += 1
            while csize > 0:
                csize -= 1
                x,y = que.pop(0)
                if visited[x][y] == True:
                    continue
                path_list.append([x,y])
                # print('x is :',x,' y is :', y)
                visited[x][y] = True
                Pixel_list = self.get_surround_pixels(ch, x, y, image, V)
                for k in Pixel_list:
                    nxt_i = k[0]
                    nxt_j = k[1]

                    if [nxt_i,nxt_j] == [x2,y2]:    
                        Flag = True
                        if cur_path_len < shorted_dist:
                            self.pathdic[(nxt_i,nxt_j)] = [x,y]
                            shorted_dist = cur_path_len
                        
                    if visited[nxt_i][nxt_j] == False:
                        que.append([nxt_i,nxt_j])
                        if (nxt_i,nxt_j) not in self.pathdic:
                            self.pathdic[(nxt_i,nxt_j)] = [x,y]
                # print('queue is :',que)
        
        if Flag == False:
            print("no path found")
            return None,path_list,shorted_dist    
        print('Path/s Found')
        self.assignshortestPath(image, From, To)
        path_list.append([x2,y2])
        return 'Yes',path_list,shorted_dist
    
    def assignshortestPath(self, image, From, To):
        x2,y2 = To[0],To[1]
        Path = [[x2,y2]]
        x,y = x2,y2
        
        while x != None and y != None:
            Path.append([x,y])
            x_y = self.pathdic[(x,y)]
            x,y = x_y[0],x_y[1]
        self.shortPath = Path

def main():

    ch = input('enter image(1) OR No of Pixels in each direction(2)')
    if ch == '1':
        image_name = input('enter the image name : ')
        image = cv2.resize(cv2.imread(image_name,cv2.IMREAD_GRAYSCALE),(32,32))
        print(image.shape)
        cv2.imshow('Used Image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        n = int(input("enter no of rows:"))
        m = int(input("enter no of cols:"))

        image = np.random.randint(2, size=(n,m))
        cv2.imshow('Used Image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    plt.grid()
    plt.imshow(image, cmap='gray')
    plt.show()
    [x1,y1] = [int(i) for i in input("enter From Pixel(x1,y1):").strip().split(',')]
    [x2,y2] = [int(i) for i in input("enter To Pixel(x2,y2):").strip().split(',')]
    
    # For N4
    print('This is for N4 connectivity.')
    p4 = paths()
    image_copy_4,image_print_4 = np.array(image),np.array(image)
    IsFound_4,path_list_4,shorted_dist_4 = p4.getpaths(image_copy_4,[x1,y1],[x2,y2],0)
    if IsFound_4 == None:
        print('shortest distance is inf.')
        p4.connectedPixels(image_print_4, path_list_4, [x1,y1])
    else:
        print('shortest path distance is :',shorted_dist_4)
        p4.plotshortestpath(image_print_4, path_list_4, [x1,y1])
    
    # For Nd
    print('This is for Nd connectivity.')
    pd = paths()
    image_copy_d,image_print_d = np.array(image),np.array(image)
    IsFound_d,path_list_d,shorted_dist_d = pd.getpaths(image_copy_d,[x1,y1],[x2,y2],1)
    if IsFound_d == None:
        print('shortest distance is inf.')
        pd.connectedPixels(image_print_d, path_list_d, [x1,y1])
    else:
        print('shortest path distance is :',shorted_dist_d)
        pd.plotshortestpath(image_print_d, path_list_d, [x1,y1])
    
    # For N8
    print('This is for N8 connectivity.')
    p8 = paths()
    image_copy_8,image_print_8 = np.array(image),np.array(image)
    IsFound_8,path_list_8,shorted_dist_8 = p8.getpaths(image_copy_8,[x1,y1],[x2,y2],2)
    if IsFound_8 == None:
        print('shortest distance is inf.')
        p8.connectedPixels(image_print_8, path_list_8, [x1,y1])
    else:
        print('shortest path distance is :',shorted_dist_8)
        p8.plotshortestpath(image_print_8, path_list_8, [x1,y1])
    
    # For M
    print('This is for M connectivity.')
    pM = paths()
    image_copy_M,image_print_M = np.array(image),np.array(image)
    IsFound_M,path_list_M,shorted_dist_M = pM.getpaths(image_copy_M,[x1,y1],[x2,y2],0)
    if IsFound_M == None:
        print('shortest distance is inf.')
        pM.connectedPixels(image_print_M, path_list_M, [x1,y1])
    else:
        print('shortest path distance is :',shorted_dist_M)
        pM.plotshortestpath(image_print_M, path_list_M, [x1,y1])

if __name__ == '__main__':
    main()
