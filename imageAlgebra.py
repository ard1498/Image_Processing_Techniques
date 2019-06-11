#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%
class Algebra:
    
    def getXor(self,image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                b = image1[i,j][0] ^ image2[i,j][0]
                g = image1[i,j][1] ^ image2[i,j][1]
                r = image1[i,j][2] ^ image2[i,j][2]
                new_img[i,j] = [b,g,r]
        return new_img
    
    def getAnd(self,image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                b = image1[i,j][0] & image2[i,j][0]
                g = image1[i,j][1] & image2[i,j][1]
                r = image1[i,j][2] & image2[i,j][2]
                new_img[i,j] = [b,g,r]
        return new_img

    def getOr(self,image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                b = image1[i,j][0] | image2[i,j][0]
                g = image1[i,j][1] | image2[i,j][1]
                r = image1[i,j][2] | image2[i,j][2]
                new_img[i,j] = [b,g,r]
        return new_img

    def getAddn(self, image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                new_img[i,j][0] = image1[i,j][0] + image2[i,j][0]
                new_img[i,j][1] = image1[i,j][1] + image2[i,j][1]
                new_img[i,j][2] = image1[i,j][2] + image2[i,j][2]
        return new_img
    
    def getSubtract(self, image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                new_img[i,j][0] = ((image1[i,j][0] - image2[i,j][0]))
                new_img[i,j][1] = ((image1[i,j][1] - image2[i,j][1]))
                new_img[i,j][2] = ((image1[i,j][2] - image2[i,j][2]))
        return new_img
    
    def getMultiplyImages(self, image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                new_img[i,j][0] = ((image1[i,j][0] * image2[i,j][0]))
                new_img[i,j][1] = ((image1[i,j][1] * image2[i,j][1]))
                new_img[i,j][2] = ((image1[i,j][2] * image2[i,j][2]))
        return new_img

    def getMultiply(self, image1, k):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image1.shape[1]
        for i in range(n):
            for j in range(m):
                new_img[i,j][0] = ((image1[i,j][0] * k))
                new_img[i,j][1] = ((image1[i,j][1] * k))
                new_img[i,j][2] = ((image1[i,j][2] * k))
        return new_img

    def getDivisionImages(self, image1, image2):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image2.shape[1]
        for i in range(n):
            for j in range(m):
                new_img[i,j][0] = ((image1[i,j][0] // image2[i,j][0])%256 + 256) % 256
                new_img[i,j][1] = ((image1[i,j][1] // image2[i,j][0])%256 + 256) % 256
                new_img[i,j][2] = ((image1[i,j][2] // image2[i,j][0])%256 + 256) % 256
        return new_img
    
    def getDivision(self, image1, k):
        new_img = np.zeros(image1.shape)
        n = image1.shape[0]
        m = image1.shape[1]
        for i in range(n):
            for j in range(m):
                new_img[i,j][0] = ((image1[i,j][0] // k)%256 + 256) % 256
                new_img[i,j][1] = ((image1[i,j][1] // k)%256 + 256) % 256
                new_img[i,j][2] = ((image1[i,j][2] // k)%256 + 256) % 256
        return new_img

def printimage(name, image):
    print(f"{name} :")
    plt.imshow(image)
    plt.show()

def main():
    im_name1 = input('enter the image 1 :')
    image1 = cv2.resize(cv2.imread(im_name1),(300,300))
    
    printimage('Original Image 1', image1)
    cv2.imshow('Original Image 1', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im_name2 = input('enter the image 2 :')
    image2 = cv2.resize(cv2.imread(im_name2),(300,300))
    
    printimage('Original Image 2', image2)
    cv2.imshow('Original Image 2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    alg_obj = Algebra()
    image_addn1, image_addn2 = image1.copy(),image2.copy()
    image_sub1, image_sub2 = image1.copy(),image2.copy()
    image_multi1, image_multi2 = image1.copy(),image2.copy()
    image_div1, image_div2 = image1.copy(),image2.copy()
    image_divScale1, image_MultiScale = image1.copy(),image1.copy()

    printimage('And of Images',alg_obj.getAnd(image1.copy(),image2.copy()))
    cv2.imshow('And of Images',alg_obj.getAnd(image1.copy(),image2.copy()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    printimage('Or of Images',alg_obj.getOr(image1.copy(),image2.copy()))
    cv2.imshow('Or of Images',alg_obj.getOr(image1.copy(),image2.copy()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    printimage('Xor of Images',alg_obj.getXor(image1.copy(),image2.copy()))
    cv2.imshow('Xor of Images',alg_obj.getXor(image1.copy(),image2.copy()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    printimage('Addition of images',alg_obj.getAddn(image_addn1.copy(),image_addn2.copy()))
    cv2.imshow('Addition of images',alg_obj.getAddn(image_addn1,image_addn2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    printimage('Subtraction of Images',alg_obj.getSubtract(image_sub1.copy(),image_sub2.copy()))
    cv2.imshow('Subtraction of Images',alg_obj.getSubtract(image_sub1,image_sub2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    printimage('Multiplication with scalar',alg_obj.getMultiply(image_MultiScale.copy(),10))
    cv2.imshow('Multiplication with scalar',alg_obj.getMultiply(image_MultiScale,10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    printimage('Multiplication of Images',alg_obj.getMultiplyImages(image_multi1.copy(),image_multi2.copy()))
    cv2.imshow('Multiplication of Images',alg_obj.getMultiplyImages(image_multi1,image_multi2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    printimage('Division with scalar',alg_obj.getDivision(image_divScale1.copy(),10))
    cv2.imshow('Division with scalar',alg_obj.getDivision(image_divScale1,10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    printimage('Division of images',alg_obj.getDivisionImages(image_div1.copy(), image_div2.copy()))
    cv2.imshow('Division of images',alg_obj.getDivisionImages(image_div1, image_div2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%%
if __name__ == '__main__':
    main()