from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from skimage import exposure
import cv2
import time
from threading import Thread


def computeNormGrayHistogram(inputRGBImage):

    print(inputRGBImage)

    flatSize = inputRGBImage.shape[0] * inputRGBImage.shape[1]

    histogram = np.zeros(256)

    for row in inputRGBImage:
        for col in row:
            luminosity = col[0]
            binIndex = int(np.floor(luminosity)) #turn grayscale image to bin index of histogram
            histogram[binIndex] = histogram[binIndex] + 1 #increment bin

    histogram = histogram / flatSize #normalize histogram bins by dividing by total pixel count

    return histogram #return histogram as 32 vector




def computeNormGrayHistogramRGBIn(inputRGBImage):

    

    flatSize = inputRGBImage.shape[0] * inputRGBImage.shape[1]

    histogram = np.zeros(256)

    for row in inputRGBImage:
        for col in row:
            luminosity = 0.3 * col[0] +  0.59 * col[1] + 0.11 * col[2] #convert rgb to grayscale using weighted method, yielding luminosity scalar
            binIndex = int(np.floor(luminosity) )#turn grayscale image to bin index of histogram
            histogram[binIndex] = histogram[binIndex] + 1 #increment bin

    histogram = histogram / flatSize #normalize histogram bins by dividing by total pixel count

    return histogram #return histogram as 32 vector




def plotHistogram(histogram, title, mode):

    if mode == "gray":
        bins = np.linspace(0,255,256)

        plt.bar(bins,histogram,color='gray')

        plt.title(title)
        plt.xlabel('bin')
        plt.ylabel('normalized frequency')

        plt.show()
    
    if mode == "rgb":
        bins = np.linspace(0,95,96)

        plt.bar(bins,histogram,color=np.repeat(['red','green','blue'],32))

        plt.title(title)
        plt.xlabel('bin')
        plt.ylabel('normalized frequency')

        plt.show()


def medianFilter(inputImage, kernelSize): #input image is RGB

    

    imagePadded = np.pad(inputImage, pad_width=((kernelSize, kernelSize), (kernelSize, kernelSize)), mode='symmetric') #pad around symmetrically with window size which is more than enough

    outputImage = np.zeros(inputImage.shape)

    for row in range(inputImage.shape[0]):


        for col in range(inputImage.shape[1]): #by iterating through the row and column, we are going through each pixel coordinate on the original image

            value = inputImage[row][col]

            rowOffset = row + kernelSize
            colOffset = col + kernelSize #center of the kernel on the padded image space


            kernelValues = imagePadded[int(rowOffset - (kernelSize-1)/2) : int(rowOffset + (kernelSize-1)/2 + 1), int(colOffset - (kernelSize-1)/2) : int(colOffset + (kernelSize-1)/2 + 1)] #extract the kernel by slicing
            
            kernelValues = np.sort(kernelValues.flatten()) #sort in ascending order to take middle value as median

            outputImage[row][col] = kernelValues[int(np.floor((kernelSize*kernelSize)/2))] #extract the middle value of the kernel values to get median


    return outputImage


def meanFilter(inputImage, kernelSize): #input image is RGB

    imagePadded = np.pad(inputImage, pad_width=((kernelSize, kernelSize), (kernelSize, kernelSize)), mode='symmetric') #pad around symmetrically with window size which is more than enough

    outputImage = np.zeros(inputImage.shape)

    for row in range(inputImage.shape[0]):


        for col in range(inputImage.shape[1]): #by iterating through the row and column, we are going through each pixel coordinate on the original image

            value = inputImage[row][col]

            rowOffset = row + kernelSize
            colOffset = col + kernelSize #center of the kernel on the padded image space


            kernelValues = imagePadded[int(rowOffset - (kernelSize-1)/2) : int(rowOffset + (kernelSize-1)/2 + 1), int(colOffset - (kernelSize-1)/2) : int(colOffset + (kernelSize-1)/2 + 1)] #extract the kernel by slicing
            
            kernelValues = np.sort(kernelValues.flatten())

            outputImage[row][col] = np.average(kernelValues) #new pixel value is the average of the kernel
    
    return outputImage









image_ = image.imread('mural.jpg')

grayHistogram = computeNormGrayHistogramRGBIn(image_)

plotHistogram(grayHistogram, 'mural original histogram', 'gray')


image_ = image.imread('mural_noise1.jpg')

grayHistogram = computeNormGrayHistogram(image_)

plotHistogram(grayHistogram, 'mural_noise1 histogram', 'gray')

image_ = image.imread('mural_noise2.jpg')

grayHistogram = computeNormGrayHistogram(image_)

plotHistogram(grayHistogram, 'mural_noise2 histogram', 'gray')




def runFilters(imagePath,name):

    image_ = cv2.imread(imagePath)
    imageGray = cv2.cvtColor(image_,cv2.COLOR_BGR2GRAY)

    start = time.time()
    imageFiltered = meanFilter(imageGray,5)
    plt.imsave(name + "mean5.png", imageFiltered, cmap='gray')
    end = time.time()
    print(imagePath + " 5x5 mean filter took " + str(end - start) + " seconds")

    start = time.time()
    imageFiltered = meanFilter(imageGray,21)
    plt.imsave(name + "mean21.png", imageFiltered, cmap='gray')
    end = time.time()
    print(imagePath + " 21x21 mean filter took " + str(end - start) + " seconds")

    start = time.time()
    imageFiltered = medianFilter(imageGray,5)
    plt.imsave(name + "median5.png", imageFiltered, cmap='gray')
    end = time.time()
    print(imagePath + " 5x5 median filter took " + str(end - start) + " seconds")

    start = time.time()
    imageFiltered = medianFilter(imageGray,21)
    plt.imsave(name + "median21.png", imageFiltered, cmap='gray')
    end = time.time()
    print(imagePath + " 21x21 median filter took " + str(end - start) + " seconds")





image_ = cv2.imread('mural_noise1mean5.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise1 mean 5x5', 'gray')

image_ = cv2.imread('mural_noise1mean21.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise1 mean 21x21', 'gray')

image_ = cv2.imread('mural_noise1median5.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise1 median 5x5', 'gray')

image_ = cv2.imread('mural_noise1median21.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise1 median 21x21', 'gray')

image_ = cv2.imread('mural_noise2mean5.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise2 mean 5x5', 'gray')

image_ = cv2.imread('mural_noise2mean21.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise2 mean 21x21', 'gray')

image_ = cv2.imread('mural_noise2median5.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise2 median 5x5', 'gray')

image_ = cv2.imread('mural_noise2median21.png')
grayHistogram = computeNormGrayHistogram(image_)
plotHistogram(grayHistogram, 'mural_noise2 median 21x21', 'gray')





runFilters("mural_noise1.jpg", "mural_noise1")
runFilters("mural_noise2.jpg", "mural_noise2")



template = cv2.imread("template.jpg",0)
mural = cv2.imread("mural.jpg",0)
result = cv2.matchTemplate(mural,template,cv2.TM_CCORR_NORMED)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
mural = cv2.rectangle(mural, maxLoc - np.array([50,50]), maxLoc + np.array([50,50]),(255,255,255),3)
plt.imshow(mural,cmap="gray")
plt.show()

template = cv2.imread("template.jpg",0)
mural = cv2.imread("mural.jpg",0)
result = cv2.matchTemplate(mural,template,cv2.TM_CCORR)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
mural = cv2.rectangle(mural, maxLoc - np.array([50,50]), maxLoc + np.array([50,50]),(255,255,255),3)
plt.imshow(mural,cmap="gray")
plt.show()


