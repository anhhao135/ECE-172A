from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from skimage import exposure




def computeNormGrayHistogram(inputRGBImage):

    flatSize = inputRGBImage.shape[0] * inputRGBImage.shape[1]

    histogram = np.zeros(32)

    for row in inputRGBImage:
        for col in row:
            luminosity = 0.3 * col[0] +  0.59 * col[1] + 0.11 * col[2] #convert rgb to grayscale using weighted method, yielding luminosity scalar
            binIndex = int(np.floor(luminosity / 8)) #turn grayscale image to bin index of histogram
            histogram[binIndex] = histogram[binIndex] + 1 #increment bin

    histogram = histogram / flatSize #normalize histogram bins by dividing by total pixel count

    return histogram #return histogram as 32 vector


def computeNormRGBHistogram(inputRGBImage):

    flatSize = inputRGBImage.shape[0] * inputRGBImage.shape[1]

    histogram = np.zeros([3,32]) #2d vector, where 3 rows represent the 32 vectors for each channel

    for row in inputRGBImage:
        for col in row:
            for i in range(3): #iterate through RGB
                value = col[i]
                binIndex = int(np.floor(value / 8)) #turn grayscale image to bin index of histogram
                histogram[i][binIndex] = histogram[i][binIndex] + 1 #increment bin corresponding to color

    histogram = histogram / flatSize #normalize histogram bins by dividing by total pixel count

    return np.concatenate((histogram[0],histogram[1],histogram[2]),axis=None) #unfold the 2d array into a 1d array of 96 elements


def plotHistogram(histogram, title, mode):

    if mode == "gray":
        bins = np.linspace(0,31,32)

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


def AHE(image, winsize): #image is grayscale to pixel values are from 0 to 1 as a float

    imagePadded = np.pad(image, pad_width=((winsize, winsize), (winsize, winsize)), mode='symmetric') #pad around symmetrically with window size which is more than enough

    outputImage = np.zeros(image.shape)

    for row in range(image.shape[0]):

        print(row) #just to keep track of the current row and see progress

        for col in range(image.shape[1]): #by iterating through the row and column, we are going through each pixel coordinate on the original image

            value = image[row][col]

            rank = 0

            x = np.linspace(col - (winsize-1)/2, col + (winsize-1)/2, winsize) #create meshgrid x y values centered around window center
            y = np.linspace(row - (winsize-1)/2, row + (winsize-1)/2, winsize)


            for i in range(winsize):
                for j in range(winsize): #now we are going through all the possible pixels within the window

                    x_padded = int(x[j] + winsize) #add offset to pixel coordinate since we have padded image by the window size all around
                    y_padded = int(y[i] + winsize)

                    sampledWindowValue = imagePadded[y_padded][x_padded]
                    
                    
                    if value > sampledWindowValue:
                        rank = rank + 1

            outputImage[row][col] = rank / (winsize * winsize) #since the max value is 1, we don't see it in the formula


    return outputImage





image = image.imread('forest.jpg')


grayHistogram = computeNormGrayHistogram(image)

plotHistogram(grayHistogram, 'gray histogram', 'gray')

RGBHistogram = computeNormRGBHistogram(image)

plotHistogram(RGBHistogram, 'rgb histogram', 'rgb')



imageFlippedHorizontally = np.fliplr(image)

grayHistogramFlipped = computeNormGrayHistogram(imageFlippedHorizontally)

plotHistogram(grayHistogramFlipped, 'gray histogram flipped', 'gray')

RGBHistogramFlipped = computeNormRGBHistogram(imageFlippedHorizontally)

plotHistogram(RGBHistogramFlipped, 'rgb histogram flipped', 'rgb')




imageDoubleReds = image
imageDoubleReds[:,:,0] = 2 * imageDoubleReds[:,:,0]

plt.imshow(imageDoubleReds)
plt.show()

grayHistogramDoubleReds = computeNormGrayHistogram(imageDoubleReds)

plotHistogram(grayHistogramDoubleReds, 'gray histogram double reds', 'gray')

RGBHistogramDoubleReds = computeNormRGBHistogram(imageDoubleReds)

plotHistogram(RGBHistogramDoubleReds, 'rgb histogram double reds', 'rgb')




image = image.imread('beach.png')

imageHE = exposure.equalize_hist(image)


plt.imsave("HE.png", imageHE, cmap='gray')

outImage = AHE(image,33)

plt.imsave("33.png", outImage, cmap='gray')

outImage = AHE(image,65)

plt.imsave("65.png", outImage, cmap='gray')

outImage = AHE(image,129)

plt.imsave("129.png", outImage, cmap='gray')


