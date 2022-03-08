from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from skimage import exposure
import cv2
import matplotlib.cm as cm


def GaussianSmoothing(inputImage):

    kernel = (1/159) * np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])

    kernelSize = kernel.shape[0]

    imagePadded = np.pad(inputImage, pad_width=((kernelSize, kernelSize), (kernelSize, kernelSize)), mode='symmetric') #pad around symmetrically with window size which is more than enough

    outputImage = np.zeros(inputImage.shape)

    for row in range(inputImage.shape[0]):

        print(row)


        for col in range(inputImage.shape[1]): #by iterating through the row and column, we are going through each pixel coordinate on the original image

            value = inputImage[row][col]

            rowOffset = row + kernelSize
            colOffset = col + kernelSize #center of the kernel on the padded image space

            kernelSampledValues = imagePadded[int(rowOffset - (kernelSize-1)/2) : int(rowOffset + (kernelSize-1)/2 + 1), int(colOffset - (kernelSize-1)/2) : int(colOffset + (kernelSize-1)/2 + 1)] #extract the kernel by slicing

            elementwiseMultiply = np.multiply(kernel,kernelSampledValues) #multiply kernel values by Gaussian weights

            outputImage[row][col] = np.sum(elementwiseMultiply) #return the sum of the kernel values
    
    return outputImage

def Sobel(inputImage):

    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) #gradient kernels

    kernelSize = kx.shape[0]

    imagePadded = np.pad(inputImage, pad_width=((kernelSize, kernelSize), (kernelSize, kernelSize)), mode='symmetric') #pad around symmetrically with window size which is more than enough


    outputGradientMagnitudes = np.zeros(inputImage.shape)
    outputGradientAngles = np.zeros(inputImage.shape)
    outputCombinedImage = np.zeros(inputImage.shape)


    for row in range(inputImage.shape[0]):

        print(row)

        for col in range(inputImage.shape[1]): #by iterating through the row and column, we are going through each pixel coordinate on the original image

            value = inputImage[row][col]

            rowOffset = row + kernelSize
            colOffset = col + kernelSize #center of the kernel on the padded image space

            kernelSampledValues = imagePadded[int(rowOffset - (kernelSize-1)/2) : int(rowOffset + (kernelSize-1)/2 + 1), int(colOffset - (kernelSize-1)/2) : int(colOffset + (kernelSize-1)/2 + 1)] #extract the kernel by slicing

            elementwiseMultiplyKx = np.multiply(kx,kernelSampledValues)
            Gx = np.sum(elementwiseMultiplyKx) #multiply by kernel, then sum values to get value of gradient

            elementwiseMultiplyKy = np.multiply(ky,kernelSampledValues)
            Gy = np.sum(elementwiseMultiplyKy)

            outputGradientMagnitudes[row][col] = np.sqrt(np.square(Gx) + np.square(Gy)) #get magnitude by performing Pythagorean theorem
            outputGradientAngles[row][col] = np.arctan(Gy / Gx) #get angle in radians with arctan; this is limited to -pi/2 to pi/2

            if outputGradientMagnitudes[row][col] > 100: #to build the magnitude-gradient image, we only want to show the pixels above a certain threshold, since angle image alone is very noisy
                outputCombinedImage[row][col] = (outputGradientAngles[row][col] + (np.pi / 2)) / (np.pi) #first add pi/2 to angle so now it is always positive from 0 to pi, then normalize to 0-1 range by dividing by pi

    return outputGradientMagnitudes, outputGradientAngles, outputCombinedImage #return all three images containing different information extracted using Sobel


def NMS(inputMagnitudesImage,inputAngleImages):
    
    inputMagnitudesImagePadded = np.pad(inputMagnitudesImage, pad_width=((1,1), (1,1)), mode='symmetric') #pad 1 pixel around image symmetrically so we can go value compare in gradient direction along edges

    outputMagntiudeImage = np.zeros(inputAngleImages.shape)

    for row in range(inputMagnitudesImage.shape[0]):

        print(row)

        for col in range(inputMagnitudesImage.shape[1]): #by iterating through the row and column, we are going through each pixel coordinate on the original image

            rowOffset = row + 1
            colOffset = col + 1 #absolute coordinates in padded image space; basically just add 1 to both
            pixelValueTarget = inputMagnitudesImagePadded[rowOffset][colOffset]

            direction = getAngleDirection(inputAngleImages[row][col]) #get value from 0 to 3 to determine direction of gradient and which neighboring pixels to check

            if direction == 0: #NS
                pixelValue1 = inputMagnitudesImagePadded[rowOffset+1][colOffset]
                pixelValue2 = inputMagnitudesImagePadded[rowOffset-1][colOffset]
                if pixelValue1 > pixelValueTarget or pixelValue2 > pixelValueTarget: #if either of the neighboring pixels has a higher value, the target pixel cannot be a local max, so suppress it
                    outputMagntiudeImage[row][col] = 0
                else:
                    outputMagntiudeImage[row][col] = pixelValueTarget

            if direction == 1: #SE-NW
                pixelValue1 = inputMagnitudesImagePadded[rowOffset-1][colOffset+1]
                pixelValue2 = inputMagnitudesImagePadded[rowOffset+1][colOffset-1]
                if pixelValue1 > pixelValueTarget or pixelValue2 > pixelValueTarget:
                    outputMagntiudeImage[row][col] = 0
                else:
                    outputMagntiudeImage[row][col] = pixelValueTarget

            if direction == 2: #EW
                pixelValue1 = inputMagnitudesImagePadded[rowOffset][colOffset+1]
                pixelValue2 = inputMagnitudesImagePadded[rowOffset][colOffset-1]
                if pixelValue1 > pixelValueTarget or pixelValue2 > pixelValueTarget:
                    outputMagntiudeImage[row][col] = 0
                else:
                    outputMagntiudeImage[row][col] = pixelValueTarget

            if direction == 3: #NE-SW
                pixelValue1 = inputMagnitudesImagePadded[rowOffset+1][colOffset+1]
                pixelValue2 = inputMagnitudesImagePadded[rowOffset-1][colOffset-1]
                if pixelValue1 > pixelValueTarget or pixelValue2 > pixelValueTarget:
                    outputMagntiudeImage[row][col] = 0
                else:
                    outputMagntiudeImage[row][col] = pixelValueTarget

    return outputMagntiudeImage

def getAngleDirection(inputAngle): #angle is output of arctan in radians, so is limited to -pi/2 and pi/2

    #return direction enums: 0: NS, 1:SE-NW, 2:EW, 3:NE-SW

    inputAngleOffset = inputAngle + np.pi/2 #add pi/2 so now angle is within 0 to pi and always positive

    rounded = np.round(inputAngleOffset / (np.pi/4)) #divide by 45 degrees and round to nearest integer to determine which direction is nearest

    if rounded == 0 or rounded == 4:
        return 0
    else:
        return rounded
    


lane = cv2.imread("lane.png",0)
laneSmooth = GaussianSmoothing(lane)
laneMag,laneAngle,laneComb = Sobel(laneSmooth)
laneMagSuppressed = NMS(laneMag,laneAngle)

plt.imshow(laneSmooth,cmap="gray")
plt.show()
plt.imshow(laneMag,cmap="gray")
plt.show()
plt.imshow(laneAngle,cmap="gray")
plt.show()
plt.imshow(laneComb,cmap="viridis")
plt.show()
plt.imshow(laneMagSuppressed,cmap="gray")
plt.show()
plt.imshow((laneMagSuppressed > 110) * laneMagSuppressed,cmap="gray")
plt.show()

