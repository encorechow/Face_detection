import numpy as np
import cv2
import load_images as li

w = 64
h = 64


def HOG(image):

        winSize = (w,h)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 1
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

        hist = hog.compute(image)
        hist = hist.transpose()
        result = hist[0]
        return result


def resizeImages(images):
        res = []
        try:
                for image in images:
                        res.append(cv2.resize(image, (w,h), interpolation = cv2.INTER_CUBIC))
                return np.array(res)
        except Exception, e:
                print e

#if __name__ == "__main__":
#        images = li.load("orl_faces")
#        reImages = resizeImages(images)
#
#        hog = HOG(reImages[0])
#
#        print hog

