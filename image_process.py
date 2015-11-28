import imutils

def pyramid(image, pyramid_scale=1.5, minSize=(64, 64)):
        yield image
        while True:
                w = int(image.shape[1] / pyramid_scale)
                image = imutils.resize(image, width=w)
                if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                        break

                yield image

def sliding_window(image, stepSize, windowSize):
        for y in xrange(0, image.shape[0], stepSize):
                for x in xrange(0, image.shape[1], stepSize):
                        yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


