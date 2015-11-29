import numpy as np
import cv2
import os

def _loadi(filename):
        try:
                return cv2.imread(filename, 0)
        except:
                print "file name incorrect"


def load(directory):
        images = []
        try:
                for (dirpath, dirnames, filenames) in os.walk(directory):
                        for filename in filenames:
                                if filename[-4:len(filename)] == '.pgm' or \
                                        filename[-4:len(filename)] == '.jpg' or\
                                        filename[-4:len(filename)] == '.png':
                                        images.append(_loadi(os.path.join(dirpath, filename)))
                return np.array(images)
        except:
                print "directory does not exist"

#if __name__ == "__main__":
#        images = load("orl_faces")
#        print images


