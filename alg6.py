import cv2
import numpy as np
from scipy.signal import fftconvolve
import pywt
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import rectangle
from skimage.filters.rank import modal, majority, maximum
from scipy.ndimage import maximum_filter
from scipy.signal import medfilt2d
import alignment

transFolder = 'TransFolder/'

class Alg6MergeTest(object):
    def startAlg(self, image_files):
        print("Algorithm5 starting.")
        # Code for algorithm 6 (from cmcguinness repository))
        print('image files', image_files)
        image_files = sorted(image_files)
        print('sorted image files', image_files)


        # focusimages = [cv2.imread(img) for img in image_files]
        focusimages = []
        print("FILES",image_files)
        for img in image_files:
            print ("Reading in file {}".format(img))
            # img = cv2.imread('step0.jpg',0)
            focusimages.append(cv2.imread("{}".format(img)))
            # focusimages.append(cv2.imread("{}".format(img), 0))

        print("Running alignment module.")
        images = alignment.align_images_compare_first(focusimages)
        print("IMAGES",focusimages)

        print ("Computing the laplacian of the blurred images")
        laps = []
        for i in range(len(images)):
            print ("Lap {}".format(i))
            laps.append(doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)))
            curr = doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))
            cv2.imwrite(transFolder + 'laplace' + str(i) + '.png', curr)

        laps = np.asarray(laps)
        print ("Shape of array of laplacians = {}".format(laps.shape))

        output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

        abs_laps = np.absolute(laps)
        maxima = abs_laps.max(axis=0)
        bool_mask = abs_laps == maxima
        mask = bool_mask.astype(np.uint8)
        for i in range(0,len(images)):
            output = cv2.bitwise_not(images[i],output, mask=mask[i])
            

        result = 255-output
        
        return result

#   Compute the gradient map of the image
def doLap(image):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
    kernel_size = 5         # Size of the laplacian window
    blur_size = 5           # How big of a kernal to use for the gaussian blur
                            # Generally, keeping these two values the same or very
                            #  close works well
                            # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)