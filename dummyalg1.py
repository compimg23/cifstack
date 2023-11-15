import cv2
import numpy as np

class DummyAlgorithm1(object):
    def startAlg(self, image_files):
        print("DummyAlgorithm1 starting.")
        img_mats = [cv2.imread(img) for img in image_files]
        num_files = len(image_files)
        firstimg = img_mats[0]
        lastimg = img_mats[num_files-1]
        print("Using maximum of each pixel in first and last image in list.")
        max_pixels = np.maximum(firstimg, lastimg)
        return max_pixels