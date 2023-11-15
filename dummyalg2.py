import cv2
import numpy as np

class DummyAlgorithm2(object):
    def startAlg(self, image_files):
        print("DummyAlgorithm2 starting.")
        img_mats = [cv2.imread(img) for img in image_files]
        num_files = len(image_files)
        firstimg = img_mats[0]
        lastimg = img_mats[num_files-1]
        print("Using minimum of each pixel in first and last image in list.")
        min_pixels = np.minimum(firstimg, lastimg)
        return min_pixels