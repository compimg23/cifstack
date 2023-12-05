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

class Alg5MergeTest(object):
    def startAlg(self, image_files):
        print("Algorithm5 starting.")
        # Code for algorithm 5 (maybe Laplacian?)