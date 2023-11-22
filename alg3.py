import cv2
import numpy as np
from scipy.signal import fftconvolve
import pywt
import matplotlib.pyplot as plt

class Alg3WaveletTest(object):
    def startAlg(self, image_files):
        print("DummyAlgorithm2 starting.")
        img_mats = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_files]
        num_files = len(image_files)
        print(type(img_mats))
        firstimg = img_mats[0]
        firstimg = pywt.data.camera()
        
        # lastimg = img_mats[num_files-1]
        print("firstimg shape", firstimg.shape)
        print("Running wavelet decomposition.")
        print(pywt.families())
        print(pywt.wavelist('haar'))
        decomp = pywt.dwt2(firstimg, 'haar', mode='per')
        # print(decomp)
        print(type(decomp))
        print(len(decomp[0]))
        print(decomp[1][0].shape)
        print(len(decomp[1][0]))
        
        LL, (LH, HL, HH) = decomp
        print("LL",LL.shape)
        titles = ['Approximation', ' Horizontal detail',
        'Vertical detail', 'Diagonal detail']
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        plt.show()

        return LL