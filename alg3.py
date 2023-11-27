import cv2
import numpy as np
from scipy.signal import fftconvolve
import pywt
import matplotlib.pyplot as plt
from PIL import Image

class Alg3WaveletTest(object):
    def startAlg(self, image_files):
        print("Algorithm3 (wavelet decomposition) starting.")
        family = 'cmor'
        waveletchoice = 'haar'
        for family in pywt.families():
            print('family', family)
            if family == 'gaus':
                print('continuing')
                continue
            for waveletchoice in pywt.wavelist(family):
            # for waveletchoice in ['haar']:
                print(pywt.families())
                print(pywt.wavelist(family))
                print('waveletchoice', waveletchoice)
                # img_mats = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_files]
                img_mats = [cv2.imread(img) for img in image_files]
                num_files = len(image_files)
                # print(type(img_mats))
                firstimg = img_mats[0]
                # firstimg = pywt.data.camera()
                
                # lastimg = img_mats[num_files-1]
                print("firstimg shape", firstimg.shape)
                print("Running wavelet decomposition.")
                decomp1 = pywt.dwt2(firstimg[:,:,0], waveletchoice, mode='per')
                decomp2 = pywt.dwt2(firstimg[:,:,1], waveletchoice, mode='per')
                decomp3 = pywt.dwt2(firstimg[:,:,2], waveletchoice, mode='per')
                recompimg = np.zeros_like(firstimg)
                # print(recompimg.shape)
                recompimg[:,:,2] = pywt.idwt2(decomp1, waveletchoice, mode='per')[0:recompimg.shape[0],:]
                recompimg[:,:,1] = pywt.idwt2(decomp2, waveletchoice, mode='per')[0:recompimg.shape[0],:]
                recompimg[:,:,0] = pywt.idwt2(decomp3, waveletchoice, mode='per')[0:recompimg.shape[0],:]
                im1 = Image.fromarray((recompimg))
                # im1 = im1.convert('RGB')
                recompname = 'OutputFolder/dwt' + waveletchoice + '_recomp.jpg'
                print('Saving recomposition')
                im1.save(recompname)
                print('Recomposition saved in ' + recompname)
                # print("DECOMP",decomp1)
                # print(type(decomp1))
                # print(len(decomp1[0]))
                # print(decomp1[1][0].shape)
                # print(decomp1[0].shape)
                # print(len(decomp1[1][0]))
                
                combinedImg = np.zeros((decomp1[0].shape[0],decomp1[0].shape[1],3))
                k = 0
                print('Looping and saving decomposition figure for each channel...')
                for decomp in [decomp1, decomp2, decomp3]:
                    # print('K', k, decomp)
                    LL, (LH, HL, HH) = decomp
                    # print("LL",LL.shape)
                    titles = ['Approximation', ' Horizontal detail',
                    'Vertical detail', 'Diagonal detail']
                    fig = plt.figure(figsize=(13, 10.8))
                    for i, a in enumerate([LL, LH, HL, HH]):
                        ax = fig.add_subplot(2, 2, i + 1)
                        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
                        ax.set_title(titles[i], fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.tight_layout()
                    LL = LL / np.max(LL)
                    # print('LL', LL)
                    combinedImg[:,:,2-k] = LL
                    k += 1
                    # plt.show()
                    figname = 'OutputFolder/dwt' + str(k) + '_' + waveletchoice + '.jpg'
                    plt.savefig(figname)
                    plt.close()
                    print('Decomposition figure saved in ' + figname)

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return combinedImg * 255