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

class Alg4MergeTest(object):
    def startAlg(self, image_files):
        print("Algorithm4 (wavelet decomposition) starting.")
        print('image files', image_files)
        image_files = sorted(image_files)
        print('sorted image files', image_files)
        family = 'cmor'
        waveletchoice = 'haar'
        for family in ['haar']: #pywt.families():
            print('family', family)
            if family == 'gaus':
                print('continuing')
                continue
            for waveletchoice in pywt.wavelist(family):
                img_mats = [cv2.imread(img) for img in image_files]
                num_files = len(image_files)
                print('numfile', num_files)
                firstimg = img_mats[0]
                decomp1 = pywt.dwt2(firstimg[:,:,0], waveletchoice, mode='per')
                decomp2 = pywt.dwt2(firstimg[:,:,1], waveletchoice, mode='per')
                decomp3 = pywt.dwt2(firstimg[:,:,2], waveletchoice, mode='per')
                newdecomp1 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                newdecomp2 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                newdecomp3 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                # print('newdecomp0', newdecomp1[0].shape)
                # print('newdecomp1', decomp2[1][0].shape)
                for j in range(num_files):
                    currimg = img_mats[j]
                    print("firstimg shape", firstimg.shape)
                    print("Running wavelet decomposition.")
                    decomp1 = pywt.dwt2(currimg[:,:,0], waveletchoice, mode='per')
                    decomp2 = pywt.dwt2(currimg[:,:,1], waveletchoice, mode='per')
                    decomp3 = pywt.dwt2(currimg[:,:,2], waveletchoice, mode='per')

                    LL1, (LH1, HL1, HH1) = decomp1
                    LL2, (LH2, HL2, HH2) = decomp2
                    LL3, (LH3, HL3, HH3) = decomp3
                    print(LH1)
                    print('Processing')
                    # LH1[0:100,:] = LH1[100:200,:]
                    # HL1[0:100,:] = HL1[100:200,:]
                    # HH1[0:100,:] = HH1[100:200,:]
                    print('ma',np.max(LH1))
                    # LH1 = np.clip(LH1, -1, 1)
                    # HL1 = np.clip(HL1, -1, 1)
                    # HH1 = np.clip(HH1, -1, 1)
                    # LH2 = np.clip(LH2, -1, 1)
                    # HL2 = np.clip(HL2, -1, 1)
                    # HH2 = np.clip(HH2, -1, 1)
                    # LH3 = np.clip(LH3, -1, 1)
                    # HL3 = np.clip(HL3, -1, 1)
                    # HH3 = np.clip(HH3, -1, 1)

                    # decomp3[1] = np.clip(decomp3[1], -100, 100)
                    print(LH1)
                    decomp1 = LL1, (LH1, HL1, HH1)
                    decomp2 = LL2, (LH2, HL2, HH2)
                    decomp3 = LL3, (LH3, HL3, HH3)
                    # decomp1 = abs(LL1), (abs(LH1), abs(HL1), abs(HH1))
                    # decomp2 = abs(LL2), (abs(LH2), abs(HL2), abs(HH2))
                    # decomp3 = abs(LL3), (abs(LH3), abs(HL3), abs(HH3))

                    # newdecomp10 = self.absmax(newdecomp1[0], decomp1[0])

                    # newdecomp10 = np.where(np.abs(newdecomp1[1][0]) + np.abs(newdecomp1[1][1]) + np.abs(newdecomp1[1][2]) > \
                    #      np.abs(decomp1[1][0]) + np.abs(decomp1[1][1]) + np.abs(decomp1[1][2]),
                    #      newdecomp1[0], decomp1[0])
                    # newdecomp110 = np.where(np.abs(newdecomp1[1][0]) + np.abs(newdecomp1[1][1]) + np.abs(newdecomp1[1][2]) > \
                    #     np.abs(decomp1[1][0]) + np.abs(decomp1[1][1]) + np.abs(decomp1[1][2]),
                    #     newdecomp1[1][0], decomp1[1][0])
                    # newdecomp111 = np.where(np.abs(newdecomp1[1][0]) + np.abs(newdecomp1[1][1]) + np.abs(newdecomp1[1][2]) > \
                    #     np.abs(decomp1[1][0]) + np.abs(decomp1[1][1]) + np.abs(decomp1[1][2]),
                    #     newdecomp1[1][1], decomp1[1][1])
                    # newdecomp112 = np.where(np.abs(newdecomp1[1][0]) + np.abs(newdecomp1[1][1]) + np.abs(newdecomp1[1][2]) > \
                    #     np.abs(decomp1[1][0]) + np.abs(decomp1[1][1]) + np.abs(decomp1[1][2]),
                    #     newdecomp1[1][2], decomp1[1][2])
                    # newdecomp1 = newdecomp10, (newdecomp110, newdecomp111, newdecomp112)
                    newdecomp1 = self.combine_decomp(newdecomp1, decomp1)

                    # newdecomp20 = self.absmax(newdecomp2[0], decomp2[0])

                    # newdecomp20 = np.where(np.abs(newdecomp2[1][0]) + np.abs(newdecomp2[1][1]) + np.abs(newdecomp2[1][2]) > \
                    #     np.abs(decomp2[1][0]) + np.abs(decomp2[1][1]) + np.abs(decomp2[1][2]),
                    #     newdecomp2[0], decomp2[0])
                    # newdecomp210 = np.where(np.abs(newdecomp2[1][0]) + np.abs(newdecomp2[1][1]) + np.abs(newdecomp2[1][2]) > \
                    #     np.abs(decomp2[1][0]) + np.abs(decomp2[1][1]) + np.abs(decomp2[1][2]),
                    #     newdecomp2[1][0], decomp2[1][0])
                    # newdecomp211 = np.where(np.abs(newdecomp2[1][0]) + np.abs(newdecomp2[1][1]) + np.abs(newdecomp2[1][2]) > \
                    #     np.abs(decomp2[1][0]) + np.abs(decomp2[1][1]) + np.abs(decomp2[1][2]),
                    #     newdecomp2[1][1], decomp2[1][1])
                    # newdecomp212 = np.where(np.abs(newdecomp2[1][0]) + np.abs(newdecomp2[1][1]) + np.abs(newdecomp2[1][2]) > \
                    #     np.abs(decomp2[1][0]) + np.abs(decomp2[1][1]) + np.abs(decomp2[1][2]),
                    #     newdecomp2[1][2], decomp2[1][2])
                    # newdecomp2 = newdecomp20, (newdecomp210, newdecomp211, newdecomp212)
                    newdecomp2 = self.combine_decomp(newdecomp2, decomp2)

                    # newdecomp30 = self.absmax(newdecomp3[0], decomp3[0])

                    # newdecomp30 = np.where(np.abs(newdecomp3[1][0]) + np.abs(newdecomp3[1][1]) + np.abs(newdecomp3[1][2]) > \
                    #     np.abs(decomp3[1][0]) + np.abs(decomp3[1][1]) + np.abs(decomp3[1][2]),
                    #     newdecomp3[0], decomp3[0])
                    # newdecomp310 = np.where(np.abs(newdecomp3[1][0]) + np.abs(newdecomp3[1][1]) + np.abs(newdecomp3[1][2]) > \
                    #     np.abs(decomp3[1][0]) + np.abs(decomp3[1][1]) + np.abs(decomp3[1][2]),
                    #     newdecomp3[1][0], decomp3[1][0])
                    # newdecomp311 = np.where(np.abs(newdecomp3[1][0]) + np.abs(newdecomp3[1][1]) + np.abs(newdecomp3[1][2]) > \
                    #     np.abs(decomp3[1][0]) + np.abs(decomp3[1][1]) + np.abs(decomp3[1][2]),
                    #     newdecomp3[1][1], decomp3[1][1])
                    # newdecomp312 = np.where(np.abs(newdecomp3[1][0]) + np.abs(newdecomp3[1][1]) + np.abs(newdecomp3[1][2]) > \
                    #     np.abs(decomp3[1][0]) + np.abs(decomp3[1][1]) + np.abs(decomp3[1][2]),
                    #     newdecomp3[1][2], decomp3[1][2])
                    # newdecomp3 = newdecomp30, (newdecomp310, newdecomp311, newdecomp312)
                    newdecomp3 = self.combine_decomp(newdecomp3, decomp3)
                    
                    # print(recompimg.shape)
                    recompimg = np.zeros_like(currimg)
                    recompimg[:,:,2] = pywt.idwt2(newdecomp1, waveletchoice, mode='per')[0:recompimg.shape[0],:]
                    recompimg[:,:,1] = pywt.idwt2(newdecomp2, waveletchoice, mode='per')[0:recompimg.shape[0],:]
                    recompimg[:,:,0] = pywt.idwt2(newdecomp3, waveletchoice, mode='per')[0:recompimg.shape[0],:]

                    im1 = Image.fromarray((recompimg))
                    recompname = 'OutputFolder/dwt_' + waveletchoice + '_recomp_' + str(j) + '.jpg'
                    print('Saving recomposition')
                    im1.save(recompname)
                    print('Recomposition saved in ' + recompname)
                    
                    combinedImg = np.zeros((decomp1[0].shape[0],decomp1[0].shape[1],3))
                    k = 0
                    print('Looping and saving decomposition figure for each channel...')
                    # Loop over RGB channels of current image
                    for decomp in [newdecomp1, newdecomp2, newdecomp3]:
                        LL, (LH, HL, HH) = decomp
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
                        print('npma',np.max(LL))
                        LL = LL / np.max(LL)
                        combinedImg[:,:,2-k] = LL
                        k += 1
                        figname = 'OutputFolder/chan' + str(k) + '_dwt' + str(j) + '_' + waveletchoice + '.jpg'
                        plt.savefig(figname)
                        plt.close()
                        print('Decomposition figure saved in ' + figname)

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return recompimg
    
    def absmax(self, a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)

    def combine_decomp(self, newdecompx, currdecomp):
        boolMat = np.abs(newdecompx[1][0]) + np.abs(newdecompx[1][1]) + np.abs(newdecompx[1][2]) > \
                         np.abs(currdecomp[1][0]) + np.abs(currdecomp[1][1]) + np.abs(currdecomp[1][2])
        # print(boolMat)
        newdecompx0 = np.where(boolMat,
                newdecompx[0], currdecomp[0])
        newdecompx10 = np.where(boolMat,
            newdecompx[1][0], currdecomp[1][0])
        newdecompx11 = np.where(boolMat,
            newdecompx[1][1], currdecomp[1][1])
        newdecompx12 = np.where(boolMat,
            newdecompx[1][2], currdecomp[1][2])
        newdecompx = newdecompx0, (newdecompx10, newdecompx11, newdecompx12)
        return newdecompx
