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

class Alg3WaveletGray(object):
    def startAlg(self, image_files, alignMethod):
        print("Algorithm3 (wavelet decomposition gray) starting.")
        print('image files', image_files)
        image_files = list(set(image_files)) # remove duplicates
        image_files = sorted(image_files)
        print('sorted image files', image_files)
        img_mats = [cv2.imread(img) for img in image_files]
        # print("SKIPPING alignment module.")
        # img_mats = alignment.align_images_compare_last(img_mats)
        img_mats = alignMethod(img_mats)
        print("typeimgmats",type(img_mats))
        num_files = len(image_files)
        print('numfile', num_files)
        family = 'cmor'
        waveletchoice = 'haar'
        # for family in ['haar']: #pywt.families():
        print(pywt.families())
        waveletchoice = 'haar'
        print('waveletchoice', waveletchoice)
        firstimg = img_mats[0]
        decomp1 = pywt.dwt2(firstimg[:,:,0], waveletchoice, mode='per')
        decomp2 = pywt.dwt2(firstimg[:,:,1], waveletchoice, mode='per')
        decomp3 = pywt.dwt2(firstimg[:,:,2], waveletchoice, mode='per')
        newdecompg = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
        newdecomp1 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
        newdecomp2 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
        newdecomp3 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
        for j in range(num_files):
            currimg = img_mats[j]
            print("firstimg shape", firstimg.shape)
            print("Running wavelet decomposition.")

            imggray = cv2.cvtColor(currimg, cv2.COLOR_BGR2GRAY)
            
            decompgray = pywt.dwt2(imggray, waveletchoice, mode='per')

            decomp1 = pywt.dwt2(currimg[:,:,0], waveletchoice, mode='per')
            decomp2 = pywt.dwt2(currimg[:,:,1], waveletchoice, mode='per')
            decomp3 = pywt.dwt2(currimg[:,:,2], waveletchoice, mode='per')

            LLg, (LHg, HLg, HHg) = decompgray

            LL1, (LH1, HL1, HH1) = decomp1
            LL2, (LH2, HL2, HH2) = decomp2
            LL3, (LH3, HL3, HH3) = decomp3
            # print(LH1)
            print('Processing')

            decompgray = LLg, (LHg, HLg, HHg)

            decomp1 = LL1, (LH1, HL1, HH1)
            decomp2 = LL2, (LH2, HL2, HH2)
            decomp3 = LL3, (LH3, HL3, HH3)

            newdecomp1, newdecompg = self.combine_decomps_gray(newdecomp1, newdecompg, decomp1, decompgray)
            newdecomp2, newdecompg = self.combine_decomps_gray(newdecomp2, newdecompg, decomp2, decompgray)
            newdecomp3, newdecompg = self.combine_decomps_gray(newdecomp3, newdecompg, decomp3, decompgray)
            
            recompimg = np.zeros_like(currimg)
            recompimggray = np.zeros_like(imggray)

            recompimggray[:,:] = pywt.idwt2(newdecompg, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,0] = pywt.idwt2(newdecomp1, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,1] = pywt.idwt2(newdecomp2, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,2] = pywt.idwt2(newdecomp3, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]

            recompname = 'OutputFolder/dwt_' + waveletchoice + '_recomp_' + str(j) + '.jpg'
            print('Saving recomposition')
            cv2.imwrite(recompname, recompimg)
            print('Recomposition saved in ' + recompname)

            recompgrayname = 'OutputFolder/dwt_' + waveletchoice + '_recomp_gray' + str(j) + '.jpg'
            print('Saving recomposition')
            cv2.imwrite(recompgrayname, recompimggray)
            print('Recomposition gray saved in ' + recompgrayname)
            
            combinedImg = np.zeros((decomp1[0].shape[0],decomp1[0].shape[1],3))
            k = 0
            print('Looping and saving decomposition figure for each channel...')
            # Loop over RGB channels of current image
            for decomp in [newdecomp1, newdecomp2, newdecomp3]:
                # Creating & saving decomposition images
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
                LL = LL / np.max(LL)
                combinedImg[:,:,2-k] = LL
                k += 1
                figname = 'OutputFolder/chan' + str(k) + '_dwt' + str(j) + '_' + waveletchoice + '.jpg'
                plt.savefig(figname)
                plt.close() #Important to close plot often, otherwise memory leak and program crashes after 50 iterations!
                print('Decomposition figure saved in ' + figname)
        print('Saving recomposition')
        # im1.save(recompname)
        cv2.imwrite(recompname, recompimg)

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return recompimg
    
    def absmax(self, a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)

    def combine_decomps(self, newdecompx, currdecomp):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        boolMat = np.abs(newdecompx[1][0]) + np.abs(newdecompx[1][1]) + np.abs(newdecompx[1][2]) > \
                         np.abs(currdecomp[1][0]) + np.abs(currdecomp[1][1]) + np.abs(currdecomp[1][2])
        # copy pixel values for all four subbands according to boolean matrix.
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

    def combine_decomps_gray(self, newdecompx, newdecompg, currdecomp, currdecompg):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        boolMat = np.abs(newdecompg[1][0]) + np.abs(newdecompg[1][1]) + np.abs(newdecompg[1][2]) > \
                         np.abs(currdecompg[1][0]) + np.abs(currdecompg[1][1]) + np.abs(currdecompg[1][2])
        # copy pixel values for all four subbands according to boolean matrix.
        newdecompx0 = np.where(boolMat,
                newdecompx[0], currdecomp[0])
        newdecompx10 = np.where(boolMat,
            newdecompx[1][0], currdecomp[1][0])
        newdecompx11 = np.where(boolMat,
            newdecompx[1][1], currdecomp[1][1])
        newdecompx12 = np.where(boolMat,
            newdecompx[1][2], currdecomp[1][2])
        newdecompx = newdecompx0, (newdecompx10, newdecompx11, newdecompx12)

        newdecompg0 = np.where(boolMat,
                newdecompg[0], currdecompg[0])
        newdecompg10 = np.where(boolMat,
            newdecompg[1][0], currdecompg[1][0])
        newdecompg11 = np.where(boolMat,
            newdecompg[1][1], currdecompg[1][1])
        newdecompg12 = np.where(boolMat,
            newdecompg[1][2], currdecompg[1][2])
        newdecompg = newdecompg0, (newdecompg10, newdecompg11, newdecompg12)

        return newdecompx, newdecompg
