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

class Alg10Waveletr2dDecompL2(object):
    def startAlg(self, image_files, alignMethod):
        print("Algorithm10 (wavelet using pywt.wavedec2 method - 2 levels) starting.")
        print('image files', image_files)
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
        for family in ['haar']: #pywt.families():
            print('family', family)
            if family == 'gaus' or family == 'mexh' or family == 'morl' or family == 'cgau' or family == 'shan' or family == 'fbsp' or family == 'cmor':
                print('continuing')
                continue
            # for waveletchoice in pywt.wavelist(family):
            print('wavelist', pywt.wavelist(family))
            for waveletchoice1 in pywt.wavelist(family):
                waveletchoice = 'haar'
                print('waveletchoice', waveletchoice)
                firstimg = img_mats[0]
                decomp1 = pywt.dwt2(firstimg[:,:,0], waveletchoice, mode='per')
                decomp2 = pywt.dwt2(firstimg[:,:,1], waveletchoice, mode='per')
                decomp3 = pywt.dwt2(firstimg[:,:,2], waveletchoice, mode='per')
                newdecompg = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                wdecompgimg = cv2.cvtColor(img_mats[0], cv2.COLOR_BGR2GRAY)
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

                    # newdecompg = self.combine_decomps(newdecompg, decompgray)

                    newdecomp1, newdecompg = self.combine_decomps_gray(newdecomp1, newdecompg, decomp1, decompgray)
                    newdecomp2, newdecompg = self.combine_decomps_gray(newdecomp2, newdecompg, decomp2, decompgray)
                    newdecomp3, newdecompg = self.combine_decomps_gray(newdecomp3, newdecompg, decomp3, decompgray)
                    
                    #ALTERNATE DECOMP
                    print("ALT DECOMP")
                    print("imggrayshape",imggray.shape)
                    print("wdecompgimgshape",wdecompgimg.shape)
                    w_level = 2
                    coeffs = pywt.wavedec2(imggray, waveletchoice, level=w_level)
                    print("lencoeffs",len(coeffs))
                    # wdecompgimg = cv2.cvtColor(wdecompgimg, cv2.COLOR_BGR2GRAY)
                    focal_coeffs = pywt.wavedec2(wdecompgimg, waveletchoice, level=w_level)
                    print("lenfocal_coeffs",len(focal_coeffs))
                    # fused_coeffs = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # fused_coeffs0 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[0], coeffs[0])]
                    # fused_coeffs1 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[1][0], coeffs[1][0])]
                    # fused_coeffs2 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[1][1], coeffs[1][1])]
                    # fused_coeffs3 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[1][2], coeffs[1][2])]
                    # # fused_coeffs = [(np.where(np.abs(focal_c[0]) > np.abs(img_c[0]), focal_c, img_c), (np.where(np.abs(focal_c[1][0]) > np.abs(img_c[1][0]), focal_c, img_c), np.where(np.abs(focal_c[1][1]) > np.abs(img_c[1][1]), focal_c, img_c), np.where(np.abs(focal_c[1][2]) > np.abs(img_c[1][2]), focal_c, img_c))) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # fused_coeffs = fused_coeffs0, (fused_coeffs1, fused_coeffs2, fused_coeffs3)
                    # fused_coeffs = fused_coeffs[0], (fused_coeffs[1][0], fused_coeffs[1][1], fused_coeffs[1][2])
                    # fused_coeffs = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # bool_coeffs = [focal_c > img_c for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    fused_coeffs4comp = [np.maximum(np.abs(focal_c), np.abs(img_c)) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    bool_coeffs0 = fused_coeffs4comp[0] == np.abs(focal_coeffs[0])
                    bool_coeffs10 = fused_coeffs4comp[1][0] == np.abs(focal_coeffs[1][0])
                    bool_coeffs11 = fused_coeffs4comp[1][1] == np.abs(focal_coeffs[1][1])
                    bool_coeffs12 = fused_coeffs4comp[1][2] == np.abs(focal_coeffs[1][2])
                    bool_coeffs20 = fused_coeffs4comp[2][0] == np.abs(focal_coeffs[2][0])
                    bool_coeffs21 = fused_coeffs4comp[2][1] == np.abs(focal_coeffs[2][1])
                    bool_coeffs22 = fused_coeffs4comp[2][2] == np.abs(focal_coeffs[2][2])
                    # print("bool_coeffs10", bool_coeffs10)
                    # fused_coeffs = fused_coeffs0, (fused_coeffs1, fused_coeffs2, fused_coeffs3)
                    # fused_coeffs = fused_coeffs[0], (fused_coeffs[1][0], fused_coeffs[1][1], fused_coeffs[1][2])
                    fused_coeffs = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]), (np.where(bool_coeffs10, focal_coeffs[1][0], coeffs[1][0]), np.where(bool_coeffs11, focal_coeffs[1][1], coeffs[1][1]), np.where(bool_coeffs12, focal_coeffs[1][2], coeffs[1][2])), \
                    (np.where(bool_coeffs20, focal_coeffs[2][0], coeffs[2][0]), np.where(bool_coeffs21, focal_coeffs[2][1], coeffs[2][1]), np.where(bool_coeffs22, focal_coeffs[2][2], coeffs[2][2]))

                    fused_coeffs = self.channel_decomp_multilevel(imggray, wdecompgimg, waveletchoice)
                    # print("coeffs", coeffs)
                    # print("fusedcoeffs", fused_coeffs)
                    # print("coeffslen", len(coeffs))
                    # print("focallen",len(focal_coeffs))
                    # print("fusedlen", len(fused_coeffs))

                    print("lenfused_coeffs",len(fused_coeffs))
                    wdecompgimg = pywt.waverec2(fused_coeffs, waveletchoice)
                    versionnum = 31
                    altname = 'OutputFolder/newdwt_v' + str(versionnum) + '_l2_multi' + waveletchoice + '_recomp_' + str(j) + '.jpg'
                    print("Saving alternate recomposition...")
                    cv2.imwrite(altname, wdecompgimg)
                    print('Alt Recomposition saved in ' + altname)
                    #END ALTERNATE DECOMP

                    recompimg = np.zeros_like(currimg)
                    recompimggray = np.zeros_like(imggray)

                    recompimggray[:,:] = pywt.idwt2(newdecompg, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    recompimg[:,:,0] = pywt.idwt2(newdecomp1, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    recompimg[:,:,1] = pywt.idwt2(newdecomp2, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    recompimg[:,:,2] = pywt.idwt2(newdecomp3, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]

                    # im1 = Image.fromarray((recompimg))
                    recompname = 'OutputFolder/dwt_' + waveletchoice + '_recomp_' + str(j) + '.jpg'
                    print('Saving recomposition')
                    # im1.save(recompname)
                    cv2.imwrite(recompname, recompimg)
                    print('Recomposition saved in ' + recompname)

                    recompgrayname = 'OutputFolder/dwt_' + waveletchoice + '_recomp_gray' + str(j) + '.jpg'
                    print('Saving recomposition')
                    # im1.save(recompname)
                    cv2.imwrite(recompgrayname, recompimggray)
                    print('Recomposition gray saved in ' + recompgrayname)
                    
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
                        # print('npmax',np.max(LL))
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
    
    def channel_decomp_multilevel(self, currimggray, progressimggray, waveletchoice):
        wavelevel = 6
        fusedlevelimg = currimggray.copy()
        for j in range(wavelevel,0,-1):
        # for j in range(1, wavelevel+1):
            fusedleveldecomp = self.channel_decomp_wavedec(fusedlevelimg, progressimggray, waveletchoice, j);
            fusedlevelimg = pywt.waverec2(fusedleveldecomp, waveletchoice)

        # fusedleveldecomp = self.channel_decomp_wavedec(fusedlevelimg, progressimggray, waveletchoice, wavelevel);
        fusedleveldecomp = pywt.wavedec2(fusedlevelimg, waveletchoice, level=wavelevel)
        return fusedleveldecomp
        

    def channel_decomp_wavedec(self, currimggray, progressimggray, waveletchoice, wavelevel):
        # wavelevel = 2
        coeffs = pywt.wavedec2(currimggray, waveletchoice, level=wavelevel)
        focal_coeffs = pywt.wavedec2(progressimggray, waveletchoice, level=wavelevel)
        fused_coeffs = focal_coeffs.copy()
        num_high_tuples = len(coeffs)
        highpass_sum = 0
        focal_highpass_sum = 0
        fused_coeffs4comp = [np.maximum(np.abs(focal_c), np.abs(img_c)) for focal_c, img_c in zip(focal_coeffs, coeffs)]
        boolcoeffs = []
        fusedcoeffs = [focal_coeffs[0]]
        focalhighsum = 0
        fusedhighsum = 0
        for i in range(1, num_high_tuples):
            # print("I",i)
            # bool_coeffs0 = fused_coeffs4comp[0] == np.abs(focal_coeffs[0])
            bool_coeffs10 = fused_coeffs4comp[i][0] == np.abs(focal_coeffs[i][0])
            bool_coeffs11 = fused_coeffs4comp[i][1] == np.abs(focal_coeffs[i][1])
            bool_coeffs12 = fused_coeffs4comp[i][2] == np.abs(focal_coeffs[i][2])
            if i == 1:
                focalhighsum += np.abs(focal_coeffs[i][0]) + np.abs(focal_coeffs[i][1]) + np.abs(focal_coeffs[i][2])
                fusedhighsum += fused_coeffs4comp[i][0] + fused_coeffs4comp[i][1] + fused_coeffs4comp[i][2]
            # boolcoeffs.append(bool_coeffs0, ((bool_coeffs10, bool_coeffs11, bool_coeffs12)))
            fusedcoeffs.append((np.where(bool_coeffs10, focal_coeffs[i][0], coeffs[i][0]), np.where(bool_coeffs11, focal_coeffs[i][1], coeffs[i][1]), np.where(bool_coeffs12, focal_coeffs[i][2], coeffs[i][2])))

            # fused_coeffs[i] = self.channel_decomp_high_sum(coeffs[i], focal_coeffs[i])
            # focal_highpass_sum = np.abs(focal_coeffs[i][0]) + np.abs(focal_coeffs[i][1]) + np.abs(focal_coeffs[i][2])
            # highpass_sum = np.abs(coeffs[i][0]) + np.abs(coeffs[i][1]) + np.abs(coeffs[i][2])

        bool_coeffs0 = focalhighsum >= fusedhighsum
        print('bool_coeffs0.shape', bool_coeffs0.shape)
        print('focal_coeffs[0].shape', focal_coeffs[0].shape)
        # if wavelevel == 1:
        #     print("WAVELEVEL2")
        #     fusedcoeffs[0] = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]) # lowpass vote

        fusedcoeffs[0] = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]) # lowpass vote
        # # replace low pass choice with majority vote.
        # bool_coeffs0 = sum([bool_coeffs10, bool_coeffs11, bool_coeffs12]) >= 2
        # bool_coeffs0 = np.abs(focal_coeffs[1][0]) + np.abs(focal_coeffs[1][1]) + np.abs(focal_coeffs[1][2]) >= fused_coeffs4comp[1][0] + fused_coeffs4comp[1][1] + fused_coeffs4comp[1][2]
        # bool_coeffs0 = focal_highpass_sum >= highpass_sum #PREV COMMIT
        # print('bool_coeffs0.shape:', bool_coeffs0.shape)
        # print('bool_coeffs10.shape:', bool_coeffs10.shape)
        # print('XXXXXXbool0.shape:', bool_coeffs0.shape)
        # print('bool0:', bool_coeffs0)
        # fused_coeffs = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]), (np.where(bool_coeffs10, focal_coeffs[1][0], coeffs[1][0]), np.where(bool_coeffs11, focal_coeffs[1][1], coeffs[1][1]), np.where(bool_coeffs12, focal_coeffs[1][2], coeffs[1][2]))
        return fusedcoeffs

    def channel_decomp_high_sum(self, coeffs, focal_coeffs):
        fused_coeffs4comp = [np.maximum(np.abs(focal_c), np.abs(img_c)) for focal_c, img_c in zip(focal_coeffs, coeffs)]
        bool_coeffs10 = fused_coeffs4comp[0] == np.abs(focal_coeffs[0])
        bool_coeffs11 = fused_coeffs4comp[1] == np.abs(focal_coeffs[1])
        bool_coeffs12 = fused_coeffs4comp[2] == np.abs(focal_coeffs[2])
        # replace low pass choice with majority vote.
        # highpass_sum += sum([bool_coeffs10, bool_coeffs11, bool_coeffs12])
        # bool_coeffs0 = np.abs(focal_coeffs[0]) + np.abs(focal_coeffs[1]) + np.abs(focal_coeffs[2]) >= fused_coeffs4comp[0] + fused_coeffs4comp[1] + fused_coeffs4comp[2]
        # print('bool_coeffs0.shape:', bool_coeffs0.shape)
        # print('bool_coeffs10.shape:', bool_coeffs10.shape)
        # print('XXXXXXbool0.shape:', bool_coeffs0.shape)
        # print('bool0:', bool_coeffs0)
        # fused_coeffs = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]), (np.where(bool_coeffs10, focal_coeffs[1][0], coeffs[1][0]), np.where(bool_coeffs11, focal_coeffs[1][1], coeffs[1][1]), np.where(bool_coeffs12, focal_coeffs[1][2], coeffs[1][2]))
        fused_coeffs = (np.where(bool_coeffs10, focal_coeffs[0], coeffs[0]), np.where(bool_coeffs11, focal_coeffs[1], coeffs[1]), np.where(bool_coeffs12, focal_coeffs[2], coeffs[2]))
        return fused_coeffs

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
