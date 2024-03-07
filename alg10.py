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
        # waveletchoice = 'haar'
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
                w_mainlevel = 1
                waveletchoice = 'db5'
                print('waveletchoice', waveletchoice)
                firstimg = img_mats[0]
                # decomp1 = pywt.dwt2(firstimg[:,:,0], waveletchoice, mode='per')
                # decomp2 = pywt.dwt2(firstimg[:,:,1], waveletchoice, mode='per')
                # decomp3 = pywt.dwt2(firstimg[:,:,2], waveletchoice, mode='per')
                decomp1 = pywt.wavedec2(firstimg[:,:,0], waveletchoice, level=w_mainlevel)
                decomp2 = pywt.wavedec2(firstimg[:,:,1], waveletchoice, level=w_mainlevel)
                decomp3 = pywt.wavedec2(firstimg[:,:,2], waveletchoice, level=w_mainlevel)
                # newdecompg = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                wdecompgimg = cv2.cvtColor(img_mats[0], cv2.COLOR_BGR2GRAY)
                newdecompg = pywt.wavedec2(wdecompgimg, waveletchoice, level=w_mainlevel)
                # newdecomp1 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                # newdecomp2 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                # newdecomp3 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                newdecomp1 = decomp1
                newdecomp2 = decomp2
                newdecomp3 = decomp3
                newdecompimg = firstimg.copy()
                for j in range(num_files):
                    currimg = img_mats[j]
                    print("firstimg shape", firstimg.shape)
                    print("Running wavelet decomposition.")

                    imggray = cv2.cvtColor(currimg, cv2.COLOR_BGR2GRAY)
                    
                    # decompgray = pywt.dwt2(imggray, waveletchoice, mode='per')

                    # decomp1 = pywt.dwt2(currimg[:,:,0], waveletchoice, mode='per')
                    # decomp2 = pywt.dwt2(currimg[:,:,1], waveletchoice, mode='per')
                    # decomp3 = pywt.dwt2(currimg[:,:,2], waveletchoice, mode='per')

                    # decomp1 = pywt.wavedec2(currimg[:,:,0], waveletchoice, w_mainlevel)
                    # decomp2 = pywt.wavedec2(currimg[:,:,1], waveletchoice, w_mainlevel)
                    # decomp3 = pywt.wavedec2(currimg[:,:,2], waveletchoice, w_mainlevel)

                    # curr_rec1 = pywt.waverec2(decomp1, waveletchoice, mode='symmetric')
                    # curr_rec2 = pywt.waverec2(decomp2, waveletchoice, mode='symmetric')
                    # curr_rec3 = pywt.waverec2(decomp3, waveletchoice, mode='symmetric')
                    
                    # currimg = np.zeros((curr_rec1.shape[0], curr_rec1.shape[1], 3))
                    # currimg[:,:,0] = curr_rec1
                    # currimg[:,:,1] = curr_rec2
                    # currimg[:,:,2] = curr_rec3

                    # LLg, (LHg, HLg, HHg) = decompgray

                    # LL1, (LH1, HL1, HH1) = decomp1
                    # LL2, (LH2, HL2, HH2) = decomp2
                    # LL3, (LH3, HL3, HH3) = decomp3
                    # # print(LH1)
                    print('Processing')

                    # decompgray = LLg, (LHg, HLg, HHg)

                    # decomp1 = LL1, (LH1, HL1, HH1)
                    # decomp2 = LL2, (LH2, HL2, HH2)
                    # decomp3 = LL3, (LH3, HL3, HH3)

                    # newdecompg = self.combine_decomps(newdecompg, decompgray)

                    # #PREVIOUS 1-LEVEL DECOMP
                    # newdecomp1, newdecompg = self.combine_decomps_gray(newdecomp1, newdecompg, decomp1, decompgray)
                    # newdecomp2, newdecompg = self.combine_decomps_gray(newdecomp2, newdecompg, decomp2, decompgray)
                    # newdecomp3, newdecompg = self.combine_decomps_gray(newdecomp3, newdecompg, decomp3, decompgray)
                    # #END PREVIOUS 1-LEVEL DECOMP

                    #NEW MULTILEVEL DECOMP
                    # print("Recompositing progress decomposition of current iteration...")
                    # newdecompimg1 = pywt.waverec2(newdecomp1, waveletchoice, mode='symmetric')
                    # newdecompimg2 = pywt.waverec2(newdecomp2, waveletchoice, mode='symmetric')
                    # newdecompimg3 = pywt.waverec2(newdecomp3, waveletchoice, mode='symmetric')
                    # newdecompimg = np.zeros((newdecompimg1.shape[0], newdecompimg1.shape[1], 3))
                    # newdecompimg[:,:,0] = newdecompimg1
                    # newdecompimg[:,:,1] = newdecompimg2
                    # newdecompimg[:,:,2] = newdecompimg3
                    # print('currimg.shape', currimg.shape)
                    # print('newdecompimg1.shape', newdecompimg1.shape)
                    print("Performing pointwise maximum comparison")
                    # newdecompg = self.channel_decomp_multilevel(imggray, wdecompgimg, waveletchoice, w_mainlevel)
                    # newdecomp1 = self.channel_decomp_multilevel(newdecompimg1, currimg[:,:,0], waveletchoice, w_mainlevel)
                    # newdecomp2 = self.channel_decomp_multilevel(newdecompimg2, currimg[:,:,1], waveletchoice, w_mainlevel)
                    # newdecomp3 = self.channel_decomp_multilevel(newdecompimg3, currimg[:,:,2], waveletchoice, w_mainlevel)

                    newdecomp1, newdecomp2, newdecomp3, newdecompg = self.channel_decomp_multilevel_3chan(currimg.copy(), newdecompimg.copy(), imggray.copy(), wdecompgimg.copy(), waveletchoice, w_mainlevel)
                    #END NEW MULTILEVEL DECOMP
                    
                    # #ALTERNATE DECOMP
                    # print("ALT DECOMP")
                    # print("imggrayshape",imggray.shape)
                    # print("wdecompgimgshape",wdecompgimg.shape)
                    # w_level = 2
                    # coeffs = pywt.wavedec2(imggray, waveletchoice, level=w_level)
                    # print("lencoeffs",len(coeffs))
                    # # wdecompgimg = cv2.cvtColor(wdecompgimg, cv2.COLOR_BGR2GRAY)
                    # focal_coeffs = pywt.wavedec2(wdecompgimg, waveletchoice, level=w_level)
                    # print("lenfocal_coeffs",len(focal_coeffs))
                    # # fused_coeffs = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # # fused_coeffs0 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[0], coeffs[0])]
                    # # fused_coeffs1 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[1][0], coeffs[1][0])]
                    # # fused_coeffs2 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[1][1], coeffs[1][1])]
                    # # fused_coeffs3 = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs[1][2], coeffs[1][2])]
                    # # # fused_coeffs = [(np.where(np.abs(focal_c[0]) > np.abs(img_c[0]), focal_c, img_c), (np.where(np.abs(focal_c[1][0]) > np.abs(img_c[1][0]), focal_c, img_c), np.where(np.abs(focal_c[1][1]) > np.abs(img_c[1][1]), focal_c, img_c), np.where(np.abs(focal_c[1][2]) > np.abs(img_c[1][2]), focal_c, img_c))) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # # fused_coeffs = fused_coeffs0, (fused_coeffs1, fused_coeffs2, fused_coeffs3)
                    # # fused_coeffs = fused_coeffs[0], (fused_coeffs[1][0], fused_coeffs[1][1], fused_coeffs[1][2])
                    # # fused_coeffs = [np.maximum(focal_c, img_c) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # # bool_coeffs = [focal_c > img_c for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # fused_coeffs4comp = [np.maximum(np.abs(focal_c), np.abs(img_c)) for focal_c, img_c in zip(focal_coeffs, coeffs)]
                    # bool_coeffs0 = fused_coeffs4comp[0] == np.abs(focal_coeffs[0])
                    # bool_coeffs10 = fused_coeffs4comp[1][0] == np.abs(focal_coeffs[1][0])
                    # bool_coeffs11 = fused_coeffs4comp[1][1] == np.abs(focal_coeffs[1][1])
                    # bool_coeffs12 = fused_coeffs4comp[1][2] == np.abs(focal_coeffs[1][2])
                    # bool_coeffs20 = fused_coeffs4comp[2][0] == np.abs(focal_coeffs[2][0])
                    # bool_coeffs21 = fused_coeffs4comp[2][1] == np.abs(focal_coeffs[2][1])
                    # bool_coeffs22 = fused_coeffs4comp[2][2] == np.abs(focal_coeffs[2][2])
                    # # print("bool_coeffs10", bool_coeffs10)
                    # # fused_coeffs = fused_coeffs0, (fused_coeffs1, fused_coeffs2, fused_coeffs3)
                    # # fused_coeffs = fused_coeffs[0], (fused_coeffs[1][0], fused_coeffs[1][1], fused_coeffs[1][2])
                    # fused_coeffs = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]), (np.where(bool_coeffs10, focal_coeffs[1][0], coeffs[1][0]), np.where(bool_coeffs11, focal_coeffs[1][1], coeffs[1][1]), np.where(bool_coeffs12, focal_coeffs[1][2], coeffs[1][2])), \
                    # (np.where(bool_coeffs20, focal_coeffs[2][0], coeffs[2][0]), np.where(bool_coeffs21, focal_coeffs[2][1], coeffs[2][1]), np.where(bool_coeffs22, focal_coeffs[2][2], coeffs[2][2]))

                    # fused_coeffs = self.channel_decomp_multilevel(imggray, wdecompgimg, waveletchoice)
                    # # print("coeffs", coeffs)
                    # # print("fusedcoeffs", fused_coeffs)
                    # # print("coeffslen", len(coeffs))
                    # # print("focallen",len(focal_coeffs))
                    # # print("fusedlen", len(fused_coeffs))

                    # print("lenfused_coeffs",len(fused_coeffs))
                    # wdecompgimg = pywt.waverec2(fused_coeffs, waveletchoice)
                    # versionnum = 32
                    # altname = 'OutputFolder/newdwt_v' + str(versionnum) + '_l2_multi' + waveletchoice + '_recomp_' + str(j) + '.jpg'
                    # print("Saving alternate recomposition...")
                    # cv2.imwrite(altname, wdecompgimg)
                    # print('Alt Recomposition saved in ' + altname)
                    # #END ALTERNATE DECOMP

                    # recompimg = np.zeros_like(currimg)
                    # recompimggray = np.zeros_like(imggray)

                    # #OLD RECOMP
                    # recompimggray[:,:] = pywt.idwt2(newdecompg, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    # recompimg[:,:,0] = pywt.idwt2(newdecomp1, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    # recompimg[:,:,1] = pywt.idwt2(newdecomp2, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    # recompimg[:,:,2] = pywt.idwt2(newdecomp3, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
                    # #END OLD RECOMP

                    # NEW RECOMP
                    # Why is gray shape different from color channel shapes?
                    print("recg",newdecompg[0].shape)
                    print("rec0", newdecomp1[0].shape)
                    print("Recompositing current image from deocmpositions.")
                    graychan = pywt.waverec2(newdecompg, waveletchoice)#[0:recompimggray.shape[0],0:recompimggray.shape[1]]
                    recchan = pywt.waverec2(newdecomp1, waveletchoice)#[0:recompimg.shape[0],0:recompimg.shape[1]]
                    recompimggray = np.zeros((graychan.shape[0], graychan.shape[1]))
                    recompimg = np.zeros((recchan.shape[0], recchan.shape[1], 3))
                    recompimggray[:,:] = pywt.waverec2(newdecompg, waveletchoice)[0:recompimggray.shape[0],0:recompimggray.shape[1]]
                    recompimg[:,:,0] = pywt.waverec2(newdecomp1, waveletchoice)[0:recompimg.shape[0],0:recompimg.shape[1]]
                    recompimg[:,:,1] = pywt.waverec2(newdecomp2, waveletchoice)[0:recompimg.shape[0],0:recompimg.shape[1]]
                    recompimg[:,:,2] = pywt.waverec2(newdecomp3, waveletchoice)[0:recompimg.shape[0],0:recompimg.shape[1]]
                    print("Saving results for next iteration...")
                    wdecompgimg = recompimggray
                    newdecompimg = recompimg
                    # END NEW RECOMP

                    # im1 = Image.fromarray((recompimg))
                    recompname = 'OutputFolder/dwtrec2_' + waveletchoice + '_recomp_' + str(j) + '.jpg'
                    print('Saving recomposition')
                    # im1.save(recompname)
                    cv2.imwrite(recompname, recompimg)
                    print('Recomposition saved in ' + recompname)

                    recompgrayname = 'OutputFolder/dwtrec2_' + waveletchoice + '_recomp_gray' + str(j) + '.jpg'
                    print('Saving recomposition')
                    # im1.save(recompname)
                    cv2.imwrite(recompgrayname, recompimggray)
                    print('Recomposition gray saved in ' + recompgrayname)
                    
                    # # DECOMPOSITION FIGURES FOR ONLY 1 LEVEL
                    # combinedImg = np.zeros((decomp1[0].shape[0],decomp1[0].shape[1],3))
                    # k = 0
                    # print('Looping and saving decomposition figure for each channel...')
                    # # Loop over RGB channels of current image
                    # for decomp in [newdecomp1, newdecomp2, newdecomp3]:
                    #     LL, (LH, HL, HH) = decomp
                    #     titles = ['Approximation', ' Horizontal detail',
                    #     'Vertical detail', 'Diagonal detail']
                    #     fig = plt.figure(figsize=(13, 10.8))
                    #     for i, a in enumerate([LL, LH, HL, HH]):
                    #         ax = fig.add_subplot(2, 2, i + 1)
                    #         ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
                    #         ax.set_title(titles[i], fontsize=10)
                    #         ax.set_xticks([])
                    #         ax.set_yticks([])
                    #     fig.tight_layout()
                    #     # print('npmax',np.max(LL))
                    #     LL = LL / np.max(LL)
                    #     combinedImg[:,:,2-k] = LL
                    #     k += 1
                    #     figname = 'OutputFolder/chan' + str(k) + '_dwt' + str(j) + '_' + waveletchoice + '.jpg'
                    #     plt.savefig(figname)
                    #     plt.close() #Important to close plot often, otherwise memory leak and program crashes after 50 iterations!
                    #     print('Decomposition figure saved in ' + figname)
                    # # END DECOMPOSITION FIGURES FOR ONLY 1 LEVEL

                print('Saving recomposition')
                # im1.save(recompname)
                cv2.imwrite(recompname, recompimg)

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return recompimg
    
    def absmax(self, a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)
    
    # def channel_decomp_multilevel(self, currimggray, progressimggray, waveletchoice, wavelevel):
    #     fusedlevelimg = currimggray.copy()
    #     for j in range(wavelevel,0,-1):
    #     # for j in range(1, wavelevel+1):
    #         fusedleveldecomp = self.channel_decomp_wavedec(fusedlevelimg, progressimggray, waveletchoice, j);
    #         fusedlevelimg = pywt.waverec2(fusedleveldecomp, waveletchoice)

    #     # fusedleveldecomp = self.channel_decomp_wavedec(fusedlevelimg, progressimggray, waveletchoice, wavelevel);
    #     fusedleveldecomp = pywt.wavedec2(fusedlevelimg, waveletchoice, level=wavelevel)
    #     return fusedleveldecomp
        
    def convertWaveletAndBack(self, imgToConvert, waveletchoice, wavelevel):
        imgdec0 = pywt.wavedec2(imgToConvert[:,:,0], waveletchoice, level=wavelevel)
        imgdec1 = pywt.wavedec2(imgToConvert[:,:,1], waveletchoice, level=wavelevel)
        imgdec2 = pywt.wavedec2(imgToConvert[:,:,2], waveletchoice, level=wavelevel)
        imgrec0 = pywt.waverec2(imgdec0, waveletchoice)
        imgrec1 = pywt.waverec2(imgdec1, waveletchoice)
        imgrec2 = pywt.waverec2(imgdec2, waveletchoice)
        newimg = np.zeros((imgrec0.shape[0], imgrec0.shape[1], 3))
        newimg[:,:,0] = imgrec0
        newimg[:,:,1] = imgrec1
        newimg[:,:,2] = imgrec2
        return newimg

    def channel_decomp_multilevel_3chan(self, currimg, progressimg, currgrayimg, progressgrayimg, waveletchoice, wavelevel):
        fusedlevelimg = currimg.copy()
        curr_coeffs0 = pywt.wavedec2(currimg[:,:,0], waveletchoice, level=wavelevel)
        recimg0 = pywt.waverec2(curr_coeffs0, waveletchoice)
        # fusedlevelimg = np.zeros((recimg0.shape[0], recimg0.shape[1], 3))
        fusedlevelimg = self.convertWaveletAndBack(fusedlevelimg, waveletchoice, wavelevel)
        fusedlevelgrayimg = currgrayimg.copy()
        print("CURRb4conv", currimg.shape)
        curr_gcoeffs0 = pywt.wavedec2(currgrayimg, waveletchoice, level=wavelevel)
        recgimg0 = pywt.waverec2(curr_gcoeffs0, waveletchoice)
        # fusedlevelgrayimg = np.zeros((recgimg0.shape[0], recgimg0.shape[1]))
        fusedlevelgrayimg = recgimg0
        proggray_coeffs = pywt.wavedec2(progressgrayimg, waveletchoice, level=wavelevel)
        recproggrayimg = pywt.waverec2(proggray_coeffs, waveletchoice)
        progressgrayimg = recproggrayimg
        currimg = self.convertWaveletAndBack(currimg, waveletchoice, wavelevel)
        progressimg = self.convertWaveletAndBack(progressimg, waveletchoice, wavelevel)
        print("CURRafterconv", currimg.shape)
        # curr_coeffs = pywt.wavedec2(currimg, waveletchoice, level=wavelevel)
        # progress_coeffs = pywt.wavedec2(progressimg, waveletchoice, level=wavelevel)
        # curr_graycoeffs = pywt.wavedec2(currgrayimg, waveletchoice, level=wavelevel)
        # progress_graycoeffs = pywt.wavedec2(progressgrayimg, waveletchoice, level=wavelevel)
        
        # LOOP FOR GRADUAL COMPARISON AT DIFFERENT WAVELET LEVELS
        # for looplevel in range(wavelevel,0,-1):
        # for looplevel in range(1, wavelevel+1):
        #     print("looplevel:", looplevel)
        #     fusedleveldecomp0, fusedleveldecomp1, fusedleveldecomp2, fusedlevelgraydecomp = self.channel_decomp_wavedec_3chan(fusedlevelimg, progressimg, fusedlevelgrayimg, progressgrayimg, waveletchoice, looplevel)
        #     fusedlevelimg[:,:,0] = pywt.waverec2(fusedleveldecomp0, waveletchoice)[0:fusedlevelimg.shape[0],0:fusedlevelimg.shape[1]]
        #     fusedlevelimg[:,:,1] = pywt.waverec2(fusedleveldecomp1, waveletchoice)[0:fusedlevelimg.shape[0],0:fusedlevelimg.shape[1]]
        #     fusedlevelimg[:,:,2] = pywt.waverec2(fusedleveldecomp2, waveletchoice)[0:fusedlevelimg.shape[0],0:fusedlevelimg.shape[1]]
        #     print('fusedleveldecomp0[0].shape', fusedleveldecomp0[0].shape)
        #     print('fusedlevelgraydecomp[0].shape', fusedlevelgraydecomp[0].shape)
        #     fusedlevelgrayimg = pywt.waverec2(fusedlevelgraydecomp, waveletchoice)
        # END LOOP FOR GRADUAL COMPARISON AT DIFFERENT WAVELET LEVELS

        # ADDED ALTERNATIVE FOR SINGLE LOOP.
        looplevel = wavelevel
        print("looplevel:", looplevel)
        fusedleveldecomp0, fusedleveldecomp1, fusedleveldecomp2, fusedlevelgraydecomp = self.channel_decomp_wavedec_3chan(fusedlevelimg, progressimg, fusedlevelgrayimg, progressgrayimg, waveletchoice, looplevel)
        # recimg0 = pywt.waverec2(fusedleveldecomp0, waveletchoice)
        # recimg1 = pywt.waverec2(fusedleveldecomp1, waveletchoice)
        # recimg2 = pywt.waverec2(fusedleveldecomp2, waveletchoice)
        # fusedlevelimg = np.zeros((recimg0.shape[0], recimg0.shape[1], 3))
        # fusedlevelimg[:,:,0] = recimg0
        # fusedlevelimg[:,:,1] = recimg1
        # fusedlevelimg[:,:,2] = recimg2
        # END ADDED ALTERNATIVE FOR SINGLE LOOP.

        print('fusedleveldecomp0[0].shape', fusedleveldecomp0[0].shape)
        print('fusedlevelgraydecomp[0].shape', fusedlevelgraydecomp[0].shape)
        # fusedlevelgrayimg = pywt.waverec2(fusedlevelgraydecomp, waveletchoice)

        ## fusedleveldecomp = self.channel_decomp_wavedec(fusedlevelimg, progressimggray, waveletchoice, wavelevel);
        
        # fusedleveldecomp0 = pywt.wavedec2(fusedlevelimg[:,:,0], waveletchoice, level=wavelevel)
        # fusedleveldecomp1 = pywt.wavedec2(fusedlevelimg[:,:,1], waveletchoice, level=wavelevel)
        # fusedleveldecomp2 = pywt.wavedec2(fusedlevelimg[:,:,2], waveletchoice, level=wavelevel)
        # fusedlevelgraydecomp = pywt.wavedec2(fusedlevelgrayimg, waveletchoice, level=wavelevel)
        return fusedleveldecomp0, fusedleveldecomp1, fusedleveldecomp2, fusedlevelgraydecomp 
        
    def channel_decomp_wavedec_3chan(self, currimg, progressimg, currgrayimg, progressgrayimg, waveletchoice, wavelevel):
        curr_coeffs0 = pywt.wavedec2(currimg[:,:,0], waveletchoice, level=wavelevel)
        curr_coeffs1 = pywt.wavedec2(currimg[:,:,1], waveletchoice, level=wavelevel)
        curr_coeffs2 = pywt.wavedec2(currimg[:,:,2], waveletchoice, level=wavelevel)
        progress_coeffs0 = pywt.wavedec2(progressimg[:,:,0], waveletchoice, level=wavelevel)
        progress_coeffs1 = pywt.wavedec2(progressimg[:,:,1], waveletchoice, level=wavelevel)
        progress_coeffs2 = pywt.wavedec2(progressimg[:,:,2], waveletchoice, level=wavelevel)
        curr_graycoeffs = pywt.wavedec2(currgrayimg, waveletchoice, level=wavelevel)
        progress_graycoeffs = pywt.wavedec2(progressgrayimg, waveletchoice, level=wavelevel)
        print('currgrayimg.shape', currgrayimg.shape)
        print('progressgrayimg.shape', progressgrayimg.shape)
        print('progress_graycoeffs[1].shape', progress_graycoeffs[1][1].shape)
        print('curr_graycoeffs[1].shape', curr_graycoeffs[1][1].shape)
        num_high_tuples = len(curr_coeffs0)
        # Init fused coeffs with LL subband, will be replaced at end.
        fusedcoeffs0 = [progress_coeffs0[0]]
        fusedcoeffs1 = [progress_coeffs1[0]]
        fusedcoeffs2 = [progress_coeffs2[0]]
        fusedgraycoeffs = [progress_graycoeffs[0]]
        for i in range(1, num_high_tuples):
            combinedecomps0, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs0[i], progress_coeffs0[i], curr_graycoeffs[i], progress_graycoeffs[i])
            combinedecomps1, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs1[i], progress_coeffs1[i], curr_graycoeffs[i], progress_graycoeffs[i])
            combinedecomps2, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs2[i], progress_coeffs2[i], curr_graycoeffs[i], progress_graycoeffs[i])

            # combinedecomps0 = self.channel_decomp_wavedec_max(currimg[:,:,0], progressimg[:,:,0], waveletchoice, wavelevel)[1]
            # combinedecomps1 = self.channel_decomp_wavedec_max(currimg[:,:,1], progressimg[:,:,1], waveletchoice, wavelevel)[1]
            # combinedecomps2 = self.channel_decomp_wavedec_max(currimg[:,:,2], progressimg[:,:,2], waveletchoice, wavelevel)[1]

            fusedcoeffs0.append((combinedecomps0[0], combinedecomps0[1], combinedecomps0[2]))
            fusedcoeffs1.append((combinedecomps1[0], combinedecomps1[1], combinedecomps1[2]))
            fusedcoeffs2.append((combinedecomps2[0], combinedecomps2[1], combinedecomps2[2]))
            print('combinedecomps0[0].shape', combinedecomps0[0].shape)
            print('combinedgraydecomps[0].shape', combinedgraydecomps[0].shape)
            fusedgraycoeffs.append((combinedgraydecomps[0], combinedgraydecomps[1], combinedgraydecomps[2]))

        print('curr_coeffs0[0].shape', curr_coeffs0[0].shape)
        lastleveldecomp0, lastlevelgraydecomp = self.combine_decomps((curr_coeffs0[0], curr_coeffs0[1]), (progress_coeffs0[0], progress_coeffs0[1]), (curr_graycoeffs[0], curr_graycoeffs[1]), (progress_graycoeffs[0], progress_graycoeffs[1]))
        lastleveldecomp1, lastlevelgraydecomp = self.combine_decomps((curr_coeffs1[0], curr_coeffs1[1]), (progress_coeffs1[0], progress_coeffs1[1]), (curr_graycoeffs[0], curr_graycoeffs[1]), (progress_graycoeffs[0], progress_graycoeffs[1]))
        lastleveldecomp2, lastlevelgraydecomp = self.combine_decomps((curr_coeffs2[0], curr_coeffs2[1]), (progress_coeffs2[0], progress_coeffs2[1]), (curr_graycoeffs[0], curr_graycoeffs[1]), (progress_graycoeffs[0], progress_graycoeffs[1]))
        fusedcoeffs0[0] = lastleveldecomp0[0]
        fusedcoeffs1[0] = lastleveldecomp1[0]
        fusedcoeffs2[0] = lastleveldecomp2[0]
        fusedgraycoeffs[0] = lastlevelgraydecomp[0]
        print('lastleveldecomp0[0]', lastleveldecomp0[0].shape)
        print('lastlevelgraydecomp[0]', lastlevelgraydecomp[0].shape)
        return fusedcoeffs0, fusedcoeffs1, fusedcoeffs2, fusedgraycoeffs
    
    # def channel_decomp_wavedec_max(self, currimggray, progressimggray, waveletchoice, wavelevel):
    #     # wavelevel = 2
    #     coeffs = pywt.wavedec2(currimggray, waveletchoice, level=wavelevel)
    #     focal_coeffs = pywt.wavedec2(progressimggray, waveletchoice, level=wavelevel)
    #     fused_coeffs = focal_coeffs.copy()
    #     num_high_tuples = len(coeffs)
    #     highpass_sum = 0
    #     focal_highpass_sum = 0
    #     fused_coeffs4comp = [np.maximum(np.abs(focal_c), np.abs(img_c)) for focal_c, img_c in zip(focal_coeffs, coeffs)]
    #     boolcoeffs = []
    #     fusedcoeffs = [focal_coeffs[0]]
    #     focalhighsum = 0
    #     fusedhighsum = 0
    #     for i in range(1, num_high_tuples):
    #         # print("I",i)
    #         # bool_coeffs0 = fused_coeffs4comp[0] == np.abs(focal_coeffs[0])
    #         bool_coeffs10 = fused_coeffs4comp[i][0] == np.abs(focal_coeffs[i][0])
    #         bool_coeffs11 = fused_coeffs4comp[i][1] == np.abs(focal_coeffs[i][1])
    #         bool_coeffs12 = fused_coeffs4comp[i][2] == np.abs(focal_coeffs[i][2])
    #         if i == 1:
    #             focalhighsum += np.abs(focal_coeffs[i][0]) + np.abs(focal_coeffs[i][1]) + np.abs(focal_coeffs[i][2])
    #             fusedhighsum += fused_coeffs4comp[i][0] + fused_coeffs4comp[i][1] + fused_coeffs4comp[i][2]
    #         # boolcoeffs.append(bool_coeffs0, ((bool_coeffs10, bool_coeffs11, bool_coeffs12)))
    #         fusedcoeffs.append((np.where(bool_coeffs10, focal_coeffs[i][0], coeffs[i][0]), np.where(bool_coeffs11, focal_coeffs[i][1], coeffs[i][1]), np.where(bool_coeffs12, focal_coeffs[i][2], coeffs[i][2])))

    #         # fused_coeffs[i] = self.channel_decomp_high_sum(coeffs[i], focal_coeffs[i])
    #         # focal_highpass_sum = np.abs(focal_coeffs[i][0]) + np.abs(focal_coeffs[i][1]) + np.abs(focal_coeffs[i][2])
    #         # highpass_sum = np.abs(coeffs[i][0]) + np.abs(coeffs[i][1]) + np.abs(coeffs[i][2])

    #     bool_coeffs0 = focalhighsum >= fusedhighsum
    #     # print('bool_coeffs0.shape', bool_coeffs0.shape)
    #     # print('focal_coeffs[0].shape', focal_coeffs[0].shape)
    #     # if wavelevel == 1:
    #     #     print("WAVELEVEL2")
    #     #     fusedcoeffs[0] = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]) # lowpass vote

    #     fusedcoeffs[0] = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]) # lowpass vote
    #     # # replace low pass choice with majority vote.
    #     # bool_coeffs0 = sum([bool_coeffs10, bool_coeffs11, bool_coeffs12]) >= 2
    #     # bool_coeffs0 = np.abs(focal_coeffs[1][0]) + np.abs(focal_coeffs[1][1]) + np.abs(focal_coeffs[1][2]) >= fused_coeffs4comp[1][0] + fused_coeffs4comp[1][1] + fused_coeffs4comp[1][2]
    #     # bool_coeffs0 = focal_highpass_sum >= highpass_sum #PREV COMMIT
    #     # print('bool_coeffs0.shape:', bool_coeffs0.shape)
    #     # print('bool_coeffs10.shape:', bool_coeffs10.shape)
    #     # print('XXXXXXbool0.shape:', bool_coeffs0.shape)
    #     # print('bool0:', bool_coeffs0)
    #     # fused_coeffs = np.where(bool_coeffs0, focal_coeffs[0], coeffs[0]), (np.where(bool_coeffs10, focal_coeffs[1][0], coeffs[1][0]), np.where(bool_coeffs11, focal_coeffs[1][1], coeffs[1][1]), np.where(bool_coeffs12, focal_coeffs[1][2], coeffs[1][2]))
    #     return fusedcoeffs
    
    def spatial_consistency_check(self, boolMatRaw):
        boolMatNew = boolMatRaw.copy()
        padMat = np.pad(boolMatRaw, [(1,1), (1,1)], mode='constant')
        boolMatNewPad = padMat.copy()
        (shapeY, shapeX) = padMat.shape
        for i in range(1, shapeY-1):
            for j in range(1, shapeX-1):
                surroundMat = padMat[i-1:i+2, j-1:j+2]
                sum3x3 = np.sum(surroundMat)
                yBorderCells = 0
                if i==1 or i == shapeY-2:
                    yBorderCells = 3
                xBorderCells = 0
                if j==1 or j == shapeX-2:
                    xBorderCells = 3
                totalBorderCells = yBorderCells + xBorderCells
                if (i==1 or i == shapeY-2) and (j==1 or j == shapeX-2):
                    totalBorderCells = totalBorderCells - 1
                # print("i:",i,"j:",j,"totalBorderCells:",totalBorderCells,"(9-totalBorderCells+1)//2:",(9-totalBorderCells+1) // 2)
                if sum3x3 >= ((9 - totalBorderCells + 1) // 2):
                    boolMatNewPad[i,j] = True
        boolMatNew = boolMatNewPad[1:shapeY-1, 1:shapeX-1]
        print('boolMatRaw.shape', boolMatRaw.shape, 'boolMatNew.shape', boolMatNew.shape)
        print('sum boolMatRaw', np.sum(boolMatRaw), 'sum boolMatNew', np.sum(boolMatNew))
        return boolMatNew

    def combine_decomps_nolow(self, currdecomp, newdecompx, currgraydecomp, newgraydecompx):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        print('newdecompx.shape', newdecompx[0].shape)
        print('currdecomp.shape', currdecomp[0].shape)
        print('newgraydecompx[0].shape', newgraydecompx[0].shape)
        print('currgraydecomp[0].shape', currgraydecomp[0].shape)
        # Old comparison
        # boolMat = np.abs(newgraydecompx[0]) + np.abs(newgraydecompx[1]) + np.abs(newgraydecompx[2]) > \
        #                  np.abs(currgraydecomp[0]) + np.abs(currgraydecomp[1]) + np.abs(currgraydecomp[2])
        # Subband consistency check, see Forster et al.
        boolMat0 = np.abs(newgraydecompx[0]) > np.abs(currgraydecomp[0])
        boolMat1 = np.abs(newgraydecompx[1]) > np.abs(currgraydecomp[1])
        boolMat2 = np.abs(newgraydecompx[2]) > np.abs(currgraydecomp[2])
        boolMatSubbandCheck = sum([boolMat0, boolMat1, boolMat2]) >= 2
        boolMat = self.spatial_consistency_check(boolMatSubbandCheck)
        # copy pixel values for all four subbands according to boolean matrix.
        # newdecompx0 = np.where(boolMat,
        #         newdecompx[0], currdecomp[0])
        print('newdecompx.shape', newdecompx[0].shape)
        print('currdecomp.shape', currdecomp[0].shape)
        print('newgraydecompx.shape', newgraydecompx[0].shape)
        print('currgraydecomp.shape', currgraydecomp[0].shape)
        print('boolMat.shape', boolMat.shape)
        newdecompx10 = np.where(boolMat,
            newdecompx[0], currdecomp[0])
        newdecompx11 = np.where(boolMat,
            newdecompx[1], currdecomp[1])
        newdecompx12 = np.where(boolMat,
            newdecompx[2], currdecomp[2])
        
        newgraydecompx10 = np.where(boolMat,
            newgraydecompx[0], currgraydecomp[0])
        newgraydecompx11 = np.where(boolMat,
            newgraydecompx[1], currgraydecomp[1])
        newgraydecompx12 = np.where(boolMat,
            newgraydecompx[2], currgraydecomp[2])
        
        print('newdecompx10.shape', newdecompx10.shape)
        print('newgraydecompx10.shape', newgraydecompx10.shape)
        newdecompx = (newdecompx10, newdecompx11, newdecompx12)
        newgraydecompx = (newgraydecompx10, newgraydecompx11, newgraydecompx12)
        return newdecompx, newgraydecompx

    def combine_decomps(self, currdecomp, newdecompx, currgraydecomp, newgraydecompx):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        # boolMat = np.abs(newgraydecompx[1][0]) + np.abs(newgraydecompx[1][1]) + np.abs(newgraydecompx[1][2]) > \
        #                  np.abs(currgraydecomp[1][0]) + np.abs(currgraydecomp[1][1]) + np.abs(currgraydecomp[1][2])
        # Subband consistency check, see Forster et al.
        boolMat0 = np.abs(newgraydecompx[1][0]) > np.abs(currgraydecomp[1][0])
        boolMat1 = np.abs(newgraydecompx[1][1]) > np.abs(currgraydecomp[1][1])
        boolMat2 = np.abs(newgraydecompx[1][2]) > np.abs(currgraydecomp[1][2])
        boolMatSubbandCheck = sum([boolMat0, boolMat1, boolMat2]) >= 2
        boolMat = self.spatial_consistency_check(boolMatSubbandCheck)
        # copy pixel values for all four subbands according to boolean matrix.
        newdecompx0 = np.where(boolMat,
                newdecompx[0], currdecomp[0])
        newdecompx10 = np.where(boolMat,
            newdecompx[1][0], currdecomp[1][0])
        newdecompx11 = np.where(boolMat,
            newdecompx[1][1], currdecomp[1][1])
        newdecompx12 = np.where(boolMat,
            newdecompx[1][2], currdecomp[1][2])
        
        newgraydecompx0 = np.where(boolMat,
                newgraydecompx[0], currgraydecomp[0])
        newgraydecompx10 = np.where(boolMat,
            newgraydecompx[1][0], currgraydecomp[1][0])
        newgraydecompx11 = np.where(boolMat,
            newgraydecompx[1][1], currgraydecomp[1][1])
        newgraydecompx12 = np.where(boolMat,
            newgraydecompx[1][2], currgraydecomp[1][2])
        
        newdecompx = newdecompx0, (newdecompx10, newdecompx11, newdecompx12)
        newgraydecompx = newgraydecompx0, (newgraydecompx10, newgraydecompx11, newgraydecompx12)
        return newdecompx, newgraydecompx

    # def combine_decomps_gray_nolow(self, newdecompx, newdecompg, currdecomp, currdecompg):
    #     # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
    #     boolMat = np.abs(newdecompg[1][0]) + np.abs(newdecompg[1][1]) + np.abs(newdecompg[1][2]) > \
    #                      np.abs(currdecompg[1][0]) + np.abs(currdecompg[1][1]) + np.abs(currdecompg[1][2])
    #     # copy pixel values for all four subbands according to boolean matrix.
    #     # newdecompx0 = np.where(boolMat,
    #     #         newdecompx[0], currdecomp[0])
    #     newdecompx10 = np.where(boolMat,
    #         newdecompx[1][0], currdecomp[1][0])
    #     newdecompx11 = np.where(boolMat,
    #         newdecompx[1][1], currdecomp[1][1])
    #     newdecompx12 = np.where(boolMat,
    #         newdecompx[1][2], currdecomp[1][2])
    #     newdecompx = (newdecompx10, newdecompx11, newdecompx12)

    #     # newdecompg0 = np.where(boolMat,
    #     #         newdecompg[0], currdecompg[0])
    #     newdecompg10 = np.where(boolMat,
    #         newdecompg[1][0], currdecompg[1][0])
    #     newdecompg11 = np.where(boolMat,
    #         newdecompg[1][1], currdecompg[1][1])
    #     newdecompg12 = np.where(boolMat,
    #         newdecompg[1][2], currdecompg[1][2])
    #     newdecompg = (newdecompg10, newdecompg11, newdecompg12)

    #     return newdecompx, newdecompg

    # def combine_decomps_gray(self, newdecompx, newdecompg, currdecomp, currdecompg):
    #     # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
    #     boolMat = np.abs(newdecompg[1][0]) + np.abs(newdecompg[1][1]) + np.abs(newdecompg[1][2]) > \
    #                      np.abs(currdecompg[1][0]) + np.abs(currdecompg[1][1]) + np.abs(currdecompg[1][2])
    #     # copy pixel values for all four subbands according to boolean matrix.
    #     newdecompx0 = np.where(boolMat,
    #             newdecompx[0], currdecomp[0])
    #     newdecompx10 = np.where(boolMat,
    #         newdecompx[1][0], currdecomp[1][0])
    #     newdecompx11 = np.where(boolMat,
    #         newdecompx[1][1], currdecomp[1][1])
    #     newdecompx12 = np.where(boolMat,
    #         newdecompx[1][2], currdecomp[1][2])
    #     newdecompx = newdecompx0, (newdecompx10, newdecompx11, newdecompx12)

    #     newdecompg0 = np.where(boolMat,
    #             newdecompg[0], currdecompg[0])
    #     newdecompg10 = np.where(boolMat,
    #         newdecompg[1][0], currdecompg[1][0])
    #     newdecompg11 = np.where(boolMat,
    #         newdecompg[1][1], currdecompg[1][1])
    #     newdecompg12 = np.where(boolMat,
    #         newdecompg[1][2], currdecompg[1][2])
    #     newdecompg = newdecompg0, (newdecompg10, newdecompg11, newdecompg12)

    #     return newdecompx, newdecompg
