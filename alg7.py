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

class Alg7WaveletLaplace(object):
    def startAlg(self, image_files, alignMethod):
        print("Algorithm7 (wavelet with laplacian for LL subband) starting.")
        print('image files', image_files)
        image_files = sorted(image_files)
        print('sorted image files', image_files)
        img_mats = [cv2.imread(img) for img in image_files]
        # print("SKIPPING alignment module.")
        # img_mats = alignment.align_images_compare_last(img_mats)
        img_mats = alignMethod(img_mats)
        print("typeimgmats",type(img_mats))
        print("typeimgmats0",type(img_mats[0][0,0,0]))
        print("imgmats0shape",img_mats[0].shape)
        print("imgmats0 max", img_mats[0].max())
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
                newdecomp1 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                newdecomp2 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))
                newdecomp3 = np.zeros_like(decomp1[0]), (np.zeros_like(decomp1[1][0]), np.zeros_like(decomp1[1][1]), np.zeros_like(decomp1[1][2]))

                laps = []
                llimgs = []
                for j in range(num_files):
                    currimg = img_mats[j]

                    # laps.append(doLap(cv2.cvtColor(currimg,cv2.COLOR_BGR2GRAY)))
                    # curr = doLap(cv2.cvtColor(currimg,cv2.COLOR_BGR2GRAY))
                    # cv2.imwrite(transFolder + 'LLlaplace' + str(i) + '.png', curr)

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
                    
                    #save Low pass image in list for Laplacian
                    llimg = np.zeros((LL1.shape[0], LL1.shape[1], 3))
                    llimg[:,:,0] = LL1
                    llimg[:,:,1] = LL2
                    llimg[:,:,2] = LL3
                    llimgs.append(llimg)

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

        print("RUNNING LAPLACIAN...")
        print("llimgs", len(llimgs))
        print("llimgs0", llimgs[0].shape)
        newLL = self.laplacian(llimgs, img_mats) #create new low pass (LL) subband
        waveletchoice = 'haar'
        decomp1 = pywt.dwt2(recompimg[:,:,0], waveletchoice, mode='per')
        self.saveDecomp(decomp1, 'final1', waveletchoice)
        decomp2 = pywt.dwt2(recompimg[:,:,1], waveletchoice, mode='per')
        self.saveDecomp(decomp1, 'final2', waveletchoice)
        decomp3 = pywt.dwt2(recompimg[:,:,2], waveletchoice, mode='per')
        self.saveDecomp(decomp1, 'final3', waveletchoice)
        # use laplacian LL instead in recompimg
        print("newLLshape", newLL.shape)
        print("decomp1shape", decomp1[1][0].shape)
        decomp1 = (newLL[:,:,0], (decomp1[1][0], decomp1[1][1], decomp1[1][2]))
        decomp2 = (newLL[:,:,1], (decomp2[1][0], decomp2[1][1], decomp2[1][2]))
        decomp3 = (newLL[:,:,2], (decomp3[1][0], decomp3[1][1], decomp3[1][2]))
        recompimg[:,:,0] = pywt.idwt2(decomp1, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
        recompimg[:,:,1] = pywt.idwt2(decomp2, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]
        recompimg[:,:,2] = pywt.idwt2(decomp3, waveletchoice, mode='per')[0:recompimg.shape[0],0:recompimg.shape[1]]

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return recompimg
    
    def saveDecomp(self, decomp, filename, waveletchoice):
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
        figname = 'OutputFolder/out' + filename + '_dwt' + str(filename) + '_' + waveletchoice + '.jpg'
        plt.savefig(figname)

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

    #   Compute the gradient map of the image
    def doLap(self, image):

        # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
        kernel_size = 5         # Size of the laplacian window
        blur_size = 5           # How big of a kernal to use for the gaussian blur
                                # Generally, keeping these two values the same or very
                                #  close works well
                                # Also, odd numbers, please...

        blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
        return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)


    def laplacian(self, images, images2):
        # print("IMAGES",images)

        print ("Computing the laplacian of the blurred images")
        laps = []
        transFolder = 'TransFolder/'
        print("type images0", type(images[0][0,0,0]))
        print("images0 shape", images[0].shape)
        print("images0 min", images[0].min())
        img0max = images[0].max()
        # images[0] = (images[0] * (255 / img0max)).astype(np.uint8)
        print("type images0b", type(images[0][0,0,0]))
        print("images0 min", images[0].min())
        print("images0 max", images[0].max())
        # images[0] = cv2.cvtColor(images[0], cv2.COLOR_RGB2HSV)
        print("images0 max normal255", images[0].max())
        cv2.imwrite(transFolder + 'images0max.png', images[0])
        for i in range(len(images)):
            print ("Lap {}".format(i))
            images[i] = (images[i] * (255 / img0max)).astype(np.uint8)
            laps.append(self.doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)))
            curr = self.doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))
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
        result = result * img0max / 255

        return result
