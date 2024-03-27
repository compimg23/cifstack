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
import gc # for garbage collection

class Alg10Waveletr2dDecompL2(object):
    def startAlg(self, image_files, alignMethod, levelarg):
        print("Algorithm10 (wavelet using pywt.wavedec2 method & consistency checks - w_mainlevel levels) starting.")
        print('image files', image_files)
        image_files = list(set(image_files)) # remove duplicates
        image_files = sorted(image_files)
        print('sorted image files', image_files)
        img_mats = [cv2.imread(img) for img in image_files]
        print(pywt.wavelist(kind='discrete'))
        w_mainlevel = levelarg #1
        waveletchoice = 'db5'
        if w_mainlevel < 1: #Use default level if too low.
            w_mainlevel = 1
            print('Real wavelet decomposition level must be at least 1. Setting it to default value 1. Use the -l command-line argument to set another value.')
        img_mats = alignMethod(img_mats)
        num_files = len(image_files)
        print('Number of input files:', num_files)
        print('Using wavelet', waveletchoice, 'with decomposition level', w_mainlevel)
        print('Outputting possible wavelets for reference...')
        print(pywt.families())
        # print('wavelist', pywt.wavelist(family)) # Use to print wavelets of a given family variable from pywt.families().
        firstimg = img_mats[0]
        decomp1 = pywt.wavedec2(firstimg[:,:,0], waveletchoice, level=w_mainlevel)
        decomp2 = pywt.wavedec2(firstimg[:,:,1], waveletchoice, level=w_mainlevel)
        decomp3 = pywt.wavedec2(firstimg[:,:,2], waveletchoice, level=w_mainlevel)
        wdecompgimg = cv2.cvtColor(img_mats[0], cv2.COLOR_BGR2GRAY)
        newdecompg = pywt.wavedec2(wdecompgimg, waveletchoice, level=w_mainlevel)
        newdecomp1 = decomp1
        newdecomp2 = decomp2
        newdecomp3 = decomp3
        newdecompimg = firstimg.copy()
        for j in range(num_files):
            gc.collect()
            currimg = img_mats[j]
            print("Running wavelet decomposition.")

            imggray = cv2.cvtColor(currimg, cv2.COLOR_BGR2GRAY)
            
            print("Performing pointwise maximum comparison")

            newdecomp1, newdecomp2, newdecomp3, newdecompg = self.channel_decomp_multilevel_3chan(currimg.copy(), newdecompimg.copy(), imggray.copy(), wdecompgimg.copy(), waveletchoice, w_mainlevel)
            #END NEW MULTILEVEL DECOMP

            # NEW RECOMP
            # Why is gray shape different from color channel shapes?
            print("Recompositing current image from deocmpositions.")
            graychan = pywt.waverec2(newdecompg, waveletchoice)#[0:recompimggray.shape[0],0:recompimggray.shape[1]]
            recchan = pywt.waverec2(newdecomp1, waveletchoice)#[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimggray = np.zeros((graychan.shape[0], graychan.shape[1]))
            recompimg = np.zeros((recchan.shape[0], recchan.shape[1], 3))
            gc.collect()
            recompimggray[:,:] = pywt.waverec2(newdecompg, waveletchoice)[0:recompimggray.shape[0],0:recompimggray.shape[1]]
            recompimg[:,:,0] = pywt.waverec2(newdecomp1, waveletchoice)[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,1] = pywt.waverec2(newdecomp2, waveletchoice)[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,2] = pywt.waverec2(newdecomp3, waveletchoice)[0:recompimg.shape[0],0:recompimg.shape[1]]
            print("Saving results for next iteration...")
            wdecompgimg = recompimggray
            newdecompimg = recompimg
            # END NEW RECOMP

            recompname = 'OutputFolder/dwtrec2_' + waveletchoice + '_recomp_' + str(j) + '.jpg'
            print('Saving color recomposition')
            cv2.imwrite(recompname, recompimg)
            print('Recomposition saved in ' + recompname)

            recompgrayname = 'OutputFolder/dwtrec2_' + waveletchoice + '_recomp_gray' + str(j) + '.jpg'
            print('Saving gray recomposition')
            cv2.imwrite(recompgrayname, recompimggray)
            print('Recomposition gray saved in ' + recompgrayname)

        print('Saving recomposition')
        # im1.save(recompname)
        cv2.imwrite(recompname, recompimg)

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return recompimg
    
    def absmax(self, a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)
        
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
        curr_gcoeffs0 = pywt.wavedec2(currgrayimg, waveletchoice, level=wavelevel)
        recgimg0 = pywt.waverec2(curr_gcoeffs0, waveletchoice)
        fusedlevelgrayimg = recgimg0
        proggray_coeffs = pywt.wavedec2(progressgrayimg, waveletchoice, level=wavelevel)
        recproggrayimg = pywt.waverec2(proggray_coeffs, waveletchoice)
        progressgrayimg = recproggrayimg
        currimg = self.convertWaveletAndBack(currimg, waveletchoice, wavelevel)
        progressimg = self.convertWaveletAndBack(progressimg, waveletchoice, wavelevel)

        # ADDED ALTERNATIVE FOR SINGLE LOOP.
        looplevel = wavelevel
        print("looplevel (# of high-pass tuples):", looplevel)
        gc.collect()
        fusedleveldecomp0, fusedleveldecomp1, fusedleveldecomp2, fusedlevelgraydecomp = self.channel_decomp_wavedec_3chan(fusedlevelimg, progressimg, fusedlevelgrayimg, progressgrayimg, waveletchoice, looplevel)

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
        num_high_tuples = len(curr_coeffs0)
        # Init fused coeffs with LL subband, will be replaced at end.
        fusedcoeffs0 = [progress_coeffs0[0]]
        fusedcoeffs1 = [progress_coeffs1[0]]
        fusedcoeffs2 = [progress_coeffs2[0]]
        fusedgraycoeffs = [progress_graycoeffs[0]]
        for i in range(1, num_high_tuples):
            gc.collect()
            print("Combining decomposition of high-pass tuple", i)
            combinedecomps0, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs0[i], progress_coeffs0[i], curr_graycoeffs[i], progress_graycoeffs[i])
            combinedecomps1, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs1[i], progress_coeffs1[i], curr_graycoeffs[i], progress_graycoeffs[i])
            combinedecomps2, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs2[i], progress_coeffs2[i], curr_graycoeffs[i], progress_graycoeffs[i])

            fusedcoeffs0.append((combinedecomps0[0], combinedecomps0[1], combinedecomps0[2]))
            fusedcoeffs1.append((combinedecomps1[0], combinedecomps1[1], combinedecomps1[2]))
            fusedcoeffs2.append((combinedecomps2[0], combinedecomps2[1], combinedecomps2[2]))
            fusedgraycoeffs.append((combinedgraydecomps[0], combinedgraydecomps[1], combinedgraydecomps[2]))

        gc.collect()
        lastleveldecomp0, lastlevelgraydecomp = self.combine_decomps((curr_coeffs0[0], curr_coeffs0[1]), (progress_coeffs0[0], progress_coeffs0[1]), (curr_graycoeffs[0], curr_graycoeffs[1]), (progress_graycoeffs[0], progress_graycoeffs[1]))
        lastleveldecomp1, lastlevelgraydecomp = self.combine_decomps((curr_coeffs1[0], curr_coeffs1[1]), (progress_coeffs1[0], progress_coeffs1[1]), (curr_graycoeffs[0], curr_graycoeffs[1]), (progress_graycoeffs[0], progress_graycoeffs[1]))
        lastleveldecomp2, lastlevelgraydecomp = self.combine_decomps((curr_coeffs2[0], curr_coeffs2[1]), (progress_coeffs2[0], progress_coeffs2[1]), (curr_graycoeffs[0], curr_graycoeffs[1]), (progress_graycoeffs[0], progress_graycoeffs[1]))
        fusedcoeffs0[0] = lastleveldecomp0[0]
        fusedcoeffs1[0] = lastleveldecomp1[0]
        fusedcoeffs2[0] = lastleveldecomp2[0]
        fusedgraycoeffs[0] = lastlevelgraydecomp[0]
        return fusedcoeffs0, fusedcoeffs1, fusedcoeffs2, fusedgraycoeffs
    
    
    def spatial_consistency_check(self, boolMatRaw):
        boolMatNew = boolMatRaw.copy()
        padMat = np.pad(boolMatRaw, [(1,1), (1,1)], mode='constant')
        boolMatNewPad = padMat.copy()
        (shapeY, shapeX) = padMat.shape
        for i in range(1, shapeY-1):
            for j in range(1, shapeX-1):
                # gc.collect() # Unfortunately this collect() causes the program to freeze.
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
        gc.collect()
        return boolMatNew

    def combine_decomps_nolow(self, currdecomp, newdecompx, currgraydecomp, newgraydecompx):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        # Old comparison
        # boolMat = np.abs(newgraydecompx[0]) + np.abs(newgraydecompx[1]) + np.abs(newgraydecompx[2]) > \
        #                  np.abs(currgraydecomp[0]) + np.abs(currgraydecomp[1]) + np.abs(currgraydecomp[2])
        # Subband consistency check, see Forster et al.
        boolMat0 = np.abs(newgraydecompx[0]) > np.abs(currgraydecomp[0])
        boolMat1 = np.abs(newgraydecompx[1]) > np.abs(currgraydecomp[1])
        boolMat2 = np.abs(newgraydecompx[2]) > np.abs(currgraydecomp[2])
        boolMatSubbandCheck = sum([boolMat0, boolMat1, boolMat2]) >= 2
        boolMat = self.spatial_consistency_check(boolMatSubbandCheck)
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
        
        newdecompx = (newdecompx10, newdecompx11, newdecompx12)
        newgraydecompx = (newgraydecompx10, newgraydecompx11, newgraydecompx12)
        gc.collect()
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
        gc.collect()
        return newdecompx, newgraydecompx