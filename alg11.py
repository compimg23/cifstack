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
import dtcwt
import gc # for garbage collection

class Alg11Waveletr2dDecompComplex(object):
    def startAlg(self, image_files, alignMethod, levelarg):
        print("Algorithm11 (complex wavelet using dtcwt.Transform2d class - 7 levels) starting.")
        print('image files input list', image_files)
        image_files = list(set(image_files)) # remove duplicates
        image_files = sorted(image_files)
        print('sorted image files input list', image_files)
        img_mats = [cv2.imread(img) for img in image_files]
        print(pywt.wavelist(kind='discrete'))
        w_mainlevel = levelarg #11
        if w_mainlevel < 2: #Use default level if too low.
            w_mainlevel = 11
            print('Complex decomposition level must be at least 2 for our selection metric. Setting it to default value 11. Use the -l command-line argument to set another.')
        img_mats = alignMethod(img_mats) #remove to test w/out alignment
        num_files = len(image_files)
        print('Number of input files:', num_files)
        print('Using complex qshift wavelet with decomposition level', w_mainlevel)
        firstimg = img_mats[0]
        print('PLEASE NOTE: Transform2d().forward() will throw warnings and automatically resize input images for evenly divisible decompositions.')
        print('This does not affect the output significantly. This algorithm has been adjusted to take into account these dimension adjustments.')
        decomp1 = dtcwt.Transform2d().forward(firstimg[:,:,0], nlevels=w_mainlevel)
        decomp2 = dtcwt.Transform2d().forward(firstimg[:,:,1], nlevels=w_mainlevel)
        decomp3 = dtcwt.Transform2d().forward(firstimg[:,:,2], nlevels=w_mainlevel)
        decomp1 = self.convert_dtcwt_to_wavedec_list(decomp1)
        decomp2 = self.convert_dtcwt_to_wavedec_list(decomp2)
        decomp3 = self.convert_dtcwt_to_wavedec_list(decomp3)
        wdecompgimg = cv2.cvtColor(img_mats[0], cv2.COLOR_BGR2GRAY)
        newdecompg = dtcwt.Transform2d().forward(wdecompgimg, nlevels=w_mainlevel)
        newdecompg = self.convert_dtcwt_to_wavedec_list(newdecompg)
        newdecomp1 = decomp1
        newdecomp2 = decomp2
        newdecomp3 = decomp3
        newdecompimg = firstimg.copy()
        for j in range(num_files):
            gc.collect()
            currimg = img_mats[j]
            print("Running wavelet decomposition, iteration", j, "of", num_files)

            imggray = cv2.cvtColor(currimg, cv2.COLOR_BGR2GRAY)

            #NEW MULTILEVEL DECOMP
            print("Performing pointwise maximum comparison")

            print('PLEASE NOTE: Transform2d().forward() will throw warnings and automatically resize input images for evenly divisible decompositions.')
            print('This does not affect the output significantly. This algorithm has been adjusted to take into account these dimension adjustments.')
            newdecomp1, newdecomp2, newdecomp3, newdecompg = self.channel_decomp_multilevel_3chan(currimg.copy(), newdecompimg.copy(), imggray.copy(), wdecompgimg.copy(), w_mainlevel)
            gc.collect()
            #END NEW MULTILEVEL DECOMP

            # NEW RECOMP
            print("Recompositing current image from decompositions.")
            dtcwt_newdecompg = self.convert_wavedec_list_to_dtcwt(newdecompg)
            graychan = dtcwt.Transform2d().inverse(dtcwt_newdecompg)
            dtcwt_newdecomp1 = self.convert_wavedec_list_to_dtcwt(newdecomp1)
            recchan = dtcwt.Transform2d().inverse(dtcwt_newdecomp1)
            recompimggray = np.zeros((graychan.shape[0], graychan.shape[1]))
            recompimg = np.zeros((recchan.shape[0], recchan.shape[1], 3))
            gc.collect()
            recompimggray[:,:] = dtcwt.Transform2d().inverse(self.convert_wavedec_list_to_dtcwt(newdecompg))[0:recompimggray.shape[0],0:recompimggray.shape[1]]
            recompimg[:,:,0] = dtcwt.Transform2d().inverse(self.convert_wavedec_list_to_dtcwt(newdecomp1))[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,1] = dtcwt.Transform2d().inverse(self.convert_wavedec_list_to_dtcwt(newdecomp2))[0:recompimg.shape[0],0:recompimg.shape[1]]
            recompimg[:,:,2] = dtcwt.Transform2d().inverse(self.convert_wavedec_list_to_dtcwt(newdecomp3))[0:recompimg.shape[0],0:recompimg.shape[1]]
            gc.collect()
            print("Saving results for next iteration...")
            wdecompgimg = recompimggray
            newdecompimg = recompimg
            # END NEW RECOMP

            recompname = 'OutputFolder/Transform2d_recomp_' + str(j) + '.jpg'
            print('Saving recomposition')
            cv2.imwrite(recompname, recompimg)
            print('Recomposition saved in ' + recompname)

            recompgrayname = 'OutputFolder/Transform2d_recomp_gray' + str(j) + '.jpg'
            print('Saving gray recomposition')
            cv2.imwrite(recompgrayname, recompimggray)
            print('Recomposition gray saved in ' + recompgrayname)
            

        print('Saving final recomposition')
        cv2.imwrite(recompname, recompimg)

        print('FINISHED METHOD. Returning low pass filtered image (smaller size).')
        return recompimg
    
    def absmax(self, a, b):
        return np.where(np.abs(a) > np.abs(b), a, b)
        
    # Used to make sure dimensions of image are consistent.
    def convertWaveletAndBack(self, imgToConvert, wavelevel):
        imgdec0 = dtcwt.Transform2d().forward(imgToConvert[:,:,0], nlevels=wavelevel)
        imgdec1 = dtcwt.Transform2d().forward(imgToConvert[:,:,1], nlevels=wavelevel)
        imgdec2 = dtcwt.Transform2d().forward(imgToConvert[:,:,2], nlevels=wavelevel)
        imgrec0 = dtcwt.Transform2d().inverse(imgdec0)
        imgrec1 = dtcwt.Transform2d().inverse(imgdec1)
        imgrec2 = dtcwt.Transform2d().inverse(imgdec2)
        newimg = np.zeros((imgrec0.shape[0], imgrec0.shape[1], 3))
        newimg[:,:,0] = imgrec0
        newimg[:,:,1] = imgrec1
        newimg[:,:,2] = imgrec2
        return newimg

    # Convert to format that we use in selection method.
    def convert_dtcwt_to_wavedec_list(self, complexDecomp):
        decList = []
        decList.append(complexDecomp.lowpass)
        highLen = len(complexDecomp.highpasses)
        for i in range(highLen-1, -1, -1):
            gc.collect()
            highpassTuple = (complexDecomp.highpasses[i][:,:,0],complexDecomp.highpasses[i][:,:,1],complexDecomp.highpasses[i][:,:,2],
                             complexDecomp.highpasses[i][:,:,3],complexDecomp.highpasses[i][:,:,4],complexDecomp.highpasses[i][:,:,5])
            decList.append(highpassTuple)
        gc.collect()
        return decList

    # Convert back to format for inverse wavelet reconstruction.
    def convert_wavedec_list_to_dtcwt(self, decList):
        decLen = len(decList)
        highpassList = []
        highpassFirst = np.zeros((decList[decLen-1][0].shape[0], decList[decLen-1][0].shape[1], 6)).astype(complex)
        for j in range(6):
            highpassFirst[:,:,j] = decList[decLen-1][j]
        highpassList.append(highpassFirst)
        highpassTuple = tuple(highpassList)
        for i in range(decLen-2, 0, -1):
            gc.collect()
            highpass = np.zeros((decList[i][0].shape[0], decList[i][0].shape[1], 6)).astype(complex)
            for j in range(6):
                highpass[:,:,j] = decList[i][j]
            highpassList.append(highpass)
        highpassTuple = tuple(highpassList)
        complexDecomp = dtcwt.Pyramid(decList[0], highpassTuple)
        gc.collect()
        return complexDecomp
    
    # Prepare progress image and currently iterated image and send it to decomposition comparison.
    def channel_decomp_multilevel_3chan(self, currimg, progressimg, currgrayimg, progressgrayimg, wavelevel):
        fusedlevelimg = currimg.copy()#
        curr_gcoeffs0 = dtcwt.Transform2d().forward(currgrayimg, nlevels=wavelevel)
        recgimg0 = dtcwt.Transform2d().inverse(curr_gcoeffs0)
        fusedlevelgrayimg = recgimg0
        proggray_coeffs = dtcwt.Transform2d().forward(progressgrayimg, nlevels=wavelevel)
        recproggrayimg = dtcwt.Transform2d().inverse(proggray_coeffs)
        progressgrayimg = recproggrayimg
        currimg = self.convertWaveletAndBack(currimg, wavelevel)
        progressimg = self.convertWaveletAndBack(progressimg, wavelevel)

        # ADDED ALTERNATIVE FOR SINGLE LOOP.
        looplevel = wavelevel
        fusedleveldecomp0, fusedleveldecomp1, fusedleveldecomp2, fusedlevelgraydecomp = self.channel_decomp_wavedec_3chan(fusedlevelimg, progressimg, fusedlevelgrayimg, progressgrayimg, looplevel)
        # END ADDED ALTERNATIVE FOR SINGLE LOOP.

        gc.collect()
        return fusedleveldecomp0, fusedleveldecomp1, fusedleveldecomp2, fusedlevelgraydecomp 
        
    # Loop through high pass decompositions for all 3 RGB channels.
    def channel_decomp_wavedec_3chan(self, currimg, progressimg, currgrayimg, progressgrayimg, wavelevel):
        gc.collect()
        curr_coeffs0 = dtcwt.Transform2d().forward(currimg[:,:,0], nlevels=wavelevel)
        gc.collect()
        curr_coeffs1 = dtcwt.Transform2d().forward(currimg[:,:,1], nlevels=wavelevel)
        gc.collect()
        curr_coeffs2 = dtcwt.Transform2d().forward(currimg[:,:,2], nlevels=wavelevel)
        gc.collect()
        curr_coeffs0 = self.convert_dtcwt_to_wavedec_list(curr_coeffs0)
        gc.collect()
        curr_coeffs1 = self.convert_dtcwt_to_wavedec_list(curr_coeffs1)
        gc.collect()
        curr_coeffs2 = self.convert_dtcwt_to_wavedec_list(curr_coeffs2)
        gc.collect()
        progress_coeffs0 = dtcwt.Transform2d().forward(progressimg[:,:,0], nlevels=wavelevel)
        gc.collect()
        progress_coeffs1 = dtcwt.Transform2d().forward(progressimg[:,:,1], nlevels=wavelevel)
        gc.collect()
        progress_coeffs2 = dtcwt.Transform2d().forward(progressimg[:,:,2], nlevels=wavelevel)
        gc.collect()
        progress_coeffs0 = self.convert_dtcwt_to_wavedec_list(progress_coeffs0)
        gc.collect()
        progress_coeffs1 = self.convert_dtcwt_to_wavedec_list(progress_coeffs1)
        gc.collect()
        progress_coeffs2 = self.convert_dtcwt_to_wavedec_list(progress_coeffs2)
        gc.collect()
        curr_graycoeffs = dtcwt.Transform2d().forward(currgrayimg, nlevels=wavelevel)
        gc.collect()
        curr_graycoeffs = self.convert_dtcwt_to_wavedec_list(curr_graycoeffs)
        gc.collect()
        progress_graycoeffs = dtcwt.Transform2d().forward(progressgrayimg, nlevels=wavelevel)
        gc.collect()
        progress_graycoeffs = self.convert_dtcwt_to_wavedec_list(progress_graycoeffs)
        gc.collect()
        num_high_tuples = len(curr_coeffs0)
        # Init fused coeffs with LL subband, will be replaced at end.
        fusedcoeffs0 = [progress_coeffs0[0]]
        fusedcoeffs1 = [progress_coeffs1[0]]
        fusedcoeffs2 = [progress_coeffs2[0]]
        fusedgraycoeffs = [progress_graycoeffs[0]]
        ## Loop through all the high-pass subband levels, 3 high-pass subbands in each level.
        for i in range(1, num_high_tuples):
            gc.collect()
            print('Processing high-pass decomposition subbands in level ', i, 'of', num_high_tuples, '.')
            combinedecomps0, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs0[i], progress_coeffs0[i], curr_graycoeffs[i], progress_graycoeffs[i])
            combinedecomps1, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs1[i], progress_coeffs1[i], curr_graycoeffs[i], progress_graycoeffs[i])
            combinedecomps2, combinedgraydecomps = self.combine_decomps_nolow(curr_coeffs2[i], progress_coeffs2[i], curr_graycoeffs[i], progress_graycoeffs[i])

            fusedcoeffs0.append((combinedecomps0[0], combinedecomps0[1], combinedecomps0[2], combinedecomps0[3], combinedecomps0[4], combinedecomps0[5]))
            fusedcoeffs1.append((combinedecomps1[0], combinedecomps1[1], combinedecomps1[2], combinedecomps1[3], combinedecomps1[4], combinedecomps1[5]))
            fusedcoeffs2.append((combinedecomps2[0], combinedecomps2[1], combinedecomps2[2], combinedecomps2[3], combinedecomps2[4], combinedecomps2[5]))
            fusedgraycoeffs.append((combinedgraydecomps[0], combinedgraydecomps[1], combinedgraydecomps[2], combinedgraydecomps[3], combinedgraydecomps[4], combinedgraydecomps[5]))

        lastleveldecomp0, lastlevelgraydecomp = self.combine_decomps((curr_coeffs0[0], curr_coeffs0[2]), (progress_coeffs0[0], progress_coeffs0[2]), (curr_graycoeffs[0], curr_graycoeffs[2]), (progress_graycoeffs[0], progress_graycoeffs[2]))
        lastleveldecomp1, lastlevelgraydecomp = self.combine_decomps((curr_coeffs1[0], curr_coeffs1[2]), (progress_coeffs1[0], progress_coeffs1[2]), (curr_graycoeffs[0], curr_graycoeffs[2]), (progress_graycoeffs[0], progress_graycoeffs[2]))
        lastleveldecomp2, lastlevelgraydecomp = self.combine_decomps((curr_coeffs2[0], curr_coeffs2[2]), (progress_coeffs2[0], progress_coeffs2[2]), (curr_graycoeffs[0], curr_graycoeffs[2]), (progress_graycoeffs[0], progress_graycoeffs[2]))
        fusedcoeffs0[0] = lastleveldecomp0#[0]
        fusedcoeffs1[0] = lastleveldecomp1#[0]
        fusedcoeffs2[0] = lastleveldecomp2#[0]
        fusedgraycoeffs[0] = lastlevelgraydecomp#[0]
        gc.collect()
        return fusedcoeffs0, fusedcoeffs1, fusedcoeffs2, fusedgraycoeffs
    
    # Check majority pixels in 3x3 Kernel, including border cases.
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
                if sum3x3 >= ((9 - totalBorderCells + 1) // 2):
                    boolMatNewPad[i,j] = True
        boolMatNew = boolMatNewPad[1:shapeY-1, 1:shapeX-1]
        return boolMatNew

    # Compare high pass subbands without low pass subband.
    def combine_decomps_nolow(self, currdecomp, newdecompx, currgraydecomp, newgraydecompx):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        # Old comparison
        # boolMat = np.abs(newgraydecompx[0]) + np.abs(newgraydecompx[1]) + np.abs(newgraydecompx[2]) > \
        #                  np.abs(currgraydecomp[0]) + np.abs(currgraydecomp[1]) + np.abs(currgraydecomp[2])
        # Subband consistency check, see Forster et al.
        boolMat0 = np.abs(newgraydecompx[0]) > np.abs(currgraydecomp[0])
        boolMat1 = np.abs(newgraydecompx[1]) > np.abs(currgraydecomp[1])
        boolMat2 = np.abs(newgraydecompx[2]) > np.abs(currgraydecomp[2])
        boolMat3 = np.abs(newgraydecompx[3]) > np.abs(currgraydecomp[3])
        boolMat4 = np.abs(newgraydecompx[4]) > np.abs(currgraydecomp[4])
        boolMat5 = np.abs(newgraydecompx[5]) > np.abs(currgraydecomp[5])
        boolMatSubbandCheck = sum([boolMat0, boolMat1, boolMat2, boolMat3, boolMat4, boolMat5]) >= 3
        boolMat = self.spatial_consistency_check(boolMatSubbandCheck)
        # copy pixel values for all four subbands according to boolean matrix.
        newdecompx10 = np.where(boolMat,
            newdecompx[0], currdecomp[0])
        newdecompx11 = np.where(boolMat,
            newdecompx[1], currdecomp[1])
        newdecompx12 = np.where(boolMat,
            newdecompx[2], currdecomp[2])
        newdecompx13 = np.where(boolMat,
            newdecompx[3], currdecomp[3])
        newdecompx14 = np.where(boolMat,
            newdecompx[4], currdecomp[4])
        newdecompx15 = np.where(boolMat,
            newdecompx[5], currdecomp[5])
        
        newgraydecompx10 = np.where(boolMat,
            newgraydecompx[0], currgraydecomp[0])
        newgraydecompx11 = np.where(boolMat,
            newgraydecompx[1], currgraydecomp[1])
        newgraydecompx12 = np.where(boolMat,
            newgraydecompx[2], currgraydecomp[2])
        newgraydecompx13 = np.where(boolMat,
            newgraydecompx[3], currgraydecomp[3])
        newgraydecompx14 = np.where(boolMat,
            newgraydecompx[4], currgraydecomp[4])
        newgraydecompx15 = np.where(boolMat,
            newgraydecompx[5], currgraydecomp[5])
        
        newdecompx = (newdecompx10, newdecompx11, newdecompx12, newdecompx13, newdecompx14, newdecompx15)
        newgraydecompx = (newgraydecompx10, newgraydecompx11, newgraydecompx12, newgraydecompx13, newgraydecompx14, newgraydecompx15)
        return newdecompx, newgraydecompx

    # Compare high pass subbands and update low pass subband.
    def combine_decomps(self, currdecomp, newdecompx, currgraydecomp, newgraydecompx):
        # Boolean matrix tells us for each pixel which image has the three high-pass subband pixels with greatest total abs value.
        # Old comparison
        # boolMat = np.abs(newgraydecompx[1][0]) + np.abs(newgraydecompx[1][1]) + np.abs(newgraydecompx[1][2]) > \
        #                  np.abs(currgraydecomp[1][0]) + np.abs(currgraydecomp[1][1]) + np.abs(currgraydecomp[1][2])
        # Subband consistency check, see Forster et al.
        boolMat0 = np.abs(newgraydecompx[1][0]) > np.abs(currgraydecomp[1][0])
        boolMat1 = np.abs(newgraydecompx[1][1]) > np.abs(currgraydecomp[1][1])
        boolMat2 = np.abs(newgraydecompx[1][2]) > np.abs(currgraydecomp[1][2])
        boolMatSubbandCheck = sum([boolMat0, boolMat1, boolMat2]) >= 2

        boolMat = np.abs(newgraydecompx[0]) > np.abs(currgraydecomp[0])
        boolMat = self.spatial_consistency_check(boolMat) #boolMat #(boolMatSubbandCheck)
        
        # copy pixel values for all four subbands according to boolean matrix.
        newdecompx0resize = newdecompx[0][0:newdecompx[1][0].shape[0],0:newdecompx[1][0].shape[1]]
        newgraydecompx0resize = newdecompx[0][0:newdecompx[1][0].shape[0],0:newdecompx[1][0].shape[1]]
        currdecomp0resize = newdecompx[0][0:newdecompx[1][0].shape[0],0:newdecompx[1][0].shape[1]]
        currgraydecomp0resize = newdecompx[0][0:newdecompx[1][0].shape[0],0:newdecompx[1][0].shape[1]]
        newdecompx0 = np.where(boolMat,
                newdecompx[0], currdecomp[0]) #[0:newdecompx[0].shape[0],0:newdecompx[0].shape[1]]
        newdecompx0[0:newdecompx[1][0].shape[0],0:newdecompx[1][0].shape[1]] = np.where(boolMatSubbandCheck,
                newdecompx0resize, currdecomp0resize)
        
        newgraydecompx0 = np.where(boolMat,
                newgraydecompx[0], currgraydecomp[0])
        newgraydecompx0[0:newdecompx[1][0].shape[0],0:newdecompx[1][0].shape[1]] = np.where(boolMatSubbandCheck,
                newgraydecompx0resize, currgraydecomp0resize)
        
        return newdecompx0, newgraydecompx0