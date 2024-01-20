import numpy as np
import cv2

transFolder = 'TransFolder/'

# This file works better for 'clock' image set. Alignment mask 0 around edges (20 pixels).
def findHomography(image_2, image_1, warp_matrix = np.eye(2, 3, dtype=np.float32)):
    # BEGIN NEW CODE FOR findTransformECC()
    warp_mode = cv2.MOTION_AFFINE
    num_iter = 50 #1000 #smaller is faster & less exact
    termination_eps = 0.001 #1e-7 #smaller is faster & less exact
    # warp_matrix = np.eye(2, 3, dtype=np.float32) #3x3 matrix for HOMOGRAPHY, 2x3 for AFFINE
    # warp_matrix[0][0] = 1.0;
    # warp_matrix[1][1] = 1.0;
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    print("WARPMAT", warp_matrix)
    onesMask = np.ones_like(image_1)
    onesMask[0:20,:] = 0
    onesMask[-20:-1,:] = 0
    onesMask[:,0:20] = 0
    onesMask[:,-20:-1] = 0
    # print("onesMask",onesMask[-23:-18,-23:-18])
    # END NEW CODE

    print("finding transform")
    # homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)
    # (cc, homography) = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_MAX_ITER, num_iter,termination_eps))
    (cc, homography) = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, termination_eps), inputMask=onesMask, gaussFiltSize=3)
    # (cc, homography) = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, termination_eps))

    return homography

def align_images_compare_last(images):
    print("New align_images_compare_last()")
    outimages = []    #   We assume that image 0 is
    numImages = len(images)
    # image1gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)

    outimages = images
    outimages[numImages-1] = images[numImages-1]
    for i in range(numImages-2, -1, -1):
    # for i in range(1,numImages): # for comparing first image instead of last.
        print ("Aligning image {}".format(i))
        compareIndex = numImages-1
        hom = findHomography(cv2.cvtColor(images[compareIndex],cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))

        #warpPerspective for MOTION_HOMOGRAPHY mode.
        # newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)

        #warpAffine for MOTION_AFFINE mode.
        # newimage = images[i]
        newimage = cv2.warpAffine(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        cv2.imwrite(transFolder + 'transimage' + str(i) + '.png', newimage)
        # outimages.append(newimage)
        outimages[i] = newimage

        # If you find that there's a large amount of ghosting, it may be because one or more of the input
        # images gets misaligned.  Outputting the aligned images may help diagnose that.
        # cv2.imwrite(transFolder + "aligned{}.png".format(i), newimage)

    # # add last image that was compared to without transforming it.
    # outimages.append(images[numImages-1])
    return outimages

def align_images_compare_first(images):
    print("New align_images_compare_first()")
    outimages = []    #   We assume that image 0 is
    numImages = len(images)
    # image1gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)

    # add last image that was compared to without transforming it.
    outimages.append(images[0])

    for i in range(1,numImages):
    # for i in range(1,numImages): # for comparing first image instead of last.
        print ("Aligning image {}".format(i))
        compareIndex = 0
        hom = findHomography(cv2.cvtColor(images[compareIndex],cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))

        #warpPerspective for MOTION_HOMOGRAPHY mode.
        # newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)

        #warpAffine for MOTION_AFFINE mode.
        # newimage = cv2.warpAffine(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        newimage = cv2.warpAffine(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        cv2.imwrite(transFolder + 'transimage' + str(i) + '.png', newimage)
        outimages.append(newimage)
        # If you find that there's a large amount of ghosting, it may be because one or more of the input
        # images gets misaligned.  Outputting the aligned images may help diagnose that.
        # cv2.imwrite(transFolder + "aligned{}.png".format(i), newimage)

    return outimages