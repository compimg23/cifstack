import numpy as np
import cv2

def findHomography(image_1, image_2):
    # BEGIN NEW CODE FOR findTransformECC()
    warp_mode = cv2.MOTION_AFFINE
    num_iter = 5000
    termination_eps = 1e-10
    warp_matrix = np.eye(2, 3, dtype=np.float32) #3x3 matrix for HOMOGRAPHY, 2x3 for AFFINE
    # END NEW CODE

    print("finding transform")
    # homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)
    # (cc, homography) = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_MAX_ITER, num_iter,termination_eps))
    (cc, homography) = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, termination_eps))

    return homography

def align_images(images):
    outimages = []    #   We assume that image 0 is
    numImages = len(images)
    # image1gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)

    for i in range(0,numImages-1):
    # for i in range(1,numImages): # for comparing first image instead of last.
        print ("Aligning image {}".format(i))
        compareIndex = numImages-1
        hom = findHomography(cv2.cvtColor(images[compareIndex],cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))

        #warpPerspective for MOTION_HOMOGRAPHY mode.
        # newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)

        #warpAffine for MOTION_AFFINE mode.
        newimage = cv2.warpAffine(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        cv2.imwrite('transimage' + str(i) + '.png', newimage)
        outimages.append(newimage)
        # If you find that there's a large amount of ghosting, it may be because one or more of the input
        # images gets misaligned.  Outputting the aligned images may help diagnose that.
        # cv2.imwrite("aligned{}.png".format(i), newimage)

    # add last image that was compared to without transforming it.
    outimages.append(images[numImages])
    return outimages