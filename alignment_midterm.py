import numpy as np
import cv2

transFolder = 'TransFolder/'

def findHomography(image_1, image_2):
    warp_mode = cv2.MOTION_AFFINE
    num_iter = 1000 #50 #1000 <- original midterm value
    termination_eps = 1e-7 #0.001 #1e-7 <- original midterm value
    warp_matrix = np.eye(2, 3, dtype=np.float32) #3x3 matrix for HOMOGRAPHY, 2x3 for AFFINE

    print("finding transform")
    (cc, homography) = cv2.findTransformECC(image_1, image_2, warp_matrix, warp_mode, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, termination_eps))

    return homography

def align_images_compare_last(images):
    print("Old align_images_compare_last()")
    outimages = []    #   We assume that image 0 is
    numImages = len(images)

    for i in range(0,numImages-1):
        print ("Aligning image {}".format(i))
        compareIndex = numImages-1
        hom = findHomography(cv2.cvtColor(images[compareIndex],cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))

        #warpAffine() for MOTION_AFFINE mode. warpPerspective() for MOTION_HOMOGRAPHY mode.
        newimage = cv2.warpAffine(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        cv2.imwrite(transFolder + 'transimage' + str(i) + '.png', newimage)
        outimages.append(newimage)

        # If you find that there's a large amount of ghosting, it may be because one or more of the input
        # images gets misaligned.  Outputting the aligned images may help diagnose that.
        # cv2.imwrite(transFolder + "aligned{}.png".format(i), newimage)

    # add last image that was compared to without transforming it.
    outimages.append(images[numImages-1])
    return outimages

def align_images_compare_first(images):
    print("Old align_images_compare_first()")
    outimages = []    #   We assume that image 0 is
    numImages = len(images)

    # add last image that was compared to without transforming it.
    outimages.append(images[0])

    for i in range(1,numImages):
        print ("Aligning image {}".format(i))
        compareIndex = 0
        hom = findHomography(cv2.cvtColor(images[compareIndex],cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY))

        #warpAffine() for MOTION_AFFINE mode. warpPerspective() for MOTION_HOMOGRAPHY mode.
        newimage = cv2.warpAffine(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        cv2.imwrite(transFolder + 'transimage' + str(i) + '.png', newimage)
        outimages.append(newimage)
        # If you find that there's a large amount of ghosting, it may be because one or more of the input
        # images gets misaligned.  Outputting the aligned images may help diagnose that.
        # cv2.imwrite(transFolder + "aligned{}.png".format(i), newimage)

    return outimages