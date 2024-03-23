#Computational Imaging WS23-24
#Focal Stacking group
#Run 2nd algorithm with: python run.py -a 2 -i InputFolder -o output.jpg
#Print lines from run.py should begin with a *

import os
from glob import glob
from argparse import ArgumentParser

import numpy as np
import cv2

import dummyalg1, dummyalg2, alg3, alg4, alg5, alg6, alg7, alg8, alg9, alg10, alg11
import alignment, alignment_midterm, alignment_alt

from PIL import Image
import PIL

import time

def main():
    _parser = ArgumentParser(prog="Tool to focus stack a list of images.")
    _parser.add_argument(
        "-a", "--alg", help="Number of algorithm to use.", required=True, type=str,
    )
    _parser.add_argument(
        "-v", "--alignver", help="Use 'old' or 'new' or 'alt' alignment version.", required=False, type=str,
    )
    _parser.add_argument(
        "-c", "--compareimg", help="Use 'last' or 'first' image for alignment comparison.", required=False, type=str,
    )
    _parser.add_argument(
        "-i",
        "--input",
        help="Directory of images to focus stack",
        required=False,
        type=str,
    )
    _parser.add_argument(
        "-o", "--output", help="Name of output image including ending.", required=True, type=str,
    )
    _parser.add_argument(
        "-g",
        "--gaussian",
        help="Size of gaussian blur kernel.",
        default=5,
        required=False,
        type=int,
    )
    _parser.add_argument(
        "-l",
        "--laplacian",
        help="Size of laplacian gradient kernel.",
        default=5,
        required=False,
        type=int,
    )
    
    args = _parser.parse_args()
    
    image_files = sum(
        [glob(f"{args.input}/*.{ext}") for ext in ["jpg", "png", "jpeg", "JPG"]], []
    )
    num_files = len(image_files)
    print("*Number of image files found:", num_files)
    if (len(image_files) < 1):
        print("No image files found in input folder! Canceling operation.")
        exit()

    if args.alignver == 'old' or args.alignver == 'Old':
        # use old (midterm) alignment
        if args.compareimg == 'first':
            alignMethod = alignment_midterm.align_images_compare_first
        else:
            alignMethod = alignment_midterm.align_images_compare_last
    else:
        if args.alignver == 'alt' or args.alignver == 'Alt':
            # use alt (better for cpu)) alignment
            if args.compareimg == 'first':
                alignMethod = alignment_alt.align_images_compare_first
            else:
                alignMethod = alignment_alt.align_images_compare_last
        else:
            # use new (better in *most* cases) alignment
            if args.compareimg == 'first':
                alignMethod = alignment.align_images_compare_first
            else:
                alignMethod = alignment.align_images_compare_last

    match args.alg:
        case '2':
            print("*Activating dummy algorithm 2.")
            alg = dummyalg2.DummyAlgorithm2()
        case '3':
            print("*Activating algorithm 3.")
            alg = alg3.Alg3WaveletGray()
        case '4':
            print("*Activating algorithm 4.")
            alg = alg4.Alg4MergeTest()
        case '5':
            print("*Activating algorithm 5.")
            alg = alg5.Alg5MergeTest()
        case '6':
            print("*Activating algorithm 6.")
            alg = alg6.Alg6MergeTest()
        case '7':
            print("*Activating algorithm 7.")
            alg = alg7.Alg7WaveletLaplace()
        case '8':
            print("*Activating algorithm 8.")
            alg = alg8.Alg8WaveletDeep()
        case '9':
            print("*Activating algorithm 9.")
            alg = alg9.Alg9Waveletr2dDecompL1()
        case '10':
            print("*Activating algorithm 10.")
            alg = alg10.Alg10Waveletr2dDecompL2()
        case '11':
            print("*Activating algorithm 11.")
            alg = alg11.Alg11Waveletr2dDecompComplex()
        case _:
            print("*Activating dummy algorithm 1 (default).")
            alg = dummyalg1.DummyAlgorithm1()

    start_time = time.time()
    
    start_time_of_day = time.localtime()

    resultImg = alg.startAlg(image_files, alignMethod)

    total_seconds = time.time() - start_time
    end_time_of_day = time.localtime()

    print("Algorithm time to run: --- %s seconds ---" % total_seconds)

    print("Algorithm was started at", start_time_of_day, "and ended at time", end_time_of_day)

    print("*Algorithm finished running")
    
    if os.path.exists(args.output):
        print(f"*Image {args.output} exists already. Canceling write operation.")
    else:
        print(f"*Writing image {args.output}")
        # im1 = Image.fromarray((resultImg * 55).astype(np.uint8))
        cv2.imwrite('OutputFolder/' + args.output, resultImg)

        #Old output with switched R-B channels.
        # im1 = Image.fromarray((resultImg).astype(np.uint8))
        # # im1 = im1.convert('RGB')
        # im1.save('OutputFolder/' + args.output)


if __name__ == "__main__":
    main()
