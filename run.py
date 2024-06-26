#Computational Imaging WS23-24
#Focal Stacking group
#Written by Manuel Gonzales & Lukas Giehl
#Example: Run 6th algorithm with: python run.py -a 6 -i InputFolder -o output.png
#Example: Get help for more options: python run.py --help

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
        "-a", "--alg", help="Number of algorithm to use. 3 = Real (Haar) wavelet. 5 = Laplacian Pyramid. 6 = Laplacian. 10 = Daubechies. 11 = Complex (Q-Shift) wavelet.", required=True, type=str,
    )
    _parser.add_argument(
        "-v", "--alignver", help="Use 'new' or 'alt' or 'old' alignment version. Default 'new'.", required=False, type=str,
    )
    _parser.add_argument(
        "-c", "--compareimg", help="Use 'last' or 'first' image for alignment comparison. Default 'last'.", required=False, type=str,
    )
    _parser.add_argument(
        "-i",
        "--input",
        help="Directory of images to focus stack",
        required=False,
        type=str,
    )
    _parser.add_argument(
        "-o", "--output", help="Name of output image including ending, without folder (saves in ./OutputFolder).", required=True, type=str,
    )
    _parser.add_argument(
        "-l", "--level", help="Decomposition level used for algorithms 10 and 11 only. Complex wavelet requires at least 2.", required=False, type=str,
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

    if args.level == None:
        levelargint = -1
    else:
        levelargint = int(args.level)
    resultImg = alg.startAlg(image_files, alignMethod, levelargint)
    
    if os.path.exists(args.output):
        print(f"*Image {args.output} exists already. Canceling write operation.")
    else:
        print(f"*Writing image {args.output}")
        cv2.imwrite('OutputFolder/' + args.output, resultImg)

    total_seconds = time.time() - start_time
    end_time_of_day = time.localtime()

    print("Algorithm", args.alg, "time to run: --- %s seconds ---" % total_seconds)

    print("Algorithm was started at time object", start_time_of_day, "and ended at time object", end_time_of_day)

    print("*Algorithm finished running")

if __name__ == "__main__":
    main()
