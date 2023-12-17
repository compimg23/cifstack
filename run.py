#Computational Imaging WS23-24
#Focal Stacking group
#Run 2nd algorithm with: python run.py -a 2 -i InputFolder -o output.jpg
#Print lines from run.py should begin with a *

import os
from glob import glob
from argparse import ArgumentParser

import numpy as np
import cv2

import dummyalg1, dummyalg2, alg3, alg4, alg5, alg6

from PIL import Image
import PIL

def main():
    _parser = ArgumentParser(prog="Tool to focus stack a list of images.")
    _parser.add_argument(
        "-a", "--alg", help="Number of algorithm to use.", required=True, type=str,
    )
    _parser.add_argument(
        "-i",
        "--input",
        help="Directory of images to focus stack",
        required=False,
        type=str,
    )
    _parser.add_argument(
        "-o", "--output", help="Name of output image.", required=True, type=str,
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
        case _:
            print("*Activating dummy algorithm 1 (default).")
            alg = dummyalg1.DummyAlgorithm1()

    resultImg = alg.startAlg(image_files)
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
