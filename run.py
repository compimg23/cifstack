#Computational Imaging WS23-24
#Focal Stacking group
#Run 2nd algorithm with: python run.py -a 2 -i InputFolder -o output.jpg
#Print lines from run.py should begin with a *

import os
from glob import glob
from argparse import ArgumentParser

import cv2

import dummyalg1, dummyalg2, alg3

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
    
    # args = _parser.parse_args()
    
    image_files = sum(
        [glob(f"{'InputFolder'}/*.{ext}") for ext in ["jpg", "png", "jpeg", "JPG"]], []
    )
    # image_files = sum(
    #     [glob(f"{args.input}/*.{ext}") for ext in ["jpg", "png", "jpeg", "JPG"]], []
    # )
    # num_files = len(image_files)
    # print("*Number of image files found:", num_files)
    # if (len(image_files) < 1):
    #     print("No image files found in input folder! Canceling operation.")
    #     exit()

    # match args.alg:
    #     case '2':
    #         print("*Activating dummy algorithm 2.")
    #         alg = dummyalg2.DummyAlgorithm2()
    #     case '3':
    #         print("*Activating algorithm 3.")
    #         alg = alg3.Alg3WaveletTest()
    #     case _:
    #         print("*Activating dummy algorithm 1 (default).")
    #         alg = dummyalg1.DummyAlgorithm1()

    print("*Activating algorithm 3.")
    alg = alg3.Alg3WaveletTest()

    resultImg = alg.startAlg(image_files)
    print("*Algorithm finished running")
    
    # if os.path.exists(args.output):
    #     print(f"*Image {args.output} exists already. Canceling write operation.")
    # else:
    #     print(f"*Writing image {args.output}")
    #     cv2.imwrite(args.output, resultImg)


if __name__ == "__main__":
    main()
