# cifstack
Focal stacking project

Based loosely on https://github.com/momonala/focus-stack, https://github.com/cmcguinness/focusstack, https://github.com/sjawhar/focus-stacking and https://github.com/PetteriAimonen/focus-stack.

For help with the arguments type: `python run.py --help`

The code starts running from run.py. 
From there the main algorithms for real Haar wavelet (algorithm 3) Laplacian pyramid (algorithm 5), simple Laplacian (algorithm 6), real Daubechies wavelet (algorithm 10) and complex Q-shift wavelet (algorithm 11) can be selectet with the parameter -a followed by the algorithm number. For algorithms 10 & 11, you may optionally provide the wavelet decomposition level with the -l parameter.

The alignment method depends on the image set used, usually -v new is sufficient.
The parameter -c last/first defines the reference image within the dataset (preferably the image with features that are in all the other images).
With -i the input folder is chosen and -o is the name of the output image file with either .png or .jpg ending (without path because it is always saved in the folder named 'OutputFolder', which must already exist).

An execution could look like this for Laplacian Pyramid: `python run.py -a 5 -v new -c last -i InputFolder -o output_pyramid.png`
An execution using complex Q-shift wavelets with decomposition level 15 could look like this: `python run.py -a 11 -l 15 -v new -c last -i InputFolder -o output_qshift.png`
