# cifstack
Focal stacking project

Based loosely on https://github.com/momonala/focus-stack, https://github.com/cmcguinness/focusstack, https://github.com/sjawhar/focus-stacking and https://github.com/PetteriAimonen/focus-stack.

The code can be run with run.py. 
From there the main algorythms for real Haar wavelet (alg3) Laplacian pyramid (alg5), simple Laplacian (alg6), real Daubechies wavelet (alg10) and complex Q-shift wavelet (alg11) can be selectet with the parameter -a.
Alignment method is depending on the image set used, usually -v new is sufficient.
Parameter -c last/first defines the reference image within the dataset.
With -i the Inputfolder is chosen and -o is the name of the output image.

An execution could look like this: python run.py -a alg11 -v new -c last -i InputFolder -o output.png
