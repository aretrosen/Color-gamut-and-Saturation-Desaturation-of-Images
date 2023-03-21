# ADIP & CV (CS60052) Assignment 3

## Installation:
Follow this link to download [Miniconda](https://docs.conda.io/en/latest/miniconda.html "Miniconda Page"). Then, to install required packages, and create environment, run
```bash
conda env create -f environment.yml
```

inside the project directory. Activate `conda` environment by running

```bash
conda activate adip-assignment-3
```

Assuming the OS you run is a Linux distro, run the `run.sh` script for
effortless viewing of the results. Each `"slide"` will show after 4 seconds. If
the OS is **NOT** Linux based, then you can run each of the lines in the
`run.sh` file successively.

All functions are heavily documented, and the operations are all
automatic. However, parameters outside the default ones may be equally good and
is worth trying. Also, changing some parameters have interesting effects. There
are many parameters, and I recommend tweaking some to get new and interesting
results.

## Experimental Results:
Read function docstrings for a overview of all the functions, and what they
do. There are many parameters involved, but I have tried to set the parameters
in a way such that they are mostly automatic and require as little human
intervention as possible. However, note, that fixed_saturation is set in both
image manually, as I think that percent_saturation is a better way to increase
saturation, so that is kept default, and to get maximal saturation as per the
given problem, fixed_saturation is set to 1 within proper saturation limits.
Image-wise results are as follows:

* **Tiger.jpg** - Save for the fixed_saturation value set, everything is
automatic.
* **Deer.jpg** - It **can** run completely automatic, but fixed_saturation value
is set to 1 for better compliance with the problem given. Also, the
saturation_lims (saturation limits) are lowered, since this image as a whole is
a bit "dull", and not saturated at all.

Standard algorithms are used for most of the assignment. The algorithm used for
the desaturation using center of gravity is as mentioned in the paper "A New
Algorithm Based on Saturation and Desaturation n the xy Chromaticity Diagram for
Enhancement and re-rendition of Color Images" by J. Mukherjee et al.

## Usage:
```bash
# Creating and activating conda environment
conda env create -f environment.yml
conda activate adip-assignment-3

# See all options; accessing the help menu
python assignment.py --help

# Without any option, it runs with the data/Tiger.jpg image, if present
# NOTE: Here it uses the percent_saturation to be at 100 by default
python assignment.py

# Set time interval in seconds between image "slides", here it is 2.5s
python assignment.py --interval 2.5

# Runs with another image
python assignment.py --image myimage.jpg

# Runs the program with a fixed saturation. However saturation below 15% is
# ignored.
python assignment.py --image myimage.jpg --fixed_saturation 1

# NOTE: Providing fixed_saturation and percent_saturation (or desaturation in
# both places) is counter-intuitive. If provided, fixed values take precedence.
# Here is an example with percent desaturation increased from default 50 to 90.
python assignment.py --percent_desaturation 90

# Runs with shifted whitepoint and different k. Default k is 0.5, whitepoint
# is D65. NOTE: It is better to run with different k and experiment with it
# for better results.
python assignment.py -k 0.8 --whitepoint 0.2 0.5

# Example with different saturation limits. Saturation limits are used to
# prevent color bleeding as much as possible. Experiment with it for better
# results.
python assignment.py --image myimg.jpg --fixed_saturation 1 --saturation_lims 10 100
```


### Author Information:
	• Name : Aritra Sen.
	• Roll no. : 19ME10101.
	• Department : Mechanical Engineering.
	• Subject : Advanced Digital Image Processing & Computer Vision (CS60052).
