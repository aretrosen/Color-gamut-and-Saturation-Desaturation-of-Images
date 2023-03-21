import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def RGB2xyY(img: np.ndarray) -> np.ndarray:
    """Converts an RGB Image to XYZ format and then to xyY format.

    Args:
        img (np.ndarray): A numpy array representing an image in RGB format.

    Returns:
        An array representing the image in xyY format.
    """
    img_f: np.ndarray = np.clip(img.astype(np.float32) / 255, 0, 1)
    img_XYZ: np.ndarray = cv2.cvtColor(img_f, cv2.COLOR_RGB2XYZ)
    sum_XYZ: np.ndarray = np.sum(img_XYZ, axis=-1)
    sum_XYZ[sum_XYZ == 0] = 1e-25
    img_xyY: np.ndarray = np.zeros_like(img_XYZ)
    img_xyY[..., :2] = img_XYZ[..., :2] / sum_XYZ[..., np.newaxis]
    img_xyY[..., 2] = img_XYZ[..., 1]
    return img_xyY


def xyY2XYZ(xyY: np.ndarray) -> np.ndarray:
    """Converts from xyY format to XYZ format.

    Args:
        xyY (np.ndarray): A numpy array representing an image in xyY format.

    Returns:
        An array representing the image in XYZ format.
    """
    XYZ: np.ndarray = np.zeros_like(xyY)
    XYZ[..., 1] = xyY[..., 2]
    xyY[..., 1][xyY[..., 1] == 0] = 1e-25
    XYZ[..., 0] = xyY[..., 0] * xyY[..., 2] / xyY[..., 1]
    XYZ[..., 2] = (1 - xyY[..., 0] - xyY[..., 1]) * xyY[..., 2] / xyY[..., 1]
    return XYZ.astype(np.float32)


def saturation(
    img: np.ndarray,
    fixed_sat: float | None = None,
    percent_increase: float = 0,
    lims: np.ndarray = np.array([0.15, 1]),
    auto_sat: bool = False,
):
    """Performs operations on the Saturation values of an Image.

    Performs an auto saturation operation if specified. Percent saturation
    increase or decrease is supported, along with fixed saturation values. In
    case of fixed saturation, there should be some limits so that color bleeding
    for near to white pixel doesn't occur,i.e., within a given circle, values
    will be unaffected.

    Args:
        img (np.ndarray): The image in RGB format.
        fixed_sat (None | float): Fixed Saturation value,if provided.
        percent_increase (float): Percent increase in saturation values. Doesn't
            work if fixed_saturation is provided.
        lims (np.ndarray): Limits within which fixed saturation is to be applied
        auto_sat (bool): An auto increase in saturation value.

    Returns:
        Saturated image.
    """
    img_f: np.ndarray = img.astype(np.float32) / 255
    img_hsv = cv2.cvtColor(img_f, cv2.COLOR_RGB2HSV)
    if auto_sat:
        k = np.amax(img_hsv[..., 1])
        img_hsv[..., 1] /= k if k != 0 else 1
    if fixed_sat is not None:
        filt = (img_hsv[..., 1] >= lims[0]) & (img_hsv[..., 1] <= lims[1])
        img_hsv[..., 1][filt] = float(fixed_sat)
    else:
        img_hsv[..., 1] = img_hsv[..., 1] * (1 + percent_increase / 100)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 1)
    img_sat_f = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_sat: np.ndarray = np.around(img_sat_f * 255).astype(np.uint8)
    return img_sat


def desaturation_cog(
    img: np.ndarray,
    max_sat_img: np.ndarray,
    whitepoint: np.ndarray = np.array([0.3127, 0.3290]),
    k: float = 0.5,
):
    """Desaturation using center of gravity method.

    Desaturation operation using the center of gravity method as described in
    the paper "A New Algorithm Based on Saturation and Desaturation n the xy
    Chromaticity Diagram for Enhancement and re-rendition of Color Images" by J.
    Mukherjee et al. This function uses the method mentioned as-is and no
    further tuning was done.

    Args:
        img (np.ndarray): The Image whose desaturation is to be done.
        max_sat_img (np.ndarray): Maximally saturated Image.
        whitepoint (np.ndarray): The whitepoint (D65 or shifted) of the image.
        k (float): The factor k as mentioned in the paper.

    Returns:
        Desaturated-Saturated Image, as specified in the paper mentioned above.
    """
    img_xyY = RGB2xyY(img)
    max_sat_img_xyY = RGB2xyY(max_sat_img)
    Y_avg = np.mean(img_xyY[..., 2])
    Y_w = k * Y_avg
    d1 = np.abs(Y_w) / whitepoint[1]
    d2 = max_sat_img_xyY[..., 2].copy()
    filt = max_sat_img_xyY[..., 1] != 0
    d2[filt] /= max_sat_img_xyY[..., 1][filt]
    xyY = np.zeros_like(img_xyY)
    deno = d1 + d2
    deno[deno == 0] = 1e-25
    xyY[..., 0] = (whitepoint[0] * d1 + max_sat_img_xyY[..., 0] * d2) / (deno)
    xyY[..., 1] = (np.abs(Y_w) + max_sat_img_xyY[..., 2]) / (deno)
    xyY[..., 2] = Y_w + max_sat_img_xyY[..., 2]
    XYZ = xyY2XYZ(xyY)
    rgb = cv2.cvtColor(XYZ, cv2.COLOR_XYZ2RGB)
    return np.clip(np.around(rgb * 255), 0, 255).astype(np.uint8)


def main():
    """Computes Saturated and Desaturated Images, plots, and display them.

    This function computes all the saturated and desaturated images, and the
    Saturated-desaturated image, and computes the xy Chromaticity plot for each
    such image. It also shows the xy Chromaticity plot, the sRGB gamut and the
    D65 and EE white points at start of the slideshow.

    WARNING: The function is a monolith, partly because it does all the trivial
    operations for which functions are a overkill. Comments an indentation are
    provided to show each section separately, and why they are used.

    Returns:
        Displays the plots and the images. Exits on any key pressed.

    """
    slideshow: list = []
    slideshow.append(("Original Image", img_rgb.copy()))

    ##############
    # Saturation #
    ##############
    sat_img: np.ndarray = saturation(
        img_rgb,
        args.fixed_saturation,
        args.percent_saturation,
        fixed_lims,
        auto_sat=True,
    )
    slideshow.append(("Saturated Image", sat_img.copy()))

    ################
    # Desaturation #
    ################
    desat_img: np.ndarray = saturation(
        img_rgb,
        args.fixed_desaturation,
        -args.percent_desaturation,
        fixed_lims,
        auto_sat=True,
    )
    slideshow.append(("Desaturated Image", desat_img.copy()))

    ###############################
    # Saturated-desaturated Image #
    ###############################
    sat_desat_img: np.ndarray = desaturation_cog(
        img_rgb, sat_img.copy(), whitept, args.k
    )
    slideshow.append(
        (f"Saturated-desaturated Image (k = {args.k})", sat_desat_img.copy())
    )

    ################################################
    # Preparing the "horse-shoe" chromaticity plot #
    ################################################

    graph_points = 256  # Number of points to consider in each axis
    xlim = 0.8  # Extent of x-axis
    ylim = 0.9  # Extent of y-axis

    cie_XYZ = np.loadtxt("data/ciexyz31_1.csv", delimiter=",", dtype=np.float64)
    sumXYZ = np.sum(cie_XYZ[:, 1:], axis=-1)
    sumXYZ[sumXYZ == 0] = 1e-25
    xyz = cie_XYZ[:, 1:] / sumXYZ[..., np.newaxis]
    cie_xy = Polygon(xyz[..., :2], fc="none", ec="k")

    xx_cie, yy_cie = np.meshgrid(
        np.linspace(0, xlim, graph_points + 1), np.linspace(ylim, 0, graph_points + 1)
    )
    xyY_cie = np.dstack((xx_cie, yy_cie, np.ones_like(xx_cie)))
    cie_rgb = np.clip(cv2.cvtColor(xyY2XYZ(xyY_cie), cv2.COLOR_XYZ2RGB), 0, 1)

    #############################################
    # Starting the slideshow of images and plot #
    #############################################
    plt.figure("Colorimetry Assignment")
    _, ax = plt.subplots()

    ##########################################################################
    # First plot: Shows sRGB Gamut, and the D65 and EE (Equal Energy) points #
    ##########################################################################
    ax.set_title("CIE 1931 Colourspace with the sRGB Gamut")
    ax.xaxis.set_ticks_position("bottom")
    ax.add_patch(cie_xy)
    img_ax = ax.imshow(cie_rgb, interpolation="bilinear", extent=(0, xlim, 0, ylim))
    img_ax.set_clip_path(cie_xy)
    ax.plot(
        [0.3333, 0.3127, 0.64, 0.3, 0.15],
        [0.3333, 0.3290, 0.33, 0.6, 0.06],
        "o",
        mfc="none",
        mec="k",
        mew=2,
        ms=7,
    )
    ax.add_patch(
        Polygon(
            np.array([[0.64, 0.33], [0.3, 0.6], [0.15, 0.06]]),
            fc="none",
            ec="k",
        )
    )
    ax.text(0.325, 0.345, "EE", weight="semibold")
    ax.text(0.30, 0.305, "D65", weight="semibold")
    ax.text(0.35, 0.56, "sRGB Gamut", weight="heavy")

    plt.pause(args.interval)

    #############################################################
    # For each image, show them and their xy chromaticity plot. #
    #############################################################
    for title, img in slideshow:
        ax.clear()
        ax.set_title(title)
        ax.matshow(img)
        plt.axis("off")
        plt.pause(args.interval)

        ax.clear()
        ax.set_title(title + " Chromaticity Plot")
        ax.xaxis.set_ticks_position("bottom")
        ax.add_patch(cie_xy)
        img_ax = ax.imshow(cie_rgb, interpolation="bilinear", extent=(0, xlim, 0, ylim))
        img_ax.set_clip_path(cie_xy)
        xyY = RGB2xyY(img)
        xaxis = xyY[..., 0]
        yaxis = xyY[..., 1]
        ax.scatter(xaxis[xaxis != 0], yaxis[yaxis != 0], c="k", s=0.1, marker=".")
        plt.pause(args.interval)

    # Closes all plot on loop exit
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saturation, Desaturation and Desaturation-Saturation of a colored Image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/Tiger.jpg",
        help="path to an image",
    )
    parser.add_argument(
        "--fixed_saturation",
        default=None,
        help="Saturating to a fixed floating point number between 0 to 1, both inclusive",
    )
    parser.add_argument(
        "--fixed_desaturation",
        default=None,
        help="Desaturating to a fixed floating point number between 0 to 1, both inclusive",
    )
    parser.add_argument(
        "--saturation_lims",
        nargs=2,
        metavar=("LOWER_LIMIT", "UPPER_LIMIT"),
        type=float,
        default=[15, 100],
        help="Changes only the saturation values within this range,in case the saturation value is fixed. Limits are in percentage",
    )
    parser.add_argument(
        "--percent_saturation",
        type=float,
        default=100,
        help="Percent increase in saturation of image",
    )
    parser.add_argument(
        "--percent_desaturation",
        type=float,
        default=50,
        help="Percent decrease in saturation of image",
    )
    parser.add_argument(
        "-k",
        type=float,
        default=0.5,
        help="Factor k for desaturation using center of gravity method",
    )
    parser.add_argument(
        "--whitepoint",
        nargs=2,
        metavar=("X", "Y"),
        type=float,
        default=[0.3127, 0.3290],
        help="The white point or type of light source in the image",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=4,
        help="time interval in seconds between each image of slide show",
    )
    args = parser.parse_args()

    # Check image validity
    if not Path(args.image).is_file():
        raise FileNotFoundError
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise Exception("Not a valid image format")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Check fixed_saturation and fixed_desaturation compatibility
    if (
        args.fixed_desaturation is not None
        and args.fixed_saturation is not None
        and args.fixed_saturation <= args.fixed_desaturation
    ):
        raise Exception("Incompatible fixed saturation and desaturation value")

    # Check if fixed saturation / desaturation is within a valid range.
    if args.fixed_saturation is not None and not (
        0 <= float(args.fixed_saturation) <= 1
    ):
        raise Exception("Value of fixed_saturation must be between 0 and 1")
    if args.fixed_desaturation is not None and not (
        0 <= float(args.fixed_desaturation) <= 1
    ):
        raise Exception("Value of fixed_desaturation must be between 0 and 1")

    # Automatically raise exception if type error found
    args.fixed_saturation = (
        float(args.fixed_saturation) if args.fixed_saturation else None
    )
    args.fixed_desaturation = (
        float(args.fixed_desaturation) if args.fixed_desaturation else None
    )

    # Preparing global values
    fixed_lims = np.array(args.saturation_lims) / 100
    whitept = np.array(args.whitepoint)
    plt.rcParams["figure.figsize"] = (9, 9)
    plt.rcParams["figure.autolayout"] = True

    # Call main function to display images. A method is used so that global
    # namespace is not polluted.
    main()
