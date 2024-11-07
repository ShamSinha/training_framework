import numpy as np
import math
from skimage import filters
import random
from random import randint, uniform
from torch import Tensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import scipy
import cv2
import copy
from typing import Callable, List, Dict, Optional, Tuple, Union

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from PIL import ImageFilter, ImageOps

auglevel = 5


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "RotateImage":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, magnitude)
    elif op_name == "saturation":
        img = F.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "InvertImage":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


def call_transform(operation, value, fill=None):
    def transform(inp):
        return _apply_op(inp, operation, value, InterpolationMode.BICUBIC, fill)

    return transform


# SSL based augmentation techniques
def image_in_painting(image):
    x = copy.deepcopy(image)
    img_rows, img_cols = x.shape
    cnt = 2
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 15, img_rows // 15)
        block_noise_size_y = random.randint(img_cols // 15, img_cols // 15)
        # block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        # noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
        ] = (
            np.random.rand(block_noise_size_x, block_noise_size_y) * 1.0
        )
        cnt -= 1
    return x


def GaussianBlur(img):
    """
    Apply Gaussian Blur to the PIL image.
    """
    radius_min = 0.1
    radius_max = 2.0

    return img.filter(
        ImageFilter.GaussianBlur(radius=random.uniform(radius_min, radius_max))
    )


def Solarization(img):
    """
    Apply Solarization to the PIL image.
    """
    return ImageOps.solarize(img)


def image_out_painting(image):
    x = copy.deepcopy(image)
    img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1]) * 1.0
    block_noise_size_x = img_rows - random.randint(
        3 * img_rows // 10, 4 * img_rows // 10
    )
    block_noise_size_y = img_cols - random.randint(
        3 * img_cols // 10, 4 * img_cols // 10
    )
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    x[
        noise_x : noise_x + block_noise_size_x,
        noise_y : noise_y + block_noise_size_y,
    ] = image_temp[
        noise_x : noise_x + block_noise_size_x,
        noise_y : noise_y + block_noise_size_y,
    ]
    cnt = 10
    # more number of times we replace the random image with the orginal image ,
    # if you want less distortion image it a large number

    while cnt > 0:
        block_noise_size_x = img_rows - random.randint(
            3 * img_rows // 10, 4 * img_rows // 10
        )
        block_noise_size_y = img_cols - random.randint(
            3 * img_cols // 10, 4 * img_cols // 10
        )

        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)

        x[
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
        ] = image_temp[
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
        ]
        cnt -= 1
    return x


def local_pixel_shuffling(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 100
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)

        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)

        window = orig_image[
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
        ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, block_noise_size_y))
        image_temp[
            noise_x : noise_x + block_noise_size_x,
            noise_y : noise_y + block_noise_size_y,
        ] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.
    Control points should be a list of lists, or list of tuples
    such as [ [1,1],
              [2,3],
              [4,5], ..[Xn, Yn] ]
     nTimes is the number of time steps, defaults to 1000
     See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)]
    )

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x):
    points = [
        [0, 0],
        [random.random(), random.random()],
        [random.random(), random.random()],
        [1, 1],
    ]
    # xpoints = [p[0] for p in points]
    # ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def convert_to_unit8(resoluion) -> Callable:
    def convert_function(x):
        return (x * resoluion).astype(np.uint8)

    return convert_function


def convolution(image, kernel, average=False):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[
        pad_height : padded_image.shape[0] - pad_height,
        pad_width : padded_image.shape[1] - pad_width,
    ] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(
                kernel * padded_image[row : row + kernel_row, col : col + kernel_col]
            )
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output


def gaussian_Kernel(size, sigma_range, twoDimensional=True):
    """
    Creates a gaussian kernel with given sigma and size, 3rd argument is for choose the kernel as 1d or 2d
    """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    if twoDimensional:
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * math.pi * sigma**2))
            * math.e
            ** (
                (-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2))
                / (2 * sigma**2)
            ),
            (size, size),
        )
    else:
        kernel = np.fromfunction(
            lambda x: math.e ** ((-1 * (x - (size - 1) / 2) ** 2) / (2 * sigma**2)),
            (size,),
        )
    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel_size, sigma):
    kernel = gaussian_Kernel(kernel_size, sigma_range=sigma)
    return convolution(image, kernel, average=True)


##########################


def type_check(im: Union[np.ndarray, dict]) -> np.ndarray:
    """
    Returns numpy array if the input is a dict
    Args:
        im: numpy array or a dict

    Returns:
        numpy array

    """
    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im, dict):
        return im["input"]


def good_return(
    im: Union[np.ndarray, dict], out: np.ndarray
) -> Optional[Union[np.ndarray, dict]]:
    """
    returns 'out' as np.ndarray if the input 'im' is an ndarray
    inserts out into im['input'] if the input 'im' is a dict
    Args:
        im: np.ndarray or dict
        out: np.ndarray

    Returns:
        returns the same type as input

    """
    if isinstance(im, np.ndarray):
        return out
    elif isinstance(im, dict):
        im["input"] = out
        return im
    return None


def scale(arr: np.ndarray) -> np.ndarray:
    """
    scales all values of numpy array to [0,1]
    Args:
        arr: numpy array

    Returns:
        scaled array

    """
    # TODO assertion on type of numpy array ?

    eps = 1e-10
    if arr.dtype in [np.float64, np.float32, np.float16]:
        eps = np.finfo(arr.dtype).eps
    arr = arr - arr.min()
    arr = arr / (arr.max() + eps)
    return arr


def clip(arr: np.ndarray) -> np.ndarray:
    """
    Clips the array between 0 and 1
    Args:
        arr: numpy array

    Returns:
        clipped numpy array

    """
    arr = np.clip(arr, a_max=1.0, a_min=0)
    return arr


def brightness(inp):
    # TODO docstring.
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    minb, maxb = (50 - 1 * auglevel, 50 + 1 * auglevel)
    p = randint(minb, maxb)
    arr = arr + scipy.special.logit(p / 100)
    return good_return(inp, arr)


def contrast(inp):
    # TODO docstring.
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    minc, maxc = 8 - 1 * auglevel, 8 + 1 * auglevel
    p = randint(minc, maxc)
    arr = arr * (p / 10)
    return good_return(inp, arr)


def gamma(inp):
    # TODO docstring.
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    ming, maxg = 10 - 1 * auglevel, 10 + 1 * auglevel
    gamma = randint(ming, maxg) / 10
    arr = np.array(arr * 255, dtype=np.uint8)
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    arr = cv2.LUT(arr, table) / 255
    return good_return(inp, arr)


def tanht(inp):
    # TODO docstring.
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    arr = scale(arr)
    arr = 2 * arr - 1

    a1 = np.random.uniform(-0.5, 0)
    a2 = np.random.uniform(0.2, 0.5)
    # a1 -> -.0.25
    # a2 -> 0.25
    w_1, b_1 = np.linalg.solve(a=[[a1, 1], [a2, 1]], b=[-0.25, 0.25])

    b1 = np.random.uniform(-0.5, 0)
    b2 = np.random.uniform(0, 0.5)
    w_2, b_2 = np.linalg.solve(a=[[b1, 1], [b2, 1]], b=[-0.25, 0.25])

    arr_orig = (np.arctanh(arr) - b_1) / (w_1 + 0.000000000001)
    arr = np.tanh(w_2 * arr_orig + b_2)

    # arr [-1, 1] -> [0, 1]
    arr = (arr + 1) / 2
    return good_return(inp, scale(arr))


def squeeze(
    inp: Union[np.ndarray, dict], th1: float = 0, th2: float = 0
) -> Union[np.ndarray, dict]:
    """
    # To be removed
    Squeezes the values of an image that lie between the two given
    thresholds. the pixel values > th2 and the pixel values < th1 are multiplied
    by constants and the final image is scaled
    Args:
        inp: numpy array
        th1: threshold 1
        th2: threshold 2

    Returns:
        squeezed array
    """
    im = type_check(inp)
    if th1 == 0 and th2 == 0:
        th1 = 0.1 * randint(0, 4)
        th2 = 0.1 * randint(5, 10)
    gtarr = (im >= th2) * im
    ltarr = (im < th1) * im
    nim = (im > th1) * (im < th2) * im
    new_im = scale(1.2 * ltarr + 0.9 * gtarr + nim)
    return good_return(inp, scale(new_im))


def sparse(
    inp: Union[np.ndarray, dict], th1: float = 0, th2: float = 0
) -> Union[np.ndarray, dict]:
    """
    # To be removed
    Opposite of squeeze, stretches out the pixel values that lie between
    the two given thresholds
    Args:
        inp: np.ndarray or dict
        th1: lower threshold
        th2: higher threshold

    Returns:
       stretched out image

    """
    im = type_check(inp)
    if th1 == 0 and th2 == 0:
        th1 = 0.1 * randint(0, 4)
        th2 = 0.1 * randint(5, 10)
    gtarr = (im >= th2) * im
    ltarr = (im < th1) * im
    nim = (im > th1) * (im < th2) * im
    nim = range_scale(nim, 0.9 * th1, 1.2 * th2)
    new_im = scale(0.9 * ltarr + 1.2 * gtarr + nim)
    return good_return(inp, scale(new_im))


def range_scale(im: np.ndarray, th1: float = 0, th2: float = 0) -> np.ndarray:
    """
    # To be removed
    Scale the image between the two given thresholds
    Args:
        im: np.ndarray or dict
        th1: lower threshold
        th2: higher threshold

    Returns:
        scaled image

    """
    non_zero_min: int = ((im == 0) * 1 + im).min()
    dif = th2 - th1
    im2 = im - non_zero_min
    im2 = (im2 > 0) * im2
    im2 = im2 * (dif / (im2.max() + 0.00001))
    im2 = im2 + th1
    im2 = (im2 > th1) * im2
    return im2


def get_borders(im: np.ndarray, th: float) -> Tuple[int, int, int, int]:
    """
    Returns borders of an image by thresholding each strip. Used in rmblack function
    #TODO vectorize the operations, currently this takes too much time
    Args:
        im: numpy array
        th: threshold

    Returns:
        left, right, up, down borders

    """
    x = im.shape[0]
    y = im.shape[1]
    mn = im.mean()
    bl, br, bu, bd = (0, x, 0, y)
    for i in range(0, x):
        strip = im[i : i + 1, :]
        if strip.std() > th or strip.mean() > mn / 2:
            bl = i
            break
    for i in range(x, 0, -1):
        strip = im[i - 1 : i, :]
        if strip.std() > th or strip.mean() > mn / 2:
            br = i
            break
    for i in range(0, y):
        strip = np.transpose(im[:, i : i + 1], (1, 0))
        if strip.std() > th or strip.mean() > mn / 2:
            bu = i
            break
    for i in range(y, 0, -1):
        strip = np.transpose(im[:, i - 1 : i], (1, 0))
        if strip.std() > th or strip.mean() > mn / 2:
            bd = i
            break
    return bl, br, bu, bd


def randomcrop(size=224, h=0, w=0):
    # TODO docstring
    def rc(inp):
        im = type_check(inp)
        new_im = im[h : h + size, w : w + size]
        return good_return(inp, scale(new_im))

    return rc


def centercrop(size=224):
    # TODO docstring
    def cc(inp):
        im = type_check(inp)
        h, w = im.shape[0], im.shape[1]
        new_im = im[
            int((h - size) / 2) : int((h + size) / 2),
            int((w - size) / 2) : int((w + size) / 2),
        ]
        return good_return(inp, scale(new_im))

    return cc


def rmblack(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Removes the black borders using strip based thresholding.
    Uses threshold as 0.2
    #TODO add threshold as argument
    Args:
        inp: numpy array or a dict

    Returns:
        array/ dict after removing the black borders

    """
    im = type_check(inp)
    # RM Black fails if image mean is 0 or standard deviation is close to 0
    # Removing those 2 conditions
    # im_mean = np.mean(im)
    # im_std = np.std(im)
    # if im_mean > 0 and im_std / im_mean > 1:
    #     imarr = th_lower(im, 10)
    #     bds = get_borders(imarr, 0.2)  # default threshold
    #     imarr = im[bds[0] : bds[1], bds[2] : bds[3]]
    # else:
    #     imarr = im
    imarr = th_lower(im, 10)
    bds = get_borders(imarr, 0.2)  # default threshold
    imarr = im[bds[0] : bds[1], bds[2] : bds[3]]
    return good_return(inp, scale(imarr))


def th_lower(inp, p=-1):
    # TODO docstring.
    arr = type_check(inp)
    if p == -1:
        p = randint(0, 20)
    ll = np.percentile(arr, p)
    narr = arr - ll
    narr = (narr > 0) * narr
    narr = scale(narr)
    ul = np.percentile(narr, 99)
    img = (narr > ul) * ul
    iml = (narr < ul) * narr
    farr = scale(img + iml)
    return good_return(inp, farr)


def add_noise(inp: Union[np.ndarray, dict], p: float = -1) -> Union[np.ndarray, dict]:
    """
    Add gaussian noise to the image
    Args:
        inp: numpy array or a dict
        p: argument to the normal distribution used to generate noise

    Returns:
        noisy image
    """
    arr = type_check(inp)
    if p == -1:
        p = 0.01 * randint(2, 8)
    noise = np.random.normal(0, p, arr.shape)
    new_arr = arr + noise
    return good_return(inp, scale(new_arr))


def smooth_noise(
    inp: Union[np.ndarray, dict], p: float = -1
) -> Union[np.ndarray, dict]:
    """
    Add smoothed gaussian noise to the image
    Args:
        inp: numpy array or a dict
        p: argument to the normal distribution used to generate noise

    Returns:
        noisy image

    """
    arr = type_check(inp)
    if p == -1:
        p = 0.01 * randint(2, 8)
    noise = np.random.normal(0, p, arr.shape)
    noise = filters.gaussian(noise, 0.6)
    new_arr = arr + noise
    return good_return(inp, scale(new_arr))


def smooth(inp: Union[np.ndarray, dict], p: float = -1) -> Union[np.ndarray, dict]:
    """
    Apply gaussian filter on the image to smooth it
    Args:
        inp: numpy array or a dict
        p: Strength of the gaussian filter to be used

    Returns:
        smoothed out image
    """
    arr = type_check(inp)
    if p == -1:
        p = randint(5, 12) * 0.1
    new_arr = filters.gaussian(arr, p)
    return good_return(inp, scale(new_arr))


def mean_clip(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    thresholds the array by it's mean and returns a scaled (0,1) array
    Args:
        inp: numpy array or a dict

    Returns:
        Array thresholded by it's mean and scaled
    """
    arr = type_check(inp)
    mn = arr.mean()
    new_arr = (arr > mn) * 1
    return good_return(inp, scale(new_arr))


def lnorm(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Local normalization, normalize the input array by its
    mean and standard deviation
    Args:
        inp: np.ndarray or dict

    Returns:
        normalized image

    """
    arr = type_check(inp)
    arr = arr - arr.mean()
    arr = arr / (arr.std() + 0.000001)
    return good_return(inp, arr)


def gaborr(inp: Union[np.ndarray, dict], fq: float = -1) -> Union[np.ndarray, dict]:
    """
    Apply gabor filter on the image
    Args:
        inp: np.ndarray or dict
        fq: frequency of gabor filter to be used
    Returns:
        output image from the gabor filter

    """
    arr = type_check(inp)
    if fq == -1:
        fq = 0.1 * randint(4, 7)
    narr, narr_i = filters.gabor(arr, frequency=fq)
    return good_return(inp, scale(narr))


def resize_hard(size: int = 224) -> Callable:
    """
    resizes to a size and scales it #TODO is this redundant ?
    Args:
        size: size to which input should be resized to

    Returns:
        a function which resizes to size and scales the array
    """

    def rz(inp):
        im = type_check(inp)
        im = scale(im)
        # new_im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        new_im = resize(size, size)(im)
        return good_return(inp, scale(new_im))

    return rz


def resize(height: int = 224, width: int = 224) -> Callable:
    """[summary]

    Args:
        height (int, optional): Image height to be resized to. Defaults to 224.
        width (int, optional): Image width to be resized to. Defaults to 224.

    Returns:
        Callable: function to do resizing
    """

    def rz(inp):
        im = type_check(inp)
        new_im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        return good_return(inp, new_im)

    return rz


def resize_hard_sz(sz0, sz1):
    def rz(inp):
        if isinstance(inp, np.ndarray):
            im = inp
            im = scale(im)
            new_im = cv2.resize(im, (sz1, sz0))
            return new_im
        elif isinstance(inp, dict):
            im = {x: scale(inp[x]) for x in inp}
            new_im = {x: cv2.resize(im[x], (sz1, sz0)) for x in im}
            return new_im

    return rz


def resize_hard_int(size=224):
    def rz(inp):
        im = type_check(inp)
        new_im = cv2.resize(im, (size, size))
        return good_return(inp, new_im)

    return rz


def sub_smooth(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Subtract smoothed image from the image
    sharpens the edges in the image
    Args:
        inp: np.ndarray or dict

    Returns:
        image subtracted by its smoothed copy

    """
    im = type_check(inp)
    im = scale(im)
    nim = filters.gaussian(im, 8)
    nim = im - nim
    nim = scale(nim)
    return good_return(inp, nim)


def scale_unity(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Scale the image between -1 and 1
    Args:
        inp: np.ndarray or dict

    Returns:
        scaled image

    """
    im = type_check(inp)
    im = scale(im)
    im = (im - 0.5) / 0.5
    return good_return(inp, im)


def smooth_norm(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Helps extract smooth and sharp features in an image
    iteratively add and subtract smoothed copies of image
    Args:
        inp: np.ndarray or dict

    Returns:
        transformed image

    """
    im = type_check(inp)
    blur1 = filters.gaussian(im, 7)
    blur2 = filters.gaussian(blur1, 7)
    blur3 = filters.gaussian(blur2, 7)
    im = im - blur1 + blur2 - blur3
    return good_return(inp, scale(im))


def seg_randomsizedcrop(output_shape, frac_range=(0.5, 1)):
    def rzcrop(inp_dict):
        ims = [inp_dict[x] for x in inp_dict]
        shp = ims[0].shape
        rat = uniform(frac_range[0], frac_range[1])
        crop_dim = int(np.sqrt(ims[0].size * rat))
        stx, sty = randint(0, shp[0] - crop_dim - 1), randint(0, shp[1] - crop_dim - 1)
        inp_dict = {
            k: resize_hard(output_shape)(
                inp_dict[k][stx : stx + crop_dim, sty : sty + crop_dim]
            )
            for k in inp_dict
        }
        return inp_dict

    return rzcrop


def seg_randomsizedcrop_int(output_shape, frac_range=(0.5, 1)):
    # TODO docstring
    def rzcrop(inp_dict):
        ims = [inp_dict[x] for x in inp_dict]
        shp = ims[0].shape
        rat = uniform(frac_range[0], frac_range[1])
        crop_dim = int(np.sqrt(ims[0].size * rat))
        stx, sty = randint(0, shp[0] - crop_dim - 1), randint(0, shp[1] - crop_dim - 1)
        inp_dict = {
            k: resize_hard_int(output_shape)(
                inp_dict[k][stx : stx + crop_dim, sty : sty + crop_dim]
            )
            for k in inp_dict
        }
        return inp_dict

    return rzcrop


def seg_randomcrop(output_shape):
    # TODO docstring
    def rc(inp_dict):
        h, w = output_shape
        ims = [inp_dict[x] for x in inp_dict]
        im_h, im_w = ims[0].shape
        st_h, st_w = randint(0, im_h - h), randint(0, im_w - w)
        inp_dict = {k: inp_dict[k][st_h : st_h + h, st_w : st_w + w] for k in inp_dict}
        return inp_dict

    return rc


class NDTransform(object):
    """Base class for all numpy based transforms.

    This class achieves the following:

    * Abstract the transform into
        * Getting parameters to apply which is only run only once per __call__.
        * Applying transform given parameters
    * Check arguments passed to a transforms for consistency

    Abstraction is especially useful when there is randomness involved with the
    transform. You don't want to have different transforms applied to different
    members of a data point.
    """

    def _argcheck(self, data):
        """Check data for arguments."""

        if isinstance(data, np.ndarray):
            assert data.ndim in {
                2,
                3,
            }, "Image should be a ndarray of shape H x W x C or H X W."
            if data.ndim == 3:
                assert (
                    data.shape[2] < data.shape[0]
                ), "Is your color axis the last? Roll axes using np.rollaxis."

            return data.shape[:2]
        elif isinstance(data, dict):
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    assert isinstance(k, str)

            shapes = {
                k: self._argcheck(img)
                for k, img in data.items()
                if isinstance(img, np.ndarray)
            }
            assert (
                len(set(shapes.values())) == 1
            ), "All member images must have same size. Instead got: {}".format(shapes)
            return set(shapes.values()).pop()
        else:
            raise TypeError("ndarray or dict of ndarray can only be passed")

    def _get_params(self, h, w, seed=None):
        """Get parameters of the transform to be applied for all member images.

        Implement this function if there are parameters to your transform which
        depend on the image size. Need not implement it if there are no such
        parameters.

        Parameters
        ----------
        h: int
            Height of the image. i.e, img.shape[0].
        w: int
            Width of the image. i.e, img.shape[1].

        Returns
        -------
        params: dict
            Parameters of the transform in a dict with string keys.
            e.g. {'angle': 30}
        """
        return {}

    def _transform(self, img, is_label, **kwargs):
        """Apply the transform on an image.

        Use the parameters returned by _get_params and apply the transform on
        img. Be wary if the image is label or not.

        Parameters
        ----------
        img: ndarray
            Image to be transformed. Can be a color (H X W X C) or
            gray (H X W)image.
        is_label: bool
            True if image is to be considered as label, else False.
        **kwargs
            kwargs will be the dict returned by get_params

        Return
        ------
        img_transformed: ndarray
            Transformed image.
        """
        raise NotImplementedError

    def __call__(self, data, seed=None):
        """
        Parameters
        ----------
        data: dict or ndarray
            Image ndarray or a dict of images. All ndarrays in the dict are
            considered as images and should be of same size. If key for a
            image in dict has string `target` in it somewhere, it is
            considered as a target segmentation map.
        """
        h, w = self._argcheck(data)
        params = self._get_params(h, w, seed=seed)

        if isinstance(data, np.ndarray):
            return self._transform(data, is_label=False, **params)
        else:
            data = data.copy()
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    if isinstance(k, str) and "target" in k:
                        is_label = True
                    else:
                        is_label = False

                    data[k] = self._transform(img.copy(), is_label, **params)
            return data


class ToTensor(NDTransform):
    """Convert ndarrays to tensors.

    Following are taken care of when converting to tensors:

    * Axes are swapped so that color axis is in front of rows and columns
    * A color axis is added in case of gray images
    * Target images are left alone and are directly converted
    * Label images is set to LongTensor by default as expected by torch's loss
      functions.

    Parameters
    ----------
    dtype: torch dtype
        If you want to convert all tensors to cuda, you can directly
        set dtype=torch.cuda.FloatTensor. This is for non label images
    dtype_label: torch dtype
        Same as above but for label images.
    """

    import torch

    def _transform(self, img, is_label):
        img = np.ascontiguousarray(img)
        if not is_label:
            # put it from HWC to CHW format
            if img.ndim == 3:
                img = np.rollaxis(img, 2, 0)
            elif img.ndim == 2:
                img = img.reshape((1,) + img.shape)
        else:
            if img.ndim == 3:  # making transforms work for multi mask models
                img = np.rollaxis(img, 2, 0)

        img = self.torch.from_numpy(img)

        if is_label:
            return img.long()
        else:
            return img.float()
