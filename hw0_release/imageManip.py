import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = io.imread(image_path)
    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = 0.5*(image**2)
    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = color.rgb2grey(image)
    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    rgb_dict = {'R':0,'G':1,'B':2}
    out = image.copy()
    out[:,:,rgb_dict[channel]].fill(0)
    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    lab_dict = {'L':0,'A':1,'B':2}
    lab = color.rgb2lab(image)
    out = image.copy().astype(np.float64)
    out.fill(0)
    out[:,:,lab_dict[channel]] = lab[:,:,lab_dict[channel]]
    # rescaling for lab
    #out = (out + [0,128,128])/[100,255,255]
    out = color.lab2rgb(out)
    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    hsv_dict = {'H':0,'S':1,'V':2}
    out = image.copy().astype(np.float64)
    out.fill(1)
    out[:,:,hsv_dict[channel]] = hsv[:,:,hsv_dict[channel]]
    out = color.hsv2rgb(out)
    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    im1 = rgb_decomposition(image1, channel1)
    im2 = rgb_decomposition(image2, channel2)
    out = image1.copy()
    out[:,:out.shape[1]//2,:] = im1[:,:out.shape[1]//2,:] #out.shape[1] gives size of width
    out[:,out.shape[1]//2:,:] = im2[:,out.shape[1]//2:,:]
    return out
