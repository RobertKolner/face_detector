#!/usr/bin/env python
from scipy import ndimage
from skimage import data, io, filters, feature
import numpy as np
import sys


def show(image):
    image -= min(0.0, np.min(image))
    image /= max(1.0, np.max(image))
    io.imshow(image)
    io.show()


if __name__ == "__main__":
    image = ndimage.imread('nikolai.png', True) / 255.0
    filtered = filters.gaussian_filter(image, 1)

    #edges = feature.corner_fast(filtered)
    #out_image = edges * 0.1 + filtered
    out_image = filters.canny(filtered)

    show(out_image)
