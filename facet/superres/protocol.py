"""Explicit bicubic sampling for auditing DIV2K degradation and SR baselines.

Keys cubic (a=-0.5), half-pixel centres, symmetric boundary extension and
antialiasing when reducing size. No image quantization is performed here.
"""
import numpy as np
from scipy.sparse import csr_matrix


def cubic(x):
    x = np.abs(x)
    return ((1.5*x**3 - 2.5*x**2 + 1) * (x <= 1)
            + (-0.5*x**3 + 2.5*x**2 - 4*x + 2) * ((x > 1) & (x <= 2)))


def resize_weights(length, output_length):
    if min(length, output_length) <= 0:
        raise ValueError('dimensions must be positive')
    scale = output_length / length
    kernel_scale = min(scale, 1.)
    centres = (np.arange(output_length) + .5) / scale - .5
    support = 4 / kernel_scale
    left = np.floor(centres - support/2).astype(int)
    indices = left[:, None] + np.arange(int(np.ceil(support)) + 2)
    weights = kernel_scale * cubic((centres[:, None] - indices) * kernel_scale)
    weights /= weights.sum(axis=1, keepdims=True)
    # Symmetric extension repeats the edge pixel, as MATLAB imresize does.
    reflected = indices % (2*length)
    reflected = np.where(reflected < length, reflected, 2*length-1-reflected)
    rows = np.broadcast_to(np.arange(output_length)[:, None], indices.shape)
    return csr_matrix((weights.ravel(), (rows.ravel(), reflected.ravel())),
                      shape=(output_length, length))


def bicubic_resize(image, shape):
    image = np.asarray(image, dtype=np.float64)
    h, w, channels = image.shape
    H, W = shape
    tmp = (resize_weights(h, H) @ image.reshape(h, -1)).reshape(H, w, channels)
    result = (resize_weights(w, W) @ tmp.transpose(1, 0, 2).reshape(w, -1))
    return result.reshape(W, H, channels).transpose(1, 0, 2)


def modcrop(image, scale=8):
    h, w = image.shape[:2]
    return image[:h-h % scale, :w-w % scale]
