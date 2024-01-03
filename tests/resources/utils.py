"""Utility methods for unit tests."""
import PIL
from PIL.Image import Image
from warnings import warn


def images_are_identical(image_1: Image, image_2: Image):
    if image_1.mode != image_2.mode:
        warn(f'Images do not have the same mode. Found {image_1.mode} and {image_2.mode}.')
        return False
    elif image_1.mode in ['1', 'L']:
        return PIL.ImageChops.difference(image_1, image_2).getextrema() == ((0, 0))
    elif image_1.mode == 'LA':
        return PIL.ImageChops.difference(image_1, image_2).getextrema() == ((0, 0), (0, 0))
    elif image_1.mode == 'RGB':
        return PIL.ImageChops.difference(image_1, image_2).getextrema() == ((0, 0), (0, 0), (0, 0))
    elif image_1.mode == 'RGBA':
        return PIL.ImageChops.difference(image_1, image_2).getextrema() == ((0, 0), (0, 0), (0, 0), (0, 0))
    else:
        raise NotImplementedError(f'Unable to compare images of type {image_1.mode}.')
