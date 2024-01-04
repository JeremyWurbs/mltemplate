"""Unit test methods for mltemplate.utils.conversions utility module."""
import numpy as np
import pytest
from PIL.Image import Image

from mltemplate.utils import (
    ascii_to_pil,
    bytes_to_pil,
    cv2_to_pil,
    ndarray_to_pil,
    pil_to_ascii,
    pil_to_bytes,
    pil_to_cv2,
    pil_to_ndarray,
    pil_to_tensor,
    tensor_to_pil,
)
from tests import MockAssets, images_are_identical

mocks = MockAssets()


def test_ascii_serialization(image: Image = mocks.image):
    ascii_image = pil_to_ascii(image)
    pil_image = ascii_to_pil(ascii_image)
    assert images_are_identical(image, pil_image)


def test_bytes_serialization(image: Image = mocks.image):
    bytes_image = pil_to_bytes(image)
    pil_image = bytes_to_pil(bytes_image)
    assert images_are_identical(image, pil_image)


def test_tensor_conversion(image: Image = mocks.image):
    tensor_image = pil_to_tensor(image)
    pil_image = tensor_to_pil(tensor_image, min_val=0, max_val=255)
    assert images_are_identical(image, pil_image)


def test_ndarray_conversion(
    image: Image = mocks.image,
    image_rgba: Image = mocks.image_rgba,
    image_float32_ndarray: np.ndarray = mocks.image_float32_ndarray,
    image_ndarray_bgr: np.ndarray = mocks.image_ndarray_bgr,
    image_ndarray_bgra: np.ndarray = mocks.image_ndarray_bgra,
):
    # Test normal 'RGB' usage
    ndarray_image = pil_to_ndarray(image)
    pil_image = ndarray_to_pil(ndarray_image)
    assert images_are_identical(image, pil_image)

    # Test image with alpha channel
    ndarray_image = pil_to_ndarray(image_rgba)
    pil_image = ndarray_to_pil(ndarray_image)
    assert images_are_identical(image_rgba, pil_image)

    # Test np.float32 image
    pil_image = ndarray_to_pil(image_float32_ndarray)
    assert images_are_identical(image, pil_image)

    # Test that an exception is thrown if the ndarray.dtype is not an integer, float or bool
    with pytest.raises(Exception):
        char_image = np.chararray(shape=(100, 100))
        pil_image = ndarray_to_pil(char_image)
        assert isinstance(pil_image, Image)

    # Test BGR image
    pil_image = ndarray_to_pil(image_ndarray_bgr, image_format="BGR")
    assert images_are_identical(image, pil_image)

    # Test BGRA image
    pil_image = ndarray_to_pil(image_ndarray_bgra, image_format="BGR")
    assert images_are_identical(image_rgba, pil_image)

    # Test that an exception is thrown if the input image does not have 1, 3 or 4 channels
    with pytest.raises(Exception):
        two_channel_image = np.zeros(shape=(100, 100, 2), dtype=np.float32)
        pil_image = ndarray_to_pil(two_channel_image)
        assert isinstance(pil_image, Image)

    # Test that an exception is thrown if the input format is not 'RGB' or 'BGR'
    with pytest.raises(Exception):
        hsv_image = np.zeros(shape=(100, 100, 3), dtype=np.float32)
        pil_image = ndarray_to_pil(hsv_image, image_format="HSV")
        assert isinstance(pil_image, Image)


def test_cv2_conversion(image: Image = mocks.image, image_rgba: Image = mocks.image_rgba):
    # Test normal 'RGB' usage
    cv2_image = pil_to_cv2(image)
    pil_image = cv2_to_pil(cv2_image)
    assert images_are_identical(image, pil_image)

    # Test image with alpha channel
    cv2_image = pil_to_cv2(image_rgba)
    pil_image = cv2_to_pil(cv2_image)
    assert images_are_identical(image_rgba, pil_image)
