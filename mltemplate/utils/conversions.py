"""Utility methods relating to image conversion."""
import base64
import io

import cv2
import numpy as np
import PIL
import torch
from PIL.Image import Image
from torchvision.transforms.v2 import functional as F


def pil_to_ascii(image: Image) -> str:
    """Serialize PIL Image to ascii.

    Example::

          import PIL
          from mltemplate.utils import pil_to_ascii, ascii_to_pil

          image = PIL.Image.open('tests/resources/hopper.png')
          ascii_image = pil_to_ascii(image)
          decoded_image = ascii_to_pil(ascii_image)
    """
    imageio = io.BytesIO()
    image.save(imageio, "png")
    bytes_image = base64.b64encode(imageio.getvalue())
    ascii_image = bytes_image.decode("ascii")
    return ascii_image


def ascii_to_pil(ascii_image: str) -> Image:
    """Convert ascii image to PIL Image.

    Example::

          import PIL
          from mltemplate.utils import pil_to_ascii, ascii_to_pil

          image = PIL.Image.open('tests/resources/hopper.png')
          ascii_image = pil_to_ascii(image)
          decoded_image = ascii_to_pil(ascii_image)
    """
    return PIL.Image.open(io.BytesIO(base64.b64decode(ascii_image)))


def pil_to_bytes(image: Image) -> bytes:
    """Serialize PIL Image into io.BytesIO stream.

    Example::

          import PIL
          from mltemplate.utils import pil_to_bytes, bytes_to_pil

          image = PIL.Image.open('tests/resources/hopper.png')
          bytes_image = pil_to_bytes(image)
          decoded_image = bytes_to_pil(ascii_image)
    """
    imageio = io.BytesIO()
    image.save(imageio, "png")
    image_stream = imageio.getvalue()
    return image_stream


def bytes_to_pil(bytes_image: bytes) -> Image:
    """Convert io.BytesIO stream to PIL Image.

    Example::

          import PIL
          from mltemplate.utils import pil_to_bytes, bytes_to_pil

          image = PIL.Image.open('tests/resources/hopper.png')
          bytes_image = pil_to_bytes(image)
          decoded_image = bytes_to_pil(ascii_image)
    """
    return PIL.Image.open(io.BytesIO(bytes_image))


def pil_to_tensor(image: Image) -> torch.Tensor:
    """Convert PIL Image to Torch Tensor.

    Example:
        from PIL import Image
        from mltemplate.utils import pil_to_tensor

        image = Image.open('tests/resources/hopper.png')
        tensor = pil_to_tensor(image)
    """
    return F.pil_to_tensor(image)


def tensor_to_pil(image: torch.Tensor, mode=None, min_val=None, max_val=None) -> Image:
    """Convert Torch Tensor to PIL Image.

    Note that PIL float images must be scaled [0, 1]. It is often the case, however, that torch tensor images may have a
    different range (e.g. zero mean or [-1, 1]). As such, the input torch tensor will automatically be scaled to fit in
    the range [0, 1]. If no min / max value is provided, the output range will be identically 0 / 1, respectively. Else
    you may pass in min / max range values explicitly.

    Args:
        image: The input image.
        mode: The mode of the *output* image. One of {'L', 'RGB', 'RGBA'}.
        min_val: The minimum value of the input image. If None, it will be inferred from the input image.
        max_val: The maximum value of the input image. If None, it will be inferred from the input image.

    Example::
        from PIL import Image
        from mltemplate.utils import pil_to_tensor, tensor_to_pil

        image = Image.open('tests/resources/hopper.png')
        tensor_image = pil_to_tensor(image)
        pil_image = tensor_to_pil(tensor_image)
    """
    min_ = min_val if min_val is not None else torch.min(image)
    max_ = max_val if max_val is not None else torch.max(image)
    return F.to_pil_image((image - min_) / (max_ - min_), mode=mode)


def pil_to_ndarray(image: Image, image_format="RGB") -> np.ndarray:
    """Convert PIL image to numpy ndarray.

    If an alpha channel is present, it will automatically be copied over as well.

    Args:
        image: The input image.
        image_format: Determines the number and order of channels in the *output* image. One of {'L', 'RGB', 'BGR'}.

    Returns:
        An np.ndarray image in the specified format.
    """
    if image.mode in ["LA", "RGBA"]:  # Alpha channel present
        if image_format == "L":
            image = image.convert(mode="L")
            return np.array(image)
        else:
            image = image.convert(mode="RGBA")
            if image_format == "RGB":
                return np.array(image)
            elif image_format == "BGR":
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

    else:  # No alpha channel
        if image_format == "L":
            image = image.convert(mode="L")
            return np.array(image)
        else:
            image = image.convert(mode="RGB")
            if image_format == "RGB":
                return np.array(image)
            elif image_format == "BGR":
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def ndarray_to_pil(image: np.ndarray, image_format: str = "RGB"):
    """Convert numpy ndarray to PIL image.

    The input image can either be a float array with values in the range [0, 1], an int array with values in the
    range [0, 255], or a bool array.

    Args:
        image: The input image. It should be a numpy array with 1, 3 or 4 channels.
        image_format: The format of the *input* image. One of {'RGB', 'BGR'}

    Returns:
        A PIL image.

    Example::

        from matplotlib import image, pyplot as plt
        from mltemplate.utils import ndarray_to_pil

        ndarray_image = image.imread('tests/resources/hopper.png')
        pil_image = ndarray_to_pil(ndarray_image, image_format='RGB')

        plt.imshow(ndarray_image)
        plt.show()
        pil_image.show()
    """
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
    elif not np.issubdtype(image.dtype, np.integer) and image.dtype != bool:
        raise AssertionError(f"Unknown image dtype {image.dtype}. Expected one of bool, np.floating or np.integer.")

    num_channels = image.shape[-1] if image.ndim == 3 else 1
    if num_channels not in [1, 3, 4]:
        raise AssertionError(
            f"Unknown image format with {num_channels} number of channels. Expected an image with 1, 3 or 4."
        )
    elif num_channels == 1 or image_format == "RGB" or image.dtype == bool:
        return PIL.Image.fromarray(image)
    elif image_format == "BGR":
        if num_channels == 3:
            return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif num_channels == 4:
            return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    raise AssertionError(f'Unknown image format "{image_format}". Expected one of "RGB" or "BGR".')


def pil_to_cv2(image: Image) -> np.ndarray:
    """Convert PIL image to cv2 image.

    Note that, in addition to cv2 images being numpy arrays, PIL Images follow RGB format while cv2 images follow BGR
    format.

    Args:
        image: The input image.

    Returns:
        An np.ndarray image in 'BGR' (cv2) format.

    Example::

          import PIL
          from mltemplate.utils import pil_to_cv2

          pil_image = PIL.Image.open('tests/resources/hopper.png')
          cv2_image = pil_to_cv2(pil_image)
    """
    return pil_to_ndarray(image, image_format="BGR")


def cv2_to_pil(image: np.ndarray) -> Image:
    """Convert PIL image to cv2 image.

    Note that, in addition to cv2 images being numpy arrays, PIL Images follow RGB format while cv2 images follow BGR
    format.

    Args:
        image: The input image. Should be a np.ndarray in 'BGR' (cv2) format.

    Returns:
        A PIL image.

    Example::

          import cv2
          from mltemplate.utils import cv2_to_pil

          cv2_image = cv2.imread('tests/resources/hopper.png')
          pil_image = cv2_to_pil(cv2_image)
    """
    return ndarray_to_pil(image, image_format="BGR")
