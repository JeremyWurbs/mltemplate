"""Module containing mock assets for unit testing."""
import cv2
import numpy as np
from PIL import Image


class MockAssets:
    """Class containing mock assets for unit tests."""

    image_path = "tests/resources/hopper.png"
    mask_path = "tests/resources/hopper_mask.png"
    audio_path = "tests/resources/write_a_fond_note.mp3"
    image = Image.open(image_path).convert("RGB")  # 1024x768 portrait image
    image_mask = Image.open(mask_path).convert("L")  # 1024x768 mask to image
    image_large = Image.open("tests/resources/hopper_large.png").convert("RGB")  # 2048x1536 image
    image_square = Image.open("tests/resources/hopper_square.png").convert("RGB")  # 1024x1024 image
    image_background = Image.open("tests/resources/office_in_a_small_city.png").convert("RGB")  # background image
    prompt = "An astronaut riding a horse"

    @property
    def image_rgba(self) -> Image:
        """Returns RGBA version of MockAssets.image."""
        rgba = self.image.copy()
        rgba.putalpha(self.image_mask)
        assert rgba.mode == "RGBA"
        return rgba

    @property
    def image_la(self) -> Image:
        """Returns LA version of MockAssets.image."""
        la = self.image.copy().convert("LA")
        la.putalpha(self.image_mask)
        assert la.mode == "LA"
        return la

    @property
    def image_wide(self) -> Image:
        """Returns an image that is wider than it is tall."""
        return self.image.copy().transpose(Image.ROTATE_90)

    @property
    def image_tall(self) -> Image:
        """Returns an image that is taller than it is wide."""
        return self.image.copy()

    @property
    def image_ndarray_bgr(self) -> np.ndarray:
        """Returns cv2 version (np.ndarray in BGR format) of MockAssets.image."""
        image = cv2.imread(self.image_path)
        assert image.shape[-1] == 3  # assert BGR format
        return image

    @property
    def image_ndarray_bgra(self) -> np.ndarray:
        """Returns a cv2 image with an alpha channel."""
        image = cv2.cvtColor(self.image_ndarray_bgr, cv2.COLOR_BGR2BGRA)
        mask = cv2.cvtColor(cv2.imread(self.mask_path), cv2.COLOR_BGR2GRAY)
        image[:, :, 3] = mask
        assert image.shape[-1] == 4  # assert BGRA format
        return image

    @property
    def image_uint8_ndarray(self) -> np.ndarray:
        """Returns a numpy uint8 ndarray image."""
        image_ndarray = np.asarray(self.image)
        assert image_ndarray.dtype == np.uint8
        return image_ndarray

    @property
    def image_float32_ndarray(self) -> np.ndarray:
        """Returns a numpy float32 ndarray image."""
        uint8_ndarray = self.image_uint8_ndarray
        fp32_ndarray = (uint8_ndarray / 255.0).astype(np.float32)
        assert fp32_ndarray.dtype == np.float32
        return fp32_ndarray
