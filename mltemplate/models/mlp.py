"""Multi-layer perceptron (MLP) model."""
from typing import List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image
from torch import nn

from mltemplate.utils import ifnone, pil_to_tensor


class MLP(nn.Module):
    """A simple multi-layer perceptron (MLP) model.

    Args:
        dim: The dimensions of the MLP. The first element is the input dimension, the last element is the output
            dimension, and the intermediate elements are the hidden dimensions.
        dropout: The dropout probability.

    Example::

        from mltemplate.data import MNIST
        from mltemplate.models import MLP

        mnist = MNIST()
        model = MLP(dim=[784, 100, 10])

        image, label = mnist.sample(stage='test', idx=100)
        prediction = model(image)
        print(f'label: {label}, prediction: {prediction}')

    """

    def __init__(self, dim: Optional[List[int]] = None, dropout: float = 0.2, **_):
        super().__init__()
        dim = ifnone(dim, default=[784, 100, 10])

        layers = []
        for i in range(len(dim) - 1):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def __call__(self, image: Union[Image, np.ndarray, torch.Tensor]):
        """Classify an image.

        Args:
            image: The image to classify. Can be a PIL Image, a numpy array, or a torch Tensor.

        Returns:
            The class logits, as a two-dimensional numpy array.
        """
        if isinstance(image, np.ndarray):
            image = torch.Tensor(image).type(torch.FloatTensor)
        elif isinstance(image, Image):
            image = pil_to_tensor(image).type(torch.FloatTensor)
        elif not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be of type Image, np.ndarray, or torch.Tensor, but got {type(image)}.")
        prediction = torch.softmax(self.forward(image), dim=1)
        return prediction

    def forward(self, x: torch.Tensor):
        """Forward pass of the model.

        Args:
            x: The input to the model.

        Returns:
            Torch Tensor output of the model.
        """
        # x has dimensions (B, C, H, W) or (C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.model(x.reshape(x.shape[0], -1))
