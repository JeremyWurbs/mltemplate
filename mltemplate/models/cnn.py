"""Convnet model."""
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image
from torch import nn

from mltemplate.utils import pil_to_tensor


class CNN(nn.Module):
    """A simple convolutional neural network model.

    Example::

        from mltemplate.data import MNIST
        from mltemplate.models import CNN

        mnist = MNIST()
        model = CNN()

        image, label = mnist.sample(stage='test', idx=100)
        prediction = model(image)
        print(f'label: {label}, prediction: {prediction}')

    """

    def __init__(self, channels=1, height=28, width=28, num_classes=10, **_):
        super().__init__()
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

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

    def forward(self, x):
        # x has dimensions (B, C, H, W) or (C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
