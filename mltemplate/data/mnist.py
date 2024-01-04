"""MNIST dataset module."""
import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from mltemplate import Config
from mltemplate.utils import ifnone


class MNISTDataModule(pl.LightningDataModule):
    """MNIST dataset module.

    The MNISTDataModule is a PyTorch Lightning DataModule which provides train, val, and test dataloaders for the
    MNIST dataset, and may be passed directly into a Pytorch Lightning Trainer. The MNIST dataset is a collection of
    70,000 28x28 grayscale images of handwritten digits (0-9) and their corresponding labels. The dataset is split into
    60,000 training images and 10,000 test images. The training images are further commonly split into 55,000 training
    and 5,000 validation images, although these numbers may be set by the user.

    Example, Manually Preparing and Using the DataModule::

        from mltemplate.data import MNIST

        # Instantiate MNIST dataset module
        dm = MNIST()
        dm.prepare_data()  # downloads data
        dm.setup()  # splits into train/val/test

        # Torch DataLoaders can then be accessed with:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

    Example, Using the DataModule with a PyTorch Lightning Trainer::

        from mltemplate.data import MNIST
        from pytorch_lightning import Trainer

        # Instantiate MNIST dataset module
        dm = MNIST()

        # Pass MNIST dataset module to PyTorch Lightning Trainer
        trainer = Trainer()
        trainer.fit(model, dm)

    """

    config = Config()

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        num_train: int = 55000,
        num_val: int = 5000,
        pin_memory: bool = False,
        data_dir: Optional[str] = None,
        **_,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(os.cpu_count() // 2, 1)
        self.num_train = num_train
        self.num_val = num_val
        self.pin_memory = pin_memory
        self.data_dir = ifnone(data_dir, default=self.config["DIR_PATHS"]["DATA"])

        self.train = None
        self.val = None
        self.test = None
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=False)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train, self.val = random_split(
            MNIST(self.data_dir, train=True, transform=transform), [self.num_train, self.num_val]
        )
        self.test = MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def sample(self, stage: str = "train", idx: int = 0):
        """Return a sample from the dataset.

        Args:
            stage: The stage of the dataset from which to sample. Must be one of: train, val, test.
            idx: The index of the sample to return.

        Returns:
            A sample from the dataset as an (image, label) tuple.
        """
        if stage == "train":
            return self.train[idx]
        elif stage == "val":
            return self.val[idx]
        elif stage == "test":
            return self.test[idx]
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be one of: train, val, test.")
