"""Dataset Loaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class DataLoader(_DataLoader):  # pylint: disable=too-few-public-methods
    """Custom DataLoader.

    The class provides a num_samples property.
    """

    @property
    def num_samples(self):
        """Total number of data samples.

        This property especially considers the drop_last parameter.
        """
        if self.drop_last:
            return len(self.sampler) - len(self.sampler) % self.batch_size
        return len(self.sampler)


class SubsetSampler(object):  # pylint: disable=too-few-public-methods
    """
    Return subset of dataset. For example to enforce overfitting.
    """

    def __init__(self, sampler=None, shuffle=True, subset_size=None):
        assert (sampler is not None or subset_size is not None), (
            "Either argument sampler or subset_size must be given.")
        if subset_size is None:
            subset_size = len(sampler)
        assert subset_size <= len(sampler), (
            f"The subset size ({subset_size}) must be smaller "
            f"or equal to the sampler size ({len(sampler)}).")
        self.subset_size = subset_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.set_random_subset()

    def set_random_subset(self):
        if self.sampler is None:
            self.random_subset = \
                torch.randperm(len(self.sampler))[:self.subset_size]
        else:
            self.random_subset = list(self.sampler)[:self.subset_size]

    def __iter__(self):
        if self.shuffle:
            # train with fixed random subset of dataset
            return iter(self.random_subset)
        # if given sampler has randomization we can not easily provide the first
        raise NotImplementedError
        # train with first subset of dataset
        return iter(range(self.subset_size))

    def __len__(self):
        return len(self.random_subset)


#
# Transforms
#

class ToTensor(transforms.ToTensor):

    def __call__(self, pic):
        if isinstance(pic, torch._TensorBase):
            pic = pic.float()
            pic.div_(255.0)
            return pic

        return super(ToTensor, self).__call__(pic)


class ToCUDA(object):

    def __init__(self, device=None):
        self._device = device

    def __call__(self, tensor):
        return tensor.cuda(self._device)


class Unsqueeze(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        tensor.unsqueeze_(dim=self.dim)
        return tensor


class Squeeze(object):

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        tensor.squeeze_(dim=self.dim)
        return tensor

