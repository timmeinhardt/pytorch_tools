"""Dataset Loaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
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


class SubsetSampler(Sampler):  # pylint: disable=too-few-public-methods
    """
    Return subset of dataset. For example to enforce overfitting.
    """

    def __init__(self, indices, subset_size, random_subset=False, shuffle=True):
        assert subset_size <= len(indices), (
            f"The subset size ({subset_size}) must be smaller "
            f"or equal to the sampler size ({len(indices)}).")
        self._subset_size = subset_size
        self._shuffle = shuffle
        self._random_subset = random_subset
        self._indices = indices
        self._subset = None
        self.set_subset()

    def set_subset(self):
        """Set subset from sampler with size self._subset_size"""
        if self._random_subset:
            perm = torch.randperm(len(self._indices))
            self._subset = self._indices[perm][:self._subset_size]
        else:
            self._subset = toch.Tensor(self._indices[:self._subset_size])

    def __iter__(self):
        """Iterate over same or shuffled subset."""
        if self._shuffle:
            perm = torch.randperm(self._subset_size)
            return iter(self._subset[perm].tolist())
        return iter(self._subset)

    def __len__(self):
        return len(self._subset)


#
# Transforms
#

class ToTensor(transforms.ToTensor):

    def __call__(self, pic):
        if torch.is_tensor(pic):
            pic = pic.float()
            pic.div_(255.0)
            return pic

        return super(ToTensor, self).__call__(pic)


class ToDevice(object):

    def __init__(self, device=None):
        self._device = device

    def __call__(self, tensor):
        t = tensor.to(self._device)
        return t

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

