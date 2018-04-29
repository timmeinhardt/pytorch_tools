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

    def __init__(self, sampler, subset_size, random_subset=False, shuffle=True):
        # assert (sampler is not None or subset_size is not None), (
        #     "Either argument sampler or subset_size must be given.")
        # if subset_size is None:
        #     subset_size = len(sampler)
        assert subset_size <= len(sampler), (
            f"The subset size ({subset_size}) must be smaller "
            f"or equal to the sampler size ({len(sampler)}).")
        self._subset_size = subset_size
        self._shuffle = shuffle
        self._random_subset = random_subset
        self._sampler = sampler
        self.set_subset()

    def set_subset(self):
        """Set subset from sampler with size self._subset_size"""
        if self._random_subset:
            self._subset = torch.randperm(len(self._sampler))[:self._subset_size]
        else:
            self._subset = toch.Tensor(list(self._sampler)[:self._subset_size])

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

