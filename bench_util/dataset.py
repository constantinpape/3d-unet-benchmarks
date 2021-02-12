import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

# enable running this without scikit image (seg_to_boundaries won't work)
try:
    from skimage.segmentation import find_boundaries
except ImportError:
    find_boundaries = None


#
# the raw and target transformations we might need
#

def normalize(raw, mean=None, std=None, eps=1e-6):
    raw = raw.astype('float32')
    mean = np.mean(raw) if mean is None else mean
    std = np.std(raw) if std is None else std
    raw -= mean
    raw /= (std + eps)
    return raw


def seg_to_mask(seg, bg_val=0):
    mask = (seg != bg_val).astype('float32')
    return mask


def seg_to_boundaries(seg):
    assert find_boundaries is not None
    boundaries = find_boundaries(seg).astype('float32')
    return boundaries


class EMDataset(Dataset):

    def _compute_sample_len(self):
        assert os.path.exists(self.raw_path)
        assert os.path.exists(self.label_path)
        with h5py.File(self.raw_path, 'r') as f:
            shape = f[self.raw_key].shape
        with h5py.File(self.label_path, 'r') as f:
            assert f[self.label_key].shape == shape

        # compute n_samples as the number of times we can fit the patch shape into the full volume
        n_samples = int(np.prod(
            [float(sh / csh) for sh, csh in zip(shape, self.patch_size)]
        ))

        return n_samples

    def __init__(self,
                 raw_path, label_path,
                 raw_key, label_key,
                 patch_size,
                 raw_transform, target_transform,
                 transform=None):
        super().__init__()
        assert len(patch_size) == 3
        self.raw_path = raw_path
        self.raw_key = raw_key

        self.label_path = label_path
        self.label_key = label_key

        self.patch_size = patch_size

        self.raw_transform = raw_transform
        self.target_transform = target_transform
        self.transform = transform

        # compute the number of samples for each volume
        self._len = self._compute_sample_len()

    # get the total number of samples
    def __len__(self):
        return self._len

    def sample_bounding_box(self, shape):
        patch_start = [
            np.random.randint(0, sh - cs) if sh - cs > 0 else 0
            for sh, cs in zip(shape, self.patch_size)
        ]
        return np.s_[patch_start[0]:patch_start[0] + self.patch_size[0],
                     patch_start[1]:patch_start[1] + self.patch_size[1],
                     patch_start[2]:patch_start[2] + self.patch_size[2]]

    def sample_random_patch(self):
        with h5py.File(self.raw_path, 'r') as fr, h5py.File(self.label_path, 'r') as fg:
            ds_raw = fr[self.raw_key]
            ds_gt = fg[self.label_key]
            shape = ds_raw.shape
            bounding_box = self.sample_bounding_box(shape)

            raw = ds_raw[bounding_box]
            target = ds_gt[bounding_box]
        return raw, target

    def ensure_tensor(self, tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        assert torch.is_tensor(tensor)
        return tensor

    def ensure_4d(self, tensor):
        if tensor.ndim == 3:
            tensor = tensor[None]
        elif tensor.ndim == 5:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        elif tensor.ndim == 4:
            pass
        else:
            raise RuntimeError
        return tensor

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # sample a random patch
        raw, target = self.sample_random_patch()

        raw = self.ensure_tensor(self.raw_transform(raw))
        target = self.ensure_tensor(self.target_transform(target))

        # joint transforms
        if self.transform is not None:
            raw, target = self.transform(raw, target)

        raw, target = self.ensure_4d(raw), self.ensure_4d(target)
        return raw, target
