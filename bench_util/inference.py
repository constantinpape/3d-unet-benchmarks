import numbers
import time
from itertools import product

import numpy as np
import torch
import torch.cuda.amp as amp

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x):
        return x


def product1d(inrange):
    for ii in inrange:
        yield ii


def slice_to_start_stop(s, size):
    """For a single dimension with a given size, normalize slice to size.
     Returns slice(None, 0) if slice is invalid."""
    if s.step not in (None, 1):
        raise ValueError('Nontrivial steps are not supported')

    if s.start is None:
        start = 0
    elif -size <= s.start < 0:
        start = size + s.start
    elif s.start < -size or s.start >= size:
        return slice(None, 0)
    else:
        start = s.start

    if s.stop is None or s.stop > size:
        stop = size
    elif s.stop < 0:
        stop = (size + s.stop)
    else:
        stop = s.stop

    if stop < 1:
        return slice(None, 0)

    return slice(start, stop)


def int_to_start_stop(i, size):
    """For a single dimension with a given size, turn an int into slice(start, stop)
    pair."""
    if -size < i < 0:
        start = i + size
    elif i >= size or i < -size:
        raise ValueError('Index ({}) out of range (0-{})'.format(i, size - 1))
    else:
        start = i
    return slice(start, start + 1)


def normalize_slices(slices, shape):
    """ Normalize slices to shape.

    Normalize input, which can be a slice or a tuple of slices / ellipsis to
    be of same length as shape and be in bounds of shape.

    Args:
        slices (int or slice or ellipsis or tuple[int or slice or ellipsis]): slices to be normalized

    Returns:
        tuple[slice]: normalized slices (start and stop are both non-None)
        tuple[int]: which singleton dimensions should be squeezed out
    """
    type_msg = 'Advanced selection inappropriate. ' \
               'Only numbers, slices (`:`), and ellipsis (`...`) are valid indices (or tuples thereof)'

    if isinstance(slices, tuple):
        slices_lst = list(slices)
    elif isinstance(slices, (numbers.Number, slice, type(Ellipsis))):
        slices_lst = [slices]
    else:
        raise TypeError(type_msg)

    ndim = len(shape)
    if len([item for item in slices_lst if item != Ellipsis]) > ndim:
        raise TypeError("Argument sequence too long")
    elif len(slices_lst) < ndim and Ellipsis not in slices_lst:
        slices_lst.append(Ellipsis)

    normalized = []
    found_ellipsis = False
    squeeze = []
    for item in slices_lst:
        d = len(normalized)
        if isinstance(item, slice):
            normalized.append(slice_to_start_stop(item, shape[d]))
        elif isinstance(item, numbers.Number):
            squeeze.append(d)
            normalized.append(int_to_start_stop(int(item), shape[d]))
        elif isinstance(item, type(Ellipsis)):
            if found_ellipsis:
                raise ValueError("Only one ellipsis may be used")
            found_ellipsis = True
            while len(normalized) + (len(slices_lst) - d - 1) < ndim:
                normalized.append(slice(0, shape[len(normalized)]))
        else:
            raise TypeError(type_msg)
    return tuple(normalized), tuple(squeeze)


def blocking(shape, block_shape, roi=None, center_blocks_at_roi=False):
    """ Generator for nd blocking.

    Args:
        shape (tuple): nd shape
        block_shape (tuple): nd block shape
        roi (tuple[slice]): region of interest (default: None)
        center_blocks_at_roi (bool): if given a roi,
            whether to center the blocks being generated
            at the roi's origin (default: False)
    """
    assert len(shape) == len(block_shape), "Invalid number of dimensions."

    if roi is None:
        # compute the ranges for the full shape
        ranges = [range(sha // bsha if sha % bsha == 0 else sha // bsha + 1)
                  for sha, bsha in zip(shape, block_shape)]
        min_coords = [0] * len(shape)
        max_coords = shape
    else:
        # make sure that the roi is valid
        roi, _ = normalize_slices(roi, shape)
        ranges = [range(rr.start // bsha,
                        rr.stop // bsha if rr.stop % bsha == 0 else rr.stop // bsha + 1)
                  for rr, bsha in zip(roi, block_shape)]
        min_coords = [rr.start for rr in roi]
        max_coords = [rr.stop for rr in roi]

    need_shift = False
    if roi is not None and center_blocks_at_roi:
        shift = [rr.start % bsha for rr, bsha in zip(roi, block_shape)]
        need_shift = sum(shift) > 0

    # product raises memory error for too large ranges,
    # because input iterators are cast to tuple
    # so far I have only seen this for 1d "open-ended" datasets
    # and hence just implemented a workaround for this case,
    # but it should be fairly easy to implement an nd version of product
    # without casting to tuple for our use case using the imglib loop trick, see also
    # https://stackoverflow.com/questions/8695422/why-do-i-get-a-memoryerror-with-itertools-product
    try:
        start_points = product(*ranges)
    except MemoryError:
        assert len(ranges) == 1
        start_points = product1d(ranges)

    for start_point in start_points:
        positions = [sp * bshape for sp, bshape in zip(start_point, block_shape)]
        if need_shift:
            positions = [pos + sh for pos, sh in zip(positions, shift)]
            if any(pos > maxc for pos, maxc in zip(positions, max_coords)):
                continue
        yield tuple(slice(max(pos, minc), min(pos + bsha, maxc))
                    for pos, bsha, minc, maxc in zip(positions, block_shape,
                                                     min_coords, max_coords))


def ensure_5d(tensor):
    if tensor.ndim == 3:
        tensor = tensor[None, None]
    elif tensor.ndim == 4:
        tensor = tensor[None]
    elif tensor.ndim == 5:
        pass
    return tensor


# we don't save any output, because this is just for benchmarking purposes
def run_inference(input_dataset, model,
                  block_shape, halo,
                  preprocess):
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    shape = input_dataset.shape

    full_block_shape = tuple(bs + 2 * ha for bs, ha in zip(block_shape, halo))
    local_bb = tuple(slice(ha, bsh - ha)
                     for bsh, ha in zip(block_shape, halo))

    def grow_bounding_box(bb):
        grown_bb = tuple(slice(max(b.start - ha, 0), min(sh, b.stop + ha))
                         for b, ha, sh in zip(bb, halo, shape))
        return grown_bb

    def ensure_block_shape(input_):
        if input_.shape != full_block_shape:
            pad_shape = [(0, bsh - sh)
                         for bsh, sh in zip(full_block_shape, input_.shape)]
            input_ = np.pad(input_, pad_shape)
        return input_

    blocks = list(blocking(shape, block_shape))
    per_block_times = []

    t_tot = time.time()
    with torch.no_grad():
        for bb in tqdm(blocks):
            bb = grow_bounding_box(bb)

            input_ = input_dataset[bb]
            input_ = ensure_block_shape(input_)

            input_ = preprocess(input_)
            input_ = ensure_5d(input_)
            t0 = time.time()
            with amp.autocast():
                input_ = torch.from_numpy(input_).to(device)
                output = model(input_)
                output = output.cpu().numpy()
            per_block_times.append(time.time() - t0)

            # this is where we would save the output ...
            output = output[0]
            output = output[(slice(None),) + local_bb]
    t_tot = time.time() - t_tot

    return t_tot, per_block_times
