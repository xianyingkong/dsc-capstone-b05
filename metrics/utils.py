"""

Directly ported from pytorch_gan_metrics with slight modification on images input
https://github.com/w86763777/pytorch-gan-metrics


"""
import os
from typing import List, Union, Tuple, Optional
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import (
    get_inception_feature,
    calculate_inception_score,
    calculate_frechet_inception_distance,
    torch_cov)


def get_inception_score_and_fid(
    images: Union[torch.FloatTensor, DataLoader],
    fid_stats_path: str,
    splits: int = 10,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[Tuple[float, float], float]:
    """Calculate Inception Score and FID.

    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        fid_stats_path: Path to pre-calculated statistic.
        splits: The number of bins of Inception Score.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        **kwargs: The arguments passed to
                  `pytorch_gan_metrics.core.get_inception_feature`.
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    acts, probs = get_inception_feature(
        images, dims=[2048, 1008], use_torch=use_torch, **kwargs)

    # Inception Score
    inception_score, std = calculate_inception_score(probs, splits, use_torch)

    # Frechet Inception Distance
    f = np.load(fid_stats_path, allow_pickle=True)
    if isinstance(f, np.ndarray):
        mu, sigma = f.item()['mu'][:], f.item()['sigma'][:]
    else:
        mu, sigma = f['mu'][:], f['sigma'][:]
        f.close()
    fid = calculate_frechet_inception_distance(acts, mu, sigma, use_torch)

    return (inception_score, std), fid


def get_fid(
    images: Union[torch.FloatTensor, DataLoader],
    fid_stats_path: str,
    use_torch: bool = False,
    **kwargs,
) -> float:
    """Calculate Frechet Inception Distance.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        fid_stats_path: Path to pre-calculated statistic.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        **kwargs: The arguments passed to
                  `pytorch_gan_metrics.core.get_inception_feature`.

    Returns:
        FID
    """
    acts, = get_inception_feature(
        images, dims=[2048], use_torch=use_torch, **kwargs)

    # Frechet Inception Distance
    f = np.load(fid_stats_path, allow_pickle=True)
    if isinstance(f, np.ndarray):
        mu, sigma = f.item()['mu'][:], f.item()['sigma'][:]
    else:
        mu, sigma = f['mu'][:], f['sigma'][:]
        f.close()
    fid = calculate_frechet_inception_distance(acts, mu, sigma, use_torch)

    return fid


def get_inception_score(
    images: Union[torch.FloatTensor, DataLoader],
    splits: int = 10,
    use_torch: bool = False,
    **kwargs,
) -> Tuple[float, float]:
    """Calculate Inception Score.

    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        splits: The number of bins of Inception Score.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        **kwargs: The arguments passed to
                  `pytorch_gan_metrics.core.get_inception_feature`.

    Returns:
        Inception Score
    """
    probs, = get_inception_feature(
        images, dims=[1008], use_torch=use_torch, **kwargs)
    inception_score, std = calculate_inception_score(probs, splits, use_torch)
    return (inception_score, std)


def calc_and_save_stats(
    imgs_dataset,
    output_path: str,
    batch_size: int = 50,
    use_torch: bool = False,
    num_workers: int = os.cpu_count(),
    verbose: bool = True,
) -> None:
    """Calculate the FID statistics and save them to a file.

    Args:
        input_path (str): Path to the image directory. This function will
                          recursively find images in all subfolders.
        output_path (str): Path to the output file.
        batch_size (int): Batch size. Defaults to 50.
        img_size (int): Image size. If None, use the original image size.
        num_workers (int): Number of dataloader workers. Default:
                           os.cpu_count().
    """
    loader = DataLoader(
        imgs_dataset, batch_size=batch_size, num_workers=num_workers)
    acts, = get_inception_feature(
        loader, dims=[2048], use_torch=use_torch, verbose=verbose)

    if use_torch:
        mu = torch.mean(acts, dim=0).cpu().numpy()
        sigma = torch_cov(acts, rowvar=False).cpu().numpy()
    else:
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)

    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
