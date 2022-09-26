import jax
import jax.numpy as jnp
import flax
import numpy as np
from PIL import Image
import os
import scipy
from tqdm import tqdm

import torch


def is_image(name):
    image_extensions = ['.jpg', 'jpeg', '.png']
    for ext in image_extensions:
        if name.lower().endswith(ext):
            return True
    return False


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=256):
        self.path = path
        self.img_size = img_size
        self.image_list = [x for x in os.listdir(path) if is_image(x)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.image_list[index]))

        if img.mode == "L":
            img = img.convert("RGB")
        # resize if not the right size
        if img.size[:2] != (self.img_size, self.img_size):
            img = img.resize(
                size=(self.img_size, self.img_size),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        return img


def compute_statistics(path, params, apply_fn, batch_size=1, img_size=256):
    if path.endswith(".npz"):
        stats = np.load(path)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma

    # images = []
    # for f in tqdm(os.listdir(path), ncols=0):
    #     img = Image.open(os.path.join(path, f))
    #     # convert if only a single channel
    #     if img.mode == "L":
    #         img = img.convert("RGB")
    #     # resize if not the right size
    #     if img_size is not None and img.size[:2] != (img_size, img_size):
    #         img = img.resize(
    #             size=(img_size, img_size),
    #             resample=Image.BILINEAR,
    #         )
    #     img = np.array(img) / 255.0
    #     images.append(img)

    # num_batches = int(len(images) // batch_size)
    # act = []
    # for i in tqdm(range(num_batches), ncols=0):
    #     x = images[i * batch_size : i * batch_size + batch_size]
    #     x = np.asarray(x)
    #     x = 2 * x - 1
    #     pred = apply_fn(params, jax.lax.stop_gradient(x))
    #     act.append(pred.squeeze(axis=1).squeeze(axis=1))
    # act = jnp.concatenate(act, axis=0)


    dataloader = torch.utils.data.DataLoader(
        ImageFolderDataset(path=path, img_size=img_size),
        batch_size=batch_size, num_workers=4, drop_last=True,
        collate_fn=lambda x: np.stack(x, axis=0)
    )
    def dataloader_iterator():
        n_devices = len(jax.local_devices())
        for x in dataloader:
            x = 2.0 * x - 1.0
            yield x.reshape(n_devices, -1, *x.shape[1:])

    jax_dataloader = flax.jax_utils.prefetch_to_device(
        dataloader_iterator(), size=4, devices=jax.local_devices()
    )
    act = []
    for x in tqdm(jax_dataloader, total=len(dataloader), ncols=0):
        pred = jax.device_get(apply_fn(params, x))
        pred = pred.reshape(-1, *pred.shape[2:])
        act.append(pred.squeeze(axis=1).squeeze(axis=1))
    act = jnp.concatenate(act, axis=0)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
