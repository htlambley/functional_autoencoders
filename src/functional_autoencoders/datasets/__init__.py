import os
import shutil
import requests
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from typing import Literal
from jax.tree_util import tree_map
from torch.utils.data import (
    Dataset,
    DataLoader,
    default_collate,
)


@dataclass
class ComplementMasking:
    encoder_point_ratio: float

    def __call__(self, u, x):
        if self.encoder_point_ratio == -1:
            return u, x, u, x
        elif self.encoder_point_ratio <= 0.0 or self.encoder_point_ratio >= 1.0:
            raise ValueError("`encoder_point_ratio` for `ComplementMasking` should be in range 0 < encoder_point_ratio < 1.")

        n_total_pts = u.shape[0]
        n_rand_pts = int(self.encoder_point_ratio * n_total_pts)
        indices = np.random.choice(n_total_pts, n_rand_pts, replace=False)
        indices_comp = np.setdiff1d(np.arange(n_total_pts), indices)

        u_enc = u[indices]
        x_enc = x[indices]

        u_dec = u[indices_comp]
        x_dec = x[indices_comp]

        return u_enc, x_enc, u_dec, x_dec


@dataclass
class RandomMasking:
    encoder_point_ratio: float
    decoder_point_ratio: float

    def __call__(self, u, x):
        if self.encoder_point_ratio == -1 and self.decoder_point_ratio == -1:
            return u, x, u, x
        elif self.encoder_point_ratio <= 0.0 or self.encoder_point_ratio >= 1.0 or self.decoder_point_ratio <= 0.0 or self.decoder_point_ratio >= 1.0:
            raise ValueError("Point ratios for `RandomMasking` should be in range 0 < point_ratio < 1.")

        n_total_pts = u.shape[0]
        n_rand_pts_enc = int(self.encoder_point_ratio * n_total_pts)
        n_rand_pts_dec = int(self.decoder_point_ratio * n_total_pts)

        indices_enc = np.random.choice(n_total_pts, n_rand_pts_enc, replace=False)
        indices_dec = np.random.choice(n_total_pts, n_rand_pts_dec, replace=False)

        u_enc = u[indices_enc]
        x_enc = x[indices_enc]

        u_dec = u[indices_dec]
        x_dec = x[indices_dec]

        return u_enc, x_enc, u_dec, x_dec


@dataclass
class RandomMissingData:
    point_ratio: float

    def __call__(self, u, x):
        if self.point_ratio <= 0.0 or self.point_ratio >= 1.0:
            raise ValueError("`point_ratio` for `RandomMissingData` should satisfy 0.0 < point_ratio < 1.0")
        n_points = int(u.shape[1] * self.point_ratio)
        indices = np.sort(np.random.choice(u.shape[1], n_points, replace=False))
        u = u[:, indices]
        x = x[indices]
        return u, x


def get_dataloaders(
    dataset_class,
    batch_size=32,
    num_workers=0,
    shuffle_train=True,
    transform_train=None,
    transform_test=None,
    which: Literal["both", "train", "test"] = "both",
    **dataset_kwargs,
):
    if which != "test":
        train_dataset = dataset_class(
            train=True,
            transform=transform_train,
            **dataset_kwargs,
        )
        train_dataloader = NumpyLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_train,
        )

    if which != "train":
        test_dataset = dataset_class(
            train=False,
            transform=transform_test,
            **dataset_kwargs,
        )
        test_dataloader = NumpyLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    if which == "train":
        return train_dataloader
    elif which == "test":
        return test_dataloader
    else:
        return train_dataloader, test_dataloader


# `_numpy_collate` and `NumpyLoader` are based on the JAX notebook https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def _numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):

        # Batch_sampler option is mutually exclusive with
        # batch_size, shuffle, sampler, and drop_last.
        if batch_sampler is not None:
            additional_args = {}
        else:
            additional_args = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "drop_last": drop_last,
            }

        super(self.__class__, self).__init__(
            dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_numpy_collate,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            **additional_args,
        )


class OnDiskDataset:
    data_base: str
    dataset_filename: str
    dataset_filename: str

    @property
    def dataset_dir(self):
        """The path to the dataset directory on disk."""
        return os.path.join(os.getcwd(), self.data_base, "data", self.dataset_name)

    @property
    def dataset_path(self):
        """The path to the dataset on disk."""
        return os.path.join(self.dataset_dir, self.dataset_filename)

    @property
    def _data_exists(self):
        """Checks whether the dataset has already been downloaded to disk."""
        try:
            return os.path.isdir(self.dataset_dir)
        except AttributeError:
            raise NotImplementedError(
                "Dataset must have attributes data_base, dataset_name and dataset_filename"
            )


class GenerableDataset(Dataset):
    """A dataset which can be generated locally

    Derived classes must implement `generate`, `__len__` and `__getitem__`.
    """

    def __init__(self, train=True, *args, **kwargs):
        self.train = train
        self.generate()
        super().__init__(*args, **kwargs)

    def generate(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class DownloadableDataset(Dataset, OnDiskDataset):
    """A dataset which manages downloading, preprocessing and loading from a remote source.

    `DownloadableDataset` checks to see if the dataset has already been downloaded;
    if not present on disk, the dataset is downloaded from the remote repository, as
    specified through the attribute `data_url`.
    The dataset is then preprocessed using the `_preprocess_data` method (which by default
    does nothing) and then in any case is loaded using `_load_data`.

    The preprocessing step is run by default after download, and can be used for tasks such as
    unzipping a compressed file. It can be forced using the `force_preprocess` option, which will
    run the preprocessing step even if the dataset is already present on disk.

    Derived classes must implement `_load_data`, `__len__` and `__getitem__`.
    They can also optionally implement `_preprocess_data`.

    ## Parameters
    download : bool
        Toggles whether to allow downloading the dataset from the remote source. (default = `True`)
        Even if `True`, will only download if not already present on disk.

    force_preprocess : bool
        If `True`, runs the preprocessing step even if the dataset has already been downloaded. (default = `False`)

    force_download : bool
        If `True`, redownloads and preprocesses the data even if already downloaded. (default = `False`)

    train : bool
        Toggles whether to use train or test split. (default = `True`)

    data_base : str
        Modifies the search path for the data directory. (default = `''`)
        The default search path is `current_working_directory/data`, and
        `data_base` alters the base path relative to `current_working_directory`.
    """

    def __init__(
        self,
        download=True,
        force_preprocess=False,
        force_download=False,
        train=True,
        data_base="",
        *args,
        **kwargs,
    ):
        self.train = train
        self.data_base = data_base

        if not self._data_exists or force_download:
            if not download:
                raise ValueError(
                    "Dataset not found and download=False. Try setting download=True."
                )
            if os.path.exists(self.dataset_dir):
                print(f"Purging old dataset")
                shutil.rmtree(self.dataset_dir)

            print(f"Downloading dataset {self.dataset_name}.")
            self._download_data()
            print(f"Preprocessing dataset {self.dataset_name}")
            self._preprocess_data()
        elif force_preprocess:
            self._preprocess_data()

        self._load_data(train)
        super().__init__(*args, **kwargs)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def _load_data(self, train):
        """Loads the downloaded and preprocessed dataset from disk.

        Parameters
        ----------

        train : bool
            Determines whether the train or test split should be loaded.
        """
        raise NotImplementedError()

    def _preprocess_data(self):
        """Preprocesses the downloaded dataset.

        For example, this can be used for extracting zipped files or other data
        preprocessing after download.

        The default action is to do no preprocessing.
        """
        pass

    def _download_data(self):
        """Downloads the dataset from the URL specified by attribute `data_url`."""
        try:
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            r = requests.get(
                self.data_url, stream=True, headers={"User-agent": "Mozilla/5.0"}
            )
            if r.status_code != 200:
                raise ValueError(
                    f"Failed to download from specified data_url: error code {r.status_code}"
                )
            total = int(r.headers.get("content-length", 0))

            with (
                open(self.download_path, "wb") as f,
                tqdm(total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar,
            ):
                for data in r.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)

        except AttributeError:
            raise NotImplementedError(
                "DownloadableDataset must have attribute data_url"
            )

    @property
    def download_path(self):
        """The path to the downloaded file on disk."""
        return os.path.join(
            os.getcwd(),
            self.data_base,
            "data",
            self.dataset_name,
            self.download_filename,
        )

