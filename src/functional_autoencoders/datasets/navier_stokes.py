import os
import zipfile
import h5py
import scipy
import numpy as np
from functional_autoencoders.util import pickle_save, pickle_load, get_raw_x
from functional_autoencoders.datasets import DownloadableDataset


class NavierStokes(DownloadableDataset):
    r"""The Navier--Stokes dataset for a viscous, incompressible fluid in two dimensions as used by Li et al. (2021),
    as described in [the paper](https://arxiv.org/abs/2408.01362).

    ## Dataset

    The dataset consists of pairs `(initial_condition, trajectory)`.
    The initial condition is generated from a Gaussian random field with mean
    zero and covariance operator

    $$ \mathcal{C} = 7^{3/2} \bigl(-\Delta + 49I\bigr)^{-5/2}.$$

    The trajectory consists of $T$ timesteps evolved from the initial condition
    with a step length $\Delta t$.
    The forcing term $\varphi$ is given by

    $$ f(x) = 0.1 \Bigl( \sin\bigl( 2\pi (x_{1} + x_{2}) \bigr) + \cos\bigl( 2\pi(x_{1} + x_{2}) \bigr) \Bigr).$$

    The possible choices of parameters are as follows:
    - `viscosity = 1e-3`, `resolution = 64`. This dataset has 5,000 trajectories of $T = 50$ seconds with step $\Delta t = 1$ second.
    - `viscosity = 1e-4`, `resolution = 64`. This dataset has 10,000 trajectories of $T = 50$ seconds with step $\Delta t = 1$ second.
    - `viscosity = 1e-4`, `resolution = 256`. This dataset has 20 trajectories of $T = 50$ seconds with step $\Delta t = 0.25$ seconds.
    - `viscosity = 1e-5`, `resolution = 64`. This dataset has 1,200 trajectories of $T = 20$ seconds with step $\Delta t = 1$ second.

    ## Notes

    In Li et al. (2021), the $\nu = 10^{-4}$ dataset is simulated only up to $T = 30$ seconds,
    but the dataset actually includes simulations up to $T = 50$.
    """

    dataset_name = "navier_stokes"
    # Data URL and filename are dynamically determined based on resolution and viscosity choice.
    data_url = ""
    download_filename = ""
    dataset_filename = ""
    _is_h5 = True

    def __init__(
        self,
        viscosity=1e-3,
        resolution=64,
        time_idx=-1,
        train_test_split_ratio=0.8,
        save_fast=False,
        load_fast=False,
        transform=None,
        *args,
        **kwargs,
    ):
        self.train_test_split_ratio = train_test_split_ratio
        self.time_idx = time_idx
        self.save_fast = save_fast
        self.load_fast = load_fast

        if viscosity not in [1e-3, 1e-4, 1e-5]:
            raise ValueError("Viscosity must take value 1e-3, 1e-4 or 1e-5")

        if resolution not in [64, 256]:
            raise ValueError(
                "Navier--Stokes dataset only available at 64x64 or 256x256 resolution. See documentation for valid combinations."
            )

        if resolution != 64 and viscosity != 1e-4:
            raise ValueError(
                "Navier--Stokes dataset only available at 256x256 when viscosity is 1e-4."
            )

        self.viscosity = viscosity
        self.resolution = resolution
        self.transform = transform

        if viscosity == 1e-3:
            self.data_url = "https://drive.usercontent.google.com/download?id=1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d&export=download&authuser=0&confirm=t&uuid=05b098fa-6a5b-40cd-9b0f-39fa5b7c9261&at=APZUnTWMn104jdp7fiMuS7y5sxL7:1702487907119"
            self.download_filename = "NavierStokes_V1e-3_N5000_T50.zip"
            self.dataset_filename = "ns_V1e-3_N5000_T50.mat"

        elif viscosity == 1e-4:
            if resolution == 64:
                self.data_url = "https://drive.usercontent.google.com/download?id=1RmDQQ-lNdAceLXrTGY_5ErvtINIXnpl3&export=download&authuser=0&confirm=t&uuid=a218d5ab-1b75-4b1c-a5da-ed0b71bd3f20&at=APZUnTXLTgsSn6kzgcqTUn2fwDTk:1702490181229"
                self.download_filename = "NavierStokes_V1e-4_N10000_T30.zip"
                self.dataset_filename = "ns_V1e-4_N10000_T30.mat"
            else:
                self.data_url = "https://drive.usercontent.google.com/download?id=1pr_Up54tNADCGhF8WLvmyTfKlCD5eEkI&export=download&authuser=0&confirm=t&uuid=d6ce8938-295e-40a4-9f87-d24ebdabf310&at=APZUnTUbM3Hu0GZiNSbQuzuyH7fr:1702639561668"
                self.download_filename = "NavierStokes_V1e-4_N20_T50_R256_test.zip"
                self.dataset_filename = "ns_data_V1e-4_N20_T50_R256test.mat"
                self._is_h5 = False

        elif viscosity == 1e-5:
            self.data_url = "https://drive.usercontent.google.com/download?id=1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5&export=download&authuser=0&confirm=t&uuid=68addf1d-8b63-4591-b32a-6ee2b4886e66&at=APZUnTU4rRh0Qwj4UUGbBBxetFUR:1702490348128"
            self.download_filename = "NavierStokes_V1e-5_N1200_T20.zip"
            self.dataset_filename = "NavierStokes_V1e-5_N1200_T20.mat"
            self._is_h5 = False

        else:
            raise NotImplementedError()

        super().__init__(*args, **kwargs)

    def _load_data(self, train):
        if self.load_fast:
            self._load_data_fast(train)
        else:
            self._load_data_slow(train)

    def _load_data_slow(self, train):
        if self._is_h5:
            data = h5py.File(self.dataset_path, "r")
            self.u_data = np.moveaxis(data["u"][self.time_idx, :, :, :], -1, 0)
        else:
            data = scipy.io.loadmat(self.dataset_path)
            self.u_data = data["u"][:, :, :, self.time_idx]

        n_train = int(self.train_test_split_ratio * self.u_data.shape[0])

        if train:
            self.u_data = self.u_data[:n_train]
        else:
            self.u_data = self.u_data[n_train:]

        self.u_data = (self.u_data - self.u_data.min()) / (
            self.u_data.max() - self.u_data.min()
        )
        self.x = get_raw_x(*self.u_data.shape[1:3])
        self.x = np.array(self.x)

        if self.save_fast:
            self._save_data_fast(train)

    def _save_data_fast(self, train):
        print("Saving fast data")

        save_filename = self._get_fast_data_filename(train)
        save_path = os.path.join(self.dataset_dir, save_filename)
        pickle_save(
            {
                "u_data": self.u_data,
                "x": self.x,
            },
            save_path,
        )

        print("Done!")

    def _load_data_fast(self, train):
        self.dataset_filename = self._get_fast_data_filename(train)
        load_path = os.path.join(self.dataset_dir, self.dataset_filename)
        data = pickle_load(load_path)
        self.u_data = data["u_data"]
        self.x = data["x"]

    def _get_fast_data_filename(self, train):
        filename_suffix = ("_train" if train else "_test") + "_fast.pkl"
        fast_data_filename = self.dataset_filename.replace(".mat", filename_suffix)
        return fast_data_filename

    def _preprocess_data(self):
        with zipfile.ZipFile(self.download_path, "r") as f:
            f.extractall(self.dataset_dir)

    def __len__(self):
        return self.u_data.shape[0]

    def __getitem__(self, idx):
        u = self.u_data[idx].reshape(-1, 1)
        x = self.x.reshape(-1, 2)

        if self.transform is not None:
            return self.transform(u, x)
        else:
            return u, x, u, x
