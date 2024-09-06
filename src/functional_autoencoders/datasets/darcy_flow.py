import os
import zipfile
import scipy
import numpy as np
from functional_autoencoders.util.anti_aliasing import AntiAliasingManagerFourier
from functional_autoencoders.util import pickle_save, pickle_load, get_raw_x
from functional_autoencoders.datasets import DownloadableDataset


class DarcyFlow(DownloadableDataset):
    r"""
    Solution pressure fields $p \colon \Omega \to \mathbb{R}$ for the Darcy flow steady-state model of flow in a porous medium,
    with domain $\Omega = [0, 1]^{2}$, as described in section 4.3.2 of [the paper](https://arxiv.org/abs/2408.01362), solving the
    partial-differential equation

    $$ - \nabla \cdot \bigl(k \nabla p \bigr) = \varphi \text{~on $\Omega$}, $$
    $$ p = 0 \text{~on $\partial \Omega$}, $$

    with $\varphi = 1$ and $k$ distributed randomly as the pushforward of the measure $N(0, (-\Delta + 9I)^{-2})$ under the map

    $$\psi(x) = 3 + 9 \cdot 1 \bigl[ x \geq 0 \bigr].$$

    This is based on the dataset of Li et al. (2021) and consists of 1,024 train and 1,024 test samples discretised on a $421 \times 421$
    grid.

    Run once with save_fast=True to save the fast data files.
    Subsequent runs can be done with load_fast=True to load the fast data files.

    ## References
    Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, and Anandkumar (2021). Fourier neural operator for parametric partial differential equations. ICLR 2021.
        arXiv:2010.08895.
    """

    data_url = "https://drive.google.com/u/0/uc?id=1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf&export=download&confirm=t&uuid=9d0c35a0-3979-4852-b8fd-c1d4afec423c&at=AB6BwCA0wHtyid20GZfaIBVJ4aQv:1702379684316"
    download_filename = "Darcy_421.zip"
    dataset_filename = ""
    dataset_name = "fno_darcy"

    def __init__(
        self,
        downscale=-1,
        save_fast=False,
        load_fast=False,
        transform=None,
        *args,
        **kwargs,
    ):
        self.transform = transform
        self.downscale = downscale
        self.save_fast = save_fast
        self.load_fast = load_fast

        super().__init__(*args, **kwargs)

    def _preprocess_data(self):
        with zipfile.ZipFile(self.download_path, "r") as f:
            f.extractall(self.dataset_dir)

    def _get_slow_data_filename(self, train):
        if train:
            return "piececonst_r421_N1024_smooth1.mat"
        else:
            return "piececonst_r421_N1024_smooth2.mat"

    def _get_fast_data_filename(self, train):
        slow_data_filename = self._get_slow_data_filename(train)
        fast_data_filename = slow_data_filename.replace(".mat", "_fast.pkl")
        return fast_data_filename

    def _load_data(self, train):
        if self.load_fast:
            self._load_data_fast(train)
        else:
            self._load_data_slow(train)

    def _load_data_slow(self, train):
        self.dataset_filename = self._get_slow_data_filename(train)

        mat = scipy.io.loadmat(self.dataset_path, variable_names=["coeff", "sol"])
        u = mat["sol"].astype(float)

        n = u.shape[-1]
        x = get_raw_x(n, n).reshape(n, n, 2)
        x = np.array(x)

        if self.downscale != -1:
            aam = AntiAliasingManagerFourier(
                cutoff_nyq=0.99,
                mask_blur_kernel_size=7,
                gaussian_sigma=0.1,
            )

            u = aam.downsample(u, self.downscale)
            x = x[::self.downscale, ::self.downscale, :]

        u = u.reshape(u.shape[0], -1, 1)
        x = x.reshape(-1, 2)

        u = (u - u.min()) / (u.max() - u.min())
        self.data = {
            "u": u,
            "x": x,
        }

        if self.save_fast:
            print("Saving fast data")

            save_filename = self._get_fast_data_filename(train)
            save_path = os.path.join(self.dataset_dir, save_filename)
            pickle_save(self.data, save_path)

            print("Done!")

    def _load_data_fast(self, train):
        self.dataset_filename = self._get_fast_data_filename(train)
        load_path = os.path.join(self.dataset_dir, self.dataset_filename)
        self.data = pickle_load(load_path)

    def __len__(self):
        return self.data["u"].shape[0]

    def __getitem__(self, idx):
        u = self.data["u"][idx].reshape(-1, 1)
        x = self.data["x"].reshape(-1, 2)

        if self.transform is not None:
            return self.transform(u, x)
        else:
            return u, x, u, x
