from functional_autoencoders.datasets import GenerableDataset
import numpy as np


class RandomDirac(GenerableDataset):
    """
    Dataset representing Dirac masses with random, uniformly chosen centre in :math:`(0, 1)`.

    For numerical purposes, the Dirac mass is represented by a function which is zero except at one mesh point,
    with height chosen such that the left/right Riemann sum has constant mass 1 at any resolution.
    """

    def __init__(
        self,
        fixed_centre,
        pts=128,
        transform=None,
        *args,
        **kwargs,
    ):
        self._fixed_centre = fixed_centre
        self._pts = pts
        self.transform = transform
        super().__init__(*args, **kwargs)

    def generate(self):
        x = np.linspace(0, 1, self._pts + 2)[1:-1]
        x = np.expand_dims(x, -1)
        if self._fixed_centre:
            centres = np.ones((2,), dtype=np.int32) * self._pts // 2
        else:
            centres = np.tile(np.arange(8)[1:-1] * int(self._pts / 8), 2)
        masses = np.ones_like(centres)
        height = self._pts + 1
        u = np.zeros((centres.shape[0], self._pts))
        for i, c in enumerate(centres):
            u[i, c] = height * masses[i]
        u = np.expand_dims(u, -1)

        self.data = {"u": u, "x": x, "masses": masses, "centres": centres}

    @property
    def x(self):
        return self.data["x"][:]

    @property
    def masses(self):
        return self.data["masses"][:]

    @property
    def centres(self):
        return self.data["centres"][:]

    def __len__(self):
        return self.data["u"].shape[0] // 2

    def __getitem__(self, idx):
        if not self.train:
            idx += self.data["u"].shape[0] // 2
        u = self.data["u"][idx]
        x = self.data["x"][:]
        if self.transform is not None:
            return self.transform(u, x)
        else:
            return u, x, u, x
