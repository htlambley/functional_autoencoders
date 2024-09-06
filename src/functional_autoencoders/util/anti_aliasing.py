import numpy as np
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter


class AntiAliasingManagerBase(ABC):
    """
    Base class that allows for antialiased downsampling and upsampling.

    Users should not instantiate this class directly, and instead use `AntiAliasingManagerFourier`.
    """

    @abstractmethod
    def lowpass(self, z, sample_rate_h, sample_rate_w):
        pass

    def upsample(self, z, scale_factor=2):
        z_up = self._upsample_with_zero_insertion(z, scale_factor)
        z_up_low = self.lowpass(z_up, z.shape[-2], z.shape[-1]) * scale_factor**2
        return z_up_low

    def downsample(self, z, scale_factor=2):
        z_low = self.lowpass(z, z.shape[-2] // scale_factor, z.shape[-1] // scale_factor)
        z_low_down = z_low[:, ::scale_factor, ::scale_factor]
        return z_low_down

    def _upsample_with_zero_insertion(self, x, stride=2):
        *cdims, Hin, Win = x.shape
        Hout = stride * Hin
        Wout = stride * Win
        out = x.new_zeros(*cdims, Hout, Wout)
        out[..., ::stride, ::stride] = x
        return out


class AntiAliasingManagerFourier(AntiAliasingManagerBase):
    """
    Allows for downsampling and upsampling using a smoothed filter applied in Fourier space.

    The low-pass filter is a mollification of an ideal $\mathrm{sinc}$ filter with bandwidth
    selected to eliminate frequencies beyond the Nyquist frequency of the target resolution,
    computed in practice by convolving the ideal filter in Fourier space with a Gaussian kernel
    of with standard deviation `gaussian_sigma`, truncated to a `mask_blur_kernel_size` convolutional filter.
    """

    def __init__(self, cutoff_nyq, mask_blur_kernel_size, gaussian_sigma):
        self.cutoff_nyq = cutoff_nyq
        self.mask_blur_kernel_size = mask_blur_kernel_size
        self.gaussian_sigma = gaussian_sigma

    def lowpass(self, z, sample_rate_h, sample_rate_w):
        dft = np.fft.fft2(z, norm="ortho")
        dft_shift = np.fft.fftshift(dft)

        mask = self._get_blurred_mask(z, sample_rate_h, sample_rate_w)
        dft_shift_masked = np.multiply(dft_shift, mask)

        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)

        z_filtered = np.fft.ifft2(back_ishift_masked, norm="ortho").real
        return z_filtered

    def _get_blurred_mask(self, z, sample_rate_h, sample_rate_w):
        radius = self.mask_blur_kernel_size - 1 // 2
        mask = self._get_ideal_square_mask(z, sample_rate_h, sample_rate_w)
        mask = gaussian_filter(mask, self.gaussian_sigma, radius=radius)
        return mask

    def _get_ideal_square_mask(self, z, sample_rate_h, sample_rate_w):
        h, w = z.shape[-2:]
        mask = np.zeros((h, w))
        modes_h = int(self.cutoff_nyq * sample_rate_h)
        modes_w = int(self.cutoff_nyq * sample_rate_w)

        if (h - modes_h) % 2 != 0:
            modes_h -= 1

        if (w - modes_w) % 2 != 0:
            modes_w -= 1

        start_idx_h = (h - modes_h) // 2
        start_idx_w = (w - modes_w) // 2

        end_idx_h = start_idx_h + modes_h
        end_idx_w = start_idx_w + modes_w

        mask[start_idx_h:end_idx_h, start_idx_w:end_idx_w] = 1

        return mask
