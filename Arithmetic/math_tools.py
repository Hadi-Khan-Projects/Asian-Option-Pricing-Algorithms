import numpy as np

# return: Fourier transform of the input function.
def fourier_transform(function, points, dx):
    return np.fft.fft(function) * dx

# return: Inverse Fourier transform, back to the spatial domain.
def inverse_fourier_transform(transform, points, dk):
    return np.fft.ifft(transform) / dk

# return: Scaled Fourier transform.
def scale_transform(transform, scale_factor):
    return transform * scale_factor
