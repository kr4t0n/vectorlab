"""
Spectral Residual (SR) is an efficient unsupervised
algorithm, which at first demonstrates outstanding performance
and robustness in the visual saliency detection tasks. It now
migrates to time series smoothing tasks.
"""
import numpy as np

from sklearn.base import TransformerMixin

from ._moving_average import MovingAverage
from .._utils import extend_series
from ...utils._check import check_valid_int, check_pairwise_1d_array

EPS = 1e-8


class SpectralResidual(TransformerMixin):
    r"""Spectral Residual (SR) is an efficient unsupervised
    algorithm, which demonstrates outstanding performance and
    robustness in the visual saliency detection tasks. The motivation
    is that the time-series anomaly detection task is similar to the
    problem of visual saliency detection essentially. When anomalies
    appear in time-series curves, they are always the most salient
    part in vision.

    The Spectral Residual (SR) algorithm consists of three major steps:

        1. Fourier Transform to get the log amplitude spectrum
        2. Calculation of `spectral residual`
        3. Inverse Fourier Transform that transforms the sequence back
           to spatial domain


    Mathematically, given a sequence `x`, we have

        .. math::
            \begin{gather*}
            A(f) = Amplitude(\mathfrak F({\bf x})) \\
            P(f) = Phrase(\mathfrak F({\bf x})) \\
            L(f) = log(A(f)) \\
            AL(f) = h_q(f) \cdot L(f) \\
            R(f) = L(f) - AL(f) \\
            S({\bf x}) = \|\mathfrak{F}^{-1} (exp(R(f) + iP(f)))\|
            \end{gather*}

    where :math:`\mathfrak F` and :math:`\mathfrak{F}^{-1}` denote Fourier
    Transform and Inverse Fourier Transform respectively. :math:`\bf x` is
    the input sequence with shape :math:`n \times 1`. `A(f)` is the amplitude
    spectrum of sequence :math:`\bf x`. `P(f)` is the corresponding phase
    spectrum of sequence :math:`\bf x`. `L(f)` is the log representation
    of `A(f)`, and `AL(f)` is the average spectrum of `L(f)` which can be
    approximated by convoluting the input sequence by :math:`h_q(f)`, where
    :math:`h_q(f)` is and :math:`q \times q` matrix defined as,

        .. math::
            h_q(f) = \dfrac{1}{q^2} \begin{bmatrix}
            1 & 1 & 1 & \ldots & 1\\
            1 & 1 & 1 & \ldots & 1\\
            \vdots & \vdots & \vdots & \ddots & \vdots\\
            1 & 1 & 1 & \ldots & 1 \end{bmatrix}

    `R(f)` is the `spectral residual`, the log spectrum `L(f)` subtracting the
    averaged log spectrum `AL(f)`. The `spectral residual` serves as a
    compressed representation of the sequence while the innovation part of
    the original sequence becomes more significant. At last, we transfer
    the sequence back to spatial domain via Inverse Fourier Transform.
    The result sequence :math:`S({\bf x})` is called the `saliency map`.

    Parameters
    ----------
    window_size : int
        How many previous data-points used to calculate the average log
        spectrum.
    extend_num : int, optional
        The number of extended points.
    look_ahead : int, optional
        The number of previous points to be considered.


    Attributes
    ----------
    window_size_ : int
        How many previous data-points used to calculate the average log
        spectrum.
    extend_num_ : int, optional
        The number of extended points.
    look_ahead_ : int, optional
        The number of previous points to be considered.
    X_ : array_like, shape (n_samples + extend_num)
        The extended input array.
    smoothed_X_ : array_like, shape (n_samples + extend_num)
        The smoothed extended input array.
    """

    def __init__(self, window_size,
                 extend_num=0, look_ahead=5):

        super().__init__()

        window_size = check_valid_int(
            window_size,
            lower=1,
            variable_name='window_size'
        )
        extend_num = check_valid_int(
            extend_num,
            lower=0,
            variable_name='extend_num'
        )
        look_ahead = check_valid_int(
            look_ahead,
            lower=1,
            variable_name='look_ahead'
        )

        self.window_size_ = window_size
        self.extend_num_ = extend_num
        self.look_ahead_ = look_ahead

        return

    def fit(self, X, Y=None):
        r"""Fit a spectral residual smoothing

        THE Spectral Residual (SR) method works better if the target point
        locates in the center of the sliding window. Thus, we add several
        `estimated points` after :math:`x_n` before inputting the sequence
        to SR model. The value of :math:`x_{n+1}` is calculated by series
        utilities `extend_series`. Detailed information can be obtained from
        that function documentation.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            SpectralResidual class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.X_ = extend_series(
            X,
            extend_num=self.extend_num_,
            look_ahead=self.look_ahead_
        )

        return self

    def transform(self, X, Y=None):
        r"""Transform a spectral residual smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In spectral residual, this function actually
        transform the input data X to smooth into smoothed_X with
        preset window size.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        smoothed_X : array_like, shape (n_samples)
            The smoothed input array.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        fft_trans = np.fft.fft(self.X_)
        amplitude = np.absolute(fft_trans)
        eps_index = np.where(amplitude <= EPS)[0]
        amplitude[eps_index] = EPS

        amplitude_log = np.log(amplitude)
        amplitude_log[eps_index] = 0
        residual = np.exp(
            amplitude_log - MovingAverage(
                window_size=self.window_size_
            ).fit_transform(amplitude_log)
        )

        fft_trans.real = fft_trans.real * residual / amplitude
        fft_trans.imag = fft_trans.imag * residual / amplitude
        fft_trans.real[eps_index] = 0
        fft_trans.imag[eps_index] = 0

        self.smoothed_X_ = np.fft.ifft(fft_trans)
        self.smoothed_X_ = np.absolute(self.smoothed_X_)

        return self.smoothed_X_[:X.shape[0]]
