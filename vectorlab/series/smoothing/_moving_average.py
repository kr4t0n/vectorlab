"""
Moving average is a straight forward smoothing method.
"""

import warnings
import numpy as np

from sklearn.base import TransformerMixin

from ...utils._check import (
    check_valid_int,
    check_valid_float,
    check_pairwise_1d_array
)


class MovingAverage(TransformerMixin):
    r"""The moving average (MA) is a simple technical analysis
    tool that smooths out data by creating a constantly updated
    average data.

    A simple moving average is the unweighted mean of the previous
    `n` data-points.

        .. math::
            \overline{p}_{n} = \frac{p_1 + p_2 + \dots + p_n}{n} =
            \frac{1}{n} \sum_{i=1}^{n}p_i


    Parameters
    ----------
    window_size : int
        How many previous data-points used to calculate
        the unweighted mean value.

    Attributes
    ----------
    window_size_ : int
        How many previous data-points used to calculate
        the unweighted mean value.
    X_ : array_like, shape (n_samples)
        The input array.
    smoothed_X_ : array_like, shape (n_samples)
        The smoothed input array.

    Raises
    ------
    ValueError
        If window_size is not an integer larger than 0,
        a ValueError is raised.
    """

    def __init__(self, window_size):

        super().__init__()

        window_size = check_valid_int(
            window_size,
            lower=1,
            variable_name='window_size'
        )

        self.window_size_ = window_size

        return

    def fit(self, X, Y=None):
        r"""Fit a moving average smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In moving average, this function does nothing,
        just copy the input X as the attribute.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            MovingAverage class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.X_ = X

        return self

    def _moving_average(self, X, Y=None):
        r"""The actual moving average function used in MovingAverage class.
        This function only reads in an 1d array at a time.

        Parameters
        ----------
        X : array_like, shape (n_smaples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        smoothed_X : array_like, shape (n_smaples)
            The smoothed input array.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        if self.window_size_ > X.shape[0]:
            warnings.warn(
                f'Window size is smaller than the length of input X. '
                f'Window size is {self.window_size_}, '
                f'while length of input is {X.shape[0]}. '
                f'Window size will change to the length of {X.shape[0]}.'
            )

            window_size = X.shape[0]
        else:
            window_size = self.window_size_

        smoothed_X = np.cumsum(X, dtype=np.float_)
        smoothed_X[window_size:] = \
            smoothed_X[window_size:] - smoothed_X[:-window_size]
        smoothed_X[window_size:] = \
            smoothed_X[window_size:] / window_size

        for i in range(0, window_size):
            smoothed_X[i] /= (i + 1)

        return smoothed_X

    def transform(self, X, Y=None):
        r"""Transform a moving average smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In moving average, this function actually transform
        the input data X to smooth into smoothed_X with preset window size.

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

        self.smoothed_X_ = self._moving_average(X, Y)

        return self.smoothed_X_


class WeightedMovingAverage(TransformerMixin):
    r"""The weighted moving average (WMA) is a simple technical analysis
    tool that smooths out data by creating a constantly updated
    average data.

    A weighted moving average is weighted mean of the previous
    `n` data-points.

        .. math::
            \overline{p}_{n} = \frac{1 \cdot p_1 + 2 \cdot p_2 +
            \dots + n \cdot p_n}{1 + 2 + \dots + n} =
            \frac{1}{1 + 2 + \dots + n} \sum_{i=1}^{n}i \cdot p_i

    Parameters
    ----------
    window_size : int
        How many previous data-points used to calculate
        the weighted mean value.

    Attributes
    ----------
    window_size_ : int
        How many previous data-points used to calculate
        the weighted mean value.
    X_ : array_like, shape (n_samples)
        The input array.
    smoothed_X_ : array_like, shape (n_samples, n_features_new)
        The smoothed input array.

    Raises
    ------
    ValueError
        If window_size is not an integer larger than 0,
        a ValueError is raised.
    """

    def __init__(self, window_size):

        super().__init__()

        window_size = check_valid_int(
            window_size,
            lower=1,
            variable_name='window_size'
        )

        self.window_size_ = window_size

        return

    def fit(self, X, Y=None):
        r"""Fit a weighted moving average smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In weighted moving average, this function does nothing,
        just copy the input X as the attribute.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            WeightedMovingAverage class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.X_ = X

        return self

    def _weighted_moving_average(self, X, Y=None):
        r"""The actual weighted moving average function used in
        WeightedMovingAverage class. This function only reads in an
        1d array at a time.

        Parameters
        ----------
        X : array_like, shape (n_smaples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        smoothed_X : array_like, shape (n_smaples)
            The smoothed input array.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        if self.window_size_ > X.shape[0]:
            warnings.warn(
                f'Window size is smaller than the length of input X. '
                f'Window size is {self.window_size_}, '
                f'while length of input is {X.shape[0]}. '
                f'Window size will change to the length of {X.shape[0]}.'
            )

            window_size = X.shape[0]
        else:
            window_size = self.window_size_

        smoothed_X = X * window_size * 1.0
        for i in range(1, window_size):
            smoothed_X += np.pad(X, (i, 0))[:X.shape[0]] * (window_size - i)

        smoothed_X[window_size:] = \
            smoothed_X[window_size:] / ((1 + window_size) * window_size / 2)

        for i in range(0, window_size):
            smoothed_X[i] /= ((window_size - i + window_size) * (i + 1) / 2)

        return smoothed_X

    def transform(self, X, Y=None):
        r"""Transform a weighted moving average smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In weighted moving average, this function
        actually transform the input data X to smooth into smoothed_X
        with preset window size.

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

        self.smoothed_X_ = self._weighted_moving_average(X)

        return self.smoothed_X_


class ExpWeightedMovingAverage(TransformerMixin):
    r"""The exponential weighted moving average (EWMA) is a
    simple technical analysis tool that smooths out data by
    creating a constantly update average data.

    A exponential weighted moving average is a first-order
    infinite impulse response filter that applies weighting
    factors which decreases exponentially.

        .. math::
            \overline{p}_{n} = \alpha^{n} \cdot p_1 + \alpha^{n - 1} \cdot p_2
            + \dots + \alpha^{1} \cdot p_n =
            \sum_{i=1}^{n}\alpha^{n - i + 1} \cdot p_i

    Parameters
    ----------
    alpha : float
        The weighting factor used in calculation.

    Attributes
    ----------
    alpha_ : float:
        The weighting factor used in calculation.
    X_ : array_like, shape (n_samples)
        The input array.
    smoothed_X_ : array_like, shape (n_samples)
        The smoothed input array.

    Raises
    ------
    ValueError
        If alpha is a float larger than 1 or smaller than 0,
        a ValueError is raised.
    """

    def __init__(self, alpha):

        super().__init__()

        alpha = check_valid_float(
            alpha,
            lower=0, upper=1,
            variable_name='alpha'
        )

        self.alpha_ = alpha

        return

    def fit(self, X, Y=None):
        r"""Fit a exponential weighted moving average smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In exponential weighted moving average, this
        function does nothing, just copy the input X as the attribute.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            ExpWeightedMovingAverage class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.X_ = X

        return self

    def _exp_weighted_moving_average(self, X, Y=None):
        r"""The actual exponential weighted moving average function used in
        ExpWeightedMovingAverage class. This function only reads in an
        1d array at a time.

        Parameters
        ----------
        X : array_like, shape (n_smaples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        smoothed_X : array_like, shape (n_smaples)
            The smoothed input array.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        alpha_add = np.frompyfunc(
            lambda x, y: self.alpha_ * x + (1 - self.alpha_) * y,
            2,
            1
        )
        smoothed_X = alpha_add.accumulate(X, dtype=np.object_).astype(np.float_)

        return smoothed_X

    def transform(self, X, Y=None):
        r"""Transform a exponential weighted moving average smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In exponential weighted moving average, this function
        actually transform the input data X to smooth into smoothed_X
        with preset alpha.

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

        self.smoothed_X_ = self._exp_weighted_moving_average(X)

        return self.smoothed_X_
