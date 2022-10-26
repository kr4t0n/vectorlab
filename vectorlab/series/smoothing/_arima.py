"""
ARIMA is an autoregressive integrated moving average.
"""

import numpy as np

from sklearn.base import TransformerMixin

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA as ARIMAX

from ...utils._check import check_pairwise_1d_array


class ARIMA(TransformerMixin):
    r"""An autoregressive integrated moving average (ARIMA) model is a
    generalization of an autoregressive moving average (ARMA) model.

    Non-seasonal ARIMA models are generally denoted ARIMA(p, d, q) where
    parameters `p`, `d`, `q` are non-negative integers, `p` is the order
    (number of time lags) of the autoregressive model, `d` is the degree
    of differencing (the number of times the data have had past values
    subtracted), and `q` is the order of the moving average model.

    The generalized formulation of ARIMA is

        .. math::
            (1 - \sum_{i=1}^{p} \phi_{i} L^{i})(1 - L)^{d} X_{t} =
            \delta + (1 + \sum_{i=1}^{q} \theta_{i} L^{i}) \epsilon_{t}

    Parameters
    ----------
    order : {'auto', list, tuple}
        The order is in format of (p, d, q). Therefore, order should be
        a list or a tuple containing these three parameters. When order is str,
        it should be set to `auto`, such (p, d, q) will be computed when
        model is fitted.

    Attributes
    ----------
    p_ : int
        The order of the autoregressive model.
    d_ : int
        The degree of differencing.
    q_ : int
        The order of the moving average model.
    order_ : {'auto', list, tuple}
        The order in format of (p, d, q).
    X_ : array_like, shape (n_samples)
        The input array.
    delta_ : float
        The first value of input data. This value is subtracted from the input
        data to make it start from zero.
    fitted_model_ : statsmodels.tsa.arima.model.ARIMA
        The fitted ARIMA model.
    smoothed_X_ : array_like, shape (n_samples)
        The smoothed input array.

    Raises
    ------
    ValueError
        If order is not in {'auto', list, tuple}, or list or tuple length is
        not equal to three, a ValueError is raised.
    """

    def __init__(self, order='auto'):

        super().__init__()

        self.p_, self.d_, self.q_ = None, None, None

        _order_error_msg = (
            f'Order parameter is {order}. '
            'However, order only supports auto or a tuple of (p, d, q).'
        )

        if isinstance(order, str):
            if order != 'auto':
                raise ValueError(_order_error_msg)
            else:
                self.order_ = order
        elif isinstance(order, tuple) or isinstance(order, list):
            if len(order) != 3:
                raise ValueError(_order_error_msg)
            else:
                self.order_ = order
                self.p_, self.d_, self.q_ = self.order_
        else:
            raise ValueError(_order_error_msg)

        return

    def _arima_fit(self, X, Y=None):
        r"""The actual ARIMA fit function used in ARIMA class.
        This function only reads in an 1d array at a time.

        When order is set to `auto`, this function will calculate ACF and
        PACF of input array. And use them to calculate the appropriate
        `p` and `q`. The `d` value is always set to be `1`, since only
        `d` is smaller than 2 is meaningful in real life situation.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        fitted_model : statsmodels.tsa.arima.model
            The fitted ARIMA model.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.delta_ = X[0]
        X = X - self.delta_

        if self.order_ == 'auto':
            lag_acf = acf(
                X,
                nlags=min(int(10 * np.log10(len(X))), len(X) // 2 - 1),
                fft=True
            )
            lag_pacf = pacf(
                X,
                nlags=min(int(10 * np.log10(len(X))), len(X) // 2 - 1),
                method='ols'
            )

            # compute upper confidential interval
            upper_ci = 1.96 / np.sqrt(len(X))
            self.p_ = np.argmax(lag_acf < upper_ci)
            self.q_ = np.argmax(lag_pacf < upper_ci)
            self.d_ = 1

        model = ARIMAX(X, order=(self.p_, self.d_, self.q_))
        fitted_model = model.fit()

        return fitted_model

    def fit(self, X, Y=None):
        r"""Fit a ARIMA smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In ARIMA, this function copy the input X as the
        attribute and fit the ARIMA model. If `order` parameter in ARIMA
        is set to `auto`, this function will try to calculate the appropriate
        `p`, `d` and `q`. If `order` is directly specified, the ARIMA will
        use it to fit the model.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            ARIMA class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.X_ = X
        self.fitted_model_ = self._arima_fit(X, Y)

        return self

    def transform(self, X, Y=None):
        r"""Transform a ARIMA smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In ARIMA, this function actually transform
        the input data X to smooth into smoothed_X with fitted order
        (p, d, q).

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

        self.smoothed_X_ = self.fitted_model_.predict(
            start=1,
            end=len(X)
        ) + self.delta_

        return self.smoothed_X_
