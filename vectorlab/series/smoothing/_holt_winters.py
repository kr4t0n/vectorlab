"""
Holt winters is an exponential smoothing.
"""

from sklearn.base import TransformerMixin

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ...utils._check import (
    check_valid_int, check_valid_option, check_pairwise_1d_array
)


class HoltWinters(TransformerMixin):
    r"""Holt Winters smoothing is a triple exponential smoothing.
    Triple exponential smoothing applies exponential smoothing
    three times, which is commonly used when there are three
    high frequency signals to be removed from a time series under
    study. There are different types of seasonality: `multiplicative`
    and `additive` in nature, much like addition and multiplication
    are basic operations in mathematics.

    For a cycle of seasonal change of length `L`. The output of
    algorithm is written as :math:`F_{t+m}`, an estimate of the value
    of :math:`x_{t+m}` at time :math:`t+m > 0` based on the raw data up
    to time `t`.

    The multiplicative seasonality is given by,
        .. math::
            \begin{gather*}
            s_0 = x_0 \\
            s_t = \alpha \frac{x_t}{c_{t-L}} +
            (1 - \alpha)(s_{t-1} + b_{t-1}) \\
            b_t = \beta(s_t - s_{t-1}) + (1 - \beta)b_{t-1} \\
            c_t = \gamma \frac{x_t}{s_t} + (1 - \gamma)c_{t-L} \\
            F_{t+m} = (s_t + mb_t)c_{t-L+1+(m-1) \mod L}
            \end{gather*}

    where :math:`\alpha (0 \leq \alpha \leq 1)` is the `data smoothing
    factor`, :math:`\beta (0 \leq \beta \leq 1)` is the `trend smoothing
    factor`, and :math:`\gamma (0 \leq \gamma \leq 1)` is the `seasonal
    change smoothing factor`.

    The general formula for the initial trend estimate `b` is
        .. math::
            b_0 = \frac{1}{L}(\frac{x_{L+1}-x_1}{L} + \frac{x_{L+2}-x_2}{L} +
            \dots + \frac{x_{L+L}-x_L}{L})

    Triple exponential smoothing with additive seasonality is given by
        .. math::
            \begin{gather*}
            s_0 = x_0 \\
            s_t = \alpha (x_t - c_{t-L}) +
            (1 - \alpha)(s_{t-1} + b_{t-1}) \\
            b_t = \beta(s_t - s_{t-1}) + (1 - \beta)b_{t-1} \\
            c_t = \gamma (x_t - s_{t-1} - b_{t-1}) + (1 - \gamma)c_{t-L} \\
            F_{t+m} = s_t + mb_t + c_{t-L+1+(m-1) \mod L}
            \end{gather*}

    Parameters
    ----------
    trend : str, {'add', 'mul', 'additive', 'multiplicative'}, optional
        Type of trend component.
    seasonal : str, {'add', 'mul', 'additive', 'multiplicative'}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle.
    initialization_method : str, {'estimated', 'heuristic',
                                  'legacy-heuristic', 'known'}, optional
        Method for initialize the recursions.


    Attributes
    ----------
    trend_ : str, {'add', 'mul', 'additive', 'multiplicative'}, optional
        Type of trend component.
    seasonal_ : str, {'add', 'mul', 'additive', 'multiplicative'}, optional
        Type of seasonal component.
    seasonal_periods_ : int, optional
        The number of periods in a complete seasonal cycle.
    initialization_method_ : str, {'estimated', 'heuristic',
                                   'legacy-heuristic', 'known'}, optional
        Method for initialize the recursions.
    X_ : array_like, shape (n_samples)
        The input array.
    fitted_model_ : statsmodels.tsa.holtwinters.ExponentialSmoothing
        The fitted Holt Winters model.
    smoothed_X_ : array_like, shape (n_samples)
        The smoothed input array.
    """

    def __init__(self, trend=None,
                 seasonal=None, seasonal_periods=None,
                 initialization_method='estimated'):

        super().__init__()

        if trend:
            trend = check_valid_option(
                trend,
                options=['add', 'mul', 'additive', 'multiplicative'],
                variable_name='trend'
            )
        if seasonal:
            seasonal = check_valid_option(
                seasonal,
                options=['add', 'mul', 'additive', 'multiplicative'],
                variable_name='seasonal'
            )
        if seasonal_periods:
            seasonal_periods = check_valid_int(
                seasonal_periods,
                lower=1,
                variable_name='seasonal_periods'
            )
        if initialization_method:
            initialization_method = check_valid_option(
                initialization_method,
                options=['estimated', 'heuristic', 'legacy-heuristic', 'known'],
                variable_name='initialization_method'
            )

        self.trend_ = trend
        self.seasonal_ = seasonal
        self.seasonal_periods_ = seasonal_periods
        self.initialization_method_ = initialization_method

        return

    def _holt_winters_fit(self, X, Y=None):
        r"""The actual Holt Winters fit function used in HoltWinters class.
        This function only reads in an 1d array at a time.

        In this function, it will use the input data `X`, and preset parameters
        to fit a Holt Winters model.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        fitted_model : statsmodels.tsa.holtwinters.ExponentialSmoothing
            The fitted Holt Winters model.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        model = ExponentialSmoothing(
            X,
            trend=self.trend_,
            seasonal=self.seasonal_,
            seasonal_periods=self.seasonal_periods_,
            initialization_method=self.initialization_method_
        )
        fitted_model = model.fit()

        return fitted_model

    def fit(self, X, Y=None):
        r"""Fit a Holt Winters smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In Holt Winters, this function copy the input X as the
        attribute and fit the Holt Winters model.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            The input array.
        Y : Ignored
            Not used, present for scikit-learn API consistency by convention.

        Returns
        -------
        self : object
            HoltWinters class object itself.
        """

        X, Y = check_pairwise_1d_array(X, Y)

        self.X_ = X
        self.fitted_model_ = self._holt_winters_fit(X, Y)

        return self

    def transform(self, X, Y=None):
        r"""Transform a Holt Winters smoothing

        All the input data is provided by X, while Y is set to None
        to be ignored. In Holt Winters, this function actually transform
        the input data X to smooth into smoothed_X with fitted model.

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
        )

        return self.smoothed_X_
