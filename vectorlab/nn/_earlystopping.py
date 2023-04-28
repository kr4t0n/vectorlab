import numpy as np

from ..base import SLMixin
from ..utils._check import check_valid_option


class EarlyStopping(SLMixin):
    r"""The EarlyStopping class to stop training in advance.

    The EarlyStopping is a form of regularization used to avoid
    overfitting when a certain metric performance stop improving.

    Parameters
    ----------
    metric_type : str
        The type of metric, ascending or descending.
    warmup : int
        The warmup iteration for training.
    laziness : int
        When certain performance stop improving, laziness allow
        it to observe few more iterations.
    tolerance : float
        Tolerance allow the metric to fluctuate in a limited range.
    """

    def __init__(self, metric_type,
                 warmup=20, laziness=4,
                 tolerance=1e-3):

        super().__init__()

        metric_type = check_valid_option(
            metric_type,
            options=['ascending', 'descending'],
            variable_name='early stopping metric type'
        )

        self.warmup_ = warmup
        self.laziness_ = laziness
        self.tolerance_ = tolerance
        self.metric_type_ = metric_type

        self.last_epoch_ = -1
        self.laziness_cnt_ = -1

        if self.metric_type_ == 'ascending':
            self.best_metric_ = -np.Inf
        elif self.metric_type_ == 'descending':
            self.best_metric_ = np.Inf

        return

    def step(self):
        r"""Ascend the epoch step.

        Returns
        -------
        self : EarlyStopping
            Return itself.
        """

        self.last_epoch_ += 1

        return self

    def record_metric(self, metric):
        r"""Record current metric value.

        Parameters
        ----------
        metric : int, float
            Current metric value.

        Returns
        -------
        self : EarlyStopping
            Return itself.
        """

        if self.metric_type_ == 'ascending':

            if metric > self.best_metric_ - self.tolerance_:
                self.laziness_cnt_ = -1
            else:
                self.laziness_cnt_ += 1

            if metric > self.best_metric_:
                self.best_metric_ = metric

        elif self.metric_type_ == 'descending':

            if metric < self.best_metric_ + self.tolerance_:
                self.laziness_cnt_ = -1
            else:
                self.laziness_cnt_ += 1

            if metric < self.best_metric_:
                self.best_metric_ = metric

        return self

    def is_done(self):
        r"""Judge if conditions are statisfied to early stopping.

        Returns
        -------
        bool
            Conditions are statisfied or not.
        """

        if (
            self.last_epoch_ >= self.warmup_
        ) and (
            self.laziness_cnt_ >= self.laziness_
        ):
            return True

        return False


class AscES(EarlyStopping):
    r"""The ascending type of EarlyStopping class to stop training in advance.

    Parameters
    ----------
    warmup : int
        The warmup iteration for training.
    laziness : int
        When certain performance stop improving, laziness allow
        it to observe few more iterations.
    tolerance : float
        Tolerance allow the metric to fluctuate in a limited range.
    """

    def __init__(self,
                 warmup=20, laziness=4,
                 tolerance=1e-3):

        super(AscES, self).__init__(
            metric_type='ascending',
            warmup=warmup, laziness=laziness,
            tolerance=tolerance
        )

        return


class DescES(EarlyStopping):
    r"""The descending type of EarlyStopping class to stop training in advance.

    Parameters
    ----------
    warmup : int
        The warmup iteration for training.
    laziness : int
        When certain performance stop improving, laziness allow
        it to observe few more iterations.
    tolerance : float
        Tolerance allow the metric to fluctuate in a limited range.
    """

    def __init__(self,
                 warmup=20, laziness=4,
                 tolerance=1e-3):

        super(DescES, self).__init__(
            metric_type='descending',
            warmup=warmup, laziness=laziness,
            tolerance=tolerance
        )

        return
