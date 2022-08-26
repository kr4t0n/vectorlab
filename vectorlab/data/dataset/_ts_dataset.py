"""
In time series dataset module, a Dataset specified for time
series data is proposed. And it can help to format and operate
your time series conveniently.
"""

import uuid
import warnings
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.preprocessing import Normalizer, MinMaxScaler

from ...base import SLMixin
from ...plot import init_plot, plot2d, show_plot
from ...utils._time import ts_to_dttz
from ...utils._check import (
    _check_ndarray, check_nd_array,
    check_valid_int, check_valid_option
)
from ...series._preprocessing import (
    format_ts, aggregate_ts, series_interpolate,
    auto_ts_step
)


class TSData(SLMixin):
    r"""TSData is a data structure to store time series data for multiple
    attributes of one entity.

    TSData also supports various preprocessing operation to directly apply
    to the attributes stored as `numpy.ndarray`, which supports all kinds
    of matrix manipulation to achieve the most efficient calculation.

    Currently, these preprocessing process includes formatting time stamp,
    and interpolation. And directly use preprocess function will try to
    do stuff automatically.

    Attributes
    ----------
    entity_ : str
        The entity identification of this TSData.
    ts_ : array_like, shape (n_samples)
        The time stamps of the entity.
    attr_names_ : array_like, shape (n_attrs)
        The attributes name of the entity.
    attrs_ : array_like, shape (n_attrs, n_samples)
        The attributes of the entity.
    n_samples_ : int
        The number of samples contained in the TSData.
    n_attrs_ : int
        The number of attributes contained in the TSData.
    """

    def __init__(self):

        super(TSData, self).__init__()

        return

    def _validate(self):
        r"""The data validation for TSData.

        In this function, time stamps and attributes data of TSData are
        checked. Their dimension information and shape of matrix are
        verified. After checking, `n_samples_` and `n_attrs_` attributes
        of class are updated.

        ***NOTICE***: It is strongly recommended that after manipulation
        to the data in TSData, this function is applied to ensure data
        consistency inside.
        """

        assert self.ts_.ndim == 1
        assert self.attr_names_.ndim == 1
        assert self.ts_.shape[0] == self.attrs_.shape[1]
        assert self.attr_names_.shape[0] == self.attrs_.shape[0]

        self.n_samples_ = self.ts_.shape[0]
        self.n_attrs_ = self.attr_names_.shape[0]

        return

    def from_pandas(self, df, entity=None, timestamp=None):
        r"""Convert TSData from pandas.DataFrame.

        This function load data from a `pandas.DataFrame`. TSData aims
        to be store data for only one and one only entity, and it is
        highly recommended to declare the entity name. If entity
        parameter is column, it will try to infer the entity name from
        such column.

        Also a time stamp columns is also need to be pointed out. If
        not, it will automatically infer the time stamp in an ascending
        integer order begin from 0.

        Besides the entity and time stamp columns, the rest of the
        data frame will all be treated as the attributes, which will
        infer the attributes names from the columns names and set the
        attribute data accordingly.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be loaded.
        entity : str, optional
            The entity name or entity column to be inferred.
        timestamp : str, optional
            The time stamp column to be inferred.

        Raises
        ------
        ValueError
            If the specified entity column containing more than one
            values inside, a ValueError is raised since TSData structure
            only supports data storage for one entity.
        """

        if entity is None:
            self.entity_ = str(uuid.uuid4())
            warnings.warn(
                f'There is no entity information being specified, TSData will '
                f'randomly generate a id for your entity, your id is '
                f'{self.entity_}. If you wish to use a given entity name, '
                f'you can either pass a name to entity parameter or '
                f'a column name contained entity name.'
            )
        elif entity in df.columns:
            if df[entity].drop_duplicates().shape[0] > 1:
                raise ValueError(
                    f'TSData object only support one entity, there are '
                    f'{df[entity].drop_duplicates().shape[0]} entities in the '
                    f'DataFrame. You can use TSDataset instead.'
                )
            else:
                self.entity_ = df[entity].iloc[0]
                df = df[set(df.columns) - set([self.entity_])]
        else:
            self.entity_ = entity

        if timestamp is None:
            warnings.warn(
                'TSData time stamp column is not specified. It will '
                'automatically add a series of time stamps according '
                'to ascending order of integers from 0.'
            )

            df['_ts'] = np.arange(df.shape[0])
            timestamp = '_ts'
        else:
            if timestamp not in df.columns:
                raise ValueError(
                    f'TSData cannot find specified time stamp column '
                    f'in DataFrame. Potential columns are: {df.columns}'
                )

        df = df.sort_values(by=timestamp)

        self.ts_ = df[timestamp].to_numpy()

        self.attr_names_ = df.columns.to_numpy()
        self.attr_names_ = self.attr_names_[self.attr_names_ != timestamp]

        self.attrs_ = df[self.attr_names_].to_numpy().T

        # Validation
        self._validate()

        return

    def to_pandas(self, with_entity=True):
        r"""Convert TSData to pandas.DataFrame.

        This function pour data to a `pandas.DataFrame`. In this way, you
        can conveniently store TSData into a `csv` file using the interface
        provided by pandas.

        Using the with_entity, it can add a column name as `entity` and a
        column name as `timestamp` will also be added automatically.

        Parameters
        ----------
        with_entity : bool, optional
            Whether to add a entity column or not.

        Returns
        -------
        pandas.DataFrame
            The output pandas.DataFrame.
        """

        if with_entity:
            df_columns = np.hstack(
                (
                    ['entity', 'timestamp'],
                    self.attr_names_
                )
            )
            df_data = np.vstack(
                (
                    np.repeat(self.entity_, self.n_samples_),
                    self.ts_,
                    self.attrs_
                )
            )
        else:
            df_columns = np.hstack(
                (
                    ['timestamp'],
                    self.attr_names_
                )
            )
            df_data = np.vstack(
                (
                    self.ts_,
                    self.attrs_
                )
            )

        df = pd.DataFrame(df_data.T, columns=df_columns)

        return df

    def add_attr(self, attr_name, attr):
        r"""Add attribute data into TSData.

        This function provides an attribute adding operation to TSData.
        When certain attribute data is transformed, your can use this
        function to add the attribute data and attribute name into TSData.

        Parameters
        ----------
        attr_name : str
            The added attribute name.
        attr : array_like, shape (n_samples, ) or (1, n_samples)
            The added attribute data.

        Raises
        ------
        ValueError
            When the added attribute data shape is not (n_samples, )
            or (1, n_samples), a ValueError is raised.
        """

        attr = check_nd_array(attr, n=1)

        if attr.shape != (self.n_samples_, ):
            raise ValueError(
                f'Cannot add attribute {attr_name} with data shape '
                f'{attr.shape}. Support data shape is ({self.n_samples_},).'
            )

        if attr_name in self.attr_names_:
            warnings.warn(
                f'Attribute {attr_name} already exists. This operation '
                f'will replace the original data.'
            )
        else:
            self.attr_names_ = np.hstack(
                (
                    self.attr_names_,
                    [attr_name]
                )
            )
            self.attrs_ = np.vstack(
                (
                    self.attrs_,
                    np.empty(self.n_samples_)
                )
            )

        self.attrs_[self.attr_names_ == attr_name, :] = attr

        self._validate()

        return

    def del_attr(self, attr_name):
        r"""Delete attribute data from TSData.

        This function provides an attribute deleting operation from TSData.
        When certain attribute data is no longer needed, your can use this
        function to delete the attribute data and attribute name from TSData.

        Parameters
        ----------
        attr_name : str
            The attribute to be deleted.

        Raises
        ------
        ValueError
            When the attribute name to be deleted does not exist in stored
            attribute names, a ValueError is raised.
        """

        attr_name = check_valid_option(
            attr_name,
            options=self.attr_names_,
            variable_name='attr_name'
        )

        self.attrs_ = self.attrs_[self.attr_names_ != attr_name]
        self.attr_names_ = self.attr_names_[self.attr_names_ != attr_name]

        self._validate()

        return

    def format_timestamp(self, step):
        r"""Formatting time stamp as well as attributes data in TSData.

        This function mainly tries to format time stamp in TSData, to
        make the time stamp into an arithmetic sequence. The full
        documentation can be found at minerva_lib.preprocessing.series.

        Parameters
        ----------
        step : int, float
            The step between two time stamps.
        """

        self.ts_, self.attrs_ = format_ts(
            self.ts_, self.attrs_,
            step=step
        )

        self._validate()

        return

    def aggregate_timestamp(self, step, agg_type):
        r"""Aggregating time stamp as well as attributes data in TSData.

        This function mainly tries to aggregate time stamp in TSData, to
        make the time stamp into an arithmetic sequence. The full
        documentation can be found at minerva_lib.preprocessing.series.

        Parameters
        ----------
        step : int, float
            The step between two time stamps.
        agg_type : str
            The aggregation method used to aggregate the time stamp.
        """

        self.ts_, self.attrs_ = aggregate_ts(
            self.ts_, self.attrs_,
            step=step,
            agg_type=agg_type
        )

        self._validate()

        return

    def interpolate(self, kind):
        r"""Interpolate and fill in the missing attributes data in TSData.

        This function mainly tries to interpolate and fill in the `np.nan`
        value with certain methods, to make sure that there is no missing
        value in the attribute data. The full documentation can be found
        at minerva_lib.preprocessing.series.

        Parameters
        ----------
        kind : str
            The method used in interpolation.
        """

        self.attrs_ = series_interpolate(
            self.ts_, self.attrs_,
            kind=kind
        )

        self._validate()

        return

    def standardize(self):
        r"""The automatic standardization process to TSData.

        This function mainly tries to automatically do some standardization
        process to TSData. It will calculate the auto time step to be used
        in the format_timestamp. And we also use `cubic` as the method in
        interpolation, since this method is kind of enough in real life
        situation.
        """

        step = auto_ts_step(self.ts_, eps=4)
        kind = 'cubic'

        self.format_timestamp(step=step)
        self.interpolate(kind=kind)

        self._validate()

        return

    def preprocessing(self, method):
        r"""Preprocessing the attribute data in TSData.

        This preprocessing is recommended to be executed after the
        standardization process of TSData. Mainly, preprocessing is used to
        transform the attribute data into certain satisfied range to be used
        in machine learning algorithm. Currently, there are two methods
        supported, `normalize` and  `minmax`.

        Parameters
        ----------
        method : str
            The method to be used in preprocessing.
        """

        method = check_valid_option(
            method,
            options=['normalize', 'minmax'],
            variable_name='TSData preprocessing method'
        )

        if method == 'normalize':
            self.scaler_ = Normalizer().fit(self.attrs_.T)
        elif method == 'minmax':
            self.scaler_ = MinMaxScaler().fit(self.attrs_.T)

        self.attrs_ = self.scaler_.transform(self.attrs_.T).T

        self._validate()

        return

    def show(self, attr_names=None, compress=False, show_date=True):
        r"""The visualization of TSData.

        This function provides visualization of the attributes stored in
        TSData. In this function, we will use one single figure to plot
        multiple attributes, therefore, as a result, there will be n_attrs
        subplots in the way of (n_attrs, 1, ).

        ***NOTICE***: It will be not recommended to directly show the TSData
        when you have a large set of attributes, it will make your figure look
        quite squeezed. You could use `compress` parameter to plot multiple
        attributes inside one figure.

        Parameters
        ----------
        attr_names : str, list
            The attributes to be plotted, if not provided, all attributes
            will be plotted.
        compress : bool
            When compress is True, all attributes will be plotted in one.
            When compress is False, every attribute will be plotted in its
            own subgraph.
        show_date : bool
            When show_date is True, the timestamp will be converted into date.
            When show_data is False, the raw timestamp will be shown.

        Raises
        ------
        ValueError
            If input attr_names has attributes which are not existed in TSData,
            a ValueError is raised.
        """

        if attr_names is None:
            attr_names = self.attr_names_
        elif isinstance(attr_names, str):
            attr_names = [attr_names]

        attr_names = _check_ndarray(attr_names)

        # validation of attribute names
        if not np.all(np.isin(attr_names, self.attr_names_)):

            invalid_attr_names = attr_names[
                ~np.isin(attr_names, self.attr_names_)
            ]

            raise ValueError(
                f'The input attribute names, {attr_names} is not valid. '
                f'The attributes, {invalid_attr_names} do(es) not exist in '
                f'{self.attr_names_}.'
            )

        n_attrs = attr_names.shape[0]

        ts = self.ts_
        if show_date:
            ts = np.vectorize(ts_to_dttz)(ts)

        attrs = self.attrs_[np.isin(self.attr_names_, attr_names), :]

        # plot the attributes

        # original height and width
        height, width = 5, 5

        # converted heigth and width
        if not compress:
            height = height * n_attrs

        width = width * int(np.log(self.n_samples_))

        init_plot(height=height, width=width)

        if compress:

            plot2d(
                x=np.tile(ts, n_attrs),
                y=attrs.ravel(),
                categories=np.repeat(attr_names, self.n_samples_),
                lines='-',
                legend_title='attributes'
            )

        else:

            for i in range(n_attrs):
                plot2d(
                    x=ts,
                    y=attrs[i].ravel(),
                    categories=np.repeat(attr_names[i], self.n_samples_),
                    lines='-',
                    legend_title='attributes',
                    ax_pos=(n_attrs, 1, i + 1)
                )

        show_plot()

        return

    def __getitem__(self, key):
        r"""The built-in __getitem__ method in TSData.

        For TSData, __getitem__ is mainly focused on returning the desired
        attributes data.

        When the key parameter in __getitem__ is a slice, it will try to
        slice the time stamp and attributes data according, and the return
        a subset of the original TSData.

        When the key parameter in __getitem__ is an integer, it will try to
        return the `i-th` sample attributes inside TSData.

        When the key parameter in __getitem__ is a string, it will try to
        treat the key as a attribute name and return the corresponding
        attribute data.

        Parameters
        ----------
        key : slice
            The slice to be used to obtain a subset of TSData.

        Returns
        -------
        TSData, array_like
            The subset TSData of the original one.

        Raises
        ------
        ValueError
            It the key parameter in __getitem__ is int, but it is larger
            than the number of samples, a ValueError is raised.

            If the key parameter in __getitem__ is string but not appeared
            in the provided attributes, a ValueError is raised.

            If the key parameter in __getitem__ is not slice, int or str,
            a ValueError is raised.
        """

        if isinstance(key, slice):

            other = deepcopy(self)

            other.ts_ = other.ts_[key]
            other.attrs_ = other.attrs_[:, key]

            other._validate()

            return other
        elif isinstance(key, int):

            key = check_valid_int(
                key,
                lower=0, upper=self.n_samples_ - 1,
                variable_name='sample index'
            )
            attr = self.attrs_[:, key]
            attr_data = dict(zip(self.attr_names_, attr))

            return attr_data
        elif isinstance(key, str):

            key = check_valid_option(
                key,
                options=self.attr_names_,
                variable_name='attribute name'
            )
            attr_data = self.attrs_[self.attr_names_ == key, :].ravel()

            return attr_data
        else:
            raise ValueError(
                f'TSDatas currently does support using {key} to get '
                f'items.'
            )

    def __repr__(self):
        r"""The built-in __repr__ method in TSData.

        For TSData, __repr__ is mainly focused on showing some statistical
        number of TSData.

        Currently, __repr__ will show the entity name, number of samples,
        number of attributes and attributes data accordingly.

        Returns
        -------
        repr_string : str
            The string to be shown when called by print function.
        """

        repr_string = [
            f'[TSData]',
            f'  Entity: {self.entity_}',
            f'    n_samples: {self.n_samples_}',
            f'    n_attrs  : {self.n_attrs_}',
            f'    attrs    : {self.attr_names_}'
        ]

        repr_string = '\n'.join(repr_string)

        return repr_string
