"""
Base classes and methods for vectorlab.
"""

import pickle
import warnings
import numpy as np


class SLMixin:
    r"""Mixin class for all save and load class in Minerva.
    """

    @staticmethod
    def save(class_object, file_path):
        r"""Save the class to the specified file path using Pickle.

        Parameters
        ----------
        file_path : str
            The file path to save the class object.
        """

        pickle.dump(class_object, open(file_path, 'wb'))

        return

    @staticmethod
    def load(file_path):
        r"""Load the class from the specified file path using Pickle.

        Parameters
        ----------
        file_path : str
            The file path to load the class object.
        """

        class_object = pickle.load(open(file_path, 'rb'))

        return class_object


class KVNode(SLMixin):
    r"""The base KV node class

    The KV node represent a key-value pair node. It is often
    used in the situation with a certain value is binded to
    an entity during the computation. For example, a KV node
    can be used in the construction of a heap, to order some
    identities based on their corresponding values.

    Therefore, this class re-implement the base Python object
    comparison method,

        - __lt__
        - __le__
        - __gt__
        - __ge__
        - __eq__
        - __hash__
        - __repr__

    These methods will be used on the class attribute `value_`.
    If value is an object in the node, it should be comparable
    between the same type.

    Parameters
    ----------
    key : object
        A Python object represents the entity.
    value : int, float, object
        The value to be compared.

    Attributes
    ----------
    key_ : object
        A Python object represents the entity.
    value_ : int, float, object
        The value to be compared.
    """

    def __init__(self, key, value):

        super().__init__()

        self.key_ = key
        self.value_ = value

        return

    def __lt__(self, other):
        return self.value_ < other.value_

    def __le__(self, other):
        return self.value_ <= other.value_

    def __gt__(self, other):
        return self.value_ > other.value_

    def __ge__(self, other):
        return self.value_ >= other.value_

    def __eq__(self, other):
        return (self.key_ == other.key_) & (self.value_ == other.value_)

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):

        print_str = f'Key: {self.key_}, Value: {self.value_}'

        return print_str


class Stack(SLMixin):
    r"""The base Stack class

    The Stack represents one of the basic data structure as stack.
    It is often used in the situation when certain data in and out
    order is desired. Stack follows the order of LIFO (last in
    first out).

    Parameters
    ----------
    dtype : object type, optional
        If dtype is not None, stack elements should be strictly followed
        that type. If dtype is None, stack is flexible.

    Attributes
    ---------
    stack_ : list
        The stack to store elements.
    dtype_ : type
        The element type desired to store in the stack.
    """

    def __init__(self, dtype=None):

        super().__init__()

        self.stack_ = []
        self.dtype_ = dtype

        return

    def push(self, element):
        r"""Push a new element into the stack.

        Parameters
        ----------
        element : object
            The new element to be pushed into the stack.

        Raises
        ------
        ValueError
            When the stack's dtype is not None and new element
            type is not equal to it, a ValueError is raised.
        """

        if self.dtype_ is not None:
            if not isinstance(element, self.dtype_):
                raise ValueError(
                    f'New element type is not allowed. '
                    f'New element type is {type(element)}, '
                    f'allowed type is {self.dtype_}.'
                )

        self.stack_.append(element)

        return

    def pop(self):
        r"""Pop a element from the stack.

        Returns
        -------
        object
            The latest element in the stack.

        Raises
        ------
        ValueError
            When the stack is empty, a ValueError is raised.
        """

        if len(self) > 0:
            return self.stack_.pop(-1)
        else:
            raise ValueError('Pop from an empty stack.')

    def top(self):
        r"""Top element appeared in the stack.

        Returns
        -------
        object
            The top element in the stack is returned but not removed.
            When the stack is empty, a None value is returned.
        """

        if len(self) > 0:
            return self.stack_[-1]
        else:
            warnings.warn('The stack is empty, return a None value.')
            return None

    def __len__(self):

        return len(self.stack_)


class Queue(SLMixin):
    r"""The base Queue class

    The Queue represents one of the basic data structure as queue.
    If is often used in the situation when certain data in and out
    order is desired. Queue follows the order of FIFO (first in
    first out).

    Parameters
    ----------
    dtype : object type, optional
        If dtype is not None, queue elements should be strictly followed
        that type. If dtype is None, queue is flexible.

    Attributes
    ----------
    queue_ : list
        The queue to store elements.
    dtype_ : type
        The element type desired to store in the queue.
    """

    def __init__(self, dtype=None):

        super().__init__()

        self.queue_ = []
        self.dtype_ = dtype

        return

    def push(self, element):
        r"""Push a new element into the queue.

        Parameters
        ----------
        element : object
            The new element to be pushed into the queue.

        Raises
        ------
        ValueError
            When the queue's dtype is not None and new element
            type is not equal to it, a ValueError is raised.
        """

        if self.dtype_ is not None:
            if not isinstance(element, self.dtype_):
                raise ValueError(
                    f'New element type is not allowed. '
                    f'New element type is {type(element)}, '
                    f'allowed type is {self.dtype_}.'
                )

        self.queue_.append(element)

        return

    def pop(self):
        r"""Pop a element from the queue.

        Returns
        -------
        object
            The earliest element in the queue.

        Raises
        ------
        ValueError
            When the queue is empty, a ValueError is raised.
        """

        if len(self) > 0:
            return self.queue_.pop(0)
        else:
            raise ValueError('Pop from an empty queue.')

    def top(self):
        r"""Top element appeared in the queue.

        Returns
        -------
        object
            The top element in the queue is returned but not removed.
            When the stack is empty, a None value is returned.
        """

        if len(self) > 0:
            return self.queue_[0]
        else:
            warnings.warn('The queue is empty, return a None value.')
            return None

    def __len__(self):

        return len(self.queue_)


class Accumulator(SLMixin):
    r"""The base Accumulator class

    The Accumulator helps to maintain the number of provided attributes
    in an accumulation manner. The may be helpful when you have to
    maintain the certain number of metrics inside a loop. Using an
    accumulator can eliminate the concatenation operation to store
    the temporal results, therefore, could be not only time efficient
    but also space efficient.

    Parameters
    ----------
    n_attrs : int
        The numebr of attributes maintained inside the accumulator.
    attrs: list, np.ndarray
        The attribute name provided.

    Attributes
    ----------
    n_attrs_ : int
        The number of attributes maintained inside the accumulator.
    attrs_ : dict
        The name of attributes to fetch the data.
    attrs_stats_ : np.ndarray
        The accumulated statistical number of attributes.

    Raises
    ------
    ValueError
        If the number of attributes and the number of attribute names
        are not equal, a ValueError will be raised.
    """

    def __init__(self, n_attrs, attrs):

        if attrs is not None and len(attrs) != n_attrs:
            raise ValueError(
                f'The number of attributes and the number of provided '
                f'attribute names are not equal. The number of attributes '
                f'is {n_attrs} and the number of provided attribute '
                f'names is {len(attrs)}.'
            )

        self.n_attrs_ = n_attrs

        self.attrs_ = {attr: idx for idx, attr in enumerate(attrs)}
        self.attrs_stats_ = np.zeros(self.n_attrs_, dtype=np.float_)

        return

    def get(self, attr):
        r"""Fetch the accumulated number using the attribute name.

        Parameters
        ----------
        attr : str
            The attribute name used to fetch the stored data.

        Returns
        -------
        float
            The stored data.
        """

        return self.attrs_stats_[self.attrs_[attr]]

    def add(self, attrs_stats):
        r"""Accumulate the stored number with provided attribute new
        statistical numbers.

        Parameters
        ----------
        attrs_stats : list, np.ndarray, dict
            When attrs_stats is provided as a list or an array, it should
            have the same number of elements as the initialized number. In
            such case, the number will accumulate correspondingly. When
            attrs_stats is provided as a dictionary, it will accumulate
            the statistical number using the name of attributes.

        Raises
        ------
        ValueError
            When attrs_stats is provided as a list or an array, if the length
            provided statistical numbers is not equal to the initialized
            number, a ValueError will be raised.
        """

        if (
            isinstance(attrs_stats, list)
        ) or (
            isinstance(attrs_stats, np.ndarray)
        ):

            attrs_stats = np.array(attrs_stats)

            if self.attrs_stats_.shape != attrs_stats.shape:
                raise ValueError(
                    f'The number of adding attributes statistics is different '
                    f'from the number of elements initialized. '
                    f'The accumulator contains {self.n_attrs_} elements '
                    f'but provided with {attrs_stats.shape[0]}.'
                )

            self.attrs_stats_ += attrs_stats

        if isinstance(attrs_stats, dict):

            attrs_stats_ = np.zeros(self.n_attrs_, dtype=np.float_)

            for attr in attrs_stats:
                attrs_stats_[self.attrs_[attr]] = attrs_stats[attr]

            self.attrs_stats_ += attrs_stats_

        return
