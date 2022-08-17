import pytest

from vectorlab.data import kwargs_expansion


@pytest.mark.parametrize('kwargs_dict', [{'a': [10, 100], 'b': ['a', 'b']}])
def test_kwargs_expansion(kwargs_dict):

    print(kwargs_dict)

    kwargs_dicts = kwargs_expansion(kwargs_dict)

    assert all(
        v in kwargs_dict[k]
        for kwargs_dict_ in kwargs_dicts
        for k, v in kwargs_dict_.items()
    )
