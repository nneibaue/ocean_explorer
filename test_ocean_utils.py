import pytest
import ocean_utils
import numpy as np



@pytest.mark.parametrize('group,expected', [
                           ('a|b', False),
                           ('c|d', False),
                           ('d|c|b|a', True),  # Test different ordering
                           ('c|a|b|d', True),
                           ('c|a|b|f|g', False),
                           (np.nan, False)]) # Test NaN vals
def test_ocean_utils_exclusive(group, expected):
 element_list = ['a', 'b', 'c', 'd']
 assert ocean_utils.check_groups(group, element_list, exclusive=True) == expected


@pytest.mark.parametrize('group,expected', [
                           ('a|b', True),
                           ('c|d', True),
                           ('d|c|b|a', True),  # Test different ordering
                           ('c|a|b|d', True),
                           ('c|a|b|f|g', False),
                           (np.nan, False)]) # Test NaN vals
def test_ocean_utils_not_exclusive(group, expected):
 element_list = ['a', 'b', 'c', 'd']
 assert ocean_utils.check_groups(group, element_list, exclusive=False) == expected