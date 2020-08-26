import pytest
import ocean_utils
import numpy as np
import os
from shutil import rmtree
import json

def test_set_noisy_scan_flag():
  ocean_utils.set_noisy_scan_flag('test_scan_true', True, base_dir='test_profile')
  ocean_utils.set_noisy_scan_flag('test_scan_false', False, base_dir='test_profile')

  fname = os.path.join('test_profile', 'settings', 'noisy_scans.json')
  assert os.path.exists(fname)
  with open(fname, 'r') as f:
    settings = json.load(f)

  assert settings['test_scan_true']
  assert not settings['test_scan_false']

  rmtree('test_profile/settings')

@pytest.mark.parametrize('group,expected', [
	                         ('b', False),  # Test single groups
                           ('a|b', False),
                           ('c|d', False),
                           ('d|c|b|a', True),  # Test different ordering
                           ('c|a|b|d', True),
                           ('c|a|b|f|g', False),
                           (np.nan, False)]) # Test NaN vals
def test_check_groups_exclusive(group, expected):
 element_list = ['a', 'b', 'c', 'd']
 assert ocean_utils.check_groups(group, element_list, exclusive=True) == expected


@pytest.mark.parametrize('group,expected', [
	                         ('b', False),  # Test single groups
                           ('a|b', True),
                           ('c|d', False),
                           ('d|c|b|a', True),  # Test different ordering
                           ('c|a|b|d', True),
                           ('c|a|b|f|g', True),
                           (np.nan, False)]) # Test NaN vals
def test_check_groups_not_exclusive(group, expected):
 element_list = ['a', 'b']
 assert ocean_utils.check_groups(group, element_list, exclusive=False) == expected