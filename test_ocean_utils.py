import pytest
import ocean_utils
import numpy as np
import os
from shutil import rmtree
import json
import ocean

def test_reset_all_noise_flags():
  d = ocean.Depth('test_profile/1m')
  for detsum in d.detsums:
    ocean_utils.set_noisy_detsum_flag(detsum, True, 'test_profile')

  ocean_utils.reset_all_noise_flags('test_profile')

  fname = os.path.join('test_profile', 'settings', 'noisy_detsums.json')
  with open(fname, 'r') as f:
    noisy_detsums_dict = json.load(f)

  for scan_name in noisy_detsums_dict:
    for element in scan_name:
      assert not noisy_detsums_dict[scan_name][element]


def test_set_noisy_detsum_flag():
  detsum1 = ocean.Detsum('test_profile/1m/scan2D_1/detsum_A_K_norm.txt')
  detsum2 = ocean.Detsum('test_profile/1m/scan2D_2/detsum_B_K_norm.txt')

  ocean_utils.set_noisy_detsum_flag(detsum1, True, base_dir='test_profile')
  ocean_utils.set_noisy_detsum_flag(detsum2, False, base_dir='test_profile')

  fname = os.path.join('test_profile', 'settings', 'noisy_detsums.json')
  assert os.path.exists(fname)

  with open(fname, 'r') as f:
    settings = json.load(f)

  assert settings['scan2D_1']['A'] == True
  assert settings['scan2D_2']['B'] == False

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