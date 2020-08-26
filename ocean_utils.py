'''Helper functions for working with ocean data.'''

import numpy as np
import plotting
import json
import os

NOISY_SCANS_FILE = 'noisy_scans.json'

def set_noisy_scan_flag(scan_name, is_noisy, base_dir=None):
  '''Flags a scan as noisy or not.

  Args:
    scan_name: str name of scan. This should match the directory name containing
      files in that scan. E.g. 'scan2D_12345'
    is_noisy: bool whether to set `scan_name` as noisy
    base_dir: str directory containing 'settings' subdir
  '''
  assert isinstance(is_noisy, bool)
  assert isinstance(scan_name, str)

  plotting._check_or_create_settings('noisy_scans', base_dir=base_dir)
  fname = os.path.join(base_dir or '',  'settings', NOISY_SCANS_FILE)

  with open(fname, 'r') as f:
    settings = json.load(f)
  settings[scan_name] = is_noisy
  
  with open(fname, 'w') as f:
    json.dump(settings, f)



def check_groups(group, element_list, exclusive=True):
  '''Checks to see if `group` is in `element_list`.

  Args:
    group: element group string. E.g. 'Cu|Br'
    element_list: list of elements. E.g. ['Cu', 'Ca', 'Br', 'K']
    exclusive: whether or not to match groups explicitly. For example,
    'Cu|Br' matches ['Cu', 'Ca', 'Br'] with `exclusive=False`, but
    not with `exclusive=True` because the group is missing 'Ca'.

  Returns: bool. Whether or not this group was a match
  '''
  if group is np.nan:
    return False
  this_group = group.split('|')
  matches = sum([element in this_group for element in element_list])
  if exclusive:
    if matches == len(element_list) == len(this_group):
      return True
    return False
  else:
    if matches == len(element_list):
      return True
    return False