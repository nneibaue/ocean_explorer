'''Helper functions for working with ocean data.'''

import numpy as np
import plotting
import json
import os
import ocean
import re

NOISY_SCANS_FILE = 'noisy_scans.json'

class FileTemplates:
  PROFILE = '^.*$'
  DEPTH = '^([0-9])+m$'
  SCAN = '^scan2D_([0-9]{1,7})$' # Assume scan number < 1E6
  DETSUM = '^detsum_([a-zA-Z]{1,2})_([A-Z])(_norm)?.txt$'

def get_scan_names(experiment_dir):
  scans = []
  for dir_or_file in os.listdir(experiment_dir):
    if re.fullmatch(FileTemplates.DEPTH, dir_or_file):
      depth_path = os.path.join(experiment_dir, dir_or_file)
      for scan_dir in os.listdir(depth_path):
        if re.fullmatch(FileTemplates.SCAN, scan_dir):
          scans.append(scan_dir)

  return scans

  
def create_noisy_scans_file(experiment_dir):
  plotting._check_or_create_settings('noisy_scans', base_dir=experiment_dir)
  fname = os.path.join(experiment_dir, 'settings', NOISY_SCANS_FILE)
  with open(fname, 'r') as f:
    noisy_scan_dict = json.load(f)

  scan_names = get_scan_names(experiment_dir)
  for name in scan_names:
    if name not in noisy_scan_dict:
      noisy_scan_dict[name] = False

  with open(fname, 'w') as f:
    json.dump(noisy_scan_dict, f)
  

def set_noisy_scan_flag(scan, isNoisy, base_dir=None):
  '''Flags a scan as noisy or not and updates the value in
     settings/noisy_scans.json.

  Args:
    scan_name: str name of scan. This should match the directory name containing
      files in that scan. E.g. 'scan2D_12345'
    is_noisy: bool whether to set `scan_name` as noisy
    base_dir: str directory containing 'settings' subdir
  '''
  assert isinstance(isNoisy, bool)
  scan.isNoisy = isNoisy
  
  plotting._check_or_create_settings('noisy_scans', base_dir=base_dir)
  fname = os.path.join(base_dir or '',  'settings', NOISY_SCANS_FILE)

  with open(fname, 'r') as f:
    settings = json.load(f)
  settings[scan.name] = isNoisy
  
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