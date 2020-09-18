'''Helper functions for working with ocean data.'''

import numpy as np
import plotting
import json
import os
import re

NOISY_DETSUMS_FILE = 'noisy_detsums.json'

class FileTemplates:
  PROFILE = '^(.*)_profile$'
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

def create_noisy_detsums_file(experiment_dir):
  plotting._check_or_create_settings('noisy_detsums', base_dir=experiment_dir)
  fname = os.path.join(experiment_dir, 'settings', NOISY_DETSUMS_FILE)
  with open(fname, 'r') as f:
    noisy_detsum_dict = json.load(f)
  for dir_or_file in os.listdir(experiment_dir):
    if re.fullmatch(FileTemplates.DEPTH, dir_or_file):
      depth_path = os.path.join(experiment_dir, dir_or_file)
      for scan_dir in os.listdir(depth_path):
        if re.fullmatch(FileTemplates.SCAN, scan_dir):
          scan_path = os.path.join(depth_path, scan_dir)
          noisy_detsum_dict[scan_dir] = {}
          for detsum in os.listdir(scan_path):
            match = re.fullmatch(FileTemplates.DETSUM, detsum)
            if match and match.group(1) not in noisy_detsum_dict[scan_dir]:
              noisy_detsum_dict[scan_dir][match.group(1)] = False

  with open(fname, 'w') as f:
    json.dump(noisy_detsum_dict, f)
  

def reset_all_noise_flags(profile_dir):
  '''Sets noise flag on all detsums to False.

  Args:
    profile_dir: directory containing a profile
  '''
  plotting._check_or_create_settings('noisy_detsums', base_dir=profile_dir)
  fname = os.path.join(profile_dir, 'settings', NOISY_DETSUMS_FILE)
  with open(fname, 'r') as f:
    noisy_detsum_dict = json.load(f)

  for scan in noisy_detsum_dict:
    for element in scan:
      noisy_detsum_dict[scan][element] = False

  with open(fname, 'w') as f:
    json.dump(noisy_detsum_dict, f)
  

def set_noisy_detsum_flag(detsum, isNoisy, base_dir=None):
  '''Flags a detsum as noisy or not and updates the value in
     settings/noisy_detsums.json.

  Args:
    detsum: Detsum object 
    is_noisy: bool whether to set `scan_name` as noisy
    base_dir: str directory containing 'settings' subdir
  '''
  assert isinstance(isNoisy, bool)
  detsum.isNoisy = isNoisy
  
  plotting._check_or_create_settings('noisy_detsums', base_dir=base_dir)
  fname = os.path.join(base_dir or '',  'settings', NOISY_DETSUMS_FILE)

  with open(fname, 'r') as f:
    settings = json.load(f)
  settings[detsum.scan_name][detsum.element] = isNoisy
  
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