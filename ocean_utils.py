'''Helper functions for working with ocean data.'''

import numpy as np


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