'''Core functions and classes for ETSP data explorer'''


class Detsum:
  file_template = '^detsum_([a-zA-Z]{1,2})_([A-Z])(_norm)?.txt$'
  def __init__(self, path):
    self.filename = path.split('/')[-1]
    self.scan_name = path.split('/')[-2]
    self.depth = path.split('/')[-3]
    if re.search('_norm.txt', self.filename):
      self.normalized = True
    else:
      self.normalized = False
    if not re.fullmatch(Detsum.file_template, self.filename):
      raise NameError(f'{self.filename} is not a valid name for Detsum')
    self.element = re.search(Detsum.file_template, self.filename).group(1)
    self.orbital = re.search(Detsum.file_template, self.filename).group(2)

    self._data_raw = np.array(np.genfromtxt(path))

    # Masks can only be added using self.add_mask
    self._masks = []

  @property
  def data(self):
    return self._apply_masks()

  @property
  def total_counts(self):
    return np.nansum(self.data)

  @property
  def mask(self):
    '''Returns full mask for this Detsum.'''
    if not self._masks:
      return np.ones_like(self._data_raw, dtype=bool)
    return np.logical_and.reduce(self._masks)

  def _apply_masks(self):
    data = self._data_raw.copy()
    if not self._masks:
      return data
    data[~self.mask] = np.nan
    return data

  def add_mask(self, mask):
    '''Removes points that aren't in the mask'''
    if mask.shape != self.data.shape:
      raise ValueError(f'Trying to add mask of shape {mask.shape} to Detsum of shape {self.data.shape}')
    self._masks.append(mask)

  def reset_mask(self):
    self._masks = []

  def plot(self, raw=False,
           figsize=(7, 7),
           ax=None,
           **imshow_kwargs):

    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)
    else:
      fig = ax.figure
    if raw:
      ax.imshow(self._data_raw, **imshow_kwargs)
    else:
      ax.imshow(self.data, aspect='equal', **imshow_kwargs)
    ax.set_title(self)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

  # def __add__(self, other):
  #   return self._create_default_mask(tot)

  def __repr__(self):
    return (f'Detsum(element: {self.element}, ' 
            f'orbital: {self.orbital}, ' 
            f'normalized: {self.normalized}, '
            f'total counts: {self.total_counts}')


