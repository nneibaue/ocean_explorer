'''Core functions and classes for ETSP data explorer'''
import altair as alt
import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
from plotting import encode_matplotlib_fig, _check_or_create_settings
import ocean_utils
import json

DRIVE_BASE = '/content/gdrive/My Drive'
PROFILE_FILE_FILTER = ['__profile__', 'available_props.json', 'property_map.json', 'settings.json']

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
    self.shape = self._data_raw.shape

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
    if len(self._masks) == 0:
      return np.ones_like(self._data_raw, dtype=bool)
    return np.logical_and.reduce(self._masks)

  def _apply_masks(self):
    data = self._data_raw.copy()
    if len(self._masks) == 0:
      return data
    data[~self.mask] = np.nan
    return data

  def add_mask(self, mask):
    '''Adds a new mask'''
    if mask.shape != self.data.shape:
      raise ValueError(f'Trying to add mask of shape {mask.shape} to Detsum of shape {self.data.shape}')
    self._masks.append(mask)

  def reset_mask(self):
    self._masks = []

  def plot(self, raw=False,
           figsize=(7, 7),
           ax=None,
           base64=False,
           **imshow_kwargs):

    # Make sure the right total is being displayed
    if raw:
      counts = np.nansum(self._data_raw)
    else:
      counts = self.total_counts
      
    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)
    else:
      fig = ax.figure
    if raw:
      ax.imshow(self._data_raw, **imshow_kwargs)
    else:
      ax.imshow(self.data, aspect='equal', **imshow_kwargs)
    ax.set_title(f'{self.element} | {self.depth} | {self.scan_name}\ntotal counts: {counts}')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if base64:
      return encode_matplotlib_fig(fig)
    plt.close(fig)
    return fig 

  # def __add__(self, other):
  #   return self._create_default_mask(tot)

  def __repr__(self):
    return (f'Detsum(element: {self.element}, ' 
            f'orbital: {self.orbital}, ' 
            f'normalized: {self.normalized}, '
            f'total counts: {self.total_counts}')


class Scan:
  file_template = '^scan2D_([0-9]{1,7})$' # Assume scan number < 1E6

  def __init__(self, path=None,
               elements_of_interest=None,
               orbitals=['K'],
               normalized=True,
               copy=None):
    self.depth = path.split('/')[-2]
    self.name = path.split('/')[-1]
    self._elements_of_interest = elements_of_interest
    self._orbitals = orbitals
    self._normalized = normalized
    if not re.fullmatch(Scan.file_template, self.name):
      raise NameError(f'{path} is not a valid name for Scan directory')
    else:
      self.path = path

    self.scan_number = re.search(Scan.file_template, self.name).group(1)

    # Build regex template based on parameters
    template = '^detsum'
    if elements_of_interest is None:
      template += '_[a-zA-Z]{1,2}'
    else:
      template += f'_({"|".join(elements_of_interest)})'
    template += f'_({"|".join(orbitals)})'
    if normalized:
      template += '_norm.txt$'
    else:
      template += '.txt$'
    self._template = template  

    # This is gross
    if copy is None:
      self.detsums = self._make_detsums(template)
    else:
      self.detsums = copy.detsums
    self.detsums = sorted(self.detsums, key = lambda d: d.element)
    

  def _get_element_groups(self):
    elements = np.array(self.elements)
    data = np.stack([d.data for d in self.detsums], axis=2)
    groups = np.zeros_like(data[..., 0]).astype(str)
    unique_elements = {}
    for i, row in enumerate(data):
      for j, _ in enumerate(row): 
        pixel = data[i, j]
        group_elements = elements[~np.isnan(pixel)]
        group_name = '|'.join(group_elements)
        groups[i, j] = group_name
    return groups

  def get_unique_groups(self, elements_to_sum, sort_by):
    '''Returns a DataFrame with counts and sums for each unique element group.
    
    Args:
      elements_to_sum: list of elements to sum over for each group. For example, 
        if this is ['Cu', 'Br'], then the resulting DataFrame will have a colum
        with Cu counts for each group and Br counts for each group. 
      sort_by: either "num_pixels" or <element>_counts, where <element> is an
        element present in `elements_to_sum`.
    Returns: pandas DataFrame with unique element groups as the index. There is
      one column per element in `elements_to_sum`, and the final column is 
      either <element>_counts or num_pixels, depending on the value passed
      to `sort_by`. The rows are sorted by the final column.
      '''
    
    if 'counts' in sort_by:
      match = re.match('^([a-zA-Z]{1,2})_counts$', sort_by)
      if match and match.group(1) not in elements_to_sum:
        raise ValueError(f'{sort_by} passed to `sort_by`, but {match.group(1)}'
                          ' not found in `elements_to_sum`!')
      elif not match:
        raise ValueError('`sort_by` must be "num_pixels" or "{element}_counts"!')
    elif sort_by != 'num_pixels':
        raise ValueError('`sort_by` must be "num_pixels" or "{element}_counts"!')

    data = self.data
    if np.all(pd.isnull(data).values):
      return pd.DataFrame()
    aggfunc = {element: np.sum for element in elements_to_sum}
    aggfunc.update(element_group = 'count')
    res = pd.pivot_table(data,
                         index='element_group',
                         aggfunc=aggfunc)
    res.columns=[f'{element}_counts' for element in elements_to_sum] + ['num_pixels']
    return res.sort_values(by=sort_by, ascending=False)

  def get_detsum(self, element):
    assert element in self.elements
    return self.detsums[self.elements.index(element)]

  @property
  def elements(self):
    return [d.element for d in self.detsums]

  @property
  def data(self):
    df = pd.DataFrame()
    for d in self.detsums:
      df[d.element] = d.data.ravel()

    df['element_group'] = self._get_element_groups().ravel()
    df = df.replace('^$', np.nan, regex=True)
    return df

  @property
  def concentrations(self):
    df = pd.DataFrame()
    for d in self.detsums:
      df[d.element] = [d.total_counts]
    return df


  def plot_concentrations(self,
                          against='Cu',
                          elements=None,
                          single_figsize=(5, 5)):

    ncols = 2
    nrows = int(np.ceil(len(self.detsums) / ncols))
    f = plt.figure(figsize=(ncols*single_figsize[0], nrows*single_figsize[1]))
    x = self.detsums[list(map(lambda d: d.element, self.detsums)).index(against)]

    #Make axes
    for i, d in enumerate(self.detsums):
      a = f.add_subplot(f'{nrows}{ncols}{i}')
      #a = f.add_subplot(f'{nrows}{ncols}{i}')
      a.scatter(x.data.ravel(), d.data.ravel())
      a.set_xlabel(x.element, fontsize=14, fontweight='bold')
      a.set_ylabel(d.element, fontsize=14, fontweight='bold')
      a.text(0.5, 0.9, f'{d.element} total: {d.total_counts:0.4f}', transform=a.transAxes)
      a.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
      a.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
      f.suptitle(f'Scan {self.scan_number}', y=0.9, fontsize=16, fontweight='bold')
     

  def interactive_plot(self,
                       element='Cu',
                       display=True,
                       save=False,
                       filename='altair_dashboard.html',
                       base_dir=None):
    # Only select pixels that have Cu values
    cu_points = self.data[self.data[element].notnull()]
    alt.data_transformers.disable_max_rows()
    cu_points['num_present_elements'] = cu_points['element_group'].apply(lambda x: len(x.split('|')))
    
    
    #unpivot the table into a new table with ['pixel', 'element', 'concentration']
    cu_points_melted = cu_points.melt(
        id_vars='element_group',
        value_vars=self.elements,
        var_name='element',
        value_name='concentration')
    
    
    selection = alt.selection_multi(fields=['element_group'])
    
    #base bar chart
    bar_base = alt.Chart(cu_points_melted).transform_filter(
        selection
        ).encode(
        y='element',
        x='sum(concentration)',
        text=alt.Text('sum(concentration)', format='0.2f')
    )
    
    #function to create scatter plot based on element
    scatter_base = lambda e: alt.Chart(cu_points).mark_circle().encode(
        x=element,
        y=e,
        tooltip=self.elements,
        color=alt.condition(selection, 'element_group:N', alt.value('lightgray'), legend=None)
    ).add_selection(
        selection
    )
    
    counts = alt.Chart(cu_points).transform_filter(
        selection
    ).transform_aggregate(
        count='count()'
    ).transform_calculate(
        text="number of points selected: " + alt.datum.count
    ).mark_text(
        dy=-20,
        baseline="top",
        align="left"
    ).encode(
        x=alt.value(100),
        y=alt.value(5),
        text='text:N',
    )
  
    # legend plot
    legend = alt.Chart(cu_points).mark_circle(size=100).encode(
        x='element_group:N',
        color=alt.condition(selection, 'element_group:N', alt.value('lightgray'))
    ).add_selection(
        selection
    )
    
    #bar plots
    bar_final = bar_base.mark_bar()
    bar_text = bar_base.mark_text(align='left', baseline='middle', dx=3)
    
    #scatter plots
    scatters = alt.vconcat(
        *[alt.hconcat(
            *[(scatter_base(e).properties(width=300, height=150) + counts)
            for e in self.elements[i:i+3] if e != 'cu']
        )
        for i in range(0, len(self.elements) - 1, 3)]
    )
    
    
    #final plot, put together with altair's fancy syntax
    final_plot = alt.vconcat(scatters, legend | (bar_final + bar_text), center=True)
  
    if save:
      if base_dir is None:
        raise ValueError('base_dir cannot be None if saving')
      final_plot.save(os.path.join(base_dir, filename))
    
    if display:
      return final_plot 

  def _make_detsums(self, template):
    detsums = []
    for f in os.listdir(self.path):
      fullpath = os.path.join(self.path, f)
      #print(self._template, f, re.fullmatch(self._template, f))
      if re.fullmatch(self._template, f):
        detsums.append(Detsum(fullpath))
    for d in detsums:
      self.__dict__[f'{d.element}_{d.orbital}'] = d
    return detsums
  
  def __repr__(self):
    nl = '\n'
    #detsums = '\n'.join([str(d) for d in self.detsums])
    return (f'Scan(scan_number: {self.scan_number}, contained_detsums:{nl}'
            f'{nl.join(map(str, self.detsums))})')

  #def plot_all_detsums(self, raw=True)


  def filter_by(self, element):
    # Create a new scan object that copies this scan
    s = Scan(self.path,
             self._elements_of_interest,
             self._orbitals,
             self._normalized,
             copy=self)

    # Extract the mask from the element of interest
    for d in s.detsums:
      if d.element == element:
        mask = d.mask

    # Add mask to other detsums 
    for d in s.detsums:
      if d.element != element:
        d.add_mask(mask)
    return s

class CombinedScan(Scan):
  def __init__(self, scans):
    
    #make sure all scans are at the same depth
    assert len(set([s.depth for s in scans])) == 1

    self.depth = scans[0].depth
    self._scans = scans


  @property
  def data(self):
    scan_data = [s.data for s in self._scans]
    return pd.concat(scan_data, ignore_index=True)
  
  def filter_by(self, element):
    new_scans = [s.filter_by(element) for s in self._scans]
    return CombinedScan(new_scans) 

  def get_detsum(self, element):
    detsums = []
    for s in self._scans:
      assert element in s.elements
      detsums.append(s.detsums[s.elements.index(element)])
    return detsums

  def __repr__(self):
    scans = '\n'.join([f'{s.__repr__()}\n--------------------\n' for s in self._scans])
    return f'CombinedScan\n===================\n{scans})'
           
class Depth:
  file_template = '^([0-9])+m$'

  def __init__(self, path,
               elements_of_interest=None,
               orbitals=['K'],
               normalized=True,
               noisy_scans=None):
    self._instance_kwargs = {
        'path': path,
        'elements_of_interest': elements_of_interest,
        'orbitals': orbitals,
        'normalized': normalized,
    }
    self.scans = []
    self.name = path.split('/')[-1]
    if not re.fullmatch(Depth.file_template, self.name):
      raise NameError(f'{self.name} is not a valid name for a Depth!')
    #self.depth = re.search(Depth.file_template, path.split('/')[-1]).group(1)
    self.depth = path.split('/')[-1]

    # Load the scans, skipping noisy ones
    for f in os.listdir(path):
      if noisy_scans is not None and f in noisy_scans:
        continue
      fullpath = os.path.join(path, f)
      try:
        self.scans.append(
            Scan(fullpath,
                 elements_of_interest=elements_of_interest,
                 orbitals=orbitals,
                 normalized=normalized))
      except NameError as e:
        print(e)
        pass

  # Returns a fresh copy of the Detsum from the source data
  def fresh_copy(self):
    return Depth(**self._instance_kwargs)

  def get_scan(self, scan_number):
    for s in self.scans:
      if s.scan_number == scan_number:
        return s
    
    raise ValueError(f'No scan found with scan number {scan_number}')

  @property
  def detsums(self):
    detsums = []
    for s in self.scans:
      detsums += s.detsums
    return detsums

  @property
  def combined_scan(self):
    return CombinedScan(self.scans)
  
  @property
  def elements(self):
    '''Returns the elements present in all scans'''
    return sorted(list(reduce(set.intersection, [set(s.elements) for s in self.scans])))

  @property
  def summary(self):
    pass

  @property
  def concentrations(self):
    df = pd.DataFrame()
    df.columns = self.elements
    for s in self.scans:
      df.loc[s.scan_number] = s.concentrations
    df.index.name = 'Scan Number'
    return df

  def apply_element_filter(self, filter_dict,
                           combine_detsums=False, inplace=True):
    '''Applies element-wise filter to all Detsums.

    Args:
      filter_dict: dictionary of the form {scan_number: {element: filter_func}}
        `filter_func` takes an array and returns a single threshold value. 
      combine_detsum: bool. Whether or not to use the data from all Detsums
        in this Depth when calculating the threshold with `filter_func`. If
        this is `False`, then the threshold will be calculated based on data
        from each individual Detsum
      inplace: bool. Whether or not to create a new Depth object or modify the
        current one. This option exists because creating a new Depth object may
        take some time, as it has to re-import data to re-create all Scans and
        Detsums. 
    '''

    # TODO: Fix this later
    if combine_detsums:
      raise NotImplementedError
    

    # for testing the functions in filter_dict
    test_arr = np.linspace(0, 1, 100)

    if inplace:
      # Reset all masks
      for d in self.detsums:
        if not np.all(d.mask):
          d.reset_mask()
      depth = self
    else:
      depth = Depth.fresh_copy(self)

    data_full = depth.combined_scan.data.copy()

    for scan in depth.scans:
      filter_dict_inner = filter_dict[scan.scan_number]
      for element in filter_dict_inner:
        if element not in scan.elements:
          print(f'{element} not present in Scan {scan}')
      for d in scan.detsums:
        get_threshold = filter_dict_inner[d.element]
        # Make sure filter_funcs are working properly.
        if not isinstance(get_threshold(test_arr), float):
          raise TypeError('Problem encountered with filter function for {d.element}. '
                          'All filter funcs must be of the form f([array]) -> [float]')
        data_for_mask = d._data_raw
  
        threshold = get_threshold(data_for_mask)
        #print(f'Threshold for {d.element}: {threshold}')
        mask = d._data_raw > threshold
        d.add_mask(mask)

    if not inplace:
      return depth


  def __repr__(self):
    return f'Depth({self.depth}, scans: {[s.scan_number for s in self.scans]})'


class Profile:
  '''A Container class to instantiate and hold Depth objects'''

  def __init__(self, experiment_dir,
               elements_of_interest=None,
               orbitals=['K'],
               normalized=True):

    self._elements_of_interest = elements_of_interest
    depths = []

    self.experiment_dir = experiment_dir

    # Get noisy scans from settings
    noisy_scans = []
    _check_or_create_settings('noisy_scans', base_dir=self.experiment_dir)
    noisy_scans_file = os.path.join(experiment_dir,
                                    'settings',
                                    ocean_utils.NOISY_SCANS_FILE)
    with open(noisy_scans_file, 'r') as f:
      scan_dict = json.load(f)
    for scan in scan_dict:
      if scan_dict[scan]:
        noisy_scans.append(scan)

    # Load depths
    for dir_or_file in os.listdir(experiment_dir):
      if dir_or_file in PROFILE_FILE_FILTER:
        continue
      try:
        fullpath = os.path.join(experiment_dir, dir_or_file)
        d = Depth(os.path.join(fullpath),
                  elements_of_interest=elements_of_interest,
                  orbitals=['K'],
                  normalized=True,
                  noisy_scans=noisy_scans or None)
        depths.append(d)
        print(f"Successfully imported data for {d.depth}")
      except NameError as e:
        print(e)
        continue

    self.depths = depths
    

    
    

  def apply_element_filter(self, filter_dict):
    '''Applies element-wise filter depth-by-depth and scan-by-scan.'''
    for depth in self.depths:
      depth_value = depth.depth
      if depth_value not in filter_dict:
        raise KeyError(f'{depth_value} not found in `filter_dict`!')
      depth.apply_element_filter(filter_dict[depth_value])
      
      
  @property
  def scans(self):
    scans = []
    for d in self.depths:
      scans += d.scans
    return scans

  @property
  def detsums(self):
    detsums = []
    for d in self.depths:
      detsums += d.detsums
    return detsums
            
