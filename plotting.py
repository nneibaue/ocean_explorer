import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cycler import cycler
import numpy as np
from io import BytesIO
import base64
import json
import os
import ipywidgets as iw

COLORS = plt.cm.tab20(np.arange(20))
PATTERNS = [None, '\\\\', '..', 'xx', '**', '++', 'oo', '00', '--', '\\\\\\', '...', 'xxx', '***', '+++', 'ooo', '000', '---']

SMALLTEXTBOX = iw.Layout(width='50px', height='25px')

def _prop_list_exists(dir_name=None):
  fname = 'available_props.json'
  if dir_name is not None:
    fname = os.path.join(dir_name, fname)
  
  return os.path.exists(fname)

def _create_prop_list(dir_name=None, overwrite=False):
  '''Creates a new list of properties and saves it to `fname`.'''

  fname = 'available_props.json'
  if dir_name is not None:
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
    fname = os.path.join(dir_name, fname)

  # Get a default list of 10 colors. I don't exactly understand *how* this
  # works, but I know that it *does* work. Here is the stack overflow for
  # this: https://stackoverflow.com/questions/49233855/matplotlib-add-more-colors-to-the-default-colors-in-axes-prop-cycle-manually
  color_cycler = cycler(facecolor=COLORS)
  pattern_cycler = cycler(hatch=PATTERNS)

  # Multiplication of cyclers results in the 'outer product'
  # https://matplotlib.org/cycler/
  props = list(iter(pattern_cycler * color_cycler))

  # Convert numpy arrays to lists for json encoding
  for prop in props:
    for key in prop:
      if isinstance(prop[key], np.ndarray):
        prop[key] = list(prop[key])

  if os.path.exists(fname) and not overwrite:
    raise ValueError(f'{fname} already exists! Overwriting will erase property data for this experiment\n'
                     f'Please set `overwrite` to True if you would like to proceed')

  with open(fname, 'w') as f:
    json.dump(props, f)


def save_new_props(element_groups, dir_name=None):
  '''Takes properties from available_props.json and assigns it to element groups.
  
  Note: if `element_group` already has a property assigned, then this does nothing.

  Args:
    element_groups: list of element group strings. E.g. ['Cu', 'Cu|Br', 'Cu|Zn']
    dir_name: directory to save prop file in. If this is None, defaults to current working directory.
    
  '''

  available_fname = 'available_props.json'
  map_fname = 'property_map.json'

  if dir_name is not None:
    available_fname = os.path.join(dir_name, available_fname)
    map_fname = os.path.join(dir_name, map_fname)

  # Read property map into memory, if exists. Else make a new one
  if os.path.exists(map_fname):
    with open(map_fname, 'r') as f:
      prop_map = json.load(f)
  else:
    prop_map = {}
  

  if not os.path.exists(available_fname):
    raise FileNotFoundError('Cannot find {available_fname}. Please check that it exists!')

  # Read available props into memory
  with open(available_fname, 'r') as f:
    available_props = json.load(f)

  # If the props run out, then create a new set (for now, props will be repeated)
  # This is kind-of an ugly hack for now, but will prevent the code from crashing
  if not available_props:
    _create_prop_list(dir_name=dir_name, overwrite=True)

  # Assign new properties from available_props
  for group in element_groups:
    if group in prop_map:
      continue
    prop_map.update({group: available_props.pop(0)})

  # Re-save available_props (now with fewer available)
  with open(available_fname, 'w') as f:
    json.dump(available_props, f)

  # Re-save new property map
  with open(map_fname, 'w') as f:
    json.dump(prop_map, f)
   
def get_single_prop(element_group, dir_name=None):
  '''Gets the property for the given element_group, if it exists.'''

  map_fname = 'property_map.json'
  if dir_name is not None:
    map_fname = os.path.join(dir_name, map_fname)
  if not os.path.exists(map_fname):
    raise FileNotFoundError(f'No property map found at {map_fname}! Use `save_new_props` to create a new map')

  with open(map_fname, 'r') as f:
    prop_map = json.load(f)
  if element_group not in prop_map:
    raise ValueError(f'{element_group} not found!')
  return prop_map[element_group]
  
def get_all_props(dir_name=None):
  '''Retrieves the entire property map from disk.'''

  map_fname = 'property_map.json'
  if dir_name is not None:
    map_fname = os.path.join(dir_name, map_fname)
  if not os.path.exists(map_fname):
    raise FileNotFoundError(f'No property map found at {map_fname}! Use `save_new_props` to create a new map')
    
  with open(map_fname, 'r') as f:
    prop_map = json.load(f)
  return prop_map


def encode_matplotlib_fig(fig):
  buf = BytesIO()
  fig.savefig(buf, format='png', bbox_inches='tight')
  data = base64.b64encode(buf.getvalue()).decode('ascii')
  plt.close(fig)
  return f"<img src='data:image/png;base64,{data}'/>"

def ribbon_plot(depths,
                element_filter,
                filter_by='Cu',
                combine_scans=True,
                combine_detsums=True,
                N=8,
                normalize_by='counts',
                base64=False,
                experiment_dir=None):
  '''Shows fractional concentration of element among different groups.

  Args:
    filter_by: string. Element under investigation
    element_filter: dict specifying how elements should be filtered. For more 
      info, see `Depth.apply_element_filter`. In Colab, this can be done by
      opening a new cell and running `Depth.apply_element_filter?`.
    combine_scans: bool. Whether or not to show individual scans as different
      bars in the ribbon plot. If this is True, then scans will be combined
    combine_detsums: bool. Whether or not to combine detsums when applying the
      filter functions defined in `element_filter`. If this is True, then all
      detsums within a given depth will be considered when applying the filter.
    N: int. The amount of groups to use. For example, N=12 means "take the
      top 12 groups from each depth"
    normalize_by: string. Which quantity to use for data normalization. This
      can either be 'counts' or 'pixels'.
    base64: bool. If this is True, will encode the graph and return html image element.
    experiment_dir: optional directory containing property information. If this is None, then properties
      will be saved to local disk (if using colab, this is the local directory on the Colab machine, which
      may be deleted when a new instance of the Notebook is loaded)
    '''

  fig, ax = plt.subplots(figsize=(16, 4))

  # Sort depth objects by depth
  depths = sorted(depths, key=lambda d: int(d.depth[:-1]), reverse=True)

  if not _prop_list_exists(experiment_dir):
    _create_prop_list(dir_name=experiment_dir, overwrite=True)

  if normalize_by == 'pixels':
    sort_by = 'num_pixels'
    col = 'num_pixels'
    normalization = lambda g: g['num_pixels'].sum()
  elif normalize_by == 'counts':
    sort_by = f'{filter_by}_counts'
    col = f'{filter_by}_counts'
    normalization = lambda g: g[f'{filter_by}_counts'].sum()
  else:
    raise ValueError("`normalize_by` must be 'pixels' or 'counts'")

  def get_scan_groups(scan):
    groups = scan.filter_by(filter_by).get_unique_groups(elements_to_sum=[filter_by], sort_by=sort_by)
    if len(groups.index):
      groups[col] /= normalization(groups)
      groups = groups.iloc[:N]
    return groups

        #prop_dict[g].update({'alpha': 0.8, 'fill': True, 'edgecolor': 'k'})
      #print(f'Adding {g} to prop_dict')

  # Properties used for this 
  these_props = {}

  # Plots single scan (one bar)
  def plot_scan(scan, i):
    groups = get_scan_groups(scan)

    # Save properties for these groups
    save_new_props(groups.index, dir_name=experiment_dir)

    # If groups are empy, do nothing
    if not len(groups.index):
      return

    left = 0
    thickness = 1
    props = get_all_props(experiment_dir)
    for group in groups.index:
      # Get props and update with a couple of things
      prop = props[group]
      prop.update({'fill': 'True', 'edgecolor': 'k'})
      these_props[group] = prop

      #Encode the group value (e.g. Cu|Mn -> 0.0778) as the rectangle width
      rect_width = groups[col].loc[group]
      ax.add_patch(mpatches.Rectangle((left, i-thickness/2),
                            rect_width, thickness, **prop))
      left += rect_width

    t=ax.text(left+0.02, i, f'{col}: {groups[col].sum():0.2f}',
            verticalalignment='center',
            fontsize=10,
            color='black',
            fontweight='bold',
            transform=ax.transData)

  i=0
  yticklabels = []
  for depth in depths:
    depth.apply_element_filter(element_filter, combine_detsums=combine_detsums)
    if combine_scans:
      scans = [depth.combined_scan]
      yticklabels.append(depth.depth)
    else:
      scans = depth.scans

    for scan in scans:
      plot_scan(scan, i)
      i += 1
      if not combine_scans:
        yticklabels.append(f'{depth.depth}: {scan.name}')

  ax.set_xlim(0, 1.13)

    # https://stackoverflow.com/questions/23696898/adjusting-text-background-transparency
    #t.set_bbox({'facecolor':'white', 'alpha':0.8})
    
  element_group_legend(ax, these_props.keys(), experiment_dir)
  
  ax.set_yticks(np.arange(0, i))
  ax.set_yticklabels(yticklabels)
  ax.set_ylim(-0.5, i-0.5)
  ax.set_title(f'{filter_by} {normalize_by}', fontsize=16, fontweight='bold')
  plt.close()
  if base64:
    return encode_matplotlib_fig(fig)
  return fig


# Make the legend using patches. Here is a helpful link
# https://stackoverflow.com/questions/53849888/make-patches-bigger-used-as-legend-inside-matplotlib
def element_group_legend(ax, groups, dir_name):
  props = get_all_props(dir_name)
  patches = []
  for group in groups:
    patches.append(mpatches.Patch(**props[group], label=group))

  leg = ax.legend(handles=patches,
                  bbox_to_anchor=(1, 1.05),
                  ncol=round(len(patches) / 10) or 1,
                  labelspacing=2,
                  bbox_transform=ax.transAxes,
                  loc='upper left')
  for patch in leg.get_patches():
    patch.set_height(22)
    patch.set_y(-10)


class DepthSelector:
  '''Object that will hold a depth selector widget using composition.'''
  def __init__(self, depths, orientation='vertical', **layout_kwargs):
    self._depths = depths
    self._boxes = [iw.Checkbox(value=False, description=depth.depth, indent=False) for depth in depths]

    assert orientation in ['vertical', 'horizontal']
    self._orientation = orientation
    self._layout = iw.Layout(**layout_kwargs)

  @property
  def selected_depths(self):
    selected = []
    for depth, box in zip(self._depths, self._boxes):
      if box.value:
        selected.append(depth)
    return selected

  @property
  def widget(self):
    if self._orientation == 'vertical':
      container = iw.VBox
    else:
      container = iw.HBox

    return container(self._boxes, layout=self._layout)


class ElementFilter:
  '''Object that will hold an element filter selector widget using composition.'''
  def __init__(self, elements, orientation='vertical', input_type='slider', **layout_kwargs):
    assert orientation in ['vertical', 'horizontal']
    self._orientation = orientation
    self._elements = elements
    self._base_func = lambda n: lambda x: np.mean(x) + n * np.std(x)


    if input_type == 'slider':
      if self._orientation == 'vertical':
        slider_orientation = 'horizontal'
      else:
        slider_orientation = 'vertical'
      element_input = lambda element: iw.FloatSlider(2.0, min=0, max=4, step=0.1,
                                                     description=element, orientation=slider_orientation)
    elif input_type == 'text':
      element_input = lambda element: iw.HBox([iw.Textarea('2', layout=SMALLTEXTBOX), iw.HTML(element)],
                                              layout=iw.Layout(padding='10px'))
      
    self._input_type = input_type
    self._inputs = [element_input(e) for e in self._elements]
    self._layout = iw.Layout(**layout_kwargs)


  def get_input(self, e):
    return self._inputs[self._elements.index(e)].children[0]

  @property
  def filter_dict(self):
    filter_dict = {}
    for e in self._elements:
      val = float(self.get_input(e).value)
      filter_dict[e] = self._base_func(val)
    return filter_dict

  @property
  def widget(self):
    if self._orientation == 'vertical':
      container = iw.VBox
      inputs = self._inputs
    else:
      container = iw.HBox
      if self._input_type == 'text':
        inputs = [iw.VBox([self._inputs[i], self._inputs[i+1]]) for i in range(0, len(self._inputs), 2)]
      else:
        inputs = self._inputs

    return container(inputs, layout=self._layout)


class ElementFilterWithBoxes(ElementFilter):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._boxes = [iw.Checkbox(value=False, description=e, indent=False) for e in self._elements]
    self._inputs = [iw.HBox([box, _input]) for box, _input in zip(self._boxes, self._inputs)]

  @property
  def selected_elements(self):
    selected = []
    for element, box in zip(self._elements, self._boxes):
      if box.value:
        selected.append(element)
    return selected
