import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cycler import cycler
import numpy as np
from io import BytesIO
import base64
import json
import os
import ipywidgets as iw
import time

COLORS = plt.cm.tab20(np.arange(20))
PATTERNS = [None, '\\\\', '..', 'xx', '**', '++', 'oo', '00', '--', '\\\\\\', '...', 'xxx', '***', '+++', 'ooo', '000', '---']
SMALLTEXTBOX = iw.Layout(width='50px', height='25px')
PROP_FILE_AVAILABLE = 'available_props.json'
PROP_FILE_MAP = 'property_map.json'
SETTINGS_FILE = 'settings.json'


def _check_or_create_settings(dir_name=None):
  '''Checks to see if a settings file exists or create a new one'''
  fname = SETTINGS_FILE
  if dir_name is not None:
    fname = os.path.join(dir_name, fname)
  
  if os.path.exists(fname):
    return
  
  # Make a blank settings file 
  with open(fname, 'w') as f:
    json.dump({}, f)


def _prop_list_exists(dir_name=None):
  fname = PROP_FILE_AVAILABLE
  if dir_name is not None:
    fname = os.path.join(dir_name, fname)
  
  return os.path.exists(fname)

def _create_prop_list(dir_name=None, overwrite=False):
  '''Creates a new list of properties and saves it to `fname`.'''

  fname = PROP_FILE_AVAILABLE
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

  available_fname = PROP_FILE_AVAILABLE
  map_fname = PROP_FILE_MAP

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

  map_fname = PROP_FILE_MAP
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

  map_fname = PROP_FILE_MAP
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


class PropSelector:
  '''Object that will hold a prop selector widget using composition.'''
  def __init__(self, props, orientation='vertical', title=None, description_func=lambda p: p, **layout_kwargs):
    self._props = props
    self._boxes = [iw.Checkbox(value=False,
                               description=description_func(prop), indent=False) for prop in props]

    assert orientation in ['vertical', 'horizontal']
    self._orientation = orientation
    self._layout = iw.Layout(**layout_kwargs)
    self._title = title

  @property
  def selected_props(self):
    selected = []
    for prop, box in zip(self._props, self._boxes):
      if box.value:
        selected.append(prop)
    return selected

  @property
  def widget(self):
    if self._orientation == 'vertical':
      container = iw.VBox
    else:
      container = iw.HBox
    if self._title is not None:
      title = iw.HTML(f'<h3>{self._title}</h3>')
      inputs = [title] + self._boxes
    else:
      inputs = self._boxes

    return container(inputs, layout=self._layout)


  def observe(self, handler):
    '''Sets each checkbox to observe `handler`.
    
    `handler` has a signature of handler(prop, val: bool)
    '''
    def handler_wrapper(prop, b):
      val = b['new']
      return handler(prop, val)

    for prop, box in zip(self._props, self._boxes):
      box.observe(lambda b, prop=prop: handler_wrapper(prop, b), names='value')


class ElementFilter:
  SETTING_KEY = 'element_filter'
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
      element_input = lambda element:iw.Textarea('2', layout=SMALLTEXTBOX)
                                              
      
    self._input_type = input_type
    self._input_widgets = [element_input(e) for e in self._elements]
    self._layout = iw.Layout(**layout_kwargs)


  def get_input_widget(self, e):
    return self._input_widgets[self._elements.index(e)]

  def save_settings(self, key='latest', experiment_dir=None):
    '''Saves the current element filter as `key` in "settings.json"'''
    if not key:
      return
    _check_or_create_settings(experiment_dir)
    fname = SETTINGS_FILE
    if experiment_dir is not None:
      fname = os.path.join(experiment_dir, fname)
      
    # Read settings into memory
    with open(fname, 'r') as f:
      settings = json.load(f)

    # Edit the settings
    skey = ElementFilter.SETTING_KEY
    if skey not in settings:
      settings[skey] = {}
    settings[skey][key] = self.value_dict

    # Write settings to disk
    with open(fname, 'w') as f:
      json.dump(settings, f)

  def load_settings(self, key='latest', experiment_dir=None):
    '''Loads settings from json file.'''
    _check_or_create_settings()
    
    fname = SETTINGS_FILE
    # Read settings into memory
    with open(fname, 'r') as f:
      settings = json.load(f)

    skey = ElementFilter.SETTING_KEY
    if skey not in settings:
      print('NO SETTINGS FILE FOUND')
      return

    value_dict = settings[skey][key]
    for e in self._elements:
      if self._input_type == 'text':
        val = str(value_dict[e])
      else:
        val = value_dict[e]
      self.get_input_widget(e).value = val


  @property
  def filter_dict(self):
    filter_dict = {}
    for e in self._elements:
      val = float(self.get_input_widget(e).value)
      filter_dict[e] = self._base_func(val)
    return filter_dict


  @property
  def value_dict(self):
    vals = {}
    for e in self._elements:
      val = float(self.get_input_widget(e).value)
      vals[e] = val
    return vals

  @property
  def values(self):
    return list(self.value_dict.values())

  @property
  def value_string(self):
    labels = [f'{e}: {val} | ' for e, val in zip(self._elements, self.values)]
    return ''.join(labels)

  @property
  def widget(self):
    if self._orientation == 'vertical':
      container = iw.VBox
      inputs = self._input_widgets
    else:
      container = iw.HBox
      if self._input_type == 'text':
        inputs = [iw.HBox([i, iw.HTML(e)], layout=iw.Layout(padding='10px')) for i, e in zip(self._input_widgets, self._elements)]
        inputs = [iw.VBox([inputs[i], inputs[i+1]]) for i in range(0, len(inputs), 2)]
      else:
        inputs = self._input_widgets

    return container(inputs, layout=self._layout)


class ElementFilterWithBoxes(ElementFilter):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._boxes = [iw.Checkbox(value=False, indent=False) for e in self._elements]

    if self._orientation == 'vertical':
      container = iw.HBox
    else:
      container = iw.VBox
    self._inputs = [container([box, _input]) for box, _input in zip(self._boxes, self._inputs)]

  @property
  def selected_elements(self):
    selected = []
    for element, box in zip(self._elements, self._boxes):
      if box.value:
        selected.append(element)
    return selected


class SettingsController:
  BUTTON_WIDTH = '70px'
  INPUT_WIDTH = '150px'
  INPUT_HEIGHT = '25px'
  def __init__(self, w, orientation='vertical', experiment_dir=None, **layout_kwargs):
    '''Creates a settings widget for widget `w`'''

    self._experiment_dir = experiment_dir
    _check_or_create_settings(experiment_dir)
    

    self._layout = iw.Layout(**layout_kwargs)
    assert orientation in ['horizontal', 'vertical']
    self._orientation = orientation

    self._w = w


    # Save Widget
    self.save_widget = iw.Textarea(value='',
        layout=iw.Layout(width=SettingsController.INPUT_WIDTH, height=SettingsController.INPUT_HEIGHT))
    self.save_button = iw.Button(description='Save',
        layout=iw.Layout(width=SettingsController.BUTTON_WIDTH))
    self.save_button.on_click(self._save_settings)

    # Load Widget
    self.load_widget = iw.Dropdown(options=['None'],
        layout=iw.Layout(width=SettingsController.INPUT_WIDTH, height=SettingsController.INPUT_HEIGHT))
    self.load_button = iw.Button(description='Load',
        layout=iw.Layout(width=SettingsController.BUTTON_WIDTH))
    self.load_button.on_click(lambda b: self._load_settings(experiment_dir))

    fname = SETTINGS_FILE
    if experiment_dir is not None:
      fname = os.path.join(experiment_dir, fname)

    # Load settings into memory
    with open(fname, 'r') as f:
      self._settings = json.load(f)
    
    if self._w.SETTING_KEY in self._settings:
      self._options = list(self._settings[self._w.SETTING_KEY].keys())
    else:
      self._options = ['None']

    self.load_widget.options = self._options


  def _save_settings(self, b):
    text = self.save_widget.value
    self._settings[self._w.SETTING_KEY].update({text: self._w.value_dict})
    self._options = list(self._settings[self._w.SETTING_KEY].keys())
    self.load_widget.options = self._options
    self.save_widget.value = ''

    self._w.save_settings(key=text)
    

  def _load_settings(self, experiment_dir):

    selected = self.load_widget.value
    self._w.load_settings(key=selected, experiment_dir=experiment_dir)

  @property
  def widget(self): 
    if self._orientation == 'vertical':
      container = iw.VBox
    else:
      container = iw.HBox
    
    save = iw.HBox([self.save_button, self.save_widget])
    load = iw.HBox([self.load_button, self.load_widget])
    title = iw.HTML('<h2>Settings</h2>')
    
    return container([title, save, load], layout=self._layout)
