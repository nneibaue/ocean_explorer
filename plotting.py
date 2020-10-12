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
import ocean_utils as utils
from IPython.display import display, HTML

COLORS = plt.cm.tab20(np.arange(20))
PATTERNS = [None, '\\\\', '..', 'xx', '**', '++', 'oo', '00', '--', '\\\\\\', '...', 'xxx', '***', '+++', 'ooo', '000', '---']
SMALLTEXTBOX = iw.Layout(width='50px', height='25px')
PROP_FILE_AVAILABLE = 'available_props.json'
PROP_FILE_MAP = 'property_map.json'


def _check_or_create_settings(setting_key, base_dir=None):

  if base_dir is not None:
    settings_dir = os.path.join(base_dir, 'settings')
  else:
    settings_dir = 'settings'

  if not os.path.isdir(settings_dir):
    os.makedirs(settings_dir) 

  fname = os.path.join(settings_dir, f'{setting_key}.json')
  if not os.path.exists(fname):
    with open(fname, 'w') as f:
      json.dump({}, f, indent=2)


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
    json.dump(props, f, indent=2)


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
    json.dump(available_props, f, indent=2)

  # Re-save new property map
  with open(map_fname, 'w') as f:
    json.dump(prop_map, f, indent=2)
   
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

def ribbon_plot(profile,
                element_filter,
                filter_by='Cu',
                combine_scans=True,
                combine_detsums=True,
                N=8,
                normalize_by='counts',
                base64=False):
  '''Shows fractional concentration of element among different groups.

  Args:
    profile: ocean.Profile object
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
    '''

  fig, ax = plt.subplots(figsize=(16, 4))

  # Sort depth objects by depth
  depths = profile.depths
  depths = sorted(depths, key=lambda d: int(d.depth[:-1]), reverse=True)
  profile.apply_element_filter(element_filter)

  #TODO tidy up enums
  experiment_dir = 'deglitched_profiles'

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
      title = iw.HTML(f'<h3>{self._title}</h3>', layout=iw.Layout(flex='auto auto'))
      for box in self._boxes:
        box.layout = iw.Layout(flex='1')
      inputs = [title] + self._boxes
      self._layout.align_items = 'center'
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


class ElementFilterPanel:
  SETTING_KEY = 'element_filter_panel'
  def __init__(self, profile, layout_kwargs={}, **element_filter_kwargs):

    self._element_filter_kwargs = element_filter_kwargs
    self._layout_kwargs = layout_kwargs

    # {depth_value: {scan_number: {ElementFilterSinglePane instance}}
    self._element_filters = {}

    for depth in profile.depths:
      this_scan_dict = self._make_scan_dict(depth)
      self._element_filters[depth.depth] = this_scan_dict

    fname = f'{self.SETTING_KEY}.json'
    self.experiment_dir = profile.experiment_dir
    self.settings_file = os.path.join(self.experiment_dir, 'settings', fname)

    # Make sure there is a settings file present with the right key for this widget
    _check_or_create_settings(self.SETTING_KEY, base_dir=self.experiment_dir)

  def _make_scan_dict(self, depth):
    this_depth = {}
    for scan in depth.scans:
      this_depth[scan.scan_number] = ElementFilterSinglePane(
          elements=scan.elements, **self._element_filter_kwargs)
    return this_depth


  @property
  def filter_dict(self):
    d = {}
    for depth, scans in self._element_filters.items():
      d[depth] = {}
      for scan_num, element_filter in scans.items():
        d[depth][scan_num] = element_filter.filter_dict

    return d

  @property
  def value_dict(self):
    d = {}
    for depth, scans in self._element_filters.items():
      d[depth] = {}
      for scan_num, element_filter in scans.items():
        d[depth][scan_num] = element_filter.value_dict

    return d
      
  def save_settings(self, key='latest'):
    '''Saves the current element filter as `key` in "settings.json"'''
    if not key:
      return

    # Read settings into memory
    with open(self.settings_file, 'r') as f:
      settings = json.load(f)
    
    # Modify setting
    settings[key] = self.value_dict

    # Write settings to disk
    with open(self.settings_file, 'w') as f:
      json.dump(settings, f, indent=2)


  def load_settings(self, key='latest'):
    '''Loads settings from json file.'''
    _check_or_create_settings(self.SETTING_KEY)
    
    # Read settings into memory
    with open(self.settings_file, 'r') as f:
      settings = json.load(f)

    value_dict = settings[key]
    for depth_value in value_dict:
      this_depth = value_dict[depth_value]
      for scan_number in this_depth:
        self._element_filters[depth_value][scan_number]._load_settings(this_depth[scan_number])
        
      
  @property
  def widget(self):
    depth_panes = []
    for depth_value in self._element_filters:
      scan_panes = []
      this_depth = self._element_filters[depth_value]
      for scan_number in this_depth:
        this_element_filter = this_depth[scan_number]
        scan_panes.append(this_element_filter.widget)
      this_panel = iw.Tab(children=scan_panes) 

      # Loop scans again to set titles
      for i, scan_number in enumerate(this_depth):
        this_panel.set_title(i, scan_number)

      depth_panes.append(this_panel)

    depth_panel = iw.Tab(children=depth_panes,
                         layout=iw.Layout(**self._layout_kwargs))

    # Loop depths again to set titles
    for i, depth_value in enumerate(self._element_filters):
      depth_panel.set_title(i, depth_value)
      

    return depth_panel

    
    
class ElementFilterSinglePane:
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

  def _load_settings(self, value_dict):
    '''Sets internal widget state.

    Args:
      value_dict: dictionary of the form {element1: value1, element2: value2, ...} 

    '''
    for e in value_dict:
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


class ElementFilterWithBoxes(ElementFilterSinglePane):
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
  TOTAL_WIDTH = '220px'
  def __init__(self, w, orientation='vertical', experiment_dir=None, **layout_kwargs):
    '''Creates a settings widget for widget `w`'''

    self.experiment_dir = w.experiment_dir
    self.settings_file = w.settings_file
    self._setting_key = w.SETTING_KEY

    _check_or_create_settings(self._setting_key, self.experiment_dir)
    

    self._layout = iw.Layout(**layout_kwargs)
    assert orientation in ['horizontal', 'vertical']
    self._orientation = orientation

    # Save Widget
    self.save_widget = iw.Textarea(value='',
        layout=iw.Layout(width=SettingsController.INPUT_WIDTH, height=SettingsController.INPUT_HEIGHT))
    self.save_button = iw.Button(description='Save',
        layout=iw.Layout(width=SettingsController.BUTTON_WIDTH))
    self.save_button.on_click(lambda b: self.save_settings())

    # Load Widget
    self.load_widget = iw.Dropdown(options=['None'],
        layout=iw.Layout(width=SettingsController.INPUT_WIDTH, height=SettingsController.INPUT_HEIGHT))
    self.load_button = iw.Button(description='Load',
        layout=iw.Layout(width=SettingsController.BUTTON_WIDTH))
    self.load_button.on_click(lambda b: self.load_settings())

    # Refresh button
    self.refresh_button = iw.Button(description='Refresh Settings',
        layout=iw.Layout(width=SettingsController.TOTAL_WIDTH))
    self.refresh_button.on_click(lambda b: self.refresh_settings())

    # Load current settings from file
    self.refresh_settings()

    self._w = w

  def save_settings(self):
    text = self.save_widget.value
    self._w.save_settings(key=text)
    self.save_widget.value = ''
    self.refresh_settings()

  def load_settings(self):
    selected = self.load_widget.value
    self._w.load_settings(key=selected)

  def refresh_settings(self):
    with open(self.settings_file, 'r') as f:
      settings = json.load(f)
    options = settings.keys()
    self.load_widget.options = options

  @property
  def widget(self): 
    if self._orientation == 'vertical':
      container = iw.VBox
    else:
      container = iw.HBox
      self.refresh_button.layout.width = self.BUTTON_WIDTH
    
    save = iw.HBox([self.save_button, self.save_widget])
    load = iw.HBox([self.load_button, self.load_widget])
    refresh = self.refresh_button
    title = iw.HTML('<h2>Settings</h2>')
    
    return container([title, save, load, refresh], layout=self._layout)


class DetsumPlot:
    def __init__(self, detsum):
        self.graph_output = iw.Output()
        self.detsum = detsum
        self.avg = np.mean(self.detsum.data)
        self.min = np.nanmin(self.detsum.data)
        self.max = np.nanmax(self.detsum.data)
        self._default_layout = iw.Layout(width='230px')
        
        self.step = self.max/100
        
        self.isNoisy_checkbox = iw.Checkbox(description='Noisy', value=self.detsum.isNoisy)
        self.isNoisy_checkbox.observe(self.set_noise_flag, names='value')
        
        self.force_refresh_button = iw.Button(description='refresh')
        self.force_refresh_button.on_click(self.refresh_graph)
        
        
        self.update_max({'new': self.max})
        
    def set_noise_flag(self, val):
        utils.set_noisy_detsum_flag(self.detsum, val['new'])
        
        
    def refresh_graph(self, _):
        self.graph_output.clear_output(wait=True)
        with self.graph_output:
            im = iw.HTML(self.detsum.plot(base64=True, raw=True))
            display(im)

    def update_max_min(self, _):
      vmin = float(self.min_slider.value)
      vmax = float(self.max_slider.value)
      try:
        plot = self.detsum.plot(base64=True,
                                raw=True,
                                vmin=vmin,
                                vmax=vmax)
        to_display = iw.HTML(plot)
      except ValueError as e:
        to_display = iw.HTML(f'<p style="red">{str(e)}!</p>')
      self.graph_output.clear_output(wait=True) 
      with self.graph_output:
        display(to_display)
          
        
        
    def update_max(self, val):
        self.graph_output.clear_output(wait=True)
        vmax = float(val['new'])
        with self.graph_output:
            im = iw.HTML(self.detsum.plot(base64=True, vmax=vmax, raw=True))
            display(im)
                                 
    def update_min(self, val):
        self.graph_output.clear_output(wait=True)
        vmin = float(val['new'])
        with self.graph_output:
            im = iw.HTML(self.detsum.plot(base64=True, vmin=vmin, raw=True))
            display(im)
            
        
    @property
    def widget(self):
        self.max_slider = iw.FloatSlider(layout=self._default_layout,
                                             min=self.step, max=2*self.max, value=self.max, step=self.step,
                                             readout_format='.2e', continuous_update=False)
        
        self.min_slider = iw.FloatSlider(layout=self._default_layout,
                                             min=self.step, max=2*self.max, value=self.step, step=self.step,
                                             readout_format='.2e', continuous_update=False)
        
        # self.max_slider.observe(self.update_max, names='value')
        # self.min_slider.observe(self.update_min, names='value')
        self.max_slider.observe(self.update_max_min, names='value')
        self.min_slider.observe(self.update_max_min, names='value')
        min_label = iw.Text(f'min counts: {self.min:0.2e}')
        max_label = iw.Text(f'max counts: {self.max:0.2e}')
        avg_label = iw.Text(f'avg counts: {self.avg:0.2e}')
        labels = iw.HTML(f'''<center><p>min: {self.min:0.2e}</p>
                                     <p>max: {self.max:0.2e}</p>
                                     <p>avg: {self.avg:0.2e}</p></center>''',
                                layout=self._default_layout)
        #labels = iw.VBox([min_label, max_label, avg_label], layout=self._default_layout)
        
        vbox = iw.VBox([self.graph_output,
                                self.min_slider,
                                self.max_slider,
                                labels,
                                iw.HBox([self.force_refresh_button, self.isNoisy_checkbox])])
        return vbox


class NoiseFlaggingUI:
    def __init__(self, profile):
        self._tab_dict = {}
        for depth in profile.depths:
            plots = [DetsumPlot(d).widget for d in depth.detsums]
            rows = []
            for i in np.arange(0, len(plots), 4):
                rows.append(iw.HBox(plots[i:i+4]))
                
            self._tab_dict[depth.depth] = iw.VBox(rows)
            
            
    @property
    def widget(self):
        children = []
        tabs = iw.Tab()
        for i, depth in enumerate(self._tab_dict):
            tabs.set_title(i, depth)
            children.append(self._tab_dict[depth])
            
        tabs.children = children
        return tabs

def noise_flagging_ui(profile):
  nf = NoiseFlaggingUI(profile)
  display(nf.widget)

def ribbon_plot_ui(profile, elements_of_interest):
  experiment_dir = profile.experiment_dir
  status_indicator = iw.Output()
  with status_indicator:
    display(iw.HTML('<h3 style="color:green">Ready</h3>'))
  graph_output = iw.Output()
  # element_inputs = {}
  # element_filter = {}
  test = {}
  smalltextbox = iw.Layout(width='50px', height='25px')
  # filter_func = lambda n: lambda x: np.mean(x) + np.std(x)*n
  
  element_filter = ElementFilterPanel(profile,
                                    input_type='text',
                                    orientation='horizontal',
                                    experiment_dir=experiment_dir)#, **layout_kwargs)
  filter_settings = SettingsController(element_filter)
  
  # for e in ELEMENTS_OF_INTEREST:
  #   element_inputs[e] = iw.Textarea(value='2', layout=smalltextbox)
  #   element_filter[e] = filter_func(2)
    
  filter_by_control = iw.Dropdown(options=elements_of_interest,
                                          value='Cu', description='Filter by:',
                                          layout=iw.Layout(width='200px'))
  
  combine_scans_checkbox = iw.Checkbox(value=True, description='Combine Scans')
  
  combine_detsums_checkbox = iw.Checkbox(value=False, description='Combine Detsums')
  
  normalize_by_control = iw.Dropdown(options=['counts', 'pixels'],
                                            value='counts',
                                            description='Normalize By',
                                            layout=iw.Layout(width='200px'))
  
  
  N_input = iw.Textarea(value='8', layout=iw.Layout(width='150px'), description='N')
  update_button = iw.Button(description='Update Plot')                          
  clear_output_control = iw.Checkbox(value=False, description='Clear output after each run')
  
  # element_filter_input = iw.HBox(
  #     [iw.VBox([iw.HTML(f'<h3>{e}</h3>'), element_inputs[e]]) for e in ELEMENTS_OF_INTEREST]
  # )
  
  save_html_button = iw.Button(description='Save HTML')
  def update_plot(b):
    element_filter.save_settings()
    status_indicator.clear_output()
    # for e in ELEMENTS_OF_INTEREST:
    #   val = float(element_inputs[e].value)
    #   element_filter[e] = filter_func(val)
    with status_indicator:
      display(iw.HTML('<h3 style="color:red">Working...</h3>'))

    info_banner_html = (f'Filter by: {filter_by_control.value} | '
                      f'Comb. Scans: {combine_scans_checkbox.value} | '
                      f'Comb. Detsums: {combine_detsums_checkbox.value} | '
                      f'N: {N_input.value} | '
                      f'Normalize By: {normalize_by_control.value} | ')
    info_banner = iw.HTML(info_banner_html)

    plot = ribbon_plot(profile, element_filter=element_filter.filter_dict,
                filter_by=filter_by_control.value,
                combine_detsums=combine_detsums_checkbox.value,
                combine_scans=combine_scans_checkbox.value,
                N=int(N_input.value),
                normalize_by=normalize_by_control.value,
                base64=True)
    if clear_output_control.value:
      graph_output.clear_output()

    with graph_output:
      #display(iw.HTML(plot))
      display(iw.VBox([iw.HTML(plot), iw.HTML(info_banner_html)]))

    status_indicator.clear_output()
    with status_indicator:
      display(iw.HTML('<h3 style="color:green">Ready</h3>'))
  
  update_button.on_click(update_plot)
  

  update_plot('this param does not matter here')  

  #https://stackoverflow.com/questions/55336771/align-ipywidget-button-to-center
  controls_bot = iw.Box([element_filter.widget, filter_settings.widget],
                                layout=iw.Layout(display='flex', align_items='center'))
  controls_top = iw.HBox([iw.VBox([update_button, status_indicator]),
                              iw.VBox([filter_by_control, normalize_by_control]),
                              iw.VBox([combine_scans_checkbox, combine_detsums_checkbox]),
                              N_input, clear_output_control])

  controls = iw.VBox([controls_top, controls_bot],
                            layout=iw.Layout(
                                border='1px solid black',
                                width='100%',
                            ))
  app = iw.VBox([graph_output, controls])
  display(app)


def image_ui(profile, elements_of_interest):
  experiment_dir = profile.experiment_dir
  plot_area = iw.Output()
  status_indicator = iw.Output()
  group_indicator = iw.Output()
  
  with group_indicator:
    display(iw.HTML('<h3 style="color:orange">No Group Selected</h3>'))

  with status_indicator:
    display(iw.HTML('<h3 style="color:green">Ready</h3>'))

  
  # Making the controls

  layout_kwargs = dict(width='85%', border='1px solid black')
  #settings_layout = dict(width='20%', border='1px solid blue')

  depth_selector = PropSelector(profile.depths, orientation='horizontal', title='Depths to plot',
                                   description_func=lambda d: d.depth, **layout_kwargs)
  element_filter = ElementFilterPanel(profile,
                                    orientation='horizontal',
                                    experiment_dir=experiment_dir)

  filter_settings = SettingsController(element_filter)

  element_plot_selector = PropSelector(elements_of_interest,
                                          orientation='horizontal',
                                          title='Elements to plot',
                                          **layout_kwargs)

  element_group_selector = PropSelector(elements_of_interest,
                                           orientation='horizontal',
                                           title='Groups to show',
                                           **layout_kwargs)

  combine_detsums_checkbox = iw.Checkbox(value=False, indent=False, description='Combine Detsums')
  update_button = iw.Button(description='Update')
  raw_data_toggle = iw.ToggleButtons(value='Filtered', options=['Filtered', 'Raw'])
  show_groups_toggle = iw.ToggleButton(value=True, description='Show Groups')
  exclusive_groups_toggle = iw.ToggleButtons(value='Exclusive', options=['Exclusive', 'Nonexclusive'])

  controls_bottom = iw.HBox([update_button,
                                          show_groups_toggle,
                                          exclusive_groups_toggle,
                                          raw_data_toggle,
                                          #combine_detsums_checkbox,
                                          status_indicator],
                                         layout=iw.Layout(padding='5px', **layout_kwargs))


  controls_right = iw.HBox([
      element_filter.widget, iw.VBox([
        combine_detsums_checkbox,
        filter_settings.widget])
      ],
      layout=iw.Layout(**layout_kwargs))
  controls_top = iw.VBox([depth_selector.widget,
                                   element_plot_selector.widget,
                                   controls_right,
                                   #iw.HBox([group_indicator, show_groups_toggle]),
                                   element_group_selector.widget,
                                   group_indicator])

  # controls_right = iw.VBox([iw.HTML('Hightlight group\ncontaining elements:'),
  #                                   element_group_selector.widget],
  #                                  layout=iw.Layout(border='1px solid black'))

  # controls = iw.HBox([controls_left, controls_right], layout=iw.Layout(width='85%'))
  controls = iw.VBox([controls_top, controls_bottom])
                            
  rows = []
  rows_raw = []
  rows_groups_exclusive = []
  rows_groups_exclusive_raw = []
  rows_groups_nonexclusive = []
  rows_groups_nonexclusive_raw = []
  
  current_group = []


  def show_plots(val):
    plot_area.clear_output()
    show = show_groups_toggle.value
    exclusive = exclusive_groups_toggle.value == 'Exclusive'
    with plot_area:
      if raw_data_toggle.value == 'Raw':
        if show and exclusive:
          display(iw.VBox(rows_groups_exclusive_raw))
        elif show and not exclusive:
          display(iw.VBox(rows_groups_nonexclusive_raw))
        elif not show:
          display(iw.VBox(rows_raw))
      elif raw_data_toggle.value == 'Filtered':
        if show and exclusive:
          display(iw.VBox(rows_groups_exclusive))
        if show and not exclusive:
          display(iw.VBox(rows_groups_nonexclusive))
        elif not show:
          display(iw.VBox(rows))
        
    status_indicator.clear_output()
    with status_indicator:
      display(iw.HTML('<h3 style="color:green">Ready</h3>'))


  def update_group(element, val):
    nonlocal current_group
    if val:
      current_group.append(element)
    elif not val:
      current_group.remove(element)

    sorted_group = [elements_of_interest[elements_of_interest.index(element)] if
                                          element in current_group else None for element in elements_of_interest]
    sorted_group = list(filter(lambda x: x, sorted_group))
    group_indicator.clear_output()
    with group_indicator:
      if not current_group:
        display(iw.HTML('<h3 style="color:orange">No group selected</h3>'))
      else:
        group_str = '  |  '.join(sorted_group)
        display(iw.HTML(f'<h3 style="color:red">Group Selected: {group_str}</hp>'))
    current_group = sorted_group

  def generate_plots(b):
    element_filter.save_settings()
    status_indicator.clear_output()
    with status_indicator:
      display(iw.HTML('<h3 style="color:red">Working....</h3>'))

    depths_to_plot = depth_selector.selected_props
    if not depths_to_plot:
      status_indicator.clear_output()
      with status_indicator:
        display(iw.HTML('<h3 style="color:orange">No Depth Selected!</h3>'))
      return

    elements_to_plot = element_plot_selector.selected_props

    group = '|'.join(element_group_selector.selected_props)

    for depth in depths_to_plot:
      depth.apply_element_filter(element_filter.filter_dict[depth.depth],
                                 combine_detsums=combine_detsums_checkbox.value)

    def get_row(depth, elements, raw, show_groups, exclusive):
      detsums = sorted(depth.detsums, key=lambda d: d.element)
      plots = []
      #group = '|'.join(current_group)
      for scan in depth.scans:
        data = scan.data['element_group'].values.reshape(scan.detsums[0].shape)
        for detsum in scan.detsums:
          if detsum.element not in elements:
            continue
          fig, ax = plt.subplots(figsize=(15, 15))
          detsum.plot(raw=raw, ax=ax)
          if show_groups and current_group:
            fn = np.vectorize(lambda group: utils.check_groups(group, current_group, exclusive=exclusive))
            rows, cols = np.where(fn(data))
            ax.scatter(cols, rows, s=20, color='red')
          plot = encode_matplotlib_fig(fig)
          plt.close()
          plots.append(iw.HTML(plot))
      return plots

    nonlocal rows
    nonlocal rows_raw
    nonlocal rows_groups_exclusive
    nonlocal rows_groups_exclusive_raw
    nonlocal rows_groups_nonexclusive
    nonlocal rows_groups_nonexclusive_raw

    
    rows = [iw.HBox(get_row(depth, elements_to_plot, False, False, False)) for depth in depths_to_plot]
    rows_raw = [iw.HBox(get_row(depth, elements_to_plot, True, False, False)) for depth in depths_to_plot]
    rows_groups_exclusive = [iw.HBox(get_row(depth, elements_to_plot, False, True, True)) for depth in depths_to_plot]
    rows_groups_exclusive_raw = [iw.HBox(get_row(depth, elements_to_plot, True, True, True)) for depth in depths_to_plot]
    rows_groups_nonexclusive = [iw.HBox(get_row(depth, elements_to_plot, False, True, False)) for depth in depths_to_plot]
    rows_groups_nonexclusive_raw = [iw.HBox(get_row(depth, elements_to_plot, True, True, False)) for depth in depths_to_plot]

    show_plots(0)


  update_button.on_click(generate_plots)
  raw_data_toggle.observe(show_plots, 'value')
  show_groups_toggle.observe(show_plots, 'value')
  exclusive_groups_toggle.observe(show_plots, 'value')
  element_group_selector.observe(update_group)
  # for selector in depth_selectors.values():
  #   selector.observe(update_plot)
  display(iw.VBox([controls, plot_area]))


def run_ui(profile_dict, ui_func, title=None):
  widgets = []
  if title is not None:
    widgets.append(iw.HTML(f'<h2>{title}</h2>'))

  dropdown = iw.Dropdown(description='Select Profile', options=profile_dict)
  run_button = iw.Button(description='Run')
  text_output = iw.Output()
  widgets.append(iw.HBox([dropdown, run_button, text_output]))

  ui_output = iw.Output()
  widgets.append(ui_output)

  with text_output:
      display(iw.HTML('<div style="color:Green">Ready...</div>'))
  def run(_):
      text_output.clear_output()
      with text_output:
          display(iw.HTML('<div style="color:red">Working...</div>'))
      ui_output.clear_output()
      with ui_output:
          ui_func(dropdown.value)
      text_output.clear_output()
      with text_output:
          display(iw.HTML('<div style="color:green">Ready...</div>'))
  run_button.on_click(run)
  display(iw.VBox(widgets))
