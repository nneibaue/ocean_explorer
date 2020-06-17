import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cycler import cycler
import numpy as np
from io import BytesIO
import base64

def encode_matplotlib_fig(fig):
  buf = BytesIO()
  fig.savefig(buf, format='png')
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
                prop_dict=None):
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
    prop_dict: optional dictionary to store properties in for the whole notebook.
      If this is None, then properties will be created for each individual plot.
  '''

  fig, ax = plt.subplots(figsize=(16, 4))

  # Sort depth objects by depth
  depths = sorted(depths, key=lambda d: int(d.depth[:-1]), reverse=True)

  # Get a default list of 10 colors. I don't exactly understand *how* this
  # works, but I know that it *does* work. Here is the stack overflow for
  # this: https://stackoverflow.com/questions/49233855/matplotlib-add-more-colors-to-the-default-colors-in-axes-prop-cycle-manually
  colors = plt.cm.tab10(np.arange(10))

  patterns = [None, '///', '..', 'xx', '**', '+++', 'OO', '\\\\\\']
  color_cycler = cycler(facecolor=colors)
  pattern_cycler = cycler(hatch=patterns)

  # Multiplication of cyclers results in the 'outer product'
  # https://matplotlib.org/cycler/
  props = iter(pattern_cycler * color_cycler)

  if prop_dict is None:
    prop_dict = {}

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

  # Assigns a unique color for each group, if the color doesn't exist yet
  def update_prop_dict(groups):
    for g in groups.index:
      if g not in prop_dict:
        prop_dict[g] = next(props)
        prop_dict[g].update({'alpha': 0.8, 'fill': True, 'edgecolor': 'k'})
      #print(f'Adding {g} to prop_dict')

  # Plots single scan (one bar)
  def plot_scan(scan, i):
    groups = get_scan_groups(scan)

    # If groups are empy, do nothing
    if not len(groups.index):
      return
    update_prop_dict(groups)
    left = 0
    thickness = 1
    for group in groups.index:
      #Encode the group value (e.g. Cu|Mn -> 0.0778) as the rectangle width
      rect_width = groups[col].loc[group]
      ax.add_patch(mpatches.Rectangle((left, i-thickness/2),
                            rect_width, thickness,
                            **prop_dict[group]))
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
    
  # Make the legend using patches. Here is a helpful link
  # https://stackoverflow.com/questions/53849888/make-patches-bigger-used-as-legend-inside-matplotlib
  patches = []
  for group in prop_dict:
    patches.append(mpatches.Patch(
       **prop_dict[group], 
       label=group,
    ))
  leg = ax.legend(handles=patches,
                  bbox_to_anchor=(1.05, 1.2),
                  ncol=round(len(patches) / 10) or 1,
                  labelspacing=2,
                  bbox_transform=fig.transFigure)
  for patch in leg.get_patches():
    patch.set_height(22)
    patch.set_y(-10)
  
  ax.set_yticks(np.arange(0, i))

  ax.set_yticklabels(yticklabels)


  ax.set_ylim(-0.5, i-0.5)
  ax.set_title(f'{filter_by} {normalize_by}', fontsize=16, fontweight='bold')
  plt.close()
  if base64:
    return encode_matplotlib_fig(fig)
  return fig