import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cycler import cycler
import numpy as np

def ribbon_plot(depths,
                filter_by='Cu',
                ax=None,
                N=8,
                normalize_by='counts'):
  '''Shows fractional concentration of element among different groups.

  Args:
    filter_by: string. Element under investigation
    ax: plt.Axes. Optional Axes to plot on
    N: int. The amount of groups to use. For example, N=12 means "take the
      top 12 groups from each depth"
    normalize_by: string. Which quantity to use for data normalization. This
      can either be 'counts' or 'pixels'.
  '''
  if ax is None:
    fig, ax = plt.subplots(figsize=(16, 4))
  else:
    fig = ax.figure()

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


  prop_dict = {}

  for i, depth in enumerate(depths):
    scan = depth.combined_scan
    # Get the unique groups and the total pixels and counts
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

    groups = scan.filter_by(filter_by).get_unique_groups(elements_to_sum=[filter_by], sort_by=sort_by)
    groups[col] /= normalization(groups)

    # Gets the top N unique groups for this scan
    groups = groups.iloc[:N]

    # Assigns a unique color for each group, if the color doesn't exist yet
    for g in groups.index:
      if g not in prop_dict:
        prop_dict[g] = next(props)
        prop_dict[g].update({'alpha': 0.8, 'fill': True, 'edgecolor': 'k'})
        #print(f'Adding {g} to prop_dict')

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
                  ncol=len(patches) // 10,
                  labelspacing=2)
  for patch in leg.get_patches():
    patch.set_height(22)
    patch.set_y(-10)
  
  ax.set_yticks(np.arange(0, len(depths)))
  ax.set_yticklabels([d.depth for d in depths])
  ax.set_ylim(-0.5, len(depths)-0.5)
  ax.set_xlim(0, 1.13)
  ax.set_title(f'{filter_by} {normalize_by}', fontsize=16, fontweight='bold')
  plt.show()