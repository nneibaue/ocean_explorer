import pytest
import ocean
import numpy as np
import matplotlib
import pandas as pd

ELEMENTS = ['A', 'B', 'C']

class TestDetsum:
  fname = 'test_profile/1m/scan2D_1/detsum_A_K_norm.txt'

  def test_instantiate_from_file(self):
    d = ocean.Detsum(self.fname)
    expected = np.array([[1, 1, 5, 5],
                         [1, 1, 10, 5],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]])
    np.testing.assert_array_equal(d._data_raw, expected) 
    
  def test_total_counts_correct(self):
    d = ocean.Detsum(self.fname)
    assert d.total_counts == 37

  def test_simple_mask(self):
    d = ocean.Detsum(self.fname)
    mask = np.array([[0, 1, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 1, 1, 0]])

    expected = np.array([[np.nan, 1, np.nan, 5],
                         [np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan],
                         [np.nan, 1, 1, np.nan]])
    d.add_mask(mask)
    np.testing.assert_array_equal(d.data, expected)
  
  def test_plot_with_defaults_does_not_crash(self):
    d = ocean.Detsum(self.fname)
    fig = d.plot()
    assert isinstance(fig, matplotlib.figure.Figure)

class TestScan:
  fname = 'test_profile/1m/scan2D_1'
    
  def test_instantiate_from_file_with_defaults(self):
    s = ocean.Scan(self.fname)
    assert isinstance(s, ocean.Scan)
    assert s.scan_number == '1'
    assert s.depth == '1m'
    assert s.name == 'scan2D_1'
    assert len(s.detsums) == 3
    assert s.elements == ['A', 'B', 'C']
    assert isinstance(s.concentrations, pd.DataFrame)

  def test_instantiate_from_file_two_elements(self):
    s = ocean.Scan(self.fname, elements_of_interest=['A', 'B'])
    assert len(s.detsums) == 2
    assert s.elements == ['A', 'B']

  def test_data_with_no_masks_has_single_element_group(self):
    s = ocean.Scan(self.fname)
    assert isinstance(s.data, pd.DataFrame)

    element_groups = s.data['element_group'].unique()
    assert len(element_groups) == 1
    assert element_groups[0] == 'A|B|C'

  def test_data_with_masks_has_correct_element_groups(self):
    s = ocean.Scan(self.fname)

    # filter out all points less than 4
    for d in s.detsums: 
      d.add_mask(d.data > 4)

    element_groups = s.data['element_group'].dropna().unique()
    assert len(element_groups) == 3 # A, A|B, B
    unique_groups_df = s.get_unique_groups(elements_to_sum=['A', 'B', 'C'],
                                           sort_by='A_counts')

    assert unique_groups_df.loc['A']['A_counts'] == 15.0
    assert unique_groups_df.loc['A']['B_counts'] == 0.0
    assert unique_groups_df.loc['A']['C_counts'] == 0.0
    
    assert unique_groups_df.loc['B']['A_counts'] == 0.0
    assert unique_groups_df.loc['B']['B_counts'] == 15.0
    assert unique_groups_df.loc['B']['C_counts'] == 0.0
    
    assert unique_groups_df.loc['A|B']['A_counts'] == 10.0
    assert unique_groups_df.loc['A|B']['B_counts'] == 5.0
    assert unique_groups_df.loc['A|B']['C_counts'] == 0.0

  def test_filter_by(self):
    s = ocean.Scan(self.fname)
    
    # filter out all points less than 4
    for d in s.detsums:
      d.add_mask(d.data > 4)

    # filter by A
    s2 = s.filter_by('A')

    b_expected = np.array([[np.nan, np.nan, np.nan, np.nan],
                           [np.nan, np.nan,  5., np.nan],
                           [np.nan, np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.nan, np.nan]])

    np.testing.assert_array_equal(s2.get_detsum('B').data, b_expected)




class TestDepth:
    
    def test_instantiation_single_scan(self):
      fname = 'test_profile/5m'
      d = ocean.Depth(fname)
      assert isinstance(d, ocean.Depth)
      assert d.depth == '5m'
      assert len(d.scans) == 1
      assert len(d.detsums) == 3

    def test_instantiation_multiple_scans(self):
      fname = 'test_profile/1m'
      d = ocean.Depth(fname)
      assert isinstance(d, ocean.Depth)
      assert d.depth == '1m'
      assert len(d.scans) == 3
      assert len(d.detsums) == 9


    def test_apply_element_filter_simple_threshold(self):
      fname = 'test_profile/1m'
      d = ocean.Depth(fname)
      filter_func_1 = lambda x: 4.0
      filter_func_2 = lambda x: 15.0 
      filter_func_3 = lambda x: 7.0

      element_filter = {
        '1': {
          'A': filter_func_1,
          'B': filter_func_1,
          'C': filter_func_1},
        '2': {
          'A': filter_func_2,
          'B': filter_func_2,
          'C': filter_func_2},
        '3': {
          'A': filter_func_3,
          'B': filter_func_3,
          'C': filter_func_3}
      }

      d.apply_element_filter(element_filter)
        
      a1_expected = np.array([[np.nan, np.nan, 5.0, 5.0],
                              [np.nan, np.nan,  10.0, 5.0],
                              [np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]])
                              
      b1_expected = np.array([[5.0, np.nan, np.nan, np.nan],
                              [np.nan, np.nan,  5.0, np.nan],
                              [np.nan, np.nan, np.nan, np.nan],
                              [5.0, np.nan, np.nan, 5.0]])

      c1_expected = np.array([[np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan,  np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]])

      a2_expected = np.array([[np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan,  50.0, np.nan],
                              [np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]])
                              
      b2_expected = np.array([[np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan,  np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]])

      c2_expected = np.array([[np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan,  np.nan, np.nan],
                              [50.0, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]])

      a3_expected = np.array([[np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan,  np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan],
                              [np.nan, 10.0, np.nan, np.nan]])
                              
      b3_expected = np.array([[10.0, 10.0, 10.0, 10.0],
                              [10.0, 10.0,  10.0, np.nan],
                              [10.0, 10.0, 10.0, 10.0],
                              [10.0, 10.0, np.nan, np.nan]])

      c3_expected = np.array([[10.0, np.nan, np.nan, np.nan],
                              [10.0, np.nan,  np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]])

      s1 = d.get_scan('1')
      s2 = d.get_scan('2')
      s3 = d.get_scan('3')
      
      np.testing.assert_array_equal(s1.get_detsum('A').data, a1_expected)
      np.testing.assert_array_equal(s1.get_detsum('B').data, b1_expected)
      np.testing.assert_array_equal(s1.get_detsum('C').data, c1_expected)
      
      np.testing.assert_array_equal(s2.get_detsum('A').data, a2_expected)
      np.testing.assert_array_equal(s2.get_detsum('B').data, b2_expected)
      np.testing.assert_array_equal(s2.get_detsum('C').data, c2_expected)

      np.testing.assert_array_equal(s3.get_detsum('A').data, a3_expected)
      np.testing.assert_array_equal(s3.get_detsum('B').data, b3_expected)
      np.testing.assert_array_equal(s3.get_detsum('C').data, c3_expected)

