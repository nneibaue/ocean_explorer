import pytest
import ocean
import numpy as np

ELEMENTS = ['A', 'B', 'C']

@pytest.fixture
def profile():
  p = ocean.Profile('test_profile',
                    elements_of_interest=ELEMENTS)
  return p

@pytest.fixture
def depth_multiple_scans():
  d = ocean.Depth('test_profile/1m')
  return d

@pytest.fixture
def depth_single_scan():
  d = ocean.Depth('test_profile/5m')
  return d


class TestDetsum:
  fname = 'test_profile/1m/scan2D_1/detsum_A_K_norm.txt'
  d = ocean.Detsum(fname)

  def test_import_from_file(self):
    expected = np.array([[1, 1, 5, 5],
                         [1, 1, 10, 5],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]])
    np.testing.assert_array_equal(self.d._data_raw, expected) 

  def test_simple_mask(self):
    mask = np.array([[0, 1, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 1, 1, 0]])

    expected = np.array([[np.nan, 1, np.nan, 5],
                         [np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan],
                         [np.nan, 1, 1, np.nan]])
    self.d.add_mask(mask)
    np.testing.assert_array_equal(self.d.data, expected)
    

