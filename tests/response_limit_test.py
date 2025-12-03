"""
Includes the tests for the ResponseLimit object. 

Copyright 2025 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import sdynpy as sdpy
import forcefinder as ff
import pytest

def test_init_different_dof_types():
    """
    This test just makes sure that the basic init works with the 
    different input types for the dofs. It goes through all the 
    permutations of how data could be supplied
    """
    # Single Limit
    limit = ff.ResponseLimit('1X+', [1,2,3], [1,2,3])
    assert limit.limit_coordinate == '1X+'
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)

    limit = ff.ResponseLimit(['1X+'], [1,2,3], [1,2,3])
    assert limit.limit_coordinate == '1X+'
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)

    limit = ff.ResponseLimit(('1X+'), [1,2,3], [1,2,3])
    assert limit.limit_coordinate == '1X+'
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)

    limit = ff.ResponseLimit(sdpy.coordinate_array(string_array='1X+'), [1,2,3], [1,2,3])
    assert limit.limit_coordinate == '1X+'
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)

    # Multiple Limits
    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]])
    assert len(limit) == 2
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)
    assert limit.limit_coordinate.shape == (2,) 
    assert all(limit.limit_coordinate == ['1X+','2X+'])

    limit = ff.ResponseLimit(('1X+','2X+'), [[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]])
    assert len(limit) == 2
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)
    assert limit.limit_coordinate.shape == (2,) 
    assert all(limit.limit_coordinate == ['1X+','2X+'])

    limit = ff.ResponseLimit(sdpy.coordinate_array(string_array=['1X+','2X+']), [[1,2,3], [1,2,3]], [[1,2,3], [1,2,3]])
    assert len(limit) == 2
    assert isinstance(limit.limit_coordinate, sdpy.CoordinateArray)
    assert limit.limit_coordinate.shape == (2,) 
    assert all(limit.limit_coordinate == ['1X+','2X+'])

def test_init_different_frequency_types():
    """
    This test makes sure that the init works with the different 
    input types for the frequency. This includes constructing things
    properly as well as throwing the appropriate exceptions. It
    goes through all the permutations of how data could be supplied
    """
    # Single Limit
    limit = ff.ResponseLimit('1X+', [1,2,3], [1,2,3])
    assert all(limit.breakpoint_frequency == [1,2,3]) 
    assert isinstance(limit.breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit('1X+', (1,2,3), [1,2,3])
    assert all(limit.breakpoint_frequency == [1,2,3]) 
    assert isinstance(limit.breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit('1X+', np.array([1,2,3]), [1,2,3])
    assert all(limit.breakpoint_frequency == [1,2,3]) 
    assert isinstance(limit.breakpoint_frequency, np.ndarray)

    with pytest.raises(ValueError, match='The breakpoint frequencies must be supplied in ascending order'):
        limit = ff.ResponseLimit('1X+', [2,1,3], [1,2,3])

    with pytest.raises(ValueError, match='The breakpoint_frequency must have only one dimension if there is only one limit DOF'):
        limit = ff.ResponseLimit('1X+', [[1,2,3]], [1,2,3])

    with pytest.raises(ValueError, match='The breakpoint_frequency must have only one dimension if there is only one limit DOF'):
        limit = ff.ResponseLimit('1X+', np.array([1,2,3])[np.newaxis,:], [1,2,3])
    
    # Multiple Limits
    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [4,5,6,7]], [[1,2,3], [1,2,3,5]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [(1,2,3), (4,5,6,7)], [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [np.array([1,2,3]), [4,5,6,7]], [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [np.array([1,2,3]), np.array([4,5,6,7])], [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], ([1,2,3], [4,5,6,7]), [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], ((1,2,3), (4,5,6,7)), [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], (np.array((1,2,3)), (4,5,6,7)), [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], (np.array((1,2,3)), np.array((4,5,6,7))), [[1,2,3], [1,2,3,4]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert isinstance(limit[0].breakpoint_frequency, np.ndarray)
    assert all(limit[1].breakpoint_frequency == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_frequency, np.ndarray)

    with pytest.raises(ValueError, match='The frequencies must be supplied as a limit or tuple of lists or 1d arrays when there are more than one limit DOF'):
        limit = ff.ResponseLimit(['1X+','2X+'], np.array([[1,2,3], [4,5,6]]), [[1,2,3], [1,2,3]])
    
    with pytest.raises(ValueError, match='The breakpoint_frequency must have the same number of breakpoint vectors as the number of limit DOFs'):
        limit = ff.ResponseLimit(['1X+','2X+'], [1,2,3], [1,2,3])

    with pytest.raises(ValueError, match='The breakpoint frequencies must be supplied in ascending order'):
        limit = ff.ResponseLimit(['1X+','2X+'], [[2,1,3], [4,5,6]], [[1,2,3], [1,2,3]])

    with pytest.raises(ValueError, match='The limit must have the same number of breakpoint frequencies and levels'):
        limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3,4], [4,5,6,7]], [[1,2,3], [1,2,3,4]])

def test_init_different_level_types():
    """
    This test makes sure that the init works with the different 
    input types for the level. This includes constructing things
    properly as well as throwing the appropriate exceptions. It
    goes through all the permutations of how data could be supplied
    """
    # Single Limit
    limit = ff.ResponseLimit('1X+', [1,2,3], [1,2,3])
    assert all(limit.breakpoint_level == [1,2,3]) 
    assert isinstance(limit.breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit('1X+', (1,2,3), (1,2,3))
    assert all(limit.breakpoint_level == [1,2,3]) 
    assert isinstance(limit.breakpoint_frequency, np.ndarray)

    limit = ff.ResponseLimit('1X+', [1,2,3], np.array([1,2,3]))
    assert all(limit.breakpoint_level == [1,2,3]) 
    assert isinstance(limit.breakpoint_level, np.ndarray)

    with pytest.raises(ValueError, match='The breakpoint_level must have only one dimension if there is only one limit DOF'):
        limit = ff.ResponseLimit('1X+', [1,2,3], [[1,2,3]])

    with pytest.raises(ValueError, match='The breakpoint_level must have only one dimension if there is only one limit DOF'):
        limit = ff.ResponseLimit('1X+', [1,2,3], np.array([1,2,3])[np.newaxis,:])
    
    # Multiple Limits
    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,5]], [[1,2,3], [4,5,6,7]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], [(1,2,3), (4,5,6,7)])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], [np.array([1,2,3]), [4,5,6,7]])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], [np.array([1,2,3]), np.array([4,5,6,7])])
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], ([1,2,3], [4,5,6,7]))
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], ((1,2,3), (4,5,6,7)))
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], (np.array((1,2,3)), (4,5,6,7)))
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [1,2,3,4]], (np.array((1,2,3)), np.array((4,5,6,7))))
    assert len(limit) == 2
    assert all(limit[0].breakpoint_level == [1,2,3])
    assert isinstance(limit[0].breakpoint_level, np.ndarray)
    assert all(limit[1].breakpoint_level == [4,5,6,7])
    assert isinstance(limit[1].breakpoint_level, np.ndarray)

    with pytest.raises(ValueError, match='The levels must be supplied as a limit or tuple of lists or 1d arrays when there are more than one limit DOF'):
        limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3], [4,5,6]], np.array([[1,2,3], [1,2,3]]))
    
    with pytest.raises(ValueError, match='The breakpoint_level must have the same number of breakpoint vectors as the number of limit DOFs'):
        limit = ff.ResponseLimit(['1X+','2X+'], [[1,2,3],[4,5,6]], [1,2,3])

def test_indexing():
    """
    This checks the indexing on the ResponseLimit for some cases
    that weren't caught earlier.
    """
    limit = ff.ResponseLimit('1X+', [1,2,3], [4,5,6])

    # Testing the indexing on an object with length 1
    assert all(limit[0].breakpoint_frequency == [1,2,3])
    assert all(limit[0].breakpoint_level == [4,5,6])
    assert len(list(limit)) == 1

    # Testing that the correct exception is thrown when indexing past the size of the object
    with pytest.raises(IndexError, match='Index 1 is out of bounds for ResponseLimit with size 1'):
        _ = limit[1]

    dof = ['1X+', '2X+', '3X+', '4X+', '5X+', '6X+']
    frequency = [np.ones(6)*(ii+1)+[1,2,3,4,5,6] for ii in range(len(dof))]
    level = [np.ones(6)*(ii+1) for ii in range(len(dof))]
    limit = ff.ResponseLimit(dof, frequency, level)

    # Testing that the iterable function works in a loop
    for ii, single_limit in enumerate(limit):
        assert all(single_limit.limit_coordinate == dof[ii])
        assert all(single_limit.breakpoint_frequency == frequency[ii])
        assert all(single_limit.breakpoint_level == level[ii])

    # Testing that I get the same answer when iterating of the object multiple times
    iterate1 = list(limit)
    iterate2 = list(limit)
    for ii in range(6):
        assert all(iterate1[ii].limit_coordinate == iterate2[ii].limit_coordinate)
        assert all(iterate1[ii].breakpoint_frequency == iterate2[ii].breakpoint_frequency)
        assert all(iterate1[ii].breakpoint_level == iterate2[ii].breakpoint_level)

def test_interpolation():
    """
    This checks that the interpolation function works as expected. 
    """
    limit = ff.ResponseLimit('1X+', [1,100], [1,100])
    
    # Basic interpolation checks
    interpolated_limit = limit.interpolate_to_full_frequency([1,10,100], interpolation_type='linearlinear')
    assert all(interpolated_limit[0,:] == [1,10,100])

    del interpolated_limit
    interpolated_limit = limit.interpolate_to_full_frequency([1,10,100], interpolation_type='loglog')
    assert all(interpolated_limit[0,:] == [1,10,100])

    # Checking for NaNs outside the limit frequencies
    del interpolated_limit
    interpolated_limit = limit.interpolate_to_full_frequency([-1,0,1,10,100,110,120], 'linearlinear')
    assert all(np.isnan(interpolated_limit[0,-2:]))
    assert all(np.isnan(interpolated_limit[0,:2]))
    assert all(interpolated_limit[0,2:5] == [1,10,100])

    # Checking for correct interpolation within the limit frequencies
    del interpolated_limit
    in_bound_limit = ff.ResponseLimit('1X+', [1,4,8,12,16,20], [1,2,3,4,5,6])
    interpolated_limit = in_bound_limit.interpolate_to_full_frequency([10,11,12,13,14,15], 'linearlinear')
    assert all(interpolated_limit[0,:] == [3.5, 3.75, 4, 4.25, 4.5, 4.75])

    # Checking for the exception when the interpolation type is not available
    with pytest.raises(ValueError, match='The specified interpolation type is not available.'):
        _ = limit.interpolate_to_full_frequency([1,10,100], interpolation_type='linearlog')

    # Making sure an exception is raised when the supplied frequency is out of order
    with pytest.raises(ValueError, match='full_frequency must be supplied in ascending order'):
        _ = limit.interpolate_to_full_frequency([1,100,10])