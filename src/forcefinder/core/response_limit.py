"""
Defines the ResponseLimit class which is used for response limiting in
the PowerSourcePathReceiver.

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
from __future__ import annotations
import sdynpy as sdpy
import numpy as np
from typing import Union, Iterator

def check_ascending(vector: Union[list, tuple, np.ndarray]) -> bool:
    """
    Checks if the supplied vector is sorted in an ascending order.

    Parameters
    ----------
    vector : list, tuple, or ndarray
        A 1d vector to evaluate. 

    Returns
    -------
    bool 
        True if the vector is sorted in ascending order, otherwise
        False.
    
    Notes
    -----
    The supplied vector is cast as an ndarray to ensure consistent 
    logic for the check.
    """
    vector = np.atleast_1d(np.array(vector, dtype=float))
    return all(vector[ii] < vector[ii+1] for ii in range(len(vector)-1))

def get_nesting_depth(data : Union[list, tuple]) -> int:
    """
    Determines the nesting depth (akin to the number of dimensions) 
    for a list or tuple.

    Parameters
    ----------
    data : list or tuple
        The data to evaluate the nesting depth for.

    Returns
    -------
    nesting_depth : it
        The maximum nesting depth of the data.
    """
    # Non-list elements have a depth of 0, need this for when the 
    # recursion hits the end of the nesting
    if not isinstance(data, (list, tuple)):
        return 0

    # Magical recursive function call
    nesting_depth = 0
    for item in data:
        nesting_depth = max(nesting_depth, get_nesting_depth(item))

    return 1 + nesting_depth

class ResponseLimit:
    """
    A class to represent a response limit as a set of breakpoints for 
    response limits in inverse problems. 

    Attributes
    ----------
    limit_coordinate : CoordinateArray
        The responses DOFs that the limits are defined for.
    breakpoint_frequency : list
        The frequency breakpoint for the limits, formatted as a list 
        of 1d arrays.
    breakpoint_level : list
        The level breakpoint for the limits, formatted as a list of 
        1d arrays.
    """
    def __init__(self, limit_coordinate: Union[sdpy.CoordinateArray, list, tuple, str], 
                 breakpoint_frequency: Union[list, np.ndarray], 
                 breakpoint_level: Union[list, np.ndarray]):
        """
        Basic set-up for the ResponseLimit class.

        Parameters
        ----------
        limit_coordinate : CoordinateArray, list, tuple, or str
            The responses DOFs that the limits are defined for. This can
            be defined as a SDynPy CoordinateArray, a list of strings, 
            a tuple of strings, or a string (when there is only one DOF 
            in the limit). The DOF strings should be supplied with a node
            number and direction, i.e., 201X+. Note that the list, tuple, 
            or CoordinateArray should be one dimensional, i.e., 
            ['1X+', '2X+'] not [['1X+'], ['2X+']]
        breakpoint_frequency : list
            The frequency breakpoint for the limits, formatted as a list 
            of lists of 1d arrays. These breakpoints must be supplied in 
            ascending order, i.e., [1,2,3] not [1,3,2] or [3,2,1]. 
        breakpoint_level : list
            The level breakpoint for the limits, formatted as a list of 
            lists of 1d arrays.

        Notes
        -----
        This function assumes that there is only one limit if limit_coordinate
        is supplied as a string. 
        """
        if isinstance(limit_coordinate, str):
            self._number_limits_ = int(1)
            self.limit_coordinate = np.atleast_1d(sdpy.coordinate_array(string_array=limit_coordinate))
        elif isinstance(limit_coordinate, (list, tuple)):
            self._number_limits_ = len(limit_coordinate)
            self.limit_coordinate = sdpy.coordinate_array(string_array=limit_coordinate)
        elif isinstance(limit_coordinate, sdpy.CoordinateArray):
            self._number_limits_ = np.atleast_1d(limit_coordinate).shape[0]
            self.limit_coordinate = np.atleast_1d(limit_coordinate)
        else:
            raise ValueError('The limit_coordinate must be supplied as a string, list, tuple, or SDynPy CoordinateArray')

        if self._number_limits_ == 1:
            # Setting the breakpoint frequencies and levels, with some format checking, if they are provided as lists or tuples
            if isinstance(breakpoint_frequency, (list,tuple)):
                if get_nesting_depth(breakpoint_frequency) != 1:
                    raise ValueError('The breakpoint_frequency must have only one dimension if there is only one limit DOF')
                self.breakpoint_frequency = np.array(breakpoint_frequency, dtype=float)
            if isinstance(breakpoint_level, (list,tuple)):
                if get_nesting_depth(breakpoint_level) != 1:
                    raise ValueError('The breakpoint_level must have only one dimension if there is only one limit DOF')
                self.breakpoint_level = np.array(breakpoint_level, dtype=float)

            # Setting the breakpoint frequencies and levels, with some format checking, if they are provided as ndarrays
            if isinstance(breakpoint_frequency, np.ndarray):
                if breakpoint_frequency.ndim != 1:
                    raise ValueError('The breakpoint_frequency must have only one dimension if there is only one limit DOF')
                self.breakpoint_frequency = breakpoint_frequency.astype(float)
            if isinstance(breakpoint_level, np.ndarray):
                if breakpoint_level.ndim != 1:
                    raise ValueError('The breakpoint_level must have only one dimension if there is only one limit DOF')
                self.breakpoint_level = breakpoint_level.astype(float)

            if not check_ascending(self.breakpoint_frequency):
                raise ValueError('The breakpoint frequencies must be supplied in ascending order')
        else: 
            if not isinstance(breakpoint_frequency, (list, tuple)):
                raise ValueError('The frequencies must be supplied as a limit or tuple of lists or 1d arrays when there are more than one limit DOF')
            if not isinstance(breakpoint_level, (list, tuple)):
                raise ValueError('The levels must be supplied as a limit or tuple of lists or 1d arrays when there are more than one limit DOF')
            if self._number_limits_ != len(breakpoint_frequency):
                raise ValueError('The breakpoint_frequency must have the same number of breakpoint vectors as the number of limit DOFs')
            if self._number_limits_ != len(breakpoint_level):
                raise ValueError('The breakpoint_level must have the same number of breakpoint vectors as the number of limit DOFs')
            
            self.breakpoint_frequency = []
            self.breakpoint_level = []
            for ii in range(self._number_limits_):
                # Things are done in this order for the function to be agnostic to the 
                # breakpoint frequencies and levels being lists, tuples, or ndarrays
                # inside the supplied list/tuple.
                loop_frequency = np.array(breakpoint_frequency[ii], dtype=float)
                loop_level = np.array(breakpoint_level[ii], dtype=float)

                if loop_frequency.shape[0] != loop_level.shape[0]:
                    raise ValueError('The limit must have the same number of breakpoint frequencies and levels')
                if not check_ascending(loop_frequency):
                    raise ValueError('The breakpoint frequencies must be supplied in ascending order')
                
                self.breakpoint_frequency.append(loop_frequency)
                self.breakpoint_level.append(loop_level)

    def __len__(self) -> int:
        return self._number_limits_
    
    def __getitem__(self, idx: int) -> ResponseLimit:
        if isinstance(idx, int):
            if idx > len(self)-1:
                raise IndexError('Index {:} is out of bounds for ResponseLimit with size {:}'.format(idx, len(self)))
            
            if len(self) == 1:
                return self
            else:    
                return ResponseLimit(self.limit_coordinate[idx],
                                     self.breakpoint_frequency[idx],
                                     self.breakpoint_level[idx])
        else:
            raise TypeError('The ResponseLimit class can only be indexed by single integers')
        
    def __iter__(self) -> Iterator[ResponseLimit]:
        for ii in range(self._number_limits_):
            yield self[ii]

    def __repr__(self):
        return 'Response limit object with {:} limit DOFs'.format(self._number_limits_)
    
    def interpolate_to_full_frequency(self, full_frequency: np.ndarray, 
                                      interpolation_type: str = 'loglog') -> np.ndarray:
        """
        Converts the limit breakpoints to a different frequency vector 
        via linear interpolation.

        Parameters
        ----------
        full_frequency : ndarray
            A 1d array of frequencies to interpolate the limit breakpoints
            over. This array must be sorted in ascending order. 
        interpolation_type : str, optional
            The type of interpolation to use when converting the breakpoints
            to full_frequencies. The options are loglog or linearlinear, 
            depending on if the interpolation should result in straight lines
            on a log or linear scale. The default is loglog.

        Returns
        -------
        full_limit : ndarray
            The limit, interpolated to the supplied frequencies. It is 
            organized [number of limits, number of frequencies] where the 
            limits are ordered the same as the ResponseLimit object. 

        Raises
        ------
        ValueError
            If full_frequency is not supplied in ascending order. 

        Notes
        -----
        The interpolated limit outside at frequencies that our higher/lower
        than the breakpoint frequency limits are returned as NaN. 

        Any zeros in full_frequency or the breakpoint information are converted
        to machine epsilon (for floats) if loglog interpolation is used.
        """
        # ensuring that full_frequency is the correct format
        full_frequency = np.array(full_frequency, dtype=float)
        if not check_ascending(full_frequency):
            raise ValueError('full_frequency must be supplied in ascending order')
        
        full_limit = np.zeros((len(self), full_frequency.shape[0]), dtype=float)

        for ii, limit in enumerate(self):
            if interpolation_type == 'linearlinear':
                full_limit[ii,...] = np.interp(full_frequency, 
                                               limit.breakpoint_frequency, 
                                               limit.breakpoint_level,
                                               left=np.nan, right=np.nan)
            elif interpolation_type == 'loglog':
                # Converting zeros to machine epsilon for proper interpolation 
                if np.any(full_frequency==0):
                    full_frequency[full_frequency==0] = np.finfo(float).eps 
                if np.any(limit.breakpoint_frequency==0):
                    limit.breakpoint_frequency[limit.breakpoint_frequency==0] = np.finfo(float).eps
                if np.any(limit.breakpoint_level==0):
                    limit.breakpoint_level[limit.breakpoint_level==0] = np.finfo(float).eps

                full_limit[ii,...] = 10**(np.interp(np.log10(full_frequency), 
                                                    np.log10(limit.breakpoint_frequency), 
                                                    np.log10(limit.breakpoint_level),
                                                    left=np.nan, right=np.nan))
            else:
                    raise ValueError('The specified interpolation type is not available.')
        return full_limit
