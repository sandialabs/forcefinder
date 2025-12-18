forcefinder.core.response_limit
===============================

.. py:module:: forcefinder.core.response_limit

.. autoapi-nested-parse::

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



Classes
-------

.. autoapisummary::

   forcefinder.core.response_limit.ResponseLimit


Functions
---------

.. autoapisummary::

   forcefinder.core.response_limit.check_ascending
   forcefinder.core.response_limit.get_nesting_depth


Module Contents
---------------

.. py:function:: check_ascending(vector: Union[list, tuple, numpy.ndarray]) -> bool

   Checks if the supplied vector is sorted in an ascending order.

   :param vector: A 1d vector to evaluate.
   :type vector: list, tuple, or ndarray

   :returns: True if the vector is sorted in ascending order, otherwise
             False.
   :rtype: bool

   .. rubric:: Notes

   The supplied vector is cast as an ndarray to ensure consistent
   logic for the check.


.. py:function:: get_nesting_depth(data: Union[list, tuple]) -> int

   Determines the nesting depth (akin to the number of dimensions)
   for a list or tuple.

   :param data: The data to evaluate the nesting depth for.
   :type data: list or tuple

   :returns: **nesting_depth** -- The maximum nesting depth of the data.
   :rtype: it


.. py:class:: ResponseLimit(limit_coordinate: Union[sdynpy.CoordinateArray, list, tuple, str], breakpoint_frequency: Union[list, numpy.ndarray], breakpoint_level: Union[list, numpy.ndarray])

   A class to represent a response limit as a set of breakpoints for
   response limits in inverse problems.

   .. attribute:: limit_coordinate

      The responses DOFs that the limits are defined for.

      :type: CoordinateArray

   .. attribute:: breakpoint_frequency

      The frequency breakpoint for the limits, formatted as a list
      of 1d arrays.

      :type: list

   .. attribute:: breakpoint_level

      The level breakpoint for the limits, formatted as a list of
      1d arrays.

      :type: list


   .. py:method:: interpolate_to_full_frequency(full_frequency: numpy.ndarray, interpolation_type: str = 'loglog') -> numpy.ndarray

      Converts the limit breakpoints to a different frequency vector
      via linear interpolation.

      :param full_frequency: A 1d array of frequencies to interpolate the limit breakpoints
                             over. This array must be sorted in ascending order.
      :type full_frequency: ndarray
      :param interpolation_type: The type of interpolation to use when converting the breakpoints
                                 to full_frequencies. The options are loglog or linearlinear,
                                 depending on if the interpolation should result in straight lines
                                 on a log or linear scale. The default is loglog.
      :type interpolation_type: str, optional

      :returns: **full_limit** -- The limit, interpolated to the supplied frequencies. It is
                organized [number of limits, number of frequencies] where the
                limits are ordered the same as the ResponseLimit object.
      :rtype: ndarray

      :raises ValueError: If full_frequency is not supplied in ascending order.

      .. rubric:: Notes

      The interpolated limit outside at frequencies that our higher/lower
      than the breakpoint frequency limits are returned as NaN.

      Any zeros in full_frequency or the breakpoint information are converted
      to machine epsilon (for floats) if loglog interpolation is used.



