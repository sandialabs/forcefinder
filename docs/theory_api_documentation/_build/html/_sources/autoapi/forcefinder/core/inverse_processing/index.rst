forcefinder.core.inverse_processing
===================================

.. py:module:: forcefinder.core.inverse_processing

.. autoapi-nested-parse::

   Defines the pre and post processing functions for the inverse methods
   in ForceFinder.

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



Functions
---------

.. autoapisummary::

   forcefinder.core.inverse_processing.linear_inverse_processing
   forcefinder.core.inverse_processing.power_inverse_processing
   forcefinder.core.inverse_processing.transient_inverse_processing


Module Contents
---------------

.. py:function:: linear_inverse_processing(method)

   This is a decorator function that does the pre and post processing to
   handle the response and reference transformations for the various inverse
   methods in the LinearSourcePathReceiver class.

   The inverse (class) method must return a force in an NDArray format, that is
   shaped [number of lines, number of forces, 1].


.. py:function:: power_inverse_processing(method)

   This is a decorator function that does the pre and post processing to
   handle the response and reference transformations for the various inverse
   methods in the PowerSourcePathReceiver class.

   The inverse (class) method must return a force in an NDArray format, that is
   shaped [number of lines, number of forces, number of forces].


.. py:function:: transient_inverse_processing(method)

   This is a decorator function that does the pre and post processing to
   handle the cola segmentation, response transformations, and reference
   transformations for the various inverse methods in the
   TransientSourcePathReceiver class.

   The inverse (class) method must return a force in an NDArray format, that is
   shaped [number of lines, number of forces, 1].


