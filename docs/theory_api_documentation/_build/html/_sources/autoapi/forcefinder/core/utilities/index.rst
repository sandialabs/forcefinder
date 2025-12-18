forcefinder.core.utilities
==========================

.. py:module:: forcefinder.core.utilities

.. autoapi-nested-parse::

   Creates some utility functions for the ForceFinder package.

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

   forcefinder.core.utilities.check_frequency_abscissa
   forcefinder.core.utilities.compare_sampling_rate
   forcefinder.core.utilities.is_cpsd
   forcefinder.core.utilities.apply_buzz_method
   forcefinder.core.utilities.reduce_drives_condition_not_met


Module Contents
---------------

.. py:function:: check_frequency_abscissa(data, reference_abscissa)

   Checks the abscissa of the data for building a source-path-receiver
   model. It validates that the data has a common abscissa for all
   the degrees of freedom and that the abscissa for data and reference
   data match.

   :param data: The data to check the abscissa on.
   :type data: NDDataArray
   :param reference_abscissa: The reference abscissa to compare the data abscissa against.
   :type reference_abscissa: ndarray

   :raises ValueError: If the abscissa from data doesn't match reference_data.


.. py:function:: compare_sampling_rate(time_data, reference_sampling_rate)

   Checks that the sampling rate of the supplied time_data matches the
   reference_reference_sampling tate. The primary purpose of this is
   to ensure that the response/force and FRFs have the same sampling
   rate when constructing a transient source-path-receiver model.

   :param time_data: The data to check the sampling rate on.
   :type time_data: TimeHistoryArray
   :param reference_sampling_rate: The reference sampling rate to compare against
   :type reference_sampling_rate: float

   :raises ValueError: If the sampling rate in time_data doesn't match the reference


.. py:function:: is_cpsd(data_array)

   Function to check if the supplied data array contains PSDs or CPSDs.

   :param data_array: data array to check.
   :type data_array: PowerSpectralDensityArray

   :returns: True if the data array contains CPSDs, False otherwise
   :rtype: bool


.. py:function:: apply_buzz_method(self)

   Applies the buzz method using the information in the SPR object.

   .. rubric:: References

   .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation,
           Aerospace Department, University of Bristol, 2014


.. py:function:: reduce_drives_condition_not_met(training_psd, reconstructed_psd, reduced_drive_reconstructed_psd, db_error_ratio)

   Evaluates the reconstructed response from the reduce_drives_update to
   see if the optimality condition was met.

   :param training_psd: The training PSDs for the SPR, which is the diagonal of the training
                        response. It should be shaped [number of lines, number of dofs].
   :type training_psd: ndarray
   :param reconstructed_psd: The reconstructed training PSDs for the SPR with the non-reduced forces,
                             which is the diagonal of the reconstructed training response. It should
                             be shaped [number of lines, number of dofs].
   :type reconstructed_psd: ndarray
   :param reconstructed_psd: The reconstructed training PSDs for the SPR with the reduced forces,
                             which is the diagonal of the reconstructed training response. It should
                             be shaped [number of lines, number of dofs].
   :type reconstructed_psd: ndarray
   :param db_error_ratio: The dB error ratio that was used in the reduced drives update.
   :type db_error_ratio: float

   :returns: Returns True if the reduce drives optimality condition is not met. Returns
             False if the condition is meth
   :rtype: bool


