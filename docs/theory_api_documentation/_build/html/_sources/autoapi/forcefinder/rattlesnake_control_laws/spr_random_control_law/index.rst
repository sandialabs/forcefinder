forcefinder.rattlesnake_control_laws.spr_random_control_law
===========================================================

.. py:module:: forcefinder.rattlesnake_control_laws.spr_random_control_law

.. autoapi-nested-parse::

   Defines the RandomControlSourcePathReceiver which is used for MIMO
   random vibration control.

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

   forcefinder.rattlesnake_control_laws.spr_random_control_law.RandomControlSourcePathReceiver


Functions
---------

.. autoapisummary::

   forcefinder.rattlesnake_control_laws.spr_random_control_law.parse_extra_parameters


Module Contents
---------------

.. py:function:: parse_extra_parameters(string)

   Parses the extra parameters from the Rattlesnake text box.

   :param string: The string of extra parameters from the Rattlesnake text box.
   :type string: str

   :returns: **extra_parameters** -- A dictionary with the parameters from the string parsed into the
             various optional arguments for the inverse. The potential keys
             are (the string in Rattlesnake should use the same names):
                 df : float
                     The frequency spacing for the control problem.
                 bp_freqs : ndarray
                     A 1D ndarray that defines the breakpoint frequencies for the
                     frequency dependent regularization.
                 ISE_technique : str
                     The ISE technique that is being used. This aligns with the
                     method names in the PowerSourcePathReceiver class.
                 inverse_method : str
                     The inverse method that is being used. This aligns with the
                     inverse methods that are in the "manual_inverse" method.
                 regularization_parameter : ndarray
                     A 1D ndarray that defines the breakpoint regularization parameters
                     (if Tikhonov regularization is being used) for the frequency
                     dependent regularization.
                 cond_num_threshold : ndarray
                     A 1D ndarray that defines the breakpoint condition number
                     threshold (if the TSVD is being used) for the frequency
                     dependent regularization.
                 num_retained_values : ndarray
                     A 1D ndarray that defines the breakpoint number of retained
                     singular values (if the TSVD is being used) for the frequency
                     dependent regularization.
                 number_regularization_values : int
                     The number of regularization values to search over in the auto
                     regularization methods. The default is 100.
                 l_curve_type : str
                     The type of L-curve that is used to find the "optimal regularization
                     parameter. The default depends on if the TSVD or Tikhonov regularization
                     are being used and the options are:

                         - forces
                             This L-curve is constructed with the "size" of the forces on the
                             Y-axis and the regularization parameter on the X-axis.

                         - standard
                             This L-curve is constructed with the residual squared error on
                             the X-axis and the "size" of the forces on the Y-axis.

                 optimality_condition : str
                     The method that is used to find an "optimal" regularization parameter.
                     The default depends on if the TSVD or Tikhonov regularization
                     is being used and the options are:

                         - curvature
                             This method searches for the regularization parameter that
                             results in maximum curvature of the L-curve. It is also referred
                             to as the L-curve criterion.

                         - distance
                             This method searches for the regularization parameter that
                             minimizes the distance between the L-curve and a "virtual origin".
                             A virtual origin is used, because the L-curve is scaled and offset
                             to always range from zero to one, in this case.

                 match_trace : bool
                     Whether or not to apply a match trace update during the control.
                 use_buzz : bool
                     Whether or not to use the buzz method for cross-term modification.
   :rtype: dict

   :raises ValueError: If the df argument isn't supplied.
   :raises ValueError: If the ISE_technique argument isn't supplied.
   :raises ValueError: If the use_buzz argument isn't supplied.

   .. rubric:: Notes

   The string is set-up by typing the parameters into the Rattlesnake text
   box. It should be set-up so that each return parameter is set equal to a
   value with a line break between each value, for example:

       df = 0.1

       bp_freqs = 10, 500, 2000

       ISE_technique = manual_inverse

       inverse_method = threshold

       cond_num_threshold = 50, 1000, 300

       use_buzz = True


.. py:class:: RandomControlSourcePathReceiver(specification: numpy.ndarray, warning_levels: numpy.ndarray, abort_levels: numpy.ndarray, extra_control_parameters: str, transfer_function: numpy.ndarray = None, noise_response_cpsd: numpy.ndarray = None, noise_reference_cpsd: numpy.ndarray = None, sysid_response_cpsd: numpy.ndarray = None, sysid_reference_cpsd: numpy.ndarray = None, multiple_coherence: numpy.ndarray = None, frames=None, total_frames=None, last_response_cpsd: numpy.ndarray = None, last_output_cpsd: numpy.ndarray = None)

   Bases: :py:obj:`forcefinder.PowerSourcePathReceiver`


   A subclass of the PowerSourcePathReceiver that is used in Rattlesnake random
   vibration control.

   .. rubric:: Notes

   This subclass technically shares all the attributes of the PowerSourcePathReceiver
   class. However, many of the private variables are left empty, because that data
   is not available in Rattlesnake (such as all the coordinate arrays). This means
   that this subclass works on all the private variables and is set-up such that it
   only uses methods, which only use the super class private variables.


   .. py:method:: system_id_update(transfer_function: numpy.ndarray = None, noise_response_cpsd: numpy.ndarray = None, noise_reference_cpsd: numpy.ndarray = None, sysid_response_cpsd: numpy.ndarray = None, sysid_reference_cpsd: numpy.ndarray = None, multiple_coherence: numpy.ndarray = None, frames=None, total_frames=None)

      Updates the system ID data throughout the test and performs the inverse source
      estimation as the system ID data is updated.

      :param transfer_function: The the training FRF array, organized with frequency on the first axis.
      :type transfer_function: ndarray, optional
      :param noise_response_cpsd: The response CPSD array that is measured during the noise floor check. This
                                  is not used in the SourcePathReceiver control law.
      :type noise_response_cpsd: ndarray, optional
      :param noise_reference_cpsd: The reference CPSD array that is measured during the noise floor check. This
                                   is not used in the SourcePathReceiver control law.
      :type noise_reference_cpsd: ndarray, optional
      :param sysid_response_cpsd: The response CPSD array that is measured during the system ID. This is used
                                  for the buzz cpsd array in the SourcePathReceiver class.
      :type sysid_response_cpsd: ndarray, optional
      :param sysid_reference_cpsd: The reference CPSD array that is measured during the system ID. This is not
                                   used in the SourcePathReceiver control law.
      :type sysid_reference_cpsd: ndarray, optional
      :param multiple_coherence: The multiple coherence that is measured during the system ID.
      :type multiple_coherence: ndarray, optional
      :param frames: The number of measurement frames aquired so far during the test.
      :type frames: int
      :param total_frames: The total number of measurement frames that will be used to comput the
                           averaged CPSD and FRF arrays.
      :type total_frames: int

      .. rubric:: Notes

      The use_transformation parameter in the inverse method must be set to false
      in the inverse source estimation since the transformations are done in
      Rattlesnake and the transformation arrays are not available to the control
      class.



   .. py:method:: control(transfer_function: numpy.ndarray, multiple_coherence: numpy.ndarray, frames, total_frames, last_response_cpsd: numpy.ndarray, last_output_cpsd: numpy.ndarray)

      Supply the drive voltage signals to Rattlesnake for the control.

      :param transfer_function: The the training FRF array, organized with frequency on the first axis.
      :type transfer_function: ndarray, optional
      :param multiple_coherence: The multiple coherence that is measured during the system ID.
      :type multiple_coherence: ndarray, optional
      :param frames: The number of measurement frames aquired so far during the test.
      :type frames: int
      :param total_frames: The total number of measurement frames that will be used to comput the
                           averaged CPSD and FRF arrays.
      :type total_frames: int
      :param last_response_cpsd: The last response cpsd array that was measured in the test, organized with
                                 frequency on the first axis.
      :type last_response_cpsd: ndarray, optional
      :param last_output_cpsd: The last output cpsd array that was estimated from the control, organized
                               with frequency on the first axis.
      :type last_output_cpsd: ndarray, optional

      :returns: The updated force array from the class.
      :rtype: ndarray

      .. rubric:: Notes

      The only algorithms in this method should be for "drive updates", such
      as the match trace method.



