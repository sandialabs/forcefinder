forcefinder.core.source_path_receiver
=====================================

.. py:module:: forcefinder.core.source_path_receiver

.. autoapi-nested-parse::

   Defines the SourcePathReceiver which is used for MIMO vibration test
   simulation or force reconstruction.

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

   forcefinder.core.source_path_receiver.SourcePathReceiver
   forcefinder.core.source_path_receiver.LinearSourcePathReceiver
   forcefinder.core.source_path_receiver.PowerSourcePathReceiver
   forcefinder.core.source_path_receiver.TransientSourcePathReceiver


Module Contents
---------------

.. py:class:: SourcePathReceiver(frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, training_frfs=None, response_transformation=None, reference_transformation=None, empty=False)

   A class to represent a source-path-receiver (SPR) model of a system for MIMO
   vibration testing or transfer path analysis. It is primarily intended to manage
   all the book keeping for the complex problem set-up.

   This is the base SPR class that is further defined for specific data types in
   subclasses.

   .. rubric:: Notes

   The ordinate in the full FRFs and target responses can be different for the ordinate
   in the training FRFs and responses (depending on the problem set-up).


   .. py:method:: save(filename)

      Saves the SourcePathReceiver object to a .npz file.

      :param filename: The file path and name for the .npz file
      :type filename: str

      .. rubric:: Notes

      The private properties of the class are saved as arguments in the .npz file, where
      the argument names match the private variable name.



   .. py:method:: load(filename)
      :classmethod:


      Loads the SourcePathReceiver object from an .npz file.

      :param filename: The file path and name for the .npz file
      :type filename: str

      .. rubric:: Notes

      The private properties of the class should have been saved as arguments in the .npz
      file, where the argument names match the private variable name.



   .. py:property:: frfs

      This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.


   .. py:property:: training_frfs

      This produces a copy of the FRFs for the training_response_coordinate as a SDynPy
      TransferFunctionArray. Any modifications to the copy will not modify the original object.


   .. py:property:: target_frfs

      This produces a copy of the FRFs for the target_response_coordinate as a SDynPy
      TransferFunctionArray. Any modifications to the copy will not modify the original object.


   .. py:property:: validation_frfs

      This produces a copy of the FRFs for the target_response_coordinate as a SDynPy
      TransferFunctionArray. Any modifications to the copy will not modify the original object.


   .. py:property:: transformed_training_frfs

      the training FRFs with the transformations applied (i.e., what is used in the inverse problem).


   .. py:method:: apply_response_weighting(physical_response_weighting=None, transformed_response_weighting=None)

      Applys a weighting to the rows or columns of the response
      transformation array.

      :param physical_response_weighting: The weighting to be applied to the physical response coordinates
                                          (i.e., the columns) of the response transformation array. The
                                          weightings should be ordered the same as the "training_response_coordinate"
                                          of the SPR object.
      :type physical_response_weighting: ndarray, optional
      :param transformed_response_weighting: The weighting to be applied to the transformed response
                                             coordinates (i.e., the rows) of the response transformation array.
                                             The weightings should be ordered the same as the "transformed_response_coordinate"
                                             of the SPR object.
      :type transformed_response_weighting: ndarray, optional

      :returns: **self** -- The SourcePathReceiver object with the response transformation that
                has been updated
      :rtype: SourcePathReceiver

      :raises ValueError: If any of the following occurs:
          
              - If the number of elements in the physical response weighting array
              does not match the number of training response coordinates.
          
              - If the number of elements in the transformed response weighting array
              does not match the number of transformed response coordinates.
          
              - If neither a physical or transformed response weighting is supplied.



   .. py:method:: apply_reference_weighting(physical_reference_weighting=None, transformed_reference_weighting=None)

      Applys a weighting to the rows or columns of the reference
      transformation array.

      :param physical_reference_weighting: The weighting to be applied to the physical reference coordinates
                                           (i.e., the columns) of the reference transformation array. The
                                           weightings should be ordered the same as the "reference_coordinate"
                                           of the SPR object.
      :type physical_reference_weighting: ndarray, optional
      :param transformed_reference_weighting: The weighting to be applied to the transformed reference
                                              coordinates (i.e., the rows) of the reference transformation array.
                                              The weightings should be ordered the same as the "transformed_reference_coordinate"
                                              of the SPR object.
      :type transformed_reference_weighting: ndarray, optional

      :returns: **self** -- The SourcePathReceiver object with the reference transformation that
                has been updated
      :rtype: SourcePathReceiver

      :raises ValueError: If an of the following occurs:
          
              - If the number of elements in the physical reference weighting array
              does not match the number of reference coordinates.
          
              - If the number of elements in the transformed reference weighting array
              does not match the number of transformed reference coordinates.
          
              - If neither a physical or transformed reference weighting is supplied.



   .. py:method:: set_response_transformation_by_normalization(method='std', normalize_transformed_coordinate=True, reset_transformation=False)

      Sets the response transformation matrix such that it will "normalize"
      the responses based on a statistical quantity from the rows of the
      FRF matrix.

      :param method: A string defining the statistical quantity that will be used
                     for the response normalization. Currently, only the standard
                     deviation is supported.
      :type method: str, optional
      :param normalized_transformed_coordinate: Whether the normalization should be applied to the transformed
                                                or physical coordinate (i.e., the rows or columns of the transfomation
                                                array) of the SPR object. The default is True.
      :type normalized_transformed_coordinate: bool, optional
      :param reset_transformation: Whether the normalization should replace the pre-existing transformation
                                   array in the SPR object or if the normalization should be applied
                                   to the transformation array (similar to a frequency dependent response
                                   weighting).
      :type reset_transformation: bool, optional

      :returns: **self** -- The SourcePathReceiver object with the response transformation that
                has been updated
      :rtype: SourcePathReceiver



   .. py:method:: set_reference_transformation_by_normalization(method='std', normalize_transformed_coordinate=True, reset_transformation=False)

      Sets the reference transformation matrix such that it will "normalize"
      the references based on a statistical quantity from the columns of the
      FRF matrix.

      :param method: A string defining the statistical quantity that will be used
                     for the reference normalization. Currently, only the standard
                     deviation is supported.
      :type method: str, optional
      :param normalized_transformed_coordinate: Whether the normalization should be applied to the transformed
                                                or physical coordinate (i.e., the rows or columns of the transfomation
                                                array) of the SPR object. The default is True.
      :type normalized_transformed_coordinate: bool, optional
      :param reset_transformation: Whether the normalization should replace the pre-existing transformation
                                   array in the SPR object or if the normalization should be applied
                                   to the transformation array (similar to a frequency dependent reference
                                   weighting).
      :type reset_transformation: bool, optional

      :returns: **self** -- The SourcePathReceiver object with the reference transformation that
                has been updated
      :rtype: SourcePathReceiver



   .. py:method:: reset_reference_transformation()

      Resets the reference transformation matrix to default (identity).



   .. py:method:: reset_response_transformation()

      Resets the response transformation matrix to default (identity).



   .. py:method:: copy()

      Returns a deepcopy of the SPR object.



   .. py:method:: extract_elements_by_abscissa(min_abscissa=None, max_abscissa=None, in_place=True)

      Extracts elements from all the components of the SPR object with abscissa
      values within the specified range.

      :param min_abscissa: Minimum abscissa value to keep.
      :type min_abscissa: float
      :param max_abscissa: Maximum abscissa value to keep.
      :type max_abscissa: float
      :param in_place: Whether or not to modify the SPR object in place or create a new SPR object.
                       The default is True (i.e., modify the SPR object in place).
      :type in_place: bool

      :rtype: SPR object that has been trimmed according to the abscissa limits

      .. rubric:: Notes

      This method currently only works on linear or power SPR objects since transient SPR
      objects have two abscissas and there is ambiguity on which one would be trimmed.



.. py:class:: LinearSourcePathReceiver(frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, training_frfs=None, response_transformation=None, reference_transformation=None, empty=False)

   Bases: :py:obj:`SourcePathReceiver`


   A subclass to represent a source-path-receiver (SPR) model with linear spectra
   (i.e., ffts) for the responses or forces.

   .. rubric:: Notes

   The "linear" term in the class name stands for the linear units in the response and
   force spectra.


   .. py:property:: transformed_force

      The force with the transformation applied.


   .. py:property:: transformed_training_response

      The training response with the transformation applied (i.e., what is used in the inverse problem).


   .. py:property:: transformed_reconstructed_response

      The reconstructed response (at the training coordinates) with the transformation applied (i.e., what
      was used in the inverse problem).


   .. py:method:: predicted_response_specific_dofs(response_dofs)

      Computes the predicted response for specific response DOFs in the SPR object.

      :param response_dofs: The response DOFs to compute the response for. It should be shaped as a 1d array.
      :type response_dofs: CoordinateArray

      :returns: **predicted_response** -- The predicted response at the desired response DOFs (in the order they were
                supplied).
      :rtype: SpectrumArray

      :raises AttributeError: If there are not forces in the SPR object.
      :raises ValueError: If any of the following occurs:
          
              - If any of the selected response DOFs are not in the response_coordinate of
              the SPR object.
          
              - If the response_dofs is not a 1d array.



   .. py:method:: global_asd_error(channel_set='training')

      Computes the global ASD error in dB of the reconstructed response,
      per the procedure in MIL-STD 810H.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional

      :returns: Returns a spectrum array of the global ASD error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: SpectrumArray

      .. rubric:: Notes

      Computes the ASD from the spectrum response by squaring the absolute value
      of the spectrum and dividing by the SPR object abscissa spacing.

      .. rubric:: References

      .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.



   .. py:method:: average_asd_error(channel_set='training')

      Computes the DOF averaged ASD error in dB of the reconstructed response.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional

      :returns: Returns a spectrum array of the average ASD error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: PowerSpectralDensityArray

      .. rubric:: Notes

      Computes the ASD from the spectrum response by squaring the absolute value
      of the spectrum and dividing by the SPR object abscissa spacing.



   .. py:method:: rms_asd_error(channel_set='training')

      Computes the root mean square (RMS) of the DOF-DOF ASD dB error of the
      reconstructed response.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional

      :returns: Returns a spectrum array of the RMS ASD error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: PowerSpectralDensityArray

      .. rubric:: Notes

      Computes the ASD from the spectrum response by squaring the absolute value
      of the spectrum and dividing by the SPR object abscissa spacing.



   .. py:method:: error_summary(channel_set='training', figure_kwargs={}, linewidth=1, plot_kwargs={})

      Plots the error summary using the method of the same name from the
      PowerSpectralDensityArray class.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional
      :param figure_kwargs: Arguments to use when creating the figure. The default is {}.
      :type figure_kwargs: dict, optional
      :param linewidth: Widths of the lines on the plot. The default is 1.
      :type linewidth: float, optional
      :param plot_kwargs: Arguments to use when plotting the lines. The default is {}.
      :type plot_kwargs: dict, optional

      :returns: A tuple of dictionaries of error metrics
      :rtype: Error Metrics

      .. rubric:: Notes

      This is a simple wrapper around the "error_summary" method, where the
      PowerSpectralDensityArray is computed from the spectrum response by
      squaring the absolute value of the spectrum and dividing by the SPR
      object abscissa spacing.



   .. py:method:: manual_inverse(method='standard', regularization_weighting_matrix=None, regularization_parameter=None, cond_num_threshold=None, num_retained_values=None, use_transformation=True, response=None, frf=None)

      Perform the inverse source estimation problem with manual settings.

      :param method: The method to be used for the FRF matrix inversions. The available
                     methods are:

                         - standard
                             Basic pseudo-inverse via numpy.linalg.pinv with the default
                             rcond parameter, this is the default method.

                         - threshold
                             Pseudo-inverse via numpy.linalg.pinv with a specified condition
                             number threshold.

                         - tikhonov
                             Pseudo-inverse using the Tikhonov regularization method.

                         - truncation
                             Pseudo-inverse where a fixed number of singular values are
                             retained for the inverse.
      :type method: str, optional
      :param regularization_weighting_matrix: Matrix used to weight input degrees of freedom via Tikhonov regularization.
                                              This matrix can also be a 3D matrix such that the the weights are different
                                              for each frequency line. The matrix should be sized
                                              [number of lines, number of references, number of references], where the number
                                              of lines either be one (the same weights at all frequencies) or the length
                                              of the abscissa (for the case where a 3D matrix is supplied).
      :type regularization_weighting_matrix: sdpy.Matrix or np.ndarray, optional
      :param regularization_parameter: Scaling parameter used on the regularization weighting matrix when the tikhonov
                                       method is chosen. A vector of regularization parameters can be provided so the
                                       regularization is different at each frequency line. The vector must match the
                                       length of the abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
      :type regularization_parameter: float or np.ndarray, optional
      :param cond_num_threshold: Condition number used for SVD truncation when the threshold method is chosen.
                                 A vector of condition numbers can be provided so it varies as a function of
                                 frequency. The vector must match the length of the abscissa in this case.
      :type cond_num_threshold: float or np.ndarray, optional
      :param num_retained_values: Number of singular values to retain in the pseudo-inverse when the truncation
                                  method is chosen. A vector of can be provided so the number of retained values
                                  can change as a function of frequency. The vector must match the length of the
                                  abscissa in this case.
      :type num_retained_values: float or np.ndarray, optional
      :param use_transformation: Whether or not the response and reference transformation from the class definition
                                 should be used (which is handled in the "linear_inverse_processing" decorator
                                 function). The default is true.
      :type use_transformation: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator function
                  and object definition. This argument should not be supplied by the user.
      :type frf: ndarray

      :returns: **force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      The "linear_inverse_processing" decorator function pre and post processes the training
      response and FRF data from the SourcePathReceiver object to use the response and
      reference transformation matrices. This method only estimates the forces, using the
      supplied FRF inverse parameters.



   .. py:method:: auto_tikhonov_by_l_curve(low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, l_curve_type='standard', optimality_condition='curvature', use_transformation=True, response=None, frf=None)

      Performs the inverse source estimation problem with Tikhonov regularization,
      where the regularization parameter is automatically selected with L-curve
      methods.

      :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                       array with a length that matches the number of frequency lines in
                                       the SourcePathReceiver object. The default is the smallest singular
                                       value of the training frf array.
      :type low_regularization_limit: ndarray
      :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                        array with a length that matches the number of frequency lines in
                                        the SourcePathReceiver object. The default is the largest singular
                                        value of the training frf array.
      :type high_regularization_limit: ndarray
      :param number_regularization_values: The number of regularization parameters to search over, where the
                                           potential parameters are geometrically spaced between the low and high
                                           regularization limits.
      :type number_regularization_values: int
      :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                           parameter. The available types are:

                               - forces
                                   This L-curve is constructed with the "size" of the forces on
                                   the Y-axis and the regularization parameter on the X-axis.

                               - standard (default)
                                   This L-curve is constructed with the residual squared error on
                                   the X-axis and the "size" of the forces on the Y-axis.
      :type l_curve_type: str
      :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                   The options are:

                                       - curvature (default)
                                           This method searches for the regularization parameter that
                                           results in maximum curvature of the L-curve. It is also referred
                                           to as the L-curve criterion.

                                       - distance
                                           This method searches for the regularization parameter that
                                           minimizes the distance between the L-curve and a "virtual origin".
                                           A virtual origin is used, because the L-curve is scaled and offset
                                           to always range from zero to one, in this case.
      :type optimality_condition: str
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "linear_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      All the settings, including the selected regularization parameters, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
      .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.
      .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
          Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.



   .. py:method:: auto_tikhonov_by_cv_rse(low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, cross_validation_type='loocv', number_folds=None, use_transformation=True, response=None, frf=None)

      Performs the inverse source estimation problem with Tikhonov regularization,
      where the regularization parameter is automatically selected with cross
      validation, where the residual squared error is use as the metric to evaluate
      the quality of fit.

      :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                       array with a length that matches the number of frequency lines in
                                       the SourcePathReceiver object. The default is the smallest singular
                                       value of the training frf array.
      :type low_regularization_limit: ndarray, optional
      :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                        array with a length that matches the number of frequency lines in
                                        the SourcePathReceiver object. The default is the largest singular
                                        value of the training frf array.
      :type high_regularization_limit: ndarray, optional
      :param number_regularization_values: The number of regularization parameters to search over, where the
                                           potential parameters are geometrically spaced between the low and high
                                           regularization limits.
      :type number_regularization_values: int, optional
      :param cross_validation_type:
                                    The cross validation method to use. The available options are:

                                        - loocv (default)
                                            Leave one out cross validation.

                                        - k-fold
                                            K fold cross validation.
      :type cross_validation_type: str, optional
      :param number_folds: The number of folds to use in the k fold cross validation. The number of
                           response DOFs must be evenly divisible by the number of folds.
      :type number_folds: int
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "linear_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool, optional
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      All the settings, including the selected regularization parameters, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] D. M. Allen, "The Relationship between Variable Selection and Data Agumentation
             and a Method for Prediction," Technometrics, vol. 16, no. 1, pp. 125-127, 1974,
             doi: 10.2307/1267500.
      .. [2] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning:
             Data Mining, Inference, and Prediction, 2nd Edition ed. New York: Springer New York,
             2017.



   .. py:method:: auto_truncation_by_l_curve(l_curve_type='standard', optimality_condition='distance', curvature_method=None, use_transformation=True, response=None, frf=None)

      Performs the inverse source estimation problem with the truncated singular
      value decomposition (TSVD). The number of singular values to retain in the
      inverse is automatically selected with L-curve methods

      :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                           parameter. The available types are:

                               - forces - This L-curve is constructed with the "size" of the
                               forces on the Y-axis and the regularization parameter on the X-axis.

                               - standard (default) - This L-curve is constructed with the residual
                               squared error on the X-axis and the "size" of the forces on the Y-axis.
      :type l_curve_type: str
      :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                   The options are:

                                       - curvature
                                           This method searches for the regularization parameter that results
                                           in maximum curvature of the L-curve. It is also referred to as the
                                           L-curve criterion.

                                       - distance (default)
                                       This method searches for the regularization parameter that minimizes
                                       the distance between the L-curve and a "virtual origin". A virtual
                                       origin is used, because the L-curve is scaled and offset to always range
                                       from zero to one, in this case.
      :type optimality_condition: str
      :param curvature_method: The method that is used to compute the curvature of the L-curve, in the
                               case that the curvature is used to find the optimal regularization
                               parameter. The default is None and the options are:

                                   - numerical
                                       This method computes the curvature of the L-curve via numerical
                                       derivatives.

                                   - cubic_spline
                                       This method fits a cubic spline to the L-curve the computes the
                                       curvature from the cubic spline (this might perform better if the
                                       L-curve isn't "smooth").
      :type curvature_method: std
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "linear_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      L-curve for the TSVD could be non-smooth and determining the number of singular
      values to retain via curvature methods could lead to erratic results.

      All the setting, including the number of singular values to retain, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
      .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.
      .. [3] L. Reichel and H. Sadok, "A new L-curve for ill-posed problems," Journal of
          Computational and Applied Mathematics, vol. 219, no. 2, pp. 493-508,
          2008/10/01/ 2008, doi: https://doi.org/10.1016/j.cam.2007.01.025.



   .. py:method:: elastic_net_by_information_criterion(alpha_parameter, number_of_lambdas=100, information_criterion='AICC', use_transformation=True, response=None, frf=None, **kwargs)

      Perform the inverse source estimation problem with the elastic net and
      perform the model selection (to determine the optimal regularization
      parameter) with an information criterion.

      :param alpha_parameter: Alpha parameter for the elastic net. This controls the balance between the
                              L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
                              than 0 and less than 1.
      :type alpha_parameter: float
      :param number_of_lambdas: This parameter is supplied if the lambda_values are being determined by
                                the code. The default is 100.
      :type number_of_lambdas: int
      :param information_criterion:
                                    The desired information criterion, the available options are:

                                        - 'BIC'
                                            The Bayesian information criterion

                                        - 'AIC'
                                            The Akaike information criterion

                                        - 'AICC' (default)
                                            The corrected Akaike information criterion
      :type information_criterion: str
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "linear_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: References

      .. [1] T. Hastie, R. Tibshirani, M. Wainright, Statistical Learning with Sparsity:
          The Lasso with Generalizations. Boca Raton, Fl: CRC Press, 2015.
      .. [2] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
          Linear Models via Coordinate Descent, Journal of Statistical Software,
          Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01.



.. py:class:: PowerSourcePathReceiver(frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, training_frfs=None, buzz_cpsd=None, response_transformation=None, reference_transformation=None, empty=False)

   Bases: :py:obj:`SourcePathReceiver`


   A subclass to represent a source-path-receiver (SPR) model with power spectra
   for the responses and forces.

   .. rubric:: Notes

   The "power" term in the class name stands for the power units (i.e., units squared) in the
   response and force spectra.


   .. py:property:: transformed_force

      The force with the transformation applied.


   .. py:property:: transformed_training_response

      The training response with the transformation applied (i.e., what is used in the inverse problem).
      The buzz method is only applied if the training response is a set of PSDs rather than CPSDs.


   .. py:property:: transformed_reconstructed_response

      The reconstructed response (at the training coordinates) with the transformation applied (i.e., what
      was used in the inverse problem). This always outputs a response CPSD matrix for comparisons
      against the "transformed_training_response".


   .. py:method:: predicted_response_specific_dofs(response_dofs)

      Computes the predicted response for specific response DOFs in the SPR object.

      :param response_dofs: The response DOFs to compute the response for. It should be shaped as a 1d array.
      :type response_dofs: CoordinateArray

      :returns: **predicted_response** -- The predicted response as a square CPSD matrix with the desired response DOFs (in
                the order they were supplied).
      :rtype: PowerSpectralDensityArray

      :raises AttributeError: If there are not forces in the SPR object.
      :raises ValueError: If any of the following occurs:
          
              - If any of the selected response DOFs are not in the response_coordinate of
              the SPR object.
          
              - If the response_dofs is not a 1d array.



   .. py:method:: make_buzz_cpsd_from_frf()

      Generates the buzz CPSD array from the training FRFs.



   .. py:method:: global_asd_error(channel_set='training')

      Computes the global ASD error in dB of the reconstructed response,
      per the procedure in MIL-STD 810H.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional

      :returns: Returns a spectrum array of the global ASD error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: SpectrumArray

      .. rubric:: References

      .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.



   .. py:method:: average_asd_error(channel_set='training')

      Computes the DOF averaged ASD error in dB of the reconstructed response.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional

      :returns: Returns a spectrum array of the average ASD error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: PowerSpectralDensityArray



   .. py:method:: rms_asd_error(channel_set='training')

      Computes the root mean square (RMS) of the DOF-DOF ASD dB error of the
      reconstructed response.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional

      :returns: Returns a spectrum array of the RMS ASD error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: PowerSpectralDensityArray



   .. py:method:: error_summary(channel_set='training', figure_kwargs={}, linewidth=1, plot_kwargs={})

      Plots the error summary using the method of the same name from the
      PowerSpectralDensityArray class.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs.

                              - validation
                                  This compares the responses at the validation DOFs.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional
      :param figure_kwargs: Arguments to use when creating the figure. The default is {}.
      :type figure_kwargs: dict, optional
      :param linewidth: Widths of the lines on the plot. The default is 1.
      :type linewidth: float, optional
      :param plot_kwargs: Arguments to use when plotting the lines. The default is {}.
      :type plot_kwargs: dict, optional

      :returns: A tuple of dictionaries of error metrics
      :rtype: Error Metrics

      .. rubric:: Notes

      This is a simple wrapper around the "error_summary" method.



   .. py:method:: manual_inverse(method='standard', regularization_weighting_matrix=None, regularization_parameter=None, cond_num_threshold=None, num_retained_values=None, use_transformation=True, use_buzz=False, update_header=True, response=None, frf=None)

      Perform the inverse source estimation problem with manual settings.

      :param method: The method to be used for the FRF matrix inversions. The available
                     methods are:

                         - standard
                             Basic pseudo-inverse via numpy.linalg.pinv with the default rcond
                             parameter, this is the default method.

                         - threshold
                             Pseudo-inverse via numpy.linalg.pinv with a specified condition
                             number threshold.

                         - tikhonov
                             Pseudo-inverse using the Tikhonov regularization method.

                         - truncation
                             Pseudo-inverse where a fixed number of singular values are
                             retained for the inverse.
      :type method: str, optional
      :param regularization_weighting_matrix: Matrix used to weight input degrees of freedom via Tikhonov regularization.
                                              This matrix can also be a 3D matrix such that the the weights are different
                                              for each frequency line. The matrix should be sized
                                              [number of lines, number of references, number of references], where the number
                                              of lines either be one (the same weights at all frequencies) or the length
                                              of the abscissa (for the case where a 3D matrix is supplied).
      :type regularization_weighting_matrix: sdpy.Matrix or np.ndarray, optional
      :param regularization_parameter: Scaling parameter used on the regularization weighting matrix when the tikhonov
                                       method is chosen. A vector of regularization parameters can be provided so the
                                       regularization is different at each frequency line. The vector must match the
                                       length of the abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
      :type regularization_parameter: float or np.ndarray, optional
      :param cond_num_threshold: Condition number used for SVD truncation when the threshold method is chosen.
                                 A vector of condition numbers can be provided so it varies as a function of
                                 frequency. The vector must match the length of the abscissa in this case.
      :type cond_num_threshold: float or np.ndarray, optional
      :param num_retained_values: Number of singular values to retain in the pseudo-inverse when the truncation
                                  method is chosen. A vector of can be provided so the number of retained values
                                  can change as a function of frequency. The vector must match the length of the
                                  abscissa in this case.
      :type num_retained_values: float or np.ndarray, optional
      :param use_transformation: Whether or not the response and reference transformation from the class definition
                                 should be used (which is handled in the "power_inverse_processing" decorator
                                 function). The default is true.
      :type use_transformation: bool
      :param use_buzz: Whether or not to use the buzz method with the buzz CPSDs from the class
                       definition (this is handled in the "power_inverse_processing" decorator
                       function). The default is false.
      :type use_buzz: bool
      :param update_header: Whether or not to update the "inverse_settings" dictionary with all the settings
                            from the inverse problem. This exists primarily for compatibility with the
                            Rattlesnake control law, where updating the header information is undesirable
                            for how the settings are parsed.
      :type update_header: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator function
                  and object definition. This argument should not be supplied by the user.
      :type frf: ndarray

      :returns: **force** -- An ndarray of the estimated sources. The "power_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      The "power_inverse_processing" decorator function pre and post processes the training
      response and FRF data from the SourcePathReceiver object to use the response and
      reference transformation matrices. This method only estimates the forces, using the
      supplied FRF inverse parameters.

      .. rubric:: References

      .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation,
          Aerospace Department, University of Bristol, 2014



   .. py:method:: auto_tikhonov_by_l_curve(low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, l_curve_type='standard', optimality_condition='curvature', use_transformation=True, use_buzz=False, update_header=True, response=None, frf=None)

      Performs the inverse source estimation problem with Tikhonov regularization,
      where the regularization parameter is automatically selected with L-curve
      methods.

      :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                       array with a length that matches the number of frequency lines in
                                       the SourcePathReceiver object. The default is the smallest singular
                                       value of the training frf array.
      :type low_regularization_limit: ndarray
      :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                        array with a length that matches the number of frequency lines in
                                        the SourcePathReceiver object. The default is the largest singular
                                        value of the training frf array.
      :type high_regularization_limit: ndarray
      :param number_regularization_values: The number of regularization parameters to search over, where the
                                           potential parameters are geometrically spaced between the low and high
                                           regularization limits.
      :type number_regularization_values: int
      :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                           parameter. The available types are:

                               - forces
                                   This L-curve is constructed with the "size" of the forces on
                                   the Y-axis and the regularization parameter on the X-axis.

                               - standard (default)
                                   This L-curve is constructed with the residual squared error on
                                   the X-axis and the "size" of the forces on the Y-axis.
      :type l_curve_type: str
      :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                   The options are:

                                       - curvature (default)
                                           This method searches for the regularization parameter that results
                                           in maximum curvature of the L-curve. It is also referred to as the
                                           L-curve criterion.

                                       - distance
                                           This method searches for the regularization parameter that minimizes
                                           the distance between the L-curve and a "virtual origin". A virtual
                                           origin is used, because the L-curve is scaled and offset to always
                                           range from zero to one, in this case.
      :type optimality_condition: str
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "power_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool
      :param use_buzz: Whether or not to use the buzz method with the buzz CPSDs from the class
                       definition (this is handled in the "power_inverse_processing" decorator
                       function). The default is false.
      :type use_buzz: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      All the settings, including the selected regularization parameters, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation,
          Aerospace Department, University of Bristol, 2014
      .. [2] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
      .. [3] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.
      .. [4] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
          Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.



   .. py:method:: auto_tikhonov_by_cv_rse(low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, cross_validation_type='loocv', number_folds=None, use_transformation=True, use_buzz=False, update_header=True, response=None, frf=None)

      Performs the inverse source estimation problem with Tikhonov regularization,
      where the regularization parameter is automatically selected with cross
      validation, where the residual squared error is use as the metric to evaluate
      the quality of fit.

      :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                       array with a length that matches the number of frequency lines in
                                       the SourcePathReceiver object. The default is the smallest singular
                                       value of the training frf array.
      :type low_regularization_limit: ndarray, optional
      :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                        array with a length that matches the number of frequency lines in
                                        the SourcePathReceiver object. The default is the largest singular
                                        value of the training frf array.
      :type high_regularization_limit: ndarray, optional
      :param number_regularization_values: The number of regularization parameters to search over, where the
                                           potential parameters are geometrically spaced between the low and high
                                           regularization limits.
      :type number_regularization_values: int, optional
      :param cross_validation_type:
                                    The cross validation method to use. The available options are:

                                        - loocv (default)
                                            Leave one out cross validation.

                                        - k-fold
                                            K fold cross validation.
      :type cross_validation_type: str, optional
      :param number_folds: The number of folds to use in the k fold cross validation. The number of
                           response DOFs must be evenly divisible by the number of folds.
      :type number_folds: int
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "linear_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool, optional
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      All the settings, including the selected regularization parameters, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] D. M. Allen, "The Relationship between Variable Selection and Data Agumentation
             and a Method for Prediction," Technometrics, vol. 16, no. 1, pp. 125-127, 1974,
             doi: 10.2307/1267500.
      .. [2] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning:
             Data Mining, Inference, and Prediction, 2nd Edition ed. New York: Springer New York,
             2017.



   .. py:method:: auto_truncation_by_l_curve(l_curve_type='standard', optimality_condition='distance', curvature_method=None, use_transformation=True, use_buzz=False, update_header=True, response=None, frf=None)

      Performs the inverse source estimation problem with the truncated singular
      value decomposition (TSVD). The number of singular values to retain in the
      inverse is automatically selected with L-curve methods

      :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                           parameter. The available types are:

                               - forces
                                   This L-curve is constructed with the "size" of the forces on
                                   the Y-axis and the regularization parameter on the X-axis.

                               - standard (default)
                                   This L-curve is constructed with the residual squared error on
                                   the X-axis and the "size" of the forces on the Y-axis.
      :type l_curve_type: str
      :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                   The options are:

                                       - curvature
                                           This method searches for the regularization parameter that
                                           results in maximum curvature of the L-curve. It is also referred
                                           to as the L-curve criterion.

                                       - distance (default)
                                           This method searches for the regularization parameter that minimizes
                                           the distance between the L-curve and a "virtual origin". A virtual
                                           origin is used, because the L-curve is scaled and offset to always
                                           range from zero to one, in this case.
      :type optimality_condition: str
      :param curvature_method: The method that is used to compute the curvature of the L-curve, in the
                               case that the curvature is used to find the optimal regularization
                               parameter. The default is None and the options are:

                                   - numerical
                                       This method computes the curvature of the L-curve via numerical
                                       derivatives.

                                   - cubic_spline
                                       This method fits a cubic spline to the L-curve the computes the
                                       curvature from the cubic spline (this might perform better if the
                                       L-curve isn't "smooth").
      :type curvature_method: std
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "power_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool
      :param use_buzz: Whether or not to use the buzz method with the buzz CPSDs from the class
                       definition (this is handled in the "power_inverse_processing" decorator
                       function). The default is false.
      :type use_buzz: bool
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      L-curve for the TSVD could be non-smooth and determining the number of singular
      values to retain via curvature methods could lead to erratic results.

      All the setting, including the number of singular values to retain, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation,
          Aerospace Department, University of Bristol, 2014
      .. [2] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
      .. [3] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.
      .. [4] L. Reichel and H. Sadok, "A new L-curve for ill-posed problems," Journal of
          Computational and Applied Mathematics, vol. 219, no. 2, pp. 493-508,
          2008/10/01/ 2008, doi: https://doi.org/10.1016/j.cam.2007.01.025.



   .. py:method:: match_trace_update(use_transformation=True, in_place=True)

      Applies a "match trace" update to the to the forces in the SPR object to
      eliminate bias error.

      :param use_transformation: Whether or not the transformation was used in the input estimation,
                                 this should match what was used in the inverse problem for the correct
                                 behavior. The default is true.
      :type use_transformation: bool, optional
      :param in_place: Whether to apply the limit to the original SPR object or not. When
                       true, The limit will be applied to the original SPR object. When
                       false, a copy will be made of the original and the limit will be
                       applied to that copy. The default is true.
      :type in_place: bool, optional

      :rtype: SPR object with the match trace update applied to the force attribute.

      .. rubric:: References

      .. [1] D. Rohe, R. Schultz, and N. Hunter, "Rattlesnake Users Manual,"
              Sandia National Laboratories, 2021.



   .. py:method:: reduce_drives_update(use_transformation=True, db_error_ratio=1.0, reduce_max_drive=False, use_warm_start=True, in_place=True)

      Minimizes the sum of the force PSDs while holding absolute dB error between
      predicted and actual responses constant, where this dB error is defined by
      the reconstructed response from the pre-existing force in the SPR object.

      :param use_transformation: Whether or not the transformation was used in the input estimation,
                                 this should match what was used in the inverse problem for the correct
                                 behavior. The default is true.
      :type use_transformation: bool, optional
      :param db_error_ratio: Recommend >=1. Constrains the predicted dB error after drive reduction
                             to this value times the prior dB error. Default is 1.0.
      :type db_error_ratio: float, optional
      :param reduce_max_drive: If true, reduces the maximum drive in each frequency bin. If false,
                               reduces the drive trace. Default is False.
      :type reduce_max_drive: bool, optional
      :param use_warm_start: Whether to use the initial forces as a warm start for the optimizer.
                             Default is True.
      :type use_warm_start: bool, optional
      :param in_place: Whether to apply the limit to the original SPR object or not. When
                       true, The limit will be applied to the original SPR object. When
                       false, a copy will be made of the original and the limit will be
                       applied to that copy. The default is true.
      :type in_place: bool, optional

      :rtype: SPR object with the reduce drives update applied to the force attribute.

      :raises AttributeError: If there are not forces in the SPR object.
      :raises warning: If the optimization fails to converge or meet the error constraints.

      .. rubric:: Notes

      This method leverages an optimization problem and warnings may be supplied
      if the optimization fails or has not converged. A failed optimization may
      have resulted in under/over predicted response or unexpected force amplitudes.



   .. py:method:: apply_response_limit(response_limit, limit_db_level=float(0), interpolation_type='loglog', in_place=True)

      Scales the force CPSD to apply a limit to the predicted PSD responses in
      the SPR object.

      :param response_limit: The ResponseLimit object that defines the limits.
      :type response_limit: ResponseLimit
      :param limit_level: The dB level for the limit. The levels in the limit_dict will be
                          modified by the dB level. The default is 0 dB (no modification).
      :type limit_level: optional, float
      :param interpolation_type: The type of interpolation to use to convert the breakpoints to all
                                 the frequency lines in the SPR object. The options are loglog or
                                 linearlinear, depending on if the frequencies and levels should be
                                 plotted on log-log or linear-linear plot axes. The default is loglog.
      :type interpolation_type: str, optional
      :param in_place: Whether to apply the limit to the original SPR object or not. When
                       true, The limit will be applied to the original SPR object. When
                       false, a copy will be made of the original and the limit will be
                       applied to that copy. The default is true.
      :type in_place: bool, optional

      :rtype: SPR object with the response limits applied.

      :raises KeyError: If the limit_dict does not have the correct keys or if the information
          for the keys are not organized correctly.
      :raises ValueError: If the specified interpolation type is not available.

      .. rubric:: Notes

      The limit is applied iteratively to each limit DOF, one at a time, rather
      than determining a globally optimal solution with the limits applied.



.. py:class:: TransientSourcePathReceiver(frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, training_frfs=None, response_transformation=None, reference_transformation=None, empty=False)

   Bases: :py:obj:`SourcePathReceiver`


   A subclass to represent a source-path-receiver (SPR) model with time traces
   for the responses and forces.

   .. rubric:: Notes

   The "transient" term in the class name refers to the intended use of this SPR model
   (transient problems).


   .. py:method:: save(filename)

      Saves the TransientSourcePathReceiver object to a .npz file.

      :param filename: The file path and name for the .npz file
      :type filename: str

      .. rubric:: Notes

      The private properties of the class are saved as arguments in the .npz file, where
      the argument names match the private variable name. The save method is specially
      defined for the TransientSourcePathReceiver because it has a "time_abscissa" private
      property, which isn't in the other SourcePathReceiver objects.



   .. py:method:: load(filename)
      :classmethod:


      Loads the TransientSourcePathReceiver object from an .npz file.

      :param filename: The file path and name for the .npz file
      :type filename: str

      .. rubric:: Notes

      The private properties of the class should have been saved as arguments in the .npz
      file, where the argument names match the private variable name. The load method is
      specially defined for the TransientSourcePathReceiver because it has a "time_abscissa"
      private property, which isn't in the other SourcePathReceiver objects.



   .. py:property:: transformed_force

      The force with the transformation applied.


   .. py:property:: transformed_training_response

      The training response with the transformation applied (i.e., what is used in the inverse problem).


   .. py:property:: transformed_reconstructed_response

      The reconstructed response (at the training coordinates) with the transformation applied (i.e., what
      was used in the inverse problem).


   .. py:method:: predicted_response_specific_dofs(response_dofs)

      Computes the predicted response for specific response DOFs in the SPR object.

      :param response_dofs: The response DOFs to compute the response for. It should be shaped as a 1d array.
      :type response_dofs: CoordinateArray

      :returns: **predicted_response** -- The predicted response with the desired response DOFs (in the order they were
                supplied).
      :rtype: TimeHistoryArray

      :raises AttributeError: If there are not forces in the SPR object.
      :raises ValueError: If any of the following occurs:
              - If any of the selected response DOFs are not in the response_coordinate of
              the SPR object.
          
              - If the response_dofs is not a 1d array.



   .. py:method:: extract_time_elements_by_abscissa(min_abscissa=None, max_abscissa=None, in_place=True)

      Extracts the time elements from all the components of the SPR object with
      abscissa values within the specified range.

      :param min_abscissa: Minimum abscissa value to keep.
      :type min_abscissa: float
      :param max_abscissa: Maximum abscissa value to keep.
      :type max_abscissa: float
      :param in_place: Whether or not to modify the SPR object in place or create a new SPR object.
                       The default is True (i.e., modify the SPR object in place).
      :type in_place: bool

      :rtype: SPR object that has been trimmed according to the abscissa limits



   .. py:method:: global_rms_error(channel_set='training', samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None)

      Computes the global RMS error in dB of the reconstructed response,
      per the procedure in MIL-STD 810H.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training DOFs
                                  in the SPR object.

                              - validation
                                  This compares the responses for the validation response DOFs
                                  in the SPR object.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional
      :param samples_per_frame: Number of samples in each measurement frame to compute the RMS
                                for. Either this argument or `frame_length` must be specified.
                                If both or neither are specified, a `ValueError` is raised. This
                                argument matches the behavior of the "split_into_frames" method.
      :type samples_per_frame: int, optional
      :param frame_length: Length of each measurement frame to compute the RMS for, in the
                           same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
                           Either this argument or `samples_per_frame` must be specified. If
                           both or neither are specified, a `ValueError` is raised. This
                           argument matches the behavior of the "split_into_frames" method.
      :type frame_length: float, optional
      :param overlap: Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
                      overlap a quarter of the frame) for the RMS calculation. Either
                      this argument or `overlap_samples` must be specified. If both are
                      specified, a `ValueError` is raised.  If neither are specified, no
                      overlap is used. This argument matches the behavior of the
                      "split_into_frames" method.
      :type overlap: float, optional
      :param overlap_samples: Number of samples in the measurement frame to overlap for the RMS
                              calculation. Either this argument or `overlap_samples` must be
                              specified.  If both are specified, a `ValueError` is raised.  If
                              neither are specified, no overlap is used. This argument matches
                              the behavior of the "split_into_frames" method.
      :type overlap_samples: int, optional

      :returns: Returns a time history array of the global RMS error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: TimeHistoryArray

      .. rubric:: References

      .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.



   .. py:method:: average_rms_error(channel_set='training', samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None)

      Computes the average RMS error in dB of the reconstructed response.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training
                                  DOFs in the SPR object.

                              - validation
                                  This compares the responses for the validation response
                                  DOFs in the SPR object.

                              - target
                                  This compares the responses for all the target response DOFs
                                  in the SPR object.
      :type channel_set: str, optional
      :param samples_per_frame: Number of samples in each measurement frame to compute the RMS
                                for. Either this argument or `frame_length` must be specified.
                                If both or neither are specified, a `ValueError` is raised. This
                                argument matches the behavior of the "split_into_frames" method.
      :type samples_per_frame: int, optional
      :param frame_length: Length of each measurement frame to compute the RMS for, in the
                           same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
                           Either this argument or `samples_per_frame` must be specified. If
                           both or neither are specified, a `ValueError` is raised. This
                           argument matches the behavior of the "split_into_frames" method.
      :type frame_length: float, optional
      :param overlap: Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
                      overlap a quarter of the frame) for the RMS calculation. Either
                      this argument or `overlap_samples` must be specified. If both are
                      specified, a `ValueError` is raised.  If neither are specified, no
                      overlap is used. This argument matches the behavior of the
                      "split_into_frames" method.
      :type overlap: float, optional
      :param overlap_samples: Number of samples in the measurement frame to overlap for the RMS
                              calculation. Either this argument or `overlap_samples` must be
                              specified.  If both are specified, a `ValueError` is raised.  If
                              neither are specified, no overlap is used. This argument matches
                              the behavior of the "split_into_frames" method.
      :type overlap_samples: int, optional

      :returns: Returns a time history array of the average RMS error in dB. The response
                coordinate for this array is made up to have a value for the DataArray
                and does not correspond to anything.
      :rtype: TimeHistoryArray



   .. py:method:: time_varying_trac(channel_set='training', samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None)

      Computes the time varying TRAC comparison between the truth and
      reconstructed responses.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training
                                  DOFs in the SPR object.

                              - validation
                                  This compares the responses for the validation response
                                  DOFs in the SPR object.

                              - target
                                  This compares the responses for all the target response
                                  DOFs in the SPR object.
      :type channel_set: str, optional
      :param samples_per_frame: Number of samples in each measurement frame to compute the RMS
                                for. Either this argument or `frame_length` must be specified.
                                If both or neither are specified, a `ValueError` is raised. This
                                argument matches the behavior of the "split_into_frames" method.
      :type samples_per_frame: int, optional
      :param frame_length: Length of each measurement frame to compute the RMS for, in the
                           same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
                           Either this argument or `samples_per_frame` must be specified. If
                           both or neither are specified, a `ValueError` is raised. This
                           argument matches the behavior of the "split_into_frames" method.
      :type frame_length: float, optional
      :param overlap: Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
                      overlap a quarter of the frame) for the RMS calculation. Either
                      this argument or `overlap_samples` must be specified. If both are
                      specified, a `ValueError` is raised.  If neither are specified, no
                      overlap is used. This argument matches the behavior of the
                      "split_into_frames" method.
      :type overlap: float, optional
      :param overlap_samples: Number of samples in the measurement frame to overlap for the RMS
                              calculation. Either this argument or `overlap_samples` must be
                              specified.  If both are specified, a `ValueError` is raised.  If
                              neither are specified, no overlap is used. This argument matches
                              the behavior of the "split_into_frames" method.
      :type overlap_samples: int, optional

      :returns: Returns a time history array of the time varying TRAC for all the
                response degrees of freedom.
      :rtype: TimeHistoryArray



   .. py:method:: time_varying_level_error(channel_set='training', level_type='rms', samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None)

      Computes the computes the time varying error for a statistical level
      (rms or max) between the truth and reconstructed responses.

      :param channel_set: The channel set to make the response comparisons between.
                          The available options are:

                              - training (default)
                                  This compares the responses for the transformed training
                                  DOFs in the SPR object.

                              - validation
                                  This compares the responses for the validation response
                                  DOFs in the SPR object.

                              - target
                                  This compares the responses for all the target response
                                  DOFs in the SPR object.
      :type channel_set: str, optional
      :param level_type:
                         The type of level to be used in the comparison. The options are:

                             - rms (default)
                                 The rms level error for each frame of data in the responses.

                             - max
                                 The error in the maximum level that is seem for each frame
                                 of data in the responses.
      :type level_type: str, optional
      :param samples_per_frame: Number of samples in each measurement frame to compute the RMS
                                for. Either this argument or `frame_length` must be specified.
                                If both or neither are specified, a `ValueError` is raised. This
                                argument matches the behavior of the "split_into_frames" method.
      :type samples_per_frame: int, optional
      :param frame_length: Length of each measurement frame to compute the RMS for, in the
                           same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
                           Either this argument or `samples_per_frame` must be specified. If
                           both or neither are specified, a `ValueError` is raised. This
                           argument matches the behavior of the "split_into_frames" method.
      :type frame_length: float, optional
      :param overlap: Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
                      overlap a quarter of the frame) for the RMS calculation. Either
                      this argument or `overlap_samples` must be specified. If both are
                      specified, a `ValueError` is raised.  If neither are specified, no
                      overlap is used. This argument matches the behavior of the
                      "split_into_frames" method.
      :type overlap: float, optional
      :param overlap_samples: Number of samples in the measurement frame to overlap for the RMS
                              calculation. Either this argument or `overlap_samples` must be
                              specified.  If both are specified, a `ValueError` is raised.  If
                              neither are specified, no overlap is used. This argument matches
                              the behavior of the "split_into_frames" method.
      :type overlap_samples: int, optional

      :returns: Returns a time history array of the time varying TRAC for all the
                response degrees of freedom.
      :rtype: TimeHistoryArray



   .. py:method:: sosfiltfilt(sos, in_place=True, **sosfiltfilt_kwargs)

      Performs forward-backward digital filtering on the time domain components
      of the spr object with the SciPy sosfiltfilt function

      :param sos: An array second-order filter coefficients from on of the SciPy filter
                  design tools (the filter output should be 'sos').
      :type sos: array_like
      :param in_place: Whether to apply the filters to the original SPR object or not. When
                       true, The filters will be applied to the original SPR object. When
                       false, a copy will be made of the original and the filters will be
                       applied to that copy. The default is true.
      :type in_place: bool, optional
      :param sosfiltfilt_kwargs: Additional keyword arguments that will be passed to the sosfiltfilt
                                 function. The axis keyword will always be set to 0 to be compatible
                                 the the organization of the data.

      :rtype: SPR object with the time domain components filtered.



   .. py:method:: attenuate_force(limit: Union[float, numpy.ndarray], full_scale: float = 1.0, in_place: bool = True)

      Attenuate peaks in the force that exceed limits by scaling the region between
      zero crossings using the local maximum. This maintains a smooth waveform that
      does not exceed the specified limits.

      :param limit: limit value or array of limit values with shape (n_signals,)
      :type limit: float | np.ndarray
      :param full_scale: global scaling factor applied to limit value, intended to be used such
                         that output waveform peaks are slightly less than physical limit,
                         (ex. use full_scale=0.97 so that output signal will not exceed 97% of specified
                         limit), by default 1.0
      :type full_scale: float, optional
      :param in_place: Whether to apply the limit to the original SPR object or not. When
                       true, The limit will be applied to the original SPR object. When
                       false, a copy will be made of the original and the limit will be
                       applied to that copy. The default is true.
      :type in_place: bool, optional

      :rtype: SPR object with the limit applied to the force attribute.

      .. rubric:: Notes

      Each region between zero crossings is scaled according to:
          `full_scale * limit / local_maximum`



   .. py:method:: manual_inverse(inverse_method='standard', regularization_weighting_matrix=None, regularization_parameter=None, cond_num_threshold=None, num_retained_values=None, cola_frame_length=None, cola_window=('tukey', 0.5), cola_overlap_samples=None, frf_interpolation_type='sinc', transformation_interpolation_type='cubic', use_transformation=False, response_generator=None, frf=None, reconstruction_generator=None)

      Performs the inverse source estimation problem with manual settings.

      :param cola_frame_length: The frame length (in samples) if the COLA method is being used. The
                                default frame length is Fs/df from the transfer function.
      :type cola_frame_length: float, optional
      :param cola_window: The desired window for the COLA procedure, must exist in the scipy
                          window library. The default is a Tukey window with an alpha of 0.5.
      :type cola_window: str, optional
      :param cola_overlap_samples: The number of overlapping samples between measurement frames in the
                                   COLA procedure.  A default is defined for the default Tukey window,
                                   otherwise the user must supply it.
      :type cola_overlap_samples: int, optional
      :param inverse_method: The method to be used for the FRF matrix inversions. The available
                             methods are:

                                 - standard
                                     Basic pseudo-inverse via numpy.linalg.pinv with the default
                                     rcond parameter, this is the default method.

                                 - threshold
                                     Pseudo-inverse via numpy.linalg.pinv with a specified condition
                                     number threshold.

                                 - tikhonov
                                     Pseudo-inverse using the Tikhonov regularization method.

                                 - truncation
                                     Pseudo-inverse where a fixed number of singular values are
                                     retained for the inverse.
      :type inverse_method: str, optional
      :param regularization_weighting_matrix: Matrix used to weight input degrees of freedom via Tikhonov regularization.
      :type regularization_weighting_matrix: sdpy.Matrix, optional
      :param regularization_parameter: Scaling parameter used on the regularization weighting matrix when the tikhonov
                                       method is chosen. A vector of regularization parameters can be provided so the
                                       regularization is different at each frequency line. The vector must match the
                                       length of the FRF abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
      :type regularization_parameter: float or np.ndarray, optional
      :param cond_num_threshold: Condition number used for SVD truncation when the threshold method is chosen.
                                 A vector of condition numbers can be provided so it varies as a function of
                                 frequency. The vector must match the length of the FRF abscissa in this case
                                 (either be size [num_lines,] or [num_lines, 1]).
      :type cond_num_threshold: float or np.ndarray, optional
      :param num_retained_values: Number of singular values to retain in the pseudo-inverse when the truncation
                                  method is chosen. A vector of can be provided so the number of retained values
                                  can change as a function of frequency. The vector must match the length of the
                                  FRF abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
      :type num_retained_values: float or np.ndarray, optional
      :param frf_interpolation_type: The type of interpolation to use on the FRFs (to account for the zero padding).
                                     This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
                                     The default is 'cubic'.
      :type frf_interpolation_type: str, optional
      :param transformation_interpolation_type: The type of interpolation to use on the FRFs (to account for the zero padding).
                                                This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
                                                The default is 'cubic'.
      :type transformation_interpolation_type: str, optional
      :param use_transformation: Whether or not to use the transformations in the ISE problem. The default is
                                 False.
      :type use_transformation: bool, optional
      :param response_generator: The generator function to create the COLA segmented responses. This is created
                                 by the decorator function and should not be supplied by the user.
      :type response_generator: function
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator function
                  and object definition. This argument should not be supplied by the user.
      :type frf: ndarray
      :param reconstruction_generator: The generator function to recompile the COLA segmented forces. This is created
                                       by the decorator function and should not be supplied by the user.
      :type reconstruction_generator: function

      :returns: **reconstructed_force** -- An ndarray array of the estimated sources.
      :rtype: ndarray

      .. rubric:: Notes

      The "transient_inverse_processing" decorator function pre and post processes the
      training response and FRF data from the SourcePathReceiver object to segment the
      data for COLA processing, apply the response and reference transformations, and
      recompile the segmented forces into a single time trace. This method only estimates the
      forces, using the supplied FRF inverse parameters.

      .. rubric:: References

      .. [1] Wikipedia, "Overlap-add Method".
          https://en.wikipedia.org/wiki/Overlap-add_method



   .. py:method:: auto_tikhonov_by_l_curve(low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, l_curve_type='standard', optimality_condition='curvature', cola_frame_length=None, cola_window=('tukey', 0.5), cola_overlap_samples=None, frf_interpolation_type='sinc', transformation_interpolation_type='cubic', use_transformation=False, response_generator=None, frf=None, reconstruction_generator=None)

      Performs the inverse source estimation problem with Tikhonov regularization,
      where the regularization parameter is automatically selected with L-curve
      methods.

      :param cola_frame_length: The frame length (in samples) if the COLA method is being used. The
                                default frame length is Fs/df from the transfer function.
      :type cola_frame_length: float, optional
      :param cola_window: The desired window for the COLA procedure, must exist in the scipy
                          window library. The default is a Tukey window with an alpha of 0.5.
      :type cola_window: str, optional
      :param cola_overlap_samples: The number of overlapping samples between measurement frames in the
                                   COLA procedure.  A default is defined for the default Tukey window,
                                   otherwise the user must supply it.
      :type cola_overlap_samples: int, optional
      :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                       array with a length that matches the number of frequency lines in
                                       the SourcePathReceiver object. The default is the smallest singular
                                       value of the training frf array.
      :type low_regularization_limit: ndarray
      :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                        array with a length that matches the number of frequency lines in
                                        the SourcePathReceiver object. The default is the largest singular
                                        value of the training frf array.
      :type high_regularization_limit: ndarray
      :param number_regularization_values: The number of regularization parameters to search over, where the
                                           potential parameters are geometrically spaced between the low and high
                                           regularization limits.
      :type number_regularization_values: int
      :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                           parameter. The available types are:

                               - forces
                                   This L-curve is constructed with the "size" of the forces on the
                                   Y-axis and the regularization parameter on the X-axis.

                               - standard (default)
                                   This L-curve is constructed with the residual squared error on
                                   the X-axis and the "size" of the forces on the Y-axis.
      :type l_curve_type: str
      :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                   The options are:

                                       - curvature (default)
                                           This method searches for the regularization parameter that results
                                           in maximum curvature of the L-curve. It is also referred to as the
                                           L-curve criterion.

                                       - distance
                                           This method searches for the regularization parameter that minimizes
                                           the distance between the L-curve and a "virtual origin". A virtual
                                           origin is used, because the L-curve is scaled and offset to always
                                           range from zero to one, in this case.
      :type optimality_condition: str
      :param frf_interpolation_type: The type of interpolation to use on the FRFs (to account for the zero padding).
                                     This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
                                     The default is 'cubic'.
      :type frf_interpolation_type: str, optional
      :param transformation_interpolation_type: The type of interpolation to use on the FRFs (to account for the zero padding).
                                                This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
                                                The default is 'cubic'.
      :type transformation_interpolation_type: str, optional
      :param use_transformation: Whether or not to use the transformations in the ISE problem. The default is
                                 False.
      :type use_transformation: bool, optional
      :param response_generator: The generator function to create the COLA segmented responses. This is created
                                 by the decorator function and should not be supplied by the user.
      :type response_generator: function
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator function
                  and object definition. This argument should not be supplied by the user.
      :type frf: ndarray
      :param reconstruction_generator: The generator function to recompile the COLA segmented forces. This is created
                                       by the decorator function and should not be supplied by the user.
      :type reconstruction_generator: function

      :returns: **reconstructed_force** -- An ndarray array of the estimated sources.
      :rtype: ndarray

      .. rubric:: Notes

      The "transient_inverse_processing" decorator function pre and post processes the
      training response and FRF data from the SourcePathReceiver object to segment the
      data for COLA processing, apply the response and reference transformations, and
      recompile the segmented forces into a single time trace. This method only estimates the
      forces, using the supplied FRF inverse parameters.

      .. rubric:: References

      .. [1] Wikipedia, "Overlap-add Method".
          https://en.wikipedia.org/wiki/Overlap-add_method
      .. [2] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
      .. [3] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.
      .. [4] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
          Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.



   .. py:method:: auto_tikhonov_by_cv_rse(low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, cross_validation_type='loocv', number_folds=None, cola_frame_length=None, cola_window=('tukey', 0.5), cola_overlap_samples=None, frf_interpolation_type='sinc', transformation_interpolation_type='cubic', use_transformation=False, response_generator=None, frf=None, reconstruction_generator=None)

      Performs the inverse source estimation problem with Tikhonov regularization,
      where the regularization parameter is automatically selected with cross
      validation, where the residual squared error is use as the metric to evaluate
      the quality of fit.

      :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                       array with a length that matches the number of frequency lines in
                                       the SourcePathReceiver object. The default is the smallest singular
                                       value of the training frf array.
      :type low_regularization_limit: ndarray, optional
      :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                        array with a length that matches the number of frequency lines in
                                        the SourcePathReceiver object. The default is the largest singular
                                        value of the training frf array.
      :type high_regularization_limit: ndarray, optional
      :param number_regularization_values: The number of regularization parameters to search over, where the
                                           potential parameters are geometrically spaced between the low and high
                                           regularization limits.
      :type number_regularization_values: int, optional
      :param cross_validation_type:
                                    The cross validation method to use. The available options are:

                                        - loocv (default)
                                            Leave one out cross validation.

                                        - k-fold
                                            K fold cross validation.
      :type cross_validation_type: str, optional
      :param number_folds: The number of folds to use in the k fold cross validation. The number of
                           response DOFs must be evenly divisible by the number of folds.
      :type number_folds: int
      :param use_transformation: Whether or not the response and reference transformation from the class
                                 definition should be used (which is handled in the "linear_inverse_processing"
                                 decorator function). The default is true.
      :type use_transformation: bool, optional
      :param response: The preprocessed response data. The preprocessing is handled by the decorator
                       function and object definition. This argument should not be supplied by the
                       user.
      :type response: ndarray
      :param frf: The preprocessed frf data. The preprocessing is handled by the decorator
                  function and object definition. This argument should not be supplied by the
                  user.
      :type frf: ndarray

      :returns: **selected_force** -- An ndarray of the estimated sources. The "linear_inverse_processing" decorator
                function applies this force to the force property of SourcePathReceiver object.
      :rtype: ndarray

      .. rubric:: Notes

      All the settings, including the selected regularization parameters, are saved to
      the "inverse_settings" class property.

      .. rubric:: References

      .. [1] Wikipedia, "Overlap-add Method".
             https://en.wikipedia.org/wiki/Overlap-add_method
      .. [2] D. M. Allen, "The Relationship between Variable Selection and Data Agumentation
             and a Method for Prediction," Technometrics, vol. 16, no. 1, pp. 125-127, 1974,
             doi: 10.2307/1267500.
      .. [3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning:
             Data Mining, Inference, and Prediction, 2nd Edition ed. New York: Springer New York,
             2017.



