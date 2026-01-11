forcefinder.core.auto_regularization
====================================

.. py:module:: forcefinder.core.auto_regularization

.. autoapi-nested-parse::

   Contains helper functions for performing automatic regularization with
   the truncated singular value decomposition, ridge regression (Tikhonov),
   and the elastic net. These functions are intended to work with data from
   a SourcePathReceiver object.

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

   forcefinder.core.auto_regularization.tikhonov_full_path_for_l_curve
   forcefinder.core.auto_regularization.compute_regularized_residual_penalty_for_l_curve
   forcefinder.core.auto_regularization.compute_regularized_svd_inv
   forcefinder.core.auto_regularization.mean_squared_error
   forcefinder.core.auto_regularization.leave_one_out_cv
   forcefinder.core.auto_regularization.k_fold_cv
   forcefinder.core.auto_regularization.l_curve_optimal_regularization
   forcefinder.core.auto_regularization.broadcasting_l_curve_criterion
   forcefinder.core.auto_regularization.broadcasting_l_curve_by_distance
   forcefinder.core.auto_regularization.select_model_by_information_criterion
   forcefinder.core.auto_regularization.compute_tikhonov_regularized_frf_pinv
   forcefinder.core.auto_regularization.compute_regularized_forces
   forcefinder.core.auto_regularization.l_curve_selection
   forcefinder.core.auto_regularization.l_curve_criterion
   forcefinder.core.auto_regularization.optimal_l_curve_by_distance
   forcefinder.core.auto_regularization.compute_residual_penalty_for_l_curve


Module Contents
---------------

.. py:function:: tikhonov_full_path_for_l_curve(frfs, response, low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100)

   Performs the inverse force estimation problem and computes the necessary residual and
   penalty for model selection with the L-curve criterion.

   :param frfs: The array of FRFs for the inverse force estimation problem, should be organized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param response: The response to compute the forces from. Should be sized [number of lines, number of responses]
                    for linear spectra or [number of lines, number of responses, number of responses] for power spectra.
   :type response: ndarray
   :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                    array with a length that matches the number of frequency lines in
                                    the SourcePathReceiver object. The default is the smallest singular
                                    value of the frf array (on a frequency-frequency basis).
   :type low_regularization_limit: ndarray, optional
   :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                     array with a length that matches the number of frequency lines in
                                     the SourcePathReceiver object. The default is the largest singular
                                     value of the frf array (on a frequency-frequency basis).
   :type high_regularization_limit: ndarray, optional
   :param number_regularization_values: The number of regularization parameters to search over, where the
                                        potential parameters are geometrically spaced between the low and high
                                        regularization limits. The default is 100.
   :type number_regularization_values: int, optional

   :returns: * **regularization_values** (*ndarray*) -- The regularization parameters that were used in the inverse. It is sized
               [number of parameters, number of lines].
             * **residual** (*ndarray*) -- The residual error (i.e., sum squared error) between the computed and truth response.
               It is sized [number of parameters, number of lines].
             * **penalty** (*ndarray*) -- The penalty to use in the L-curve parameter selection, which is the squared sum
               of the forces. It is sized [number of parameters, number of lines].

   .. rubric:: Notes

   The regularization values are spread over a geometric sequence that spans
   the low and high regularization limits.

   The high and low regularization parameter limits cannot be set to zero in this function.
   Any values that are zero are reset to machine epsilon.


.. py:function:: compute_regularized_residual_penalty_for_l_curve(frfs, response, Uh, regularized_S, V)

   Computes the residual and penalty from the regularized SVD components of an FRF
   for an automatic Tikhonov regularization problem.

   :param frfs: The array of FRFs for the inverse force estimation problem, should be organized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param response: The response to compute the forces from. Should be sized [number of lines, number of responses]
                    for linear spectra or [number of lines, number of responses, number of responses] for power spectra.
   :type response: ndarray
   :param Uh: The conjugate-transpose of the left singular vectors of the supplied FRFs.
              It is shaped [number of lines, number of responses, number of references].
   :type Uh: ndarray
   :param V: The right singular vectors of the supplied FRFs. It is shaped
             [number of lines, number of references, number of references].
   :type V: ndarray
   :param regularized_S: The inverted and Tikhonov regularized singular values of the supplied
                         FRFs. It is shaped [number of values, number of lines, number of references].
   :type regularized_S: ndarray

   :returns: * **regularization_values** (*ndarray*) -- The regularization parameters that were used in the inverse. It is sized
               [number of parameters, number of lines].
             * **residual** (*ndarray*) -- The residual error (i.e., sum squared error) between the computed and truth response.
               It is sized [number of parameters, number of lines].
             * **penalty** (*ndarray*) -- The penalty to use in the L-curve parameter selection, which is the squared sum
               of the forces. It is sized [number of parameters, number of lines].


.. py:function:: compute_regularized_svd_inv(frfs, low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100)

   Computes the inverted regularized SVD components for an automatic Tikhonov
   regularization problem.

   :param frfs: The array of FRFs for the inverse force estimation problem, should be organized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                    array with a length that matches the number of frequency lines in
                                    the SourcePathReceiver object. The default is the smallest singular
                                    value of the frf array (on a frequency-frequency basis).
   :type low_regularization_limit: ndarray, optional
   :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                     array with a length that matches the number of frequency lines in
                                     the SourcePathReceiver object. The default is the largest singular
                                     value of the frf array (on a frequency-frequency basis).
   :type high_regularization_limit: ndarray, optional
   :param number_regularization_values: The number of regularization parameters to search over, where the
                                        potential parameters are geometrically spaced between the low and high
                                        regularization limits. The default is 100.
   :type number_regularization_values: int, optional

   :returns: * **regularization_values** (*ndarray*) -- The regularization parameters that were used in the inverse. It is sized
               [number of parameters, number of lines].
             * **Uh** (*ndarray*) -- The conjugate-transpose of the left singular vectors of the supplied FRFs.
               It is shaped [number of lines, number of responses, number of references].
             * **V** (*ndarray*) -- The right singular vectors of the supplied FRFs. It is shaped
               [number of lines, number of references, number of references].
             * **regularized_S** (*ndarray*) -- The inverted and Tikhonov regularized singular values of the supplied
               FRFs. It is shaped [number of values, number of lines, number of references].

   .. rubric:: Notes

   The regularization values are spread over a geometric sequence that spans
   the low and high regularization limits.

   The high and low regularization parameter limits cannot be set to zero in this function.
   Any values that are zero are reset to machine epsilon.


.. py:function:: mean_squared_error(validation_frfs, validation_response, frf_inverse, training_response)

   Computes the mean squared error of the validation response(s) from the
   inverse problem.

   :param validation_frfs: The FRFs that are used to compute the predicted response(s) from the
                           predicted forces. Should be sized [number of lines, number of validation responses, number of references].
   :type validation_frfs: ndarray
   :param validation_response: The validation response to compare the predicted response against.
                               Should be sized [number of lines, number of validation responses].
   :type validation_response: ndarray
   :param frf_inverse: The inverted FRFs that are used to compute the predicted forces.
                       Should be sized [number of lines, number of references, number of training responses].
   :type frf_inverse: ndarray
   :param training_response: The training responses that are used to compute the predicted forces.
                             Should be sized [number of lines, number of training responses].
   :type training_response: ndarray

   :returns: **mean_squared_error** -- The mean squared error of the predicted response. It is a 1D array that
             is sized the number of lines.
   :rtype: ndarray

   .. rubric:: Notes

   This function assumes that there are at least two DOFs in the training response.


.. py:function:: leave_one_out_cv(frfs, response, low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, error_function=mean_squared_error)

   Estimates the error in a force estimation problem with leave one out cross
   validation.

   :param frfs: The full FRF array to use in the inverse problem. It should be sized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param response: The full response array to use in the inverse problem. It should be
                    sized [number of lines, number of responses].
   :type response: ndarray
   :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                    array with a length that matches the number of frequency lines in
                                    the SourcePathReceiver object. The default is the smallest singular
                                    value of the frf array (on a frequency-frequency basis).
   :type low_regularization_limit: ndarray, optional
   :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                     array with a length that matches the number of frequency lines in
                                     the SourcePathReceiver object. The default is the largest singular
                                     value of the frf array (on a frequency-frequency basis).
   :type high_regularization_limit: ndarray, optional
   :param number_regularization_values: The number of regularization parameters to search over, where the
                                        potential parameters are geometrically spaced between the low and high
                                        regularization limits. The default is 100.
   :type number_regularization_values: int, optional
   :param error_function: The function that is used to compute the error in the inverse
                          problem. The function should take the following parameters:
                              - validation_frfs - ndarray that is sized [number of lines, number of validation responses, number of references].
                              - validation_response - ndarray that is sized [number of lines, number of responses].
                              - frf_inverse - ndarray that is sized [number of lines, number of references, number of training responses].
                              - training_response - ndarray that is sized [number of lines, number of training responses, 1].
                          The function should output:
                              - error - a 1d array that is sized number of lines.
                          The default function computes the residual squared error.
   :type error_function: function, optional

   :returns: * **regularization_values** (*ndarray*) -- The regularization values for the inverse problem. It is sized
               [number_regularization_values, number of lines].
             * **error** (*ndarray*) -- The prediction error from the inverse problem. It is sized
               [number_regularization_values, number of responses, number of lines].


.. py:function:: k_fold_cv(frfs, response, number_folds=None, random_seed=randbits(128), low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100, error_function=mean_squared_error)

   Estimates the error in a force estimation problem with K-fold cross
   validation.

   :param frfs: The full FRF array to use in the inverse problem. It should be sized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param response: The full response array to use in the inverse problem. It should be
                    sized [number of lines, number of responses].
   :type response: ndarray
   :param number_folds: The number of folds to use in the cross validation. The number of
                        response DOFs must be evenly divisible by the number of folds.
   :type number_folds: int
   :param random_seed: The seed for the random number generator that selects response DOFs
                       that are used in the different folds of the cross validation. The
                       default is a randomly selected integer.
   :type random_seed: int
   :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                    array with a length that matches the number of frequency lines in
                                    the SourcePathReceiver object. The default is the smallest singular
                                    value of the frf array (on a frequency-frequency basis).
   :type low_regularization_limit: ndarray, optional
   :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                     array with a length that matches the number of frequency lines in
                                     the SourcePathReceiver object. The default is the largest singular
                                     value of the frf array (on a frequency-frequency basis).
   :type high_regularization_limit: ndarray, optional
   :param number_regularization_values: The number of regularization parameters to search over, where the
                                        potential parameters are geometrically spaced between the low and high
                                        regularization limits. The default is 100.
   :type number_regularization_values: int, optional
   :param error_function: The function that is used to compute the error in the inverse
                          problem. The function should take the following parameters:
                              - validation_frfs - ndarray that is sized [number of lines, number of validation responses, number of references].

                              - validation_response - ndarray that is sized [number of lines, number of responses].

                              - frf_inverse - ndarray that is sized [number of lines, number of references, number of training responses].

                              - training_response - ndarray that is sized [number of lines, number of training responses, 1].
                          The function should return a 1d array that is sized number of lines.
                          The default function computes the mean squared error.
   :type error_function: function, optional

   :returns: * **regularization_values** (*ndarray*) -- The regularization values for the inverse problem. It is sized
               [number_regularization_values, number of lines].
             * **error** (*ndarray*) -- The prediction error from the inverse problem. It is sized
               [number_regularization_values, number of folds, number of lines].

   :raises ValueError: If the number of response DOFs is not evenly divisible by the number
       of folds.

   .. rubric:: Notes

   The response DOFs for the cross validation folds are selected by a
   random number generator. This can lead to different results for multiple
   function runs, since the default behavior is to use a different random
   number generator every time the function is ran. A random seed can be
   supplied to avoid the variability in the response DOF selection.
   However, different computers may still select different response DOFs
   for the different folds (even with the same seed), for a variety of
   reasons.


.. py:function:: l_curve_optimal_regularization(regularization_values, penalty, residual, l_curve_type='forces', optimality_condition='curvature')

   Selects the optimal regularization parameter using L-curve methods.

   :param regularization_values: The regularization values that were searched over. This should be sized
                                 [number of regularization values, number of frequency lines].
   :type regularization_values: ndarray
   :param penalty: The penalty from the regularized least squares problem. This should be sized
                   [number of regularization values, number of frequency lines].
   :type penalty: ndarray
   :param residual: The residual from the regularized least squares problem (typically the
                    mean squared error). This should be sized
                    [number of regularization values, number of frequency lines].
   :type residual: ndarray
   :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                        parameter. The available types are:
                            - forces (default) - This L-curve is constructed with the "size"
                            of the forces on the Y-axis and the regularization parameter on the
                            X-axis.

                            - standard - This L-curve is constructed with the residual squared
                            error on the X-axis and the "size" of the forces on the Y-axis.
   :type l_curve_type: str
   :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                The options are:
                                    - curvature (default) - This method searches for the regularization
                                    parameter that results in maximum curvature of the L-curve. It is
                                    also referred to as the L-curve criterion.

                                    - distance - This method searches for the regularization parameter that
                                    minimizes the distance between the L-curve and a "virtual origin". A
                                    virtual origin is used, because the L-curve is scaled and offset to always
                                    range from zero to one, in this case.
   :type optimality_condition: str

   :returns: **optimal_regularization** -- A vector of the optimal regularization values, as defined by the L-curve.
             The length matches that number of frequency lines.
   :rtype: ndarray

   :raises ValueError: If the requested L-curve type is not available.
   :raises ValueError: If the requested optimality condition is not available.

   .. rubric:: References

   .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
       of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
       vol. 14, no. 6, pp. 1487-1503, 1993.
   .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
       problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
       2000, pp. 119-142.
   .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
       Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.


.. py:function:: broadcasting_l_curve_criterion(x_axis, y_axis, regularization_values, return_curvature=False)

   Finds the "optimal" regularization value from an L-curve via the
   location where its curvature is at a maximum (the L-curve criterion).

   :param x_axis: This is a vector that defines the X-axis of the L-curve. The variable
                  that is used for this depends on the type of L-curve that is being
                  used. This should be sized [number of regularization values, number of frequency lines].
   :type x_axis: ndarray
   :param y_axis: This is a vector that defines the Y-axis of the L-curve. The variable
                  that is used for this depends on the type of L-curve that is being
                  used. This should be sized [number of regularization values, number of frequency lines].
   :type y_axis: ndarray
   :param regularization_values: This is a vector of regularization values that were used in the linear
                                 regression problem that created the L-curve. This should be sized
                                 [number of regularization values, number of frequency lines].
   :type regularization_values: ndarray
   :param return_curvature: Whether or not to return the computed curvature function. The default is
                            False.
   :type return_curvature: bool

   :returns: * **optimal_regularization** (*ndarray*) -- This is the optimal regularization value based on the L-curve
               criterion. It is sized [number of frequency lines]
             * **idx** (*ndarray*) -- The index that correspond to the optimal curvature. It is sized
               [number of frequency lines]
             * **curvature** (*ndarray, optional*) -- A vector of the curvature of the L-curve for the give sequence
               of regularization_values. This is not returned by default. It is
               sized [number of regularization values, number of frequency lines].

   .. rubric:: Notes

   The code will automatically skip frequency lines where the regularization parameter
   doesn't change (e.g., 0 Hz).

   .. rubric:: References

   .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
   .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.


.. py:function:: broadcasting_l_curve_by_distance(x_axis, y_axis, regularization_values)

   Finds the "optimal" regularization from an L-curve by finding the parameter
   that puts the curve closest to the "virtual origin".

   :param x_axis: This is a vector that defines the X-axis of the L-curve. The variable
                  that is used for this depends on the L-curve type this should be sized
                  [number of regularization values, number of frequency lines].
   :type x_axis: ndarray
   :param y_axis: This is a vector that defines the Y-axis of the L-curve. The variable
                  that is used for this depends on the L-curve type this should be sized
                  [number of regularization values, number of frequency lines].
   :type y_axis: ndarray
   :param regularization_values: This is a vector of regularization values that were used in the linear
                                 regression problem that created the L-curve. This should be sized
                                 [number of regularization values, number of frequency lines].
   :type regularization_values: ndarray

   :returns: * **optimal_regularization** (*float*) -- This is the optimal regularization value based on the L-curve distance
               from the origin.
             * **idx** (*int*) -- The index that correspond to the optimal curvature.

   .. rubric:: Notes

   This technique applies a scale and offset to the L-curve so the X and Y-axis
   always ranges from zero to one. This is required to obtain predictable
   behavior from the method, but can also distort the shape of the curve.


.. py:function:: select_model_by_information_criterion(H, x, f, method)

   Performs model selection with the desired information criterion.

   :param H: The FRFs that were used to estimate the forces, sized:
             [number_lines, number_responses, number_references].
   :type H: ndarray
   :param x: The responses that were used in the force estimation, sized:
             [number_lines, number_responses].
   :type x: ndarray
   :param f: The estimated forces from the model, sized:
             [number_lines, number_models, number_references]
   :type f: ndarray
   :param method:
                  The desired information criterion, the available options are:

                      - 'BIC': the Bayesian information criterion

                      - 'AIC': the Akaike information criterion

                      - 'AICC': the corrected Akaike information criterion
   :type method: str

   :returns: **selected_forces** -- The forces that were selected using the desired model, sized:
             [number_lines, number_references]
   :rtype: ndarray

   :raises NotImplementedError: If the supplied responses are a CPSD matrix.


.. py:function:: compute_tikhonov_regularized_frf_pinv(frfs, low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100)

   Computes the Tikhonov regularized FRF pseudo-inverse for many regularization parameters.

   :param frfs: The array of FRFs to be inverted, should be organized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param low_regularization_limit: The low regularization limit to search through. This should be a 1d
                                    array with a length that matches the number of frequency lines in
                                    the SourcePathReceiver object. The default is the smallest singular
                                    value of the frf array (on a frequency-frequency basis).
   :type low_regularization_limit: ndarray
   :param high_regularization_limit: The high regularization limit to search through. This should be a 1d
                                     array with a length that matches the number of frequency lines in
                                     the SourcePathReceiver object. The default is the largest singular
                                     value of the frf array (on a frequency-frequency basis).
   :type high_regularization_limit: ndarray
   :param number_regularization_values: The number of regularization parameters to search over, where the
                                        potential parameters are geometrically spaced between the low and high
                                        regularization limits. The default is 100.
   :type number_regularization_values: int

   :returns: * **regularized_frf_pinv** (*ndarray*) -- Regularized pseudo-inverse of the FRF array. It is sized
               [number of parameters, number of lines, number of responses, number of references].
             * **lambda_values** (*ndarray*) -- The regularization parameters that were used in the inverse. It is sized
               [number of parameters, number of lines].


.. py:function:: compute_regularized_forces(regularized_frf_inverse, response)

   Computes the forces from a regularized inverse of an FRF matrix.

   :param regularized_frf_inverse: Regularized pseudo-inverse of the FRF array. Should be sized
                                   [number of parameters, number of lines, number of references, number of responses].
   :type regularized_frf_inverse: ndarray
   :param response: The response to compute the forces from. Should be sized [number of lines, number of responses]
                    for linear spectra or [number of lines, number of responses, number of responses] for power spectra.
   :type response: ndarray

   :returns: **regularized_forces** -- The regularized forces from the supplied parameters. It is sized
             [number of parameters, number of lines, number of references] for linear spectra
             or [number of parameters, number of lines, number of references, number of references]
             for power spectra.
   :rtype: ndarray


.. py:function:: l_curve_selection(regularization_values, penalty, residual, forces_full_path, l_curve_type='forces', optimality_condition='curvature', curvature_method='numerical')

   Selects the optimal regularization parameter and forces using L-curve methods

   :param regularization_values: The regularization values that were searched over. This should be sized
                                 [number of frequency lines, number of regularization values]
   :type regularization_values: ndarray
   :param penalty: The penalty from the regularized least squares problem. This should be sized
                   [number of frequency lines, number of regularization values]
   :type penalty: ndarray
   :param residual: The residual from the regularized least squares problem (typically the
                    mean squared error). This should be sized [number of frequency lines, number of regularization values]
   :type residual: ndarray
   :param forces_full_path: The forces that were estimated in the regularized least squares problem.
                            It should be sized such that the number of frequency lines is on the
                            first axis and the number of regularization values is on the second axis.
   :type forces_full_path: ndarray
   :param l_curve_type: The type of L-curve that is used to find the "optimal regularization
                        parameter. The available types are:
                            - forces (default) - This L-curve is constructed with the "size"
                            of the forces on the Y-axis and the regularization parameter on the
                            X-axis.
                            - standard - This L-curve is constructed with the residual squared
                            error on the X-axis and the "size" of the forces on the Y-axis.
   :type l_curve_type: str
   :param optimality_condition: The method that is used to find an "optimal" regularization parameter.
                                The options are:
                                    - curvature (default) - This method searches for the regularization
                                    parameter that results in maximum curvature of the L-curve. It is
                                    also referred to as the L-curve criterion.
                                    - distance - This method searches for the regularization parameter that
                                    minimizes the distance between the L-curve and a "virtual origin". A
                                    virtual origin is used, because the L-curve is scaled and offset to always
                                    range from zero to one, in this case.
   :type optimality_condition: str
   :param curvature_method: The method that is used to compute the curvature of the L-curve, in the
                            case that the curvature is used to find the optimal regularization
                            parameter. The options are:
                                - numerical (default) - this method computes the curvature of
                                the L-curve via numerical derivatives
                                - cubic_spline - this method fits a cubic spline to the L-curve
                                the computes the curvature from the cubic spline (this might
                                perform better if the L-curve isn't "smooth")
   :type curvature_method: str

   :raises ValueError: If the requested L-curve type is not available.
   :raises ValueError: If the requested optimality condition is not available.

   :returns: * **chosen_force** (*ndarray*) -- The force at the optimal regularization value, as defined by the L-curve.
               It is sized [number of frequency lines, force array size].
             * **optimal_regularization** (*ndarray*) -- A vector of the optimal regularization values, as defined by the L-curve.
               The length matches that number of frequency lines.

   .. rubric:: Notes

   This function can handle forces as either power spectra or linear spectra.

   .. rubric:: References

   .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
       of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
       vol. 14, no. 6, pp. 1487-1503, 1993.
   .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
       problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
       2000, pp. 119-142.
   .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
       Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.


.. py:function:: l_curve_criterion(x_axis, y_axis, regularization_values, method='numerical', return_curvature=False)

   Finds the "optimal" regularization value from an L-curve via the
   location where its curvature is at a maximum (the L-curve criterion).

   :param x_axis: This is a vector that defines the X-axis of the L-curve. The variable
                  that is used for this depends on the type of L-curve that is being
                  used.
   :type x_axis: ndarray
   :param y_axis: This is a vector that defines the Y-axis of the L-curve. The variable
                  that is used for this depends on the type of L-curve that is being
                  used.
   :type y_axis: ndarray
   :param regularization_values: This is a vector of regularization values that were used in the linear
                                 regression problem that created the L-curve.
   :type regularization_values: ndarray
   :param method: This is the method by which the curvature is computed, the
                  available methods are:
                      - numerical (default) - this method computes the curvature of
                        the L-curve via numerical derivatives
                      - cubic_spline - this method fits a cubic spline to the L-curve
                        the computes the curvature from the cubic spline (this might
                        perform better if the L-curve isn't "smooth")
   :type method: str

   :returns: * **optimal_regularization** (*float*) -- This is the optimal regularization value based on the L-curve
               criterion.
             * **idx** (*int*) -- The index that correspond to the optimal curvature.
             * **curvature** (*ndarray*) -- A vector of the curvature of the L-curve for the give sequence
               of regularization_values.

   .. rubric:: References

   .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization
          of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
          vol. 14, no. 6, pp. 1487-1503, 1993.
   .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
          problems," in Computational Inverse Problems in Electrocardiology," WIT Press,
          2000, pp. 119-142.


.. py:function:: optimal_l_curve_by_distance(x_axis, y_axis, regularization_values)

   Finds the "optimal" regularization from an L-curve by finding the parameter
   that puts the curve closest to the "virtual origin".

   :param x_axis: This is a vector that defines the X-axis of the L-curve. The variable
                  that is used for this depends on the type of L-curve that is being
                  used.
   :type x_axis: ndarray
   :param y_axis: This is a vector that defines the Y-axis of the L-curve. The variable
                  that is used for this depends on the type of L-curve that is being
                  used.
   :type y_axis: ndarray
   :param regularization_values: This is a vector of regularization values that were used in the linear
                                 regression problem that created the L-curve.
   :type regularization_values: ndarray

   :returns: * **optimal_regularization** (*float*) -- This is the optimal regularization value based on the L-curve distance
               from the origin.
             * **idx** (*int*) -- The index that correspond to the optimal curvature.

   .. rubric:: Notes

   This technique applies a scale and offset to the L-curve so the X and Y-axis
   always ranges from zero to one. This is required to obtain predictable
   behavior from the method, but can also distort the shape of the curve.


.. py:function:: compute_residual_penalty_for_l_curve(frfs, forces, response)

   Computes the residual and penalty for an inverse force estimation problem
   in preparation for model selection with the L-curve criterion.

   :param frfs: The FRFs that are used to compute the system response. Should be sized
                [number of lines, number of responses, number of references].
   :type frfs: ndarray
   :param forces: The estimated forces from the inverse problem. Should be sized
                  [number of parameters, number of lines, number of references] for linear spectra
                  or [number of parameters, number of lines, number of references, number of references]
                  for power spectra
   :type forces: ndarray
   :param response: The "truth" responses to compare the computed responses to. Should be
                    sized [number of lines, number of responses] for linear spectra or
                    [number of lines, number of responses, number of responses] for power spectra.
   :type response: ndarray

   :returns: * **residual** (*ndarray*) -- The residual error (i.e., sum squared error) between the computed and truth response.
               It is sized [number of parameters, number of lines].
             * **penalty** (*ndarray*) -- The penalty to use in the L-curve parameter selection, which is the squared sum
               of the forces. It is sized [number of parameters, number of lines].

   .. rubric:: Notes

   The returned residual and penalty are the squared norm of the residual error vector
   and force vector, respectively.


