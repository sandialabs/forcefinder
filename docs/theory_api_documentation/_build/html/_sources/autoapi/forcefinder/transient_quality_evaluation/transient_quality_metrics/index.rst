forcefinder.transient_quality_evaluation.transient_quality_metrics
==================================================================

.. py:module:: forcefinder.transient_quality_evaluation.transient_quality_metrics

.. autoapi-nested-parse::

   Defines the various response prediction quality metrics for transient
   source estimation problems.

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

   forcefinder.transient_quality_evaluation.transient_quality_metrics.preprocess_data_for_quality_metric
   forcefinder.transient_quality_evaluation.transient_quality_metrics.compute_global_rms_error
   forcefinder.transient_quality_evaluation.transient_quality_metrics.compute_average_rms_error
   forcefinder.transient_quality_evaluation.transient_quality_metrics.compute_time_varying_trac
   forcefinder.transient_quality_evaluation.transient_quality_metrics.compute_time_varying_level_error
   forcefinder.transient_quality_evaluation.transient_quality_metrics.compute_error_stft
   forcefinder.transient_quality_evaluation.transient_quality_metrics.compute_stft


Module Contents
---------------

.. py:function:: preprocess_data_for_quality_metric(spr_object, channel_set='training', samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None, window='boxcar')

   Selects the correct datasets from an SPR object and splits the data into frames
   for the transient response prediction quality metrics.

   :param spr_object: The SPR object to pull the data from.
   :type spr_object: TransientSourcePathReceiver
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
   :param window: The window to apply to the segmented data. This should a window that
                  is available from the `scipy.signal.get_window` function. The default
                  is boxcar.
   :type window: str, optional

   :returns: * **truth** (*TimeHistoryArray*) -- The truth response, segmented into frames per the selected options.
             * **reconstructed** (*TimeHistoryArray*) -- The reconstructed response, segmented into frames per the selected options.

   .. rubric:: Notes

   This returns the raw training response rather than the reconstructed training
   response.


.. py:function:: compute_global_rms_error(truth_response, check_response)

   Computes the global RMS error in dB of the reconstructed response,
   per the procedure in MIL-STD 810H.

   :param truth_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the truth response in the comparison.
   :type truth_response: TimeHistoryArray
   :param check_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the check response in the comparison.
   :type check_response: TimeHistoryArray

   :returns: Returns a time history array of the global RMS error in dB. The response
             coordinate for this array is made up to have a value for the DataArray
             and does not correspond to anything.
   :rtype: TimeHistoryArray

   :raises ValueError: If the response coordinate (and coordinate ordering) are not the same
       for the truth and check responses.
   :raises ValueError: If the ordinate for the truth and check responses are not the same
       shape.
   :raises ValueError: If the abscissa for the truth and check responses are not the same.

   .. rubric:: Notes

   This function assumes that both the truth and check responses have been
   segmented using the `split_into_frames` method.

   .. rubric:: References

   .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.


.. py:function:: compute_average_rms_error(truth_response, check_response)

   Computes the average RMS error in dB of the check response compared
   to the truth response.

   :param truth_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the truth response in the comparison.
   :type truth_response: TimeHistoryArray
   :param check_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the check response in the comparison.
   :type check_response: TimeHistoryArray

   :returns: Returns a time history array of the average RMS error in dB. The response
             coordinate for this array is made up to have a value for the DataArray
             and does not correspond to anything.
   :rtype: TimeHistoryArray

   :raises ValueError: If the response coordinate (and coordinate ordering) are not the same
       for the truth and check responses.
   :raises ValueError: If the ordinate for the truth and check responses are not the same
       shape.
   :raises ValueError: If the abscissa for the truth and check responses are not the same.

   .. rubric:: Notes

   This function assumes that both the truth and check responses have been
   segmented using the `split_into_frames` method.


.. py:function:: compute_time_varying_trac(truth_response, check_response)

   Computes the time varying time response assurance criterion (TRAC)
   between the check response and truth response for all the supplied
   degrees of freedom.

   :param truth_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the truth response in the comparison.
   :type truth_response: TimeHistoryArray
   :param check_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the check response in the comparison.
   :type check_response: TimeHistoryArray

   :returns: Returns a time history array of the time varying TRAC for all the
             response degrees of freedom.
   :rtype: TimeHistoryArray

   :raises ValueError: If the response coordinate (and coordinate ordering) are not the same
       for the truth and check responses.
   :raises ValueError: If the ordinate for the truth and check responses are not the same
       shape.
   :raises ValueError: If the abscissa for the truth and check responses are not the same.

   .. rubric:: Notes

   This function assumes that both the truth and check responses have been
   segmented using the `split_into_frames` method.


.. py:function:: compute_time_varying_level_error(truth_response, check_response, level_type='rms')

   Computes the computes the time varying error for a statistical level (rms
   or max) between the check and truth responses.

   :param truth_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the truth response in the comparison.
   :type truth_response: TimeHistoryArray
   :param check_response: The segmented TimeHistoryArray (from the split_into_frames method)
                          for the check response in the comparison.
   :type check_response: TimeHistoryArray
   :param level_type:
                      The type of level to be used in the comparison. The options are:

                          - rms (default)
                              The rms level error for each frame of data in the responses

                          - max
                              The error in the maximum level that is seem for each frame
                              of data in the responses.
   :type level_type: str, optional

   :returns: Returns a time history array of the time varying level error in dB for
             all the response degrees of freedom.
   :rtype: TimeHistoryArray

   :raises ValueError: If the response coordinate (and coordinate ordering) are not the same
       for the truth and check responses.
   :raises ValueError: If the ordinate for the truth and check responses are not the same
       shape.
   :raises ValueError: If the abscissa for the truth and check responses are not the same.

   .. rubric:: Notes

   This function assumes that both the truth and check responses have been
   segmented using the `split_into_frames` method.


.. py:function:: compute_error_stft(spr_object, channel_set='training', samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None, window='hann', normalize_by_rms=False)

   Computes the dB error short time Fourier transform (STFT) for the given SPR object.

   :param spr_object: The SPR object to pull the data from.
   :type spr_object: TransientSourcePathReceiver
   :param channel_set: The channel set to make the response comparisons between.
                       The available options are:

                           - training (default)
                               This compares the responses for the transformed training
                               DOFs in the SPR object.

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
   :param window: The window to apply to the segmented data. This should a window that
                  is available from the `scipy.signal.get_window` function. The default
                  is hann.
   :type window: str, optional
   :param normalize_by_rms: Whether or not to normalize the error magnitudes (at each time slice)
                            by the relative magnitude of the RMS level (for each DOF) of the truth
                            data. The normalization can help show the relative significance of the
                            spectrum error based on the amplitude of the time response (the errors
                            are scaled down where the RMS level is small). The default is false.
   :type normalize_by_rms: bool, option

   :returns: **db_error_stft** --

             A dictionary with the following keys:

                 - time (ndarray)
                     The time vector for the STFT.

                 - frequency (ndarray)
                     The frequency vector for the STFT.

                 - response_coordinate (coordinate_array)
                     The response coordinate array for the STFT.

                 - amplitude (ndarray)
                     The dB error STFT, it is organized [response coordinate,
                     frequency axis, time axis].
   :rtype: dict

   .. rubric:: Notes

   The dB error is computed from the magnitude of the STFT.


.. py:function:: compute_stft(data, samples_per_frame=None, frame_length=None, overlap=None, overlap_samples=None, window='hann')

   Computes the short time Fourier transform (STFT) for the given
   TimeHistoryArray and packages the data into a dictionary that can be
   used with the SpectrogramGUI tool.

   :param data: The data to compute the STFT for.
   :type data: TimeHistoryArray
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
   :param window: The window to apply to the segmented data. This should a window that
                  is available from the `scipy.signal.get_window` function. The default
                  is hann.
   :type window: str, optional

   :returns: **stft** --

             A dictionary with the following keys:

                 - time (ndarray)
                     The time vector for the STFT.

                 - frequency (ndarray)
                     The frequency vector for the STFT.

                 - response_coordinate (coordinate_array)
                     The response coordinate array for the STFT.

                 - amplitude (ndarray)
                     The STFT, it is organized [response coordinate,
                     frequency axis, time axis].
   :rtype: dict

   .. rubric:: Notes

   The STFT is the linear spectrum magnitude of the Fourier transform.


