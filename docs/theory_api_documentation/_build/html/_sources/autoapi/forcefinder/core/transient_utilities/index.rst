forcefinder.core.transient_utilities
====================================

.. py:module:: forcefinder.core.transient_utilities

.. autoapi-nested-parse::

   Contains helper functions for transient source estimation problems.

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

   forcefinder.core.transient_utilities.prepare_for_zero_padding
   forcefinder.core.transient_utilities.generate_zero_padded_response_fft
   forcefinder.core.transient_utilities.generate_signal_from_cola_frames
   forcefinder.core.transient_utilities.frequency_interpolation
   forcefinder.core.transient_utilities.sinc_interpolation
   forcefinder.core.transient_utilities.attenuate_signal
   forcefinder.core.transient_utilities.create_zero_padded_response_fft
   forcefinder.core.transient_utilities.reconstruct_cola_frames


Module Contents
---------------

.. py:function:: prepare_for_zero_padding(spr_object, cola_frame_length=None, cola_window=('tukey', 0.5), cola_overlap_samples=None)

   Identifies and creates several parameters that will be used when zero
   padding the data in a transient source estimation problem.

   :param spr_object: The transient SPR object that is being used for the source estimation
                      problem.
   :type spr_object: TransientSourcePathReceiver
   :param cola_frame_length: The frame length (in samples) for the COLA processing. The default frame
                             length is approximately Fs/df from the FRFs.
   :type cola_frame_length: int, optional
   :param cola_window: The window to use in the COLA processing. This input is passed directly to
                       the SciPy get_window function. The default is a Tukey window with an alpha
                       of 0.5.
   :type cola_window: string, float, or tuple
   :param cola_overlap_samples: The overlap (in samples) between the frames in the COLA processing, such that
                                cola_overlap_samples/cola_frame_length results in the overlap percentage. A
                                default is defined for the default Tukey window.
   :type cola_overlap_samples: int, optional

   :returns: * **padded_response** (*ndarray*) -- The training response of the SPR object with zero padding on the start and
               end, to account for the windowing in the COLA procedure. It is shaped
               [number of samples, number of responses].
             * **signal_sizes** (*dict*) -- The various signal sizes that were used in the zero padding operation. The
               dictionary includes the following keys:
                   - pre_data_blank_frame_length - The amount of zeros that were added to
                   the beginning of the training response to enable a perfect COLA
                   reconstruction.

                   - post_data_blank_frame_length - The amount of zeros that were added to
                   the end of the training response to enable a perfect COLA reconstruction.

                   - left_zero_pad_length - The amount of zeros that were used to pad the
                   left side of the COLA frames.

                   - right_zero_pad_length - The amount of zeros that were used to pad the
                   right side of the COLA frames.

                   - number_cola_frames - The number of frames to segment the data into for
                   the COLA processing.

                   - cola_frame_length - The frame length (in samples) that was use for the
                   COLA processing (corresponds to the cola_frame_length input parameter).

                   - cola_overlap_samples - The overlap (in samples) between the frames in
                   the COLA processing (corresponds to the cola_overlap_samples input parameter).

                   - zero_padded_signal_length - The frame length (in samples) of each zero
                   padded COLA frame.
             * **frame_indices** (*ndarray*) -- The indices to segment the padded_response into the COLA frames. It is shaped
               [number of frames, cola frame length].
             * **window** (*ndarray*) -- The window that was used in the COLA processing.

   .. rubric:: Notes

   The zero_padded_signal_length is determined by an algorithm that uses the SciPy
   next_fast_len function, which can return an odd length. Care should be taken when
   reconstructing the signal with an inverse FFT.


.. py:function:: generate_zero_padded_response_fft(full_data, frame_indices, signal_sizes, window, response_transform, use_transformation)

   Yields an FFT for a zero padded segment of the provided data.

   :param full_data: The data to segment and FFT. It should be shaped
                     [number of samples, number of channels].
   :type full_data: ndarray
   :param frame_indices: The indices to segment the data. It should be shaped
                         [number of frames, number of samples per frame].
   :type frame_indices: ndarray
   :param signal_sizes: A dictionary with the signal sizes for the segmentation and zero
                        padding. The keys in the dictionary should include:
                            - left_zero_pad_length - The amount of zeros that were used to pad the
                            left side of the COLA frames.

                            - right_zero_pad_length - The amount of zeros that were used to pad the
                            right side of the COLA frames.

                            - number_cola_frames - The number of frames to segment the data into for
                            the COLA processing.

                            - zero_padded_signal_length - The frame length (in samples) of each zero
                            padded COLA frame.
   :type signal_sizes: dict
   :param window: The window (as a 1D array) to apply to the segmented data frames.
   :type window: ndarray
   :param response_transform: The response transformation to apply to the FFT of responses. It should be
                              shaped [number of frequency lines, number of response, number of responses].
   :type response_transform: ndarray
   :param use_transformation: Whether or not to use the response transformation.
   :type use_transformation: bool

   :Yields: **padded_response_fft** (*ndarray*) -- The FFT of the zero padded segment of data.


.. py:function:: generate_signal_from_cola_frames(signal_sizes, return_signal_length, cola_window, number_of_dofs, reference_transform, use_transformation)

   Reconstructs a full time signal from cola frame FFTs.

   :param signal_sizes: The various signal sizes that were used in the original COLA segmentation
                        and zero padding operation. The dictionary should include the following keys:
                            - pre_data_blank_frame_length - The amount of zeros that were added to
                            the beginning of the training response to enable a perfect COLA
                            reconstruction.

                            - post_data_blank_frame_length - The amount of zeros that were added to
                            the end of the training response to enable a perfect COLA reconstruction.

                            - left_zero_pad_length - The amount of zeros that were used to pad the
                            left side of the COLA frames.

                            - right_zero_pad_length - The amount of zeros that were used to pad the
                            right side of the COLA frames.

                            - number_cola_frames - The number of frames to segment the data into for
                            the COLA processing.

                            - cola_frame_length - The frame length (in samples) that was use for the
                            COLA processing (corresponds to the cola_frame_length input parameter).

                            - cola_overlap_samples - The overlap (in samples) between the frames in
                            the COLA processing (corresponds to the cola_overlap_samples input parameter).

                            - zero_padded_signal_length - The frame length (in samples) of each zero
                            padded COLA frame.
   :type signal_sizes: dict
   :param return_signal_length: The desired number of samples in the returned signal.
   :type return_signal_length: int
   :param cola_window: The window (as a 1d array) that was used in the original COLA segmentation.
   :type cola_window: ndarray
   :param number_of_dofs: The number of DOFs in the reconstructed signal.
   :type number_of_dofs: int
   :param reference_transform: The reference transformation to apply to the FFT of responses. It should be
                               shaped [number of frequency lines, number of references, number of references].
   :type reference_transform: ndarray
   :param use_transformation: Whether or not to use the reference transformation.
   :type use_transformation: bool

   :Yields: **reconstructed_signal** (*ndarray*) -- The reconstructed signal, it is sized [number of samples, number of DOFs]

   .. rubric:: Notes

   Although it isn't directly passed as a parameter for the function, the user must
   pass a "segment_fft" variable to the generator to recompile the signal in the
   time domain. This variable should be shaped
   [number of frequency lines, number of dofs, 1].


.. py:function:: frequency_interpolation(original_frequency, original_ordinate, padded_length, dt, interpolation_type='cubic')

   Interpolates a frequency domain function with the specified technique to account
   for differing block sizes in frequency and time domain data.

   :param original_frequency: A 1D array for the frequency axis of the original data.
   :type original_frequency: ndarray
   :param original_ordinate: The ordinate for the original data, should be shaped
                             [number of frequency lines, ...].
   :type original_ordinate: ndarray
   :param padded_length: The number of samples in the time data that accompanies the frequency data.
   :type padded_length: int
   :param dt: The sampling period for the time data that accompanies the frequency data.
   :type dt: float
   :param interpolation: The type of interpolation to use, this should match one of the available
                         types in scipy.interpolate.interp1d. The default is 'cubic'.
   :type interpolation: str, optional

   :returns: **interpolated_ordinate** -- The ordinate for the interpolated data. It is shaped
             [number of frequency lines, ...].
   :rtype: ndarray

   .. rubric:: Notes

   This is essentially a wrapper around scipy.interpolate.interp1d. It sets all
   frequencies outside the band of interest to zero. The first (i.e. 0 Hz) and
   last frequencies are set to zero, regardless.


.. py:function:: sinc_interpolation(original_ordinate, padded_length)

   Sinc interpolates a frequency domain function (via an FFT) to account for
   differing block sizes in frequency and time domain data.

   :param original_ordinate: The ordinate for the original data, should be shaped
                             [number of frequency lines, ...].
   :type original_ordinate: ndarray
   :param padded_length: The number of samples in the time data that accompanies the frequency data.
   :type padded_length: int

   :returns: **interpolated_ordinate** -- The ordinate for the interpolated data. It is shaped
             [number of frequency lines, ...].
   :rtype: ndarray

   .. rubric:: Notes

   This function does not account for any apparent non-causalities in the
   original ordinate. Spline interpolation should be used for data with apparent
   non-causalities.


.. py:function:: attenuate_signal(input_waveform: numpy.ndarray, limit: Union[float, numpy.ndarray], full_scale: float = 1.0) -> numpy.ndarray

   Attenuate peaks that exceed limits in time domain by scaling the region between zero
   crossings using the local maximum. This maintains a smooth waveform that does not
   exceed the specified limits.

   :param input_waveform: 1D or 2D array with shape (n_signals, n_samples)
   :type input_waveform: np.ndarray
   :param limit: limit value or array of limit values with shape (n_signals,)
   :type limit: float | np.ndarray
   :param full_scale: global scaling factor applied to limit value, intended to be used such
                      that output waveform peaks are slightly less than physical limit,
                      (ex. use full_scale=0.97 so that output signal will not exceed 97% of specified limit),
                      by default 1.0
   :type full_scale: float, optional

   :returns: clipped waveform, with shape (n_signals, n_samples)
   :rtype: np.ndarray

   .. rubric:: Notes

   Each region between zero crossings is scaled according to:
       `full_scale * limit / local_maximum`


.. py:function:: create_zero_padded_response_fft(spr_object, cola_frame_length=None, cola_window=('tukey', 0.5), cola_overlap_samples=None)

   Creates the zero padded response FFT for a COLA source estimation problem.

   :param spr_object: The transient SPR object that is being used for the source estimation
                      problem.
   :type spr_object: TransientSourcePathReceiver
   :param cola_frame_length: The frame length (in samples) for the COLA processing. The default frame
                             length is approximately Fs/df from the FRFs.
   :type cola_frame_length: int, optional
   :param cola_window: The window to use in the COLA processing. This input is passed directly to
                       the SciPy get_window function. The default is a Tukey window with an alpha
                       of 0.5.
   :type cola_window: string, float, or tuple
   :param cola_overlap_samples: The overlap (in samples) between the frames in the COLA processing, such that
                                cola_overlap_samples/cola_frame_length results in the overlap percentage. A
                                default is defined for the default Tukey window.
   :type cola_overlap_samples: int, optional

   :returns: * **padded_response_fft** (*ndarray*) -- The zero padded response FFT. It is shaped
               [number of COLA frames, number of frequency lines, number of responses].
             * **signal_sizes** (*dict*) -- The various signal sizes that were used in the zero padding operation. The
               dictionary includes the following keys:
                   - pre_data_blank_frame_length - The amount of zeros that were added to
                   the beginning of the training response to enable a perfect COLA
                   reconstruction.

                   - post_data_blank_frame_length - The amount of zeros that were added to
                   the end of the training response to enable a perfect COLA reconstruction.

                   - left_zero_pad_length - The amount of zeros that were used to pad the
                   left side of the COLA frames.

                   - right_zero_pad_length - The amount of zeros that were used to pad the
                   right side of the COLA frames.

                   - cola_frame_length - The frame length (in samples) that was use for the
                   COLA processing (corresponds to the cola_frame_length input parameter).

                   - cola_overlap_samples - The overlap (in samples) between the frames in
                   the COLA processing (corresponds to the cola_overlap_samples input parameter).

                   - zero_padded_signal_length - The frame length (in samples) of each zero
                   padded COLA frame.
             * **window** (*ndarray*) -- The window that was used in the COLA processing.

   .. rubric:: Notes

   The zero_padded_signal_length is determined by an algorithm that uses the SciPy
   next_fast_len function, which can return an odd length. As such, care should be
   taken when reconstructing the signal with an inverse FFT.


.. py:function:: reconstruct_cola_frames(cola_ffts, signal_sizes, return_signal_length, cola_window)

   Reconstructs a full time signal from cola frame FFTs.

   :param cola_ffts: The cola frame FFTs to reconstruct the signal from. It should be sized
                     [number of frames, number of frequency lines, number of DOFs]
   :type cola_ffts: ndarray
   :param signal_sizes: The various signal sizes that were used in the original COLA segmentation
                        and zero padding operation. The dictionary should include the following keys:
                            - pre_data_blank_frame_length - The amount of zeros that were added to
                            the beginning of the training response to enable a perfect COLA
                            reconstruction.

                            - post_data_blank_frame_length - The amount of zeros that were added to
                            the end of the training response to enable a perfect COLA reconstruction.

                            - left_zero_pad_length - The amount of zeros that were used to pad the
                            left side of the COLA frames.

                            - right_zero_pad_length - The amount of zeros that were used to pad the
                            right side of the COLA frames.

                            - cola_frame_length - The frame length (in samples) that was use for the
                            COLA processing (corresponds to the cola_frame_length input parameter).

                            - cola_overlap_samples - The overlap (in samples) between the frames in
                            the COLA processing (corresponds to the cola_overlap_samples input parameter).

                            - zero_padded_signal_length - The frame length (in samples) of each zero
                            padded COLA frame.
   :type signal_sizes: dict
   :param return_signal_length: The desired number of samples in the returned signal.
   :type return_signal_length: int
   :param cola_window: The window (as a 1d array) that was used in the original COLA segmentation.
   :type cola_window: ndarray

   :returns: **reconstructed_signal** -- The reconstructed signal, it is sized [number of samples, number of DOFs]
   :rtype: ndarray


