"""
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
"""
import numpy as np
from scipy.signal import get_window, check_COLA
from scipy.fft import rfft, irfft, next_fast_len, rfftfreq
from scipy.interpolate import interp1d

def prepare_for_zero_padding(spr_object, 
                             cola_frame_length=None, 
                             cola_window=('tukey', 0.5),
                             cola_overlap_samples=None):
    """
    Identifies and creates several parameters that will be used when zero
    padding the data in a transient source estimation problem.

    Parameters
    ----------
    spr_object : TransientSourcePathReceiver
        The transient SPR object that is being used for the source estimation
        problem. 
    cola_frame_length : int, optional
        The frame length (in samples) for the COLA processing. The default frame 
        length is approximately Fs/df from the FRFs.
    cola_window : string, float, or tuple
        The window to use in the COLA processing. This input is passed directly to 
        the SciPy get_window function. The default is a Tukey window with an alpha 
        of 0.5.
    cola_overlap_samples : int, optional
        The overlap (in samples) between the frames in the COLA processing, such that
        cola_overlap_samples/cola_frame_length results in the overlap percentage. A
        default is defined for the default Tukey window.

    Returns
    -------
    padded_response : ndarray
        The training response of the SPR object with zero padding on the start and 
        end, to account for the windowing in the COLA procedure. It is shaped 
        [number of samples, number of responses].
    signal_sizes : dict
        The various signal sizes that were used in the zero padding operation. The
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
    frame_indices : ndarray
        The indices to segment the padded_response into the COLA frames. It is shaped
        [number of frames, cola frame length]. 
    window : ndarray
        The window that was used in the COLA processing. 

    Notes
    -----
    The zero_padded_signal_length is determined by an algorithm that uses the SciPy 
    next_fast_len function, which can return an odd length. Care should be taken when 
    reconstructing the signal with an inverse FFT.
    """
    model_order = 4*int(np.ceil((2*spr_object.abscissa.max()/spr_object.abscissa_spacing)/4)) # slightly strange operations to guarantee a model order that is divisible by 4
    if cola_frame_length is None:
        cola_frame_length = model_order

    window = get_window(cola_window, cola_frame_length)[..., np.newaxis]
    if cola_window[0] == 'tukey' and cola_overlap_samples is None:
        cola_overlap_samples = int(cola_frame_length*(cola_window[1]/2))
    if not check_COLA(window[...,0], model_order, cola_overlap_samples):
        raise ValueError('The selected window and overlap will not result in a COLA scenario')

    number_of_frames = int(np.floor((spr_object.time_abscissa.shape[0]-cola_overlap_samples)/(cola_frame_length - cola_overlap_samples)))+2
    frame_indices = ([0, cola_frame_length] + np.arange(number_of_frames)[:, np.newaxis]*(cola_frame_length-cola_overlap_samples)).astype(int)

    pre_data_pad_length = cola_frame_length-cola_overlap_samples # zero padding the initial data to ensure a cola over the full time trace
    post_data_pad_length = frame_indices.max()-(spr_object._training_response_array_.shape[0]+pre_data_pad_length) # zero padding the end of the signal to allow partial cola frames 
    padded_response = np.concatenate((np.zeros((pre_data_pad_length, spr_object.training_response_coordinate.shape[0]),dtype=float), 
                                      spr_object._training_response_array_,
                                      np.zeros((post_data_pad_length, spr_object.training_response_coordinate.shape[0]),dtype=float)), axis=0)

    zero_pad_signal_length = next_fast_len(cola_frame_length+2*model_order)
    left_samples = (zero_pad_signal_length-cola_frame_length)//2
    right_samples = cola_frame_length+left_samples

    signal_sizes = {'pre_data_blank_frame_length':pre_data_pad_length,
                    'post_data_blank_frame_length':post_data_pad_length,
                    'left_zero_pad_length':left_samples,
                    'right_zero_pad_length':right_samples,
                    'number_cola_frames':number_of_frames,
                    'cola_frame_length':cola_frame_length,
                    'cola_overlap_samples':cola_overlap_samples,
                    'zero_padded_signal_length':zero_pad_signal_length}
    
    return np.ascontiguousarray(padded_response), signal_sizes, frame_indices, window

def generate_zero_padded_response_fft(full_data, frame_indices, signal_sizes, 
                                      window, response_transform, use_transformation):
    """
    Yields an FFT for a zero padded segment of the provided data.

    Parameters
    ----------
    full_data : ndarray
        The data to segment and FFT. It should be shaped 
        [number of samples, number of channels].
    frame_indices : ndarray
        The indices to segment the data. It should be shaped
        [number of frames, number of samples per frame].
    signal_sizes : dict
        A dictionary with the signal sizes for the segmentation and zero 
        padding. The keys in the dictionary should include:
            - left_zero_pad_length - The amount of zeros that were used to pad the 
            left side of the COLA frames. 
            
            - right_zero_pad_length - The amount of zeros that were used to pad the 
            right side of the COLA frames. 
            
            - number_cola_frames - The number of frames to segment the data into for 
            the COLA processing.
            
            - zero_padded_signal_length - The frame length (in samples) of each zero 
            padded COLA frame. 
    window : ndarray
        The window (as a 1D array) to apply to the segmented data frames.
    response_transform : ndarray
        The response transformation to apply to the FFT of responses. It should be 
        shaped [number of frequency lines, number of response, number of responses].
    use_transformation : bool
        Whether or not to use the response transformation. 

    Yields
    ------
    padded_response_fft : ndarray
        The FFT of the zero padded segment of data.  
    """
    zero_padded_response = np.zeros((signal_sizes['zero_padded_signal_length'], full_data.shape[1]), dtype=float, order='C')
    for ii in range(signal_sizes['number_cola_frames']):
        # Using in-place multiplication in an attempt to save memory
        np.multiply(full_data[frame_indices[ii,0]:frame_indices[ii,-1], ...], window, out=zero_padded_response[signal_sizes['left_zero_pad_length']:signal_sizes['right_zero_pad_length'], :])
        
        if use_transformation:
            yield response_transform@rfft(zero_padded_response, axis=0, norm='backward')[..., np.newaxis]
        else:
            yield rfft(zero_padded_response, axis=0, norm='backward')[..., np.newaxis]

def generate_signal_from_cola_frames(signal_sizes, return_signal_length, cola_window, 
                                     number_of_dofs, reference_transform, use_transformation):
    """
    Reconstructs a full time signal from cola frame FFTs.

    Parameters
    ----------
    signal_sizes : dict
        The various signal sizes that were used in the original COLA segmentation 
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
    return_signal_length : int
        The desired number of samples in the returned signal. 
    cola_window : ndarray
        The window (as a 1d array) that was used in the original COLA segmentation.
    number_of_dofs : int
        The number of DOFs in the reconstructed signal.
    reference_transform : ndarray
        The reference transformation to apply to the FFT of responses. It should be 
        shaped [number of frequency lines, number of references, number of references].
    use_transformation : bool
        Whether or not to use the reference transformation.

    Yields
    -------
    reconstructed_signal : ndarray
        The reconstructed signal, it is sized [number of samples, number of DOFs]

    Notes
    -----
    Although it isn't directly passed as a parameter for the function, the user must 
    pass a "segment_fft" variable to the generator to recompile the signal in the 
    time domain. This variable should be shaped 
    [number of frequency lines, number of dofs, 1].
    """
    reconstruction_frame_indices = ([0, signal_sizes['zero_padded_signal_length']] + np.arange(signal_sizes['number_cola_frames'])[:, np.newaxis]*((signal_sizes['cola_frame_length']-signal_sizes['cola_overlap_samples']))).astype(int)
    reconstructed_signal = np.zeros((return_signal_length+signal_sizes['pre_data_blank_frame_length']+signal_sizes['post_data_blank_frame_length']+signal_sizes['right_zero_pad_length'], number_of_dofs), dtype=float)

    segment_fft = yield []

    for ii in range(signal_sizes['number_cola_frames']):
        if use_transformation:
            segment_force = irfft((reference_transform@segment_fft)[...,0], axis=0, n=signal_sizes['zero_padded_signal_length'], norm='backward')
        else:
            segment_force = irfft(segment_fft[...,0], axis=0, n=signal_sizes['zero_padded_signal_length'], norm='backward')
        reconstructed_signal[reconstruction_frame_indices[ii,0]:reconstruction_frame_indices[ii,-1],...] += segment_force
        
        if ii != signal_sizes['number_cola_frames']-1:
            # Continues through the loop as long as we haven't hit the end
            segment_fft = yield [] 
        else:
            # Yield the final reconstructed signal if all the segments have been compiled
            step = signal_sizes['cola_frame_length'] - signal_sizes['cola_overlap_samples']
            reconstructed_signal /= np.median(sum(cola_window[ii*step:(ii+1)*step] for ii in range(signal_sizes['cola_frame_length']//step)))

            yield reconstructed_signal[(signal_sizes['pre_data_blank_frame_length']+signal_sizes['left_zero_pad_length'])+np.arange(return_signal_length), :]

def frequency_interpolation(original_frequency, original_ordinate, padded_length, dt, interpolation_type='cubic'):
    """
    Interpolates a frequency domain function with the specified technique to account 
    for differing block sizes in frequency and time domain data. 

    Parameters
    ----------
    original_frequency : ndarray
        A 1D array for the frequency axis of the original data.
    original_ordinate : ndarray
        The ordinate for the original data, should be shaped 
        [number of frequency lines, ...]. 
    padded_length : int
        The number of samples in the time data that accompanies the frequency data.
    dt : float
        The sampling period for the time data that accompanies the frequency data.
    interpolation : str, optional
        The type of interpolation to use, this should match one of the available 
        types in scipy.interpolate.interp1d. The default is 'cubic'.

    Returns
    -------
    interpolated_ordinate : ndarray
        The ordinate for the interpolated data. It is shaped 
        [number of frequency lines, ...].

    Notes
    -----
    This is essentially a wrapper around scipy.interpolate.interp1d. It sets all 
    frequencies outside the band of interest to zero. The first (i.e. 0 Hz) and 
    last frequencies are set to zero, regardless. 
    """
    padded_frequency = rfftfreq(padded_length, dt)
    interpolator = interp1d(original_frequency, original_ordinate, kind=interpolation_type, axis=0, fill_value=0) 
    interpolated_ordinate = interpolator(padded_frequency)
    interpolated_ordinate[0,...] = 0
    interpolated_ordinate[-1,...] = 0
    return np.ascontiguousarray(interpolated_ordinate)

def sinc_interpolation(original_ordinate, padded_length):
    """
    Sinc interpolates a frequency domain function (via an FFT) to account for 
    differing block sizes in frequency and time domain data. 

    Parameters
    ----------
    original_ordinate : ndarray
        The ordinate for the original data, should be shaped 
        [number of frequency lines, ...]. 
    padded_length : int
        The number of samples in the time data that accompanies the frequency data.
    
    Returns
    -------
    interpolated_ordinate : ndarray
        The ordinate for the interpolated data. It is shaped 
        [number of frequency lines, ...].

    Notes
    -----
    This function does not account for any apparent non-causalities in the 
    original ordinate. Spline interpolation should be used for data with apparent
    non-causalities. 
    """
    if original_ordinate.shape[0] % 2 == 1:
        # assume an even length time function
        num_samples = 2*(original_ordinate.shape[0]-1)
    else:
        # assume an odd length time function
        num_samples = 2*(original_ordinate.shape[0]-1)+1
        
    zero_padded_ifft = np.zeros((padded_length, original_ordinate.shape[1], original_ordinate.shape[2]), dtype=float)
    zero_padded_ifft[:int((original_ordinate.shape[0]-1)*2), ...] = irfft(original_ordinate, axis=0, n=num_samples, norm='backward')

    interpolated_ordinate = rfft(zero_padded_ifft, axis=0, norm='backward')

    interpolated_ordinate[0,...] = 0
    if padded_length % 2 == 0:
        interpolated_ordinate[-1,...] = 0

    return np.ascontiguousarray(interpolated_ordinate)

def attenuate_signal(input_waveform: np.ndarray, 
                     limit: float | np.ndarray, 
                     full_scale: float = 1.0) -> np.ndarray:
    """
    Attenuate peaks that exceed limits in time domain by scaling the region between zero 
    crossings using the local maximum. This maintains a smooth waveform that does not 
    exceed the specified limits.

    Parameters
    ----------
    input_waveform : np.ndarray
        1D or 2D array with shape (n_signals, n_samples)
    limit : float | np.ndarray
        limit value or array of limit values with shape (n_signals,)
    full_scale : float, optional
        global scaling factor applied to limit value, intended to be used such
        that output waveform peaks are slightly less than physical limit,
        (ex. use full_scale=0.97 so that output signal will not exceed 97% of specified limit), 
        by default 1.0

    Returns
    -------
    np.ndarray
        clipped waveform, with shape (n_signals, n_samples)

    Notes
    -----
    Each region between zero crossings is scaled according to:
        `full_scale * limit / local_maximum`
    """
    if input_waveform.ndim == 1:
        input_waveform = input_waveform[np.newaxis, :]
    if isinstance(limit, int | float):
        limit = np.ones(input_waveform.shape[0]) * limit
    elif isinstance(limit, list | tuple):
        limit = np.array(limit)
    # pre-scale the limit value to the full scale percentage
    limit = np.abs(limit * full_scale)[:, np.newaxis]
    shape = input_waveform.shape
    # pre-scale the waveform by the limit (so that we can flatten the array with limits already applied)
    # waveform > limit equivalent to waveform / limit > 1
    scaled_waveform = (input_waveform / limit).flatten()
    samples = input_waveform.shape[-1]
    # find each index after the waveform crosses zero
    zero_crossings = np.diff(np.signbit(scaled_waveform)).nonzero()[0] + 1
    # construct list of slices representing each segment between zero crossings
    slices = [slice(zero_crossings[i], zero_crossings[i+1]) for i in range(len(zero_crossings)-1)]
    # initialize scaling array (scaling = limit for values that won't be clipped)
    scaling = np.tile(limit, samples).flatten()
    for sl in slices:
        if (sl.stop - sl.start) < 2:
            continue # skip empty slices
        if (sl.stop % samples) + (sl.start % samples) == samples:
            continue # edge case - skip slices that span between waveforms (since we flattened everything)
        segment = scaled_waveform[sl]
        abs_max = np.max(np.abs(segment))
        if abs_max > 1:
            # for segments that exceed the limit, scaling array set to limit * local_max
            scaling[sl] *= abs_max
    return scaled_waveform.reshape(shape) * limit**2 / scaling.reshape(shape)

#%% Deprecated Functions
def create_zero_padded_response_fft(spr_object, 
                                    cola_frame_length=None, 
                                    cola_window=('tukey', 0.5),
                                    cola_overlap_samples=None):
    """
    Creates the zero padded response FFT for a COLA source estimation problem.

    Parameters
    ----------
    spr_object : TransientSourcePathReceiver
        The transient SPR object that is being used for the source estimation
        problem. 
    cola_frame_length : int, optional
        The frame length (in samples) for the COLA processing. The default frame 
        length is approximately Fs/df from the FRFs.
    cola_window : string, float, or tuple
        The window to use in the COLA processing. This input is passed directly to 
        the SciPy get_window function. The default is a Tukey window with an alpha 
        of 0.5.
    cola_overlap_samples : int, optional
        The overlap (in samples) between the frames in the COLA processing, such that
        cola_overlap_samples/cola_frame_length results in the overlap percentage. A
        default is defined for the default Tukey window.

    Returns
    -------
    padded_response_fft : ndarray
        The zero padded response FFT. It is shaped 
        [number of COLA frames, number of frequency lines, number of responses].
    signal_sizes : dict
        The various signal sizes that were used in the zero padding operation. The
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
    window : ndarray
        The window that was used in the COLA processing. 

    Notes
    -----
    The zero_padded_signal_length is determined by an algorithm that uses the SciPy 
    next_fast_len function, which can return an odd length. As such, care should be 
    taken when reconstructing the signal with an inverse FFT.
    """
    model_order = 4*int(np.ceil((2*spr_object.abscissa.max()/spr_object.abscissa_spacing)/4)) # slightly strange operations to guarantee a model order that is divisible by 4
    if cola_frame_length is None:
        cola_frame_length = model_order

    window = get_window(cola_window, cola_frame_length)[..., np.newaxis]
    if cola_window[0] == 'tukey' and cola_overlap_samples is None:
        cola_overlap_samples = int(cola_frame_length*(cola_window[1]/2))
    if not check_COLA(window[...,0], model_order, cola_overlap_samples):
        raise ValueError('The selected window and overlap will not result in a COLA scenario')

    number_of_frames = int(np.floor((spr_object.time_abscissa.shape[0]-cola_overlap_samples)/(cola_frame_length - cola_overlap_samples)))+2
    frame_indices = (np.arange(cola_frame_length) + np.arange(number_of_frames)[:, np.newaxis]*(cola_frame_length-cola_overlap_samples)).astype(int)

    pre_data_pad_length = cola_frame_length-cola_overlap_samples # zero padding the initial data to ensure a cola over the full time trace
    post_data_pad_length = frame_indices.max()+1-(spr_object._training_response_array_.shape[0]+pre_data_pad_length) # zero padding the end of the signal to allow partial cola frames 
    response = np.concatenate((np.zeros((pre_data_pad_length, spr_object.training_response_coordinate.shape[0]),dtype=float), 
                               spr_object._training_response_array_,
                               np.zeros((post_data_pad_length, spr_object.training_response_coordinate.shape[0]),dtype=float)), axis=0)

    zero_pad_signal_length = next_fast_len(cola_frame_length+2*model_order)
    left_samples = (zero_pad_signal_length-cola_frame_length)//2
    right_samples = cola_frame_length+left_samples

    zero_padded_response = np.zeros((zero_pad_signal_length, response.shape[1]), dtype=float)
    padded_response_fft = np.zeros((number_of_frames, int(zero_pad_signal_length/2+1), response.shape[1]), dtype=complex)
    for ii in range(number_of_frames):
        zero_padded_response[left_samples:right_samples, :] = response[frame_indices[ii,:], ...]*window
        padded_response_fft[ii, ...] = rfft(zero_padded_response, axis=0, norm='backward')

    signal_sizes = {'pre_data_blank_frame_length':pre_data_pad_length,
                    'post_data_blank_frame_length':post_data_pad_length,
                    'left_zero_pad_length':left_samples,
                    'right_zero_pad_length':right_samples,
                    'cola_frame_length':cola_frame_length,
                    'cola_overlap_samples':cola_overlap_samples,
                    'zero_padded_signal_length':zero_pad_signal_length}
    
    return padded_response_fft, signal_sizes, window

def reconstruct_cola_frames(cola_ffts,
                            signal_sizes, 
                            return_signal_length,
                            cola_window):
    """
    Reconstructs a full time signal from cola frame FFTs.

    Parameters
    ----------
    cola_ffts : ndarray
        The cola frame FFTs to reconstruct the signal from. It should be sized
        [number of frames, number of frequency lines, number of DOFs]
    signal_sizes : dict
        The various signal sizes that were used in the original COLA segmentation 
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
    return_signal_length : int
        The desired number of samples in the returned signal. 
    cola_window : ndarray
        The window (as a 1d array) that was used in the original COLA segmentation.

    Returns
    -------
    reconstructed_signal : ndarray
        The reconstructed signal, it is sized [number of samples, number of DOFs]
    """
    reconstruction_frame_indices = ([0, signal_sizes['zero_padded_signal_length']] + np.arange(cola_ffts.shape[0])[:, np.newaxis]*((signal_sizes['cola_frame_length']-signal_sizes['cola_overlap_samples']))).astype(int)
    reconstructed_signal = np.zeros((return_signal_length+signal_sizes['pre_data_blank_frame_length']+signal_sizes['post_data_blank_frame_length']+signal_sizes['right_zero_pad_length'], cola_ffts.shape[-1]), dtype=float)

    for ii in range(reconstruction_frame_indices.shape[0]):
        reconstructed_signal[reconstruction_frame_indices[ii,0]:reconstruction_frame_indices[ii,-1],...] += irfft(cola_ffts[ii,...], axis=0, n=signal_sizes['zero_padded_signal_length'], norm='backward')

    step = signal_sizes['cola_frame_length'] - signal_sizes['cola_overlap_samples']
    reconstructed_signal /= np.median(sum(cola_window[ii*step:(ii+1)*step] for ii in range(signal_sizes['cola_frame_length']//step)))

    return reconstructed_signal[(signal_sizes['pre_data_blank_frame_length']+signal_sizes['left_zero_pad_length'])+np.arange(return_signal_length), :]