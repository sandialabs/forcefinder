"""
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
"""
from sdynpy.core.sdynpy_data import data_array
from sdynpy.core.sdynpy_data import FunctionTypes
from sdynpy.core.sdynpy_coordinate import coordinate_array
from sdynpy.signal_processing.sdynpy_correlation import trac
import numpy as np
from scipy.linalg import norm

def preprocess_data_for_quality_metric(spr_object,
                                       channel_set='training',
                                       samples_per_frame=None,
                                       frame_length=None,
                                       overlap=None,
                                       overlap_samples=None,
                                       window='boxcar'):
    """
    Selects the correct datasets from an SPR object and splits the data into frames 
    for the transient response prediction quality metrics.

    Parameters
    ----------
    spr_object : TransientSourcePathReceiver
        The SPR object to pull the data from.
    channel_set : str, optional
        The channel set to make the response comparisons between.
        The available options are:
            - training (default) - This compares the responses for the 
            transformed training DOFs in the SPR object.

            - validation - This compares the responses for the validation
            response DOFs in the SPR object.

            - target - This compares the responses for all the target 
            response DOFs in the SPR object. 
    samples_per_frame : int, optional
        Number of samples in each measurement frame to compute the RMS 
        for. Either this argument or `frame_length` must be specified.  
        If both or neither are specified, a `ValueError` is raised. This 
        argument matches the behavior of the "split_into_frames" method.
    frame_length : float, optional
        Length of each measurement frame to compute the RMS for, in the 
        same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
        Either this argument or `samples_per_frame` must be specified. If
        both or neither are specified, a `ValueError` is raised. This 
        argument matches the behavior of the "split_into_frames" method.
    overlap : float, optional
        Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
        overlap a quarter of the frame) for the RMS calculation. Either 
        this argument or `overlap_samples` must be specified. If both are
        specified, a `ValueError` is raised.  If neither are specified, no
        overlap is used. This argument matches the behavior of the 
        "split_into_frames" method.
    overlap_samples : int, optional
        Number of samples in the measurement frame to overlap for the RMS
        calculation. Either this argument or `overlap_samples` must be 
        specified.  If both are specified, a `ValueError` is raised.  If 
        neither are specified, no overlap is used. This argument matches 
        the behavior of the "split_into_frames" method.
    window : str, optional
        The window to apply to the segmented data. This should a window that 
        is available from the `scipy.signal.get_window` function. The default
        is boxcar.  

    Returns
    -------
    truth : TimeHistoryArray
        The truth response, segmented into frames per the selected options.
    reconstructed : TimeHistoryArray
        The reconstructed response, segmented into frames per the selected options.
    
    Notes
    -----
    This returns the raw training response rather than the reconstructed training 
    response.
    """
    if channel_set == 'training':
        truth = spr_object.training_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                               frame_length=frame_length, 
                                                               overlap=overlap, 
                                                               overlap_samples=overlap_samples,
                                                               window=window,
                                                               allow_fractional_frames=True)
        reconstructed = spr_object.reconstructed_training_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                                     frame_length=frame_length, 
                                                                                     overlap=overlap, 
                                                                                     overlap_samples=overlap_samples,
                                                                                     window=window,
                                                                                     allow_fractional_frames=True)
    elif channel_set == 'validation':
        truth = spr_object.validation_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                            frame_length=frame_length, 
                                                            overlap=overlap, 
                                                            overlap_samples=overlap_samples,
                                                            window=window,
                                                            allow_fractional_frames=True)
        reconstructed = spr_object.reconstructed_validation_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                                    frame_length=frame_length, 
                                                                                    overlap=overlap, 
                                                                                    overlap_samples=overlap_samples,
                                                                                    window=window,
                                                                                    allow_fractional_frames=True)
    elif channel_set == 'target':
        truth = spr_object.target_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                frame_length=frame_length, 
                                                overlap=overlap, 
                                                overlap_samples=overlap_samples,
                                                window=window,
                                                allow_fractional_frames=True)
        reconstructed = spr_object.reconstructed_target_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                                frame_length=frame_length, 
                                                                                overlap=overlap, 
                                                                                overlap_samples=overlap_samples,
                                                                                window=window,
                                                                                allow_fractional_frames=True)
    else:
        raise ValueError('Selected channel set is not available')
    return truth, reconstructed

def compute_global_rms_error(truth_response, check_response):
    """
    Computes the global RMS error in dB of the reconstructed response, 
    per the procedure in MIL-STD 810H.

    Parameters
    ----------
    truth_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the truth response in the comparison. 
    check_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the check response in the comparison. 

    Returns
    -------
    TimeHistoryArray
        Returns a time history array of the global RMS error in dB. The response
        coordinate for this array is made up to have a value for the DataArray
        and does not correspond to anything.

    Raises
    ------
    ValueError 
        If the response coordinate (and coordinate ordering) are not the same
        for the truth and check responses. 
    ValueError
        If the ordinate for the truth and check responses are not the same 
        shape.
    ValueError
        If the abscissa for the truth and check responses are not the same.

    Notes
    -----
    This function assumes that both the truth and check responses have been 
    segmented using the `split_into_frames` method.
    
    References
    ----------
    .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.
    """
    if not np.all(truth_response[0,:].response_coordinate==check_response[0,:].response_coordinate):
        raise ValueError('The truth and check responses must hav the same response coordinate (and coordinate ordering)')
    if not np.all(truth_response.ordinate.shape==check_response.ordinate.shape):
        raise ValueError('The truth and check response must have the same shape to perform a comparison')
    if not np.allclose(truth_response.abscissa, check_response.abscissa):
        raise ValueError('The truth and check response must have the same abscissa to perform a comparison')
    
    truth_rms = truth_response.rms().transpose()
    weights = (truth_rms**2)/(norm(truth_rms, ord=2, axis=0)**2)
    
    check_rms = check_response.rms().transpose()

    rms_error = 20*np.log10(check_rms/truth_rms)
    rms_time = np.average(truth_response.abscissa[:, 0, :], axis=1)
    global_rms_error = np.sum(rms_error*weights, axis=0)

    return data_array(FunctionTypes.TIME_RESPONSE, rms_time, global_rms_error, coordinate_array(node=1, direction=1))

def compute_average_rms_error(truth_response, check_response):
    """
    Computes the average RMS error in dB of the check response compared
    to the truth response.

    Parameters
    ----------
    truth_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the truth response in the comparison. 
    check_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the check response in the comparison. 

    Returns
    -------
    TimeHistoryArray
        Returns a time history array of the average RMS error in dB. The response
        coordinate for this array is made up to have a value for the DataArray
        and does not correspond to anything.

    Raises
    ------
    ValueError 
        If the response coordinate (and coordinate ordering) are not the same
        for the truth and check responses.
    ValueError
        If the ordinate for the truth and check responses are not the same 
        shape.
    ValueError
        If the abscissa for the truth and check responses are not the same.

    Notes
    -----
    This function assumes that both the truth and check responses have been 
    segmented using the `split_into_frames` method.
    """
    if not np.all(truth_response[0,:].response_coordinate==check_response[0,:].response_coordinate):
        raise ValueError('The truth and check responses must hav the same response DOFs (and DOF ordering)')
    if not np.all(truth_response.ordinate.shape==check_response.ordinate.shape):
        raise ValueError('The truth and check response must have the same shape to perform a comparison')
    if not np.allclose(truth_response.abscissa, check_response.abscissa):
        raise ValueError('The truth and check response must have the same abscissa to perform a comparison')
    
    truth_rms = truth_response.rms().transpose()        
    check_rms = check_response.rms().transpose()

    rms_error = 20*np.log10(check_rms/truth_rms)
    rms_time = np.average(truth_response.abscissa[:, 0, :], axis=1)

    average_rms_error = 20*np.log10(np.average(10**(rms_error/20), axis=0))
    return data_array(FunctionTypes.TIME_RESPONSE, rms_time, average_rms_error, coordinate_array(node=1, direction=1))

def compute_time_varying_trac(truth_response, check_response):
    """
    Computes the time varying time response assurance criterion (TRAC) 
    between the check response and truth response for all the supplied 
    degrees of freedom.

    Parameters
    ----------
    truth_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the truth response in the comparison. 
    check_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the check response in the comparison. 

    Returns
    -------
    TimeHistoryArray
        Returns a time history array of the time varying TRAC for all the 
        response degrees of freedom.

    Raises
    ------
    ValueError 
        If the response coordinate (and coordinate ordering) are not the same
        for the truth and check responses.
    ValueError
        If the ordinate for the truth and check responses are not the same 
        shape.
    ValueError
        If the abscissa for the truth and check responses are not the same.

    Notes
    -----
    This function assumes that both the truth and check responses have been 
    segmented using the `split_into_frames` method.
    """
    if not np.all(truth_response[0,:].response_coordinate==check_response[0,:].response_coordinate):
        raise ValueError('The truth and check responses must hav the same response DOFs (and DOF ordering)')
    if not np.all(truth_response.ordinate.shape==check_response.ordinate.shape):
        raise ValueError('The truth and check response must have the same shape to perform a comparison')
    if not np.allclose(truth_response.abscissa, check_response.abscissa):
        raise ValueError('The truth and check response must have the same abscissa to perform a comparison')
    
    response_coordinate = truth_response[0,:].response_coordinate
    trac_time = np.average(truth_response.abscissa[:, 0, :], axis=1)
    trac_values = trac(truth_response.ordinate, check_response.ordinate)

    return data_array(FunctionTypes.TIME_RESPONSE, trac_time, trac_values.T, response_coordinate[..., np.newaxis])

def compute_time_varying_level_error(truth_response, check_response, level_type='rms'):
    """
    Computes the computes the time varying error for a statistical level (rms
    or max) between the check and truth responses.

    Parameters
    ----------
    truth_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the truth response in the comparison. 
    check_response : TimeHistoryArray
        The segmented TimeHistoryArray (from the split_into_frames method)
        for the check response in the comparison. 
    level_type : str, optional
        The type of level to be used in the comparison. The options are:
            - rms - The rms level error for each frame of data in the 
            responses. This is the default.
            - max - The error in the maximum level that is seem for each 
            frame of data in the responses.

    Returns
    -------
    TimeHistoryArray
        Returns a time history array of the time varying level error in dB for 
        all the response degrees of freedom.

    Raises
    ------
    ValueError 
        If the response coordinate (and coordinate ordering) are not the same
        for the truth and check responses.
    ValueError
        If the ordinate for the truth and check responses are not the same 
        shape.
    ValueError
        If the abscissa for the truth and check responses are not the same.

    Notes
    -----
    This function assumes that both the truth and check responses have been 
    segmented using the `split_into_frames` method.
    """
    if not np.all(truth_response[0,:].response_coordinate==check_response[0,:].response_coordinate):
        raise ValueError('The truth and check responses must hav the same response DOFs (and DOF ordering)')
    if not np.all(truth_response.ordinate.shape==check_response.ordinate.shape):
        raise ValueError('The truth and check response must have the same shape to perform a comparison')
    if not np.allclose(truth_response.abscissa, check_response.abscissa):
        raise ValueError('The truth and check response must have the same abscissa to perform a comparison')
    
    response_coordinate = truth_response[0,:].response_coordinate
    frame_times = np.average(truth_response.abscissa[:, 0, :], axis=1)

    if level_type == 'rms':
        level_error = 20*np.log10(check_response.rms()/truth_response.rms())
    elif level_type == 'max':
        truth_level = np.abs(truth_response.ordinate).max(axis=2)
        check_level = np.abs(check_response.ordinate).max(axis=2)
        level_error = 20*np.log10(check_level/truth_level)

    return data_array(FunctionTypes.TIME_RESPONSE, frame_times, level_error.T, response_coordinate[..., np.newaxis])

def compute_error_stft(spr_object,
                       channel_set='training',
                       samples_per_frame=None,
                       frame_length=None,
                       overlap=None,
                       overlap_samples=None,
                       window='hann',
                       normalize_by_rms=False):
    """
    Computes the dB error short time Fourier transform (STFT) for the given SPR object.

    Parameters
    ----------
    spr_object : TransientSourcePathReceiver
        The SPR object to pull the data from.
    channel_set : str, optional
        The channel set to make the response comparisons between.
        The available options are:
            - training (default) - This compares the responses for the 
            transformed training DOFs in the SPR object.

            - validation - This compares the responses for the validation
            response DOFs in the SPR object.
            
            - target - This compares the responses for all the target 
            response DOFs in the SPR object. 
    samples_per_frame : int, optional
        Number of samples in each measurement frame to compute the RMS 
        for. Either this argument or `frame_length` must be specified.  
        If both or neither are specified, a `ValueError` is raised. This 
        argument matches the behavior of the "split_into_frames" method.
    frame_length : float, optional
        Length of each measurement frame to compute the RMS for, in the 
        same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
        Either this argument or `samples_per_frame` must be specified. If
        both or neither are specified, a `ValueError` is raised. This 
        argument matches the behavior of the "split_into_frames" method.
    overlap : float, optional
        Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
        overlap a quarter of the frame) for the RMS calculation. Either 
        this argument or `overlap_samples` must be specified. If both are
        specified, a `ValueError` is raised.  If neither are specified, no
        overlap is used. This argument matches the behavior of the 
        "split_into_frames" method.
    overlap_samples : int, optional
        Number of samples in the measurement frame to overlap for the RMS
        calculation. Either this argument or `overlap_samples` must be 
        specified.  If both are specified, a `ValueError` is raised.  If 
        neither are specified, no overlap is used. This argument matches 
        the behavior of the "split_into_frames" method.
    window : str, optional
        The window to apply to the segmented data. This should a window that 
        is available from the `scipy.signal.get_window` function. The default
        is hann.
    normalize_by_rms : bool, option
        Whether or not to normalize the error magnitudes (at each time slice) 
        by the relative magnitude of the RMS level (for each DOF) of the truth
        data. The normalization can help show the relative significance of the 
        spectrum error based on the amplitude of the time response (the errors 
        are scaled down where the RMS level is small). The default is false. 

    Returns
    -------
    db_error_stft : dict
        A dictionary with the following keys:
            - time (ndarray) - The time vector for the STFT.
            - frequency (ndarray) - The frequency vector for the STFT. 
            - response_coordinate (coordinate_array) - The response 
              coordinate array for the STFT.
            - amplitude (ndarray) - The dB error STFT, it is organized 
              [response coordinate, frequency axis, time axis].

    Notes 
    -----
    The dB error is computed from the magnitude of the STFT. 
    """
    truth, reconstructed = preprocess_data_for_quality_metric(spr_object, 
                                                              channel_set=channel_set,
                                                              samples_per_frame=samples_per_frame,
                                                              frame_length=frame_length,
                                                              overlap=overlap,
                                                              overlap_samples=overlap_samples,
                                                              window=window)

    truth_stft = truth.fft(norm='forward')*2
    reconstructed_stft = reconstructed.fft(norm='forward')*2
    
    stft_times = np.mean(truth.abscissa[:, 0, :], axis=1)
    stft_freqs = truth_stft[0, 0].abscissa
    stft_response_coordinate = truth_stft[0,:].response_coordinate

    stft_db_error = np.moveaxis(20*np.log10(np.abs(reconstructed_stft.ordinate).astype(float)) - 20*np.log10(np.abs(truth_stft.ordinate).astype(float)),0,-1)
    if normalize_by_rms:
        stft_db_error *= np.moveaxis(truth.rms()/truth.rms().max(axis=0), 0,-1)[:,np.newaxis,:]
    
    return {'time':stft_times, 'frequency':stft_freqs, 'response_coordinate':stft_response_coordinate, 'amplitude':stft_db_error}

def compute_stft(data,
                 samples_per_frame=None,
                 frame_length=None,
                 overlap=None,
                 overlap_samples=None,
                 window='hann'):
    """
    Computes the short time Fourier transform (STFT) for the given
    TimeHistoryArray and packages the data into a dictionary that can be
    used with the SpectrogramGUI tool.

    Parameters
    ----------
    data : TimeHistoryArray
        The data to compute the STFT for.
    samples_per_frame : int, optional
        Number of samples in each measurement frame to compute the RMS 
        for. Either this argument or `frame_length` must be specified.  
        If both or neither are specified, a `ValueError` is raised. This 
        argument matches the behavior of the "split_into_frames" method.
    frame_length : float, optional
        Length of each measurement frame to compute the RMS for, in the 
        same units as the `time_abscissa` field (`samples_per_frame` = `frame_length`/`self.time_abscissa_spacing`).
        Either this argument or `samples_per_frame` must be specified. If
        both or neither are specified, a `ValueError` is raised. This 
        argument matches the behavior of the "split_into_frames" method.
    overlap : float, optional
        Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
        overlap a quarter of the frame) for the RMS calculation. Either 
        this argument or `overlap_samples` must be specified. If both are
        specified, a `ValueError` is raised.  If neither are specified, no
        overlap is used. This argument matches the behavior of the 
        "split_into_frames" method.
    overlap_samples : int, optional
        Number of samples in the measurement frame to overlap for the RMS
        calculation. Either this argument or `overlap_samples` must be 
        specified.  If both are specified, a `ValueError` is raised.  If 
        neither are specified, no overlap is used. This argument matches 
        the behavior of the "split_into_frames" method.
    window : str, optional
        The window to apply to the segmented data. This should a window that 
        is available from the `scipy.signal.get_window` function. The default
        is hann.  

    Returns
    -------
    stft : dict
        A dictionary with the following keys:
            - time (ndarray) - The time vector for the STFT.
            - frequency (ndarray) - The frequency vector for the STFT. 
            - response_coordinate (coordinate_array) - The response 
              coordinate array for the STFT.
            - amplitude (ndarray) - The STFT, it is organized 
              [response coordinate, frequency axis, time axis].

    Notes 
    -----
    The STFT is the linear spectrum magnitude of the Fourier transform. 
    """

    segmented_data = data.split_into_frames(samples_per_frame=samples_per_frame,
                                            frame_length=frame_length, 
                                            overlap=overlap, 
                                            overlap_samples=overlap_samples,
                                            window=window,
                                            allow_fractional_frames=True)
    
    data_stft = segmented_data.fft(norm='forward')*2
    
    stft_times = np.mean(segmented_data.abscissa[:, 0, :], axis=1)
    stft_freqs = data_stft[0, 0].abscissa
    stft_response_coordinate = data_stft[0,:].response_coordinate

    stft_amplitude = np.moveaxis(np.abs(data_stft.ordinate).astype(float),0,-1)
    return {'time':stft_times, 'frequency':stft_freqs, 'response_coordinate':stft_response_coordinate, 'amplitude':stft_amplitude}