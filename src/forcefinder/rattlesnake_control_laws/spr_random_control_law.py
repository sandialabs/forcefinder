"""
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
"""
import numpy as np
from scipy.interpolate import interp1d
from forcefinder import PowerSourcePathReceiver # Import from ForceFinder directly because of how the imports are handled from Rattlesnake

def trace(cpsd):
    return np.einsum ('ijj->i', cpsd)

def parse_extra_parameters(string):
    """
    Parses the extra parameters from the Rattlesnake text box.

    Parameters
    ----------
    string : str
        The string of extra parameters from the Rattlesnake text box.

    Returns
    -------
    extra_parameters : dict
        A dictionary with the parameters from the string parsed into the 
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
                - forces - This L-curve is constructed with the "size" of the 
                forces on the Y-axis and the regularization parameter on the X-axis. 
                - standard - This L-curve is constructed with the residual squared 
                error on the X-axis and the "size" of the forces on the Y-axis. 
        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
            The default depends on if the TSVD or Tikhonov regularization
            is being used and the options are:
                - curvature - This method searches for the regularization parameter 
                that results in maximum curvature of the L-curve. It is also referred 
                to as the L-curve criterion. 
                - distance - This method searches for the regularization parameter 
                that minimizes the distance between the L-curve and a "virtual origin". 
                A virtual origin is used, because the L-curve is scaled and offset to 
                always range from zero to one, in this case.
        match_trace : bool
            Whether or not to apply a match trace update during the control.
        use_buzz : bool
            Whether or not to use the buzz method for cross-term modification. 

    Raises
    ------
    ValueError
        If the df argument isn't supplied.
    ValueError
        If the ISE_technique argument isn't supplied.
    ValueError
        If the use_buzz argument isn't supplied.
    
    Notes
    -----
    The string is set-up by typing the parameters into the Rattlesnake text
    box. It should be set-up so that each return parameter is set equal to a 
    value with a line break between each value, for example:

        df = 0.1
        bp_freqs = 10, 500, 2000
        ISE_technique = manual_inverse
        inverse_method = threshold
        cond_num_threshold = 50, 1000, 300
        use_buzz = True
    """
    extra_parameters = {}

    for line in string.split('\n'):
        try:
            name, value = line.split('=')
            name = name.lower().strip()
            value = value.strip()
        except ValueError:
            continue #can only interpret lines with equal signs
        if name.lower() == 'df':
            extra_parameters['df'] = float(value)
        elif name.lower() == 'bp_freqs':
            extra_parameters['bp_freqs'] = np.array(value.split(','), dtype=float)
        elif name.lower() == 'ise_technique':
            extra_parameters['ISE_technique'] = value.lower()
        elif name.lower() == 'inverse_method':
            extra_parameters['method'] = value.lower()
        elif name.lower() == 'regularization_parameter':
            extra_parameters['regularization_parameter'] = np.array(value.split(','), dtype=float)
        elif name.lower() == 'cond_num_threshold':
            extra_parameters['cond_num_threshold'] = np.array(value.split(','), dtype=float)
        elif name.lower() == 'num_retained_values':
            extra_parameters['num_retained_values'] = np.array(value.split(','), dtype=float)
        elif name.lower() == 'number_regularization_values':
            extra_parameters['number_regularization_values'] = int(value)
        elif name.lower() == 'l_curve_type':
            extra_parameters['l_curve_type'] = value.lower()
        elif name.lower() == 'optimality_condition':
            extra_parameters['optimality_condition'] = value.lower()
        elif name.lower() == 'use_buzz':
            if value.lower() == 'true':
                extra_parameters['use_buzz'] = True
            elif value.lower() == 'false':
                extra_parameters['use_buzz'] = False
        elif name.lower() == 'match_trace':
            if value.lower() == 'true':
                extra_parameters['match_trace'] = True
            elif value.lower() == 'false':
                extra_parameters['match_trace'] = False
    if 'match_trace' not in extra_parameters:
        extra_parameters['match_trace'] = False
        

    if 'df' not in extra_parameters:
        raise ValueError('the df argument must be supplied')
    if 'ISE_technique' not in extra_parameters:
        raise ValueError('The ISE_technique argument must be supplied')
    if 'use_buzz' not in extra_parameters:
        raise ValueError('The use_buzz argument must be supplied')

    return extra_parameters

class RandomControlSourcePathReceiver(PowerSourcePathReceiver):
    """
    A subclass of the PowerSourcePathReceiver that is used in Rattlesnake random 
    vibration control.

    Notes
    -----
    This subclass technically shares all the attributes of the PowerSourcePathReceiver
    class. However, many of the private variables are left empty, because that data
    is not available in Rattlesnake (such as all the coordinate arrays). This means 
    that this subclass works on all the private variables and is set-up such that it
    only uses methods, which only use the super class private variables. 
    """

    def __init__(self, specification : np.ndarray, 
                 warning_levels : np.ndarray,
                 abort_levels : np.ndarray,
                 extra_control_parameters : str,
                 transfer_function : np.ndarray = None,
                 noise_response_cpsd : np.ndarray = None,
                 noise_reference_cpsd : np.ndarray = None,
                 sysid_response_cpsd : np.ndarray = None,
                 sysid_reference_cpsd : np.ndarray = None,
                 multiple_coherence : np.ndarray = None,
                 frames = None,
                 total_frames = None,
                 last_response_cpsd : np.ndarray = None, 
                 last_output_cpsd : np.ndarray = None):
        """
        Parameters
        ----------
        specification : ndarray
            The training response CPSD array, organized with frequency on the first axis.
        warning_levels : ndarray
            The warning levels provided with the specification. This is not used in the 
            SourcePathReceiver control law.
        abort_levels : ndarray
            The warning levels provided with the specification. This is not used in the 
            SourcePathReceiver control law.
        extra_control_parameters : str
            The extra control parameters that are written in the text box in the 
            Rattlesnake GUI.
        transfer_function : ndarray, optional
            The the training FRF array, organized with frequency on the first axis.
        noise_response_cpsd : ndarray
            The response CPSD array that is measured during the noise floor check. This
            is not used in the SourcePathReceiver control law.
        noise_reference_cpsd : ndarray
            The reference CPSD array that is measured during the noise floor check. This
            is not used in the SourcePathReceiver control law.
        sysid_response_cpsd : ndarray, optional
            The response CPSD array that is measured during the system ID. This is used
            for the buzz cpsd array in the SourcePathReceiver class.
        sysid_reference_cpsd : ndarray, optional
            The reference CPSD array that is measured during the system ID. This is not
            used in the SourcePathReceiver control law.
        multiple_coherence : ndarray, optional
            The multiple coherence that is measured during the system ID.
        frames : int
            The number of measurement frames aquired so far during the test.
        total_frames : int
            The total number of measurement frames that will be used to comput the 
            averaged CPSD and FRF arrays.
        last_response_cpsd : ndarray, optional
            The last response cpsd array that was measured in the test, organized with 
            frequency on the first axis.
        last_output_cpsd : ndarray, optional
            The last output cpsd array that was estimated from the control, organized 
            with frequency on the first axis.

        Notes
        -----
        This initializes an "empty" PowerSourcePathReceiver object and writes to the 
        private variables, as necessary.
        """
        super().__init__(empty=True) 
        self._training_response_array_=specification
        self._warning_levels_=warning_levels
        self._abort_levels_=abort_levels
        self.inverse_settings = parse_extra_parameters(extra_control_parameters)
        self._training_frf_array_=transfer_function
        self._noise_response_cpsd_array_=noise_response_cpsd
        self._noise_reference_cpsd_array_=noise_reference_cpsd
        self._buzz_cpsd_array_=sysid_response_cpsd
        self._sysid_reference_cpsd_array_=sysid_reference_cpsd
        self._multiple_coherence_array_=multiple_coherence
        self._frames_=frames
        self._total_frames_=total_frames
        self._last_response_array_=last_response_cpsd
        self._last_output_array_=last_output_cpsd
        
        if transfer_function is not None:
            self.system_id_update(transfer_function, 
                                  noise_response_cpsd,
                                  noise_reference_cpsd,
                                  sysid_response_cpsd,
                                  sysid_reference_cpsd,
                                  multiple_coherence,
                                  frames,
                                  total_frames)
        
        
    def system_id_update(self, transfer_function : np.ndarray = None, 
                         noise_response_cpsd : np.ndarray = None,
                         noise_reference_cpsd : np.ndarray = None,
                         sysid_response_cpsd : np.ndarray = None,
                         sysid_reference_cpsd : np.ndarray = None,
                         multiple_coherence : np.ndarray = None,
                         frames = None,
                         total_frames = None):
        """
        Updates the system ID data throughout the test and performs the inverse source
        estimation as the system ID data is updated.  

        Parameters
        ----------
        transfer_function : ndarray, optional
            The the training FRF array, organized with frequency on the first axis.
        noise_response_cpsd : ndarray, optional
            The response CPSD array that is measured during the noise floor check. This
            is not used in the SourcePathReceiver control law.
        noise_reference_cpsd : ndarray, optional
            The reference CPSD array that is measured during the noise floor check. This
            is not used in the SourcePathReceiver control law.
        sysid_response_cpsd : ndarray, optional
            The response CPSD array that is measured during the system ID. This is used
            for the buzz cpsd array in the SourcePathReceiver class.
        sysid_reference_cpsd : ndarray, optional
            The reference CPSD array that is measured during the system ID. This is not
            used in the SourcePathReceiver control law.
        multiple_coherence : ndarray, optional
            The multiple coherence that is measured during the system ID.
        frames : int
            The number of measurement frames aquired so far during the test.
        total_frames : int
            The total number of measurement frames that will be used to comput the 
            averaged CPSD and FRF arrays.
        
        Notes
        -----
        The use_transformation parameter in the inverse method must be set to false 
        in the inverse source estimation since the transformations are done in 
        Rattlesnake and the transformation arrays are not available to the control 
        class. 
        """
        self._training_frf_array_=transfer_function
        self._noise_response_cpsd_array_=noise_response_cpsd
        self._noise_reference_cpsd_array_=noise_reference_cpsd
        self._buzz_cpsd_array_=sysid_response_cpsd
        self._sysid_reference_cpsd_array_=sysid_reference_cpsd
        self._multiple_coherence_array_=multiple_coherence
        self._frames_=frames
        self._total_frames_=total_frames
        
        # Need to create an inverse argument dictionary so I can remove keys that are
        # unnecessary for the ISE technique (unexpected kwargs cause errors)
        inverse_arguments = self.inverse_settings.copy()
        del inverse_arguments['df']
        del inverse_arguments['ISE_technique']
        del inverse_arguments['match_trace']
        
        if self.inverse_settings['ISE_technique'].lower() == 'auto_tikhonov_by_l_curve':
            self.auto_tikhonov_by_l_curve(use_transformation=False, update_header=False, **inverse_arguments)
        elif self.inverse_settings['ISE_technique'].lower() == 'auto_truncation_by_l_curve':
            self.auto_truncation_by_l_curve(use_transformation=False, update_header=False, **inverse_arguments)
        elif self.inverse_settings['ISE_technique'].lower() == 'manual_inverse':
            if 'bp_freqs' in inverse_arguments:
                # need to interpolate the regularization parameters if breakpoints are supplied
                if inverse_arguments['bp_freqs'].shape != self._training_response_array_.shape[0]:
                    abscissa = np.arange(self._training_response_array_.shape[0])*self.inverse_settings['df']
                    if 'regularization_parameter' in inverse_arguments:
                        inverse_arguments['regularization_parameter'] = interp1d(inverse_arguments['bp_freqs'], 
                                                                                 inverse_arguments['regularization_parameter'],
                                                                                 'linear',
                                                                                 bounds_error=False,
                                                                                 fill_value=(inverse_arguments['regularization_parameter'][0],inverse_arguments['regularization_parameter'][-1]),
                                                                                 assume_sorted=True)(abscissa)
                    if 'cond_num_threshold' in inverse_arguments:
                        inverse_arguments['cond_num_threshold'] = interp1d(inverse_arguments['bp_freqs'], 
                                                                            inverse_arguments['cond_num_threshold'],
                                                                            'linear',
                                                                            bounds_error=False,
                                                                            fill_value=(inverse_arguments['cond_num_threshold'][0],inverse_arguments['cond_num_threshold'][-1]),
                                                                            assume_sorted=True)(abscissa)
                    if 'num_retained_values' in inverse_arguments:
                        inverse_arguments['num_retained_values'] = interp1d(inverse_arguments['bp_freqs'], 
                                                                            inverse_arguments['num_retained_values'],
                                                                            'previous',
                                                                            bounds_error=False,
                                                                            fill_value=(inverse_arguments['num_retained_values'][0],inverse_arguments['num_retained_values'][-1]),
                                                                            assume_sorted=True)(abscissa)
                    del inverse_arguments['bp_freqs']
            self.manual_inverse(use_transformation=False, update_header=False, **inverse_arguments)
        else:
            raise ValueError('The specified ISE technique is not available.')
            
    
    def control(self,transfer_function : np.ndarray,
                multiple_coherence : np.ndarray,
                frames, 
                total_frames,
                last_response_cpsd : np.ndarray, 
                last_output_cpsd : np.ndarray): 
        """
        Supply the drive voltage signals to Rattlesnake for the control. 
        
        Parameters
        ----------
        transfer_function : ndarray, optional
            The the training FRF array, organized with frequency on the first axis.
        multiple_coherence : ndarray, optional
            The multiple coherence that is measured during the system ID.
        frames : int
            The number of measurement frames aquired so far during the test.
        total_frames : int
            The total number of measurement frames that will be used to comput the 
            averaged CPSD and FRF arrays.
        last_response_cpsd : ndarray, optional
            The last response cpsd array that was measured in the test, organized with 
            frequency on the first axis.
        last_output_cpsd : ndarray, optional
            The last output cpsd array that was estimated from the control, organized 
            with frequency on the first axis.

        Returns
        -------
        ndarray
            The updated force array from the class. 

        Notes
        -----
        The only algorithms in this method should be for "drive updates", such
        as the match trace method. 
        """
        self._training_frf_array_=transfer_function
        self._last_response_array_=last_response_cpsd
        self._last_output_array_=last_output_cpsd
        if not self.inverse_settings['match_trace']:
            return self._force_array_
        elif self.inverse_settings['match_trace']:
            if self._last_output_array_ is None:
                return self._force_array_
            else:
                trace_ratio = trace(self._training_response_array_) / trace(self._last_response_array_)
                trace_ratio[np.isnan(trace_ratio)] = 0
                return self._last_output_array_*trace_ratio[..., np.newaxis, np.newaxis]