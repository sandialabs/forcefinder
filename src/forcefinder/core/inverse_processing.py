"""
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
"""
from .utilities import (is_cpsd, apply_buzz_method)
from .transient_utilities import (frequency_interpolation, prepare_for_zero_padding, sinc_interpolation,
                                  generate_zero_padded_response_fft, generate_signal_from_cola_frames)
import numpy as np
import warnings
warnings.simplefilter('always', category=RuntimeWarning)

def linear_inverse_processing(method):
    """
    This is a decorator function that does the pre and post processing to 
    handle the response and reference transformations for the various inverse 
    methods in the LinearSourcePathReceiver class. 

    The inverse (class) method must return a force in an NDArray format, that is 
    shaped [number of lines, number of forces, 1].
    """
    def wrapper(self, **kwargs):
        use_transformation = kwargs.pop('use_transformation', True) #Sets the default to use the transformation
        if use_transformation:
            # The logic is to handle if the transformation arrays are three dimensional 
            # (unique array per frequency line) or two dimensional (applied broadband).
            if self._response_transformation_array_.ndim == 2:        
                response_transform = self._response_transformation_array_[np.newaxis, ...] 
            elif self._response_transformation_array_.ndim == 3:        
                response_transform = self._response_transformation_array_
            else:
                raise ValueError('The shape of the response transformation array is not compatible with the inverse method')
            
            # the reference transformation matrix is inverted because it is formatted to 
            # transform the forces from "physical" to transformed. 
            if self._reference_transformation_array_.ndim == 2:        
                reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis, ...])
            elif self._reference_transformation_array_.ndim == 3:        
                reference_transform = np.linalg.pinv(self._reference_transformation_array_)
            else:
                raise ValueError('The shape of the reference transformation array is not compatible with the inverse method')

            preprocessed_response = response_transform@self._training_response_array_[..., np.newaxis]
            preprocessed_frf = response_transform@self._training_frf_array_@reference_transform
        elif not use_transformation:
            preprocessed_response = self._training_response_array_[..., np.newaxis]
            preprocessed_frf = self._training_frf_array_
            reference_transform = np.eye(self.reference_coordinate.shape[0])

        # "method" is the class method that is doing the inverse
        force = method(self, response=preprocessed_response, frf=preprocessed_frf, **kwargs)

        if use_transformation:
            self._force_array_ = (reference_transform@force)[..., 0] 
        else:
            self._force_array_ = force[..., 0]
        return self
    return wrapper

def power_inverse_processing(method):
    """
    This is a decorator function that does the pre and post processing to 
    handle the response and reference transformations for the various inverse 
    methods in the PowerSourcePathReceiver class. 

    The inverse (class) method must return a force in an NDArray format, that is 
    shaped [number of lines, number of forces, number of forces].
    """
    def wrapper(self, **kwargs):
        use_transformation = kwargs.pop('use_transformation', True) #Sets the default to use the transformation
        use_buzz = kwargs.pop('use_buzz', False) #Sets the default to not use the buzz method
        if use_buzz:
            response_cpsd = apply_buzz_method(self)
        elif not use_buzz:
            if not is_cpsd(self._training_response_array_):
                raise AttributeError('The buzz method must be used when the training responses are a vector of PSDs')
            response_cpsd = self._training_response_array_
        if use_transformation:
            # The logic is to handle if the transformation arrays are three dimensional 
            # (unique array per frequency line) or two dimensional (applied broadband).
            if self._response_transformation_array_.ndim == 2:        
                response_transform = self._response_transformation_array_[np.newaxis, ...] 
            elif self._response_transformation_array_.ndim == 3:        
                response_transform = self._response_transformation_array_
            else:
                raise ValueError('The shape of the response transformation array is not compatible with the inverse method')
            
            # the reference transformation matrix is inverted because it is formatted to 
            # transform the forces from "physical" to transformed. 
            if self._reference_transformation_array_.ndim == 2:        
                reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis, ...])
            elif self._reference_transformation_array_.ndim == 3:        
                reference_transform = np.linalg.pinv(self._reference_transformation_array_)
            else:
                raise ValueError('The shape of the reference transformation array is not compatible with the inverse method')

            preprocessed_response = response_transform@response_cpsd@np.transpose(response_transform.conj(), (0, 2, 1))
            preprocessed_frf = response_transform@self._training_frf_array_@reference_transform
        elif not use_transformation:
            preprocessed_response = response_cpsd
            preprocessed_frf = self._training_frf_array_
            reference_transform = np.eye(self._training_frf_array_.shape[-1])[np.newaxis, ...]

        # "method" is the class method that is doing the inverse
        force = method(self, response=preprocessed_response, frf=preprocessed_frf, **kwargs)
        
        if use_transformation:
            self._force_array_ = reference_transform@force@np.transpose(reference_transform.conj(), (0, 2, 1))
        else:
            self._force_array_ = force
        return self
    return wrapper

def transient_inverse_processing(method):
    """
    This is a decorator function that does the pre and post processing to 
    handle the cola segmentation, response transformations, and reference 
    transformations for the various inverse methods in the 
    TransientSourcePathReceiver class. 

    The inverse (class) method must return a force in an NDArray format, that is 
    shaped [number of lines, number of forces, 1].
    """
    def wrapper(self, **kwargs):
        cola_frame_length = kwargs.pop('cola_frame_length', None)
        cola_window = kwargs.pop('cola_window', ('tukey', 0.5))
        cola_overlap_samples = kwargs.pop('cola_overlap_samples', None)
        use_transformation = kwargs.pop('use_transformation', False)
        frf_interpolation_type = kwargs.pop('frf_interpolation_type', 'sinc')
        transformation_interpolation_type = kwargs.pop('transformation_interpolation_type', 'cubic')

        # Padded response here just pads the beginning and end of the full signal
        # to allow for full COLA frames at the beginning and end of the signal, where 
        # the signal starts at x% overlap in the first frame. 
        padded_response, signal_sizes, frame_indices, window = prepare_for_zero_padding(self,
                                                                                        cola_frame_length=cola_frame_length, 
                                                                                        cola_window=cola_window,
                                                                                        cola_overlap_samples=cola_overlap_samples)

        if frf_interpolation_type == 'sinc':
            interpolated_frfs = sinc_interpolation(self._training_frf_array_, signal_sizes['zero_padded_signal_length'])
        else:
            interpolated_frfs = frequency_interpolation(self.abscissa, self._training_frf_array_, signal_sizes['zero_padded_signal_length'], 
                                                        self.time_abscissa_spacing, interpolation_type=frf_interpolation_type)

        if use_transformation:
            if self._response_transformation_array_.ndim == 2:
                response_transform = self._response_transformation_array_[np.newaxis,...]
            if self._response_transformation_array_.ndim == 3:
                warnings.warn('Frequency dependent response transformations may result in non-causal behavior', category=RuntimeWarning)
                response_transform = frequency_interpolation(self.abscissa, self._response_transformation_array_, 
                                                             signal_sizes['zero_padded_signal_length'], self.time_abscissa_spacing,
                                                             interpolation_type=transformation_interpolation_type)
            if self._reference_transformation_array_.ndim == 2:
                reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis,...])
            if self._reference_transformation_array_.ndim == 3:
                warnings.warn('Frequency dependent reference transformations may result in non-causal behavior', category=RuntimeWarning)
                reference_transform = frequency_interpolation(self.abscissa, self._reference_transformation_array_, 
                                                              signal_sizes['zero_padded_signal_length'], self.time_abscissa_spacing,
                                                              interpolation_type=transformation_interpolation_type)
                reference_transform = np.linalg.pinv(reference_transform)
                
            preprocessed_frf = response_transform@interpolated_frfs@reference_transform
        else:
            preprocessed_frf = interpolated_frfs
            # making the transformation empty to be compatible with the generator
            response_transform = []
            reference_transform = [] 
        
        # Initializing the generator for reconstructing the segmented force
        reconstruction_generator = generate_signal_from_cola_frames(signal_sizes, self.time_abscissa.shape[0], window[...,0], 
                                                                    self.reference_coordinate.shape[0], reference_transform, use_transformation)
        _ = next(reconstruction_generator)

        # "method" is the class method that is doing the inverse
        self._force_array_ = method(self,  frf=preprocessed_frf, reconstruction_generator=reconstruction_generator, 
                                    response_generator=generate_zero_padded_response_fft(padded_response, frame_indices, signal_sizes, 
                                                                                         window, response_transform, use_transformation), **kwargs)
        
        return self
    return wrapper