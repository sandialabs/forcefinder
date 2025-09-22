"""
Defines the SourcePathReceiver which is used for MIMO vibration test 
simulation or transfer path analysis.

Copyright 2022 National Technology & Engineering Solutions of Sandia,
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

import sdynpy as sdpy
from sdynpy.signal_processing.sdynpy_frf_inverse import (frf_inverse,
                                                         pinv_by_truncation,
                                                         pinv_by_tikhonov)
from sdynpy.core.sdynpy_data import FunctionTypes
from sdynpy.core.sdynpy_coordinate import outer_product, coordinate_array
from sdynpy.signal_processing.sdynpy_cpsd import cpsd_coherence, cpsd_from_coh_phs
import numpy as np
from scipy.linalg import norm
from scipy.signal import sosfiltfilt
from scipy.signal.windows import get_window 
from scipy.fft import irfft
from auto_regularization import (tikhonov_full_path,
                                 tikhonov_full_path_single_frequency,
                                 l_curve_criterion,
                                 l_curve_selection,
                                 optimal_l_curve_by_distance,
                                 select_model_by_information_criterion)
from sparse_functions import elastic_net_full_path_all_frequencies_parallel
from copy import deepcopy

def check_frequency_abscissa(data, reference_abscissa):
    """
    Checks the abscissa of the data for building a source-path-receiver 
    model. It validates that the data has a common abscissa for all
    the degrees of freedom and that the abscissa for data and reference
    data match. 

    Parameters
    ----------
    data : NDDataArray
        The data to check the abscissa on.
    reference_abscissa : ndarray
        The reference abscissa to compare the data abscissa against.

    Raises
    ------
    ValueError
        If the abscissa from data doesn't match reference_data.
    """
    data.abscissa_spacing
    if not np.alltrue(data.flatten()[0].abscissa==reference_abscissa):
        raise ValueError('The abscissa for the data does not match')

def compare_sampling_rate(time_data, reference_sampling_rate):
    """
    Checks that the sampling rate of the supplied time_data matches the 
    reference_reference_sampling tate. The primary purpose of this is 
    to ensure that the response/force and FRFs have the same sampling 
    rate when constructing a transient source-path-receiver model.

    Parameters
    ----------
    time_data : TimeHistoryArray
        The data to check the sampling rate on.
    reference_sampling_rate : float
        The reference sampling rate to compare against

    Raises
    ------
    ValueError
        If the sampling rate in time_data doesn't match the reference   
    """
    fs_time_data = 1/time_data.abscissa_spacing
    if not np.isclose(fs_time_data, reference_sampling_rate):
        raise ValueError('The sampling rate for the data does not match')

def is_cpsd(data_array):
    """
    Function to check if the supplied data array contains PSDs or CPSDs.

    Parameters
    ----------
    data_array : PowerSpectralDensityArray  
        data array to check.

    Returns
    -------
    bool
        True if the data array contains CPSDs, False otherwise
    """
    if isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
        try: 
            data_array.reshape_to_matrix()
            return True
        except:
            return False
    elif data_array.ndim == 3:
        return True
    else:
        return False

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

            preprocessed_response = response_transform@self._target_response_array_[..., np.newaxis]
            preprocessed_frf = response_transform@self._target_frf_array_@reference_transform
        elif not use_transformation:
            preprocessed_response = self._target_response_array_[..., np.newaxis]
            preprocessed_frf = self._target_frf_array_
            reference_transform = np.eye(self.reference_coordinate.shape[0])

        # "method" is the class method that is doing the inverse
        force = method(self, response=preprocessed_response, frf=preprocessed_frf, **kwargs)
        self._force_array_ = (reference_transform@force)[..., 0] 
        return self
    return wrapper

def apply_buzz_method(self):
    """
    Applies the buzz method using the information in the SPR object.

    References
    ----------
    .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation, 
            Aerospace Department, University of Bristol, 2014
    """
    if self._buzz_cpsd_array_ is None:
        raise AttributeError('A buzz CPSD has not been supplied for the SPR object.')
    phase = np.angle(self._buzz_cpsd_array_)
    coherence = cpsd_coherence(self._buzz_cpsd_array_)
    if self._target_response_array_.ndim == 3:
        asds = np.diagonal(self._target_response_array_, axis1=1, axis2=2)
        new_cpsd = cpsd_from_coh_phs(asds, coherence, phase)
    elif self._target_response_array_.ndim == 2:
        new_cpsd = cpsd_from_coh_phs(self._target_response_array_, coherence, phase)
    return new_cpsd    

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
        use_buzz = kwargs.pop('use_buzz', False) #Sets the default to use the transformation
        if use_buzz:
            response_cpsd = apply_buzz_method(self)
        elif not use_buzz:
            if not is_cpsd(self._target_response_array_):
                raise AttributeError('The buzz method must be used when the target responses are a vector of PSDs')
            response_cpsd = self._target_response_array_
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
            preprocessed_frf = response_transform@self._target_frf_array_@reference_transform
        elif not use_transformation:
            preprocessed_response = response_cpsd
            preprocessed_frf = self._target_frf_array_
            reference_transform = np.eye(self._target_frf_array_.shape[-1])[np.newaxis, ...]

        # "method" is the class method that is doing the inverse
        force = method(self, response=preprocessed_response, frf=preprocessed_frf, **kwargs)
        self._force_array_ = reference_transform@force@np.transpose(reference_transform.conj(), (0, 2, 1))
        return self
    return wrapper

class SourcePathReceiver:
    """
    A class to represent a source-path-receiver (SPR) model of a system for MIMO 
    vibration testing or transfer path analysis. It is primarily intended to manage 
    all the book keeping for the complex problem set-up.

    This is the base SPR class that is further defined for specific data types in 
    subclasses.

    Attributes
    ----------
    frfs : TransferFunctionArray
        The "full" FRFs that define the path of the SPR object.
    response
        The measured responses that define the receiver of the SPR object. This is 
        the response at all the locations that are represented in the FRFs (the 
        target locations and cross validation locations).
    force
        The forces that define the source in the SPR model. The force degrees of freedom 
        (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
        set via an inverse method in the class, but the user can also set them manually. 
    target_response
        The measured responses that will be used to estimate the forces in the SPR object 
        (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
        response if a target response or target_response_coordinate is not supplied .
    transformed_target_response
        The target_response with the response_transformation applied.
    target_frfs : TransferFunctionArray
        A subset of the FRFs that have the same response DOFs as target response.
    transformed_target_frfs : TransferFunctionArray
        The target_frfs with the response_transformation and reference_transformation
        applied.
    reconstructed_response
        The computed responses from the FRFs and forces. This cannot be set by the user. 
    transformed_reconstructed_response
        The reconstructed response (at the target_response_coordinate) with the 
        response_transformation applied.
    response_coordinate : coordinate_array
        The response coordinates of the SPR object, based on the FRFs.
    target_response_coordinate : coordinate_array
        The target response coordinates of teh SPR object, based on the target responses
    reference_coordinate : coordinate_array
        The reference coordinates of the SPR object, based on the FRFs.
    response_transformation : Matrix
        The response transformation that is used in the inverse problem. The default is 
        identity. 
    transformed_response_coordinate : coordinate_array
        The coordinates that the response is transformed into through the response 
        transformation array.
    reference_transformation : Matrix
        The reference transformation that is used in the inverse problem. The default is 
        identity.
    transformed_reference_coordinate : coordinate_array
        The coordinates that the reference is transformed into through the reference 
        transformation array.
    abscissa : float
        The frequency or time vector of the SPR model.
    abscissa_spacing : float
        The abscissa spacing (frequency resolution or sampling time) for the SPR model.
    inverse_settings : dictionary
        The settings that were to estimate the sources in the SourcePathReceiver object.

    Notes
    -----
    The ordinate in the full FRFs and responses can be different for the ordinate in the
    target FRFs and responses (depending on the problem set-up).
    """

    def __init__(self, frfs=None, response=None, force=None, target_response=None, target_response_coordinate=None, 
                 target_frfs=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Basic set-up for all the SourcePathReceiver (SPR) classes. The FRFs are set-up here, 
        but everything else is set-up in the subclasses. 

        Parameters
        ----------
        frfs : TransferFunctionArray
            The "full" FRFs that define the path of the SPR object.
        response
            The measured responses that define the receiver of the SPR object. This is 
            the response at all the locations that are represented in the FRFs (the 
            target locations and cross validation locations).
        force : optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        target_response : optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a target response or target_response_coordinate is not supplied .
        target_response_coordinate : coordinate_array, optional
            The target response coordinates of teh SPR object, based on the target responses.
        response_transformation : Matrix, optional
            The response transformation that is used in the inverse problem. The default is 
            identity. 
        reference_transformation : Matrix, optional
            The reference transformation that is used in the inverse problem. The default is
            identity.
        empty : bool, optional
            Whether or not to create an "empty" SPR object where all the attributes are None.
            The default is False (to create a "full" SPR object). 

        Notes
        -----
        Although most of the initialization is performed here, many of the properties are 
        passed to the subclass for assignment.         
        """
        # These are the parameters that could or should be set in an SPR object
        self._response_array_=None
        self._frf_array_=None
        self._force_array_=None
        self._target_response_array_=None
        self._target_frf_array_=None
        self._response_transformation_array_=None
        self._reference_transformation_array_=None
        self._target_response_coordinate_=None
        self._response_coordinate_=None
        self._reference_coordinate_=None
        self._transformed_response_coordinate_=None
        self._transformed_reference_coordinate_=None
        self._abscissa_=None
        self.inverse_settings = {}

        if not empty:
            if frfs is None:
                raise AttributeError('FRF data is required to initialize a SourcePathReceiver object')
            self.frfs = frfs
            
            if response is None and target_response is None:
                raise AttributeError('Response data is required to initialize a SourcePathReceiver object')
            if response is not None:
                self.response = response
            
            # Adding the target response coordinate, target responses and forces, if they are supplied
            # the ordering here is important for the logic to flow correctly for setting the target responses
            if target_response_coordinate is not None and target_response is None:
                self._target_response_coordinate_ = np.unique(target_response_coordinate)
                # the target response setter handles the indexing to the target response coordinate
                self.target_response = response
            elif target_response is not None:
                self.target_response = target_response
            else:
                self.target_response_coordinate = self.response_coordinate
                self._target_response_array_ = deepcopy(self._response_array_)
            
            if target_frfs is not None:
                self.target_frfs = target_frfs
            else:
                self.target_frfs = self.frfs[outer_product(self.target_response_coordinate, self.reference_coordinate)]
            
            if force is not None:
                self.force = force
            
            if response_transformation is None:
                self._response_transformation_array_ = np.eye(self._target_response_coordinate_.shape[0])
                self._transformed_response_coordinate_ = deepcopy(self._target_response_coordinate_)
            else:
                self.response_transformation = response_transformation
            
            if reference_transformation is None:
                self._reference_transformation_array_ = np.eye(self.reference_coordinate.shape[0])
                self._transformed_reference_coordinate_ = deepcopy(self._reference_coordinate_)
            else:
                self.reference_transformation = reference_transformation        

    def __repr__(self) -> str:
        """
        Generates a string representation of the SPR object.
        """
        return repr('{:} object with {:} reference coordinates, {:} response coordinates, and {:} target response coordinates'.format(self.__class__.__name__, 
                                                                                                                                      str(self.reference_coordinate.size), 
                                                                                                                                      str(self.response_coordinate.size), 
                                                                                                                                      str(self.target_response_coordinate.size)))

    def save(self, filename):
        """
        Saves the SourcePathReceiver object to a .npz file.

        Parameters
        ----------
        filename : str
            The file path and name for the .npz file
        
        Notes
        -----
        The private properties of the class are saved as arguments in the .npz file, where 
        the argument names match the private variable name. 
        """
        np.savez(filename, 
                 response=self._response_array_,
                 frf=self._frf_array_,
                 force=self._force_array_,
                 target_response=self._target_response_array_,
                 target_frf=self._target_frf_array_,
                 response_transformation=self._response_transformation_array_,
                 reference_transformation=self._reference_transformation_array_,
                 target_response_coordinate=self._target_response_coordinate_.string_array(),
                 response_coordinate=self._response_coordinate_.string_array(),
                 reference_coordinate=self._reference_coordinate_.string_array(),
                 transformed_response_coordinate=self._transformed_response_coordinate_.string_array(),
                 transformed_reference_coordinate=self._transformed_reference_coordinate_.string_array(),
                 abscissa=self._abscissa_) 
    
    @classmethod
    def load(cls, filename):
        """
        Loads the SourcePathReceiver object from an .npz file.

        Parameters
        ----------
        filename : str
            The file path and name for the .npz file
        
        Notes
        -----
        The private properties of the class should have been saved as arguments in the .npz 
        file, where the argument names match the private variable name. 
        """
        loaded_spr = np.load(filename, allow_pickle=True)
        spr_object = cls.__new__(cls)
        spr_object._response_array_ = loaded_spr['response']
        spr_object._frf_array_=loaded_spr['frf']
        spr_object._force_array_ = loaded_spr['force'] if np.all(loaded_spr['force'] != np.array(None)) else None
        spr_object._target_response_array_ = loaded_spr['target_response']
        spr_object._target_frf_array_ = loaded_spr['target_frf']
        spr_object._response_transformation_array_ = loaded_spr['response_transformation']
        spr_object._reference_transformation_array_ = loaded_spr['reference_transformation']
        spr_object._target_response_coordinate_ = coordinate_array(string_array=loaded_spr['target_response_coordinate'])
        spr_object._response_coordinate_ = coordinate_array(string_array=loaded_spr['response_coordinate'])
        spr_object._reference_coordinate_ = coordinate_array(string_array=loaded_spr['reference_coordinate'])
        spr_object._transformed_response_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_response_coordinate'])
        spr_object._transformed_reference_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_reference_coordinate'])
        spr_object._abscissa_ = loaded_spr['abscissa']
        return spr_object

    @property
    def frfs(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(self._frf_array_, 0, -1), sdpy.coordinate.outer_product(self._response_coordinate_, self._reference_coordinate_))
    
    @frfs.setter
    def frfs(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.TransferFunctionArray):
            raise TypeError('The FRFs must be a SDynPy TransferFunctionArray')
        data_array = data_array.reshape_to_matrix()
        # All the logic is here to ensure that only the ordinate of the FRFs can
        # change after the SourcePathReceiver object is defined.
        if self.response_coordinate is not None and not np.all(data_array[:, 0].response_coordinate==self.response_coordinate):
            raise ValueError('The FRFs in a SourcePathReceiver object cannot be reset with different response coordinates') 
        elif self.response_coordinate is None:
            self.response_coordinate = data_array[:, 0].response_coordinate 
        if self.reference_coordinate is not None and not np.all(data_array[0, :].reference_coordinate==self.reference_coordinate):
            raise ValueError('The FRFs in a SourcePathReceiver object cannot be reset with different reference coordinates')
        elif self.reference_coordinate is None:
            self.reference_coordinate = data_array[0, :].reference_coordinate
        if self._abscissa_ is not None and not np.all(data_array[0,0].abscissa==self.abscissa):
            raise ValueError('The FRFs in a SourcePathReceiver object cannot be reset with a different abscissa')
        elif self._abscissa_ is None:
            self._abscissa_ = data_array[0, 0].abscissa
        self._frf_array_ = np.moveaxis(data_array.ordinate, -1, 0)
    
    @property
    def target_frfs(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(self._target_frf_array_, 0, -1), sdpy.coordinate.outer_product(self._target_response_coordinate_, self._reference_coordinate_))

    @target_frfs.setter
    def target_frfs(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.TransferFunctionArray):
            raise TypeError('The target FRFs must be a SDynPy TransferFunctionArray')
        data_array = data_array.reshape_to_matrix()
        if not np.all(data_array[:, 0].response_coordinate==self._target_response_coordinate_):
            raise ValueError('The target FRF response DOFs do not match the target response DOFs of the SourcePathReceiver object')
        if not np.all(data_array[0, :].reference_coordinate==self._reference_coordinate_):
            raise ValueError('The target FRF reference DOFs do not match the SourcePathReceiver object')
        check_frequency_abscissa(data_array, self._abscissa_)
        self._target_frf_array_ = np.moveaxis(data_array.ordinate, -1, 0)

    @property
    def transformed_target_frfs(self):
        """
        the target FRFs with the transformations applied (i.e., what is used in the inverse problem).
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        if self._reference_transformation_array_.ndim == 2:        
            reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis, ...])
        elif self._reference_transformation_array_.ndim == 3:        
            reference_transform = np.linalg.pinv(self._reference_transformation_array_)
        transformed_frf = response_transform@self._target_frf_array_@reference_transform
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(transformed_frf, 0, -1), 
                               outer_product(self._transformed_response_coordinate_, self._transformed_reference_coordinate_))

    @property
    def response_coordinate(self):
        return self._response_coordinate_
    
    @response_coordinate.setter
    def response_coordinate(self, coordinate_array):
        if self._response_coordinate_ is not None:
            raise AttributeError('The response coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._response_coordinate_ = coordinate_array
    
    @property
    def reference_coordinate(self):
        return self._reference_coordinate_

    @reference_coordinate.setter
    def reference_coordinate(self, coordinate_array):
        if self._reference_coordinate_ is not None:
            raise AttributeError('The reference coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._reference_coordinate_ = coordinate_array

    @property
    def target_response_coordinate(self):
        return self._target_response_coordinate_
    
    @target_response_coordinate.setter
    def target_response_coordinate(self, target_response_coordinate):
        if not np.all(np.isin(target_response_coordinate, self.response_coordinate)):
            raise ValueError('Target response {:} is not in the SPR model'.format(target_response_coordinate[~np.isin(target_response_coordinate, self.response_coordinate)].string_array()))
        if self._target_response_coordinate_ is not None:
            raise AttributeError('The target response coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._target_response_coordinate_ = np.unique(target_response_coordinate)  

    @property
    def response_transformation(self):
        if self._response_transformation_array_ is None:
            raise AttributeError('A response transformation was not defined for this object')
        return sdpy.matrix(self._response_transformation_array_, self._transformed_response_coordinate_, self._target_response_coordinate_)
    
    @response_transformation.setter
    def response_transformation(self, transformation_matrix):
        if not isinstance(transformation_matrix, sdpy.Matrix):
            raise TypeError('The response transformation must be defined as a SDynPy Matrix')
        self._response_transformation_array_ = transformation_matrix[np.unique(transformation_matrix.row_coordinate), self._target_response_coordinate_]
        self._transformed_response_coordinate_ = np.unique(transformation_matrix.row_coordinate)

    @property
    def transformed_response_coordinate(self):
        return self._transformed_response_coordinate_

    @property
    def reference_transformation(self):
        if self._reference_transformation_array_ is None:
            raise AttributeError('A reference transformation was not defined for this object')
        return sdpy.matrix(self._reference_transformation_array_, self._transformed_reference_coordinate_, self._reference_coordinate_)
    
    @reference_transformation.setter
    def reference_transformation(self, transformation_matrix):
        if not isinstance(transformation_matrix, sdpy.Matrix):
            raise TypeError('The reference transformation must be defined as a SDynPy Matrix')
        self._reference_transformation_array_ = transformation_matrix[np.unique(transformation_matrix.row_coordinate), self._reference_coordinate_]
        self._transformed_reference_coordinate_ = np.unique(transformation_matrix.row_coordinate)

    @property
    def transformed_reference_coordinate(self):
        return self._transformed_reference_coordinate_
    
    @property
    def abscissa(self):
        return self._abscissa_

    @property
    def abscissa_spacing(self):
        return np.mean(np.diff(self._abscissa_))

    @property
    def response(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass
    @response.setter
    def response(self):
        pass
    @property
    def force(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass
    @force.setter
    def force(self):
        pass
    @property
    def target_response(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass
    @target_response.setter
    def target_response(self):
        pass
    @property
    def transformed_target_response(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass
    @property
    def reconstructed_response(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass
    @property
    def transformed_reconstructed_response(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass
    @property
    def transformed_force(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        pass

    def set_response_transformation_by_normalization(self, method = 'std'):
        """
        Sets the response transformation matrix such that it will "normalize"
        the responses based on a statistical quantity from the rows of the 
        FRF matrix.

        Parameters
        ----------
        method : str, optional
            A string defining the statistical quantity that will be used 
            for the response normalization. Currently, only the standard
            deviation is supported.
        
        Returns
        -------
        self : SourcePathReceiver 
            The SourcePathReceiver object with the response transformation that
            has been updated
        """
        if method == 'std':
            res_statistic = np.std(self._target_frf_array_, axis=-1)
        else:
            raise ValueError('The selected statistical quantity is not available for response normalization')
        num_target_coordinate = self._target_response_coordinate_.shape[0]
        response_transformation_array = np.broadcast_to(np.eye(num_target_coordinate)[np.newaxis, ...], (self.abscissa.shape[0], num_target_coordinate, num_target_coordinate))/res_statistic[..., np.newaxis]
        self.response_transformation = sdpy.matrix(response_transformation_array, self.target_response_coordinate, self.target_response_coordinate)
        return self
    
    def set_reference_transformation_by_normalization(self, method = 'std'):
        """
        Sets the reference transformation matrix such that it will "normalize"
        the references based on a statistical quantity from the columns of the 
        FRF matrix.

        Parameters
        ----------
        method : str, optional
            A string defining the statistical quantity that will be used 
            for the response normalization. Currently, only the standard
            deviation is supported.

        Returns
        -------
        self : SourcePathReceiver 
            The SourcePathReceiver object with the reference transformation that
            has been updated
        """
        if method == 'std':
            ref_statistic = np.std(self._target_frf_array_, axis=-2)
        else:
            raise ValueError('The selected statistical quantity is not available for response normalization')
        num_target_coordinate = self._reference_coordinate_.shape[0]
        reference_transformation_array = np.broadcast_to(np.eye(num_target_coordinate)[np.newaxis, ...], (self.abscissa.shape[0], num_target_coordinate, num_target_coordinate))/ref_statistic[..., np.newaxis]
        self.reference_transformation = sdpy.matrix(reference_transformation_array, self.reference_coordinate, self.reference_coordinate)
        return self
    
    def reset_reference_transformation(self):
        """
        Resets the reference transformation matrix to default (identity).
        """
        self.reference_transformation = sdpy.matrix(np.eye(self.reference_coordinate.shape[0]), self.reference_coordinate, self.reference_coordinate)
        return self
    
    def reset_response_transformation(self):
        """
        Resets the response transformation matrix to default (identity).
        """
        self.response_transformation = sdpy.matrix(np.eye(self.target_response_coordinate.shape[0]), self.target_response_coordinate, self.target_response_coordinate)
        return self
    
    def copy(self):
        """
        Returns a deepcopy of the SPR object.
        """
        return deepcopy(self)
    
    def extract_elements_by_abscissa(self, min_abscissa=None, max_abscissa=None, in_place=True):
        """
        Extracts elements from all the components of the SPR object with abscissa 
        values within the specified range.

        Parameters
        ----------
        min_abscissa : float
            Minimum abscissa value to keep.
        max_abscissa : float
            Maximum abscissa value to keep.
        in_place : bool
            Whether or not to modify the SPR object in place or create a new SPR object.
            The default is True (i.e., modify the SPR object in place).

        Returns
        ------- 
            SPR object that has been trimmed according to the abscissa limits
        
        Notes
        -----
        This method currently only works on linear or power SPR objects since transient SPR 
        objects have two abscissas and there is ambiguity on which one would be trimmed. 
        """
        if hasattr(self, '_time_abscissa_'):
            raise NotImplementedError('The extract_elements_by_abscissa method has not been implemented for transient SPR objects')
        
        work_object = self if in_place else self.copy()
        
        if min_abscissa is None and max_abscissa is None:
            raise ValueError('A bounding abscissa (min_abscissa, max_abscissa, or both) must be supplied')
        if min_abscissa is None:
            abscissa_indices = (work_object._abscissa_ <= max_abscissa)
        elif max_abscissa is None:
            abscissa_indices = (work_object._abscissa_ >= min_abscissa)
        else:
            abscissa_indices = (work_object._abscissa_ >= min_abscissa) & (work_object._abscissa_ <= max_abscissa)

        work_object._abscissa_ = work_object._abscissa_[abscissa_indices, ...]
        if work_object._response_array_ is not None:
            work_object._response_array_ = work_object._response_array_[abscissa_indices, ...]
        if work_object._frf_array_ is not None:
            work_object._frf_array_ = work_object._frf_array_[abscissa_indices, ...]
        if work_object._force_array_ is not None:
            work_object._force_array_ = work_object._force_array_[abscissa_indices, ...]
        if work_object._target_response_array_ is not None:
            work_object._target_response_array_ = work_object._target_response_array_[abscissa_indices, ...]
        if work_object._target_frf_array_ is not None:
            work_object._target_frf_array_ = work_object._target_frf_array_[abscissa_indices, ...]
        if work_object._response_transformation_array_.ndim == 3:
            work_object._response_transformation_array_ = work_object._response_transformation_array_[abscissa_indices, ...]
        if work_object._reference_transformation_array_.ndim == 3:
            work_object._reference_transformation_array_ = work_object._reference_transformation_array_[abscissa_indices, ...]
        if hasattr(work_object, '_buzz_cpsd_array_'):
            if work_object._buzz_cpsd_array_ is not None:
                work_object._buzz_cpsd_array_ = work_object._buzz_cpsd_array_[abscissa_indices, ...]
        
        if in_place:
            return self
        else:
            return work_object


class LinearSourcePathReceiver(SourcePathReceiver):
    """
    A subclass to represent a source-path-receiver (SPR) model of a system for MIMO 
    vibration testing or transfer path analysis. The responses and forces in this
    subclass are linear spectra (i.e., ffts).

    Attributes
    ----------
    frfs : TransferFunctionArray
        The "full" FRFs that define the path of the SPR object.
    response : SpectrumArray
        The measured responses that define the receiver of the SPR object. This is 
        the response at all the locations that are represented in the FRFs (the 
        target locations and cross validation locations).
    force : SpectrumArray
        The forces that define the source in the SPR model. The force degrees of freedom 
        (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
        set via an inverse method in the class, but the user can also set them manually. 
    target_response : SpectrumArray
        The measured responses that will be used to estimate the forces in the SPR object 
        (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
        response if a target response or target_response_coordinate is not supplied.
    transformed_target_response : SpectrumArray
        The target_response with the response_transformation applied.
    target_frfs : TransferFunctionArray
        A subset of the FRFs that have the same response DOFs as target response.
    transformed_target_frfs : TransferFunctionArray
        The target_frfs with the response_transformation and reference_transformation
        applied.
    reconstructed_response : SpectrumArray
        The computed responses from the FRFs and forces. This cannot be set by the user.
    transformed_reconstructed_response : SpectrumArray
        The reconstructed response (at the target_response_coordinate) with the 
        response_transformation applied. 
    response_coordinate : coordinate_array
        The response coordinates of the SPR object, based on the FRFs.
    target_response_coordinate : coordinate_array
        The target response coordinates of teh SPR object, based on the target responses.
    reference_coordinate : coordinate_array
        The reference coordinates of the SPR object, based on the FRFs.
    response_transformation : Matrix, optional
        The response transformation that is used in the inverse problem. The default is 
        identity. 
    transformed_response_coordinate : coordinate_array
        The coordinates that the response is transformed into through the response 
        transformation array.
    reference_transformation : Matrix, optional
        The reference transformation that is used in the inverse problem. The default is 
        identity.
    transformed_reference_coordinate : coordinate_array
        The coordinates that the reference is transformed into through the reference 
        transformation array.
    abscissa : float
        The frequency vector of the SPR model.
    abscissa_spacing : float
        The frequency resolution for the SPR model.
    inverse_settings : dict
        The settings that were to estimate the sources in the SourcePathReceiver object.

    Notes
    -----
    The ordinate in the full FRFs and responses can be different for the ordinate in the
    target FRFs and responses (depending on the problem set-up).

    The "linear" term in the class name stands for the linear units in the response and
    force spectra.
    """

    def __init__(self, frfs=None, response=None, force=None, target_response=None, target_response_coordinate=None, 
                 target_frfs=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Parameters
        ----------
        frfs : TransferFunctionArray
            The "full" FRFs that define the path of the SPR object.
        response : SpectrumArray
            The measured responses that define the receiver of the SPR object. This is 
            the response at all the locations that are represented in the FRFs (the 
            target locations and cross validation locations).
        force : SpectrumArray, optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        target_response : SpectrumArray, optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a target response or target_response_coordinate is not supplied .
        target_response_coordinate : coordinate_array, optional
            The target response coordinates of teh SPR object, based on the target responses.
        response_transformation : Matrix, optional
            The response transformation that is used in the inverse problem. The default is 
            identity. 
        reference_transformation : Matrix, optional
            The reference transformation that is used in the inverse problem. The default is 
            identity.
        empty : bool, optional
            Whether or not to create an "empty" SPR object where all the attributes are None.
            The default is False (to create a "full" SPR object).

        Notes
        -----
        Much of the initialization is inherited from the base SourcePathReceiver class.
        """
        # Inheriting the initial set-up from 
        super().__init__(frfs=frfs, response=response, force=force, target_response=target_response, target_response_coordinate=target_response_coordinate, 
                         target_frfs=target_frfs, response_transformation=response_transformation, reference_transformation=reference_transformation, empty=empty)

    @property
    def response(self):
        if self._response_array_ is None:
            raise AttributeError('A response array was not defined for this object')
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(self._response_array_, 0, -1), self._response_coordinate_[..., np.newaxis])
    
    @response.setter
    def response(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
            raise TypeError('The response must be a SDynPy SpectrumArray')
        check_frequency_abscissa(data_array, self.abscissa)
        self._response_array_ = np.moveaxis(data_array[self._response_coordinate_[..., np.newaxis]].ordinate, -1, 0)
            
    @property
    def force(self):
        if self._force_array_ is None:
            raise AttributeError('A force array is not defined for this object')
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(self._force_array_, 0, -1), self._reference_coordinate_[..., np.newaxis])
    
    @force.setter
    def force(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
            raise TypeError('The force must be a SDynPy SpectrumArray')
        if not np.all(np.isin(data_array.response_coordinate, self.reference_coordinate)):
            raise ValueError('Force {:} is not in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.reference_coordinate)].string_array()))
        check_frequency_abscissa(data_array, self._abscissa_)
        self._force_array_ = np.moveaxis(data_array[self.reference_coordinate[..., np.newaxis]].ordinate, -1, 0)

    @property
    def transformed_force(self):
        """
        The force with the transformation applied.
        """
        if self._reference_transformation_array_.ndim == 2:        
            reference_transform = self._reference_transformation_array_[np.newaxis, ...]
        elif self._reference_transformation_array_.ndim == 3:
            reference_transform = self._reference_transformation_array_
        else:
            raise ValueError('The shape of the reference transformation array is not compatible with the object')
        transformed_force_array = (reference_transform@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(transformed_force_array, 0, -1), self._transformed_reference_coordinate_[..., np.newaxis])

    @property
    def target_response(self):
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(self._target_response_array_, 0, -1), self._target_response_coordinate_[..., np.newaxis])
    
    @target_response.setter
    def target_response(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
            raise TypeError('The target response must be a SDynPy SpectrumArray')
        if not np.all(np.isin(data_array.response_coordinate, self.response_coordinate)):
            raise ValueError('Target response {:} is not in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.response_coordinate)].string_array()))
        if self._target_response_array_ is not None and not np.all(data_array.response_coordinate==self._target_response_coordinate_):
            raise ValueError('The target response coordinates do not match the SourcePathReceiver object')
        check_frequency_abscissa(data_array, self._abscissa_)
        # The numpy unique is used when setting the coordinate to make sure that the DOF ordering
        # in the target_response_array matches the other data.
        if self._target_response_coordinate_ is None:
            self.target_response_coordinate = np.unique(data_array.response_coordinate)
        self._target_response_array_ = np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0) 

    @property
    def transformed_target_response(self):
        """
        The target response with the transformation applied (i.e., what is used in the inverse problem).
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        transformed_response = (response_transform@self._target_response_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(transformed_response, 0, -1), self._transformed_response_coordinate_[..., np.newaxis])


    @property
    def reconstructed_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        reconstructed_response = (self._frf_array_@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(reconstructed_response, 0, -1), self._response_coordinate_[..., np.newaxis])
    
    @property
    def transformed_reconstructed_response(self):
        """
        The reconstructed response (at the target coordinates) with the transformation applied (i.e., what
        was used in the inverse problem).
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        reconstructed_response = (response_transform@self._target_frf_array_@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(reconstructed_response, 0, -1), self._transformed_response_coordinate_[..., np.newaxis])

    def global_asd_error(self, 
                         channel_set='target'):
        """
        Computes the global ASD error in dB of the reconstructed response, 
        per the procedure in MIL-STD 810H.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - validation - This compares the responses at the DOFs that
                are not included in the untransformed target DOFs. 
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 

        Returns
        -------
        SpectrumArray
            Returns a spectrum array of the global ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.
        
        Notes
        -----
        Computes the ASD from the spectrum response by squaring the absolute value
        of the spectrum and dividing by the SPR object abscissa spacing.  

        References
        ----------
        .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.
        """
        if channel_set == 'target':
            truth = (np.abs(self.transformed_target_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'validation':
            validation_coordinate = self._response_coordinate_[~np.isin(self._response_coordinate_, self._target_response_coordinate_)]
            truth = (np.abs(self.response[validation_coordinate[..., np.newaxis]].ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_response[validation_coordinate[..., np.newaxis]].ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'full':
            truth = (np.abs(self.response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_response.ordinate)**2)/self.abscissa_spacing
        
        weights = truth/norm(truth, axis=0, ord=2)
        asd_error = 10*np.log10(reconstructed/truth)
        global_asd_error = np.sum(asd_error*weights, axis=0)
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, global_asd_error, coordinate_array(node=1, direction=1))
    
    def average_asd_error(self, 
                          channel_set='target'):
        """
        Computes the average ASD error in dB of the reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - validation - This compares the responses at the DOFs that
                are not included in the untransformed target DOFs. 
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 

        Returns
        -------
        PowerSpectralDensityArray
            Returns a spectrum array of the average ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.

        Notes
        -----
        Computes the ASD from the spectrum response by squaring the absolute value
        of the spectrum and dividing by the SPR object abscissa spacing.       
        """
        if channel_set == 'target':
            truth = (np.abs(self.transformed_target_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'validation':
            validation_coordinate = self._response_coordinate_[~np.isin(self._response_coordinate_, self._target_response_coordinate_)]
            truth = (np.abs(self.response[validation_coordinate[..., np.newaxis]].ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_response[validation_coordinate[..., np.newaxis]].ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'full':
            truth = (np.abs(self.response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_response.ordinate)**2)/self.abscissa_spacing
        
        asd_error = 10*np.log10(reconstructed/truth)
        average_asd_error = 10*np.log10(np.average(10**(asd_error/10), axis=0))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, average_asd_error, coordinate_array(node=1, direction=1))

    def error_summary(self,
                      channel_set='target',
                      figure_kwargs={},
                      linewidth=1,
                      plot_kwargs={}):
        """
        Plots the error summary using the method of the same name from the 
        PowerSpectralDensityArray class.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - validation - This compares the responses at the DOFs that
                are not included in the untransformed target DOFs. 
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 
        figure_kwargs : dict, optional
            Arguments to use when creating the figure. The default is {}.
        linewidth : float, optional 
            Widths of the lines on the plot. The default is 1.
        plot_kwargs : dict, optional
            Arguments to use when plotting the lines. The default is {}.

        Returns
        -------
        Error Metrics
            A tuple of dictionaries of error metrics

        Notes
        -----
        This is a simple wrapper around the "error_summary" method, where the 
        PowerSpectralDensityArray is computed from the spectrum response by 
        squaring the absolute value of the spectrum and dividing by the SPR 
        object abscissa spacing.
        """
        if channel_set == 'target':
            truth = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.transformed_target_response.ordinate)**2)/self.abscissa_spacing, self._transformed_response_coordinate_[..., np.newaxis])

            reconstructed = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing, self._transformed_response_coordinate_[..., np.newaxis])

            return_data = truth.error_summary(figure_kwargs=figure_kwargs, 
                                              linewidth=linewidth, 
                                              plot_kwargs=plot_kwargs,
                                              cpsd_matrices=reconstructed)
        elif channel_set == 'validation':
            validation_coordinate = self._response_coordinate_[~np.isin(self._response_coordinate_, self._target_response_coordinate_)]
            
            truth = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.response[validation_coordinate[..., np.newaxis]].ordinate)**2)/self.abscissa_spacing, validation_coordinate[..., np.newaxis])

            reconstructed = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.reconstructed_response[validation_coordinate[..., np.newaxis]].ordinate)**2)/self.abscissa_spacing, validation_coordinate[..., np.newaxis])
            
            return_data = truth.error_summary(figure_kwargs=figure_kwargs, 
                                              linewidth=linewidth, 
                                              plot_kwargs=plot_kwargs,
                                              cpsd_matrices=reconstructed)
        elif channel_set == 'full':
            truth = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.response.ordinate)**2)/self.abscissa_spacing, self._response_coordinate_[..., np.newaxis])

            reconstructed = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.reconstructed_response.ordinate)**2)/self.abscissa_spacing, self._response_coordinate_[..., np.newaxis])

            return_data = truth.error_summary(figure_kwargs=figure_kwargs, 
                                              linewidth=linewidth, 
                                              plot_kwargs=plot_kwargs,
                                              cpsd_matrices=reconstructed)
        return return_data

    @linear_inverse_processing
    def manual_inverse(self, method='standard',
                       regularization_weighting_matrix=None,
                       regularization_parameter=None,
                       cond_num_threshold=None,
                       num_retained_values=None,
                       use_transformation=True,
                       response=None, frf=None):
        """
        Perform the inverse source estimation problem with manual settings. 

        Parameters
        ----------
        method : str, optional
            The method to be used for the FRF matrix inversions. The available 
            methods are:
                - standard - basic pseudo-inverse via numpy.linalg.pinv with the
                  default rcond parameter, this is the default method
                - threshold - pseudo-inverse via numpy.linalg.pinv with a specified
                  condition number threshold
                - tikhonov - pseudo-inverse using the Tikhonov regularization method
                - truncation - pseudo-inverse where a fixed number of singular values
                  are retained for the inverse 
        regularization_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Matrix used to weight input degrees of freedom via Tikhonov regularization. 
            This matrix can also be a 3D matrix such that the the weights are different
            for each frequency line. The matrix should be sized 
            [number of lines, number of references, number of references], where the number 
            of lines either be one (the same weights at all frequencies) or the length
            of the abscissa (for the case where a 3D matrix is supplied).
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the regularization weighting matrix when the tikhonov
            method is chosen. A vector of regularization parameters can be provided so the 
            regularization is different at each frequency line. The vector must match the 
            length of the abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        cond_num_threshold : float or np.ndarray, optional
            Condition number used for SVD truncation when the threshold method is chosen. 
            A vector of condition numbers can be provided so it varies as a function of 
            frequency. The vector must match the length of the abscissa in this case.
        num_retained_values : float or np.ndarray, optional
            Number of singular values to retain in the pseudo-inverse when the truncation 
            method is chosen. A vector of can be provided so the number of retained values 
            can change as a function of frequency. The vector must match the length of the 
            abscissa in this case.
        use_transformation : bool
            Whether or not the response and reference transformation from the class definition
            should be used (which is handled in the "linear_inverse_processing" decorator 
            function). The default is true. 
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator function
            and object definition. This argument should not be supplied by the user.

        Returns
        -------
        force : ndarray
            An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        Notes
        -----
        The "linear_inverse_processing" decorator function pre and post processes the target
        response and FRF data from the SourcePathReceiver object to use the response and 
        reference transformation matrices. This method only estimates the forces, using the 
        supplied FRF inverse parameters. 
        """
        H_pinv = frf_inverse(frf, method = method,
                             regularization_weighting_matrix = regularization_weighting_matrix,
                             regularization_parameter = regularization_parameter,
                             cond_num_threshold = cond_num_threshold,
                             num_retained_values = num_retained_values)
        force = H_pinv@response

        self.inverse_settings.update({'ISE_technique':'manual_inverse',
                                      'inverse_method':method,
                                      'regularization_weighting_matrix':regularization_weighting_matrix,
                                      'regularization_parameter':regularization_parameter,
                                      'cond_num_threshold':cond_num_threshold,
                                      'num_retained_values':num_retained_values,
                                      'use_transformation':use_transformation})
        return force
    
    @linear_inverse_processing
    def auto_tikhonov_by_l_curve(self,
                                 low_regularization_limit = None, 
                                 high_regularization_limit = None,
                                 number_regularization_values=100,
                                 l_curve_type = 'forces',
                                 optimality_condition = 'curvature',
                                 curvature_method = 'numerical',
                                 use_transformation=True,
                                 response=None, frf=None,
                                 parallel = False,
                                 num_jobs = -2):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with L-curve 
        methods.

        Parameters
        ----------
        low_regularization_limit : ndarray
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the smallest singular
            value of the target frf array.
        high_regularization_limit : ndarray
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object.
        number_regularization_values : int
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:
                - forces (default) - This L-curve is constructed with the "size" 
                of the forces on the Y-axis and the regularization parameter on the 
                X-axis. 
                - standard - This L-curve is constructed with the residual squared 
                error on the X-axis and the "size" of the forces on the Y-axis. 
        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
            The options are:
                - curvature (default) - This method searches for the regularization
                parameter that results in maximum curvature of the L-curve. It is 
                also referred to as the L-curve criterion. 
                - distance - This method searches for the regularization parameter that
                minimizes the distance between the L-curve and a "virtual origin". A 
                virtual origin is used, because the L-curve is scaled and offset to always 
                range from zero to one, in this case.
        curvature_method : std
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The options are:
                - numerical (default) - this method computes the curvature of 
                the L-curve via numerical derivatives
                - cubic_spline - this method fits a cubic spline to the L-curve
                the computes the curvature from the cubic spline (this might 
                perform better if the L-curve isn't "smooth")
        use_transformation : bool
            Whether or not the response and reference transformation from the class 
            definition should be used (which is handled in the "linear_inverse_processing" 
            decorator function). The default is true. 
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user.
        parallel : bool
            Whether or not to parallelize the computation using Joblib. The default is 
            False. 
        num_jobs : int
            The number of processors to use when parallelizing the code. The default is 
            -2, which uses all the available processors except one. Refer to the joblib
            documentation for more details. 

        Returns
        -------
        selected_force : ndarray
            An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        Raises
        ------
        ValueError
            If the requested L-curve type is not available.
        ValueError
            If the requested optimality condition is not available.

        Notes
        -----
        Parallelizing generally isn't faster for "small" inverse problems because of the 
        overhead involved in the parallelizing. Some experience has shown that the 
        parallelization adds ~1-1.5 minutes to the computation, but this will depend on 
        the specific computer that is being used.

        All the setting, including the selected regularization parameters, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
        .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
            of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
            vol. 14, no. 6, pp. 1487-1503, 1993.
        .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
            problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
            2000, pp. 119-142.  
        .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
            Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008. 
        """
        forces_full_path, regularization_values, residual, penalty = tikhonov_full_path(frf, response[..., 0], # Need the removing the trailing axis from the end of the response
                                                                                        low_regularization_limit=low_regularization_limit, 
                                                                                        high_regularization_limit=high_regularization_limit,
                                                                                        number_regularization_values=number_regularization_values,
                                                                                        parallel=parallel,
                                                                                        num_jobs=num_jobs)
            
        selected_force, optimal_regularization = l_curve_selection(regularization_values, penalty, residual, forces_full_path,
                                                                   l_curve_type=l_curve_type,
                                                                   optimality_condition=optimality_condition,
                                                                   curvature_method =curvature_method)
        
        self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                      'inverse_method':'Tikhonov regularization',
                                      'number_regularization_values_searched':number_regularization_values,
                                      'regularization_parameter':optimal_regularization,
                                      'l_curve_type':l_curve_type,
                                      'optimality_condition':optimality_condition,
                                      'curvature_method':curvature_method,
                                      'use_transformation':use_transformation})

        return selected_force[..., np.newaxis]

    @linear_inverse_processing
    def auto_truncation_by_l_curve(self,
                                   l_curve_type = 'standard',
                                   optimality_condition = 'distance',
                                   curvature_method = None,
                                   use_transformation=True,
                                   response=None, frf=None):
        """
        Performs the inverse source estimation problem with the truncated singular
        value decomposition (TSVD). The number of singular values to retain in the 
        inverse is automatically selected with L-curve methods
        
        Parameters
        ----------
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:
                - forces - This L-curve is constructed with the "size" of the 
                forces on the Y-axis and the regularization parameter on the X-axis. 
                - standard (default) - This L-curve is constructed with the residual 
                squared error on the X-axis and the "size" of the forces on the Y-axis. 
        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
            The options are:
                - curvature - This method searches for the regularization parameter 
                that results in maximum curvature of the L-curve. It is also referred 
                to as the L-curve criterion. 
                - distance (default) - This method searches for the regularization 
                parameter that minimizes the distance between the L-curve and a "virtual 
                origin". A virtual origin is used, because the L-curve is scaled and 
                offset to always range from zero to one, in this case.
        curvature_method : std
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The default is None and the options are:
                - numerical - this method computes the curvature of the L-curve via 
                numerical derivatives
                - cubic_spline - this method fits a cubic spline to the L-curve
                the computes the curvature from the cubic spline (this might 
                perform better if the L-curve isn't "smooth")
        use_transformation : bool
            Whether or not the response and reference transformation from the class 
            definition should be used (which is handled in the "linear_inverse_processing" 
            decorator function). The default is true. 
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user.

        Returns
        -------
        selected_force : ndarray
            An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        Notes
        -----
        L-curve for the TSVD could be non-smooth and determining the number of singular 
        values to retain via curvature methods could lead to erratic results.  

        All the setting, including the number of singular values to retain, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
        .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
            of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
            vol. 14, no. 6, pp. 1487-1503, 1993.
        .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
            problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
            2000, pp. 119-142.  
        .. [3] L. Reichel and H. Sadok, "A new L-curve for ill-posed problems," Journal of 
            Computational and Applied Mathematics, vol. 219, no. 2, pp. 493-508, 
            2008/10/01/ 2008, doi: https://doi.org/10.1016/j.cam.2007.01.025.    
        """
        num_forces = frf.shape[2]
        regularization_values=np.arange(num_forces)+1

        forces_full_path = np.zeros((self._abscissa_.shape[0], num_forces, num_forces), dtype=complex)
        for ii in range(num_forces):
            forces_full_path[:, ii, :] = (pinv_by_truncation(frf, ii+1)@response)[..., 0]
        penalty = norm(forces_full_path, axis = -1, ord = 2)**2
        residual = norm((response[:, np.newaxis, :] - frf[:, np.newaxis, ...]@forces_full_path[..., np.newaxis])[..., 0], axis=-1, ord=2)**2

        selected_force, optimal_regularization = l_curve_selection(np.broadcast_to(regularization_values, (self._abscissa_.shape[0], num_forces)), 
                                                                   penalty, residual, forces_full_path,
                                                                   l_curve_type=l_curve_type,
                                                                   optimality_condition=optimality_condition,
                                                                   curvature_method =curvature_method)
        
        self.inverse_settings.update({'ISE_technique':'auto_truncation_by_l_curve',
                                      'inverse_method':'truncated singular value decomposition',
                                      'number_retained_values':optimal_regularization,
                                      'l_curve_type':l_curve_type,
                                      'optimality_condition':optimality_condition,
                                      'curvature_method':curvature_method,
                                      'use_transformation':use_transformation})

        return selected_force[..., np.newaxis]

    @linear_inverse_processing
    def elastic_net_by_information_criterion(self, alpha_parameter,
                                             number_of_lambdas=100,
                                             information_criterion='AICC',
                                             use_transformation=True,
                                             response=None, frf=None,
                                             **kwargs):
        """
        Perform the inverse source estimation problem with the elastic net and 
        perform the model selection (to determine the optimal regularization 
        parameter) with an information criterion.

        Parameters
        ----------
        alpha_parameter : float
            Alpha parameter for the elastic net. This controls the balance between the
            L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
            than 0 and less than 1. 
        number_of_lambdas : int   
            This parameter is supplied if the lambda_values are being determined by
            the code. The default is 100. 
        information_criterion : str
            The desired information criterion, the available options are:
                - 'BIC' - the Bayesian information criterion
                - 'AIC' - the Akaike information criterion
                - 'AICC' (default) - the corrected Akaike information criterion
        use_transformation : bool
            Whether or not the response and reference transformation from the class 
            definition should be used (which is handled in the "linear_inverse_processing" 
            decorator function). The default is true. 
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user.

        Returns
        -------
        selected_force : ndarray
            An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        References
        ----------
        .. [1] T. Hastie, R. Tibshirani, M. Wainright, Statistical Learning with Sparsity:
            The Lasso with Generalizations. Boca Raton, Fl: CRC Press, 2015. 
        .. [2] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
            Linear Models via Coordinate Descent, Journal of Statistical Software, 
            Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01. 
        """
        forces_full_path, regularization_values = elastic_net_full_path_all_frequencies_parallel(frf, 
                                                                                                 response[..., 0], 
                                                                                                 alpha_parameter, 
                                                                                                 number_of_lambdas = number_of_lambdas)
        
        forces_full_path = np.moveaxis(forces_full_path, -1, -2) # The forces get reshaped to match what happens in the other methods
        selected_force = select_model_by_information_criterion(frf, response, forces_full_path, information_criterion)

        self.inverse_settings.update({'ISE_technique':'elastic_net_by_information_criterion',
                                      'inverse_method':'elastic net',
                                      'alpha_parameter':alpha_parameter,
                                      'information_criterion':information_criterion,
                                      'use_transformation':use_transformation})
        return selected_force[..., np.newaxis]
        
class PowerSourcePathReceiver(SourcePathReceiver):
    """
    A subclass to represent a source-path-receiver (SPR) model of a system for MIMO 
    vibration testing or transfer path analysis. The responses and forces in this
    subclass are power spectra.

    Attributes
    ----------
    frfs : TransferFunctionArray
        The "full" FRFs that define the path of the SPR object.
    response : PowerSpectralDensityArray
        The measured responses that define the receiver of the SPR object. This is 
        the response at all the locations that are represented in the FRFs (the 
        target locations and cross validation locations).
    force : PowerSpectralDensityArray
        The forces that define the source in the SPR model. The force degrees of freedom 
        (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
        set via an inverse method in the class, but the user can also set them manually. 
    target_response : PowerSpectralDensityArray
        The measured responses that will be used to estimate the forces in the SPR object 
        (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
        response if a target response or target_response_coordinate is not supplied.
    transformed_target_response : PowerSpectralDensityArray
        The target_response with the response_transformation applied.
    target_frfs : TransferFunctionArray
        A subset of the FRFs that have the same response DOFs as target response.
    transformed_target_frfs : TransferFunctionArray
        The target_frfs with the response_transformation and reference_transformation
        applied.
    buzz_cpsd : PowerSpectralDensityArray
        The cpsd matrix from the system ID matrix to use the so-called "buzz method"
        in the inverse source estimation.
    reconstructed_response : PowerSpectralDensityArray
        The computed responses from the FRFs and forces. This cannot be set by the user. 
    transformed_reconstructed_response : PowerSpectralDensityArray
        The reconstructed response (at the target_response_coordinate) with the 
        response_transformation applied. 
    response_coordinate : coordinate_array
        The response coordinates of the SPR object, based on the FRFs.
    target_response_coordinate : coordinate_array
        The target response coordinates of teh SPR object, based on the target responses.
    reference_coordinate : coordinate_array
        The reference coordinates of the SPR object, based on the FRFs.
    response_transformation : Matrix, optional
        The response transformation that is used in the inverse problem. The default is 
        identity. 
    transformed_response_coordinate : coordinate_array
        The coordinates that the response is transformed into through the response 
        transformation array.
    reference_transformation : Matrix, optional
        The reference transformation that is used in the inverse problem. The default is 
        identity.
    transformed_reference_coordinate : coordinate_array
        The coordinates that the reference is transformed into through the reference 
        transformation array.
    abscissa : float
        The frequency vector of the SPR model.
    abscissa_spacing : float
        The frequency resolution for the SPR model.
    inverse_settings : dict
        The settings that were to estimate the sources in the SourcePathReceiver object.

    Notes
    -----
    The ordinate in the full FRFs and responses can be different for the ordinate in the
    target FRFs and responses (depending on the problem set-up).

    The "power" term in the class name stands for the power units in the response and
    force spectra.
    """

    def __init__(self, frfs=None, response=None, force=None, target_response=None, target_response_coordinate=None, 
                 target_frfs=None, buzz_cpsd=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Parameters
        ----------
        frfs : TransferFunctionArray
            The "full" FRFs that define the path of the SPR object.
        response : PowerSpectralDensityArray
            The measured responses that define the receiver of the SPR object. This is 
            the response at all the locations that are represented in the FRFs (the 
            target locations and cross validation locations).
        force : PowerSpectralDensityArray, optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        target_response : PowerSpectralDensityArray, optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a target response or target_response_coordinate is not supplied.
        buzz_cpsd : PowerSpectralDensityArray, optional
            The cpsd matrix from the system ID matrix to use the so-called "buzz method"
            in the inverse source estimation. Defaults to None. 
        target_response_coordinate : coordinate_array, optional
            The target response coordinates of teh SPR object, based on the target responses.
        response_transformation : Matrix, optional
            The response transformation that is used in the inverse problem. The default is 
            identity. 
        reference_transformation : Matrix, optional
            The reference transformation that is used in the inverse problem. The default is 
            identity.
        empty : bool, optional
            Whether or not to create an "empty" SPR object where all the attributes are None.
            The default is False (to create a "full" SPR object).

        Notes
        -----
        Much of the initialization is inherited from the base SourcePathReceiver class.
        """
        # Inheriting the initial set-up from 
        super().__init__(frfs=frfs, response=response, force=force, target_response=target_response, target_response_coordinate=target_response_coordinate, 
                         target_frfs=target_frfs, response_transformation=response_transformation, reference_transformation=reference_transformation, empty=empty)
        self.buzz_cpsd = buzz_cpsd

    @property
    def response(self):
        if self._response_array_ is None:
            raise AttributeError('A response array was not defined for this object')
        if is_cpsd(self._response_array_):
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(self._response_array_, 0, -1), 
                                   outer_product(self._response_coordinate_, self._response_coordinate_))
        else: 
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(self._response_array_, 0, -1), 
                                   np.column_stack((self._response_coordinate_, self._response_coordinate_)))
    
    @response.setter
    def response(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
            raise TypeError('The response must be a SDynPy PowerSpectralDensityArray')
        check_frequency_abscissa(data_array, self._abscissa_)
        if is_cpsd(data_array):
            self._response_array_ = np.moveaxis(data_array[outer_product(self.response_coordinate, self.response_coordinate)].ordinate, -1, 0)
        else:
            self._response_array_ = np.moveaxis(data_array[self.response_coordinate[..., np.newaxis]].ordinate, -1, 0)
            
    @property
    def force(self):
        if self._force_array_ is None:
            raise AttributeError('A force array is not defined for this object')
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(self._force_array_, 0, -1), 
                               outer_product(self._reference_coordinate_, self._reference_coordinate_))
    
    @force.setter
    def force(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
            raise TypeError('The force must be a SDynPy PowerSpectralDensityArray')
        if not np.all(np.isin(np.unique(data_array.response_coordinate), self.reference_coordinate)):
            raise ValueError('Force {:} is not in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.reference_coordinate)].string_array()))
        check_frequency_abscissa(data_array, self._abscissa_)
        self._force_array_ = np.moveaxis(data_array[outer_product(self.reference_coordinate, self.reference_coordinate)].ordinate, -1, 0)

    @property
    def transformed_force(self):
        """
        The force with the transformation applied.
        """
        if self._reference_transformation_array_.ndim == 2:        
            reference_transform = self._reference_transformation_array_[np.newaxis, ...]
        elif self._reference_transformation_array_.ndim == 3:
            reference_transform = self._reference_transformation_array_
        else:
            raise ValueError('The shape of the reference transformation array is not compatible with the object')
        transformed_force_array = reference_transform@self._target_response_array_@np.transpose(reference_transform.conj(), (0, 2, 1))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(transformed_force_array, 0, -1), 
                               outer_product(self._transformed_reference_coordinate_, self._transformed_reference_coordinate_))

    @property
    def target_response(self):
        if is_cpsd(self._target_response_array_):
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, np.moveaxis(self._target_response_array_, 0, -1), 
                                   outer_product(self._target_response_coordinate_, self._target_response_coordinate_))
        else:
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, np.moveaxis(self._target_response_array_, 0, -1), 
                                   np.column_stack((self._target_response_coordinate_, self._target_response_coordinate_)))
    
    @target_response.setter
    def target_response(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
            raise TypeError('The target response must be a SDynPy PowerSpectralDensityArray')
        if not np.all(np.isin(np.unique(data_array.response_coordinate), self.response_coordinate)):
            raise ValueError('Target response {:} is not in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.response_coordinate)].string_array()))
        if self._target_response_array_ is not None and not np.all(np.unique(data_array.response_coordinate)==self._target_response_coordinate_):
            raise ValueError('The target response coordinates do not match the SourcePathReceiver object')
        check_frequency_abscissa(data_array, self._abscissa_)
        # The numpy unique is used when setting the coordinate to make sure that the DOF ordering
        # in the target_response_array matches the other data.
        if self._target_response_coordinate_ is None:
            self.target_response_coordinate = np.unique(data_array.response_coordinate)
        if is_cpsd(data_array):
            self._target_response_array_ = np.moveaxis(data_array[outer_product(self._target_response_coordinate_, self._target_response_coordinate_)].ordinate, -1, 0) 
        else:
            self._target_response_array_ = np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0)

    @property
    def buzz_cpsd(self):
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, np.moveaxis(self._buzz_cpsd_array_, 0, -1), 
                               outer_product(self._target_response_coordinate_, self._target_response_coordinate_))
    
    @buzz_cpsd.setter
    def buzz_cpsd(self, data_array):
        if data_array is None:
            # Doing this to assign None as the default value for buzz_cpsd_array
            self._buzz_cpsd_array_ = None
        else:
            if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
                raise TypeError('The target response must be a SDynPy PowerSpectralDensityArray')
            if not np.all(np.isin(self.target_response_coordinate, np.unique(data_array.response_coordinate))):
                raise ValueError('Data for response coordinate {:} is missing from the buzz CPSD array'.format(self.target_response_coordinate[~np.isin(self.target_response_coordinate, data_array.response_coordinate)].string_array()))
            check_frequency_abscissa(data_array, self._abscissa_)
            self._buzz_cpsd_array_ = np.moveaxis(data_array[outer_product(self._target_response_coordinate_, self._target_response_coordinate_)].ordinate, -1, 0) 

    @property
    def transformed_target_response(self):
        """
        The target response with the transformation applied (i.e., what is used in the inverse problem). 
        The buzz method is only applied if the target response is a set of PSDs rather than CPSDs. 
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        if is_cpsd(self._target_response_array_):
            transformed_response = response_transform@self._target_response_array_@np.transpose(response_transform.conj(), (0, 2, 1))
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(transformed_response, 0, -1), 
                                   outer_product(self._transformed_response_coordinate_, self._transformed_response_coordinate_))
        else:
            target_response_with_buzz = apply_buzz_method(self)
            transformed_response = response_transform@target_response_with_buzz@np.transpose(response_transform.conj(), (0, 2, 1))
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(transformed_response, 0, -1), 
                                   outer_product(self._transformed_response_coordinate_, self._transformed_response_coordinate_))

    @property
    def reconstructed_response(self):
        """
        Outputs a response CPSD matrix if "response" is a CPSD. Otherwise, the the reconstructed 
        response is indexed to output the PSDs.
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        reconstructed_response = self._frf_array_@self._force_array_@np.transpose(self._frf_array_.conj(), (0, 2, 1))
        
        # Need this logic in case there isn't a response array associated with the SPR object (which could be the case if only a specification was provided)
        if self._response_array_ is not None:
            return_cpsd = is_cpsd(self._response_array_)
        else:
            return_cpsd = is_cpsd(self._target_response_array_)
        
        if return_cpsd:    
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(reconstructed_response, 0, -1), 
                                   outer_product(self._response_coordinate_, self._response_coordinate_)) 
        else:
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(reconstructed_response.diagonal(axis1=1, axis2=2), 0, -1), 
                                   np.column_stack((self._response_coordinate_, self._response_coordinate_)))

    @property
    def transformed_reconstructed_response(self):
        """
        The reconstructed response (at the target coordinates) with the transformation applied (i.e., what 
        was used in the inverse problem). This always outputs a response CPSD matrix for comparisons 
        against the "transformed_target_response".
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        reconstructed_response = self._target_frf_array_@self._force_array_@np.transpose(self._target_frf_array_.conj(), (0, 2, 1))
        reconstructed_response = response_transform@reconstructed_response@np.transpose(response_transform.conj(), (0, 2, 1))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(reconstructed_response, 0, -1), 
                               outer_product(self._transformed_response_coordinate_, self._transformed_response_coordinate_))

    def make_buzz_cpsd_from_frf(self):
        """
        Generates the buzz CPSD array from the target FRFs.
        """
        self._buzz_cpsd_array_ = self._target_frf_array_@self._target_frf_array_.conj().transpose((0, 2, 1))
        return self

    def global_asd_error(self, 
                         channel_set='target'):
        """
        Computes the global ASD error in dB of the reconstructed response, 
        per the procedure in MIL-STD 810H.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - validation - This compares the responses at the DOFs that
                are not included in the untransformed target DOFs. 
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 

        Returns
        -------
        SpectrumArray
            Returns a spectrum array of the global ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.
        
        References
        ----------
        .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.
        """
        if channel_set == 'target':
            truth = self.transformed_target_response.get_asd().ordinate
            reconstructed = self.transformed_reconstructed_response.get_asd().ordinate
        elif channel_set == 'validation':
            validation_coordinate = self._response_coordinate_[~np.isin(self._response_coordinate_, self._target_response_coordinate_)]
            if is_cpsd(self._response_array_):
                truth = self.response[outer_product(validation_coordinate, validation_coordinate)].get_asd().ordinate
                reconstructed = self.reconstructed_response[outer_product(validation_coordinate, validation_coordinate)].get_asd().ordinate
            else:
                truth = self.response[validation_coordinate[..., np.newaxis]].get_asd().ordinate
                reconstructed = self.reconstructed_response[validation_coordinate[..., np.newaxis]].get_asd().ordinate
        elif channel_set == 'full':
            truth = self.response.get_asd().ordinate
            reconstructed = self.reconstructed_response.get_asd().ordinate
        
        weights = truth/norm(truth, axis=0, ord=2)
        asd_error = 10*np.log10(reconstructed/truth)
        global_asd_error = np.sum(asd_error*weights, axis=0)
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, global_asd_error, coordinate_array(node=1, direction=1))

    def average_asd_error(self, 
                          channel_set='target'):
        """
        Computes the average ASD error in dB of the reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - validation - This compares the responses at the DOFs that
                are not included in the untransformed target DOFs. 
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 

        Returns
        -------
        PowerSpectralDensityArray
            Returns a spectrum array of the average ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.       
        """
        if channel_set == 'target':
            truth = self.transformed_target_response.get_asd().ordinate
            reconstructed = self.transformed_reconstructed_response.get_asd().ordinate
        elif channel_set == 'validation':
            validation_coordinate = self._response_coordinate_[~np.isin(self._response_coordinate_, self._target_response_coordinate_)]
            if is_cpsd(self._response_array_):
                truth = self.response[outer_product(validation_coordinate, validation_coordinate)].get_asd().ordinate
                reconstructed = self.reconstructed_response[outer_product(validation_coordinate, validation_coordinate)].get_asd().ordinate
            else:
                truth = self.response[validation_coordinate[..., np.newaxis]].get_asd().ordinate
                reconstructed = self.reconstructed_response[validation_coordinate[..., np.newaxis]].get_asd().ordinate
        elif channel_set == 'full':
            truth = self.response.get_asd().ordinate
            reconstructed = self.reconstructed_response.get_asd().ordinate
        
        asd_error = 10*np.log10(reconstructed/truth)
        average_asd_error = 10*np.log10(np.average(10**(asd_error/10), axis=0))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, average_asd_error, coordinate_array(node=1, direction=1))

    def error_summary(self,
                      channel_set='target',
                      figure_kwargs={},
                      linewidth=1,
                      plot_kwargs={}):
        """
        Plots the error summary using the method of the same name from the 
        PowerSpectralDensityArray class.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - validation - This compares the responses at the DOFs that
                are not included in the untransformed target DOFs. 
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 
        figure_kwargs : dict, optional
            Arguments to use when creating the figure. The default is {}.
        linewidth : float, optional 
            Widths of the lines on the plot. The default is 1.
        plot_kwargs : dict, optional
            Arguments to use when plotting the lines. The default is {}.

        Returns
        -------
        Error Metrics
            A tuple of dictionaries of error metrics

        Notes
        -----
        This is a simple wrapper around the "error_summary" method.
        """
        if channel_set == 'target':
            return_data = self.transformed_target_response.error_summary(figure_kwargs=figure_kwargs, 
                                                                         linewidth=linewidth, 
                                                                         plot_kwargs=plot_kwargs,
                                                                         cpsd_matrices=self.transformed_reconstructed_response)
        elif channel_set == 'validation':
            validation_coordinate = self._response_coordinate_[~np.isin(self._response_coordinate_, self._target_response_coordinate_)]
            if is_cpsd(self._response_array_):
                return_data = self.response[outer_product(validation_coordinate, validation_coordinate)].error_summary(figure_kwargs=figure_kwargs, 
                                                                                                                    linewidth=linewidth, 
                                                                                                                    plot_kwargs=plot_kwargs,
                                                                                                                    cpsd_matrices=self.reconstructed_response[outer_product(validation_coordinate, validation_coordinate)])
            else:
                return_data = self.response[validation_coordinate[..., np.newaxis]].error_summary(figure_kwargs=figure_kwargs, 
                                                                                                  linewidth=linewidth, 
                                                                                                  plot_kwargs=plot_kwargs,
                                                                                                  cpsd_matrices=self.reconstructed_response[validation_coordinate[..., np.newaxis]])
        elif channel_set == 'full':
            return_data = self.response.error_summary(figure_kwargs=figure_kwargs, 
                                                      linewidth=linewidth, 
                                                      plot_kwargs=plot_kwargs,
                                                      cpsd_matrices=self.reconstructed_response)
        return return_data

    @power_inverse_processing
    def manual_inverse(self, method='standard',
                       regularization_weighting_matrix=None,
                       regularization_parameter=None,
                       cond_num_threshold=None,
                       num_retained_values=None,
                       use_transformation=True,
                       use_buzz=False,
                       update_header=True,
                       response=None, frf=None):
        """
        Perform the inverse source estimation problem with manual settings. 

        Parameters
        ----------
        method : str, optional
            The method to be used for the FRF matrix inversions. The available 
            methods are:
                - standard - basic pseudo-inverse via numpy.linalg.pinv with the
                  default rcond parameter, this is the default method
                - threshold - pseudo-inverse via numpy.linalg.pinv with a specified
                  condition number threshold
                - tikhonov - pseudo-inverse using the Tikhonov regularization method
                - truncation - pseudo-inverse where a fixed number of singular values
                  are retained for the inverse 
        regularization_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Matrix used to weight input degrees of freedom via Tikhonov regularization. 
            This matrix can also be a 3D matrix such that the the weights are different
            for each frequency line. The matrix should be sized 
            [number of lines, number of references, number of references], where the number 
            of lines either be one (the same weights at all frequencies) or the length
            of the abscissa (for the case where a 3D matrix is supplied).
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the regularization weighting matrix when the tikhonov
            method is chosen. A vector of regularization parameters can be provided so the 
            regularization is different at each frequency line. The vector must match the 
            length of the abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        cond_num_threshold : float or np.ndarray, optional
            Condition number used for SVD truncation when the threshold method is chosen. 
            A vector of condition numbers can be provided so it varies as a function of 
            frequency. The vector must match the length of the abscissa in this case.
        num_retained_values : float or np.ndarray, optional
            Number of singular values to retain in the pseudo-inverse when the truncation 
            method is chosen. A vector of can be provided so the number of retained values 
            can change as a function of frequency. The vector must match the length of the 
            abscissa in this case.
        use_transformation : bool
            Whether or not the response and reference transformation from the class definition
            should be used (which is handled in the "power_inverse_processing" decorator 
            function). The default is true. 
        use_buzz : bool
            Whether or not to use the buzz method with the buzz CPSDs from the class 
            definition (this is handled in the "power_inverse_processing" decorator
            function). The default is false. 
        update_header : bool
            Whether or not to update the "inverse_settings" dictionary with all the settings
            from the inverse problem. This exists primarily for compatibility with the 
            Rattlesnake control law, where updating the header information is undesirable 
            for how the settings are parsed. 
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator function
            and object definition. This argument should not be supplied by the user.

        Returns
        -------
        force : ndarray
            An ndarray of the estimated sources. The "power_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        Notes
        -----
        The "power_inverse_processing" decorator function pre and post processes the target
        response and FRF data from the SourcePathReceiver object to use the response and 
        reference transformation matrices. This method only estimates the forces, using the 
        supplied FRF inverse parameters. 
        
        References
        ----------
        .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation, 
            Aerospace Department, University of Bristol, 2014
        """
        H_pinv = frf_inverse(frf, method = method,
                             regularization_weighting_matrix = regularization_weighting_matrix,
                             regularization_parameter = regularization_parameter,
                             cond_num_threshold = cond_num_threshold,
                             num_retained_values = num_retained_values)
        force = H_pinv@response@np.transpose(H_pinv.conj(), (0, 2, 1))

        if update_header:
            self.inverse_settings.update({'ISE_technique':'manual_inverse',
                                        'inverse_method':method,
                                        'regularization_weighting_matrix':regularization_weighting_matrix,
                                        'regularization_parameter':regularization_parameter,
                                        'cond_num_threshold':cond_num_threshold,
                                        'num_retained_values':num_retained_values,
                                        'use_transformation':use_transformation,
                                        'use_buzz':use_buzz})
        return force
    
    @power_inverse_processing
    def auto_tikhonov_by_l_curve(self,
                                 low_regularization_limit = None, 
                                 high_regularization_limit = None,
                                 number_regularization_values=100,
                                 l_curve_type = 'forces',
                                 optimality_condition = 'curvature',
                                 curvature_method = 'numerical',
                                 use_transformation=True,
                                 use_buzz = False,
                                 response = None, frf = None,
                                 parallel = False,
                                 num_jobs = -2):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with L-curve 
        methods.

        Parameters
        ----------
        low_regularization_limit : ndarray
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the smallest singular
            value of the target frf array.
        high_regularization_limit : ndarray
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object.
        number_regularization_values : int
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:
                - forces (default) - This L-curve is constructed with the "size" 
                of the forces on the Y-axis and the regularization parameter on the 
                X-axis. 
                - standard - This L-curve is constructed with the residual squared 
                error on the X-axis and the "size" of the forces on the Y-axis. 
        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
            The options are:
                - curvature (default) - This method searches for the regularization
                parameter that results in maximum curvature of the L-curve. It is 
                also referred to as the L-curve criterion. 
                - distance - This method searches for the regularization parameter that
                minimizes the distance between the L-curve and a "virtual origin". A 
                virtual origin is used, because the L-curve is scaled and offset to always 
                range from zero to one, in this case.
        curvature_method : std
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The options are:
                - numerical (default) - this method computes the curvature of 
                the L-curve via numerical derivatives
                - cubic_spline - this method fits a cubic spline to the L-curve
                the computes the curvature from the cubic spline (this might 
                perform better if the L-curve isn't "smooth")
        use_transformation : bool
            Whether or not the response and reference transformation from the class 
            definition should be used (which is handled in the "power_inverse_processing" 
            decorator function). The default is true. 
        use_buzz : bool
            Whether or not to use the buzz method with the buzz CPSDs from the class 
            definition (this is handled in the "power_inverse_processing" decorator
            function). The default is false. 
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user.
        parallel : bool
            Whether or not to parallelize the computation using Joblib. The default is 
            False. 
        num_jobs : int
            The number of processors to use when parallelizing the code. The default is 
            -2, which uses all the available processors except one. Refer to the joblib
            documentation for more details. 

        Returns
        -------
        selected_force : ndarray
            An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        Notes
        -----
        Parallelizing generally isn't faster for "small" inverse problems because of the 
        overhead involved in the parallelizing. Some experience has shown that the 
        parallelization adds ~1-1.5 minutes to the computation, but this will depend on 
        the specific computer that is being used.

        All the setting, including the selected regularization parameters, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
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
        """
        forces_full_path, regularization_values, residual, penalty = tikhonov_full_path(frf, response,
                                                                                        low_regularization_limit=low_regularization_limit, 
                                                                                        high_regularization_limit=high_regularization_limit,
                                                                                        number_regularization_values=number_regularization_values,
                                                                                        parallel=parallel,
                                                                                        num_jobs=num_jobs)
            
        selected_force, optimal_regularization = l_curve_selection(regularization_values, penalty, residual, forces_full_path,
                                                                   l_curve_type=l_curve_type,
                                                                   optimality_condition=optimality_condition,
                                                                   curvature_method =curvature_method)
        
        self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                      'inverse_method':'tikhonov',
                                      'number_regularization_values_searched':number_regularization_values,
                                      'regularization_parameter':optimal_regularization,
                                      'l_curve_type':l_curve_type,
                                      'optimality_condition':optimality_condition,
                                      'curvature_method':curvature_method,
                                      'use_transformation':use_transformation,
                                      'use_buzz':use_buzz})

        return selected_force
    
    @power_inverse_processing
    def auto_truncation_by_l_curve(self,
                                   l_curve_type = 'standard',
                                   optimality_condition = 'distance',
                                   curvature_method = None,
                                   use_transformation = True,
                                   use_buzz = False,
                                   response = None, frf = None):
        """
        Performs the inverse source estimation problem with the truncated singular
        value decomposition (TSVD). The number of singular values to retain in the 
        inverse is automatically selected with L-curve methods
        
        Parameters
        ----------
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:
                - forces - This L-curve is constructed with the "size" of the 
                forces on the Y-axis and the regularization parameter on the X-axis. 
                - standard (default) - This L-curve is constructed with the residual 
                squared error on the X-axis and the "size" of the forces on the Y-axis. 
        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
            The options are:
                - curvature - This method searches for the regularization parameter 
                that results in maximum curvature of the L-curve. It is also referred 
                to as the L-curve criterion. 
                - distance (default) - This method searches for the regularization 
                parameter that minimizes the distance between the L-curve and a "virtual 
                origin". A virtual origin is used, because the L-curve is scaled and 
                offset to always range from zero to one, in this case.
        curvature_method : std
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The default is None and the options are:
                - numerical - this method computes the curvature of the L-curve via 
                numerical derivatives
                - cubic_spline - this method fits a cubic spline to the L-curve
                the computes the curvature from the cubic spline (this might 
                perform better if the L-curve isn't "smooth")
        use_transformation : bool
            Whether or not the response and reference transformation from the class 
            definition should be used (which is handled in the "power_inverse_processing" 
            decorator function). The default is true. 
        use_buzz : bool
            Whether or not to use the buzz method with the buzz CPSDs from the class 
            definition (this is handled in the "power_inverse_processing" decorator
            function). The default is false.  
        response : ndarray
            The preprocessed response data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator 
            function and object definition. This argument should not be supplied by the 
            user.

        Returns
        -------
        selected_force : ndarray
            An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
            function applies this force to the force property of SourcePathReceiver object.

        Notes
        -----
        L-curve for the TSVD could be non-smooth and determining the number of singular 
        values to retain via curvature methods could lead to erratic results.  

        All the setting, including the number of singular values to retain, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
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
        """
        num_forces = frf.shape[-1]
        regularization_values=np.arange(num_forces)+1

        forces_full_path = np.zeros((frf.shape[0], num_forces, num_forces, num_forces), dtype=complex)
        for ii in range(num_forces):
            frf_pinv = pinv_by_truncation(frf, ii+1)
            forces_full_path[:, ii, ...] = frf_pinv@response@np.transpose(frf_pinv.conj(), (0,2,1))
        penalty = norm(forces_full_path, axis=(-2, -1), ord='fro')**2
        residual = norm(response[:, np.newaxis, ...] - frf[:, np.newaxis, ...]@forces_full_path@frf.conj().transpose((0,2,1))[:, np.newaxis, ...], axis=(-2,-1), ord='fro')**2
    
        selected_force, optimal_regularization = l_curve_selection(np.broadcast_to(regularization_values, (frf.shape[0], num_forces)), 
                                                                   penalty, residual, forces_full_path,
                                                                   l_curve_type=l_curve_type,
                                                                   optimality_condition=optimality_condition,
                                                                   curvature_method=curvature_method)
        
        self.inverse_settings.update({'ISE_technique':'auto_truncation_by_l_curve',
                                      'inverse_method':'truncation',
                                      'number_retained_values':optimal_regularization,
                                      'l_curve_type':l_curve_type,
                                      'optimality_condition':optimality_condition,
                                      'curvature_method':curvature_method,
                                      'use_transformation':use_transformation,
                                      'use_buzz':use_buzz})

        return selected_force
    
    def match_trace_update(self, use_transformation=True):
        """
        Applies a "match trace" update to the to the forces in the SPR object to 
        eliminate bias error.

        Parameters
        ----------
        use_transformation : bool, optional
            Whether or not the transformation was used in the input estimation, 
            this should match what was used in the inverse problem for the correct
            behavior. The default is true. 

        References
        ----------
        .. [1] D. Rohe, R. Schultz, and N. Hunter, "Rattlesnake Users Manual," 
                Sandia National Laboratories, 2021. 
        """
        if use_transformation:
            reconstructed_response_trace = np.trace(self.transformed_reconstructed_response.ordinate) 
            target_response_trace = np.trace(self.transformed_target_response.ordinate)
        elif not use_transformation:
            reconstructed_response_trace = np.trace(self.reconstructed_response[self._target_response_coordinate_].ordinate) 
            target_response_trace = np.trace(self.target_response.ordinate)
        
        trace_ratio = target_response_trace / reconstructed_response_trace
        self._force_array_ = self._force_array_*trace_ratio[..., np.newaxis, np.newaxis]

        self.inverse_settings.update({'match_trace_applied':True,
                                      'match_trace_use_transformation':use_transformation})
        
        return self

class TransientSourcePathReceiver(SourcePathReceiver):
    """
    A subclass to represent a source-path-receiver (SPR) model of a system for MIMO 
    vibration testing or transfer path analysis. The responses and forces in this
    subclass are time traces.

    Attributes
    ----------
    frfs : TransferFunctionArray
        The "full" FRFs that define the path of the SPR object.
    response : TimeHistoryArray
        The measured responses that define the receiver of the SPR object. This is 
        the response at all the locations that are represented in the FRFs (the 
        target locations and cross validation locations).
    force : TimeHistoryArray
        The forces that define the source in the SPR model. The force degrees of freedom 
        (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
        set via an inverse method in the class, but the user can also set them manually. 
    target_response : TimeHistoryArray
        The measured responses that will be used to estimate the forces in the SPR object 
        (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
        response if a target response or target_response_coordinate is not supplied.
    transformed_target_response : TimeHistoryArray
        The target_response with the response_transformation applied.
    target_frfs : TransferFunctionArray
        A subset of the FRFs that have the same response DOFs as target response.
    transformed_target_frfs : TransferFunctionArray
        The target_frfs with the response_transformation and reference_transformation
        applied.
    reconstructed_response : TimeHistoryArray
        The computed responses from the FRFs and forces. This cannot be set by the user. 
    transformed_reconstructed_response : TimeHistoryArray
        The reconstructed response (at the target_response_coordinate) with the 
        response_transformation applied. 
    response_coordinate : coordinate_array
        The response coordinates of the SPR object, based on the FRFs.
    target_response_coordinate : coordinate_array
        The target response coordinates of teh SPR object, based on the target responses.
    reference_coordinate : coordinate_array
        The reference coordinates of the SPR object, based on the FRFs.
    response_transformation : Matrix, optional
        The response transformation that is used in the inverse problem. The default is 
        identity. 
    transformed_response_coordinate : coordinate_array
        The coordinates that the response is transformed into through the response 
        transformation array.
    reference_transformation : Matrix, optional
        The reference transformation that is used in the inverse problem. The default is 
        identity.
    transformed_reference_coordinate : coordinate_array
        The coordinates that the reference is transformed into through the reference 
        transformation array.
    time_abscissa : float
        The time vector of the SPR model.
    time_abscissa_spacing : float
        The sampling time for the SPR model.
    inverse_settings : dict
        The settings that were to estimate the sources in the SourcePathReceiver object.

    Notes
    -----
    The ordinate in the full FRFs and responses can be different for the ordinate in the
    target FRFs and responses (depending on the problem set-up).

    The "transient" term in the class name refers to the intended use of this SPR model
    (transient problems).
    """

    def __init__(self, frfs=None, response=None, force=None, target_response=None, target_response_coordinate=None, 
                 target_frfs=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Parameters
        ----------
        frfs : TransferFunctionArray
            The "full" FRFs that define the path of the SPR object.
        response : TimeHistoryArray
            The measured responses that define the receiver of the SPR object. This is 
            the response at all the locations that are represented in the FRFs (the 
            target locations and cross validation locations).
        force : TimeHistoryArray, optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        target_response : TimeHistoryArray, optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a target response or target_response_coordinate is not supplied .
        target_response_coordinate : coordinate_array, optional
            The target response coordinates of teh SPR object, based on the target responses.
        response_transformation : Matrix, optional
            The response transformation that is used in the inverse problem. The default is 
            identity. 
        reference_transformation : Matrix, optional
            The reference transformation that is used in the inverse problem. The default is 
            identity.
        empty : bool, optional
            Whether or not to create an "empty" SPR object where all the attributes are None.
            The default is False (to create a "full" SPR object).

        Notes
        -----
        Much of the initialization is inherited from the base SourcePathReceiver class.
        """
        # Inheriting the initial set-up from 
        super().__init__(frfs=frfs, response=response, force=force, target_response=target_response, target_response_coordinate=target_response_coordinate, 
                         target_frfs=target_frfs, response_transformation=response_transformation, reference_transformation=reference_transformation, empty=empty)
    
    def save(self, filename):
        """
        Saves the TransientSourcePathReceiver object to a .npz file.

        Parameters
        ----------
        filename : str
            The file path and name for the .npz file
        
        Notes
        -----
        The private properties of the class are saved as arguments in the .npz file, where 
        the argument names match the private variable name. 

        The save method is specially defined for the TransientSourcePathReceiver because 
        it has a "time_abscissa" private property, which isn't in the other SourcePathReceiver 
        objects.
        """
        np.savez(filename, 
                 response=self._response_array_,
                 frf=self._frf_array_,
                 force=self._force_array_,
                 target_response=self._target_response_array_,
                 target_frf=self._target_frf_array_,
                 response_transformation=self._response_transformation_array_,
                 reference_transformation=self._reference_transformation_array_,
                 target_response_coordinate=self._target_response_coordinate_.string_array(),
                 response_coordinate=self._response_coordinate_.string_array(),
                 reference_coordinate=self._reference_coordinate_.string_array(),
                 transformed_response_coordinate=self._transformed_response_coordinate_.string_array(),
                 transformed_reference_coordinate=self._transformed_reference_coordinate_.string_array(),
                 abscissa=self._abscissa_,
                 time_abscissa=self._time_abscissa_) 
    
    @classmethod
    def load(cls, filename):
        """
        Loads the TransientSourcePathReceiver object from an .npz file.

        Parameters
        ----------
        filename : str
            The file path and name for the .npz file
        
        Notes
        -----
        The private properties of the class should have been saved as arguments in the .npz 
        file, where the argument names match the private variable name. 

        The load method is specially defined for the TransientSourcePathReceiver because 
        it has a "time_abscissa" private property, which isn't in the other SourcePathReceiver 
        objects.
        """
        loaded_spr = np.load(filename, allow_pickle=True)
        spr_object = cls.__new__(cls)
        spr_object._response_array_ = loaded_spr['response']
        spr_object._frf_array_ = loaded_spr['frf']
        spr_object._force_array_ = loaded_spr['force'] if np.all(loaded_spr['force'] != np.array(None)) else None
        spr_object._target_response_array_ = loaded_spr['target_response']
        spr_object._target_frf_array_ = loaded_spr['target_frf']
        spr_object._response_transformation_array_ = loaded_spr['response_transformation']
        spr_object._reference_transformation_array_ = loaded_spr['reference_transformation']
        spr_object._target_response_coordinate_ = coordinate_array(string_array=loaded_spr['target_response_coordinate'])
        spr_object._response_coordinate_ = coordinate_array(string_array=loaded_spr['response_coordinate'])
        spr_object._reference_coordinate_ = coordinate_array(string_array=loaded_spr['reference_coordinate'])
        spr_object._transformed_response_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_response_coordinate'])
        spr_object._transformed_reference_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_reference_coordinate'])
        spr_object._abscissa_ = loaded_spr['abscissa']
        spr_object._time_abscissa_ = loaded_spr['time_abscissa']
        return spr_object

    @property
    def response(self):
        if self._response_array_ is None:
            raise AttributeError('A response array was not defined for this object')
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(self._response_array_, 0, -1), self._response_coordinate_[..., np.newaxis])
    
    @response.setter
    def response(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.TimeHistoryArray):
            raise TypeError('The response must be a SDynPy TimeHistoryArray')
        compare_sampling_rate(data_array, self._abscissa_.max()*2)
        self._time_abscissa_=data_array.flatten()[0].abscissa
        self._response_array_ = np.moveaxis(data_array[self._response_coordinate_[..., np.newaxis]].ordinate, -1, 0)
            
    @property
    def force(self):
        if self._force_array_ is None:
            raise AttributeError('A force array is not defined for this object')
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(self._force_array_, 0, -1), self._reference_coordinate_[..., np.newaxis])
    
    @force.setter
    def force(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.TimeHistoryArray):
            raise TypeError('The force must be a SDynPy TimeHistoryArray')
        if not np.all(np.isin(data_array.response_coordinate, self.reference_coordinate)):
            raise ValueError('Force {:} is not in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.reference_coordinate)].string_array()))
        compare_sampling_rate(data_array, self._abscissa_.max()*2)
        self._force_array_ = np.moveaxis(data_array[self.reference_coordinate[..., np.newaxis]].ordinate, -1, 0)

    @property
    def transformed_force(self):
        """
        The force with the transformation applied.
        """
        if self._reference_transformation_array_.ndim == 2:        
            transformed_force_array = (self._reference_transformation_array_[np.newaxis, ...]@self._force_array_[..., np.newaxis])[..., 0]
        else:
            raise ValueError('The shape of the reference transformation array is not compatible with the object')
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(transformed_force_array, 0, -1), self._transformed_reference_coordinate_[..., np.newaxis])

    @property
    def target_response(self):
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(self._target_response_array_, 0, -1), self._target_response_coordinate_[..., np.newaxis])
    
    @target_response.setter
    def target_response(self, data_array):
        if not isinstance(data_array, sdpy.core.sdynpy_data.TimeHistoryArray):
            raise TypeError('The target response must be a SDynPy TimeHistoryArray')
        if not np.all(np.isin(data_array.response_coordinate, self.response_coordinate)):
            raise ValueError('Target response {:} is not in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.response_coordinate)].string_array()))
        if self._target_response_array_ is not None and not np.all(data_array.response_coordinate==self._target_response_coordinate_):
            raise ValueError('The target response coordinates do not match the SourcePathReceiver object')
        compare_sampling_rate(data_array, self._abscissa_.max()*2)
        # The numpy unique is used when setting the coordinate to make sure that the DOF ordering
        # in the target_response_array matches the other data.
        if self._target_response_coordinate_ is None:
            self.target_response_coordinate = np.unique(data_array.response_coordinate)
        self._target_response_array_ = np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0) 

    @property
    def transformed_target_response(self):
        """
        The target response with the transformation applied (i.e., what is used in the inverse problem).
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            raise NotImplementedError('A frequency/time dependent response transformation is not implemented for transient problems')
        transformed_response = (response_transform@self._target_response_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(transformed_response, 0, -1), self._transformed_response_coordinate_[..., np.newaxis])

    
    @property
    def reconstructed_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        return self.force.mimo_forward(self.frfs)
    
    @property
    def transformed_reconstructed_response(self):
        """
        The reconstructed response (at the target coordinates) with the transformation applied (i.e., what
        was used in the inverse problem).
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            raise NotImplementedError('A frequency/time dependent response transformation is not implemented for transient problems')
        reconstructed_response = response_transform@np.moveaxis(self.force.mimo_forward(self.target_frfs).ordinate, -1, 0)[..., np.newaxis]
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(reconstructed_response[..., 0], 0, -1), self._transformed_response_coordinate_[..., np.newaxis])
    
    @property
    def time_abscissa(self):
        return self._time_abscissa_

    @property
    def time_abscissa_spacing(self):
        return np.mean(np.diff(self._time_abscissa_))
    
    def global_rms_error(self, 
                         channel_set='target',
                         samples_per_frame=None,
                         frame_length=None,
                         overlap=None,
                         overlap_samples=None):
        """
        Computes the global RMS error in dB of the reconstructed response, 
        per the procedure in MIL-STD 810H.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 
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

        Returns
        -------
        TimeHistoryArray
            Returns a time history array of the global RMS error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.
        
        References
        ----------
        .. [1] MIL-STD-810H: Environmental Engineering Considerations and Laboratory Tests. US Military, 2019.
        """
        if channel_set == 'target':
            truth = self.transformed_target_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                       frame_length=frame_length, 
                                                                       overlap=overlap, 
                                                                       overlap_samples=overlap_samples,
                                                                       allow_fractional_frames=True)
            reconstructed = self.transformed_reconstructed_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                                      frame_length=frame_length, 
                                                                                      overlap=overlap, 
                                                                                      overlap_samples=overlap_samples,
                                                                                      allow_fractional_frames=True)
        elif channel_set == 'full':
            truth = self.response.split_into_frames(samples_per_frame=samples_per_frame,
                                                    frame_length=frame_length, 
                                                    overlap=overlap, 
                                                    overlap_samples=overlap_samples,
                                                    allow_fractional_frames=True)
            reconstructed = self.reconstructed_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                          frame_length=frame_length, 
                                                                          overlap=overlap, 
                                                                          overlap_samples=overlap_samples,
                                                                          allow_fractional_frames=True)
        truth_rms = truth.rms().transpose()
        weights = (truth_rms**2)/(norm(truth_rms, ord=2, axis=0)**2)
        
        reconstructed_rms = reconstructed.rms().transpose()

        rms_error = 20*np.log10(reconstructed_rms/truth_rms)
        rms_time = np.average(truth.abscissa[:, 0, :], axis=1)
        global_rms_error = np.sum(rms_error*weights, axis=0)
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, rms_time, global_rms_error, coordinate_array(node=1, direction=1))
    
    def average_rms_error(self, 
                          channel_set='target',
                          samples_per_frame=None,
                          frame_length=None,
                          overlap=None,
                          overlap_samples=None):
        """
        Computes the average RMS error in dB of the reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:
                - target (default) - This compares the responses for the 
                transformed target DOFs.
                - full - This compares the responses for all the response 
                DOFs in the SPR object. 
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

        Returns
        -------
        TimeHistoryArray
            Returns a time history array of the average RMS error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.
        """
        if channel_set == 'target':
            truth = self.transformed_target_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                       frame_length=frame_length, 
                                                                       overlap=overlap, 
                                                                       overlap_samples=overlap_samples,
                                                                       allow_fractional_frames=True)
            reconstructed = self.transformed_reconstructed_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                                      frame_length=frame_length, 
                                                                                      overlap=overlap, 
                                                                                      overlap_samples=overlap_samples,
                                                                                      allow_fractional_frames=True)
        elif channel_set == 'full':
            truth = self.response.split_into_frames(samples_per_frame=samples_per_frame,
                                                    frame_length=frame_length, 
                                                    overlap=overlap, 
                                                    overlap_samples=overlap_samples,
                                                    allow_fractional_frames=True)
            reconstructed = self.reconstructed_response.split_into_frames(samples_per_frame=samples_per_frame,
                                                                          frame_length=frame_length, 
                                                                          overlap=overlap, 
                                                                          overlap_samples=overlap_samples,
                                                                          allow_fractional_frames=True)
        truth_rms = truth.rms().transpose()        
        reconstructed_rms = reconstructed.rms().transpose()

        rms_error = 20*np.log10(reconstructed_rms/truth_rms)
        rms_time = np.average(truth.abscissa[:, 0, :], axis=1)
        
        average_rms_error = 20*np.log10(np.average(10**(rms_error/20), axis=0))
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, rms_time, average_rms_error, coordinate_array(node=1, direction=1))


    def sosfiltfilt(self, sos, in_place=True, **sosfiltfilt_kwargs):
        """
        Performs forward-backward digital filtering on the time domain components 
        of the spr object with the SciPy sosfiltfilt function

        Parameters
        ----------
        sos :  array_like
            An array second-order filter coefficients from on of the SciPy filter
            design tools (the filter output should be 'sos').
        in_place : bool, optional
            Whether to apply the filters to the original SPR object or not. When 
            true, The filters will be applied to the original SPR object. When 
            false, a copy will be made of the original and the filters will be 
            applied to that copy. The default is true. 
        sosfiltfilt_kwargs : 
            Additional keyword arguments that will be passed to the sosfiltfilt 
            function. The axis keyword will always be set to 0 to be compatible 
            the the organization of the data. 

        Returns
        -------
            SPR object with the time domain components filtered.
        """
        if sosfiltfilt_kwargs is None:
            sosfiltfilt_kwargs = {}
        sosfiltfilt_kwargs['axis']=0

        # setting work object to self is the same as filter self.
        work_object = self if in_place else self.copy()

        if work_object._force_array_ is not None:
            work_object._force_array_ = sosfiltfilt(sos, work_object._force_array_, **sosfiltfilt_kwargs)
        if work_object._response_array_ is not None:
            work_object._response_array_ = sosfiltfilt(sos, work_object._response_array_, **sosfiltfilt_kwargs)
        if work_object._target_response_array_ is not None:
            work_object._target_response_array_ = sosfiltfilt(sos, work_object._target_response_array_, **sosfiltfilt_kwargs)
 
        if in_place:
            return self
        else:
            return work_object

    def manual_inverse(self, 
                       time_method = 'single_frame', 
                       cola_frame_length = None,
                       cola_window = 'hann',
                       cola_overlap = None,
                       zero_pad_length = None,
                       inverse_method = 'standard',
                       regularization_weighting_matrix = None,
                       regularization_parameter = None,
                       cond_num_threshold = None,
                       num_retained_values = None,
                       use_transformation = True):
        """
        Performs the inverse source estimation problem with manual settings.

        Parameters
        ----------
        time_method : str, optional
            The method to used to handle the time data for the inverse source 
            estimation. The available options are:
                - single_frame - this method performs the Fourier deconvolution
                  via an FFT on a single frame that encompases the entire time 
                  signal. 
                - COLA - this method performs the Fourier deconvolution via a 
                  series of FFTs on relatively small frames of the time signal 
                  using a "constant overlap and add" method. This method may be 
                  faster than the single_frame method.
        cola_frame_length : float, optional
            The frame length (in samples) if the COLA method is being used. The
            default frame length is Fs/df from the transfer function. 
        cola_window : str, optional
            The desired window for the COLA procedure, must exist in the scipy
            window library. The default is a hann window.
        cola_overlap : int, optional
            The number of overlapping samples between measurement frames in the
            COLA procedure.  If not specified, a default value of half the
            cola_frame_length is used.
        zero_pad_length : int, optional
            The number of zeros used to pre and post pad the response data, to 
            avoid convolution wrap-around error. The default is to use the 
            "determine_zero_pad_length" function to determine the zero_pad_length.
        inverse_method : str, optional
            The method to be used for the FRF matrix inversions. The available 
            methods are:
                - standard - basic pseudo-inverse via numpy.linalg.pinv with the
                  default rcond parameter, this is the default method
                - threshold - pseudo-inverse via numpy.linalg.pinv with a specified
                  condition number threshold
                - tikhonov - pseudo-inverse using the Tikhonov regularization method
                - truncation - pseudo-inverse where a fixed number of singular values
                  are retained for the inverse 
        regularization_weighting_matrix : sdpy.Matrix, optional
            Matrix used to weight input degrees of freedom via Tikhonov regularization.
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the regularization weighting matrix when the tikhonov
            method is chosen. A vector of regularization parameters can be provided so the 
            regularization is different at each frequency line. The vector must match the 
            length of the FRF abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        cond_num_threshold : float or np.ndarray, optional
            Condition number used for SVD truncation when the threshold method is chosen. 
            A vector of condition numbers can be provided so it varies as a function of 
            frequency. The vector must match the length of the FRF abscissa in this case
            (either be size [num_lines,] or [num_lines, 1]).
        num_retained_values : float or np.ndarray, optional
            Number of singular values to retain in the pseudo-inverse when the truncation 
            method is chosen. A vector of can be provided so the number of retained values 
            can change as a function of frequency. The vector must match the length of the 
            FRF abscissa in this case (either be size [num_lines,] or [num_lines, 1]).

        Returns
        -------
        TimeHistoryArray
            Time history array of the estimated sources

        Notes
        -----
        This method leverages the SDynPy mimo_inverse method for TimeHistoryArrays where the
        transformed FRFs and responses are supplied to the method.

        References
        ----------
        .. [1] Wikipedia, "Overlap-add Method".
            https://en.wikipedia.org/wiki/Overlap-add_method
        """   
        if use_transformation:
            transformed_forces = self.transformed_target_response.mimo_inverse(self.transformed_target_frfs,
                                                                            time_method=time_method, 
                                                                            cola_frame_length=cola_frame_length,
                                                                            cola_window=cola_window,
                                                                            cola_overlap=cola_overlap,
                                                                            zero_pad_length=zero_pad_length,
                                                                            inverse_method=inverse_method,
                                                                            regularization_weighting_matrix=regularization_weighting_matrix,
                                                                            regularization_parameter=regularization_parameter,
                                                                            cond_num_threshold=cond_num_threshold,
                                                                            num_retained_values=num_retained_values)
            
            if self._reference_transformation_array_.ndim == 2:        
                reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis, ...])
            elif self._reference_transformation_array_.ndim == 3:        
                raise NotImplementedError('A frequency/time dependent response transformation is not implemented for transient problems')

            self._force_array_ = (reference_transform@np.moveaxis(transformed_forces.ordinate,-1,0)[..., np.newaxis])[..., 0]
        else:
            forces = self.target_response.mimo_inverse(self.target_frfs,
                                                       time_method=time_method, 
                                                       cola_frame_length=cola_frame_length,
                                                       cola_window=cola_window,
                                                       cola_overlap=cola_overlap,
                                                       zero_pad_length=zero_pad_length,
                                                       inverse_method=inverse_method,
                                                       regularization_weighting_matrix=regularization_weighting_matrix,
                                                       regularization_parameter=regularization_parameter,
                                                       cond_num_threshold=cond_num_threshold,
                                                       num_retained_values=num_retained_values)
            self._force_array_ = np.moveaxis(forces.ordinate,-1,0)

        self.inverse_settings.update({'ISE_technique':'manual',
                                      'time_method':time_method, 
                                      'cola_frame_length':cola_frame_length,
                                      'cola_window':cola_window,
                                      'cola_overlap':cola_overlap,
                                      'zero_pad_length':zero_pad_length,
                                      'inverse_method':inverse_method,
                                      'regularization_weighting_matrix':regularization_weighting_matrix,
                                      'regularization_parameter':regularization_parameter,
                                      'cond_num_threshold':cond_num_threshold,
                                      'num_retained_values':num_retained_values})
        
        return self 

    def auto_tikhonov_by_l_curve(self, 
                                 time_method = 'single_frame', 
                                 cola_frame_length = None,
                                 cola_window = 'hann',
                                 cola_overlap = None,
                                 zero_pad_length = None,
                                 low_regularization_limit = None, 
                                 high_regularization_limit = None,
                                 number_regularization_values=100,
                                 l_curve_type = 'standard',
                                 optimality_condition = 'distance',
                                 curvature_method = 'numerical',
                                 use_transformation=True):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with L-curve 
        methods.

        Parameters
        ----------
        time_method : str, optional
            The method to used to handle the time data for the inverse source 
            estimation. The available options are:
                - single_frame - this method performs the Fourier deconvolution
                  via an FFT on a single frame that encompases the entire time 
                  signal. 
                - COLA - this method performs the Fourier deconvolution via a 
                  series of FFTs on relatively small frames of the time signal 
                  using a "constant overlap and add" method. This method may be 
                  faster than the single_frame method. This method will also 
                  select different regularization parameters for each frame of
                  data.
        cola_frame_length : float, optional
            The frame length (in samples) if the COLA method is being used. The
            default frame length is Fs/df from the transfer function. 
        cola_window : str, optional
            The desired window for the COLA procedure, must exist in the scipy
            window library. The default is a hann window.
        cola_overlap : int, optional
            The number of overlapping samples between measurement frames in the
            COLA procedure.  If not specified, a default value of half the
            cola_frame_length is used.
        zero_pad_length : int, optional
            The number of zeros used to pre and post pad the response data, to 
            avoid convolution wrap-around error. The default is to use the 
            "determine_zero_pad_length" function to determine the zero_pad_length.
        low_regularization_limit : ndarray, optional
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the zero padded FRFs (not the target FRFs in the SPR object). The 
            default is the smallest singular values of the zero padded target 
            FRF array. 
        high_regularization_limit : ndarray, optional
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the zero padded FRFs (not the target FRFs in the SPR object). The 
            default is the largest singular values of the zero padded target 
            FRF array.
        number_regularization_values : int, optional
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        l_curve_type : str, optional
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:
                - forces - This L-curve is constructed with the "size" of the 
                forces on the Y-axis and the regularization parameter on the X-axis. 
                - standard (default) - This L-curve is constructed with the residual 
                squared error on the X-axis and the "size" of the forces on the Y-axis. 
        optimality_condition : str, optional
            The method that is used to find an "optimal" regularization parameter.
            The options are:
                - curvature - This method searches for the regularization parameter 
                that results in maximum curvature of the L-curve. It is also referred 
                to as the L-curve criterion. 
                - distance (default) - This method searches for the regularization 
                parameter that minimizes the distance between the L-curve and a 
                "virtual origin". A virtual origin is used, because the L-curve is 
                scaled and offset to always range from zero to one, in this case.
        curvature_method : std, optional
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The options are:
                - numerical (default) - this method computes the curvature of 
                the L-curve via numerical derivatives
                - cubic_spline - this method fits a cubic spline to the L-curve
                the computes the curvature from the cubic spline (this might 
                perform better if the L-curve isn't "smooth")
        use_transformation : bool, optional
            Whether or not the response and reference transformation from the class 
            definition should be used. The default is true. 

        Returns
        -------
        forces : ndarray
            An ndarray of the estimated sources that is assigned to to the force 
            property  of SourcePathReceiver object.

        Raises
        ------
        ValueError
            If the selected time method is not available.
        ValueError
            If the requested L-curve type is not available.
        ValueError
            If the requested optimality condition is not available.

        Notes
        -----
        Some of the default settings are different than what is used in the 
        PowerSourcePathReceiver and LinearSourcePathReceiver classes, based on results 
        from initial tests. 
        
        All the setting, including the selected regularization parameters, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
        .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
            of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
            vol. 14, no. 6, pp. 1487-1503, 1993.
        .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
            problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
            2000, pp. 119-142.  
        .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
            Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008. 
        .. [4] Wikipedia, "Overlap-add Method".
            https://en.wikipedia.org/wiki/Overlap-add_method
        """
        if use_transformation:
            frf_object = self.transformed_target_frfs
            response_object = self.transformed_target_response
        else:
            frf_object = self.target_frfs
            response_object = self.target_response
        
        model_order = frf_object.ifft().num_elements

        # Preparing the response data and FRFs for the source estimation
        if time_method == 'single_frame':
            # Zero pad for convolution wrap-around
            if zero_pad_length is None:
                padded_response = response_object.zero_pad(2*model_order, left=True, right = True,
                                                           use_next_fast_len = True)
            else:
                padded_response = response_object.zero_pad(zero_pad_length, left=True, right = True)
            actual_zero_pad = padded_response.num_elements - response_object.num_elements
            # Now make the FRFs the same size
            modified_frfs = frf_object.interpolate_by_zero_pad(padded_response.num_elements)
            padded_frequency_domain_data = padded_response.fft(norm='backward')
        elif time_method == 'cola':
            if cola_frame_length is None:
                cola_frame_length = int(round(model_order/2)*2) # This is a slightly strange operation to gaurantee an even frame length
            if cola_overlap is None:
                cola_overlap = cola_frame_length//2
            # Split into measurement frames
            segmented_data = response_object.split_into_frames(samples_per_frame = cola_frame_length, 
                                                               overlap_samples = cola_overlap,
                                                               window = cola_window,
                                                               check_cola = True,
                                                               allow_fractional_frames = True)
            # Zero pad
            if zero_pad_length is None:
                zero_padded_data = segmented_data.zero_pad(2*model_order,left=True,right=True,use_next_fast_len = True)
            else:
                zero_padded_data = segmented_data.zero_pad(zero_pad_length, left=True, right = True)
            actual_zero_pad = zero_padded_data.num_elements - segmented_data.num_elements
            modified_frfs = frf_object.interpolate_by_zero_pad(zero_padded_data.num_elements)
            padded_frequency_domain_data = zero_padded_data.fft(norm='backward')
        else:
            raise ValueError('The selected time method is not available')

        # Solving the regularized inverse problem
        frf = np.moveaxis(modified_frfs.ordinate, -1, 0)
        if np.any(low_regularization_limit) or np.any(high_regularization_limit) is None:
            s = np.linalg.svd(frf, compute_uv=False)
        if np.any(low_regularization_limit) is None:
            low_regularization_limit = s[:, -1]
        if np.any(high_regularization_limit) is None:
            high_regularization_limit = s[:, 0]

        # The 1:-1 indexing is because we don't evaluate at the first or last frequency, which must be zero
        lambda_values = np.geomspace(low_regularization_limit[1:-1], high_regularization_limit[1:-1], num = number_regularization_values)
        
        frf_inv = np.zeros((number_regularization_values, frf.shape[0], frf.shape[2], frf.shape[1]), dtype=complex)
        for ii in range(frf_inv.shape[1]-2):
            for jj, l in enumerate(lambda_values[:, ii]):
                frf_inv[jj, ii, ...] = pinv_by_tikhonov(frf[ii+1, ...], regularization_parameter=l)

        padded_frequency_domain_data_ord = np.moveaxis(padded_frequency_domain_data.ordinate,-1,-2)[...,np.newaxis]
        forces_frequency_domain = frf_inv[:, np.newaxis, ...]@padded_frequency_domain_data_ord[np.newaxis, ...]
        penalty = norm(forces_frequency_domain[..., 0], axis = -1, ord = 2)**2
        
        # Selecting the forces with "optimal" regularization
        selected_force = np.zeros(forces_frequency_domain.shape[1:-1], dtype=complex)
        optimal_regularization = np.zeros(forces_frequency_domain.shape[1:3], dtype=float)
        # Something is crashing the python kernel, but I don't know what the deal is
        lambda_values = np.moveaxis(lambda_values, 0, -1)
        penalty = np.moveaxis(penalty, 0, -1) 
        for ii in range(forces_frequency_domain.shape[1]):
            residual = norm((padded_frequency_domain_data_ord[ii, ...]-frf[np.newaxis, ...]@forces_frequency_domain[:, ii, ...])[..., 0], axis = -1, ord = 2)**2
            selected_force[ii, 1:-1, :], optimal_regularization[ii, 1:-1] = l_curve_selection(lambda_values, penalty[ii, 1:-1, :], np.moveaxis(residual, 0, -1)[1:-1, :], 
                                                                                              np.moveaxis(forces_frequency_domain, 0, 2)[ii, 1:-1, :, :, 0],
                                                                                              optimality_condition=optimality_condition, l_curve_type=l_curve_type,
                                                                                              curvature_method=curvature_method)

        
        inverse_start_index = np.argmax(modified_frfs[0,0].abscissa >= self._abscissa_[0])
        selected_force[:, :inverse_start_index, :] = 0
        # Reassembling the forces to the time domain
        forces_time_domain_with_padding = irfft(selected_force, axis = -2, norm = 'backward')
        if use_transformation:
            if self._reference_transformation_array_.ndim == 2:    
                reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis, ...])
            elif self._reference_transformation_array_.ndim == 3:
                raise NotImplementedError('A frequency/time dependent response transformation is not implemented for transient problems')
            forces_time_domain_with_padding = (reference_transform@forces_time_domain_with_padding[..., np.newaxis])[..., 0]
        
        pre_pad_length = actual_zero_pad//2
        post_pad_length = actual_zero_pad - pre_pad_length
        
        if time_method == 'single_frame':
            self._force_array_ = forces_time_domain_with_padding[0, pre_pad_length:-post_pad_length, ...]

        elif time_method == 'cola':
            forces_time_domain_with_padding = sdpy.data_array(FunctionTypes.TIME_RESPONSE,
                                                              zero_padded_data.abscissa[...,:1,:],
                                                              np.moveaxis(forces_time_domain_with_padding,-1,-2),
                                                              self._reference_coordinate_[:,np.newaxis])
            
            # Assemble the COLA
            forces_time_domain_with_padding = sdpy.TimeHistoryArray.overlap_and_add(forces_time_domain_with_padding,
                                                                                    overlap_samples = actual_zero_pad + cola_overlap)
            start_index = cola_frame_length - cola_overlap + pre_pad_length
            end_index = start_index + self.transformed_target_response.num_elements
            self._force_array_ = np.moveaxis(forces_time_domain_with_padding.idx_by_el[start_index:end_index].ordinate, 0, -1)
            
            # Compute COLA weighting
            window_fn = get_window(cola_window, cola_frame_length)
            step = cola_frame_length - cola_overlap
            weighting = np.median(sum(window_fn[ii*step:(ii+1)*step] for ii in range(cola_frame_length//step)))
            self._force_array_ /= weighting
        
        self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                      'inverse_method':'Tikhonov regularization',
                                      'time_method':time_method, 
                                      'cola_frame_length':cola_frame_length,
                                      'cola_window':cola_window,
                                      'cola_overlap':cola_overlap,
                                      'zero_pad_length':zero_pad_length,
                                      'regularization_paramter':optimal_regularization,
                                      'l_curve_type':l_curve_type,
                                      'optimality_condition':optimality_condition,
                                      'curvature_method':curvature_method,
                                      'use_transformation':use_transformation})
        
        return self
"""
Deprecated functions/methods

@linear_inverse_processing
def auto_tikhonov_by_l_curve(self,
                                low_regularization_limit = None, 
                                high_regularization_limit = None,
                                number_regularization_values=100,
                                l_curve_type = 'forces',
                                optimality_condition = 'curvature',
                                curvature_method = 'numerical',
                                use_transformation=True,
                                response=None, frf=None,
                                parallel = False,
                                num_jobs = -2):
    
    Performs the inverse source estimation problem with Tikhonov regularization, 
    where the regularization parameter is automatically selected with L-curve 
    methods.

    Parameters
    ----------
    low_regularization_limit : ndarray
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the target frf array.
    high_regularization_limit : ndarray
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object.
    number_regularization_values : int
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits.  
    l_curve_type : str
        The type of L-curve that is used to find the "optimal regularization 
        parameter. The available types are:
            - forces (default) - This L-curve is constructed with the "size" 
            of the forces on the Y-axis and the regularization parameter on the 
            X-axis. 
            - standard - This L-curve is constructed with the residual squared 
            error on the X-axis and the "size" of the forces on the Y-axis. 
    optimality_condition : str
        The method that is used to find an "optimal" regularization parameter.
        The options are:
            - curvature (default) - This method searches for the regularization
            parameter that results in maximum curvature of the L-curve. It is 
            also referred to as the L-curve criterion. 
            - distance - This method searches for the regularization parameter that
            minimizes the distance between the L-curve and a "virtual origin". A 
            virtual origin is used, because the L-curve is scaled and offset to always 
            range from zero to one, in this case.
    curvature_method : std
        The method that is used to compute the curvature of the L-curve, in the 
        case that the curvature is used to find the optimal regularization 
        parameter. The options are:
            - numerical (default) - this method computes the curvature of 
            the L-curve via numerical derivatives
            - cubic_spline - this method fits a cubic spline to the L-curve
            the computes the curvature from the cubic spline (this might 
            perform better if the L-curve isn't "smooth")
    use_transformation : bool
        Whether or not the response and reference transformation from the class 
        definition should be used (which is handled in the "linear_inverse_processing" 
        decorator function). The default is true. 
    response : ndarray
        The preprocessed response data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user. 
    frf : ndarray
        The preprocessed frf data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user.
    parallel : bool
        Whether or not to parallelize the computation using Joblib. The default is 
        False. 
    num_jobs : int
        The number of processors to use when parallelizing the code. The default is 
        -2, which uses all the available processors except one. Refer to the joblib
        documentation for more details. 

    Returns
    -------
    selected_force : ndarray
        An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
        function applies this force to the force property of SourcePathReceiver object.

    Raises
    ------
    ValueError
        If the requested L-curve type is not available.
    ValueError
        If the requested optimality condition is not available.

    Notes
    -----
    Parallelizing generally isn't faster for "small" inverse problems because of the 
    overhead involved in the parallelizing. Some experience has shown that the 
    parallelization adds ~1-1.5 minutes to the computation, but this will depend on 
    the specific computer that is being used.

    All the setting, including the selected regularization parameters, are saved to 
    the "inverse_settings" class property. 

    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
        of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
        vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
        problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
        2000, pp. 119-142.  
    .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
        Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008. 
    
    # Need the removing the trailing axis from the end of the response
    response = response[..., 0]
    if use_transformation==True:
        num_forces = self._transformed_reference_coordinate_.shape[0]
    else:
        num_forces = self._reference_coordinate_.shape[0]

    if low_regularization_limit or high_regularization_limit is None:
        s = np.linalg.svd(frf, compute_uv=False)
    if low_regularization_limit is None:
        low_regularization_limit = s[:, -1]
    if high_regularization_limit is None:
        high_regularization_limit = s[:, 0]

    if parallel==True:
        results = Parallel(n_jobs=int(num_jobs))(delayed(tikhonov_full_path_single_frequency)(frf[ii, ...], 
                                                                                                response[ii, ...], 
                                                                                                low_regularization_limit=low_regularization_limit[ii], 
                                                                                                high_regularization_limit=high_regularization_limit[ii],
                                                                                                number_regularization_values=int(number_regularization_values)) for ii in range(self._abscissa_.shape[0]))
        forces_full_path, regularization_values, residual, penalty = zip(*results)
        # Making sure that the unpacked results are ndarrays
        forces_full_path = np.array(forces_full_path)
        regularization_values = np.array(regularization_values)
        residual = np.array(residual)
        penalty = np.array(penalty)
    elif parallel==False:
        forces_full_path = np.zeros((self._abscissa_.shape[0], number_regularization_values, num_forces), dtype=complex)
        regularization_values = np.zeros((self._abscissa_.shape[0], number_regularization_values), dtype=float) 
        residual = np.zeros((self._abscissa_.shape[0], number_regularization_values), dtype=float) 
        penalty = np.zeros((self._abscissa_.shape[0], number_regularization_values), dtype=float)
        for ii in range(self._abscissa_.shape[0]):
            forces_full_path[ii, ...], regularization_values[ii, ...], residual[ii, ...], penalty[ii, ...] = tikhonov_full_path_single_frequency(frf[ii, ...], 
                                                                                                                                                    response[ii, ...], 
                                                                                                                                                    low_regularization_limit=low_regularization_limit[ii], 
                                                                                                                                                    high_regularization_limit=high_regularization_limit[ii],
                                                                                                                                                    number_regularization_values=int(number_regularization_values))
    selected_force = np.zeros((self._abscissa_.shape[0], num_forces), dtype=complex)
    optimal_regularization = np.zeros(self._abscissa_.shape[0], dtype=float)
    for ii in range(self._abscissa_.shape[0]):
        if l_curve_type=='forces':
            if optimality_condition=='curvature':
                optimal_regularization[ii] = l_curve_criterion(regularization_values[ii, ...], penalty[ii, ...], regularization_values[ii, ...], method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii] = optimal_l_curve_by_distance(regularization_values[ii, ...], penalty[ii, ...], regularization_values[ii, ...])
            else:
                raise ValueError('The selected optimality_condition is not available')
        elif l_curve_type=='standard':
            if optimality_condition=='curvature':
                optimal_regularization[ii] = l_curve_criterion(residual[ii, ...], penalty[ii, ...], regularization_values[ii, ...], method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii] = optimal_l_curve_by_distance(residual[ii, ...], penalty[ii, ...], regularization_values[ii, ...])
            else:
                raise ValueError('The selected optimality_condition is not available')
        else:
            raise ValueError('The selected L-curve type is not available')
        selected_force[ii, ...] = forces_full_path[ii, np.argmin(np.abs(regularization_values[ii, ...]-optimal_regularization[ii])), :]
    
    self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                    'inverse_method':'Tikhonov regularization',
                                    'number_regularization_values_searched':number_regularization_values,
                                    'regularization_parameter':optimal_regularization,
                                    'l_curve_type':l_curve_type,
                                    'optimality_condition':optimality_condition,
                                    'curvature_method':curvature_method,
                                    'use_transformation':use_transformation})

    return selected_force[..., np.newaxis]

@linear_inverse_processing
def auto_truncation_by_l_curve(self,
                                l_curve_type = 'standard',
                                optimality_condition = 'distance',
                                curvature_method = None,
                                use_transformation=True,
                                response=None, frf=None):
    
    Performs the inverse source estimation problem with the truncated singular
    value decomposition (TSVD). The number of singular values to retain in the 
    inverse is automatically selected with L-curve methods
    
    Parameters
    ----------
    l_curve_type : str
        The type of L-curve that is used to find the "optimal regularization 
        parameter. The available types are:
            - forces - This L-curve is constructed with the "size" of the 
            forces on the Y-axis and the regularization parameter on the X-axis. 
            - standard (default) - This L-curve is constructed with the residual 
            squared error on the X-axis and the "size" of the forces on the Y-axis. 
    optimality_condition : str
        The method that is used to find an "optimal" regularization parameter.
        The options are:
            - curvature - This method searches for the regularization parameter 
            that results in maximum curvature of the L-curve. It is also referred 
            to as the L-curve criterion. 
            - distance (default) - This method searches for the regularization 
            parameter that minimizes the distance between the L-curve and a "virtual 
            origin". A virtual origin is used, because the L-curve is scaled and 
            offset to always range from zero to one, in this case.
    curvature_method : std
        The method that is used to compute the curvature of the L-curve, in the 
        case that the curvature is used to find the optimal regularization 
        parameter. The default is None and the options are:
            - numerical - this method computes the curvature of the L-curve via 
            numerical derivatives
            - cubic_spline - this method fits a cubic spline to the L-curve
            the computes the curvature from the cubic spline (this might 
            perform better if the L-curve isn't "smooth")
    use_transformation : bool
        Whether or not the response and reference transformation from the class 
        definition should be used (which is handled in the "linear_inverse_processing" 
        decorator function). The default is true. 
    response : ndarray
        The preprocessed response data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user. 
    frf : ndarray
        The preprocessed frf data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user.

    Returns
    -------
    selected_force : ndarray
        An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
        function applies this force to the force property of SourcePathReceiver object.

    Raises
    ------
    ValueError
        If the requested L-curve type is not available.
    ValueError
        If the requested optimality condition is not available.

    Notes
    -----
    L-curve for the TSVD could be non-smooth and determining the number of singular 
    values to retain via curvature methods could lead to erratic results.  

    All the setting, including the number of singular values to retain, are saved to 
    the "inverse_settings" class property. 

    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
        of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
        vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
        problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
        2000, pp. 119-142.  
    .. [3] L. Reichel and H. Sadok, "A new L-curve for ill-posed problems," Journal of 
        Computational and Applied Mathematics, vol. 219, no. 2, pp. 493-508, 
        2008/10/01/ 2008, doi: https://doi.org/10.1016/j.cam.2007.01.025.    
    
    if use_transformation==True:
        num_forces = self._transformed_reference_coordinate_.shape[0]
    else:
        num_forces = self._reference_coordinate_.shape[0]
    regularization_values=np.arange(num_forces)+1

    forces_full_path = np.zeros((num_forces, self._abscissa_.shape[0], num_forces), dtype=complex)
    for ii in range(num_forces):
        forces_full_path[ii, ...] = (pinv_by_truncation(frf, ii+1)@response)[..., 0]
    penalty = norm(forces_full_path, axis = -1, ord = 2)**2
    residual = norm((response[np.newaxis,  ...] - frf[np.newaxis,  ...]@forces_full_path[..., np.newaxis])[..., 0], axis=-1, ord=2)**2
    
    optimal_regularization = np.zeros(self._abscissa_.shape[0], dtype=float)
    selected_force = np.zeros((self._abscissa_.shape[0], self.reference_coordinate.shape[0]), dtype=complex)
    for ii in range(self._abscissa_.shape[0]):
        if l_curve_type=='forces':
            if optimality_condition=='curvature':
                optimal_regularization[ii] = l_curve_criterion(regularization_values, penalty[:, ii], regularization_values, method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii] = optimal_l_curve_by_distance(regularization_values, penalty[:, ii], regularization_values)
            else:
                raise ValueError('The selected optimality_condition is not available')
        elif l_curve_type=='standard':
            if optimality_condition=='curvature':
                optimal_regularization[ii] = l_curve_criterion(residual[:, ii], penalty[:, ii], regularization_values, method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii] = optimal_l_curve_by_distance(residual[:, ii], penalty[:, ii], regularization_values)
            else:
                raise ValueError('The selected optimality_condition is not available')
        else:
            raise ValueError('The selected L-curve type is not available')
        selected_force[ii, ...] = forces_full_path[int(optimal_regularization[ii])-1, ii, :]
    
    self.inverse_settings.update({'ISE_technique':'auto_truncation_by_l_curve',
                                    'inverse_method':'truncated singular value decomposition',
                                    'number_retained_values':optimal_regularization,
                                    'l_curve_type':l_curve_type,
                                    'optimality_condition':optimality_condition,
                                    'curvature_method':curvature_method,
                                    'use_transformation':use_transformation})

    return selected_force[..., np.newaxis]

@power_inverse_processing
def auto_tikhonov_by_l_curve(self,
                                low_regularization_limit = None, 
                                high_regularization_limit = None,
                                number_regularization_values=100,
                                l_curve_type = 'forces',
                                optimality_condition = 'curvature',
                                curvature_method = 'numerical',
                                use_transformation=True,
                                response=None, frf=None,
                                parallel = False,
                                num_jobs = -2):
    
    Performs the inverse source estimation problem with Tikhonov regularization, 
    where the regularization parameter is automatically selected with L-curve 
    methods.

    Parameters
    ----------
    low_regularization_limit : ndarray
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the target frf array.
    high_regularization_limit : ndarray
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object.
    number_regularization_values : int
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits.  
    l_curve_type : str
        The type of L-curve that is used to find the "optimal regularization 
        parameter. The available types are:
            - forces (default) - This L-curve is constructed with the "size" 
            of the forces on the Y-axis and the regularization parameter on the 
            X-axis. 
            - standard - This L-curve is constructed with the residual squared 
            error on the X-axis and the "size" of the forces on the Y-axis. 
    optimality_condition : str
        The method that is used to find an "optimal" regularization parameter.
        The options are:
            - curvature (default) - This method searches for the regularization
            parameter that results in maximum curvature of the L-curve. It is 
            also referred to as the L-curve criterion. 
            - distance - This method searches for the regularization parameter that
            minimizes the distance between the L-curve and a "virtual origin". A 
            virtual origin is used, because the L-curve is scaled and offset to always 
            range from zero to one, in this case.
    curvature_method : std
        The method that is used to compute the curvature of the L-curve, in the 
        case that the curvature is used to find the optimal regularization 
        parameter. The options are:
            - numerical (default) - this method computes the curvature of 
            the L-curve via numerical derivatives
            - cubic_spline - this method fits a cubic spline to the L-curve
            the computes the curvature from the cubic spline (this might 
            perform better if the L-curve isn't "smooth")
    use_transformation : bool
        Whether or not the response and reference transformation from the class 
        definition should be used (which is handled in the "linear_inverse_processing" 
        decorator function). The default is true. 
    response : ndarray
        The preprocessed response data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user. 
    frf : ndarray
        The preprocessed frf data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user.
    parallel : bool
        Whether or not to parallelize the computation using Joblib. The default is 
        False. 
    num_jobs : int
        The number of processors to use when parallelizing the code. The default is 
        -2, which uses all the available processors except one. Refer to the joblib
        documentation for more details. 

    Returns
    -------
    selected_force : ndarray
        An ndarray of the estimated sources. The "linear_inverse_processing" decorator 
        function applies this force to the force property of SourcePathReceiver object.

    Raises
    ------
    ValueError
        If the requested L-curve type is not available.
    ValueError
        If the requested optimality condition is not available.

    Notes
    -----
    Parallelizing generally isn't faster for "small" inverse problems because of the 
    overhead involved in the parallelizing. Some experience has shown that the 
    parallelization adds ~1-1.5 minutes to the computation, but this will depend on 
    the specific computer that is being used.

    All the setting, including the selected regularization parameters, are saved to 
    the "inverse_settings" class property. 

    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
        of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
        vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
        problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
        2000, pp. 119-142.  
    .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
        Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008. 
    
    if use_transformation==True:
        num_forces = self._transformed_reference_coordinate_.shape[0]
    else:
        num_forces = self._reference_coordinate_.shape[0]

    if low_regularization_limit or high_regularization_limit is None:
        s = np.linalg.svd(frf, compute_uv=False)
    if low_regularization_limit is None:
        low_regularization_limit = s[:, -1]
    if high_regularization_limit is None:
        high_regularization_limit = s[:, 0]

    if parallel==True:
        results = Parallel(n_jobs=int(num_jobs))(delayed(tikhonov_full_path_single_frequency)(frf[ii, ...], 
                                                                                                response[ii, ...], 
                                                                                                low_regularization_limit=low_regularization_limit[ii], 
                                                                                                high_regularization_limit=high_regularization_limit[ii],
                                                                                                number_regularization_values=int(number_regularization_values)) for ii in range(self.abscissa.shape[0]))
        forces_full_path, regularization_values, residual, penalty = zip(*results)
        # Making sure that the unpacked results are ndarrays
        forces_full_path = np.array(forces_full_path)
        regularization_values = np.array(regularization_values)
        residual = np.array(residual)
        penalty = np.array(penalty)
    elif parallel==False:
        forces_full_path = np.zeros((self.abscissa.shape[0], number_regularization_values, num_forces, num_forces), dtype=complex)
        regularization_values = np.zeros((self.abscissa.shape[0], number_regularization_values), dtype=float) 
        residual = np.zeros((self.abscissa.shape[0], number_regularization_values), dtype=float) 
        penalty = np.zeros((self.abscissa.shape[0], number_regularization_values), dtype=float)
        for ii in range(self.abscissa.shape[0]):
            forces_full_path[ii, ...], regularization_values[ii, ...], residual[ii, ...], penalty[ii, ...] = tikhonov_full_path_single_frequency(frf[ii, ...], 
                                                                                                                                                    response[ii, ...], 
                                                                                                                                                    low_regularization_limit=low_regularization_limit[ii], 
                                                                                                                                                    high_regularization_limit=high_regularization_limit[ii],
                                                                                                                                                    number_regularization_values=int(number_regularization_values))
        
    selected_force = np.zeros((self.abscissa.shape[0], num_forces, num_forces), dtype=complex)
    optimal_regularization = np.zeros(self.abscissa.shape[0], dtype=float)
    for ii in range(self.abscissa.shape[0]):
        if l_curve_type=='forces':
            if optimality_condition=='curvature':
                optimal_regularization[ii] = l_curve_criterion(regularization_values[ii, ...], penalty[ii, ...], regularization_values[ii, ...], method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii] = optimal_l_curve_by_distance(regularization_values[ii, ...], penalty[ii, ...], regularization_values[ii, ...])
            else:
                raise ValueError('The selected optimality_condition is not available')
        elif l_curve_type=='standard':
            if optimality_condition=='curvature':
                optimal_regularization[ii] = l_curve_criterion(residual[ii, ...], penalty[ii, ...], regularization_values[ii, ...], method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii] = optimal_l_curve_by_distance(residual[ii, ...], penalty[ii, ...], regularization_values[ii, ...])
            else:
                raise ValueError('The selected optimality_condition is not available')
        else:
            raise ValueError('The selected L-curve type is not available')
        selected_force[ii, ...] = forces_full_path[ii, np.argmin(np.abs(regularization_values[ii, ...]-optimal_regularization[ii])), ...]
    
    self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                    'inverse_method':'Tikhonov regularization',
                                    'number_regularization_values_searched':number_regularization_values,
                                    'regularization_parameter':optimal_regularization,
                                    'l_curve_type':l_curve_type,
                                    'optimality_condition':optimality_condition,
                                    'curvature_method':curvature_method,
                                    'use_transformation':use_transformation})

    return selected_force
"""