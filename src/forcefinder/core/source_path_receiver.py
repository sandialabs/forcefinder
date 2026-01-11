"""
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
"""

import sdynpy as sdpy
from sdynpy.signal_processing.sdynpy_frf_inverse import (frf_inverse,
                                                         pinv_by_truncation)
from sdynpy.core.sdynpy_data import FunctionTypes
from sdynpy.core.sdynpy_coordinate import outer_product, coordinate_array
import numpy as np
from scipy.linalg import norm
from scipy.signal import sosfiltfilt 
from scipy.fft import rfftfreq
from scipy.interpolate import interp1d
from .utilities import (check_frequency_abscissa, compare_sampling_rate, is_cpsd, apply_buzz_method,
                        reduce_drives_condition_not_met)
from .inverse_processing import (linear_inverse_processing, power_inverse_processing, transient_inverse_processing)
from .auto_regularization import (tikhonov_full_path_for_l_curve,
                                  l_curve_optimal_regularization,
                                  l_curve_selection,
                                  select_model_by_information_criterion,
                                  compute_regularized_svd_inv,
                                  compute_regularized_residual_penalty_for_l_curve,
                                  leave_one_out_cv, k_fold_cv)
from .sparse_functions import elastic_net_full_path_all_frequencies_parallel
from ..transient_quality_evaluation.transient_quality_metrics import (preprocess_data_for_quality_metric, 
                                                                      compute_global_rms_error,
                                                                      compute_average_rms_error,
                                                                      compute_time_varying_trac,
                                                                      compute_time_varying_level_error)
from .transient_utilities import attenuate_signal
import cvxpy as cp
from joblib import delayed, Parallel
import warnings
from copy import deepcopy
from typing import Union

class SourcePathReceiver:
    """
    A class to represent a source-path-receiver (SPR) model of a system for MIMO 
    vibration testing or transfer path analysis. It is primarily intended to manage 
    all the book keeping for the complex problem set-up.

    This is the base SPR class that is further defined for specific data types in 
    subclasses.

    Notes
    -----
    The ordinate in the full FRFs and target responses can be different for the ordinate 
    in the training FRFs and responses (depending on the problem set-up).
    """

    def __init__(self, frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, 
                 training_frfs=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Basic set-up for all the SourcePathReceiver (SPR) classes. The FRFs are set-up here, 
        but everything else is set-up in the subclasses. 

        Parameters
        ----------
        frfs : TransferFunctionArray
            The "full" FRFs that define the path of the SPR object.
        target_response
            The measured responses at both the training and validation DOFs that define the 
            receiver of the SPR object.
        force : optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        training_response : optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a training response or training_response_coordinate is not supplied .
        training_response_coordinate : CoordinateArray, optional
            The training response coordinates of teh SPR object, based on the training responses.
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
        
        self._frf_array_=None
        self._target_frf_array_=None
        self._training_frf_array_=None
        self._validation_frf_array_=None
        self._force_array_=None
        self._target_response_array_=None
        self._training_response_array_=None
        self._response_transformation_array_=None
        self._reference_transformation_array_=None
        self._response_coordinate_=None
        self._training_response_coordinate_=None
        self._validation_response_coordinate_=None
        self._target_response_coordinate_=None
        self._reference_coordinate_=None
        self._transformed_response_coordinate_=None
        self._transformed_reference_coordinate_=None
        self._abscissa_=None
        self.inverse_settings = {}

        if not empty:
            if frfs is None:
                raise AttributeError('FRF data is required to initialize a SourcePathReceiver object')
            self.frfs = frfs
            
            if target_response is None and training_response is None:
                raise AttributeError('Response data is required to initialize a SourcePathReceiver object')
            elif target_response is not None:
                self.target_response_coordinate = np.intersect1d(np.unique(self.response_coordinate), np.unique(target_response.response_coordinate))
                self.target_response = target_response
            elif target_response is None and training_response is not None:
                self.target_response_coordinate = np.intersect1d(np.unique(self.response_coordinate), np.unique(training_response.response_coordinate))
                self.target_response = training_response

            # Adding the training response coordinate, training responses and forces, if they are supplied
            # the ordering here is important for the logic to flow correctly for setting the training responses
            if training_response_coordinate is not None and training_response is None:
                self._training_response_coordinate_ = np.unique(training_response_coordinate)
                # the training response setter handles the indexing to the training response coordinate
                self.training_response = target_response
            elif training_response is not None:
                self.training_response = training_response
            else:
                self.training_response_coordinate = self.target_response_coordinate
                self._training_response_array_ = deepcopy(self._target_response_array_)
            
            self.validation_response_coordinate = np.setxor1d(self.target_response_coordinate, self.training_response_coordinate)

            if training_frfs is not None:
                self.training_frfs = training_frfs
            else:
                self.training_frfs = self.frfs[outer_product(self.training_response_coordinate, self.reference_coordinate)]
            
            self._target_frf_array_ = np.moveaxis(self.frfs[outer_product(self._target_response_coordinate_, self._reference_coordinate_)].ordinate, -1, 0)
            self._validation_frf_array_ = np.moveaxis(self.frfs[outer_product(self._validation_response_coordinate_, self._reference_coordinate_)].ordinate, -1, 0)

            if force is not None:
                self.force = force
            
            if response_transformation is None:
                self._response_transformation_array_ = np.eye(self._training_response_coordinate_.shape[0])
                self._transformed_response_coordinate_ = deepcopy(self._training_response_coordinate_)
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
        return repr('{:} object with {:} reference coordinates, {:} target response coordinates, and {:} training response coordinates'.format(self.__class__.__name__, 
                                                                                                                                      str(self.reference_coordinate.size), 
                                                                                                                                      str(self.target_response_coordinate.size), 
                                                                                                                                      str(self.training_response_coordinate.size)))

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
                 target_response=self._target_response_array_,
                 frf=self._frf_array_,
                 force=self._force_array_,
                 training_response=self._training_response_array_,
                 target_frf=self._target_frf_array_,
                 training_frf=self._training_frf_array_,
                 validation_frf=self._validation_frf_array_,
                 response_transformation=self._response_transformation_array_,
                 reference_transformation=self._reference_transformation_array_,
                 response_coordinate=self._response_coordinate_.string_array(),
                 training_response_coordinate=self._training_response_coordinate_.string_array(),
                 validation_response_coordinate=self._validation_response_coordinate_.string_array(),
                 target_response_coordinate=self._target_response_coordinate_.string_array(),
                 reference_coordinate=self._reference_coordinate_.string_array(),
                 transformed_response_coordinate=self._transformed_response_coordinate_.string_array(),
                 transformed_reference_coordinate=self._transformed_reference_coordinate_.string_array(),
                 abscissa=self._abscissa_,
                 inverse_settings=np.array(self.inverse_settings, dtype=object)) 
    
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
        spr_object._target_response_array_ = loaded_spr['target_response']
        spr_object._frf_array_=loaded_spr['frf']
        spr_object._force_array_ = loaded_spr['force'] if np.all(loaded_spr['force'] != np.array(None)) else None
        spr_object._training_response_array_ = loaded_spr['training_response']
        spr_object._target_frf_array_ = loaded_spr['target_frf']
        spr_object._training_frf_array_ = loaded_spr['training_frf']
        spr_object._validation_frf_array_ = loaded_spr['validation_frf']
        spr_object._response_transformation_array_ = loaded_spr['response_transformation']
        spr_object._reference_transformation_array_ = loaded_spr['reference_transformation']
        spr_object._training_response_coordinate_ = coordinate_array(string_array=loaded_spr['training_response_coordinate'])
        spr_object._validation_response_coordinate_ = coordinate_array(string_array=loaded_spr['validation_response_coordinate'])
        spr_object._response_coordinate_ = coordinate_array(string_array=loaded_spr['response_coordinate'])
        spr_object._target_response_coordinate_ = coordinate_array(string_array=loaded_spr['target_response_coordinate'])
        spr_object._reference_coordinate_ = coordinate_array(string_array=loaded_spr['reference_coordinate'])
        spr_object._transformed_response_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_response_coordinate'])
        spr_object._transformed_reference_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_reference_coordinate'])
        spr_object._abscissa_ = loaded_spr['abscissa']
        spr_object.inverse_settings = loaded_spr['inverse_settings'].item()
        return spr_object

    @property
    def frfs(self):
        """
        This produces a copy of the FRFs as a SDynPy NDDataArray and any modifications to the copy will not modify the original object.
        """
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(self._frf_array_, 0, -1), outer_product(self._response_coordinate_, self._reference_coordinate_))
    
    @frfs.setter
    def frfs(self, data_array):
        if self._frf_array_ is not None:
            raise AttributeError('The FRFs of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.TransferFunctionArray):
            raise TypeError('The FRFs must be a SDynPy TransferFunctionArray')
        
        data_array = data_array.reshape_to_matrix()

        self.response_coordinate = data_array[:, 0].response_coordinate
        self.reference_coordinate = data_array[0, :].reference_coordinate
        self._abscissa_ = data_array[0, 0].abscissa
        self._frf_array_ = np.moveaxis(data_array.ordinate, -1, 0)
    
    @property
    def training_frfs(self):
        """
        This produces a copy of the FRFs for the training_response_coordinate as a SDynPy 
        TransferFunctionArray. Any modifications to the copy will not modify the original object.
        """
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(self._training_frf_array_, 0, -1), 
                               outer_product(self._training_response_coordinate_, self._reference_coordinate_))

    @training_frfs.setter
    def training_frfs(self, data_array):
        if self._training_frf_array_ is not None:
            raise AttributeError('The training FRFs of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.TransferFunctionArray):
            raise TypeError('The training FRFs must be a SDynPy TransferFunctionArray')
        data_array = data_array.reshape_to_matrix()
        if not np.all(data_array[:, 0].response_coordinate==self._training_response_coordinate_):
            raise ValueError('The training FRF response DOFs do not match the training response DOFs of the SourcePathReceiver object')
        if not np.all(data_array[0, :].reference_coordinate==self._reference_coordinate_):
            raise ValueError('The training FRF reference DOFs do not match the SourcePathReceiver object')
        check_frequency_abscissa(data_array, self._abscissa_)
        self._training_frf_array_ = np.moveaxis(data_array.ordinate, -1, 0)

    @property
    def target_frfs(self):
        """
        This produces a copy of the FRFs for the target_response_coordinate as a SDynPy 
        TransferFunctionArray. Any modifications to the copy will not modify the original object.
        """
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(self._target_frf_array_, 0, -1), 
                               outer_product(self._target_response_coordinate_, self._reference_coordinate_))
    
    @property
    def validation_frfs(self):
        """
        This produces a copy of the FRFs for the target_response_coordinate as a SDynPy 
        TransferFunctionArray. Any modifications to the copy will not modify the original object.
        """
        return sdpy.data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.abscissa, np.moveaxis(self._validation_frf_array_, 0, -1), 
                               outer_product(self._validation_response_coordinate_, self._reference_coordinate_))

    @property
    def transformed_training_frfs(self):
        """
        the training FRFs with the transformations applied (i.e., what is used in the inverse problem).
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        if self._reference_transformation_array_.ndim == 2:        
            reference_transform = np.linalg.pinv(self._reference_transformation_array_[np.newaxis, ...])
        elif self._reference_transformation_array_.ndim == 3:        
            reference_transform = np.linalg.pinv(self._reference_transformation_array_)
        transformed_frf = response_transform@self._training_frf_array_@reference_transform
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
    def target_response_coordinate(self):
        return self._target_response_coordinate_
    
    @target_response_coordinate.setter
    def target_response_coordinate(self, coordinate_array):
        if self._target_response_coordinate_ is not None:
            raise AttributeError('The target response coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._target_response_coordinate_ = coordinate_array
    
    @property
    def reference_coordinate(self):
        return self._reference_coordinate_

    @reference_coordinate.setter
    def reference_coordinate(self, coordinate_array):
        if self._reference_coordinate_ is not None:
            raise AttributeError('The reference coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._reference_coordinate_ = coordinate_array

    @property
    def training_response_coordinate(self):
        return self._training_response_coordinate_
    
    @training_response_coordinate.setter
    def training_response_coordinate(self, training_response_coordinate):
        if not np.all(np.isin(training_response_coordinate, self.target_response_coordinate)):
            raise ValueError('Training response {:} is not in the SPR model'.format(training_response_coordinate[~np.isin(training_response_coordinate, self.target_response_coordinate)].string_array()))
        if self._training_response_coordinate_ is not None:
            raise AttributeError('The training response coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._training_response_coordinate_ = np.unique(training_response_coordinate)  

    @property
    def validation_response_coordinate(self):
        return self._validation_response_coordinate_

    @validation_response_coordinate.setter
    def validation_response_coordinate(self, coordinate_array):
        if self._validation_response_coordinate_ is not None:
            raise AttributeError('The validation response coordinate for a SourcePathReceiver object cannot be reset after it is initialized')
        self._validation_response_coordinate_ = coordinate_array

    @property
    def response_transformation(self):
        if self._response_transformation_array_ is None:
            raise AttributeError('A response transformation was not defined for this object')
        return sdpy.matrix(self._response_transformation_array_, self._transformed_response_coordinate_, self._training_response_coordinate_)
    
    @response_transformation.setter
    def response_transformation(self, transformation_matrix):
        if not isinstance(transformation_matrix, sdpy.Matrix):
            raise TypeError('The response transformation must be defined as a SDynPy Matrix')
        self._response_transformation_array_ = transformation_matrix[np.unique(transformation_matrix.row_coordinate), self._training_response_coordinate_]
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
    def target_response(self):
        pass
    @target_response.setter
    def target_response(self):
        pass
    @property
    def validation_response(self):
        pass
    @property
    def force(self):
        pass
    @force.setter
    def force(self):
        pass
    @property
    def training_response(self):
        pass
    @training_response.setter
    def training_response(self):
        pass
    @property
    def transformed_training_response(self):
        pass
    @property
    def predicted_response(self):
        pass
    @property
    def reconstructed_target_response(self):
        pass
    @property
    def reconstructed_validation_response(self):
        pass
    @property
    def transformed_reconstructed_response(self):
        pass
    @property
    def transformed_force(self):
        pass

    def apply_response_weighting(self, physical_response_weighting = None,
                                 transformed_response_weighting = None):
        """
        Applys a weighting to the rows or columns of the response 
        transformation array.

        Parameters
        ----------
        physical_response_weighting : ndarray, optional
            The weighting to be applied to the physical response coordinates
            (i.e., the columns) of the response transformation array. The 
            weightings should be ordered the same as the "training_response_coordinate"
            of the SPR object.
        transformed_response_weighting : ndarray, optional
            The weighting to be applied to the transformed response 
            coordinates (i.e., the rows) of the response transformation array.
            The weightings should be ordered the same as the "transformed_response_coordinate"
            of the SPR object.
            
        Returns
        -------
        self : SourcePathReceiver
            The SourcePathReceiver object with the response transformation that 
            has been updated

        Raises
        ------
        ValueError
            If any of the following occurs:

                - If the number of elements in the physical response weighting array
                does not match the number of training response coordinates.

                - If the number of elements in the transformed response weighting array
                does not match the number of transformed response coordinates.

                - If neither a physical or transformed response weighting is supplied.

        """
        if physical_response_weighting is not None:
            if physical_response_weighting.shape != self._training_response_coordinate_.shape:
                raise ValueError('The number of elements in the physical response weighting array does not match the number of training response coordinates')
            self._response_transformation_array_ *= physical_response_weighting.astype(float)
        if transformed_response_weighting is not None:
            if transformed_response_weighting.shape != self._transformed_response_coordinate_.shape:
                raise ValueError('The number of elements in the transformed response weighting array does not match the number of transformed response coordinates')
            self._response_transformation_array_ *= transformed_response_weighting.astype(float)[:,np.newaxis]
        if physical_response_weighting is None and transformed_response_weighting is None:
            raise ValueError('Either a physical or transformed response weighting must be supplied as a keyword argument')
        return self
    
    def apply_reference_weighting(self, physical_reference_weighting = None,
                                  transformed_reference_weighting = None):
        """
        Applys a weighting to the rows or columns of the reference 
        transformation array.

        Parameters
        ----------
        physical_reference_weighting : ndarray, optional
            The weighting to be applied to the physical reference coordinates
            (i.e., the columns) of the reference transformation array. The 
            weightings should be ordered the same as the "reference_coordinate"
            of the SPR object.
        transformed_reference_weighting : ndarray, optional
            The weighting to be applied to the transformed reference 
            coordinates (i.e., the rows) of the reference transformation array.
            The weightings should be ordered the same as the "transformed_reference_coordinate"
            of the SPR object.
            
        Returns
        -------
        self : SourcePathReceiver
            The SourcePathReceiver object with the reference transformation that 
            has been updated

        Raises
        ------
        ValueError
            If an of the following occurs:

                - If the number of elements in the physical reference weighting array
                does not match the number of reference coordinates.

                - If the number of elements in the transformed reference weighting array
                does not match the number of transformed reference coordinates.

                - If neither a physical or transformed reference weighting is supplied.

        """
        if physical_reference_weighting is not None:
            if physical_reference_weighting.shape != self._reference_coordinate_.shape:
                raise ValueError('The number of elements in the physical reference weighting array does not match the number of reference coordinates')
            self._reference_transformation_array_ *= physical_reference_weighting.astype(float)
        if transformed_reference_weighting is not None:
            if transformed_reference_weighting.shape != self._transformed_reference_coordinate_.shape:
                raise ValueError('The number of elements in the transformed reference weighting array does not match the number of transformed reference coordinates')
            self._reference_transformation_array_ *= transformed_reference_weighting.astype(float)[:,np.newaxis]
        if physical_reference_weighting is None and transformed_reference_weighting is None:
            raise ValueError('Either a physical or transformed reference weighting must be supplied as a keyword argument')
        return self

    def set_response_transformation_by_normalization(self, method = 'std', 
                                                     normalize_transformed_coordinate = True, 
                                                     reset_transformation = False):
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
        normalized_transformed_coordinate : bool, optional
            Whether the normalization should be applied to the transformed
            or physical coordinate (i.e., the rows or columns of the transfomation
            array) of the SPR object. The default is True.
        reset_transformation : bool, optional
            Whether the normalization should replace the pre-existing transformation
            array in the SPR object or if the normalization should be applied 
            to the transformation array (similar to a frequency dependent response
            weighting).
        
        Returns
        -------
        self : SourcePathReceiver 
            The SourcePathReceiver object with the response transformation that
            has been updated
        """
        if normalize_transformed_coordinate is True and reset_transformation is True:
            raise ValueError('The normalization cannot be applied to the transformed coordinate if the transformation is being reset')

        # Get the FRF ordinate, depending on the selected option
        if normalize_transformed_coordinate:
            frf_ordinate = np.moveaxis(self.training_frfs.ordinate,-1,0)
        else:
            frf_ordinate = np.moveaxis(self.transformed_training_frfs.ordinate,-1,0)

        # Compute the statistic for the normalization
        if method == 'std':
            res_statistic = np.std(frf_ordinate, axis=-1)
        else:
            raise ValueError('The selected statistical quantity is not available for response normalization')
        
        if np.any(res_statistic==0):
            res_statistic[np.where(res_statistic==0)] = 1 

        # Compute the updated transformation array with the normalization applied
        num_training_coordinate = self._training_response_coordinate_.shape[0]
        response_transformation_array = np.broadcast_to(np.eye(num_training_coordinate)[np.newaxis, ...], (self.abscissa.shape[0], num_training_coordinate, num_training_coordinate))/res_statistic[..., np.newaxis]
        if reset_transformation:
            self.response_transformation = sdpy.matrix(response_transformation_array, self.training_response_coordinate, self.training_response_coordinate)
        else:
            if normalize_transformed_coordinate:
                self._response_transformation_array_ = response_transformation_array@self._response_transformation_array_
            else:
                self._response_transformation_array_ = self._response_transformation_array_@response_transformation_array
        return self
    
    def set_reference_transformation_by_normalization(self, method = 'std', 
                                                      normalize_transformed_coordinate = True, 
                                                      reset_transformation = False):
        """
        Sets the reference transformation matrix such that it will "normalize"
        the references based on a statistical quantity from the columns of the 
        FRF matrix.

        Parameters
        ----------
        method : str, optional
            A string defining the statistical quantity that will be used 
            for the reference normalization. Currently, only the standard
            deviation is supported.
        normalized_transformed_coordinate : bool, optional
            Whether the normalization should be applied to the transformed
            or physical coordinate (i.e., the rows or columns of the transfomation
            array) of the SPR object. The default is True.
        reset_transformation : bool, optional
            Whether the normalization should replace the pre-existing transformation
            array in the SPR object or if the normalization should be applied 
            to the transformation array (similar to a frequency dependent reference
            weighting).

        Returns
        -------
        self : SourcePathReceiver 
            The SourcePathReceiver object with the reference transformation that
            has been updated
        """
        if normalize_transformed_coordinate is True and reset_transformation is True:
            raise ValueError('The normalization cannot be applied to the transformed coordinate if the transformation is being reset')

        # Get the FRF ordinate, depending on the selected option
        if normalize_transformed_coordinate:
            frf_ordinate = np.moveaxis(self.training_frfs.ordinate,-1,0)
        else:
            frf_ordinate = np.moveaxis(self.transformed_training_frfs.ordinate,-1,0)

        # Compute the statistic for the normalization
        if method == 'std':
            ref_statistic = np.std(frf_ordinate, axis=-2)
        else:
            raise ValueError('The selected statistical quantity is not available for reference normalization')
        
        if np.any(ref_statistic==0):
            ref_statistic[np.where(ref_statistic==0)] = 1

        # Compute the updated transformation array with the normalization applied
        num_reference_coordinate = self._reference_coordinate_.shape[0]
        reference_transformation_array = np.broadcast_to(np.eye(num_reference_coordinate)[np.newaxis, ...], (self.abscissa.shape[0], num_reference_coordinate, num_reference_coordinate))/ref_statistic[..., np.newaxis]
        if reset_transformation:
            self.reference_transformation = sdpy.matrix(reference_transformation_array, self.reference_coordinate, self.reference_coordinate)
        else:
            if normalize_transformed_coordinate:
                self._reference_transformation_array_ = reference_transformation_array@self._reference_transformation_array_
            else:
                self._reference_transformation_array_ = self._reference_transformation_array_@reference_transformation_array
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
        self.response_transformation = sdpy.matrix(np.eye(self.training_response_coordinate.shape[0]), self.training_response_coordinate, self.training_response_coordinate)
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
        if work_object._target_response_array_ is not None:
            work_object._target_response_array_ = work_object._target_response_array_[abscissa_indices, ...]
        if work_object._frf_array_ is not None:
            work_object._frf_array_ = work_object._frf_array_[abscissa_indices, ...]
        if work_object._target_frf_array_ is not None:
            work_object._target_frf_array_ = work_object._target_frf_array_[abscissa_indices, ...]
        if work_object._force_array_ is not None:
            work_object._force_array_ = work_object._force_array_[abscissa_indices, ...]
        if work_object._training_response_array_ is not None:
            work_object._training_response_array_ = work_object._training_response_array_[abscissa_indices, ...]
        if work_object._training_frf_array_ is not None:
            work_object._training_frf_array_ = work_object._training_frf_array_[abscissa_indices, ...]
        if work_object._validation_frf_array_ is not None:
            work_object._validation_frf_array_ = work_object._validation_frf_array_[abscissa_indices, ...]
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
    A subclass to represent a source-path-receiver (SPR) model with linear spectra 
    (i.e., ffts) for the responses or forces.

    Notes
    -----
    The "linear" term in the class name stands for the linear units in the response and
    force spectra.
    """

    def __init__(self, frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, 
                 training_frfs=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Parameters
        ----------
        target_response : SpectrumArray
            The measured responses at both the training and validation DOFs that define the 
            receiver of the SPR object.
        force : SpectrumArray, optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        training_response : SpectrumArray, optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a training response or training_response_coordinate is not supplied .

        Notes
        -----
        Much of the initialization is inherited from the base SourcePathReceiver class.
        """
        # Inheriting the initial set-up from the super class
        super().__init__(frfs=frfs, target_response=target_response, force=force, training_response=training_response, training_response_coordinate=training_response_coordinate, 
                         training_frfs=training_frfs, response_transformation=response_transformation, reference_transformation=reference_transformation, empty=empty)

    @property
    def target_response(self):
        if self._target_response_array_ is None:
            raise AttributeError('A target response array was not defined for this object')
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(self._target_response_array_, 0, -1), 
                               self._target_response_coordinate_[..., np.newaxis])
    
    @target_response.setter
    def target_response(self, data_array):
        if self._target_response_array_ is not None:
            raise AttributeError('The target responses of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
            raise TypeError('The target response must be a SDynPy SpectrumArray')
        check_frequency_abscissa(data_array, self.abscissa)
        self._target_response_array_ = np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0)

    @property
    def validation_response(self):
        if self._target_response_array_ is None:
            raise AttributeError('A target response array was not defined for this object')
        return self.target_response[self._validation_response_coordinate_[..., np.newaxis]]

    @property
    def force(self):
        if self._force_array_ is None:
            raise AttributeError('A force array is not defined for this object')
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(self._force_array_, 0, -1), 
                               self._reference_coordinate_[..., np.newaxis])
    
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
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(transformed_force_array, 0, -1), 
                               self._transformed_reference_coordinate_[..., np.newaxis])

    @property
    def training_response(self):
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(self._training_response_array_, 0, -1), 
                               self._training_response_coordinate_[..., np.newaxis])
    
    @training_response.setter
    def training_response(self, data_array):
        if self._training_response_array_ is not None:
            raise AttributeError('The training responses of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
            raise TypeError('The training response must be a SDynPy SpectrumArray')
        if not np.all(np.isin(data_array.response_coordinate, self.target_response_coordinate)):
            raise ValueError('Training response {:} is not a target response coordinate in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self._target_response_coordinate_)].string_array()))
        if self._training_response_coordinate_ is not None and not np.all(np.isin(self._training_response_coordinate_, data_array.response_coordinate)):
            raise ValueError('Training response {:} is not available in the supplied data'.format(self._training_response_coordinate_[~np.isin(self._training_response_coordinate_, data_array.response_coordinate)].string_array()))
        check_frequency_abscissa(data_array, self._abscissa_)
        # The numpy unique is used when setting the coordinate to make sure that the DOF ordering
        # in the training_response_array matches the other data.
        if self._training_response_coordinate_ is None:
            self.training_response_coordinate = np.unique(data_array.response_coordinate)
        self._training_response_array_ = np.moveaxis(data_array[self._training_response_coordinate_[..., np.newaxis]].ordinate, -1, 0) 

    @property
    def transformed_training_response(self):
        """
        The training response with the transformation applied (i.e., what is used in the inverse problem).
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        transformed_response = (response_transform@self._training_response_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(transformed_response, 0, -1), 
                               self._transformed_response_coordinate_[..., np.newaxis])
    @property
    def predicted_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        predicted_response = (self._frf_array_@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(predicted_response, 0, -1), 
                               self._response_coordinate_[..., np.newaxis])

    @property
    def reconstructed_target_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        reconstructed_target_response = (self._target_frf_array_@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(reconstructed_target_response, 0, -1), 
                               self._target_response_coordinate_[..., np.newaxis])
    
    @property
    def reconstructed_validation_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        reconstructed_validation_response = (self._validation_frf_array_@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(reconstructed_validation_response, 0, -1), 
                               self._validation_response_coordinate_[..., np.newaxis])

    @property
    def transformed_reconstructed_response(self):
        """
        The reconstructed response (at the training coordinates) with the transformation applied (i.e., what
        was used in the inverse problem).
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        reconstructed_response = (response_transform@self._training_frf_array_@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(reconstructed_response, 0, -1), 
                               self._transformed_response_coordinate_[..., np.newaxis])

    def predicted_response_specific_dofs(self, response_dofs):
        """
        Computes the predicted response for specific response DOFs in the SPR object.

        Parameters
        ----------
        response_dofs : CoordinateArray
            The response DOFs to compute the response for. It should be shaped as a 1d array. 

        Returns
        -------
        predicted_response : SpectrumArray
            The predicted response at the desired response DOFs (in the order they were 
            supplied). 
        
        Raises
        ------
        AttributeError
            If there are not forces in the SPR object.
        ValueError
            If any of the following occurs:

                - If any of the selected response DOFs are not in the response_coordinate of 
                the SPR object. 

                - If the response_dofs is not a 1d array.

        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so predicted responses cannot be computed')
        if len(response_dofs.shape) != 1:
            raise ValueError('The supplied response DOFs must be a 1D array')
        if np.any(~np.isin(response_dofs, self._response_coordinate_)):
            raise ValueError('All the supplied response DOFs are not included in the SPR object')
        
        prediction_frfs = self.frfs[sdpy.coordinate.outer_product(response_dofs, self.reference_coordinate)]
        prediction_frfs = np.moveaxis(prediction_frfs.ordinate, -1, 0)

        predicted_response = (prediction_frfs@self._force_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.SPECTRUM, self.abscissa, np.moveaxis(predicted_response, 0, -1), 
                               response_dofs[..., np.newaxis])

    def global_asd_error(self, 
                         channel_set='training'):
        """
        Computes the global ASD error in dB of the reconstructed response, 
        per the procedure in MIL-STD 810H.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:

                - training (default)
                    This compares the responses for the transformed training DOFs.

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

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
        if channel_set == 'training':
            truth = (np.abs(self.transformed_training_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'validation':
            truth = (np.abs(self.validation_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_validation_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'target':
            truth = (np.abs(self.target_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_target_response.ordinate)**2)/self.abscissa_spacing
        else:
            raise ValueError('Selected channel set is not available')
        
        weights = (truth**2)/(norm(truth, axis=0, ord=2)**2)
        asd_error = 10*np.log10(reconstructed/truth)
        global_asd_error = np.sum(asd_error*weights, axis=0)
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, global_asd_error, coordinate_array(node=1, direction=1))
    
    def average_asd_error(self, 
                          channel_set='training'):
        """
        Computes the DOF averaged ASD error in dB of the reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:

                - training (default)
                    This compares the responses for the transformed training DOFs.

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object.

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
        if channel_set == 'training':
            truth = (np.abs(self.transformed_training_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'validation':
            truth = (np.abs(self.validation_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_validation_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'target':
            truth = (np.abs(self.target_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_target_response.ordinate)**2)/self.abscissa_spacing
        else:
            raise ValueError('Selected channel set is not available')
        
        asd_error = 10*np.log10(reconstructed/truth)
        average_asd_error = 10*np.log10(np.average(10**(asd_error/10), axis=0))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, average_asd_error, coordinate_array(node=1, direction=1))
    
    def rms_asd_error(self, 
                      channel_set='training'):
        """
        Computes the root mean square (RMS) of the DOF-DOF ASD dB error of the 
        reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:

                - training (default)
                    This compares the responses for the transformed training DOFs.

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

        Returns
        -------
        PowerSpectralDensityArray
            Returns a spectrum array of the RMS ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.   

        Notes
        -----
        Computes the ASD from the spectrum response by squaring the absolute value
        of the spectrum and dividing by the SPR object abscissa spacing.           
        """
        if channel_set == 'training':
            truth = (np.abs(self.transformed_training_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'validation':
            truth = (np.abs(self.validation_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_validation_response.ordinate)**2)/self.abscissa_spacing
        elif channel_set == 'target':
            truth = (np.abs(self.target_response.ordinate)**2)/self.abscissa_spacing
            reconstructed = (np.abs(self.reconstructed_target_response.ordinate)**2)/self.abscissa_spacing
        else:
            raise ValueError('Selected channel set is not available')
                
        asd_error = 10*np.log10(reconstructed/truth)
        rms_asd_error = np.sqrt(np.mean(asd_error**2, axis=0))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, rms_asd_error, coordinate_array(node=1, direction=1))

    def error_summary(self,
                      channel_set='training',
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

                - training (default)
                    This compares the responses for the transformed training DOFs.    

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object.

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
        if channel_set == 'training':
            truth = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.transformed_training_response.ordinate)**2)/self.abscissa_spacing, self._transformed_response_coordinate_[..., np.newaxis])

            reconstructed = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.transformed_reconstructed_response.ordinate)**2)/self.abscissa_spacing, self._transformed_response_coordinate_[..., np.newaxis])

            return_data = truth.error_summary(figure_kwargs=figure_kwargs, 
                                              linewidth=linewidth, 
                                              plot_kwargs=plot_kwargs,
                                              cpsd_matrices=reconstructed)
        elif channel_set == 'validation':           
            truth = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.validation_response.ordinate)**2)/self.abscissa_spacing, 
                      self._validation_response_coordinate_[..., np.newaxis])

            reconstructed = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.reconstructed_validation_response.ordinate)**2)/self.abscissa_spacing, 
                      self._validation_response_coordinate_[..., np.newaxis])
            
            return_data = truth.error_summary(figure_kwargs=figure_kwargs, 
                                              linewidth=linewidth, 
                                              plot_kwargs=plot_kwargs,
                                              cpsd_matrices=reconstructed)
        elif channel_set == 'target':
            truth = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.target_response.ordinate)**2)/self.abscissa_spacing, self._target_response_coordinate_[..., np.newaxis])

            reconstructed = sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, 
                      (np.abs(self.reconstructed_target_response.ordinate)**2)/self.abscissa_spacing, 
                      self._target_response_coordinate_[..., np.newaxis])

            return_data = truth.error_summary(figure_kwargs=figure_kwargs, 
                                              linewidth=linewidth, 
                                              plot_kwargs=plot_kwargs,
                                              cpsd_matrices=reconstructed)
        else:
            raise ValueError('Selected channel set is not available')
        
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
        The "linear_inverse_processing" decorator function pre and post processes the training
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
                                 l_curve_type = 'standard',
                                 optimality_condition = 'curvature',
                                 use_transformation=True,
                                 response=None, frf=None):
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
            value of the training frf array.
        high_regularization_limit : ndarray
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the largest singular
            value of the training frf array.
        number_regularization_values : int
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:

                - forces
                    This L-curve is constructed with the "size" of the forces on 
                    the Y-axis and the regularization parameter on the X-axis. 

                - standard (default)
                    This L-curve is constructed with the residual squared error on 
                    the X-axis and the "size" of the forces on the Y-axis. 

        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
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
        All the settings, including the selected regularization parameters, are saved to 
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
        regularization_values, residual, penalty = tikhonov_full_path_for_l_curve(frf, response[...,0], # Need the removing the trailing axis from the end of the response
                                                                                  low_regularization_limit=low_regularization_limit, 
                                                                                  high_regularization_limit=high_regularization_limit,
                                                                                  number_regularization_values=number_regularization_values)
        
        optimal_regularization = l_curve_optimal_regularization(regularization_values, penalty, residual, 
                                                                l_curve_type=l_curve_type,
                                                                optimality_condition=optimality_condition)
        
        regularized_frf_inverse = frf_inverse(frf, method='tikhonov', regularization_parameter=optimal_regularization)
        regularized_force = regularized_frf_inverse@response

        self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                      'inverse_method':'Tikhonov regularization',
                                      'number_regularization_values_searched':number_regularization_values,
                                      'regularization_parameter':optimal_regularization,
                                      'l_curve_type':l_curve_type,
                                      'optimality_condition':optimality_condition,
                                      'use_transformation':use_transformation})
        
        return regularized_force

    @linear_inverse_processing
    def auto_tikhonov_by_cv_rse(self,
                                low_regularization_limit = None, 
                                high_regularization_limit = None,
                                number_regularization_values=100,
                                cross_validation_type='loocv',
                                number_folds=None,
                                use_transformation=True,
                                response=None, frf=None):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with cross 
        validation, where the residual squared error is use as the metric to evaluate
        the quality of fit.

        Parameters
        ----------
        low_regularization_limit : ndarray, optional
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the smallest singular
            value of the training frf array.
        high_regularization_limit : ndarray, optional
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the largest singular
            value of the training frf array.
        number_regularization_values : int, optional
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        cross_validation_type : str, optional
            The cross validation method to use. The available options are:

                - loocv (default)
                    Leave one out cross validation.

                - k-fold
                    K fold cross validation.

        number_folds : int
            The number of folds to use in the k fold cross validation. The number of 
            response DOFs must be evenly divisible by the number of folds.
        use_transformation : bool, optional
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
        All the settings, including the selected regularization parameters, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
        .. [1] D. M. Allen, "The Relationship between Variable Selection and Data Agumentation 
               and a Method for Prediction," Technometrics, vol. 16, no. 1, pp. 125-127, 1974, 
               doi: 10.2307/1267500.
        .. [2] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: 
               Data Mining, Inference, and Prediction, 2nd Edition ed. New York: Springer New York, 
               2017.
        """
        if cross_validation_type == 'loocv':
            regularization_values, mse = leave_one_out_cv(frf, response[...,0], 
                                                          low_regularization_limit=low_regularization_limit,
                                                          high_regularization_limit=high_regularization_limit, 
                                                          number_regularization_values=number_regularization_values)

            mse = np.sum(mse, axis=1)
        elif cross_validation_type == 'k-fold':
            regularization_values, mse = k_fold_cv(frf, response[...,0], 
                                                   low_regularization_limit=low_regularization_limit,
                                                   high_regularization_limit=high_regularization_limit, 
                                                   number_regularization_values=number_regularization_values, 
                                                   number_folds=number_folds)

            mse = np.sum(mse, axis=1)
        else:
            raise NotImplementedError('The selected cross validation method has not been implemented yet')
        
        optimal_regularization = regularization_values[np.argmin(mse, axis=0),np.arange(regularization_values.shape[1])]

        regularized_frf_inverse = frf_inverse(frf, method='tikhonov', regularization_parameter=optimal_regularization)
        regularized_force = regularized_frf_inverse@response

        self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_cv_mse',
                                      'inverse_method':'Tikhonov regularization',
                                      'number_regularization_values_searched':number_regularization_values,
                                      'regularization_parameter':optimal_regularization,
                                      'cross_validation_type':cross_validation_type,
                                      'use_transformation':use_transformation})
        
        return regularized_force

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

                - curvature
                    This method searches for the regularization parameter that results 
                    in maximum curvature of the L-curve. It is also referred to as the 
                    L-curve criterion. 

                - distance (default)
                This method searches for the regularization parameter that minimizes 
                the distance between the L-curve and a "virtual origin". A virtual 
                origin is used, because the L-curve is scaled and offset to always range 
                from zero to one, in this case.

        curvature_method : std
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The default is None and the options are:

                - numerical
                    This method computes the curvature of the L-curve via numerical 
                    derivatives.

                - cubic_spline 
                    This method fits a cubic spline to the L-curve the computes the 
                    curvature from the cubic spline (this might perform better if the 
                    L-curve isn't "smooth").
    
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
                                                                   curvature_method=curvature_method)
        
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

                - 'BIC'
                    The Bayesian information criterion

                - 'AIC'
                    The Akaike information criterion

                - 'AICC' (default)
                    The corrected Akaike information criterion

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
    A subclass to represent a source-path-receiver (SPR) model with power spectra
    for the responses and forces.

    Notes
    -----
    The "power" term in the class name stands for the power units (i.e., units squared) in the 
    response and force spectra.
    """

    def __init__(self, frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, 
                 training_frfs=None, buzz_cpsd=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Parameters
        ----------
        target_response : PowerSpectralDensityArray
            The measured responses at both the training and validation DOFs that define the 
            receiver of the SPR object.
        force : PowerSpectralDensityArray, optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        training_response : PowerSpectralDensityArray, optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a training response or training_response_coordinate is not supplied.
        buzz_cpsd : PowerSpectralDensityArray, optional
            The cpsd matrix from the system ID matrix to use the so-called "buzz method"
            in the inverse source estimation. Defaults to None. 

        Notes
        -----
        Much of the initialization is inherited from the base SourcePathReceiver class.
        """
        # Inheriting the initial set-up from the super class
        super().__init__(frfs=frfs, target_response=target_response, force=force, training_response=training_response, training_response_coordinate=training_response_coordinate, 
                         training_frfs=training_frfs, response_transformation=response_transformation, reference_transformation=reference_transformation, empty=empty)
        self.buzz_cpsd = buzz_cpsd

    @property
    def target_response(self):
        if self._target_response_array_ is None:
            raise AttributeError('A target response array was not defined for this object')
        if is_cpsd(self._target_response_array_):
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(self._target_response_array_, 0, -1), 
                                   outer_product(self._target_response_coordinate_, self._target_response_coordinate_))
        else: 
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(self._target_response_array_, 0, -1), 
                                   np.column_stack((self._target_response_coordinate_, self._target_response_coordinate_)))
    
    @target_response.setter
    def target_response(self, data_array):
        if self._target_response_array_ is not None:
            raise AttributeError('The target responses of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
            raise TypeError('The target response must be a SDynPy PowerSpectralDensityArray')
        check_frequency_abscissa(data_array, self._abscissa_)
        if is_cpsd(data_array):
            self._target_response_array_ = np.moveaxis(data_array[outer_product(self._target_response_coordinate_, self._target_response_coordinate_)].ordinate, -1, 0)
        else:
            self._target_response_array_ = np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0)

    @property
    def validation_response(self):
        if self._target_response_array_ is None:
            raise AttributeError('A target response array was not defined for this object')
        if is_cpsd(self._target_response_array_):
            return self.target_response[outer_product(self._validation_response_coordinate_, self._validation_response_coordinate_)]
        else: 
            return self.target_response[self._validation_response_coordinate_[..., np.newaxis]]
      
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
        transformed_force_array = reference_transform@self._force_array_@np.transpose(reference_transform.conj(), (0, 2, 1))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(transformed_force_array, 0, -1), 
                               outer_product(self._transformed_reference_coordinate_, self._transformed_reference_coordinate_))

    @property
    def training_response(self):
        training_cpsd_coordinate = outer_product(self._training_response_coordinate_, self._training_response_coordinate_)
        if is_cpsd(self._training_response_array_):
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, np.moveaxis(self._training_response_array_, 0, -1), 
                                   training_cpsd_coordinate)
        else:
            # Apply buzz method if the _training_response_array_ is a vector of PSDs, since the buzz is required when only PSDs are supplied
            training_response_with_buzz = apply_buzz_method(self)
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, np.moveaxis(training_response_with_buzz, 0, -1), 
                                   training_cpsd_coordinate)
    
    @training_response.setter
    def training_response(self, data_array):
        if self._training_response_array_ is not None:
            raise AttributeError('The responses of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
            raise TypeError('The training response must be a SDynPy PowerSpectralDensityArray')
        if not np.all(np.isin(np.unique(data_array.response_coordinate), self.target_response_coordinate)):
            raise ValueError('Training response {:} is not a target response coordinate in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self._target_response_coordinate_)].string_array()))
        if self._training_response_coordinate_ is not None and not np.all(np.isin(self._training_response_coordinate_, data_array.response_coordinate)):
            raise ValueError('Training response {:} is not available in the supplied data'.format(self._training_response_coordinate_[~np.isin(self._training_response_coordinate_, data_array.response_coordinate)].string_array()))
        check_frequency_abscissa(data_array, self._abscissa_)
        # The numpy unique is used when setting the coordinate to make sure that the DOF ordering
        # in the training_response_array matches the other data.
        if self._training_response_coordinate_ is None:
            self.training_response_coordinate = np.unique(data_array.response_coordinate)
        if is_cpsd(data_array):
            self._training_response_array_ = np.moveaxis(data_array[outer_product(self._training_response_coordinate_, self._training_response_coordinate_)].ordinate, -1, 0) 
        else:
            self._training_response_array_ = np.moveaxis(data_array[self._training_response_coordinate_[..., np.newaxis]].ordinate, -1, 0)

    @property
    def buzz_cpsd(self):
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, np.moveaxis(self._buzz_cpsd_array_, 0, -1), 
                               outer_product(self._training_response_coordinate_, self._training_response_coordinate_))
    
    @buzz_cpsd.setter
    def buzz_cpsd(self, data_array):
        if data_array is None:
            # Doing this to assign None as the default value for buzz_cpsd_array
            self._buzz_cpsd_array_ = None
        else:
            if not isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
                raise TypeError('The training response must be a SDynPy PowerSpectralDensityArray')
            if not np.all(np.isin(self.training_response_coordinate, np.unique(data_array.response_coordinate))):
                raise ValueError('Data for response coordinate {:} is missing from the buzz CPSD array'.format(self.training_response_coordinate[~np.isin(self.training_response_coordinate, data_array.response_coordinate)].string_array()))
            check_frequency_abscissa(data_array, self._abscissa_)
            self._buzz_cpsd_array_ = np.moveaxis(data_array[outer_product(self._training_response_coordinate_, self._training_response_coordinate_)].ordinate, -1, 0) 

    @property
    def transformed_training_response(self):
        """
        The training response with the transformation applied (i.e., what is used in the inverse problem). 
        The buzz method is only applied if the training response is a set of PSDs rather than CPSDs. 
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        if is_cpsd(self._training_response_array_):
            transformed_response = response_transform@self._training_response_array_@np.transpose(response_transform.conj(), (0, 2, 1))
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(transformed_response, 0, -1), 
                                   outer_product(self._transformed_response_coordinate_, self._transformed_response_coordinate_))
        else:
            training_response_with_buzz = apply_buzz_method(self)
            transformed_response = response_transform@training_response_with_buzz@np.transpose(response_transform.conj(), (0, 2, 1))
            return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(transformed_response, 0, -1), 
                                   outer_product(self._transformed_response_coordinate_, self._transformed_response_coordinate_))

    @property
    def predicted_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so predicted responses cannot be reconstructed')
        predicted_response = self._frf_array_@self._force_array_@np.transpose(self._frf_array_.conj(), (0, 2, 1))
         
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(predicted_response, 0, -1), 
                               outer_product(self._response_coordinate_, self._response_coordinate_)) 

    @property
    def reconstructed_target_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so target responses cannot be reconstructed')
        reconstructed_target_response = self._target_frf_array_@self._force_array_@np.transpose(self._target_frf_array_.conj(), (0, 2, 1))
        
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(reconstructed_target_response, 0, -1), 
                               outer_product(self._target_response_coordinate_, self._target_response_coordinate_)) 
        
    @property
    def reconstructed_validation_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so validation responses cannot be reconstructed')
        reconstructed_validation_response = self._validation_frf_array_@self._force_array_@np.transpose(self._validation_frf_array_.conj(), (0, 2, 1))
 
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(reconstructed_validation_response, 0, -1), 
                               outer_product(self._validation_response_coordinate_, self._validation_response_coordinate_)) 
        
    @property
    def transformed_reconstructed_response(self):
        """
        The reconstructed response (at the training coordinates) with the transformation applied (i.e., what 
        was used in the inverse problem). This always outputs a response CPSD matrix for comparisons 
        against the "transformed_training_response".
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            response_transform = self._response_transformation_array_
        reconstructed_response = self._training_frf_array_@self._force_array_@np.transpose(self._training_frf_array_.conj(), (0, 2, 1))
        reconstructed_response = response_transform@reconstructed_response@np.transpose(response_transform.conj(), (0, 2, 1))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(reconstructed_response, 0, -1), 
                               outer_product(self._transformed_response_coordinate_, self._transformed_response_coordinate_))

    def predicted_response_specific_dofs(self, response_dofs):
        """
        Computes the predicted response for specific response DOFs in the SPR object.

        Parameters
        ----------
        response_dofs : CoordinateArray
            The response DOFs to compute the response for. It should be shaped as a 1d array. 

        Returns
        -------
        predicted_response : PowerSpectralDensityArray
            The predicted response as a square CPSD matrix with the desired response DOFs (in 
            the order they were supplied). 
        
        Raises
        ------
        AttributeError
            If there are not forces in the SPR object.
        ValueError
            If any of the following occurs:

                - If any of the selected response DOFs are not in the response_coordinate of 
                the SPR object. 

                - If the response_dofs is not a 1d array.

        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so predicted responses cannot be computed')
        if len(response_dofs.shape) != 1:
            raise ValueError('The supplied response DOFs must be a 1D array')
        if np.any(~np.isin(response_dofs, self._response_coordinate_)):
            raise ValueError('The supplied response DOFs are not included in the SPR object')
        
        prediction_frfs = self.frfs[sdpy.coordinate.outer_product(response_dofs, self.reference_coordinate)]
        prediction_frfs = np.moveaxis(prediction_frfs.ordinate, -1, 0)

        predicted_response = prediction_frfs@self._force_array_@np.transpose(prediction_frfs.conj(), (0, 2, 1))

        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.abscissa, np.moveaxis(predicted_response, 0, -1), 
                               outer_product(response_dofs, response_dofs))

    def make_buzz_cpsd_from_frf(self):
        """
        Generates the buzz CPSD array from the training FRFs.
        """
        self._buzz_cpsd_array_ = self._training_frf_array_@self._training_frf_array_.conj().transpose((0, 2, 1))
        return self

    def global_asd_error(self, 
                         channel_set='training'):
        """
        Computes the global ASD error in dB of the reconstructed response, 
        per the procedure in MIL-STD 810H.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:

                - training (default)
                    This compares the responses for the transformed training DOFs.

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

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
        if channel_set == 'training':
            truth = self.transformed_training_response.get_asd().ordinate
            reconstructed = self.transformed_reconstructed_response.get_asd().ordinate
        elif channel_set == 'validation':
            truth = self.validation_response.get_asd().ordinate
            reconstructed = self.reconstructed_validation_response.get_asd().ordinate
        elif channel_set == 'target':
            truth = self.target_response.get_asd().ordinate
            reconstructed = self.reconstructed_target_response.get_asd().ordinate
        else:
            raise ValueError('Selected channel set is not available')
        
        weights = (truth**2)/(norm(truth, axis=0, ord=2)**2)
        asd_error = 10*np.log10(reconstructed/truth)
        global_asd_error = np.sum(asd_error*weights, axis=0)
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, global_asd_error, coordinate_array(node=1, direction=1))

    def average_asd_error(self, 
                          channel_set='training'):
        """
        Computes the DOF averaged ASD error in dB of the reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:

                - training (default)
                    This compares the responses for the transformed training DOFs.

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

        Returns
        -------
        PowerSpectralDensityArray
            Returns a spectrum array of the average ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.       
        """
        if channel_set == 'training':
            truth = self.transformed_training_response.get_asd().ordinate
            reconstructed = self.transformed_reconstructed_response.get_asd().ordinate
        elif channel_set == 'validation':
            truth = self.validation_response.get_asd().ordinate
            reconstructed = self.reconstructed_validation_response.get_asd().ordinate
        elif channel_set == 'target':
            truth = self.target_response.get_asd().ordinate
            reconstructed = self.reconstructed_target_response.get_asd().ordinate
        else:
            raise ValueError('Selected channel set is not available')
        
        asd_error = 10*np.log10(reconstructed/truth)
        average_asd_error = 10*np.log10(np.average(10**(asd_error/10), axis=0))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, average_asd_error, coordinate_array(node=1, direction=1))
    
    def rms_asd_error(self, 
                      channel_set='training'):
        """
        Computes the root mean square (RMS) of the DOF-DOF ASD dB error of the 
        reconstructed response.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
            The available options are:

                - training (default)
                    This compares the responses for the transformed training DOFs.

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

        Returns
        -------
        PowerSpectralDensityArray
            Returns a spectrum array of the RMS ASD error in dB. The response
            coordinate for this array is made up to have a value for the DataArray
            and does not correspond to anything.       
        """
        if channel_set == 'training':
            truth = self.transformed_training_response.get_asd().ordinate
            reconstructed = self.transformed_reconstructed_response.get_asd().ordinate
        elif channel_set == 'validation':
            truth = self.validation_response.get_asd().ordinate
            reconstructed = self.reconstructed_validation_response.get_asd().ordinate
        elif channel_set == 'target':
            truth = self.target_response.get_asd().ordinate
            reconstructed = self.reconstructed_target_response.get_asd().ordinate
        else:
            raise ValueError('Selected channel set is not available')
        
        asd_error = 10*np.log10(reconstructed/truth)
        rms_asd_error = np.sqrt(np.mean(asd_error**2, axis=0))
        return sdpy.data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self._abscissa_, rms_asd_error, coordinate_array(node=1, direction=1))

    def error_summary(self,
                      channel_set='training',
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

                - training (default)
                    This compares the responses for the transformed training DOFs.    

                - validation
                    This compares the responses at the validation DOFs.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

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
        if channel_set == 'training':
            return_data = self.transformed_training_response.error_summary(figure_kwargs=figure_kwargs, 
                                                                          linewidth=linewidth, 
                                                                          plot_kwargs=plot_kwargs,
                                                                          cpsd_matrices=self.transformed_reconstructed_response)
        elif channel_set == 'validation':
            return_data = self.validation_response.error_summary(figure_kwargs=figure_kwargs, 
                                                                 linewidth=linewidth, 
                                                                 plot_kwargs=plot_kwargs,
                                                                 cpsd_matrices=self.reconstructed_validation_response)
        elif channel_set == 'target':
            return_data = self.target_response.error_summary(figure_kwargs=figure_kwargs, 
                                                             linewidth=linewidth, 
                                                             plot_kwargs=plot_kwargs,
                                                             cpsd_matrices=self.reconstructed_target_response)
        else:
            raise ValueError('Selected channel set is not available')
        
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
        The "power_inverse_processing" decorator function pre and post processes the training
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
                                 l_curve_type = 'standard',
                                 optimality_condition = 'curvature',
                                 use_transformation=True,
                                 use_buzz = False,
                                 update_header=True,
                                 response = None, frf = None):
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
            value of the training frf array.
        high_regularization_limit : ndarray
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the largest singular
            value of the training frf array.
        number_regularization_values : int
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:

                - forces
                    This L-curve is constructed with the "size" of the forces on 
                    the Y-axis and the regularization parameter on the X-axis. 

                - standard (default)
                    This L-curve is constructed with the residual squared error on 
                    the X-axis and the "size" of the forces on the Y-axis. 

        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
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
        All the settings, including the selected regularization parameters, are saved to 
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
        regularization_values, residual, penalty = tikhonov_full_path_for_l_curve(frf, response,
                                                                                  low_regularization_limit=low_regularization_limit, 
                                                                                  high_regularization_limit=high_regularization_limit,
                                                                                  number_regularization_values=number_regularization_values)
            
        optimal_regularization = l_curve_optimal_regularization(regularization_values, penalty, residual, 
                                                                l_curve_type=l_curve_type, 
                                                                optimality_condition=optimality_condition)
        
        regularized_frf_inverse = frf_inverse(frf, method='tikhonov', regularization_parameter=optimal_regularization)
        regularized_force = regularized_frf_inverse@response@np.moveaxis(regularized_frf_inverse.conj(),-2,-1)

        if update_header:
            self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_l_curve',
                                        'inverse_method':'tikhonov',
                                        'number_regularization_values_searched':number_regularization_values,
                                        'regularization_parameter':optimal_regularization,
                                        'l_curve_type':l_curve_type,
                                        'optimality_condition':optimality_condition,
                                        'use_transformation':use_transformation,
                                        'use_buzz':use_buzz})

        return regularized_force
    
    @power_inverse_processing
    def auto_tikhonov_by_cv_rse(self,
                                low_regularization_limit = None, 
                                high_regularization_limit = None,
                                number_regularization_values=100,
                                cross_validation_type='loocv',
                                number_folds=None,
                                use_transformation=True,
                                use_buzz=False,
                                update_header=True,
                                response=None, frf=None):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with cross 
        validation, where the residual squared error is use as the metric to evaluate
        the quality of fit.

        Parameters
        ----------
        low_regularization_limit : ndarray, optional
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the smallest singular
            value of the training frf array.
        high_regularization_limit : ndarray, optional
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the largest singular
            value of the training frf array.
        number_regularization_values : int, optional
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        cross_validation_type : str, optional
            The cross validation method to use. The available options are:

                - loocv (default)
                    Leave one out cross validation.

                - k-fold
                    K fold cross validation.

        number_folds : int
            The number of folds to use in the k fold cross validation. The number of 
            response DOFs must be evenly divisible by the number of folds.
        use_transformation : bool, optional
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
        All the settings, including the selected regularization parameters, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
        .. [1] D. M. Allen, "The Relationship between Variable Selection and Data Agumentation 
               and a Method for Prediction," Technometrics, vol. 16, no. 1, pp. 125-127, 1974, 
               doi: 10.2307/1267500.
        .. [2] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: 
               Data Mining, Inference, and Prediction, 2nd Edition ed. New York: Springer New York, 
               2017.
        """
        if cross_validation_type == 'loocv':
            regularization_values, mse = leave_one_out_cv(frf, response, 
                                                          low_regularization_limit=low_regularization_limit,
                                                          high_regularization_limit=high_regularization_limit, 
                                                          number_regularization_values=number_regularization_values)

            mse = np.sum(mse, axis=1)
        elif cross_validation_type == 'k-fold':
            regularization_values, mse = k_fold_cv(frf, response, 
                                                   low_regularization_limit=low_regularization_limit,
                                                   high_regularization_limit=high_regularization_limit, 
                                                   number_regularization_values=number_regularization_values, 
                                                   number_folds=number_folds)

            mse = np.sum(mse, axis=1)
        else:
            raise NotImplementedError('The selected cross validation method has not been implemented yet')
        
        optimal_regularization = regularization_values[np.argmin(mse, axis=0),np.arange(regularization_values.shape[1])]

        regularized_frf_inverse = frf_inverse(frf, method='tikhonov', regularization_parameter=optimal_regularization)
        regularized_force = regularized_frf_inverse @ response @ np.moveaxis(regularized_frf_inverse.conj(),-2,-1)

        if update_header:
            self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_cv_mse',
                                        'inverse_method':'Tikhonov regularization',
                                        'number_regularization_values_searched':number_regularization_values,
                                        'regularization_parameter':optimal_regularization,
                                        'cross_validation_type':cross_validation_type,
                                        'use_transformation':use_transformation,
                                        'use_buzz':use_buzz})
        
        return regularized_force

    @power_inverse_processing
    def auto_truncation_by_l_curve(self,
                                   l_curve_type = 'standard',
                                   optimality_condition = 'distance',
                                   curvature_method = None,
                                   use_transformation = True,
                                   use_buzz = False,
                                   update_header=True,
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

                - forces
                    This L-curve is constructed with the "size" of the forces on 
                    the Y-axis and the regularization parameter on the X-axis. 

                - standard (default)
                    This L-curve is constructed with the residual squared error on 
                    the X-axis and the "size" of the forces on the Y-axis. 

        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
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

        curvature_method : std
            The method that is used to compute the curvature of the L-curve, in the 
            case that the curvature is used to find the optimal regularization 
            parameter. The default is None and the options are:

                - numerical
                    This method computes the curvature of the L-curve via numerical 
                    derivatives.

                - cubic_spline
                    This method fits a cubic spline to the L-curve the computes the 
                    curvature from the cubic spline (this might perform better if the 
                    L-curve isn't "smooth").

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
        
        if update_header:
            self.inverse_settings.update({'ISE_technique':'auto_truncation_by_l_curve',
                                        'inverse_method':'truncation',
                                        'number_retained_values':optimal_regularization,
                                        'l_curve_type':l_curve_type,
                                        'optimality_condition':optimality_condition,
                                        'curvature_method':curvature_method,
                                        'use_transformation':use_transformation,
                                        'use_buzz':use_buzz})

        return selected_force
    
    def match_trace_update(self, use_transformation=True,
                           in_place=True):
        """
        Applies a "match trace" update to the to the forces in the SPR object to 
        eliminate bias error.

        Parameters
        ----------
        use_transformation : bool, optional
            Whether or not the transformation was used in the input estimation, 
            this should match what was used in the inverse problem for the correct
            behavior. The default is true. 
        in_place : bool, optional
            Whether to apply the limit to the original SPR object or not. When 
            true, The limit will be applied to the original SPR object. When 
            false, a copy will be made of the original and the limit will be 
            applied to that copy. The default is true. 

        Returns
        -------
            SPR object with the match trace update applied to the force attribute.

        References
        ----------
        .. [1] D. Rohe, R. Schultz, and N. Hunter, "Rattlesnake Users Manual," 
                Sandia National Laboratories, 2021. 
        """
        if self._force_array_ is None:
            raise AttributeError('The SPR object must have pre-existing forces for the update to work')
    
        # setting work object to self is the same as filter self.
        work_object = self if in_place else self.copy()

        if use_transformation:
            reconstructed_response_trace = np.trace(work_object.transformed_reconstructed_response.ordinate) 
            training_response_trace = np.trace(work_object.transformed_training_response.ordinate)
        elif not use_transformation:
            reconstructed_response_trace = np.trace(work_object.reconstructed_target_response[work_object._training_response_coordinate_].ordinate) 
            training_response_trace = np.trace(work_object.training_response.ordinate)
        
        trace_ratio = training_response_trace / reconstructed_response_trace
        work_object._force_array_ = work_object._force_array_*trace_ratio[..., np.newaxis, np.newaxis]

        work_object.inverse_settings.update({'match_trace_applied':True,
                                             'match_trace_use_transformation':use_transformation})
        
        if in_place:
            return self
        else:
            return work_object

    def reduce_drives_update(self, use_transformation=True,
                             db_error_ratio=1.0,
                             reduce_max_drive=False,
                             use_warm_start=True,
                             in_place=True):
        """
        Minimizes the sum of the force PSDs while holding absolute dB error between 
        predicted and actual responses constant, where this dB error is defined by 
        the reconstructed response from the pre-existing force in the SPR object. 

        Parameters
        ----------
        use_transformation : bool, optional
            Whether or not the transformation was used in the input estimation, 
            this should match what was used in the inverse problem for the correct
            behavior. The default is true.
        db_error_ratio : float, optional
            Recommend >=1. Constrains the predicted dB error after drive reduction
            to this value times the prior dB error. Default is 1.0.
        reduce_max_drive : bool, optional
            If true, reduces the maximum drive in each frequency bin. If false,
            reduces the drive trace. Default is False.
        use_warm_start : bool, optional
            Whether to use the initial forces as a warm start for the optimizer.
            Default is True.
        in_place : bool, optional
            Whether to apply the limit to the original SPR object or not. When 
            true, The limit will be applied to the original SPR object. When 
            false, a copy will be made of the original and the limit will be 
            applied to that copy. The default is true. 

        Returns
        -------
            SPR object with the reduce drives update applied to the force attribute.

        Raises
        ------
        AttributeError
            If there are not forces in the SPR object.
        warning
            If the optimization fails to converge or meet the error constraints. 

        Notes
        -----
        This method leverages an optimization problem and warnings may be supplied
        if the optimization fails or has not converged. A failed optimization may 
        have resulted in under/over predicted response or unexpected force amplitudes.
        """
        if self._force_array_ is None:
            raise AttributeError('The SPR object must have pre-existing forces for the update to work')
        
        if use_transformation is True:
            # Checking that there is not a reference transformation
            num_reference_coord = self._reference_coordinate_.shape[0]
            if self._reference_transformation_array_.shape != (num_reference_coord, num_reference_coord):
                raise NotImplementedError('The reduce drives update does not currently work with SPR objects that have non-identity reference transformations')
            else:
                if not np.all(self._reference_transformation_array_ == np.eye(num_reference_coord)):
                    raise NotImplementedError('The reduce drives update does not currently work with SPR objects that have non-identity reference transformations')

        # Initial function for the drive minimization
        if db_error_ratio < 1.0:
            use_warm_start=False # initial value must be feasible
            
        #  Drive minimizer
        def minimize_reference(y, H, y_prior, X0, db_error_ratio):
            # y is (N_response,) slice of response asd, H is slice of FRF, y_prior
            # is (N_ref,) slice of prior forces
            m,n = H.shape
            X   = cp.Variable((n, n), hermitian=True)
            if use_warm_start:
                try:
                    X.value = X0 # warm start from prior solution
                except:
                    pass
                
            # Get LB and UB based on db difference
            db_diffs = db_error_ratio*10*np.log10(y_prior/y)
            y_lb = y*10**(-np.abs(db_diffs)/10) # LB is current dB error below spec
            y_ub = y*10**(np.abs(db_diffs)/10) # UB is current dB error above spec
            
            # Constraints
            constraints  = [X >> 0]                     # PSD constraint
            for i, h in enumerate(H):
                expr = cp.real(h[np.newaxis,:] @ X @ cp.conj(h[:,np.newaxis]))    # affine in X
                constraints += [expr >= y_lb[i], expr <= y_ub[i]]
        
            # objective
            if reduce_max_drive:
                obj = cp.Minimize(cp.norm(cp.diag(X),"inf"))
            else:
                obj = cp.Minimize(cp.trace(X))
        
            # solve
            prob = cp.Problem(obj, constraints)
            
            try:
                prob.solve(solver="SCS", warm_start=use_warm_start, verbose=False) # "SCS" or "CLARABEL"
                return X.value
            except:
                warnings.warn("optimization failed in frequency bin")
                return np.zeros((n,n))
        
        # setting work object to self is the same as filter self.
        work_object = self if in_place else self.copy()

        # Get numpy arrays
        if use_transformation:
            reconstructed_response_asd = np.abs(np.einsum('jji->ij',work_object.transformed_reconstructed_response.ordinate))
            asd_response = np.abs(np.einsum('jji->ij',work_object.transformed_training_response.ordinate))
            frf = work_object.transformed_training_frfs.ordinate.transpose(2,0,1)
            prior_drives = work_object.transformed_force.ordinate.transpose(2,0,1)
        elif not use_transformation:
            training_coords = sdpy.coordinate.outer_product(work_object.training_response_coordinate, work_object.training_response_coordinate)
            reconstructed_response_asd = np.abs(np.einsum('jji->ij', work_object.reconstructed_target_response[training_coords].ordinate))
            asd_response = np.abs(np.einsum('jji->ij', work_object.training_response.ordinate))
            frf = work_object._training_frf_array_
            prior_drives = work_object.force.ordinate.transpose(2,0,1)
        
        # Other pre-processing
        n_freq = reconstructed_response_asd.shape[0]
        target_exists = np.sum(asd_response,axis=1)>1e-15 # don't solve if target is 0
        
        # Parallel execution
        X_solved = Parallel(n_jobs=-1)(delayed(minimize_reference)(asd_response[i], frf[i],
                                                                   reconstructed_response_asd[i], prior_drives[i], db_error_ratio)
                                       for i in np.arange(n_freq)[target_exists])
        
        # Logic to ignore frequency lines where the optimization returned None
        for ii, index in enumerate(np.where(target_exists==True)[0]):
            if X_solved[ii] is None:
                target_exists[index] = False

        work_object._force_array_[target_exists] = np.array([X_solved[ii] for ii in range(len(X_solved)) if X_solved[ii] is not None]) 
        work_object.inverse_settings.update({'reduce_drives_applied':True,
                                             'reduce_drives_use_transformation':use_transformation})

        if use_transformation:
            reduced_reconstructed = np.abs(np.einsum('jji->ij',work_object.transformed_reconstructed_response.ordinate))
        elif not use_transformation:
            reduced_reconstructed = np.abs(np.einsum('jji->ij', work_object.reconstructed_target_response[training_coords].ordinate))

        if reduce_drives_condition_not_met(asd_response, reconstructed_response_asd, reduced_reconstructed, db_error_ratio):
            warnings.warn('The reduce drives update failed and resulted in an under or over predicted response at some frequencies')
        
        if in_place:
            return self
        else:
            return work_object
        
    def apply_response_limit(self, response_limit,
                             limit_db_level=float(0), 
                             interpolation_type='loglog',
                             in_place=True):
        """
        Scales the force CPSD to apply a limit to the predicted PSD responses in 
        the SPR object.

        Parameters
        ----------
        response_limit : ResponseLimit
            The ResponseLimit object that defines the limits.
        limit_level : optional, float
            The dB level for the limit. The levels in the limit_dict will be 
            modified by the dB level. The default is 0 dB (no modification). 
        interpolation_type : str, optional
            The type of interpolation to use to convert the breakpoints to all
            the frequency lines in the SPR object. The options are loglog or 
            linearlinear, depending on if the frequencies and levels should be 
            plotted on log-log or linear-linear plot axes. The default is loglog.
        in_place : bool, optional
            Whether to apply the limit to the original SPR object or not. When 
            true, The limit will be applied to the original SPR object. When 
            false, a copy will be made of the original and the limit will be 
            applied to that copy. The default is true. 

        Returns
        -------
            SPR object with the response limits applied.

        Raises
        ------
        KeyError
            If the limit_dict does not have the correct keys or if the information
            for the keys are not organized correctly.
        ValueError
            If the specified interpolation type is not available. 
        
        Notes
        -----
        The limit is applied iteratively to each limit DOF, one at a time, rather
        than determining a globally optimal solution with the limits applied.
        """
        # setting work object to self is the same as filter self.
        work_object = self if in_place else self.copy()

        limit_multiplier = 10**(float(limit_db_level)/10)

        for dof_limit in response_limit:
            predicted_response = np.abs(work_object.predicted_response_specific_dofs(dof_limit.limit_coordinate).ordinate)[0,0,:]            
            
            limit = dof_limit.interpolate_to_full_frequency(self.abscissa, interpolation_type)*limit_multiplier
            limit = limit[0,:]

            if np.any(predicted_response > limit):
                limit_inds = predicted_response > limit
                work_object._force_array_[limit_inds, ...] *= (limit[limit_inds] / predicted_response[limit_inds])[:, np.newaxis, np.newaxis]
        
        if in_place:
            return self
        else:
            return work_object

class TransientSourcePathReceiver(SourcePathReceiver):
    """
    A subclass to represent a source-path-receiver (SPR) model with time traces 
    for the responses and forces.

    Notes
    -----
    The "transient" term in the class name refers to the intended use of this SPR model
    (transient problems).
    """

    def __init__(self, frfs=None, target_response=None, force=None, training_response=None, training_response_coordinate=None, 
                 training_frfs=None, response_transformation=None, reference_transformation=None, empty=False):
        """
        Parameters
        ----------
        target_response : TimeHistoryArray
            The measured responses at both the training and validation DOFs that define the 
            receiver of the SPR object.
        force : TimeHistoryArray, optional
            The forces that define the source in the SPR model. The force degrees of freedom 
            (DOFs) should match the reference DOFs in the FRFs. The forces will typically be 
            set via an inverse method in the class, but the user can also set them manually. 
        training_response : TimeHistoryArray, optional
            The measured responses that will be used to estimate the forces in the SPR object 
            (e.g., the specified responses in a MIMO vibration test). Defaults to the "full"
            response if a training response or training_response_coordinate is not supplied .

        Notes
        -----
        Much of the initialization is inherited from the base SourcePathReceiver class.
        """
        # Inheriting the initial set-up from the super class
        super().__init__(frfs=frfs, target_response=target_response, force=force, training_response=training_response, training_response_coordinate=training_response_coordinate, 
                         training_frfs=training_frfs, response_transformation=response_transformation, reference_transformation=reference_transformation, empty=empty)
    
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
        the argument names match the private variable name. The save method is specially 
        defined for the TransientSourcePathReceiver because it has a "time_abscissa" private 
        property, which isn't in the other SourcePathReceiver objects.
        """
        np.savez(filename, 
                 target_response=self._target_response_array_,
                 frf=self._frf_array_,
                 force=self._force_array_,
                 training_response=self._training_response_array_,
                 target_frf=self._target_frf_array_,
                 training_frf=self._training_frf_array_,
                 validation_frf=self._validation_frf_array_,
                 response_transformation=self._response_transformation_array_,
                 reference_transformation=self._reference_transformation_array_,
                 response_coordinate=self._response_coordinate_.string_array(),
                 training_response_coordinate=self._training_response_coordinate_.string_array(),
                 validation_response_coordinate=self._validation_response_coordinate_.string_array(),
                 target_response_coordinate=self._target_response_coordinate_.string_array(),
                 reference_coordinate=self._reference_coordinate_.string_array(),
                 transformed_response_coordinate=self._transformed_response_coordinate_.string_array(),
                 transformed_reference_coordinate=self._transformed_reference_coordinate_.string_array(),
                 abscissa=self._abscissa_,
                 time_abscissa=self._time_abscissa_,
                 inverse_settings=np.array(self.inverse_settings, dtype=object)) 
    
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
        file, where the argument names match the private variable name. The load method is 
        specially defined for the TransientSourcePathReceiver because it has a "time_abscissa" 
        private property, which isn't in the other SourcePathReceiver objects.
        """
        loaded_spr = np.load(filename, allow_pickle=True)
        spr_object = cls.__new__(cls)
        spr_object._target_response_array_ = loaded_spr['target_response']
        spr_object._frf_array_=loaded_spr['frf']
        spr_object._force_array_ = loaded_spr['force'] if np.all(loaded_spr['force'] != np.array(None)) else None
        spr_object._training_response_array_ = loaded_spr['training_response']
        spr_object._target_frf_array_ = loaded_spr['target_frf']
        spr_object._training_frf_array_ = loaded_spr['training_frf']
        spr_object._validation_frf_array_ = loaded_spr['validation_frf']
        spr_object._response_transformation_array_ = loaded_spr['response_transformation']
        spr_object._reference_transformation_array_ = loaded_spr['reference_transformation']
        spr_object._training_response_coordinate_ = coordinate_array(string_array=loaded_spr['training_response_coordinate'])
        spr_object._validation_response_coordinate_ = coordinate_array(string_array=loaded_spr['validation_response_coordinate'])
        spr_object._response_coordinate_ = coordinate_array(string_array=loaded_spr['response_coordinate'])
        spr_object._target_response_coordinate_ = coordinate_array(string_array=loaded_spr['target_response_coordinate'])
        spr_object._reference_coordinate_ = coordinate_array(string_array=loaded_spr['reference_coordinate'])
        spr_object._transformed_response_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_response_coordinate'])
        spr_object._transformed_reference_coordinate_ = coordinate_array(string_array=loaded_spr['transformed_reference_coordinate'])
        spr_object._abscissa_ = loaded_spr['abscissa']
        spr_object._time_abscissa_ = loaded_spr['time_abscissa']
        spr_object.inverse_settings = loaded_spr['inverse_settings'].item()
        return spr_object

    @property
    def target_response(self):
        if self._target_response_array_ is None:
            raise AttributeError('A target response array was not defined for this object')
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(self._target_response_array_, 0, -1), 
                               self._target_response_coordinate_[..., np.newaxis])
    
    @target_response.setter
    def target_response(self, data_array):
        if self._target_response_array_ is not None:
            raise AttributeError('The target responses of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.TimeHistoryArray):
            raise TypeError('The target response must be a SDynPy TimeHistoryArray')
        compare_sampling_rate(data_array, self._abscissa_.max()*2)
        self._time_abscissa_=data_array.flatten()[0].abscissa
        self._target_response_array_ = np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0)
    
    @property
    def validation_response(self):
        if self._target_response_array_ is None:
            raise AttributeError('A target response array was not defined for this object')
        return self.target_response[self._validation_response_coordinate_[..., np.newaxis]]

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
        self._force_array_ = np.moveaxis(data_array[self._reference_coordinate_[..., np.newaxis]].ordinate, -1, 0)

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
    def training_response(self):
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(self._training_response_array_, 0, -1), self._training_response_coordinate_[..., np.newaxis])
    
    @training_response.setter
    def training_response(self, data_array):
        if self._training_response_array_ is not None:
            raise AttributeError('The training responses of an SPR object cannot be reset once the object is initialized')
        if not isinstance(data_array, sdpy.core.sdynpy_data.TimeHistoryArray):
            raise TypeError('The training response must be a SDynPy TimeHistoryArray')
        if not np.all(np.isin(data_array.response_coordinate, self._target_response_coordinate_)):
            raise ValueError('Training response {:} is not a target response coordinate in the SPR model'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self._target_response_coordinate_)].string_array()))
        if self._training_response_coordinate_ is not None and not np.all(np.isin(self._training_response_coordinate_, data_array.response_coordinate)):
            raise ValueError('Training response {:} is not available in the supplied data'.format(self._training_response_coordinate_[~np.isin(self._training_response_coordinate_, data_array.response_coordinate)].string_array()))
        compare_sampling_rate(data_array, self._abscissa_.max()*2)
        # The numpy unique is used when setting the coordinate to make sure that the DOF ordering
        # in the training_response_array matches the other data.
        if self._training_response_coordinate_ is None:
            self.training_response_coordinate = np.unique(data_array.response_coordinate)
        self._training_response_array_ = np.moveaxis(data_array[self._training_response_coordinate_[..., np.newaxis]].ordinate, -1, 0) 

    @property
    def transformed_training_response(self):
        """
        The training response with the transformation applied (i.e., what is used in the inverse problem).
        """
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            raise NotImplementedError('A frequency dependent response transformation has not been implemented for the training response property')
        transformed_response = (response_transform@self._training_response_array_[..., np.newaxis])[..., 0]
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(transformed_response, 0, -1), self._transformed_response_coordinate_[..., np.newaxis])

    @property
    def predicted_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so predicted responses cannot be reconstructed')
        return self.force.mimo_forward(self.frfs)
    
    @property
    def reconstructed_target_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so target responses cannot be reconstructed')
        return self.force.mimo_forward(self.target_frfs)
    
    @property
    def reconstructed_training_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so training responses cannot be reconstructed')
        return self.force.mimo_forward(self.training_frfs)
    
    @property
    def reconstructed_validation_response(self):
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so validation responses cannot be reconstructed')
        return self.force.mimo_forward(self.validation_frfs)
    
    @property
    def transformed_reconstructed_response(self):
        """
        The reconstructed response (at the training coordinates) with the transformation applied (i.e., what
        was used in the inverse problem).
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so responses cannot be reconstructed')
        if self._response_transformation_array_.ndim == 2:        
            response_transform = self._response_transformation_array_[np.newaxis, ...] 
        elif self._response_transformation_array_.ndim == 3:        
            raise NotImplementedError('A frequency dependent response transformation has not been implemented for the training response property')
        reconstructed_response = response_transform@np.moveaxis(self.force.mimo_forward(self.training_frfs).ordinate, -1, 0)[..., np.newaxis]
        return sdpy.data_array(FunctionTypes.TIME_RESPONSE, self._time_abscissa_, np.moveaxis(reconstructed_response[..., 0], 0, -1), self._transformed_response_coordinate_[..., np.newaxis])
    
    @property
    def time_abscissa(self):
        return self._time_abscissa_

    @property
    def time_abscissa_spacing(self):
        return np.mean(np.diff(self._time_abscissa_))
    
    def predicted_response_specific_dofs(self, response_dofs):
        """
        Computes the predicted response for specific response DOFs in the SPR object.

        Parameters
        ----------
        response_dofs : CoordinateArray
            The response DOFs to compute the response for. It should be shaped as a 1d array. 

        Returns
        -------
        predicted_response : TimeHistoryArray
            The predicted response with the desired response DOFs (in the order they were 
            supplied). 
        
        Raises
        ------
        AttributeError
            If there are not forces in the SPR object.
        ValueError
            If any of the following occurs:
                - If any of the selected response DOFs are not in the response_coordinate of 
                the SPR object. 

                - If the response_dofs is not a 1d array.
                
        """
        if self._force_array_ is None:
            raise AttributeError('There is no force array in this object so predicted responses cannot be computed')
        if len(response_dofs.shape) != 1:
            raise ValueError('The supplied response DOFs must be a 1D array')
        if np.any(~np.isin(response_dofs, self._response_coordinate_)):
            raise ValueError('All the supplied response DOFs are not included in the SPR object')
        
        prediction_frfs = self.frfs[sdpy.coordinate.outer_product(response_dofs, self.reference_coordinate)]

        return self.force.mimo_forward(prediction_frfs)

    def extract_time_elements_by_abscissa(self, min_abscissa=None, max_abscissa=None, in_place=True):
        """
        Extracts the time elements from all the components of the SPR object with 
        abscissa values within the specified range.

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
        """
        
        work_object = self if in_place else self.copy()
        
        if min_abscissa is None and max_abscissa is None:
            raise ValueError('A bounding abscissa (min_abscissa, max_abscissa, or both) must be supplied')
        if min_abscissa is None:
            abscissa_indices = (work_object._time_abscissa_ <= max_abscissa)
        elif max_abscissa is None:
            abscissa_indices = (work_object._time_abscissa_ >= min_abscissa)
        else:
            abscissa_indices = (work_object._time_abscissa_ >= min_abscissa) & (work_object._time_abscissa_ <= max_abscissa)

        work_object._time_abscissa_ = work_object._time_abscissa_[abscissa_indices, ...]
        if work_object._target_response_array_ is not None:
            work_object._target_response_array_ = work_object._target_response_array_[abscissa_indices, ...]
        if work_object._force_array_ is not None:
            work_object._force_array_ = work_object._force_array_[abscissa_indices, ...]
        if work_object._training_response_array_ is not None:
            work_object._training_response_array_ = work_object._training_response_array_[abscissa_indices, ...]
          
        if in_place:
            return self
        else:
            return work_object

    def global_rms_error(self, 
                         channel_set='training',
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

                - training (default)
                    This compares the responses for the transformed training DOFs 
                    in the SPR object.

                - validation
                    This compares the responses for the validation response DOFs 
                    in the SPR object.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

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
        truth, reconstructed = preprocess_data_for_quality_metric(self,
                                                                  channel_set=channel_set,
                                                                  samples_per_frame=samples_per_frame,
                                                                  frame_length=frame_length,
                                                                  overlap=overlap,
                                                                  overlap_samples=overlap_samples)
        return compute_global_rms_error(truth, reconstructed)

    def average_rms_error(self, 
                          channel_set='training',
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

                - training (default)
                    This compares the responses for the transformed training 
                    DOFs in the SPR object.

                - validation
                    This compares the responses for the validation response 
                    DOFs in the SPR object.

                - target
                    This compares the responses for all the target response DOFs 
                    in the SPR object. 

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
        truth, reconstructed = preprocess_data_for_quality_metric(self,
                                                                  channel_set=channel_set,
                                                                  samples_per_frame=samples_per_frame,
                                                                  frame_length=frame_length,
                                                                  overlap=overlap,
                                                                  overlap_samples=overlap_samples)
        return compute_average_rms_error(truth, reconstructed)
    
    def time_varying_trac(self, 
                          channel_set='training',
                          samples_per_frame=None,
                          frame_length=None,
                          overlap=None,
                          overlap_samples=None):
        """
        Computes the time varying TRAC comparison between the truth and 
        reconstructed responses.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
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
            Returns a time history array of the time varying TRAC for all the 
            response degrees of freedom.
        """
        truth, reconstructed = preprocess_data_for_quality_metric(self,
                                                                  channel_set=channel_set,
                                                                  samples_per_frame=samples_per_frame,
                                                                  frame_length=frame_length,
                                                                  overlap=overlap,
                                                                  overlap_samples=overlap_samples)
        return compute_time_varying_trac(truth, reconstructed)
    
    def time_varying_level_error(self, 
                                 channel_set='training',
                                 level_type='rms',
                                 samples_per_frame=None,
                                 frame_length=None,
                                 overlap=None,
                                 overlap_samples=None):
        """
        Computes the computes the time varying error for a statistical level 
        (rms or max) between the truth and reconstructed responses.

        Parameters
        ----------
        channel_set : str, optional
            The channel set to make the response comparisons between.
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

        level_type : str, optional
            The type of level to be used in the comparison. The options are:

                - rms (default)
                    The rms level error for each frame of data in the responses. 

                - max
                    The error in the maximum level that is seem for each frame 
                    of data in the responses.

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
            Returns a time history array of the time varying TRAC for all the 
            response degrees of freedom.
        """
        truth, reconstructed = preprocess_data_for_quality_metric(self,
                                                                  channel_set=channel_set,
                                                                  samples_per_frame=samples_per_frame,
                                                                  frame_length=frame_length,
                                                                  overlap=overlap,
                                                                  overlap_samples=overlap_samples)
        return compute_time_varying_level_error(truth, reconstructed, level_type=level_type)

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
        if work_object._target_response_array_ is not None:
            work_object._target_response_array_ = sosfiltfilt(sos, work_object._target_response_array_, **sosfiltfilt_kwargs)
        if work_object._training_response_array_ is not None:
            work_object._training_response_array_ = sosfiltfilt(sos, work_object._training_response_array_, **sosfiltfilt_kwargs)
 
        if in_place:
            return self
        else:
            return work_object
        
    def attenuate_force(self, limit: Union[float, np.ndarray], 
                        full_scale: float = 1.0,
                        in_place: bool =True):
        """
        Attenuate peaks in the force that exceed limits by scaling the region between
        zero crossings using the local maximum. This maintains a smooth waveform that
        does not exceed the specified limits.

        Parameters
        ----------
        limit : float | np.ndarray
            limit value or array of limit values with shape (n_signals,)
        full_scale : float, optional
            global scaling factor applied to limit value, intended to be used such
            that output waveform peaks are slightly less than physical limit,
            (ex. use full_scale=0.97 so that output signal will not exceed 97% of specified 
            limit), by default 1.0
        in_place : bool, optional
            Whether to apply the limit to the original SPR object or not. When 
            true, The limit will be applied to the original SPR object. When 
            false, a copy will be made of the original and the limit will be 
            applied to that copy. The default is true. 
        
        Returns
        -------
            SPR object with the limit applied to the force attribute.

        Notes
        -----
        Each region between zero crossings is scaled according to:
            `full_scale * limit / local_maximum`
        """
        # setting work object to self is the same as filter self.
        work_object = self if in_place else self.copy()

        work_object._force_array_ = attenuate_signal(work_object._force_array_.T, 
                                                    limit=limit, full_scale=full_scale).T
        if in_place:
            return self
        else:
            return work_object

    @transient_inverse_processing
    def manual_inverse(self, 
                       inverse_method = 'standard',
                       regularization_weighting_matrix = None,
                       regularization_parameter = None,
                       cond_num_threshold = None,
                       num_retained_values = None,
                       cola_frame_length = None,
                       cola_window = ('tukey', 0.5),
                       cola_overlap_samples = None,
                       frf_interpolation_type = 'sinc',
                       transformation_interpolation_type = 'cubic',
                       use_transformation = False,
                       response_generator = None, 
                       frf = None, 
                       reconstruction_generator = None):
        """
        Performs the inverse source estimation problem with manual settings.

        Parameters
        ----------
        cola_frame_length : float, optional
            The frame length (in samples) if the COLA method is being used. The
            default frame length is Fs/df from the transfer function. 
        cola_window : str, optional
            The desired window for the COLA procedure, must exist in the scipy
            window library. The default is a Tukey window with an alpha of 0.5.
        cola_overlap_samples : int, optional
            The number of overlapping samples between measurement frames in the
            COLA procedure.  A default is defined for the default Tukey window, 
            otherwise the user must supply it.
        inverse_method : str, optional
            The method to be used for the FRF matrix inversions. The available 
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
        frf_interpolation_type : str, optional
            The type of interpolation to use on the FRFs (to account for the zero padding).
            This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
            The default is 'cubic'.
        transformation_interpolation_type : str, optional
            The type of interpolation to use on the FRFs (to account for the zero padding).
            This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
            The default is 'cubic'.
        use_transformation : bool, optional
            Whether or not to use the transformations in the ISE problem. The default is 
            False.
        response_generator : function
            The generator function to create the COLA segmented responses. This is created 
            by the decorator function and should not be supplied by the user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator function
            and object definition. This argument should not be supplied by the user. 
        reconstruction_generator : function
            The generator function to recompile the COLA segmented forces. This is created 
            by the decorator function and should not be supplied by the user.

        Returns
        -------
        reconstructed_force : ndarray
            An ndarray array of the estimated sources. 

        Notes
        -----
        The "transient_inverse_processing" decorator function pre and post processes the 
        training response and FRF data from the SourcePathReceiver object to segment the 
        data for COLA processing, apply the response and reference transformations, and 
        recompile the segmented forces into a single time trace. This method only estimates the 
        forces, using the supplied FRF inverse parameters.

        References
        ----------
        .. [1] Wikipedia, "Overlap-add Method".
            https://en.wikipedia.org/wiki/Overlap-add_method
        """
        if inverse_method=='threshold' and cond_num_threshold.size>1:
            padded_frequency = rfftfreq((frf.shape[0]-1)*2, self.time_abscissa_spacing)
            interpolator = interp1d(self.abscissa, cond_num_threshold, kind='cubic', axis=0, fill_value=0) 
            cond_num_threshold = interpolator(padded_frequency)
        if inverse_method=='tikhonov' and regularization_parameter.size>1:
            padded_frequency = rfftfreq((frf.shape[0]-1)*2, self.time_abscissa_spacing)
            interpolator = interp1d(self.abscissa, regularization_parameter, kind='cubic', axis=0, fill_value=0) 
            regularization_parameter = interpolator(padded_frequency)
        if inverse_method=='truncation' and num_retained_values.size>1:
            padded_frequency = rfftfreq((frf.shape[0]-1)*2, self.time_abscissa_spacing)
            interpolator = interp1d(self.abscissa, num_retained_values, kind='previous', axis=0, fill_value=0) 
            num_retained_values = interpolator(padded_frequency)

        frf_pinv = np.ascontiguousarray(frf_inverse(frf, method = inverse_method,
                                                    regularization_weighting_matrix = regularization_weighting_matrix,
                                                    regularization_parameter = regularization_parameter,
                                                    cond_num_threshold = cond_num_threshold,
                                                    num_retained_values = num_retained_values))
        for segment_fft in response_generator:
            reconstructed_force = reconstruction_generator.send(frf_pinv@segment_fft)
    
        try:
            self.inverse_settings.update({'ISE_technique':'manual', 
                                            'cola_frame_length':cola_frame_length,
                                            'cola_window':cola_window,
                                            'cola_overlap':cola_overlap_samples,
                                            'inverse_method':inverse_method,
                                            'FRF_interpolation_type':frf_interpolation_type,
                                            'transformation_interpolation_type':transformation_interpolation_type,
                                            'use_transformation':use_transformation,
                                            'regularization_weighting_matrix':regularization_weighting_matrix,
                                            'regularization_parameter':regularization_parameter,
                                            'cond_num_threshold':cond_num_threshold,
                                            'num_retained_values':num_retained_values})
        except AttributeError: 
            self.inverse_settings = {'ISE_technique':'manual', 
                                        'cola_frame_length':cola_frame_length,
                                        'cola_window':cola_window,
                                        'cola_overlap':cola_overlap_samples,
                                        'inverse_method':inverse_method,
                                        'FRF_interpolation_type':frf_interpolation_type,
                                        'transformation_interpolation_type':transformation_interpolation_type,
                                        'use_transformation':use_transformation,
                                        'regularization_weighting_matrix':regularization_weighting_matrix,
                                        'regularization_parameter':regularization_parameter,
                                        'cond_num_threshold':cond_num_threshold,
                                        'num_retained_values':num_retained_values}

        return reconstructed_force 
       
    @transient_inverse_processing
    def auto_tikhonov_by_l_curve(self, 
                                 low_regularization_limit = None, 
                                 high_regularization_limit = None,
                                 number_regularization_values=100,
                                 l_curve_type = 'standard',
                                 optimality_condition = 'curvature',
                                 cola_frame_length = None,
                                 cola_window = ('tukey', 0.5),
                                 cola_overlap_samples = None,
                                 frf_interpolation_type = 'sinc',
                                 transformation_interpolation_type = 'cubic',
                                 use_transformation = False,
                                 response_generator = None, 
                                 frf = None, 
                                 reconstruction_generator = None):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with L-curve 
        methods.

        Parameters
        ----------
        cola_frame_length : float, optional
            The frame length (in samples) if the COLA method is being used. The
            default frame length is Fs/df from the transfer function. 
        cola_window : str, optional
            The desired window for the COLA procedure, must exist in the scipy
            window library. The default is a Tukey window with an alpha of 0.5.
        cola_overlap_samples : int, optional
            The number of overlapping samples between measurement frames in the
            COLA procedure.  A default is defined for the default Tukey window, 
            otherwise the user must supply it.
        low_regularization_limit : ndarray
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the smallest singular
            value of the training frf array.
        high_regularization_limit : ndarray
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the largest singular
            value of the training frf array.
        number_regularization_values : int
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        l_curve_type : str
            The type of L-curve that is used to find the "optimal regularization 
            parameter. The available types are:

                - forces
                    This L-curve is constructed with the "size" of the forces on the 
                    Y-axis and the regularization parameter on the X-axis. 

                - standard (default)
                    This L-curve is constructed with the residual squared error on 
                    the X-axis and the "size" of the forces on the Y-axis. 

        optimality_condition : str
            The method that is used to find an "optimal" regularization parameter.
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

        frf_interpolation_type : str, optional
            The type of interpolation to use on the FRFs (to account for the zero padding).
            This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
            The default is 'cubic'.
        transformation_interpolation_type : str, optional
            The type of interpolation to use on the FRFs (to account for the zero padding).
            This can be 'sinc' or any type that is allowed with scipy.interpolate.interp1d.
            The default is 'cubic'.
        use_transformation : bool, optional
            Whether or not to use the transformations in the ISE problem. The default is 
            False.
        response_generator : function
            The generator function to create the COLA segmented responses. This is created 
            by the decorator function and should not be supplied by the user. 
        frf : ndarray
            The preprocessed frf data. The preprocessing is handled by the decorator function
            and object definition. This argument should not be supplied by the user. 
        reconstruction_generator : function
            The generator function to recompile the COLA segmented forces. This is created 
            by the decorator function and should not be supplied by the user.

        Returns
        -------
        reconstructed_force : ndarray
            An ndarray array of the estimated sources. 

        Notes
        -----
        The "transient_inverse_processing" decorator function pre and post processes the 
        training response and FRF data from the SourcePathReceiver object to segment the 
        data for COLA processing, apply the response and reference transformations, and 
        recompile the segmented forces into a single time trace. This method only estimates the 
        forces, using the supplied FRF inverse parameters.

        References
        ----------
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
        """
        regularization_values, Uh, V, regularized_S = compute_regularized_svd_inv(frf, 
                                                                                  low_regularization_limit=low_regularization_limit,
                                                                                  high_regularization_limit=high_regularization_limit, 
                                                                                  number_regularization_values=number_regularization_values)
        
        for segment_fft in response_generator:
            residual, penalty = compute_regularized_residual_penalty_for_l_curve(frf, segment_fft[...,0], # Need the removing the trailing axis from the end of the response
                                                                                 Uh, regularized_S, V)
            
            optimal_regularization = l_curve_optimal_regularization(regularization_values, penalty, residual, 
                                                                    l_curve_type=l_curve_type,
                                                                    optimality_condition=optimality_condition)

            frf_pinv = np.ascontiguousarray(frf_inverse(frf, method = 'tikhonov',
                                                        regularization_parameter = optimal_regularization))
            reconstructed_force = reconstruction_generator.send(frf_pinv@segment_fft)
    
        try:
            self.inverse_settings.update({'ISE_technique':'manual', 
                                            'cola_frame_length':cola_frame_length,
                                            'cola_window':cola_window,
                                            'cola_overlap':cola_overlap_samples,
                                            'FRF_interpolation_type':frf_interpolation_type,
                                            'transformation_interpolation_type':transformation_interpolation_type,
                                            'use_transformation':use_transformation})
        except AttributeError: 
            self.inverse_settings = {'ISE_technique':'manual', 
                                     'cola_frame_length':cola_frame_length,
                                     'cola_window':cola_window,
                                     'cola_overlap':cola_overlap_samples,
                                     'FRF_interpolation_type':frf_interpolation_type,
                                     'transformation_interpolation_type':transformation_interpolation_type,
                                     'use_transformation':use_transformation}

        return reconstructed_force 
    
    @transient_inverse_processing
    def auto_tikhonov_by_cv_rse(self,
                                low_regularization_limit=None, 
                                high_regularization_limit=None,
                                number_regularization_values=100,
                                cross_validation_type='loocv',
                                number_folds=None,
                                cola_frame_length=None,
                                cola_window=('tukey', 0.5),
                                cola_overlap_samples=None,
                                frf_interpolation_type='sinc',
                                transformation_interpolation_type='cubic',
                                use_transformation=False,
                                response_generator=None, 
                                frf=None, 
                                reconstruction_generator=None):
        """
        Performs the inverse source estimation problem with Tikhonov regularization, 
        where the regularization parameter is automatically selected with cross 
        validation, where the residual squared error is use as the metric to evaluate
        the quality of fit.

        Parameters
        ----------
        low_regularization_limit : ndarray, optional
            The low regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the smallest singular
            value of the training frf array.
        high_regularization_limit : ndarray, optional
            The high regularization limit to search through. This should be a 1d
            array with a length that matches the number of frequency lines in 
            the SourcePathReceiver object. The default is the largest singular
            value of the training frf array.
        number_regularization_values : int, optional
            The number of regularization parameters to search over, where the 
            potential parameters are geometrically spaced between the low and high
            regularization limits.  
        cross_validation_type : str, optional
            The cross validation method to use. The available options are:

                - loocv (default)
                    Leave one out cross validation.

                - k-fold
                    K fold cross validation.

        number_folds : int
            The number of folds to use in the k fold cross validation. The number of 
            response DOFs must be evenly divisible by the number of folds.
        use_transformation : bool, optional
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
        All the settings, including the selected regularization parameters, are saved to 
        the "inverse_settings" class property. 

        References
        ----------
        .. [1] Wikipedia, "Overlap-add Method".
               https://en.wikipedia.org/wiki/Overlap-add_method
        .. [2] D. M. Allen, "The Relationship between Variable Selection and Data Agumentation 
               and a Method for Prediction," Technometrics, vol. 16, no. 1, pp. 125-127, 1974, 
               doi: 10.2307/1267500.
        .. [3] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: 
               Data Mining, Inference, and Prediction, 2nd Edition ed. New York: Springer New York, 
               2017.
        """
        # Setting the regularization limits based on the full FRF array to have 
        # consistent parameters for all the subsamples. This is done here instead of in the 
        # cross validation function to prevent doing the SVD over and over.
        if low_regularization_limit is None or high_regularization_limit is None:
            S = np.linalg.svd(frf, full_matrices=False, compute_uv=False)
            if low_regularization_limit is None:
                low_regularization_limit = S[:,-1]
            if high_regularization_limit is None:
                high_regularization_limit = S[:,0]
            del S

        for segment_fft in response_generator:
            if cross_validation_type == 'loocv':
                regularization_values, mse = leave_one_out_cv(frf, segment_fft[...,0], 
                                                              low_regularization_limit=low_regularization_limit,
                                                              high_regularization_limit=high_regularization_limit, 
                                                              number_regularization_values=number_regularization_values)

                mse = np.sum(mse, axis=1)
            elif cross_validation_type == 'k-fold':
                regularization_values, mse = k_fold_cv(frf, segment_fft[...,0], 
                                                       low_regularization_limit=low_regularization_limit,
                                                       high_regularization_limit=high_regularization_limit, 
                                                       number_regularization_values=number_regularization_values, 
                                                       number_folds=number_folds)

                mse = np.sum(mse, axis=1)
            else:
                raise NotImplementedError('The selected cross validation method has not been implemented yet')
            
            optimal_regularization = regularization_values[np.argmin(mse, axis=0),np.arange(regularization_values.shape[1])]
            frf_pinv = np.ascontiguousarray(frf_inverse(frf, method = 'tikhonov',
                                                        regularization_parameter = optimal_regularization))
            reconstructed_force = reconstruction_generator.send(frf_pinv@segment_fft)

        try:
            self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_cv_rse',
                                          'inverse_method':'Tikhonov regularization',
                                          'number_regularization_values_searched':number_regularization_values,
                                          'regularization_parameter':optimal_regularization,
                                          'cross_validation_type':cross_validation_type,
                                          'cola_frame_length':cola_frame_length,
                                          'cola_window':cola_window,
                                          'cola_overlap':cola_overlap_samples,
                                          'FRF_interpolation_type':frf_interpolation_type,
                                          'transformation_interpolation_type':transformation_interpolation_type,
                                          'use_transformation':use_transformation})
        except AttributeError: 
            self.inverse_settings.update({'ISE_technique':'auto_tikhonov_by_cv_rse',
                                          'inverse_method':'Tikhonov regularization',
                                          'number_regularization_values_searched':number_regularization_values,
                                          'regularization_parameter':optimal_regularization,
                                          'cross_validation_type':cross_validation_type,
                                          'cola_frame_length':cola_frame_length,
                                          'cola_window':cola_window,
                                          'cola_overlap':cola_overlap_samples,
                                          'FRF_interpolation_type':frf_interpolation_type,
                                          'transformation_interpolation_type':transformation_interpolation_type,
                                          'use_transformation':use_transformation})

        return reconstructed_force