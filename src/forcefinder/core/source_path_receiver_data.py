"""
Defines the SourcePathReceiverData object, which is used to gather the 
data for inverse source estimation problems.

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
from dataclasses import dataclass
import numpy as np
import sdynpy as sdpy
from .utilities import check_abscissa
from sdynpy.core.sdynpy_coordinate import outer_product

@dataclass
class LabeledData:
    """
    A basic dataclass to store the data attributes in the SourcePathReceiverData class. This only
    has two attributes; label and data. 
    """
    label:str
    data:np.ndarray

class SourcePathReceiverData:
    """
    A basic class to store the data that is used to compile the data for a single realization of a 
    test. At a minimum, it contains responses but it can optionally contain all the data that is 
    normally stored in a SourcePathReceiver object.
    """
    __slots__ = ['_frf_array_', '_training_frf_array_', '_force_array_', '_target_response_array_',
                 '_training_response_array_', '_response_transformation_array_', '_response_coordinate_',
                 '_reference_transformation_array_', '_target_response_coordinate_', '_reference_coordinate_',
                 '_training_response_coordinate_', '_transformed_response_coordinate_', '_abscissa_',
                 '_transformed_reference_coordinate_', '_frozen_']

    def __init__(self, frfs=None, training_frfs=None, target_response=None, training_response=None, 
                 force=None, response_transformation=None, reference_transformation=None, 
                 training_response_coordinate=None):
        self._frozen_=False
        self._frf_array_=None #
        self._training_frf_array_=None #
        self._force_array_=None #
        self._target_response_array_=None #
        self._training_response_array_=None #
        self._response_transformation_array_=None #
        self._reference_transformation_array_=None #
        self._response_coordinate_=None #
        self._target_response_coordinate_=None #
        self._training_response_coordinate_=None #
        self._reference_coordinate_=None #
        self._transformed_response_coordinate_=None
        self._transformed_reference_coordinate_=None
        self._abscissa_=None #
        
        # Setting the responses, the logic in the setters also sets the target response coordinate, 
        # training response coordinate, and abscissa
        if target_response is None and training_response is None:
            raise AttributeError('Response data is required to initialize SourcePathReceiverData object')
        if target_response is not None:
            self.target_response = target_response
        if training_response is not None:
            self.training_response = training_response

        # Setting the FRFs, the logic in the setters also sets the response and reference coordinate
        if frfs is not None:
            self.frfs = frfs
        if training_frfs is not None:
            self.training_frfs = training_frfs

        if training_response_coordinate is not None:
            if self._training_response_coordinate_ is None:
                self.training_response_coordinate = training_response_coordinate
            else:
                if not np.all(np.sort(training_response_coordinate) == self._training_response_coordinate_):
                    raise ValueError('The supplied training response coordinate does not match the data in the object')

        if force is not None:
            self.force = force

        # Setting the transformation matrices, the logic in the setters also set the response and 
        # reference coordinate
        if response_transformation is not None:
            self.response_transformation = response_transformation
        if reference_transformation is not None:
            self.reference_transformation = reference_transformation

        self.freeze()

    @classmethod
    def empty(cls):
        return cls.__new__(cls)

    def __setattr__(self, name, data):
        if hasattr(self, '_frozen_'):
            if name != '_frozen_' and self._frozen_:
                raise AttributeError(f'{self.__class__.__name__} cannot be modified after initialization')
        super().__setattr__(name, data)

    #def __repr__(self):
        # need to figure out how to look up all the attributes and find which is none
        # return repr('{:} object with {:} reference coordinates, {:} target response coordinates, and {:} training response coordinates'.format())

    def freeze(self):
        """
        Freezes the instance of the class so attributes cannot be modified or set.
        """
        self._frozen_ = True

    def thaw(self):
        """
        Thaws the instance of the class so attributes can be set. Note that the 
        class properties are treated as immutable, so the attributes cannot be 
        modified with the property setter.
        """
        self._frozen_ = False

    @property
    def target_response(self):
        pass
    
    @target_response.setter
    def target_response(self):
        pass

    @property
    def target_response_coordinate(self):
        if self._target_response_coordinate_ is None:
            return self._training_response_coordinate_
        else:
            return self._target_response_coordinate_
    
    @target_response_coordinate.setter
    def target_response_coordinate(self, coordinate_array):
        if self._target_response_coordinate_ is not None:
            raise AttributeError('The target response coordinate cannot be reset after it is initialized')
        self._target_response_coordinate_ = coordinate_array

    @property
    def abscissa(self):
        return self._abscissa_
    
    @abscissa.setter
    def abscissa(self, data_array):
        if self._abscissa_ is not None:
            raise AttributeError('The abscissa cannot be reset after it is initialized')
        spacing = np.diff(data_array, axis=-1)
        mean_spacing = np.mean(spacing)
        if not np.allclose(spacing, mean_spacing):
            raise ValueError('The supplied abscissa must be evenly spaced')
        self._abscissa_ = data_array

    @property
    def training_response(self):
        pass
    
    @training_response.setter
    def training_response(self):
        pass

    @property
    def training_response_coordinate(self):
        if self._training_response_coordinate_ is None:
            return self._target_response_coordinate_
        else:
            return self._training_response_coordinate_
    
    @training_response_coordinate.setter
    def training_response_coordinate(self, coordinate_array):
        if self._training_response_coordinate_ is not None:
            raise AttributeError('The training response coordinate cannot be reset after it is initialized')
        if self._target_response_coordinate_ is not None:
            if not np.all(np.isin(coordinate_array, self._target_response_coordinate_)):
                        raise ValueError('The training response coordinate {:} is missing the target response coordinate'.
                                format(coordinate_array[~np.isin(coordinate_array, self.target_response_coordinate)].string_array()))
        self._training_response_coordinate_ = np.sort(coordinate_array)

    @property
    def frfs(self):
        if self._frf_array_ is None:
            if self._training_frf_array_ is not None:
                return self.training_frfs
            else:
                return self._frf_array_
        else:
            if self._frf_array_.data is None:
                return self._frf_array_.data
            else:
                return sdpy.transfer_function_array(self._abscissa_, np.moveaxis(self._frf_array_.data, 0, -1), 
                                        outer_product(self._response_coordinate_, self._reference_coordinate_))
    
    @frfs.setter
    def frfs(self, data):
        if self._frf_array_ is not None:
            raise AttributeError('The FRFs cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            data_array = data[label]
        else:
            label = ''
            data_array = data
        if data_array is None:
            self._frf_array_ = LabeledData(label, data_array)
        else:
            if not isinstance(data_array, sdpy.core.sdynpy_data.TransferFunctionArray):
                raise TypeError('The FRFs must be a SDynPy TransferFunctionArray')
            
            check_abscissa(data_array, self._abscissa_)

            data_array = data_array.reshape_to_matrix()

            self.response_coordinate = data_array[:, 0].response_coordinate
            self.reference_coordinate = data_array[0, :].reference_coordinate
            self._frf_array_ = LabeledData(label, np.moveaxis(data_array.ordinate, -1, 0))

    @property
    def response_coordinate(self):
        return self._response_coordinate_
    
    @response_coordinate.setter
    def response_coordinate(self, coordinate_array):
        if self._response_coordinate_ is not None:
            raise AttributeError('The response coordinate cannot be reset after it is initialized')
        if not np.all(np.isin(self.target_response_coordinate, coordinate_array)):
            raise ValueError('The FRF response coordinate is missing the {:} target response coordinate'.format(self.target_response_coordinate[~np.isin(self.target_response_coordinate, coordinate_array)].string_array()))
        self._response_coordinate_ = coordinate_array

    @property
    def reference_coordinate(self):
        return self._reference_coordinate_

    @reference_coordinate.setter
    def reference_coordinate(self, coordinate_array):
        if self._reference_coordinate_ is not None:
            raise AttributeError('The reference coordinate cannot be reset after it is initialized')
        self._reference_coordinate_ = coordinate_array
    
    @property
    def training_frfs(self):
        if self._training_frf_array_ is None:
            if self._frf_array_ is not None:
                if self._frf_array_.data is None:
                    return self._frf_array_.data
                else:
                    frf_coordinate = outer_product(self.training_response_coordinate, self._reference_coordinate_)
                    return self.frfs[frf_coordinate]
            else:
                return self._training_frf_array_
        else:
            if self._training_frf_array_.data is None:
                return self._training_frf_array_.data
            else:
                return sdpy.transfer_function_array(self._abscissa_, np.moveaxis(self._training_frf_array_.data, 0, -1), 
                                outer_product(self.training_response_coordinate, self._reference_coordinate_))

    @training_frfs.setter
    def training_frfs(self, data):
        if self._training_frf_array_ is not None:
            raise AttributeError('The training FRFs of an SPR object cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            data_array = data[label]
        else:
            label = ''
            data_array = data
        if data_array is None:
            self._training_frf_array_ = LabeledData(label, data_array)
        else:
            if not isinstance(data_array, sdpy.core.sdynpy_data.TransferFunctionArray):
                raise TypeError('The training FRFs must be a SDynPy TransferFunctionArray')
            data_array = data_array.reshape_to_matrix()
            if self._training_response_coordinate_ is not None:
                if not np.all(data_array[:, 0].response_coordinate==self._training_response_coordinate_):
                    raise ValueError('The training FRF response DOFs do not match the training response DOFs in the object')
            else:
                self.training_response_coordinate = data_array[:, 0].response_coordinate
            if self._reference_coordinate_ is not None:
                if not np.all(data_array[0, :].reference_coordinate==self._reference_coordinate_):
                    raise ValueError('The training FRF reference DOFs do not match reference DOFs in the object')
            else:
                self.reference_coordinate = data_array[0,:].reference_coordinate
            check_abscissa(data_array, self._abscissa_)
            frf_coordinate = outer_product(self.training_response_coordinate, self._reference_coordinate_)
            self._training_frf_array_ = LabeledData(label, np.moveaxis(data_array[frf_coordinate].ordinate, -1, 0))

    @property
    def force(self):
        pass
    
    @force.setter
    def force(self):
        pass
    
    @property
    def response_transformation(self):
        if self._response_transformation_array_ is None:
            return self._response_transformation_array_
        elif self._response_transformation_array_.data is None:
            return self._response_transformation_array_.data
        else:
            return sdpy.matrix(self._response_transformation_array_.data, self.transformed_response_coordinate, 
                               self.training_response_coordinate)
    
    @response_transformation.setter
    def response_transformation(self, data):
        if self._response_transformation_array_ is not None:
            raise AttributeError('The response transformation cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            transformation_matrix = data[label]
        else:
            label = ''
            transformation_matrix = data
        if transformation_matrix is None:
            self._response_transformation_array_ = LabeledData(label, transformation_matrix)
        else:
            if not isinstance(transformation_matrix, sdpy.Matrix):
                raise TypeError('The response transformation must be defined as a SDynPy Matrix')
            self._transformed_response_coordinate_ = np.sort(transformation_matrix.row_coordinate)
            self._response_transformation_array_ = LabeledData(label, transformation_matrix[self.transformed_response_coordinate, 
                                                                     self.training_response_coordinate])
        
    @property
    def transformed_response_coordinate(self):
        return self._transformed_response_coordinate_
    
    @property
    def reference_transformation(self):
        if self._reference_transformation_array_ is None:
            return self._reference_transformation_array_
        elif self._reference_transformation_array_.data is None:
            return self._reference_transformation_array_.data
        else:
            return sdpy.matrix(self._reference_transformation_array_.data, self.transformed_reference_coordinate, 
                               self.reference_coordinate)
    
    @reference_transformation.setter
    def reference_transformation(self, data):
        if self._reference_transformation_array_ is not None:
            raise AttributeError('The reference transformation cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            transformation_matrix = data[label]
        else:
            label = ''
            transformation_matrix = data
        if transformation_matrix is None:
            self._reference_transformation_array_ = LabeledData(label, transformation_matrix)
        else:
            if not isinstance(transformation_matrix, sdpy.Matrix):
                raise TypeError('The reference transformation must be defined as a SDynPy Matrix')
            self._transformed_reference_coordinate_ = np.sort(transformation_matrix.row_coordinate)
            self._reference_transformation_array_ = LabeledData(label, transformation_matrix[self.transformed_reference_coordinate, 
                                                                      self.reference_coordinate])

    @property
    def transformed_reference_coordinate(self):
        return self._transformed_reference_coordinate_
    
    @property
    def attributes(self):
        attribute_labels = []
        for class_label in self.__class__.__mro__:
            if '__slots__' in class_label.__dict__:
                attribute_labels.extend(class_label.__slots__)
        return attribute_labels
    
class LinearSourcePathReceiverData(SourcePathReceiverData):
    """
    A subclass to represent a SourcePathReceiverData object with linear spectra 
    (i.e., ffts) for the responses or forces.

    Notes
    -----
    The "linear" term in the class name stands for the linear units in the response and
    force spectra.
    """
    __slots__ = ()

    def __init__(self, frfs=None, training_frfs=None, target_response=None, training_response=None, 
                 force=None, response_transformation=None, reference_transformation=None, 
                 training_response_coordinate=None):
        # Inheriting the initial set-up from the parent class
        super().__init__(frfs=frfs, training_frfs=training_frfs, target_response=target_response, 
                         training_response=training_response, force=force, 
                         response_transformation=response_transformation, 
                         reference_transformation=reference_transformation, 
                         training_response_coordinate=training_response_coordinate)

    @property
    def target_response(self):
        if self._target_response_array_ is None:
            if self._training_response_array_ is None:
                return self._target_response_array_   
            else: 
                return sdpy.spectrum_array(self._abscissa_, np.moveaxis(self._training_response_array_.data, 0, -1), 
                                        self.training_response_coordinate[..., np.newaxis])
        else:
            if self._target_response_array_.data is None:
                return self._target_response_array_.data 
            else:
                return sdpy.spectrum_array(self._abscissa_, np.moveaxis(self._target_response_array_.data, 0, -1), 
                                       self.target_response_coordinate[..., np.newaxis])
    
    @target_response.setter
    def target_response(self, data):
        if self._target_response_array_ is not None:
            raise AttributeError('The target response cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            data_array = data[label]
        else:
            label = ''
            data_array = data
        if data_array is None:
            self._target_response_array_ = LabeledData(label, data_array)
        else:
            if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
                raise TypeError('The target response must be a SDynPy SpectrumArray')
            self.target_response_coordinate = np.sort(data_array.response_coordinate)
            self.abscissa = data_array.ravel()[0].abscissa
            self._target_response_array_ = LabeledData(label, 
                            np.moveaxis(data_array[self._target_response_coordinate_[..., np.newaxis]].ordinate, -1, 0))

    @property
    def training_response(self):
        if self._training_response_array_ is None:
            if self._target_response_array_ is None:
                return self._training_response_array_
            else:
                return self.target_response[self.training_response_coordinate[..., np.newaxis]]
        else:
            if self._training_response_array_.data is None:
                return self._training_response_array_.data 
            else:
                return sdpy.spectrum_array(self._abscissa_, np.moveaxis(self._training_response_array_.data, 0, -1), 
                                self.training_response_coordinate[..., np.newaxis])
    
    @training_response.setter
    def training_response(self, data):
        if self._training_response_array_ is not None:
            raise AttributeError('The training responses cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            data_array = data[label]
        else:
            label = ''
            data_array = data
        if data_array is None:
            self._training_response_array_ = LabeledData(label, data_array)
        else:
            if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
                raise TypeError('The training response must be a SDynPy SpectrumArray')
            
            if self._abscissa_ is None:
                self.abscissa = data_array.ravel()[0].abscissa
            elif self._abscissa_ is not None:
                check_abscissa(data_array, self._abscissa_)
            
            self.training_response_coordinate = np.sort(data_array.response_coordinate)
            self._training_response_array_ = LabeledData(label,
                    np.moveaxis(data_array[self._training_response_coordinate_[..., np.newaxis]].ordinate, -1, 0)) 

    @property
    def force(self):
        if self._force_array_ is None:
            return self._force_array_
        else:
            return sdpy.spectrum_array(self._abscissa_, np.moveaxis(self._force_array_.data, 0, -1), 
                                self._reference_coordinate_[..., np.newaxis])
    
    @force.setter
    def force(self, data):
        if self._force_array_ is not None:
            raise AttributeError('The force data cannot be reset once the object is initialized')
        if isinstance(data, dict):
            label = list(data)[0]
            data_array = data[label]
        else:
            label = ''
            data_array = data
        if not isinstance(data_array, sdpy.core.sdynpy_data.SpectrumArray):
            raise TypeError('The force must be a SDynPy SpectrumArray')
        if not np.all(np.isin(data_array.response_coordinate, self.reference_coordinate)):
            raise ValueError('Force {:} is not in the reference_coordinate'.format(data_array.response_coordinate[~np.isin(data_array.response_coordinate, self.reference_coordinate)].string_array()))
        check_abscissa(data_array, self._abscissa_)
        self._force_array_ = LabeledData(label,
                    np.moveaxis(data_array[self.reference_coordinate[..., np.newaxis]].ordinate, -1, 0))