"""
Defines the SourcePathReceiverSet object, which is used to gather the 
data for several inverse source estimation problems.

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
from __future__ import annotations
import numpy as np
import sdynpy as sdpy
from dataclasses import dataclass, field
from typing import List
from .utilities import is_dataset_labeled
from .source_path_receiver_data import SourcePathReceiverData
import forcefinder.core.source_path_receiver_data as spr_data

@dataclass
class DataPool:
    """
    A basic dataclass to store the data from multiple test runs. 
    """
    label: List[str] = field(default_factory=list)
    ordinate: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.label, list):
            self.label = [self.label]
        if not isinstance(self.ordinate, list):
            self.ordinate = [self.ordinate]
        self.append('none', None)
        
    def append(self, new_label: str, new_data: np.ndarray):
        """
        Adds new data to the DataPool object.

        Parameters
        ----------
        label : str
            The label for the data that is being added to the 
            DataPool object.
        data : ndarray
            The data to add to the DataPool object.

        Notes
        -----
        This method can only append one ndarray to the DataPool 
        object at time. 
        """
        self.label.append(new_label)
        self.ordinate.append(new_data)

    def __len__(self) -> int:
        return len(self.label)
    
    def __getitem__(self, idx: int) -> DataPool:
        if isinstance(idx, int):
            if idx > len(self)-1:
                raise IndexError('Index {:} is out of bounds for DataPool with size {:}'.format(idx, len(self)))
            #if len(self) == 1:
            #    return spr_data.LabeledData(self.label, self.ordinate)
            #else:    
            return spr_data.LabeledData(self.label[idx], self.ordinate[idx])
        else:
            raise TypeError('The DataPool class can only be indexed by single integers')
        
@dataclass
class PoolIdx:
    """
    A basic dataclass to track the mapping between the various pools of data and a dataset.
    """
    frf_pool_id: int = None
    training_frf_pool_id: int = None
    force_pool_id: int = None
    target_response_pool_id: int = None
    training_response_pool_id: int = None
    response_transformation_pool_id: int = None
    response_coordinate_pool_id: int = None
    reference_transformation_pool_id: int = None
    target_response_coordinate_pool_id: int = None
    reference_coordinate_pool_id: int = None
    training_response_coordinate_pool_id: int = None
    transformed_response_coordinate_pool_id: int = None
    abscissa_pool_id: int = None
    transformed_reference_coordinate_pool_id: int = None
    set_label_id: int = None
    spr_type: int = None

data_attribute_pairs = {'_frf_array_':'_frf_pool_', 
                        '_training_frf_array_':'_training_frf_pool_',
                        '_target_response_array_':'_target_response_pool_',
                        '_training_response_array_':'_training_response_pool_',
                        '_force_array_':'_force_pool_',
                        '_response_transformation_array_':'_response_transformation_pool_',
                        '_reference_transformation_array_':'_reference_transformation_pool_'}

coordinate_attribute_pairs = {'_response_coordinate_':'_response_coordinate_pool_',
                              '_target_response_coordinate_':'_target_response_coordinate_pool_',
                              '_training_response_coordinate_':'_training_response_coordinate_pool_',
                              '_reference_coordinate_':'_reference_coordinate_pool_',
                              '_transformed_response_coordinate_':'_transformed_response_coordinate_pool_',
                              '_transformed_reference_coordinate_':'_transformed_reference_coordinate_pool_'}

class SourcePathReceiverSet:
    """
    A parent class for sets of SourcePathReceiverData objects. 
    """
    __slots__ = ['_frf_pool_', '_training_frf_pool_', '_force_pool_', '_target_response_pool_',
                 '_training_response_pool_', '_response_transformation_pool_', '_response_coordinate_pool_',
                 '_reference_transformation_pool_', '_target_response_coordinate_pool_', '_reference_coordinate_pool_',
                 '_training_response_coordinate_pool_', '_transformed_response_coordinate_pool_', '_abscissa_pool_',
                 '_transformed_reference_coordinate_pool_', '_set_label_', '_pool_ids_']
    
    _spr_type_ = ['LinearSourcePathReceiverData', 
                  'PowerSourcePathReceiverData', 
                  'TransientSourcePathReceiverData']
    
    def __init__(self, datasets: dict | None = None):
        self._frf_pool_ = DataPool()
        self._training_frf_pool_ = DataPool()
        self._target_response_pool_ = DataPool()
        self._training_response_pool_ = DataPool()
        self._force_pool_ = DataPool()
        self._response_transformation_pool_ = DataPool()
        self._reference_transformation_pool_ = DataPool()
        self._abscissa_pool_ = []
        self._response_coordinate_pool_ = []
        self._target_response_coordinate_pool_ = []
        self._training_response_coordinate_pool_ = []
        self._reference_coordinate_pool_ = []
        self._transformed_response_coordinate_pool_ = []
        self._transformed_reference_coordinate_pool_ = []
        self._set_label_ = []
        self._pool_ids_ = []
        
        self.append(datasets)
    
    def __len__(self) -> int:
        return len(self._pool_ids_)
    
    
    def __getitem__(self, idx: int) -> SourcePathReceiverData:
        if isinstance(idx, int):
            if idx > len(self)-1:
                raise IndexError('Index {:} is out of bounds for SourcePathReceiverSet with length {:}'.format(idx, len(self)))
            
            empty_object = getattr(spr_data, self._spr_type_[self._pool_ids_[idx].spr_type]).empty()
            for attribute in empty_object.attributes:
                if attribute == '_abscissa_':
                    empty_object._abscissa_ = self._abscissa_pool_[self._pool_ids_[idx].abscissa_pool_id]
                if attribute in data_attribute_pairs:
                    attribute_pool_id = getattr(self._pool_ids_[idx], data_attribute_pairs[attribute][1:]+'id')
                    loop_data = getattr(self, data_attribute_pairs[attribute])[attribute_pool_id]
                    if loop_data.data is None:
                        empty_object.__setattr__(attribute, None)
                    else:
                        empty_object.__setattr__(attribute, loop_data)
                if attribute in coordinate_attribute_pairs:
                    attribute_pool_id = getattr(self._pool_ids_[idx], coordinate_attribute_pairs[attribute][1:]+'id')
                    empty_object.__setattr__(attribute, getattr(self, coordinate_attribute_pairs[attribute])[attribute_pool_id])
            
            empty_object.freeze()

            return empty_object
        else:
            raise TypeError('The SourcePathReceiverSet class can only be indexed by single integers')
    
    def __repr__(self):
        return repr('SourcePathReceiverSet object with {:} datasets'.format(len(self)))

    def append(self, datasets: dict):
        """
        Adds a SourcePathReceiverData object to the set. 

        Parameters
        ----------
        datasets : SourcePathReceiverData
            The SourcePathReceiverData objects to add to the set. It should be supplied as a 
            dictionary where the keys are the dataset labels and the values are the 
            SourcePathReceiverData object.
        """
        if not isinstance(datasets, dict):
            raise ValueError('The dataset must be supplied as a dictionary where the key is the dataset labels')
        
        for dataset in datasets:
            loop_dataset = datasets[dataset]
            if not is_dataset_labeled(loop_dataset):
                raise ValueError('All the data in the supplied dataset should be labeled')

            self._set_label_.append(dataset)
            
            dataset_pool_ids = PoolIdx()
            dataset_pool_ids.set_label_id = self._set_label_.index(dataset)
            for ii, spr_type in enumerate(self._spr_type_):
                if spr_type in loop_dataset.__class__.__name__:
                    dataset_pool_ids.spr_type = ii

            for attribute in loop_dataset.attributes:
                loop_attribute = getattr(loop_dataset, attribute)

                # Use a series of logic to find or set the data in the data pool if thats
                # how it is stored under the hood. The DataPool object is used to minimize
                # the duplication of potentially large data arrays. 
                if attribute in data_attribute_pairs:
                    class_attribute = getattr(self, data_attribute_pairs[attribute])
                    if loop_attribute is None:
                        dataset_pool_ids.__setattr__(data_attribute_pairs[attribute][1:]+'id', 
                                                     class_attribute.label.index('none'))
                    elif loop_attribute.label in class_attribute.label:
                        if class_attribute[class_attribute.label.index(loop_attribute.label)].data is None:
                            class_attribute[class_attribute.label.index(loop_attribute.label)].data = loop_attribute.data
                        elif not np.all(class_attribute[class_attribute.label.index(loop_attribute.label)].data == loop_attribute.data):
                            if loop_attribute.data is not None:
                                # Don't want to fail this check if the data is None
                                raise ValueError('The {:} in dataset {:} does not match the data with the same name in the SPR set object'.format(loop_attribute.label, dataset))
                        dataset_pool_ids.__setattr__(data_attribute_pairs[attribute][1:]+'id', 
                                                     class_attribute.label.index(loop_attribute.label))
                    else:
                        dataset_pool_ids.__setattr__(data_attribute_pairs[attribute][1:]+'id',
                                                    len(class_attribute))
                        class_attribute.append(loop_attribute.label, loop_attribute.data)
                        
                # Simply appending the different coordinates to a list, since they don't take
                # up much memory.
                if attribute in coordinate_attribute_pairs:
                    class_attribute = getattr(self, coordinate_attribute_pairs[attribute])
                    dataset_pool_ids.__setattr__(coordinate_attribute_pairs[attribute][1:]+'id', 
                                                 len(class_attribute))
                    class_attribute.append(loop_attribute)
                    
                # Manually adding the abscissa, not worried about duplication here since the 
                # vector doesn't take up much memory. 
                if attribute == '_abscissa_':
                    dataset_pool_ids.abscissa_pool_id = len(self._abscissa_pool_)
                    self._abscissa_pool_.append(loop_attribute)
                    
            self._pool_ids_.append(dataset_pool_ids)