"""
Includes the tests for the SourcePathReceiverDSet object. 

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
import sdynpy as sdpy
import forcefinder as ff
import pytest

def make_linear_spectra_data(number_dofs=4, number_transformed_dofs=2, dof_offset=0, 
                             sampling_frequency=4, frequency_resolution=1):
    """
    Generates random linear spectra data for unit tests on the SourcePathReceiver 
    object. The FRF and response ordinates are scaled by a random vector so each 
    set of data is different.

    Parameters
    ----------
    number_dofs : int, optional
        The number of DOFs to include in the data. Each DOF is a driving point. 
        The default number of DOFs is four.
    number_transformed_dofs : int, optiona
        The number of DOFs to use in the transformations. The default number is
        two. 
    dof_offset : int, optional
        The offset to apply to the DOFs so there can be unique DOFs for different 
        data sets. The default is zero.
    sampling_frequency : float, optional
        The sampling frequency for the data. The default is 4 Hz.
    frequency_resolution : float, optional
        The frequency resolution for the data. The default is 1 Hz.

    Returns
    -------
    frf : TransferFunctionArray
        The FRFs for the dataset.
    response : SpectrumArray
        The responses for the dataset.
    force : SpectrumArray
        The force for the dataset.
    response_transformation : Matrix
        The response transformation for the dataset.
    reference_transformation : Matrix
        The reference transformation for the dataset.

    Notes
    -----
    The data is nonsensical and should not be interpreted physically.
    """
    dof = sdpy.coordinate_array(node=np.arange(number_dofs)+1+dof_offset, direction=1)
    transformed_dof = sdpy.coordinate_array(node=np.arange(number_transformed_dofs)+1+dof_offset+100, direction=1)
    frf_dof = sdpy.coordinate.outer_product(dof, dof)
    abscissa = np.arange(sampling_frequency/(frequency_resolution*2)+1)*frequency_resolution

    ordinate_multiplier = np.round(np.random.randn(number_dofs),2)

    force_ordinate = np.array([(np.random.randn(number_dofs))*ii for ii in range(1,abscissa.shape[0]+1)])
    response_ordinate = force_ordinate.copy()*ordinate_multiplier
    frf_ordinate = np.array([np.eye(number_dofs)*ordinate_multiplier]*abscissa.shape[0])

    response = sdpy.spectrum_array(abscissa, np.moveaxis(response_ordinate,0,-1), dof[...,np.newaxis])
    force = sdpy.spectrum_array(abscissa, np.moveaxis(force_ordinate,0,-1), dof[...,np.newaxis])
    frf = sdpy.transfer_function_array(abscissa, np.moveaxis(frf_ordinate,0,-1), frf_dof)

    response_transformation = sdpy.matrix(np.random.randn(transformed_dof.shape[0], dof.shape[0]),
                                          transformed_dof, dof)
    reference_transformation = sdpy.matrix(np.random.randn(transformed_dof.shape[0], dof.shape[0]),
                                           transformed_dof, dof)

    return frf, response, force, response_transformation, reference_transformation

@pytest.fixture(scope='function')
def default_dataset_realization_1():
    return make_linear_spectra_data()

@pytest.fixture(scope='function')
def default_dataset_realization_2():
    return make_linear_spectra_data()

@pytest.fixture(scope='function')
def larger_dataset_realization_1():
    return make_linear_spectra_data(number_dofs=8)

@pytest.fixture(scope='function')
def larger_dataset_different_dofs():
    return make_linear_spectra_data(number_dofs=8, dof_offset=10)

def test_data_recovery(default_dataset_realization_1, 
                       default_dataset_realization_2):
    spr_data1 = ff.LinearSourcePathReceiverData({'frfs_1':default_dataset_realization_1[0]}, 
                                                {'training_frfs_1':default_dataset_realization_2[0]},
                                                {'target_response_1':default_dataset_realization_1[1]},
                                                {'training_response_1':default_dataset_realization_2[1]},
                                                {'force_realization_1':default_dataset_realization_1[2]},
                                                {'res_xform_realization_1':default_dataset_realization_1[3]},
                                                {'ref_xform_realization_1':default_dataset_realization_1[4]},
                                                default_dataset_realization_1[1].response_coordinate)
    
    spr_data2 = ff.LinearSourcePathReceiverData({'frfs_2':default_dataset_realization_2[0]}, 
                                                {'training_frfs_2':default_dataset_realization_1[0]},
                                                {'target_response_2':default_dataset_realization_2[1]},
                                                {'training_response_2':default_dataset_realization_1[1]},
                                                {'force_realization_2':default_dataset_realization_2[2]},
                                                {'res_xform_realization_2':default_dataset_realization_2[3]},
                                                {'ref_xform_realization_2':default_dataset_realization_2[4]},
                                                default_dataset_realization_2[1].response_coordinate)
    
    spr_set = ff.SourcePathReceiverSet({'dataset_1':spr_data1,
                                        'dataset_2':spr_data2})
    
    assert spr_set[0] == spr_data1
    assert spr_set[1] == spr_data2

def test_data_recovery_with_append(default_dataset_realization_1, 
                                   default_dataset_realization_2):
    spr_data1 = ff.LinearSourcePathReceiverData({'frfs_1':default_dataset_realization_1[0]}, 
                                                {'training_frfs_1':default_dataset_realization_2[0]},
                                                {'target_response_1':default_dataset_realization_1[1]},
                                                {'training_response_1':default_dataset_realization_2[1]},
                                                {'force_realization_1':default_dataset_realization_1[2]},
                                                {'res_xform_realization_1':default_dataset_realization_1[3]},
                                                {'ref_xform_realization_1':default_dataset_realization_1[4]},
                                                default_dataset_realization_1[1].response_coordinate)
    
    spr_data2 = ff.LinearSourcePathReceiverData({'frfs_2':default_dataset_realization_2[0]}, 
                                                {'training_frfs_2':default_dataset_realization_1[0]},
                                                {'target_response_2':default_dataset_realization_2[1]},
                                                {'training_response_2':default_dataset_realization_1[1]},
                                                {'force_realization_2':default_dataset_realization_2[2]},
                                                {'res_xform_realization_2':default_dataset_realization_2[3]},
                                                {'ref_xform_realization_2':default_dataset_realization_2[4]},
                                                default_dataset_realization_2[1].response_coordinate)
    
    spr_set = ff.SourcePathReceiverSet({'dataset_1':spr_data1})
    
    assert spr_set[0] == spr_data1

    spr_set.append({'dataset_2':spr_data2})
    assert spr_set[0] == spr_data1
    assert spr_set[1] == spr_data2

def test_data_recovery_missing_data(default_dataset_realization_1, 
                                    default_dataset_realization_2):
    spr_data1 = ff.LinearSourcePathReceiverData(frfs={'frfs_1':default_dataset_realization_1[0]}, 
                                                target_response={'target_response_1':default_dataset_realization_1[1]},
                                                force={'force_realization_1':default_dataset_realization_1[2]})
    
    spr_missing_data2 = ff.LinearSourcePathReceiverData(frfs={'frfs_1':None}, 
                                                        target_response={'target_response_2':default_dataset_realization_2[1]},
                                                        force={'force_realization_1':None})
    
    spr_data2 = ff.LinearSourcePathReceiverData(frfs={'frfs_1':default_dataset_realization_1[0]}, 
                                                target_response={'target_response_2':default_dataset_realization_2[1]},
                                                force={'force_realization_1':default_dataset_realization_1[2]})
    
    spr_set = ff.SourcePathReceiverSet({'dataset_1':spr_data1,
                                        'dataset_2':spr_missing_data2})
    
    assert spr_set[0] == spr_data1
    assert spr_set[1] == spr_data2