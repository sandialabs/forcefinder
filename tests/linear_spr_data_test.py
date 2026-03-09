"""
Includes the tests for the SourcePathReceiverData object. 

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

"""
Basic checks to provide data to the object and then make sure that all the 
data is stored in the object as expected.

Needed tests:
    - Have all the data, but the training dofs are a subset of the full data
        - a passing object construction where everything matches
        - a failing object construction where the training DOFs aren't in the full DOFs
        - need to check supplying training DOFs with the correct and incorrect overlaps
    - 
"""

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
def larger_dataset_realization_2():
    return make_linear_spectra_data(number_dofs=8)

@pytest.fixture(scope='function')
def larger_dataset_different_dofs():
    return make_linear_spectra_data(number_dofs=4, dof_offset=10)

def test_basic_construction(default_dataset_realization_1,
                            default_dataset_realization_2):
    spr_data = ff.LinearSourcePathReceiverData(default_dataset_realization_1[0], 
                                               default_dataset_realization_2[0],
                                               default_dataset_realization_1[1],
                                               default_dataset_realization_2[1],
                                               default_dataset_realization_1[2],
                                               default_dataset_realization_1[3],
                                               default_dataset_realization_1[4],
                                               default_dataset_realization_1[1].response_coordinate)
    
    assert np.all(spr_data._frf_array_ == np.moveaxis(default_dataset_realization_1[0].ordinate,-1,0))
    assert np.all(spr_data._training_frf_array_ == np.moveaxis(default_dataset_realization_2[0].ordinate,-1,0))
    assert np.all(spr_data._target_response_array_ == np.moveaxis(default_dataset_realization_1[1].ordinate,-1,0))
    assert np.all(spr_data._training_response_array_ == np.moveaxis(default_dataset_realization_2[1].ordinate,-1,0))
    assert np.all(spr_data._force_array_ == np.moveaxis(default_dataset_realization_1[2].ordinate,-1,0))
    assert np.all(spr_data._response_coordinate_ == default_dataset_realization_1[0][:,0].response_coordinate)
    assert np.all(spr_data._target_response_coordinate_ == default_dataset_realization_1[1].response_coordinate)
    assert np.all(spr_data._training_response_coordinate_ == default_dataset_realization_2[1].response_coordinate)
    assert np.all(spr_data._reference_coordinate_ == default_dataset_realization_1[0][0,:].reference_coordinate)
    assert np.all(spr_data._response_transformation_array_ == default_dataset_realization_1[3].matrix)
    assert np.all(spr_data._transformed_response_coordinate_ == default_dataset_realization_1[3].row_coordinate)
    assert np.all(spr_data._reference_transformation_array_ == default_dataset_realization_1[4].matrix)
    assert np.all(spr_data._transformed_reference_coordinate_ == default_dataset_realization_1[4].row_coordinate)


def test_construction_without_training_data(default_dataset_realization_1):
    spr_data = ff.LinearSourcePathReceiverData(frfs=default_dataset_realization_1[0], 
                                               target_response=default_dataset_realization_1[1])
    
    assert np.all(spr_data._frf_array_ == np.moveaxis(default_dataset_realization_1[0].ordinate,-1,0))
    assert spr_data._training_frf_array_ == None
    assert np.all(spr_data.training_frfs.ordinate == default_dataset_realization_1[0].ordinate)
    assert np.all(spr_data._target_response_array_ == np.moveaxis(default_dataset_realization_1[1].ordinate,-1,0))
    assert spr_data._training_response_array_ == None
    assert np.all(spr_data.training_response.ordinate == default_dataset_realization_1[1].ordinate)
    assert spr_data._training_response_coordinate_ == None
    assert np.all(spr_data.training_response_coordinate == default_dataset_realization_1[1].response_coordinate)

def test_construction_without_target_data(default_dataset_realization_1):
    spr_data = ff.LinearSourcePathReceiverData(training_frfs=default_dataset_realization_1[0], 
                                               training_response=default_dataset_realization_1[1])
    
    assert np.all(spr_data._training_frf_array_ == np.moveaxis(default_dataset_realization_1[0].ordinate,-1,0))
    assert spr_data._frf_array_ == None
    assert np.all(spr_data.frfs.ordinate == default_dataset_realization_1[0].ordinate)
    assert np.all(spr_data._training_response_array_ == np.moveaxis(default_dataset_realization_1[1].ordinate,-1,0))
    assert spr_data._target_response_array_ == None
    assert np.all(spr_data.target_response.ordinate == default_dataset_realization_1[1].ordinate)
    assert spr_data._target_response_coordinate_ == None
    assert np.all(spr_data.target_response_coordinate == default_dataset_realization_1[1].response_coordinate)

def test_freeze_thaw(default_dataset_realization_1):
    spr_data = ff.LinearSourcePathReceiverData(training_response=default_dataset_realization_1[1])

    with pytest.raises(AttributeError, match='LinearSourcePathReceiverData cannot be modified after initialization'):
        spr_data.frfs = default_dataset_realization_1[0]

    spr_data.thaw()
    spr_data.frfs = default_dataset_realization_1[0]
    assert np.all(spr_data.frfs.ordinate == default_dataset_realization_1[0].ordinate)

def test_immutability(default_dataset_realization_1,
                      default_dataset_realization_2):
    spr_data = ff.LinearSourcePathReceiverData(default_dataset_realization_1[0], 
                                               default_dataset_realization_2[0],
                                               default_dataset_realization_1[1],
                                               default_dataset_realization_2[1],
                                               default_dataset_realization_1[2],
                                               default_dataset_realization_1[3],
                                               default_dataset_realization_1[4],
                                               default_dataset_realization_1[1].response_coordinate)
    spr_data.thaw()

    with pytest.raises(AttributeError, match='The FRFs cannot be reset once the object is initialized'):
        spr_data.frfs = default_dataset_realization_2[0]
    assert np.all(spr_data._frf_array_==np.moveaxis(default_dataset_realization_1[0].ordinate,-1,0))

    with pytest.raises(AttributeError, match='The training FRFs of an SPR object cannot be reset once the object is initialized'):
        spr_data.training_frfs = default_dataset_realization_1[0]
    assert np.all(spr_data._training_frf_array_==np.moveaxis(default_dataset_realization_2[0].ordinate,-1,0))

    with pytest.raises(AttributeError, match='The force data cannot be reset once the object is initialized'):
        spr_data.force = default_dataset_realization_2[2]
    assert np.all(spr_data._force_array_==np.moveaxis(default_dataset_realization_1[2].ordinate,-1,0))

    with pytest.raises(AttributeError, match='The target response cannot be reset once the object is initialized'):
        spr_data.target_response = default_dataset_realization_2[1]
    assert np.all(spr_data._target_response_array_==np.moveaxis(default_dataset_realization_1[1].ordinate,-1,0))

    with pytest.raises(AttributeError, match='The training responses cannot be reset once the object is initialized'):
        spr_data.training_response = default_dataset_realization_1[1]
    assert np.all(spr_data._training_response_array_==np.moveaxis(default_dataset_realization_2[1].ordinate,-1,0))

    with pytest.raises(AttributeError, match='The response transformation cannot be reset once the object is initialized'):
        spr_data.response_transformation = default_dataset_realization_2[3]
    assert np.all(spr_data._response_transformation_array_==default_dataset_realization_1[3].matrix)

    with pytest.raises(AttributeError, match='The reference transformation cannot be reset once the object is initialized'):
        spr_data.reference_transformation = default_dataset_realization_2[4].matrix
    assert np.all(spr_data._reference_transformation_array_==default_dataset_realization_1[4].matrix)

    # Can't check that these stayed the same because the different 
    # datasets have the same values for these attributes
    with pytest.raises(AttributeError, match='The response coordinate cannot be reset after it is initialized'):
        spr_data.response_coordinate = default_dataset_realization_1[0][:,0].response_coordinate

    with pytest.raises(AttributeError, match='The reference coordinate cannot be reset after it is initialized'):
        spr_data.reference_coordinate = default_dataset_realization_1[0][0,:].reference_coordinate

    with pytest.raises(AttributeError, match='The target response coordinate cannot be reset after it is initialized'):
        spr_data.target_response_coordinate = default_dataset_realization_1[1].response_coordinate

    with pytest.raises(AttributeError, match='The training response coordinate cannot be reset after it is initialized'):
        spr_data.training_response_coordinate = default_dataset_realization_2[1].response_coordinate

    with pytest.raises(AttributeError, match='The abscissa cannot be reset after it is initialized'):
        spr_data.abscissa = default_dataset_realization_2[1][0].abscissa

def test_slots(default_dataset_realization_1):
    spr_data = ff.LinearSourcePathReceiverData(target_response=default_dataset_realization_1[1])
    spr_data.thaw()
    with pytest.raises(AttributeError):
        spr_data._test_attribute_ = 5

def test_no_response(default_dataset_realization_1):
    with pytest.raises(AttributeError, match='Response data is required to initialize SourcePathReceiverData object'):
        ff.LinearSourcePathReceiverData(frfs=default_dataset_realization_1[0])

def test_good_training_data(larger_dataset_realization_1,
                            larger_dataset_realization_2):
    spr_data = ff.LinearSourcePathReceiverData(larger_dataset_realization_1[0], 
                                               larger_dataset_realization_2[0][:4,:],
                                               larger_dataset_realization_1[1],
                                               larger_dataset_realization_2[1][:4],
                                               training_response_coordinate=larger_dataset_realization_2[1][:4].response_coordinate)
    
    assert np.all(spr_data._frf_array_ == np.moveaxis(larger_dataset_realization_1[0].ordinate,-1,0))
    assert np.all(spr_data._training_frf_array_ == np.moveaxis(larger_dataset_realization_2[0][:4,:].ordinate,-1,0))
    assert np.all(spr_data._target_response_array_ == np.moveaxis(larger_dataset_realization_1[1].ordinate,-1,0))
    assert np.all(spr_data._training_response_array_ == np.moveaxis(larger_dataset_realization_2[1][:4].ordinate,-1,0))
    assert np.all(spr_data._response_coordinate_ == larger_dataset_realization_1[0][:,0].response_coordinate)
    assert np.all(spr_data._target_response_coordinate_ == larger_dataset_realization_1[1].response_coordinate)
    assert np.all(spr_data._training_response_coordinate_ == larger_dataset_realization_2[1][:4].response_coordinate)

    spr_data1 = ff.LinearSourcePathReceiverData(frfs=larger_dataset_realization_1[0], 
                                                target_response=larger_dataset_realization_1[1],
                                                training_response_coordinate=larger_dataset_realization_1[1][:4].response_coordinate)
    
    assert np.all(spr_data1._frf_array_ == np.moveaxis(larger_dataset_realization_1[0].ordinate,-1,0))
    assert np.all(spr_data1.training_frfs.ordinate == larger_dataset_realization_1[0][:4,:].ordinate)
    assert np.all(spr_data1._target_response_array_ == np.moveaxis(larger_dataset_realization_1[1].ordinate,-1,0))
    assert np.all(spr_data1.training_response.ordinate == larger_dataset_realization_1[1][:4].ordinate)
    assert np.all(spr_data1._response_coordinate_ == larger_dataset_realization_1[0][:,0].response_coordinate)
    assert np.all(spr_data1._target_response_coordinate_ == larger_dataset_realization_1[1].response_coordinate)
    assert np.all(spr_data1._training_response_coordinate_ == larger_dataset_realization_1[1][:4].response_coordinate)

def test_bad_training_data(larger_dataset_realization_1,
                            larger_dataset_different_dofs):
    with pytest.raises(ValueError):
        ff.LinearSourcePathReceiverData(target_response=larger_dataset_realization_1[1],
                                training_response_coordinate=larger_dataset_different_dofs[1].response_coordinate)
    
    with pytest.raises(ValueError):
        ff.LinearSourcePathReceiverData(target_response=larger_dataset_realization_1[1],
                                training_response=larger_dataset_different_dofs[1])
        
    with pytest.raises(ValueError):
        ff.LinearSourcePathReceiverData(target_response=larger_dataset_realization_1[1],
                                training_frfs=larger_dataset_different_dofs[0])
        
    with pytest.raises(ValueError):
        ff.LinearSourcePathReceiverData(target_response=larger_dataset_realization_1[1],
                                        training_response=larger_dataset_realization_1[1][:4],
                                        training_frfs=larger_dataset_different_dofs[0])
        
    with pytest.raises(ValueError):
        ff.LinearSourcePathReceiverData(target_response=larger_dataset_realization_1[1],
                                        training_response=larger_dataset_realization_1[1][:4],
                                        training_response_coordinate=larger_dataset_different_dofs[0][:4].response_coordinate)