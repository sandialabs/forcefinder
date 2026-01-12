"""
Includes the tests for the LinearSourcePathReceiver inverse methods. 

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
from test_system_generation import (create_beam_system_truth_frfs, 
                                    create_beam_system_ise_frfs, 
                                    create_broadband_random_spectral_excitation,
                                    additive_noise_linear_spectra,
                                    create_rigid_system)
import pytest

#%% Set-up for the tests on the elastic system
@pytest.fixture(scope='module')
def beam_force_spectrum():
    force_spectrum, _ = create_broadband_random_spectral_excitation()
    return force_spectrum

@pytest.fixture(scope='module')
def beam_truth_frfs():
    system_a_truth_frfs, system_b_truth_frfs = create_beam_system_truth_frfs()
    return system_a_truth_frfs, system_b_truth_frfs

@pytest.fixture(scope='module')
def beam_ise_frfs():
    system_a_ise_frfs, system_b_ise_frfs = create_beam_system_ise_frfs()
    return system_a_ise_frfs, system_b_ise_frfs

@pytest.fixture(scope='module')
def system_a_truth_response(beam_truth_frfs, beam_force_spectrum):
    system_a_truth_frfs, _ = beam_truth_frfs
    force_spectrum = beam_force_spectrum
    system_a_truth_spectra_ord = np.moveaxis(system_a_truth_frfs.ordinate,-1,0)@np.moveaxis(force_spectrum.ordinate,0,-1)[...,np.newaxis]
    return sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, force_spectrum.ravel().abscissa[0], 
                           np.moveaxis(system_a_truth_spectra_ord[...,0],0,-1), 
                           system_a_truth_frfs[:,0].response_coordinate[...,np.newaxis])

@pytest.fixture(scope='module')
def system_a_noised_response(system_a_truth_response):
    return additive_noise_linear_spectra(system_a_truth_response)

@pytest.fixture(scope='module')
def system_b_truth_response(beam_truth_frfs, beam_force_spectrum):
    _, system_b_truth_frfs = beam_truth_frfs
    force_spectrum = beam_force_spectrum
    system_b_truth_spectra_ord = np.moveaxis(system_b_truth_frfs.ordinate,-1,0)@np.moveaxis(force_spectrum.ordinate,0,-1)[...,np.newaxis]
    return sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, force_spectrum.ravel().abscissa[0], 
                           np.moveaxis(system_b_truth_spectra_ord[...,0],0,-1), 
                           system_b_truth_frfs[:,0].response_coordinate[...,np.newaxis])

@pytest.fixture(scope='module')
def linear_spr_a_truth(beam_ise_frfs, system_a_truth_response):
    system_a_ise_frfs, _ = beam_ise_frfs
    system_a_linear_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_truth_response)
    system_a_linear_spr.manual_inverse()
    return system_a_linear_spr

@pytest.fixture()
def linear_spr_b(beam_ise_frfs, system_b_truth_response):
    _, system_b_ise_frfs = beam_ise_frfs
    return ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response)
#%% Testing the basic inverse for the linear SPR

def test_system_a_roundtrip(linear_spr_a_truth):
    """
    This test verifies that the linear SPR object can be used to compute a 
    reasonable set of pseudo-forces in an ideal case. It is a five step test:

        1. truth FRFs and truth excitation are created for the test beam system, where
        the truth excitation is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        response of the beam system.
        3. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam.
        4. The ISE FRFs (from step three) and truth responses (from step two) are used to 
        estimate pseudo-forces.
        5. The pseudo-forces (from step four) and ISE FRFs (from step 3) will be used to 
        reconstruct the responses on the receiver beam.  

    The test passes if the reconstructed responses are the "same" as the truth responses,
    based on a NumPy allclose comparison.  
    """
    assert np.allclose(linear_spr_a_truth.target_response.ordinate, linear_spr_a_truth.reconstructed_target_response.ordinate)

def test_truth_a_b_roundtrip(linear_spr_b, linear_spr_a_truth):
    """
    This test verifies that the linear SPR object can be used to compute a predictive set 
    of pseudo-forces in an ideal case. It is a five step test:

        1. truth FRFs and truth excitation are created for two test beam systems, referred
        to as system A and B. The truth excitation is the same for both beam systems and 
        is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        responses of systems A and B.
        3. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam for both system A and B.
        4. The ISE FRFs (from step three) and truth responses (from step two) system A are 
        used to estimate pseudo-forces.
        5. The pseudo-forces from system A (from step four) and ISE FRFs from system B (from 
        step 3) will be used to reconstruct the responses on the receiver beam of system B.  

    The test passes if the reconstructed responses on system B are the "same" as the truth 
    responses on system B, based on a NumPy allclose comparison.  
    """
    linear_spr_b.force = linear_spr_a_truth.force
    assert np.allclose(linear_spr_b.target_response.ordinate, linear_spr_b.reconstructed_target_response.ordinate)

#%% Tests for the auto-regularization methods
def test_auto_tikhonov_by_l_curve(beam_ise_frfs, system_a_noised_response, system_b_truth_response):
    """
    This test verifies that the "auto_tikhonov_by_l_curve" inverse method performs as 
    expected. It is a seven step test:

        1. truth FRFs and truth excitation are created for two test beam systems, referred
        to as system A and B. The truth excitation is the same for both beam systems and 
        is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        responses of systems A and B.
        3. Random error is added to the system A truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam for both system A and B.
        5. The ISE FRFs (from step three) and noised responses (from step three) system A are 
        used to estimate pseudo-forces using the standard pseudo-inverse method.
        6. The ISE FRFs (from step three) and noised responses (from step three) system A are 
        used to estimate pseudo-forces using the prescribed auto-regularization method.
        7. The pseudo-forces from system A (from steps five and 6) and ISE FRFs from system B 
        (from step 3) will be used to reconstruct the responses on the receiver beam of system B.  

    The test passes if two conditions are met:
        1. The summed RMS level of the estimated forces is lower for the auto-regularization 
        method than the standard pseudo-inverse method.
        2. The response prediction accuracy on system B (from step 7 above) is greater for the
        auto-regularization method than the standard pseudo-inverse method. The response 
        prediction accuracy is evaluated with the "average_asd_error" method for the linear SPR.
    """
    system_a_ise_frfs, system_b_ise_frfs = beam_ise_frfs

    system_a_manual_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    system_b_manual_spr = ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    system_a_auto_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_l_curve()

    system_b_auto_spr = ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)

    assert np.abs(system_a_manual_spr.force.ordinate).sum() > np.abs(system_a_auto_spr.force.ordinate).sum()
    assert np.real(system_b_manual_spr.average_asd_error().ordinate).sum() > np.real(system_b_auto_spr.average_asd_error().ordinate).sum()

def test_auto_truncation_by_l_curve(beam_ise_frfs, system_a_noised_response, system_b_truth_response):
    """
    This test verifies that the "auto_truncation_by_l_curve" inverse method performs as 
    expected. It is a seven step test:

        1. truth FRFs and truth excitation are created for two test beam systems, referred
        to as system A and B. The truth excitation is the same for both beam systems and 
        is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        responses of systems A and B.
        3. Random error is added to the system A truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam for both system A and B.
        5. The ISE FRFs (from step three) and noised responses (from step three) system A are 
        used to estimate pseudo-forces using the standard pseudo-inverse method.
        6. The ISE FRFs (from step three) and noised responses (from step three) system A are 
        used to estimate pseudo-forces using the prescribed auto-regularization method.
        7. The pseudo-forces from system A (from steps five and 6) and ISE FRFs from system B 
        (from step 3) will be used to reconstruct the responses on the receiver beam of system B.  

    The test passes if two conditions are met:
        1. The summed RMS level of the estimated forces is lower for the auto-regularization 
        method than the standard pseudo-inverse method.
        2. The response prediction accuracy on system B (from step 7 above) is greater for the
        auto-regularization method than the standard pseudo-inverse method. The response 
        prediction accuracy is evaluated with the "average_asd_error" method for the linear SPR.
    """
    system_a_ise_frfs, system_b_ise_frfs = beam_ise_frfs

    system_a_manual_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    system_b_manual_spr = ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    system_a_auto_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_truncation_by_l_curve()

    system_b_auto_spr = ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)

    assert np.abs(system_a_manual_spr.force.ordinate).sum() > np.abs(system_a_auto_spr.force.ordinate).sum()
    assert np.real(system_b_manual_spr.average_asd_error().ordinate).sum() > np.real(system_b_auto_spr.average_asd_error().ordinate).sum()

def test_auto_tikhonov_by_cv_rse_loocv(beam_ise_frfs, system_a_noised_response, system_b_truth_response):
    """
    This test verifies that the "auto_tikhonov_by_cv_rse" inverse method, with the loocv 
    cross validation option selected, performs as expected. It is a seven step test:

        1. truth FRFs and truth excitation are created for two test beam systems, referred
        to as system A and B. The truth excitation is the same for both beam systems and 
        is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        responses of systems A and B.
        3. Random error is added to the system A truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam for both system A and B.
        5. The ISE FRFs (from step three) and noised responses (from step three) system A are 
        used to estimate pseudo-forces using the standard pseudo-inverse method.
        6. The ISE FRFs (from step three) and noised responses (from step three) system A are 
        used to estimate pseudo-forces using the prescribed auto-regularization method.
        7. The pseudo-forces from system A (from steps five and 6) and ISE FRFs from system B 
        (from step 3) will be used to reconstruct the responses on the receiver beam of system B.  

    The test passes if two conditions are met:
        1. The summed RMS level of the estimated forces is lower for the auto-regularization 
        method than the standard pseudo-inverse method.
        2. The response prediction accuracy on system B (from step 7 above) is greater for the
        auto-regularization method than the standard pseudo-inverse method. The response 
        prediction accuracy is evaluated with the "average_asd_error" method for the linear SPR.
    """
    system_a_ise_frfs, system_b_ise_frfs = beam_ise_frfs

    system_a_manual_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    system_b_manual_spr = ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    system_a_auto_spr = ff.LinearSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_cv_rse(cross_validation_type='loocv')

    system_b_auto_spr = ff.LinearSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)

    assert np.abs(system_a_manual_spr.force.ordinate).sum() > np.abs(system_a_auto_spr.force.ordinate).sum()
    assert np.real(system_b_manual_spr.average_asd_error().ordinate).sum() > np.real(system_b_auto_spr.average_asd_error().ordinate).sum()

#%% Set-up for the test on the rigidized system

@pytest.fixture(scope='module')
def rigid_system():
    _, system, _ = create_rigid_system()
    return system

@pytest.fixture(scope='module')
def rigid_transformation():
    _, _, transformation = create_rigid_system()
    return transformation

@pytest.fixture(scope='module')
def rigid_truth_frfs(rigid_system):
    frequency = np.arange(501)
    point_coordinate = sdpy.coordinate_array(node=101, direction = [1,3,5])
    beam_translation_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()
    return rigid_system.frequency_response(frequencies=frequency, responses=beam_translation_coordinate, references=point_coordinate)

@pytest.fixture(scope='module')
def rigid_ise_frfs(rigid_system):
    frequency = np.arange(501)
    beam_translation_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()
    return rigid_system.frequency_response(frequencies=frequency, responses=beam_translation_coordinate, 
                                           references=beam_translation_coordinate)
@pytest.fixture(scope='module')
def rigid_excitation(rigid_truth_frfs):
    force_spectrum, _ = create_broadband_random_spectral_excitation(rigid_truth_frfs[0,:].reference_coordinate, rigid_truth_frfs.ravel().abscissa[0])
    return force_spectrum

@pytest.fixture(scope='module')
def rigid_truth_response(rigid_truth_frfs, rigid_excitation):
    truth_spectra_ord = np.moveaxis(rigid_truth_frfs.ordinate,-1,0)@np.moveaxis(rigid_excitation.ordinate,0,-1)[...,np.newaxis]
    return sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, rigid_excitation.ravel().abscissa[0], 
                           np.moveaxis(truth_spectra_ord[...,0],0,-1), rigid_truth_frfs[:,0].response_coordinate[...,np.newaxis])
#%% Test on the rigidized system

def test_transformed_spr_inverse(rigid_ise_frfs, rigid_truth_response, 
                                 rigid_excitation, rigid_transformation):
    """
    This test verifies that the transformations work in the inverse problem. It 
    is a six step test:

        1. It generates a system with a rigid beam with an external node that is 
        rigidly connected to all the nodes of the beam.
        2. Truth FRFs and truth excitation are made for the system, where the 
        truth excitation is applied to the external node and the truth responses
        are computed on the beam. 
        3. Rigid coordinate transformations are created to transform responses
        and forces on the beam to responses and forces on the external node. 
        4. ISE FRFs are created for reference and response DOFs on the beam. 
        5. The ISE FRFs (from step four), truth responses (from step two), and
        transformations (from step three) are combined to estimate forces acting
        on the beam. 
        6. The transformations (from step three) are combined with the estimated
        forces (from step five) to reconstruct the forces on the external node.

    The test passes if the reconstructed forces match the truth forces are the 
    same, based on a NumPy allclose comparison.
    """
    point_coordinate = sdpy.coordinate_array(node=101, direction = [1,3,5])
    beam_translation_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()

    force_transformation = sdpy.matrix(rigid_transformation[point_coordinate, beam_translation_coordinate],
                                       point_coordinate, beam_translation_coordinate)

    response_transformation_array = np.linalg.pinv(rigid_transformation[point_coordinate, beam_translation_coordinate].T)
    response_transformation = sdpy.matrix(response_transformation_array,
                                          point_coordinate, beam_translation_coordinate)
    transform_spr = ff.LinearSourcePathReceiver(rigid_ise_frfs, rigid_truth_response, 
                                                response_transformation=response_transformation,
                                                reference_transformation=force_transformation)
    transform_spr.manual_inverse(use_transformation=True)
    
    assert np.allclose(rigid_excitation.ordinate, transform_spr.transformed_force.ordinate)

#%% Set-up for the simple unit tests

@pytest.fixture(scope='module')
def unit_test_frf():
    abscissa = np.array([0,1,2])
    dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    frf_dof = sdpy.coordinate.outer_product(dof, dof)
    return sdpy.transfer_function_array(abscissa, np.moveaxis(np.array([np.eye(4)*[1,2,3,4]]*3),0,-1), frf_dof)

@pytest.fixture(scope='module')
def unit_test_force():
    abscissa = np.array([0,1,2])
    dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    return sdpy.spectrum_array(abscissa, np.moveaxis(np.array([[1,2,3,4], [2,4,6,8], [4,8,12,16]]),0,-1), 
                               dof[...,np.newaxis])

@pytest.fixture(scope='module')
def unit_test_response():
    abscissa = np.array([0,1,2])
    dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    return sdpy.spectrum_array(abscissa, np.moveaxis(np.array([[1,2,3,4], [2,4,6,8], [4,8,12,16]])*[1,2,3,4],0,-1), 
                               dof[...,np.newaxis])

@pytest.fixture()
def unit_test_spr(unit_test_frf, unit_test_force, unit_test_response):
    return ff.LinearSourcePathReceiver(unit_test_frf, unit_test_response, unit_test_force)

#%% Unit tests on the simple system

def test_predicted_response_specific_dofs(unit_test_spr, unit_test_response):
    """
    This makes sure that the `predicted_response_specific_dofs` works as expected. It
    checks that the predicted response computes the response for the correct DOFs and 
    makes sure that the correct errors are raised, as necessary.
    """
    predicted_response_one_dof = unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+'])).ordinate 
    assert np.all(predicted_response_one_dof == unit_test_response.ordinate[0,:])

    predicted_response_skipping_dofs = unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+', '3X+'])).ordinate 
    assert np.all(predicted_response_skipping_dofs == unit_test_response.ordinate[[0,2],:])

    predicted_response_flipped_dofs = unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['2X+', '1X+', '3X+'])).ordinate 
    assert np.all(predicted_response_flipped_dofs == unit_test_response.ordinate[[1,0,2],:])

    with pytest.raises(ValueError, match='The supplied response DOFs must be a 1D array'):
        unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+'])[...,np.newaxis])

    with pytest.raises(ValueError, match='All the supplied response DOFs are not included in the SPR object'):
        unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['10X+']))
    
    test_spr = unit_test_spr.copy()
    test_spr._force_array_ = None
    with pytest.raises(AttributeError, match='There is no force array in this object so predicted responses cannot be computed'):
        test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+']))

def test_reconstructed_response(unit_test_frf, unit_test_force, unit_test_response):
    """
    This makes sure that the reconstructed response properties work as expected. It
    checks that the properties computes the response for the correct DOFs and 
    makes sure that the correct errors are raised, as necessary.
    """
    target_dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    training_dof = sdpy.coordinate_array(node=[1,2], direction=1)
    validation_dof = sdpy.coordinate_array(node=[3,4], direction=1)

    unit_test_spr = ff.LinearSourcePathReceiver(unit_test_frf, 
                                                unit_test_response, 
                                                unit_test_force, 
                                                training_response_coordinate=training_dof)

    # Validating that the sample splitting worked as expected
    assert np.all(unit_test_spr.training_response_coordinate == training_dof)
    assert np.all(unit_test_spr.validation_response_coordinate == validation_dof)
    assert np.all(unit_test_spr.target_response_coordinate == target_dof)

    reconstructed_target_response = unit_test_spr.reconstructed_target_response
    assert np.all(reconstructed_target_response.ordinate == unit_test_response.ordinate)

    reconstructed_training_response = unit_test_spr.reconstructed_training_response
    assert np.all(reconstructed_training_response.ordinate == unit_test_response.ordinate[:2,:])

    reconstructed_validation_response = unit_test_spr.reconstructed_validation_response
    assert np.all(reconstructed_validation_response.ordinate == unit_test_response.ordinate[2:,:])

    no_force_spr = ff.LinearSourcePathReceiver(unit_test_frf, 
                                               unit_test_response,  
                                               training_response_coordinate=training_dof)

    with pytest.raises(AttributeError, match='There is no force array in this object so target responses cannot be reconstructed'):
        no_force_spr.reconstructed_target_response

    with pytest.raises(AttributeError, match='There is no force array in this object so training responses cannot be reconstructed'):
        no_force_spr.reconstructed_training_response
    
    with pytest.raises(AttributeError, match='There is no force array in this object so validation responses cannot be reconstructed'):
        no_force_spr.reconstructed_validation_response

def test_sample_splitting(unit_test_frf, unit_test_response):
    """
    This makes sure that the sample splitting works as expected. It checks that 
    the training, validation, and target responses/FRFs are split up as necessary.
    It also makes sure that the correct errors are raised, as necessary.
    """
    target_dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    training_dof = sdpy.coordinate_array(node=[1,2], direction=1)
    validation_dof = sdpy.coordinate_array(node=[3,4], direction=1)

    unit_test_spr = ff.LinearSourcePathReceiver(unit_test_frf, 
                                                unit_test_response,
                                                training_response_coordinate=training_dof)
    
    # Validating that the sample splitting worked as expected
    assert np.all(unit_test_spr.training_response_coordinate == training_dof)
    assert np.all(unit_test_spr.validation_response_coordinate == validation_dof)
    assert np.all(unit_test_spr.target_response_coordinate == target_dof)

    # Checking the responses
    assert np.all(unit_test_spr.target_response.ordinate == unit_test_response.ordinate)
    assert np.all(unit_test_spr.training_response.ordinate == unit_test_response.ordinate[:2,:])
    assert np.all(unit_test_spr.validation_response.ordinate == unit_test_response.ordinate[2:,:])

    # Checking the FRFs
    assert np.all(unit_test_spr.target_frfs.ordinate == unit_test_frf.ordinate)
    assert np.all(unit_test_spr.training_frfs.ordinate == unit_test_frf.ordinate[:2,:,:])
    assert np.all(unit_test_spr.validation_frfs.ordinate == unit_test_frf.ordinate[2:,:,:])

    with pytest.raises(AttributeError, match='The training response coordinate for a SourcePathReceiver object cannot be reset after it is initialized'):
        unit_test_spr.training_response_coordinate = training_dof

    bad_training_dof = sdpy.coordinate_array(node=[1,5], direction=1)
    with pytest.raises(ValueError):
        unit_test_spr = ff.LinearSourcePathReceiver(unit_test_frf, 
                                                unit_test_response,
                                                training_response_coordinate=bad_training_dof)
    
    misordered_training_dof = sdpy.coordinate_array(node=[4,1], direction=1)
    misordered_spr = ff.LinearSourcePathReceiver(unit_test_frf, 
                                                unit_test_response,
                                                training_response_coordinate=misordered_training_dof)
    
    # Validating that the sample splitting worked as expected
    assert np.all(misordered_spr.training_response_coordinate == sdpy.coordinate_array(node=[1,4], direction=1))
    assert np.all(misordered_spr.validation_response_coordinate == sdpy.coordinate_array(node=[2,3], direction=1))
    assert np.all(misordered_spr.target_response_coordinate == target_dof)

    # Checking the responses
    assert np.all(misordered_spr.target_response.ordinate == unit_test_response.ordinate)
    assert np.all(misordered_spr.training_response.ordinate == unit_test_response.ordinate[[0,3],:])
    assert np.all(misordered_spr.validation_response.ordinate == unit_test_response.ordinate[[1,2],:])

    # Checking the FRFs
    assert np.all(misordered_spr.target_frfs.ordinate == unit_test_frf.ordinate)
    assert np.all(misordered_spr.training_frfs.ordinate == unit_test_frf.ordinate[[0,3],:,:])
    assert np.all(misordered_spr.validation_frfs.ordinate == unit_test_frf.ordinate[[1,2],:,:])