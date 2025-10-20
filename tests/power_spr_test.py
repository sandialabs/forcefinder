"""
Includes the tests for the PowerSourcePathReceiver inverse methods. 

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
                                    additive_noise_power_spectra,
                                    create_rigid_system)
import pytest

#%% Set-up for the tests on the elastic system
@pytest.fixture(scope='module')
def beam_force_cpsd():
    _, force_cpsd = create_broadband_random_spectral_excitation()
    return force_cpsd

@pytest.fixture(scope='module')
def beam_truth_frfs():
    system_a_truth_frfs, system_b_truth_frfs = create_beam_system_truth_frfs()
    return system_a_truth_frfs, system_b_truth_frfs

@pytest.fixture(scope='module')
def beam_ise_frfs():
    system_a_ise_frfs, system_b_ise_frfs = create_beam_system_ise_frfs()
    return system_a_ise_frfs, system_b_ise_frfs

@pytest.fixture(scope='module')
def system_a_truth_response(beam_truth_frfs, beam_force_cpsd):
    system_a_truth_frfs, _ = beam_truth_frfs
    return beam_force_cpsd.mimo_forward(system_a_truth_frfs)

@pytest.fixture(scope='module')
def system_a_noised_response(system_a_truth_response):
    return additive_noise_power_spectra(system_a_truth_response)

@pytest.fixture(scope='module')
def system_b_truth_response(beam_truth_frfs, beam_force_cpsd):
    _, system_b_truth_frfs = beam_truth_frfs
    return beam_force_cpsd.mimo_forward(system_b_truth_frfs)

@pytest.fixture(scope='module')
def power_spr_a_truth(beam_ise_frfs, system_a_truth_response):
    system_a_ise_frfs, _ = beam_ise_frfs
    system_a_power_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_truth_response)
    system_a_power_spr.manual_inverse()
    return system_a_power_spr

@pytest.fixture()
def power_spr_b(beam_ise_frfs, system_b_truth_response):
    _, system_b_ise_frfs = beam_ise_frfs
    return ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response)

#%% Testing the basic inverse functions for the linear SPR
def test_system_a_roundtrip(power_spr_a_truth):
    """
    This test verifies that the power SPR object can be used to compute a 
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
        5. The pseudo-forces (from step four) and ISE FRFs (from step three) will be used to 
        reconstruct the responses on the receiver beam.  

    The test passes if the reconstructed responses are the "same" as the truth responses,
    based on a NumPy allclose comparison.  
    """
    assert np.allclose(power_spr_a_truth.target_response.ordinate, power_spr_a_truth.reconstructed_target_response.ordinate)

def test_truth_a_b_roundtrip(power_spr_b, power_spr_a_truth):
    """
    This test verifies that the power SPR object can be used to compute a predictive set 
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
        step three) will be used to reconstruct the responses on the receiver beam of system B.  

    The test passes if the reconstructed responses on system B are the "same" as the truth 
    responses on system B, based on a NumPy allclose comparison.  
    """
    power_spr_b.force = power_spr_a_truth.force
    assert np.allclose(power_spr_b.target_response.ordinate, power_spr_b.reconstructed_target_response.ordinate)

def test_match_trace_update(beam_ise_frfs, system_a_noised_response):
    """
    This test verifies that the match trace method in the power SPR object works as
    expected. It is a five step test:

        1. truth FRFs and truth excitation are created for the test beam system, where
        the truth excitation is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        response of the beam system.
        3. Random error is added to the truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam.
        5. The ISE FRFs (from step four) and noised responses (from step three) are used to 
        estimate pseudo-forces with the "auto_tikhonov_by_l_curve" method to induce bias
        error in the solution.
        6. The pseudo-forces (from step five) and ISE FRFs (from step four) will be used to 
        reconstruct the responses on the receiver beam.
        7. Steps 4-6 will be repeated and a match trace update will be applied to the 
        pseudo-forces. 
        8. The pseudo-forces with a match trace update (from step seven) and ISE FRFs (from 
        step four) will be used to reconstruct the responses on the receiver beam.

    The test passes if the response trace error from step six is greater than the trace error 
    from step eight  
    """
    system_a_ise_frfs, _ = beam_ise_frfs

    system_a_auto_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_l_curve()

    system_a_match_trace_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_match_trace_spr.auto_tikhonov_by_l_curve()
    system_a_match_trace_spr.match_trace_update()

    training_response_sum = np.real(system_a_auto_spr.transformed_training_response.ordinate.diagonal(axis1=0, axis2=1).sum(axis=1))
    auto_reconstructed_response_sum = np.real(system_a_auto_spr.transformed_reconstructed_response.ordinate.diagonal(axis1=0, axis2=1).sum(axis=1))
    auto_trace_error_sum = np.sum(np.abs(auto_reconstructed_response_sum-training_response_sum))
    
    match_trace_reconstructed_response_sum = np.real(system_a_match_trace_spr.transformed_reconstructed_response.ordinate.diagonal(axis1=0, axis2=1).sum(axis=1))
    match_trace_error_sum = np.sum(np.abs(match_trace_reconstructed_response_sum-training_response_sum))

    assert  auto_trace_error_sum > match_trace_error_sum

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

    system_a_manual_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    system_b_manual_spr = ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    system_a_auto_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_l_curve()

    system_b_auto_spr = ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)
    
    assert system_a_manual_spr.force.rms().sum() > system_a_auto_spr.force.rms().sum()
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

    system_a_manual_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    system_b_manual_spr = ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    system_a_auto_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_truncation_by_l_curve()

    system_b_auto_spr = ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)
    
    assert system_a_manual_spr.force.rms().sum() > system_a_auto_spr.force.rms().sum()
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

    system_a_manual_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    system_b_manual_spr = ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    system_a_auto_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_cv_rse(cross_validation_type='loocv')

    system_b_auto_spr = ff.PowerSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)
    
    assert system_a_manual_spr.force.rms().sum() > system_a_auto_spr.force.rms().sum()
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
    _, force_cpsd = create_broadband_random_spectral_excitation(rigid_truth_frfs[0,:].reference_coordinate, rigid_truth_frfs.ravel().abscissa[0])
    return force_cpsd

@pytest.fixture(scope='module')
def rigid_truth_response(rigid_truth_frfs, rigid_excitation):
    return rigid_excitation.mimo_forward(rigid_truth_frfs)

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
    transform_spr = ff.PowerSourcePathReceiver(rigid_ise_frfs, rigid_truth_response, 
                                                response_transformation=response_transformation,
                                                reference_transformation=force_transformation)
    transform_spr.manual_inverse(use_transformation=True)
    
    assert np.allclose(rigid_excitation.ordinate, transform_spr.transformed_force.ordinate)
