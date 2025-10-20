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
                                    create_transient_excitation, 
                                    create_rigid_system)
import pytest

def time_nrmse(truth, predicted):
    """
    Computes the normalized root mean squared error of the predicted
    time trace.

    Parameters
    ----------
    truth : ndarray
        An ndarray of the truth time traces, organized with the DOFs
        on the first axis and the time on the second axis.
    predicted : ndarray
        An ndarray of the predicted time traces, organized with the 
        DOFs on the first axis and the time on the second axis.
    
    Returns
    -------
    NRMSE : ndarray
        A 1d array with the NRMSE for the different DOFs.

    Notes
    -----
    This code is primarily coded to work with the transient SPR 
    tests, which is why there are no checks for data format and
    organization.
    """
    truth_range = truth.max(axis=1)-truth.min(axis=1)
    rmse = np.sqrt(np.mean((truth-predicted)**2, axis=1))
    return rmse/truth_range

def rms_error(truth, predicted):
    """
    Computes the RMS percent error of the predicted time trace.

    Parameters
    ----------
    truth : NDDataArray
        An NDDataArray of the truth quantity with the rms 
        method (e.g., a TimeHistoryArray).
    predicted : NDDataArray
        An NDDataArray of the predicted quantity with the rms 
        method (e.g., a TimeHistoryArray).

    Returns
    -------
    rms_error : ndarray
        A 1D array with the RMS percent error for the different 
        DOFs. 

    Notes
    -----
    This code is primarily coded to work with the transient SPR 
    tests, which is why there are no checks for data format and
    organization.
    """
    truth_rms = truth.rms()
    predicted_rms = predicted.rms()
    return np.abs(predicted_rms-truth_rms)/truth_rms

@pytest.fixture(scope='module')
def truth_frfs():
    system_a_truth_frfs, system_b_truth_frfs = create_beam_system_truth_frfs(transient=True)
    system_a_truth_frfs = system_a_truth_frfs.enforce_causality()
    system_b_truth_frfs = system_b_truth_frfs.enforce_causality()
    return system_a_truth_frfs, system_b_truth_frfs

@pytest.fixture(scope='module')
def ise_frfs():
    system_a_ise_frfs, system_b_ise_frfs = create_beam_system_ise_frfs(transient=True)
    system_a_ise_frfs = system_a_ise_frfs.enforce_causality()
    system_b_ise_frfs = system_b_ise_frfs.enforce_causality()
    return system_a_ise_frfs, system_b_ise_frfs

@pytest.fixture(scope='module')
def truth_transient_force():
    return create_transient_excitation()

@pytest.fixture(scope='module')
def system_a_truth_response(truth_frfs, truth_transient_force):
    system_a_truth_frfs, _ = truth_frfs
    return truth_transient_force.mimo_forward(system_a_truth_frfs)

@pytest.fixture(scope='module')
def system_a_noised_response(system_a_truth_response):
    """will only add noise to the 0-5s time period, since that's the only part with excitation"""
    system_a_noise_response = system_a_truth_response.copy()
    
    start_index = np.where(system_a_truth_response.ravel().abscissa[0]==0)[0][0]
    end_index = np.where(system_a_truth_response.ravel().abscissa[0]==5)[0][0]

    system_a_noise = np.random.randn(np.unique(system_a_noise_response.response_coordinate).shape[0], (end_index-start_index))*system_a_noise_response.rms()[...,np.newaxis]*0.5
    system_a_noise_response.ordinate[:,start_index:end_index] += system_a_noise
    return system_a_noise_response

@pytest.fixture(scope='module')
def system_b_truth_response(truth_frfs, truth_transient_force):
    _, system_b_truth_frfs = truth_frfs
    return truth_transient_force.mimo_forward(system_b_truth_frfs)

@pytest.fixture(scope='module')
def transient_spr_a_truth(ise_frfs, system_a_truth_response):
    system_a_ise_frfs, _ = ise_frfs
    system_a_spr = ff.TransientSourcePathReceiver(system_a_ise_frfs, system_a_truth_response)
    system_a_spr.manual_inverse()
    return system_a_spr

@pytest.fixture()
def transient_spr_b(ise_frfs, system_b_truth_response):
    _, system_b_ise_frfs = ise_frfs
    return ff.TransientSourcePathReceiver(system_b_ise_frfs, system_b_truth_response)

def test_force_roundtrip(ise_frfs):
    """
    This test verifies that the basic linear algebra for force estimation works in the 
    transient SPR object. It is a three step test:

        1. FRFs and excitation are created for the test beam system. The ISE FRFs are 
        used in this case because they are better conditioned than the truth FRFs. 
        2. The FRFs and excitation are combined to compute the response of the beam 
        system. 
        3. The FRFs (from step one) and computed responses (from step two) are used to 
        reconstruct the forces (from step one). 

    The test passes if the reconstructed forces are the "same" as the truth forces. Note
    that the extra processing in the COLA framework means that the reconstructed forces 
    will not pass a NumPy allclose comparison. As such, the  reconstructed forces are 
    compared to the truth forces via the time response assurance criterion, normalized 
    root mean squared error, and root mean square level error, where a good comparison 
    has been pre-defined. Note that the test only compares the 0-5s time period, since 
    that is the only part with excitation. 
    """
    frfs, _ = ise_frfs
    excitation = create_transient_excitation(force_coordinate=np.unique(frfs.reference_coordinate))
    response = excitation.mimo_forward(frfs)

    transient_spr = ff.TransientSourcePathReceiver(frfs, response)
    transient_spr.manual_inverse(use_transformation=False)
    
    trac_comparison = sdpy.correlation.trac(transient_spr.force.extract_elements_by_abscissa(0,5).ordinate, 
                                            excitation.extract_elements_by_abscissa(0,5).ordinate)
    assert trac_comparison.min() > 0.99

    nrmse_comparison = time_nrmse(transient_spr.force.extract_elements_by_abscissa(0,5).ordinate, 
                                  excitation.extract_elements_by_abscissa(0,5).ordinate)
    assert nrmse_comparison.max() < 0.001

    rms_percent_error = rms_error(transient_spr.force.extract_elements_by_abscissa(0,5),
                                  excitation.extract_elements_by_abscissa(0,5))
    assert rms_percent_error.max() < 0.001

def test_system_a_roundtrip(transient_spr_a_truth):
    """
    This test verifies that the transient SPR object can be used to compute a 
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

    The test passes if the reconstructed responses are the "same" as the truth responses. 
    Note that the extra processing in the COLA framework and causality enforcement means 
    that the reconstructed responses will not pass a NumPy allclose comparison. As such, 
    the reconstructed responses are compared to the truth forces via the time response 
    assurance criterion, normalized root mean squared error, and root mean square level 
    error, where a good comparison has been pre-defined. Note that the test only compares 
    the 0-5s time period, since that is the only part with excitation. 
    """
    trac_comparison = sdpy.correlation.trac(transient_spr_a_truth.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                            transient_spr_a_truth.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate)
    assert trac_comparison.min() > 0.95

    nrmse_comparison = time_nrmse(transient_spr_a_truth.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                  transient_spr_a_truth.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate)
    assert nrmse_comparison.max() < 0.02

    rms_percent_error = rms_error(transient_spr_a_truth.target_response.extract_elements_by_abscissa(0,5),
                                  transient_spr_a_truth.reconstructed_target_response.extract_elements_by_abscissa(0,5))
    assert rms_percent_error.max() < 0.02

def test_truth_a_b_roundtrip(transient_spr_b, transient_spr_a_truth):
    """
    This test verifies that the transient SPR object can be used to compute a 
    predictive set of pseudo-forces in an ideal case. It is a five step test:

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
    responses on system B. Note that the extra processing in the COLA framework and causality 
    enforcement means that the reconstructed responses will not pass a NumPy allclose 
    comparison. As such, the reconstructed responses are compared to the truth forces via the 
    time response assurance criterion, normalized root mean squared error, and root mean square 
    level error, where a good comparison has been pre-defined. Note that the test only compares 
    the 0-5s time period, since that is the only part with excitation. 
    """
    transient_spr_b.force = transient_spr_a_truth.force
    trac_comparison = sdpy.correlation.trac(transient_spr_b.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                            transient_spr_b.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate)
    assert trac_comparison.min() > 0.95

    nrmse_comparison = time_nrmse(transient_spr_b.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                  transient_spr_b.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate)
    assert nrmse_comparison.max() < 0.05

    rms_percent_error = rms_error(transient_spr_b.target_response.extract_elements_by_abscissa(0,5),
                                  transient_spr_b.reconstructed_target_response.extract_elements_by_abscissa(0,5))
    assert rms_percent_error.max() < 0.05
    
def test_auto_tikhonov_by_l_curve(ise_frfs, system_a_noised_response, system_b_truth_response):
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
        prediction accuracy is evaluated with the time response assurance criterion, normalized 
        root mean squared error, and root mean square level error.
    
    Note that the test only evaluates the 0-5s time period, since that is the only part with 
    truth excitation. 
    """
    system_a_ise_frfs, system_b_ise_frfs = ise_frfs

    system_a_manual_spr = ff.TransientSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    manual_force_rms = system_a_manual_spr.force.rms().sum()

    system_b_manual_spr = ff.TransientSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    trac_comparison_manual = sdpy.correlation.trac(system_b_manual_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                                   system_b_manual_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()
    
    nrmse_comparison_manual = time_nrmse(system_b_manual_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                         system_b_manual_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()

    rms_percent_error_manual = rms_error(system_b_manual_spr.target_response.extract_elements_by_abscissa(0,5),
                                         system_b_manual_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5)).sum()

    system_a_auto_spr = ff.TransientSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_l_curve()

    auto_force_rms = system_a_auto_spr.force.rms().sum()

    system_b_auto_spr = ff.TransientSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)

    trac_comparison_auto = sdpy.correlation.trac(system_b_auto_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                                   system_b_auto_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()
    
    nrmse_comparison_auto = time_nrmse(system_b_auto_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                         system_b_auto_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()

    rms_percent_error_auto = rms_error(system_b_auto_spr.target_response.extract_elements_by_abscissa(0,5),
                                         system_b_auto_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5)).sum()

    assert manual_force_rms > auto_force_rms
    assert trac_comparison_manual < trac_comparison_auto
    assert nrmse_comparison_manual > nrmse_comparison_auto
    assert rms_percent_error_manual > rms_percent_error_auto

def test_auto_tikhonov_by_cv_rse_loocv(ise_frfs, system_a_noised_response, system_b_truth_response):
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
        prediction accuracy is evaluated with the time response assurance criterion, normalized 
        root mean squared error, and root mean square level error.
    
    Note that the test only evaluates the 0-5s time period, since that is the only part with 
    truth excitation. 
    """
    system_a_ise_frfs, system_b_ise_frfs = ise_frfs

    system_a_manual_spr = ff.TransientSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_manual_spr.manual_inverse()

    manual_force_rms = system_a_manual_spr.force.rms().sum()

    system_b_manual_spr = ff.TransientSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_manual_spr.force)

    trac_comparison_manual = sdpy.correlation.trac(system_b_manual_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                                   system_b_manual_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()
    
    nrmse_comparison_manual = time_nrmse(system_b_manual_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                         system_b_manual_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()

    rms_percent_error_manual = rms_error(system_b_manual_spr.target_response.extract_elements_by_abscissa(0,5),
                                         system_b_manual_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5)).sum()

    system_a_auto_spr = ff.TransientSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_cv_rse(cross_validation_type='loocv')

    auto_force_rms = system_a_auto_spr.force.rms().sum()

    system_b_auto_spr = ff.TransientSourcePathReceiver(system_b_ise_frfs, system_b_truth_response, system_a_auto_spr.force)

    trac_comparison_auto = sdpy.correlation.trac(system_b_auto_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                                   system_b_auto_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()
    
    nrmse_comparison_auto = time_nrmse(system_b_auto_spr.target_response.extract_elements_by_abscissa(0,5).ordinate, 
                                         system_b_auto_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5).ordinate).sum()

    rms_percent_error_auto = rms_error(system_b_auto_spr.target_response.extract_elements_by_abscissa(0,5),
                                         system_b_auto_spr.reconstructed_target_response.extract_elements_by_abscissa(0,5)).sum()

    assert manual_force_rms > auto_force_rms
    assert trac_comparison_manual < trac_comparison_auto
    assert nrmse_comparison_manual > nrmse_comparison_auto
    assert rms_percent_error_manual > rms_percent_error_auto

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
def rigid_frfs(rigid_system):
    frequency = np.arange(501)
    return rigid_system.frequency_response(frequencies=frequency).enforce_causality()

@pytest.fixture()
def rigid_beam_frfs(rigid_frfs):
    beam_translation_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()
    return rigid_frfs[sdpy.coordinate.outer_product(beam_translation_coordinate, beam_translation_coordinate)]

@pytest.fixture()
def rigid_point_frfs(rigid_frfs):
    point_coordinate = sdpy.coordinate_array(node=101, direction = [1,3,5])
    return rigid_frfs[sdpy.coordinate.outer_product(point_coordinate, point_coordinate)]

@pytest.fixture(scope='module')
def rigid_excitation(rigid_frfs):
    transient_excitation = create_transient_excitation(force_coordinate=np.unique(rigid_frfs.reference_coordinate),
                                                       sampling_rate=rigid_frfs.abscissa.max()*2,
                                                       number_samples=4000,
                                                       zero_pad_samples=8000)
    return transient_excitation

@pytest.fixture(scope='module')
def rigid_truth_response(rigid_frfs, rigid_excitation):
    point_coordinate = sdpy.coordinate_array(node=101, direction = [1,3,5])
    truth_frfs = rigid_frfs[sdpy.coordinate.outer_product(np.unique(rigid_frfs.response_coordinate), point_coordinate)]
    return rigid_excitation.mimo_forward(truth_frfs)

#%% Test on the rigidized system

def test_transformed_spr_inverse(rigid_beam_frfs, rigid_point_frfs, 
                                 rigid_truth_response, rigid_transformation):
    """
    This test verifies that the transformations work in the inverse problem. It 
    is a eight step test:

        1. It generates a system with a rigid beam with an external node that is 
        rigidly connected to all the nodes of the beam.
        2. Truth FRFs and truth excitation are made for the system, where the 
        truth excitation is applied to the external node and the truth responses
        are computed on the beam and external node. 
        3. Rigid coordinate transformations are created to transform responses
        and forces on the beam to responses and forces on the external node. 
        4. ISE FRFs are created for reference and response DOFs on the beam. 
        5. The ISE FRFs (from step four), truth responses (from step two), and
        transformations (from step three) are combined to estimate forces acting
        on the beam. 
        6. The transformations (from step three) are combined with the estimated
        forces (from step five) to reconstruct the forces on the external node.
        7. ISE FRFs are created for reference and response DOFs on the external 
        node.
        8. The ISE FRFs (from step seven) and truth responses (from step two) are
        combined to directly reconstruct the forces acting on the external node.
        
    The test passes if the reconstructed forces from the transformed problem 
    (from step six) match the directly reconstructed forces (from step eight), 
    based on a NumPy allclose comparison.
    """
    point_coordinate = sdpy.coordinate_array(node=101, direction = [1,3,5])
    beam_translation_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()

    force_transformation = sdpy.matrix(rigid_transformation[point_coordinate, beam_translation_coordinate],
                                   point_coordinate, beam_translation_coordinate)

    response_transformation_array = np.linalg.pinv(rigid_transformation[point_coordinate, beam_translation_coordinate].T)
    response_transformation = sdpy.matrix(response_transformation_array,
                                        point_coordinate, beam_translation_coordinate)
    
    transform_beam_spr = ff.TransientSourcePathReceiver(rigid_beam_frfs, 
                                                        rigid_truth_response[beam_translation_coordinate[...,np.newaxis]],
                                                        response_transformation=response_transformation,
                                                        reference_transformation=force_transformation)
    transform_beam_spr.manual_inverse(use_transformation=True)

    point_spr = ff.TransientSourcePathReceiver(rigid_point_frfs, 
                                               rigid_truth_response[point_coordinate[...,np.newaxis]])
    point_spr.manual_inverse(use_transformation=False)

    assert np.allclose(transform_beam_spr.transformed_force.extract_elements_by_abscissa(0,5).ordinate, 
                       point_spr.force.extract_elements_by_abscissa(0,5).ordinate)
