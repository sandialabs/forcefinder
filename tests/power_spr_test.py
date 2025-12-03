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
import warnings

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
    from step eight. This test is repeated in this function to verify that the in_place kwarg
    works as expected (i.e., if the forces in the original SPR object are changed by the update).  
    """
    system_a_ise_frfs, _ = beam_ise_frfs

    system_a_auto_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_auto_spr.auto_tikhonov_by_l_curve()
    original_force = system_a_auto_spr.force.copy()

    def trace_error_sum(spr_object):
        """
        Computes the trace error for the SPR object against the training response. This error
        is summed over all frequency lines for a single number metric. The training response
        is deliberately being pulled from the `system_a_auto_spr` for all cases.
        """
        training_response_sum = np.real(system_a_auto_spr.transformed_training_response.ordinate.diagonal(axis1=0, axis2=1).sum(axis=1))
        reconstructed_response_sum = np.real(spr_object.transformed_reconstructed_response.ordinate.diagonal(axis1=0, axis2=1).sum(axis=1))
        return np.sum(np.abs(reconstructed_response_sum-training_response_sum))

    auto_trace_error_sum = trace_error_sum(system_a_auto_spr)

    system_a_match_trace_spr = system_a_auto_spr.match_trace_update(in_place=False)
    new_instance_match_trace_error_sum = trace_error_sum(system_a_match_trace_spr)
    
    assert  auto_trace_error_sum > new_instance_match_trace_error_sum
    assert  np.all(system_a_auto_spr.force.ordinate == original_force.ordinate)

    system_a_auto_spr.match_trace_update(in_place=True)
    in_place_match_trace_error_sum = trace_error_sum(system_a_auto_spr)
    
    assert  auto_trace_error_sum > in_place_match_trace_error_sum
    assert  np.any(system_a_auto_spr.force.ordinate != original_force.ordinate)

def test_reduce_drives_update(beam_ise_frfs, system_a_noised_response):
    """
    This test verifies that the reduce drives update in the power SPR object works as
    expected. It is a seven step test:

        1. truth FRFs and truth excitation are created for the test beam system, where
        the truth excitation is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        response of the beam system.
        3. Random error is added to the truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam.
        5. The ISE FRFs (from step four) and noised responses (from step three) are used to 
        estimate pseudo-forces with the "manual_inverse" standard method.
        6. Steps 4-6 will be repeated and a reduce drives update will be applied to the 
        pseudo-forces. 
        7. This check is repeated several times to ensure that exceptions aren't thrown 
        for different combinations of kwargs. 

    The test passes if the forces from the reduce drives update are smaller than the forces
    from the manual inverse. 
    """
    system_a_ise_frfs, _ = beam_ise_frfs

    system_a_standard_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_standard_spr.manual_inverse()
    original_force = system_a_standard_spr.force.copy()
    with pytest.warns(UserWarning) as record:
        new_instance_system_a_reduce_drives_spr = system_a_standard_spr.reduce_drives_update(in_place=False)

    assert  np.all(system_a_standard_spr.force.ordinate == original_force.ordinate)
    assert  system_a_standard_spr.force.rms().sum() > new_instance_system_a_reduce_drives_spr.force.rms().sum()

    reduce_force = new_instance_system_a_reduce_drives_spr.force.copy()
    with pytest.warns(UserWarning) as record:
        new_instance_system_a_reduce_drives_spr = system_a_standard_spr.reduce_drives_update(db_error_ratio=10, 
                                                                                             in_place=False)
    assert  np.any(new_instance_system_a_reduce_drives_spr.force.ordinate != reduce_force.ordinate)
    assert  new_instance_system_a_reduce_drives_spr.force.rms().sum() > reduce_force.rms().sum()

    with pytest.warns(UserWarning) as record:
        new_instance_system_a_reduce_drives_spr = system_a_standard_spr.reduce_drives_update(reduce_max_drive=True, 
                                                                                             in_place=False)
    assert  np.any(new_instance_system_a_reduce_drives_spr.force.ordinate != reduce_force.ordinate)
    assert  system_a_standard_spr.force.rms().sum() > new_instance_system_a_reduce_drives_spr.force.rms().sum()
        
    with pytest.warns(UserWarning) as record:
        new_instance_system_a_reduce_drives_spr = system_a_standard_spr.reduce_drives_update(use_warm_start=False, 
                                                                                             in_place=False)
    assert  system_a_standard_spr.force.rms().sum() > new_instance_system_a_reduce_drives_spr.force.rms().sum()

    in_place_system_a_reduce_drives_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    in_place_system_a_reduce_drives_spr.manual_inverse()
    with pytest.warns(UserWarning) as record:
        in_place_system_a_reduce_drives_spr.reduce_drives_update(in_place=True)
    assert  np.all(new_instance_system_a_reduce_drives_spr.force.ordinate == reduce_force.ordinate)
    assert  system_a_standard_spr.force.rms().sum() > in_place_system_a_reduce_drives_spr.force.rms().sum()

def test_reduce_drives_condition_checker():
    """
    This test checks that the reduced drives condition checker behaves as expected. 
    It goes through the following process:

        1. A generic "truth" PSD array is made (where the PSD is all ones).
        2. Random error is added to the PSD array from step one, to represent the
        reconstructed PSD that the upper and lower bound for the optimization is 
        based off of. 
        3. Various values are added to the spoofed reconstructed PSD so the different
        states can be checked. 

    This test passes if the return values from the checker function are expected.
    """
    truth = np.ones((200, 10), dtype=float)
    noise = np.random.randn(200,10)*0.05
    reconstructed = np.ones((200, 10), dtype=float) + noise

    # The reduced_reconstructed is within the error bounds
    reduced_reconstructed = reconstructed.copy()
    in_bounds_value = ff.core.utilities.reduce_drives_condition_not_met(truth, 
                                                                        reconstructed, 
                                                                        reduced_reconstructed, 
                                                                        1)
    assert in_bounds_value == False

    # The reduced_reconstructed is within the error bounds by numerical precision
    reduced_reconstructed = reconstructed.copy() + 1e-15
    in_bounds_by_precision_value = ff.core.utilities.reduce_drives_condition_not_met(truth, 
                                                                        reconstructed, 
                                                                        reduced_reconstructed, 
                                                                        1)
    assert in_bounds_by_precision_value == False

    # The reduced_reconstructed is within the error bounds with the db_error_ratio
    db_error_ratio = 2
    reduced_reconstructed = truth + noise*(10**(db_error_ratio/10))
    in_bounds_by_precision_value = ff.core.utilities.reduce_drives_condition_not_met(truth, 
                                                                        reconstructed, 
                                                                        reduced_reconstructed, 
                                                                        db_error_ratio)
    assert in_bounds_by_precision_value == False

    # The reduced_reconstructed is above and outside the error bounds
    reduced_reconstructed = reconstructed.copy() + 1e-5
    above_bounds_value = ff.core.utilities.reduce_drives_condition_not_met(truth, 
                                                                           reconstructed, 
                                                                           reduced_reconstructed, 
                                                                           1)
    assert above_bounds_value == True

    # The reduced_reconstructed is below and outside the error bounds
    reduced_reconstructed = reconstructed.copy() - 1e-5
    below_bounds_value = ff.core.utilities.reduce_drives_condition_not_met(truth, 
                                                                           reconstructed, 
                                                                           reduced_reconstructed, 
                                                                           1)
    assert below_bounds_value == True

def test_reduce_drives_transformation_exception(beam_ise_frfs, system_a_noised_response):
    """
    This test verifies that the reduce drives update in the power SPR object returns
    a NotImplementedException when a reference transformation is in the SPR object. 
    It is a eight step test:

        1. truth FRFs and truth excitation are created for the test beam system, where
        the truth excitation is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        response of the beam system.
        3. Random error is added to the truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam.
        5. A sample reference transformation is made.
        6. The ISE FRFs (from step four), noised responses (from step three), and sample 
        transformation are used to estimate pseudo-forces with the "manual_inverse" 
        standard method.
        7. A reduce drives update is applied to the SPR object.
        8. Steps 5-7 are repeated with different kwargs and transformations.

    This test passes if exceptions are raised in the expected conditions. 
    """
    system_a_ise_frfs, _ = beam_ise_frfs
    system_a_standard_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_standard_spr.manual_inverse()
    original_force = system_a_standard_spr.force.copy()

    # Testing with a shape changing transformation
    system_a_ref_transformed_spr = system_a_standard_spr.copy()
    
    sample_transformation_array = np.array([[1,0,0,0],
                                            [0,1,0,0],
                                            [0,0,1,0]])
    sample_transformation = sdpy.matrix(sample_transformation_array, 
                                        sdpy.coordinate_array(string_array=['1001X+', '1001X+', '1002X+']),
                                        system_a_ref_transformed_spr.reference_coordinate)
    
    system_a_ref_transformed_spr.reference_transformation = sample_transformation
    
    # should raise an exception
    with pytest.raises(NotImplementedError, match='The reduce drives update does not currently work with SPR objects that have non-identity reference transformations'):
        system_a_ref_transformed_spr.reduce_drives_update(use_transformation=True)
    
    # should not raise an exception
    with pytest.warns(UserWarning) as record: #used to catch warnings that might get passed in the test
        system_a_ref_transformed_spr.reduce_drives_update(use_transformation=False)
    assert  original_force.rms().sum() > system_a_ref_transformed_spr.force.rms().sum()

    # Testing with a transformation that applies DOF weighting
    system_a_weighted_spr = system_a_standard_spr.copy()
    system_a_weighted_spr.apply_reference_weighting(np.array([1,2,3,4]))

    # should raise an exception
    with pytest.raises(NotImplementedError, match='The reduce drives update does not currently work with SPR objects that have non-identity reference transformations'):
        system_a_weighted_spr.reduce_drives_update(use_transformation=True)

def test_reduce_drives_update_warnings(beam_ise_frfs, system_a_noised_response):
    """
    This test verifies that the reduce drives update in the power SPR object returns
    warnings if the upper or lower bound in the constraint isn't met. For this test, 
    we expect both the upper and lower bound in the constrain to be violated. However, 
    there is logic in the test to only apply an assertion if this is the case. It is a 
    eight step test:

        1. truth FRFs and truth excitation are created for the test beam system, where
        the truth excitation is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        response of the beam system.
        3. Random error is added to the truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam.
        5. The ISE FRFs (from step four) and noised responses (from step three) are used to 
        estimate pseudo-forces with the "manual_inverse" standard method.
        6. Steps 4-6 will be repeated and a reduce drives update will be applied to the 
        pseudo-forces.
        7. The upper and lower bound for the reduce drives constrained optimization are 
        computed from the results in step five. 
        8. A logical check is used to see if the constraints are violated.

    The test passes if the constraints are violated and an accompanying warning is 
    returned.
    """
    system_a_ise_frfs, _ = beam_ise_frfs

    system_a_standard_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_standard_spr.manual_inverse()

    truth = np.abs(system_a_standard_spr.transformed_training_response.ordinate.diagonal(axis1=0,axis2=1))
    reconstructed = np.abs(system_a_standard_spr.transformed_reconstructed_response.ordinate.diagonal(axis1=0,axis2=1))

    db_diffs = 10*np.log10(reconstructed/truth)
    y_lb = truth*10**(-np.abs(db_diffs)/10) 
    y_ub = truth*10**(np.abs(db_diffs)/10) 

    with pytest.warns(UserWarning) as record:
        system_a_standard_spr.reduce_drives_update(in_place=True)
    
    reduced_reconstructed = np.abs(system_a_standard_spr.transformed_reconstructed_response.ordinate.diagonal(axis1=0,axis2=1))
    warning_list = [str(record[ii].message) for ii in range(len(record))]

    if np.any(reduced_reconstructed < y_lb) or np.any(reduced_reconstructed > y_ub):
        expected_warning = 'The reduce drives update failed and resulted in an under or over predicted response at some frequencies'
        assert np.any(np.isin(warning_list, expected_warning))
    else: 
        warnings.warn('The reduce drives warnings test did not fail the optimization and did not test the desired behavior')

def test_reduce_drives_update_indexing(beam_ise_frfs, system_a_noised_response):
    """
    This test attempts to verify that the indexing in the reduce_drives_update
    is working as expected. It is a seven step test:

        1. truth FRFs and truth excitation are created for the test beam system, where
        the truth excitation is on the source beam. 
        2. The truth FRFs and truth excitation are combined to compute the truth 
        response of the beam system.
        3. Random error is added to the truth response.
        4. A reduced set of "ISE" FRFs are computed with responses DOF on the "receiver" 
        beam and reference DOFs on the "source" beam at the interface DOFs between the 
        source and receiver beam.
        5. The ISE FRFs (from step four) and noised responses (from step three) are 
        used to estimate pseudo-forces with the "manual_inverse" standard method.
        6. Select frequency indices of the force and response arrays are set to 1e-18.
        7. A reduce drives update is applied to the SPR object.

    This test passes if the forces at the select frequency indices remained the same 
    value after the reduce drives update. While not a complete test, this provides a
    rough check that the indexing in the reduce drives update is working as expected. 
    """
    system_a_ise_frfs, _ = beam_ise_frfs

    system_a_spr = ff.PowerSourcePathReceiver(system_a_ise_frfs, system_a_noised_response)
    system_a_spr.manual_inverse()

    zero_inds = [10, 100, 200, 300, 400]
    small_value = 1e-18
    system_a_spr._training_response_array_[zero_inds,...] = small_value
    system_a_spr._force_array_[zero_inds,...] = small_value
    original_force = system_a_spr.force.copy()

    with pytest.warns(UserWarning) as record:
        system_a_spr.reduce_drives_update()
    assert original_force.rms().sum() > system_a_spr.force.rms().sum()
    assert np.all(system_a_spr._force_array_[zero_inds,...]==small_value)

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

#%% Set-up for the simple unit tests

@pytest.fixture(scope='module')
def unit_test_frf():
    abscissa = np.array([1,2,3,4,5,6])
    dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    frf_dof = sdpy.coordinate.outer_product(dof, dof)
    return sdpy.transfer_function_array(abscissa, np.moveaxis(np.array([np.eye(4)*[1,2,3,4]]*6),0,-1), frf_dof)

@pytest.fixture(scope='module')
def unit_test_force():
    abscissa = np.array([1,2,3,4,5,6])
    dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    ord = np.array([np.diag([1,2,3,4]), 
                    np.diag([2,4,6,8]), 
                    np.diag([4,8,12,16]),
                    np.diag([8,16,24,32]),
                    np.diag([16,32,48,64]),
                    np.diag([32,64,96,128])])
    return sdpy.power_spectral_density_array(abscissa, np.moveaxis(ord,0,-1), 
                                             sdpy.coordinate.outer_product(dof, dof))

@pytest.fixture(scope='module')
def unit_test_response():
    abscissa = np.array([1,2,3,4,5,6])
    dof = sdpy.coordinate_array(node=[1,2,3,4], direction=1)
    ord = np.array([np.diag([1,2,3,4]), 
                    np.diag([2,4,6,8]), 
                    np.diag([4,8,12,16]),
                    np.diag([8,16,24,32]),
                    np.diag([16,32,48,64]),
                    np.diag([32,64,96,128])])
    return sdpy.power_spectral_density_array(abscissa, np.moveaxis((ord*[1,2,3,4])*[1,2,3,4],0,-1), 
                                             sdpy.coordinate.outer_product(dof, dof))

@pytest.fixture()
def unit_test_spr(unit_test_frf, unit_test_force, unit_test_response):
    return ff.PowerSourcePathReceiver(unit_test_frf, unit_test_response, unit_test_force)

#%% Unit tests on the simple system

def test_predicted_response_specific_dofs(unit_test_spr, unit_test_response):
    """
    This makes sure that the `predicted_response_specific_dofs` works as expected. It
    checks that the predicted response computes the response for the correct DOFs and 
    makes sure that the correct errors are raised, as necessary.
    """
    predicted_response_one_dof = unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+'])).ordinate 
    assert np.all(predicted_response_one_dof == unit_test_response.ordinate[0,0,:])

    predicted_response_skipping_dofs = unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+', '3X+'])).ordinate 
    row_index, column_index = np.ix_([0,2], [0,2])
    assert np.all(predicted_response_skipping_dofs == unit_test_response.ordinate[row_index,column_index,:])

    predicted_response_flipped_dofs = unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['2X+', '1X+', '3X+'])).ordinate 
    row_index, column_index = np.ix_([1,0,2], [1,0,2])
    assert np.all(predicted_response_flipped_dofs == unit_test_response.ordinate[row_index,column_index,:])

    with pytest.raises(ValueError, match='The supplied response DOFs must be a 1D array'):
        unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+'])[...,np.newaxis])

    with pytest.raises(ValueError, match='The supplied response DOFs are not included in the SPR object'):
        unit_test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['10X+']))
    
    test_spr = unit_test_spr.copy()
    test_spr._force_array_ = None
    with pytest.raises(AttributeError, match='There is no force array in this object so predicted responses cannot be computed'):
        test_spr.predicted_response_specific_dofs(sdpy.coordinate_array(string_array=['1X+']))

def test_response_limit_method(unit_test_spr):
    """
    This checks that the basic functionality of `apply_response_limit` works as
    expected. It checks the following things:
        1. That the response limit works if it is applied to a single DOF.
        2. That the response limit works if it is applied to two DOFs.
        3. That the `limit_db_level` makes the limit more or less strict.  
    """
    # Testing the single DOF Limit
    single_dof_limit = ff.ResponseLimit('1X+', [1,2,5,6], [5,5,9.4,9.4])
    
    # Testing when generating a new SPR object
    single_dof_limit_spr = unit_test_spr.copy()
    limited_spr = single_dof_limit_spr.apply_response_limit(single_dof_limit, in_place=False)
    assert np.all(limited_spr.reconstructed_target_response.ordinate[0,0,-2:] == 9.4)
    assert np.all(np.abs(limited_spr.reconstructed_target_response.ordinate[:,:,:4]) == 
                  np.abs(unit_test_spr.reconstructed_target_response.ordinate[:,:,:4]))
    assert np.all(single_dof_limit_spr._force_array_ == unit_test_spr._force_array_)

    # Testing with in place
    single_dof_limit_spr = unit_test_spr.copy()
    single_dof_limit_spr.apply_response_limit(single_dof_limit, in_place=True)
    assert np.all(single_dof_limit_spr.reconstructed_target_response.ordinate[0,0,-2:] == 9.4)
    assert np.all(np.abs(limited_spr.reconstructed_target_response.ordinate[:,:,:4]) == 
                  np.abs(unit_test_spr.reconstructed_target_response.ordinate[:,:,:4]))

    # Testing the two DOF limit, its set-up so different channels are impacted by the limit differently
    two_dof_limit = ff.ResponseLimit(['1X+', '2X+'],
                                     [[1,2,5,6],
                                      [1,3,4,5,6]],
                                     [[1.5,1.5,40,40],
                                      [17,36,20,20,20]])
    two_dof_limit_spr = unit_test_spr.copy()
    two_dof_limit_spr.apply_response_limit(two_dof_limit)
    assert np.all(two_dof_limit_spr.reconstructed_target_response.ordinate[0,0,1] == 1.5)
    assert np.all(np.isclose(two_dof_limit_spr.reconstructed_target_response.ordinate[1,1,-3:], 20))
    assert np.all(np.abs(single_dof_limit_spr.reconstructed_target_response.ordinate[:,:,0]) == 
                  np.abs(unit_test_spr.reconstructed_target_response.ordinate[:,:,0]))

    # Testing limit_db_level with an increased limit. This is a fairly predictable change
    single_dof_limit_spr = unit_test_spr.copy()
    single_dof_limit_spr.apply_response_limit(single_dof_limit, limit_db_level=3)
    assert np.all(single_dof_limit_spr.reconstructed_target_response.ordinate[0,0,-1] == 9.4*10**(3/10))
    assert np.all(np.abs(single_dof_limit_spr.reconstructed_target_response.ordinate[:,:,:5]) == 
                  np.abs(unit_test_spr.reconstructed_target_response.ordinate[:,:,:5]))
    
    # Testing limit_db_level with an decreased limit. This is more difficult to predict because of how
    # the reduction influences the ramp from 2-5 Hz.
    single_dof_limit_spr = unit_test_spr.copy()
    single_dof_limit_spr.apply_response_limit(single_dof_limit, limit_db_level=-3)
    assert np.all(single_dof_limit_spr.reconstructed_target_response.ordinate[0,0,4:] == 9.4*10**(-3/10))
    assert np.all(single_dof_limit_spr.reconstructed_target_response.ordinate[0,0,3] < 
                  unit_test_spr.reconstructed_target_response.ordinate[0,0,3])
    assert np.all(np.abs(single_dof_limit_spr.reconstructed_target_response.ordinate[:,:,:2]) == 
                  np.abs(unit_test_spr.reconstructed_target_response.ordinate[:,:,:2]))
    
    # Testing the in band limit
    in_band_limit = ff.ResponseLimit('1X+', [2,5], [1,1])
    in_band_limit_spr = unit_test_spr.copy()
    in_band_limit_spr.apply_response_limit(in_band_limit)
    assert np.all(in_band_limit_spr._force_array_[0,...] == unit_test_spr._force_array_[0,...])
    assert np.all(in_band_limit_spr._force_array_[-1,...] == unit_test_spr._force_array_[-1,...])
    assert np.all(in_band_limit_spr._force_array_[1:-1,...].diagonal(axis1=1,axis2=2) != 
                  unit_test_spr._force_array_[1:-1,...].diagonal(axis1=1,axis2=2))
    assert np.all(in_band_limit_spr.reconstructed_target_response.ordinate[0,0,1:-1] == 1)

    # Testing the out of band limit
    in_band_limit = ff.ResponseLimit('1X+', [0,10], [1,1])
    in_band_limit_spr = unit_test_spr.copy()
    in_band_limit_spr.apply_response_limit(in_band_limit)
    assert np.all(in_band_limit_spr.reconstructed_target_response.ordinate[0,0,:] == 1)