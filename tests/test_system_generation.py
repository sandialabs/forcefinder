"""
Contains some utilities for the integration tests. 

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
from scipy.linalg import eigh
from scipy.fft import rfft
from scipy.signal.windows import hann, tukey
from scipy.signal import butter, chirp, sosfiltfilt
from scipy.interpolate import CubicSpline

def create_beam_systems():
    """
    Creates the beam systems for testing the various inverse 
    methods in the different SourcePathReceiver objects.  

    Returns
    -------
    system_a_fem : sdpy.Geometry
        The geometry object for beam system A.
    system_a : sdpy.System
        The system object for beam system A.
    system_b_fem : sdpy.Geometry
        The geometry object for beam system B.
    system_b : sdpy.System
        The system object for beam system B.

    Notes
    -----
    This function is intended to create a repeatable system for 
    test functions, which is why there aren't any options. 
    """
    # Basic beam properties
    beam_length = 2 # meters
    beam_width = 0.02 # meters
    beam_height = 0.015 # meters
    number_nodes = 5

    youngs_modulus = 69.8e9 # pascals
    density = 2700 # kg/m^3
    poissons_ratio = 0.33

    translating_interface_spring = 1e9 #N/m
    torsion_interface_spring = 0.11e3 #N/m

    # Making receiver beam system matrices
    beam_stiffness, beam_mass = sdpy.beam.beamkm_2d(beam_length, beam_width, beam_height, 
                                                    number_nodes, youngs_modulus, density, 
                                                    poissons_ratio, axial=False)

    eig_vals, eig_vect = eigh(a=beam_stiffness, b=beam_mass)
    eig_vals[eig_vals<0] = 0

    wn = np.sqrt(np.real(eig_vals))

    z = 0.02 
    modal_damping = np.diag(z*(2*wn)) #Using a modal formulation since the stiffnesses for the zero Hz modes are a little goofy
    beam_damping = np.linalg.pinv(eig_vect.T)@modal_damping@np.linalg.pinv(eig_vect)

    # Making source beam system matrices
    source_beam_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+101, direction=[3,5]).flatten()    
    source_beam_stiffness = beam_stiffness.copy()

    source_beam_stiffness[0,0] += 1e9 # N/m, translating spring to ground on left side
    source_beam_stiffness[1,1] += 1e2 # Nm/rad, torsion spring to ground on left side 

    source_beam_stiffness[-2,-2] += 1e9 # N/m, translating spring to ground on right side
    source_beam_stiffness[-1,-1] += 1e2 # Nm/rad, torsion spring to ground on right side 

    source_eig_vals, source_eig_vect = eigh(a=source_beam_stiffness, b=beam_mass)
    source_eig_vals[source_eig_vals<0] = 0

    source_wn = np.sqrt(np.real(source_eig_vals))
    
    source_modal_damping = np.diag(z*(2*source_wn)) #Using a modal formulation since the stiffnesses for the zero Hz modes are a little goofy
    source_beam_damping = np.linalg.pinv(source_eig_vect.T)@source_modal_damping@np.linalg.pinv(source_eig_vect)

    # Making the assembled system mass matrix (which is the same for both systems)
    assembled_system_mass = np.zeros((beam_mass.shape[0]*2, beam_mass.shape[0]*2), dtype=float)
    assembled_system_mass[:beam_mass.shape[0], :beam_mass.shape[0]] = beam_mass
    assembled_system_mass[beam_mass.shape[0]:, beam_mass.shape[0]:] = beam_mass

    # Assembling system A
    system_a_stiffness = np.zeros((beam_stiffness.shape[0]*2, beam_stiffness.shape[0]*2), dtype=float)
    system_a_stiffness[:beam_stiffness.shape[0], :beam_stiffness.shape[0]] = source_beam_stiffness
    system_a_stiffness[beam_stiffness.shape[0]:, beam_stiffness.shape[0]:] = beam_stiffness

    system_a_stiffness[4,4] += translating_interface_spring
    system_a_stiffness[-4,-4] += translating_interface_spring
    system_a_stiffness[4,-4] -= translating_interface_spring
    system_a_stiffness[-4,4] -= translating_interface_spring

    system_a_stiffness[5,5] += torsion_interface_spring
    system_a_stiffness[-3,-3] += torsion_interface_spring
    system_a_stiffness[5,-3] -= torsion_interface_spring
    system_a_stiffness[-3,5] -= torsion_interface_spring

    system_a_stiffness[6,6] += translating_interface_spring
    system_a_stiffness[-2,-2] += translating_interface_spring
    system_a_stiffness[6,-2] -= translating_interface_spring
    system_a_stiffness[-2,6] -= translating_interface_spring

    system_a_stiffness[7,7] += torsion_interface_spring
    system_a_stiffness[-1,-1] += torsion_interface_spring
    system_a_stiffness[7,-1] -= torsion_interface_spring
    system_a_stiffness[-1,7] -= torsion_interface_spring

    system_a_damping = np.zeros((beam_damping.shape[0]*2, beam_damping.shape[0]*2), dtype=float)
    system_a_damping[:beam_damping.shape[0], :beam_damping.shape[0]] = source_beam_damping
    system_a_damping[beam_damping.shape[0]:, beam_damping.shape[0]:] = beam_damping

    receiver_beam_a_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+201, direction=[3,5]).flatten()
    system_a = sdpy.System(np.concatenate((np.unique(source_beam_coordinate), np.unique(receiver_beam_a_coordinate))),
                           assembled_system_mass, system_a_stiffness, system_a_damping)

    # Assembling system B
    system_b_stiffness = np.zeros((beam_stiffness.shape[0]*2, beam_stiffness.shape[0]*2), dtype=float)
    system_b_stiffness[:beam_stiffness.shape[0], :beam_stiffness.shape[0]] = source_beam_stiffness
    system_b_stiffness[beam_stiffness.shape[0]:, beam_stiffness.shape[0]:] = beam_stiffness

    system_b_stiffness[4,4] += translating_interface_spring
    system_b_stiffness[10,10] += translating_interface_spring
    system_b_stiffness[4,10] -= translating_interface_spring
    system_b_stiffness[10,4] -= translating_interface_spring

    system_b_stiffness[5,5] += torsion_interface_spring
    system_b_stiffness[11,11] += torsion_interface_spring
    system_b_stiffness[5,11] -= torsion_interface_spring
    system_b_stiffness[11,5] -= torsion_interface_spring

    system_b_stiffness[6,6] += translating_interface_spring
    system_b_stiffness[12,12] += translating_interface_spring
    system_b_stiffness[6,12] -= translating_interface_spring
    system_b_stiffness[12,6] -= translating_interface_spring

    system_b_stiffness[7,7] += torsion_interface_spring
    system_b_stiffness[13,13] += torsion_interface_spring
    system_b_stiffness[7,13] -= torsion_interface_spring
    system_b_stiffness[13,7] -= torsion_interface_spring

    system_b_damping = np.zeros((beam_damping.shape[0]*2, beam_damping.shape[0]*2), dtype=float)
    system_b_damping[:beam_damping.shape[0], :beam_damping.shape[0]] = source_beam_damping
    system_b_damping[beam_damping.shape[0]:, beam_damping.shape[0]:] = beam_damping

    receiver_beam_b_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+301, direction=[3,5]).flatten()
    system_b = sdpy.System(np.concatenate((np.unique(source_beam_coordinate), np.unique(receiver_beam_b_coordinate))),
                           assembled_system_mass, system_b_stiffness, system_b_damping)

    # Making the system geometries
    source_beam_locations = np.array([[0,   0, 0],
                                      [0.5, 0, 0],
                                      [1,   0, 0],
                                      [1.5, 0, 0],
                                      [2,   0, 0]], dtype=float)
    
    assembled_system_cs = sdpy.geometry.coordinate_system_array()
    
    receiver_beam_a_locations = source_beam_locations.copy() + [-0.5, 0, 0.02]

    receiver_beam_a_node = sdpy.geometry.node_array(id = np.concatenate((np.unique(source_beam_coordinate.node), np.unique(receiver_beam_a_coordinate.node))),
                                                coordinate = np.concatenate((source_beam_locations, receiver_beam_a_locations)))

    system_a_fem = sdpy.geometry.Geometry(node = receiver_beam_a_node,
                                        coordinate_system = assembled_system_cs)

    system_a_fem.add_traceline(np.unique(source_beam_coordinate.node))
    system_a_fem.add_traceline(np.unique(receiver_beam_a_coordinate.node))
    system_a_fem.add_traceline([103,204,0,205,104])

    receiver_beam_b_locations = source_beam_locations.copy() + [1, 0, 0.02]

    receiver_beam_b_node = sdpy.geometry.node_array(id = np.concatenate((np.unique(source_beam_coordinate.node), np.unique(receiver_beam_b_coordinate.node))),
                                                    coordinate = np.concatenate((source_beam_locations, receiver_beam_b_locations)))

    system_b_fem = sdpy.geometry.Geometry(node = receiver_beam_b_node,
                                        coordinate_system = assembled_system_cs)

    system_b_fem.add_traceline(np.unique(source_beam_coordinate.node))
    system_b_fem.add_traceline(np.unique(receiver_beam_b_coordinate.node))
    system_b_fem.add_traceline([103,301,0,302,104])

    return system_a_fem, system_a, system_b_fem, system_b

def create_rigid_system():
    """
    Creates a beam system with an external node that has been rigidized. 
    This system is intended to test the transformations in the 
    SourcePathReceiver objects. 

    Returns
    -------
    fem : sdpy.Geometry
        The geometry for the rigid system.
    transformed_system : sdpy.System
        The rigidized system matrices.
    transformation : sdpy.Matrix
        The transformation that is used to convert beam motion to the 
        external node motion.

    Notes
    -----
    This function is intended to create a repeatable system for 
    test functions, which is why there aren't any options. 
    """
    beam_length = 1
    youngs_modulus = 69.8e9 # pascals
    density = 2700 # kg/m^3
    poissons_ratio = 0.33

    beam_k, beam_m = sdpy.beam.beamkm_2d(beam_length, 0.05, 0.05, 4, youngs_modulus, density, poissons_ratio)

    physical_m = np.zeros((beam_m.shape[0]+3, beam_m.shape[1]+3), dtype=float)
    physical_m[3:, 3:] = beam_m
    physical_m[0,0] = 0.01
    physical_m[1,1] = 0.01
    physical_m[2,2] = 0.01

    physical_k = np.zeros((beam_k.shape[0]+3, beam_k.shape[1]+3), dtype=float)
    physical_k[3:, 3:] = beam_k
    physical_k[0,0] = 10e6
    physical_k[1,1] = 10e6
    physical_k[2,2] = 10e3

    point_coordinate = sdpy.coordinate_array(node=101, direction = [1,3,5])
    beam_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3,5]).flatten()
    system_coordinate = np.concatenate((point_coordinate, beam_coordinate))

    node_locations = np.array([[ 0,     0, 0],
                               [-0.5,   0, 0.1],
                               [-0.165, 0, 0.1],
                               [ 0.165, 0, 0.1],
                               [ 0.5,   0, 0.1]])

    node_array = sdpy.node_array([101,201,202,203,204], node_locations)

    traceline_array = sdpy.traceline_array(id=1, connectivity=[201,202,203,204,0,101,201,0,101,202,0,101,203,0,101,204])

    fem = sdpy.Geometry(node_array, sdpy.coordinate_system_array(id=1), traceline_array)

    transformation_array = np.array([[1, 0, 0, 1,   0,   0, 1,   0,     0, 1,    0,     0, 1,    0,   0],  
                                     [0, 1, 0, 0,   1,   0, 0,   1,     0, 0,    1,     0, 0,    1,   0],  
                                     [0, 0, 1, 0.1, 0.5, 0, 0.1, 0.165, 0, 0.1, -0.165, 0, 0.1, -0.5, 0]])  
    transformation = sdpy.matrix(transformation_array, point_coordinate, system_coordinate)

    transformed_m = transformation_array@physical_m@transformation_array.T
    transformed_k = transformation_array@physical_k@transformation_array.T

    lam, phi = eigh(transformed_k, transformed_m)
    shape_pinv = np.linalg.pinv(phi)
    transformed_c = shape_pinv@(np.eye(3)*2)@shape_pinv.T

    transformed_system = sdpy.System(system_coordinate, transformed_m, transformed_k, transformed_c, transformation=transformation_array.T)

    return fem, transformed_system, transformation

def create_beam_system_truth_frfs(transient=False):
    """
    Creating the truth FRFs from the beam systems (for creating the truth
    responses) to test the various inverse methods in the different 
    SourcePathReceiver objects.   

    Returns
    -------
    system_a_truth_frfs : sdpy.TransferFunctionArray
        The FRFs that will be used to create the truth data for system A.
    system_b_truth_frfs : sdpy.TransferFunctionArray
        The FRFs that will be used to create the truth data for system B.

    Notes
    -----
    This function is intended to create repeatable FRFs for test 
    functions, which is why there aren't any options. 
    """
    system_a_fem, system_a, system_b_fem, system_b = create_beam_systems()

    source_beam_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+101, direction=[3,5]).flatten()
    receiver_beam_a_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+201, direction=[3,5]).flatten()
    receiver_beam_b_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+301, direction=[3,5]).flatten()

    if transient:
        compute_frequency = np.arange(1001)*0.2
        ise_frequency = np.arange(501)*0.2
    else:
        compute_frequency = np.arange(start=5, stop=1001)*0.2
        ise_frequency = np.arange(start=5, stop=501)*0.2

    # Making the FRFs for system A
    system_a_truth_frfs = system_a.frequency_response(frequencies=compute_frequency, responses=receiver_beam_a_coordinate, 
                                                      references=source_beam_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())
    
    # Making the FRFs for system B
    system_b_truth_frfs = system_b.frequency_response(frequencies=compute_frequency, responses=receiver_beam_b_coordinate, 
                                                      references=source_beam_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())

    return system_a_truth_frfs, system_b_truth_frfs

def create_beam_system_ise_frfs(transient=False):
    """
    Creating the FRFs from the beam systems (to perform the ISE) to test 
    the various inverse methods in the different SourcePathReceiver objects.   

    Returns
    -------
    system_a_ise_frfs : sdpy.TransferFunctionArray
        The FRFs for system A that will be used to evaluate the different 
        inverse methods.
    system_b_ise_frfs : sdpy.TransferFunctionArray
        The FRFs for system B that will be used to evaluate the different 
        inverse methods.

    Notes
    -----
    This function is intended to create repeatable FRFs for test 
    functions, which is why there aren't any options. 
    """
    system_a_fem, system_a, system_b_fem, system_b = create_beam_systems()

    ise_reference_coordinate = sdpy.coordinate_array(node=np.array([103,104])[...,np.newaxis], direction=[3,5]).flatten()
    receiver_beam_a_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+201, direction=[3,5]).flatten()
    receiver_beam_b_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+301, direction=[3,5]).flatten()

    if transient:
        compute_frequency = np.arange(1001)*0.2
        ise_frequency = np.arange(501)*0.2
    else:
        compute_frequency = np.arange(start=5, stop=1001)*0.2
        ise_frequency = np.arange(start=5, stop=501)*0.2

    # Making the FRFs for system A
    system_a_truth_frfs = system_a.frequency_response(frequencies=compute_frequency, responses=receiver_beam_a_coordinate, 
                                                      references=ise_reference_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())

    # Making the FRFs for system B
    system_b_truth_frfs = system_b.frequency_response(frequencies=compute_frequency, responses=receiver_beam_b_coordinate, 
                                                      references=ise_reference_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())

    return system_a_truth_frfs, system_b_truth_frfs

def create_broadband_random_spectral_excitation(force_coordinate=sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+101, direction=[3,5]).flatten(),
                                                frequency=np.arange(start=5, stop=501)*0.2):
    """
    Creates broadband random excitation to test the various inverse 
    methods in the different linear and power SourcePathReceiver objects.  

    Parameters
    ----------
    force_coordinate : sdpy.CoordinateArray, optional
        The coordinate array for the force degrees of freedom. The default
        is to match the test beam system
    frequency : ndarray, optional
        A 1d array for the frequencies to compute the force over. The default
        is to match the test beam system.

    Returns
    -------
    force_spectrum : sdpy.SpectrumArray
        The force in linear spectrum format.
    force_power : sdpy.PowerSpectralDensityArray
        The force in power spectrum format as a cross-power spectral
        density matrix. 

    Notes
    -----
    This function is intended to create a repeatable inputs for 
    test functions, which is why there aren't many options. 
    """
    number_frames = 500
    force_fft = np.zeros((force_coordinate.shape[0], frequency.shape[0]), dtype=complex)

    for ii in range(number_frames):
        frame_force = np.random.randn(force_coordinate.shape[0], frequency.shape[0]*2-1)*hann(frequency.shape[0]*2-1)
        force_fft += rfft(frame_force)

    force_fft /= number_frames
    force_spectrum = sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, frequency, force_fft, force_coordinate[...,np.newaxis])

    force_cpsd = np.moveaxis(force_fft,0,1)[...,np.newaxis]@np.moveaxis(force_fft.conj(),0,1)[:,np.newaxis,:]
    force_power = sdpy.data_array(sdpy.data.FunctionTypes.POWER_SPECTRAL_DENSITY, frequency, np.moveaxis(force_cpsd,0,-1), 
                                        sdpy.coordinate.outer_product(force_coordinate,force_coordinate))
    
    return force_spectrum, force_power

def additive_noise_linear_spectra(original_spectra):
    """
    Adds random noise to the supplied linear spectra.

    Parameters
    ----------
    original_spectra : SpectrumArray
        The noise free spectra. 
    
    Returns
    -------
    noised_spectra : SpectrumArray
        The original spectra with random noise added to it

    Notes
    -----
    This is intended to work with the test functions, which 
    is why there aren't many options and significant assumptions 
    about the data format. 
    """
    original_spectra_ord = np.moveaxis(original_spectra.ordinate, -1, 0)
    
    noise = np.moveaxis(rfft(np.random.randn(original_spectra_ord.shape[-1], original_spectra_ord.shape[0]*2-1)),0,-1)*np.abs(original_spectra_ord)/20
    noised_spectra_ord = original_spectra_ord+noise

    return sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, original_spectra.ravel().abscissa[0], 
                           np.moveaxis(noised_spectra_ord,0,-1), original_spectra.response_coordinate[...,np.newaxis])

def additive_noise_power_spectra(original_cpsd):
    """
    Adds random noise to the supplied cross power spectral density (CPSD)
    matrix.

    Parameters
    ----------
    original_cpsd : SpectrumArray
        The noise free CPSD matrix. 
    
    Returns
    -------
    noised_spectra : SpectrumArray
        The original CPSD with random noise added to it

    Notes
    -----
    This is intended to work with the test functions, which 
    is why there aren't many options and significant assumptions 
    about the data format. 
    """
    original_cpsd = original_cpsd.reshape_to_matrix()
    cpsd_coordinate = sdpy.coordinate.outer_product(original_cpsd[:,0].response_coordinate, original_cpsd[0,:].reference_coordinate)

    original_cpsd_ord = np.moveaxis(original_cpsd.ordinate, -1, 0)
    original_psd_ord = np.real(np.diagonal(original_cpsd.ordinate, axis1=0, axis2=1))

    noise = np.moveaxis(rfft(np.random.randn(original_psd_ord.shape[-1], original_psd_ord.shape[0]*2-1)),0,-1)*np.sqrt(original_psd_ord)/20
    noise_cpsd = np.einsum('ij,ik->ijk', noise, noise.conj())
    noised_cpsd_ord = original_cpsd_ord+noise_cpsd
    return sdpy.data_array(sdpy.data.FunctionTypes.POWER_SPECTRAL_DENSITY, original_cpsd.ravel().abscissa[0], 
                           np.moveaxis(noised_cpsd_ord,0,-1), cpsd_coordinate)

def create_transient_excitation(force_coordinate=sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+101, direction=[3,5]),
                                sampling_rate=200,
                                number_samples=1000,
                                zero_pad_samples=4000):
    """
    Creates the transient excitation to test the various inverse 
    methods in the transient SourcePathReceiver objects.  

    Parameters
    ----------
    force_coordinate : sdpy.CoordinateArray, optional
        The coordinate array for the force degrees of freedom. The default
        is to match the test beam system
    sampling_rate : float, optional
        The sampling rate for the force in Hz. The default is 200 Hz, which
        matches the test beam system.
    number_samples : int, optional
        The number of samples to use in the force signal. The default is 
        1000.
    zero_pad_samples : int, optional
        The number of samples to zero pad the force before and after the force
        signal. The default is 4000.

    Returns
    -------
    force : sdpy.TimeHistoryArray
        The transient force in the time domain.

    Notes
    -----
    This function is intended to create a repeatable inputs for 
    test functions, which is why there aren't many options.

    The supplied force has significant zero padding applied to mitigate
    issues related to convolution wraparound errors.  
    """
    force_coordinate = force_coordinate.flatten()

    time = np.arange(number_samples)/sampling_rate

    random_excitation = np.random.randn(force_coordinate.shape[0], number_samples)
    random_excitation_scaler = np.abs(np.random.uniform(low=1, high=3, size=np.unique(force_coordinate).shape[0]))
    random_excitation /= (random_excitation.max()*random_excitation_scaler*0.75)[..., np.newaxis]

    chirp_base = np.zeros((force_coordinate.shape[0], number_samples), dtype=float)
    chirp_start = np.zeros(force_coordinate.shape[0], dtype=int)
    chirp_stop = np.zeros(force_coordinate.shape[0], dtype=int)
    for ii in range(force_coordinate.shape[0]):
        chirp_start[ii] = int(np.random.uniform(low=int(number_samples*0.05), high=int(number_samples*0.15)))
        chirp_stop[ii] = int(np.random.uniform(low=int(number_samples*0.30), high=int(number_samples*0.45)))
        
        chirp_start_frequency = np.random.uniform(low=0.05*sampling_rate/2, high=0.3*sampling_rate/2)
        chirp_stop_frequency = np.random.uniform(low=0.6*sampling_rate/2, high=0.9*sampling_rate/2)

        chirp_window = tukey(chirp_stop[ii]-chirp_start[ii], alpha=0.05)*0.75

        chirp_base[ii, chirp_start[ii]:chirp_stop[ii]] = chirp(time[chirp_start[ii]:chirp_stop[ii]], chirp_start_frequency, 
                                                            time[chirp_stop[ii]], chirp_stop_frequency)
        chirp_base[ii, :] /= chirp_base[ii, :].max()*np.random.uniform(low=1, high=5)
        chirp_base[ii, chirp_start[ii]:chirp_stop[ii]] *= chirp_window

    pulse_base = np.zeros((force_coordinate.shape[0], number_samples), dtype=float)
    pulse_start = np.zeros(force_coordinate.shape[0], dtype=int)
    pulse_stop = np.zeros(force_coordinate.shape[0], dtype=int)
    pulse = (np.sin(np.pi*time[:20]/0.1)**2) #samples
    for ii in range(force_coordinate.shape[0]):
        pulse_start[ii] = int(np.random.uniform(low=int(number_samples*0.05), high=int(number_samples*0.9)))
        pulse_stop[ii] = pulse_start[ii]+pulse.shape[0]

        pulse_base[ii, pulse_start[ii]:pulse_stop[ii]] = pulse
        pulse_base[ii, :] *= np.random.uniform(low=1, high=5)

    shape_number_times = 30
    shape_times = time[0:-1:int(time.shape[0]/shape_number_times)]
    random_shaper = np.abs(np.random.laplace(loc=0, scale=10, size=shape_number_times+1))
    random_shaper[np.intersect1d(np.where(shape_times>=time[chirp_start.min()]), np.where(shape_times<=time[chirp_stop.max()]))] = 0.25
    cs = CubicSpline(shape_times, random_shaper, bc_type='natural')
    smoothed_shape = cs(time)
    smoothed_shape[np.where(smoothed_shape<0)] = 0
    smoothed_shape /= smoothed_shape.max()
    random_excitation *= smoothed_shape[np.newaxis, ...]

    window = tukey(number_samples, alpha=0.75)
    sos_filter = butter(10, [0.05*sampling_rate/2,0.95*sampling_rate/2], btype='bandpass', output='sos', fs=sampling_rate)

    force_ordinate = sosfiltfilt(sos_filter, (random_excitation+chirp_base+pulse_base)*window[np.newaxis, ...], axis=-1)
    force = sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE, time, force_ordinate, force_coordinate[...,np.newaxis])
    force = force.zero_pad(num_samples=zero_pad_samples,left=True,right=True)
    return force