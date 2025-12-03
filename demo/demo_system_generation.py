"""
Contains some utilities for generating the benchmark tests. 

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
from scipy.fft import irfft, rfft
from scipy.signal.windows import hann, tukey
from scipy.signal import butter, chirp, sosfiltfilt
from scipy.interpolate import CubicSpline

def create_beam_systems():
    """
    Creates the beam systems for benchmarking the various inverse 
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
    benchmarking functions, which is why there aren't any options. 
    """
    # Basic beam properties
    beam_length = 4 # meters
    beam_width = 0.02 # meters
    beam_height = 0.015 # meters
    number_nodes = 9

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
    modal_damping = np.diag(2*z*wn) #Using a modal formulation since the stiffnesses for the zero Hz modes are a little goofy
    beam_damping = np.linalg.pinv(eig_vect.T)@modal_damping@np.linalg.pinv(eig_vect)

    # Making source beam system matrices
    source_beam_coordinate = sdpy.coordinate_array(node=np.arange(number_nodes)[...,np.newaxis]+101, direction=[3,5]).flatten()    
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
    system_a_stiffness[24,24] += translating_interface_spring
    system_a_stiffness[4,24] -= translating_interface_spring
    system_a_stiffness[24,4] -= translating_interface_spring

    system_a_stiffness[5,5] += torsion_interface_spring
    system_a_stiffness[25,25] += torsion_interface_spring
    system_a_stiffness[5,25] -= torsion_interface_spring
    system_a_stiffness[25,5] -= torsion_interface_spring

    system_a_stiffness[6,6] += translating_interface_spring
    system_a_stiffness[26,26] += translating_interface_spring
    system_a_stiffness[6,26] -= translating_interface_spring
    system_a_stiffness[26,6] -= translating_interface_spring

    system_a_stiffness[7,7] += torsion_interface_spring
    system_a_stiffness[27,27] += torsion_interface_spring
    system_a_stiffness[7,27] -= torsion_interface_spring
    system_a_stiffness[27,7] -= torsion_interface_spring

    system_a_damping = np.zeros((beam_damping.shape[0]*2, beam_damping.shape[0]*2), dtype=float)
    system_a_damping[:beam_damping.shape[0], :beam_damping.shape[0]] = source_beam_damping
    system_a_damping[beam_damping.shape[0]:, beam_damping.shape[0]:] = beam_damping

    receiver_beam_a_coordinate = sdpy.coordinate_array(node=np.arange(number_nodes)[...,np.newaxis]+201, direction=[3,5]).flatten()
    system_a = sdpy.System(np.concatenate((np.unique(source_beam_coordinate), np.unique(receiver_beam_a_coordinate))),
                            assembled_system_mass, system_a_stiffness, system_a_damping)

    # Assembling system B
    system_b_stiffness = np.zeros((beam_stiffness.shape[0]*2, beam_stiffness.shape[0]*2), dtype=float)
    system_b_stiffness[:beam_stiffness.shape[0], :beam_stiffness.shape[0]] = source_beam_stiffness
    system_b_stiffness[beam_stiffness.shape[0]:, beam_stiffness.shape[0]:] = beam_stiffness

    system_b_stiffness[4,4] += translating_interface_spring
    system_b_stiffness[18,18] += translating_interface_spring
    system_b_stiffness[4,18] -= translating_interface_spring
    system_b_stiffness[18,4] -= translating_interface_spring

    system_b_stiffness[5,5] += torsion_interface_spring
    system_b_stiffness[19,19] += torsion_interface_spring
    system_b_stiffness[5,19] -= torsion_interface_spring
    system_b_stiffness[19,5] -= torsion_interface_spring

    system_b_stiffness[6,6] += translating_interface_spring
    system_b_stiffness[20,20] += translating_interface_spring
    system_b_stiffness[6,20] -= translating_interface_spring
    system_b_stiffness[20,6] -= translating_interface_spring

    system_b_stiffness[7,7] += torsion_interface_spring
    system_b_stiffness[21,21] += torsion_interface_spring
    system_b_stiffness[7,21] -= torsion_interface_spring
    system_b_stiffness[21,7] -= torsion_interface_spring

    system_b_damping = np.zeros((beam_damping.shape[0]*2, beam_damping.shape[0]*2), dtype=float)
    system_b_damping[:beam_damping.shape[0], :beam_damping.shape[0]] = source_beam_damping
    system_b_damping[beam_damping.shape[0]:, beam_damping.shape[0]:] = beam_damping

    receiver_beam_b_coordinate = sdpy.coordinate_array(node=np.arange(number_nodes)[...,np.newaxis]+301, direction=[3,5]).flatten()
    system_b = sdpy.System(np.concatenate((np.unique(source_beam_coordinate), np.unique(receiver_beam_b_coordinate))),
                            assembled_system_mass, system_b_stiffness, system_b_damping)

    # Making the system geometries
    source_beam_locations = np.array([[0,   0, 0], #
                                      [0.5, 0, 0],
                                      [1, 0, 0], #
                                      [1.5, 0, 0],
                                      [2,   0, 0], #
                                      [2.5, 0, 0],
                                      [3, 0, 0], #
                                      [3.5, 0, 0],
                                      [4,   0, 0]], dtype=float) #
    
    assembled_system_cs = sdpy.geometry.coordinate_system_array()
    
    receiver_beam_a_locations = source_beam_locations.copy() + [-0.5, 0, 0.2]

    receiver_beam_a_node = sdpy.geometry.node_array(id = np.concatenate((np.unique(source_beam_coordinate.node), np.unique(receiver_beam_a_coordinate.node))),
                                                coordinate = np.concatenate((source_beam_locations, receiver_beam_a_locations)))

    system_a_fem = sdpy.geometry.Geometry(node = receiver_beam_a_node,
                                        coordinate_system = assembled_system_cs)

    system_a_fem.add_traceline(np.unique(source_beam_coordinate.node))
    system_a_fem.add_traceline(np.unique(receiver_beam_a_coordinate.node))
    system_a_fem.add_traceline([103,204,0,205,104])

    receiver_beam_b_locations = source_beam_locations.copy() + [1, 0, 0.2]

    receiver_beam_b_node = sdpy.geometry.node_array(id = np.concatenate((np.unique(source_beam_coordinate.node), np.unique(receiver_beam_b_coordinate.node))),
                                                    coordinate = np.concatenate((source_beam_locations, receiver_beam_b_locations)))

    system_b_fem = sdpy.geometry.Geometry(node = receiver_beam_b_node,
                                        coordinate_system = assembled_system_cs)

    system_b_fem.add_traceline(np.unique(source_beam_coordinate.node))
    system_b_fem.add_traceline(np.unique(receiver_beam_b_coordinate.node))
    system_b_fem.add_traceline([103,301,0,302,104])

    return system_a_fem, system_a, system_b_fem, system_b

def compute_benchmark_system_time_response(Fs = 400,
                                           T = 5,
                                           number_averages = 100,
                                           overlap = 0.5):
    """
    Computes the time response of the benchmark systems to random 
    excitation, which is applied to all the source beam DOFs.

    Parameters
    ----------
    Fs : float, optional
        The sampling frequency for computing the time response. The
        default is 400 Hz.
    T : float, optional
        The block time for each average that will be eventually used
        to compute the frequency domain response. The default is 5 s.
    number_averages : int, optional
        The number of averages that will eventually be used to compute
        the frequency domain response. The default is 100.
    overlap : float, optional
        The overlap (as a fraction) that will eventually be used to 
        compute the frequency domain response. The default is 0.5. 

    Returns
    -------
    random_excitation : TimeHistoryArray
        The excitation that was applied to system A and B. 
    system_a_response : TimeHistoryArray
        The time response of system A.
    system_b_response : TimeHistoryArray
        The time response of system B.

    Notes
    -----
    The time response is computed via time integration. 

    This function is intended to create repeatable responses for the 
    benchmarking functions, which is why there aren't many options. 
    """
    _, system_a, _, system_b = create_beam_systems()
    source_beam_coordinate = sdpy.coordinate_array(node=np.arange(9)[...,np.newaxis]+101, direction=[3,5]).flatten()
    response_coordinate_a = sdpy.coordinate_array(node=np.arange(start=0, stop=9, step=2)[...,np.newaxis]+201, direction=[3,5]).flatten()
    response_coordinate_b = sdpy.coordinate_array(node=np.arange(start=0, stop=9, step=2)[...,np.newaxis]+301, direction=[3,5]).flatten()

    N = int(T*Fs*(number_averages+1)*overlap+1)

    excitation_time = np.arange(N)/Fs
    rng = np.random.default_rng(seed=42)
    excitation_ordinate = rng.normal(loc=0, size=(source_beam_coordinate.shape[0], N))
    random_excitation = sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE, excitation_time, excitation_ordinate, source_beam_coordinate[...,np.newaxis])

    system_a_response, _ = system_a.time_integrate(random_excitation, responses=response_coordinate_a, integration_oversample=10)
    system_b_response, _ = system_b.time_integrate(random_excitation, responses=response_coordinate_b, integration_oversample=10)
    return random_excitation, system_a_response, system_b_response

def quantize_and_add_noise(time_data, snr=20, number_bits=16, range_utilization=20):
    """
    Adds measurement error to the time data via quantization error and 
    random noise. 

    Parameters
    ----------
    time_data : TimeHistoryArray
        The time data to add the measurement error to.
    snr : float, optional
        The target signal to noise ratio (in dB), based on the average RMS 
        level of the time_data. The default is 20.
    number_bits : int, optional
        The number of bits for the for the simulated quantization. The 
        default is 16.
    range_utilization : float, optional
        The proportion of the range in the simulated analog to digital 
        conversion. The range is computed by the maximum value of the 
        time_data*2*range_utilization. The default is 20.

    Returns
    -------
    noised_data : TimeHistoryArray
        The time_data with the simulated measurement errors. 
    """
    time_rms = np.mean(time_data.rms())
    rms_multiplier = 10**(snr/20)

    rng = np.random.default_rng(seed=504)
    noise_time = rng.normal(loc=0, size=(time_data.ordinate.shape[0], time_data.ordinate.shape[1]))*time_rms/rms_multiplier
    noised_data = time_data.ordinate + noise_time

    number_bins = 2**int(number_bits)
    adc_resolution = (np.abs(time_data.ordinate).max()*2*range_utilization)/number_bins
    noised_quantized_data = adc_resolution*np.round(noised_data/adc_resolution)

    return sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE, time_data.ravel().abscissa[0], 
                           noised_quantized_data, time_data.response_coordinate[...,np.newaxis])

def create_beam_system_ise_frfs(abscissa=None):
    """
    Creating the FRFs from the beam systems (to perform the ISE) to benchmark 
    the various inverse methods in the different SourcePathReceiver objects.   

    Parameters
    ----------
    abscissa : ndarray
        A 1d array of the abscissa to compute the FRFs over. 

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
    This function is intended to create repeatable FRFs for benchmarking 
    functions, which is why there aren't any options. 
    """
    system_a_fem, system_a, system_b_fem, system_b = create_beam_systems()

    ise_reference_coordinate = sdpy.coordinate_array(node=np.array([103,104])[...,np.newaxis], direction=[3,5]).flatten()
    receiver_beam_a_coordinate = sdpy.coordinate_array(node=np.arange(start=0, stop=9, step=2)[...,np.newaxis]+201, direction=[3,5]).flatten()
    receiver_beam_b_coordinate = sdpy.coordinate_array(node=np.arange(start=0, stop=9, step=2)[...,np.newaxis]+301, direction=[3,5]).flatten()

    if abscissa is None:
        compute_frequency = np.arange(2001)*0.2
        ise_frequency = np.arange(1001)*0.2
    else:
        compute_frequency = abscissa
        ise_frequency = abscissa

    # Making the FRFs for system A
    system_a_ise_frfs = system_a.frequency_response(frequencies=compute_frequency, responses=receiver_beam_a_coordinate, 
                                                      references=ise_reference_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())

    # Making the FRFs for system B
    system_b_ise_frfs = system_b.frequency_response(frequencies=compute_frequency, responses=receiver_beam_b_coordinate, 
                                                      references=ise_reference_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())

    return system_a_ise_frfs, system_b_ise_frfs

def create_transient_excitation(force_coordinate=sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+101, direction=[3,5]),
                                sampling_rate=200,
                                number_samples=1000,
                                zero_pad_samples=4000):
    """
    Creates the transient excitation to benchmark the various inverse 
    methods in the transient SourcePathReceiver objects.  

    Parameters
    ----------
    force_coordinate : sdpy.CoordinateArray, optional
        The coordinate array for the force degrees of freedom. The default
        is to match the benchmark beam system
    sampling_rate : float, optional
        The sampling rate for the force in Hz. The default is 200 Hz, which
        matches the benchmark beam system.
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
    benchmarking functions, which is why there aren't many options.

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
        
        chirp_start_frequency = np.random.uniform(low=5, high=30)
        chirp_stop_frequency = np.random.uniform(low=60, high=90)

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
    sos_filter = butter(10, [5,95], btype='bandpass', output='sos', fs=1/(time[1]-time[0]))

    force_ordinate = sosfiltfilt(sos_filter, (random_excitation+chirp_base+pulse_base)*window[np.newaxis, ...], axis=-1)
    force = sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE, time, force_ordinate, force_coordinate[...,np.newaxis])
    force = force.zero_pad(num_samples=zero_pad_samples,left=True,right=True)
    return force