"""
Contains some utilities for the benchmark and unit tests. 

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
from scipy.signal.windows import hann

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

    source_beam_stiffness[0,0] = 1e9 # N/m, translating spring to ground on left side
    source_beam_stiffness[1,1] = 1e2 # Nm/rad, torsion spring to ground on left side 

    source_beam_stiffness[-2,-2] = 1e9 # N/m, translating spring to ground on right side
    source_beam_stiffness[-1,-1] = 1e2 # Nm/rad, torsion spring to ground on right side 

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

def create_beam_system_frfs():
    """
    Creating the FRFs from the beam systems to benchmark the various 
    inverse methods in the different SourcePathReceiver objects.   

    Returns
    -------
    system_a_truth_frfs : sdpy.TransferFunctionArray
        The FRFs that will be used to create the truth data for system A.
    system_a_ise_frfs : sdpy.TransferFunctionArray
        The FRFs for system A that will be used to evaluate the different 
        inverse methods.
    system_b_truth_frfs : sdpy.TransferFunctionArray
        The FRFs that will be used to create the truth data for system B.
    system_b_ise_frfs : sdpy.TransferFunctionArray
        The FRFs for system B that will be used to evaluate the different 
        inverse methods.
    """
    system_a_fem, system_a, system_b_fem, system_b = create_beam_systems()

    source_beam_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+101, direction=[3,5]).flatten()
    ise_reference_coordinate = sdpy.coordinate_array(node=np.array([103,104])[...,np.newaxis], direction=[3,5]).flatten()
    receiver_beam_a_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+201, direction=[3,5]).flatten()
    receiver_beam_b_coordinate = sdpy.coordinate_array(node=np.arange(5)[...,np.newaxis]+301, direction=[3,5]).flatten()

    compute_frequency = np.arange(1001)*0.2
    ise_frequency = np.arange(501)*0.2

    # Making the FRFs for system A
    system_a_truth_frfs = system_a.frequency_response(frequencies=compute_frequency, responses=receiver_beam_a_coordinate, 
                                                      references=source_beam_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())
    
    ise_frf_coordinate_a = sdpy.coordinate.outer_product(receiver_beam_a_coordinate, ise_reference_coordinate) 
    system_a_ise_frfs = system_a_truth_frfs[ise_frf_coordinate_a]

    # Making the FRFs for system B
    system_b_truth_frfs = system_b.frequency_response(frequencies=compute_frequency, responses=receiver_beam_b_coordinate, 
                                                      references=source_beam_coordinate, displacement_derivative=2).extract_elements_by_abscissa(ise_frequency.min(), ise_frequency.max())
    
    ise_frf_coordinate_b = sdpy.coordinate.outer_product(receiver_beam_b_coordinate, ise_reference_coordinate) 
    system_b_ise_frfs = system_b_truth_frfs[ise_frf_coordinate_b]

    return system_a_truth_frfs, system_a_ise_frfs, system_b_truth_frfs, system_b_ise_frfs

def create_broadband_random_spectral_excitation(force_coordinate, frequency):
    """
    Creates broadband random excitation to benchmark the various inverse 
    methods in the different linear and power SourcePathReceiver objects.  

    Parameters
    ----------
    force_coordinate : sdpy.CoordinateArray
        The coordinate array for the force degrees of freedom.
    frequency : ndarray
        A 1d array for the frequencies to compute the force over.

    Returns
    -------
    force_spectrum : sdpy.SpectrumArray
        The force in linear spectrum format.
    force_power : sdpy.PowerSpectralDensityArray
        The force in power spectrum format as a cross-power spectral
        density matrix. 
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