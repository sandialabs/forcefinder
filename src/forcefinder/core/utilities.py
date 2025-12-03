"""
Creates some utility functions for the ForceFinder package.

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
from sdynpy.signal_processing.sdynpy_cpsd import cpsd_coherence, cpsd_from_coh_phs

def check_frequency_abscissa(data, reference_abscissa):
    """
    Checks the abscissa of the data for building a source-path-receiver 
    model. It validates that the data has a common abscissa for all
    the degrees of freedom and that the abscissa for data and reference
    data match. 

    Parameters
    ----------
    data : NDDataArray
        The data to check the abscissa on.
    reference_abscissa : ndarray
        The reference abscissa to compare the data abscissa against.

    Raises
    ------
    ValueError
        If the abscissa from data doesn't match reference_data.
    """
    data.abscissa_spacing
    if not np.all(data.flatten()[0].abscissa==reference_abscissa):
        raise ValueError('The abscissa for the data does not match')

def compare_sampling_rate(time_data, reference_sampling_rate):
    """
    Checks that the sampling rate of the supplied time_data matches the 
    reference_reference_sampling tate. The primary purpose of this is 
    to ensure that the response/force and FRFs have the same sampling 
    rate when constructing a transient source-path-receiver model.

    Parameters
    ----------
    time_data : TimeHistoryArray
        The data to check the sampling rate on.
    reference_sampling_rate : float
        The reference sampling rate to compare against

    Raises
    ------
    ValueError
        If the sampling rate in time_data doesn't match the reference   
    """
    fs_time_data = 1/time_data.abscissa_spacing
    if not np.isclose(fs_time_data, reference_sampling_rate):
        raise ValueError('The sampling rate for the data does not match')

def is_cpsd(data_array):
    """
    Function to check if the supplied data array contains PSDs or CPSDs.

    Parameters
    ----------
    data_array : PowerSpectralDensityArray  
        data array to check.

    Returns
    -------
    bool
        True if the data array contains CPSDs, False otherwise
    """
    if isinstance(data_array, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
        try: 
            data_array.reshape_to_matrix()
            return True
        except:
            return False
    elif data_array.ndim == 3:
        return True
    else:
        return False
    
def apply_buzz_method(self):
    """
    Applies the buzz method using the information in the SPR object.

    References
    ----------
    .. [1] P. Daborn, "Smarter dynamic testing of critical structures," PhD dissertation, 
            Aerospace Department, University of Bristol, 2014
    """
    if self._buzz_cpsd_array_ is None:
        raise AttributeError('A buzz CPSD has not been supplied for the SPR object.')
    phase = np.angle(self._buzz_cpsd_array_)
    coherence = cpsd_coherence(self._buzz_cpsd_array_)
    if self._training_response_array_.ndim == 3:
        asds = np.diagonal(self._training_response_array_, axis1=1, axis2=2)
        new_cpsd = cpsd_from_coh_phs(asds, coherence, phase)
    elif self._training_response_array_.ndim == 2:
        new_cpsd = cpsd_from_coh_phs(self._training_response_array_, coherence, phase)
    return new_cpsd

def reduce_drives_condition_not_met(training_psd, reconstructed_psd, 
                                      reduced_drive_reconstructed_psd, 
                                      db_error_ratio):
    """
    Evaluates the reconstructed response from the reduce_drives_update to
    see if the optimality condition was met. 

    Parameters 
    ----------
    training_psd : ndarray
        The training PSDs for the SPR, which is the diagonal of the training 
        response. It should be shaped [number of lines, number of dofs].
    reconstructed_psd : ndarray
        The reconstructed training PSDs for the SPR with the non-reduced forces, 
        which is the diagonal of the reconstructed training response. It should 
        be shaped [number of lines, number of dofs].
    reconstructed_psd : ndarray
        The reconstructed training PSDs for the SPR with the reduced forces, 
        which is the diagonal of the reconstructed training response. It should 
        be shaped [number of lines, number of dofs].
    db_error_ratio : float
        The dB error ratio that was used in the reduced drives update. 

    Returns
    -------
    bool
        Returns True if the reduce drives optimality condition is not met. Returns
        False if the condition is meth
    """    
    db_diffs = db_error_ratio*np.log10(reconstructed_psd/training_psd)
    y_lb = training_psd*(10**(-np.abs(db_diffs))) # LB is current dB error below spec
    y_ub = training_psd*(10**(np.abs(db_diffs))) # UB is current dB error above spec

    condition_checker = (reduced_drive_reconstructed_psd-y_lb)/(y_ub-y_lb)

    if not (np.isclose(condition_checker.min(), 0) or condition_checker.min() >= 0):
        return True
    elif not (np.isclose(condition_checker.max(), 1) or condition_checker.max() <= 1):
        return True
    else:
        return False