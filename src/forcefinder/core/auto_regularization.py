"""
Contains helper functions for performing automatic regularization with
the truncated singular value decomposition, ridge regression (Tikhonov), 
and the elastic net. These functions are intended to work with data from
a SourcePathReceiver object.

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
from sdynpy.signal_processing.sdynpy_frf_inverse import pinv_by_tikhonov
from .utilities import is_cpsd
from scipy.linalg import norm
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from secrets import randbits

#%% Functions for L-curve automatic regularization 

def tikhonov_full_path_for_l_curve(frfs, response, 
                                   low_regularization_limit=None,
                                   high_regularization_limit=None, 
                                   number_regularization_values=100):
    """
    Performs the inverse force estimation problem and computes the necessary residual and
    penalty for model selection with the L-curve criterion.

    Parameters
    ----------
    frfs : ndarray
        The array of FRFs for the inverse force estimation problem, should be organized
        [number of lines, number of responses, number of references].
    response : ndarray
        The response to compute the forces from. Should be sized [number of lines, number of responses]
        for linear spectra or [number of lines, number of responses, number of responses] for power spectra.
    low_regularization_limit : ndarray, optional
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the frf array (on a frequency-frequency basis). 
    high_regularization_limit : ndarray, optional
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the largest singular
        value of the frf array (on a frequency-frequency basis). 
    number_regularization_values : int, optional
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits. The default is 100.

    Returns
    -------
    regularization_values : ndarray
        The regularization parameters that were used in the inverse. It is sized
        [number of parameters, number of lines].
    residual : ndarray
        The residual error (i.e., sum squared error) between the computed and truth response. 
        It is sized [number of parameters, number of lines].
    penalty : ndarray
        The penalty to use in the L-curve parameter selection, which is the squared sum
        of the forces. It is sized [number of parameters, number of lines].

    Notes
    -----
    The regularization values are spread over a geometric sequence that spans
    the low and high regularization limits. 

    The high and low regularization parameter limits cannot be set to zero in this function.
    Any values that are zero are reset to machine epsilon. 
    """
    U, S, Vh = np.linalg.svd(frfs, full_matrices=False, compute_uv=True)
    Uh = np.moveaxis(U.conj(),-2,-1)
    V = np.moveaxis(Vh.conj(),-2,-1)
    del U, Vh
    
    if low_regularization_limit is None:
        low_regularization_limit = S[:, -1]
    if high_regularization_limit is None:
        high_regularization_limit = S[:, 0]
    
    if np.any(low_regularization_limit==0):
        low_regularization_limit[np.where(low_regularization_limit==0)] = np.finfo(float).eps 
    if np.any(high_regularization_limit==0):
        high_regularization_limit[np.where(high_regularization_limit==0)] = np.finfo(float).eps

    regularization_values = np.geomspace(low_regularization_limit, high_regularization_limit, num=number_regularization_values, axis=0)
    regularized_S = S[np.newaxis,...]/(S[np.newaxis,...]**2+regularization_values[...,np.newaxis])
    del S
    
    residual = np.zeros((number_regularization_values, frfs.shape[0]), dtype=float)
    penalty = np.zeros((number_regularization_values, frfs.shape[0]), dtype=float)
    
    # Everything is done in a single loop over the available regularization parameters to minimize 
    # overhead in some broadcasting operations and unnecessarily loading large arrays (such as the 
    # FRF inverse array) in memory.
    if not is_cpsd(response): #Linear Spectra
        for ii in range(number_regularization_values):
            regularized_frf_pinv = V@(regularized_S[ii,...,np.newaxis]*Uh)
            regularized_forces = regularized_frf_pinv@response[...,np.newaxis]
            residual[ii,...] = norm(response - np.squeeze(frfs@regularized_forces), axis=-1, ord=2)**2
            penalty[ii,...] = norm(regularized_forces[...,0], axis=-1, ord=2)**2
    
    elif is_cpsd(response): #Power Spectra
        for ii in range(number_regularization_values):
            regularized_frf_pinv = V@(regularized_S[ii,...,np.newaxis]*Uh) 
            regularized_forces = regularized_frf_pinv @ response @ np.moveaxis(regularized_frf_pinv.conj(),-1,-2) 
            residual[ii,...] = norm(response - frfs @ regularized_forces @ np.moveaxis(frfs.conj(),-2,-1), axis=(-2,-1), ord='fro')**2 
            penalty[ii,...] = norm(regularized_forces, axis=(-2,-1), ord='fro')**2
    
    return regularization_values, residual, penalty

def compute_regularized_residual_penalty_for_l_curve(frfs, response,
                                                     Uh, regularized_S, V):
    """
    Computes the residual and penalty from the regularized SVD components of an FRF 
    for an automatic Tikhonov regularization problem. 

    Parameters
    ----------
    frfs : ndarray
        The array of FRFs for the inverse force estimation problem, should be organized
        [number of lines, number of responses, number of references].
    response : ndarray
        The response to compute the forces from. Should be sized [number of lines, number of responses]
        for linear spectra or [number of lines, number of responses, number of responses] for power spectra.
    Uh : ndarray
        The conjugate-transpose of the left singular vectors of the supplied FRFs. 
        It is shaped [number of lines, number of responses, number of references].
    V : ndarray
        The right singular vectors of the supplied FRFs. It is shaped
        [number of lines, number of references, number of references].
    regularized_S : ndarray
        The inverted and Tikhonov regularized singular values of the supplied 
        FRFs. It is shaped [number of values, number of lines, number of references].

    Returns
    -------
    regularization_values : ndarray
        The regularization parameters that were used in the inverse. It is sized
        [number of parameters, number of lines].
    residual : ndarray
        The residual error (i.e., sum squared error) between the computed and truth response. 
        It is sized [number of parameters, number of lines].
    penalty : ndarray
        The penalty to use in the L-curve parameter selection, which is the squared sum
        of the forces. It is sized [number of parameters, number of lines].
    """
    number_regularization_values = regularized_S.shape[0]

    residual = np.zeros((number_regularization_values, frfs.shape[0]), dtype=float)
    penalty = np.zeros((number_regularization_values, frfs.shape[0]), dtype=float)
    
    # Everything is done in a single loop over the available regularization parameters to minimize 
    # overhead in some broadcasting operations and unnecessarily loading large arrays (such as the 
    # FRF inverse array) in memory.
    if not is_cpsd(response): #Linear Spectra
        for ii in range(number_regularization_values):
            regularized_frf_pinv = V@(regularized_S[ii,...,np.newaxis]*Uh)
            regularized_forces = regularized_frf_pinv@response[...,np.newaxis]
            residual[ii,...] = norm(response - np.squeeze(frfs@regularized_forces), axis=-1, ord=2)**2
            penalty[ii,...] = norm(regularized_forces[...,0], axis=-1, ord=2)**2
    
    elif is_cpsd(response): #Power Spectra
        for ii in range(number_regularization_values):
            regularized_frf_pinv = V@(regularized_S[ii,...,np.newaxis]*Uh) 
            regularized_forces = regularized_frf_pinv @ response @ np.moveaxis(regularized_frf_pinv.conj(),-1,-2) 
            residual[ii,...] = norm(response - frfs @ regularized_forces @ np.moveaxis(frfs.conj(),-2,-1), axis=(-2,-1), ord='fro')**2 
            penalty[ii,...] = norm(regularized_forces, axis=(-2,-1), ord='fro')**2
    
    return residual, penalty

def compute_regularized_svd_inv(frfs, 
                                low_regularization_limit=None,
                                high_regularization_limit=None, 
                                number_regularization_values=100):
    """
    Computes the inverted regularized SVD components for an automatic Tikhonov
    regularization problem. 

    Parameters
    ----------
    frfs : ndarray
        The array of FRFs for the inverse force estimation problem, should be organized
        [number of lines, number of responses, number of references].
    low_regularization_limit : ndarray, optional
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the frf array (on a frequency-frequency basis). 
    high_regularization_limit : ndarray, optional
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the largest singular
        value of the frf array (on a frequency-frequency basis). 
    number_regularization_values : int, optional
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits. The default is 100.

    Returns
    -------
    regularization_values : ndarray
        The regularization parameters that were used in the inverse. It is sized
        [number of parameters, number of lines].
    Uh : ndarray
        The conjugate-transpose of the left singular vectors of the supplied FRFs. 
        It is shaped [number of lines, number of responses, number of references].
    V : ndarray
        The right singular vectors of the supplied FRFs. It is shaped
        [number of lines, number of references, number of references].
    regularized_S : ndarray
        The inverted and Tikhonov regularized singular values of the supplied 
        FRFs. It is shaped [number of values, number of lines, number of references].

    Notes
    -----
    The regularization values are spread over a geometric sequence that spans
    the low and high regularization limits. 

    The high and low regularization parameter limits cannot be set to zero in this function.
    Any values that are zero are reset to machine epsilon. 
    """
    U, S, Vh = np.linalg.svd(frfs, full_matrices=False, compute_uv=True)
    Uh = np.moveaxis(U.conj(),-2,-1)
    V = np.moveaxis(Vh.conj(),-2,-1)
    del U, Vh
    
    if low_regularization_limit is None:
        low_regularization_limit = S[:, -1]
    if high_regularization_limit is None:
        high_regularization_limit = S[:, 0]
    
    if np.any(low_regularization_limit==0):
        low_regularization_limit[np.where(low_regularization_limit==0)] = np.finfo(float).eps 
    if np.any(high_regularization_limit==0):
        high_regularization_limit[np.where(high_regularization_limit==0)] = np.finfo(float).eps

    regularization_values = np.geomspace(low_regularization_limit, high_regularization_limit, num=number_regularization_values, axis=0)
    regularized_S = S[np.newaxis,...]/(S[np.newaxis,...]**2+regularization_values[...,np.newaxis])
    
    return regularization_values, Uh, V, regularized_S

#%% Functions for cross-validation automatic regularization

def mean_squared_error(validation_frfs, validation_response, frf_inverse, training_response):
    """
    Computes the mean squared error of the validation response(s) from the 
    inverse problem.

    Parameters
    ----------
    validation_frfs : ndarray
        The FRFs that are used to compute the predicted response(s) from the 
        predicted forces. Should be sized [number of lines, number of validation responses, number of references]. 
    validation_response : ndarray
        The validation response to compare the predicted response against. 
        Should be sized [number of lines, number of validation responses].
    frf_inverse : ndarray
        The inverted FRFs that are used to compute the predicted forces. 
        Should be sized [number of lines, number of references, number of training responses].
    training_response : ndarray
        The training responses that are used to compute the predicted forces.
        Should be sized [number of lines, number of training responses].

    Returns
    -------
    mean_squared_error : ndarray
        The mean squared error of the predicted response. It is a 1D array that
        is sized the number of lines. 

    Notes
    -----
    This function assumes that there are at least two DOFs in the training response.
    """
    if not is_cpsd(training_response):
        predicted_response = validation_frfs@(frf_inverse@training_response[...,np.newaxis])
        
        if validation_response.ndim == 1: # loocv
            error = np.abs(validation_response[...,np.newaxis,np.newaxis] - predicted_response)
        else: # k-fold
            error = np.abs(validation_response[...,np.newaxis] - predicted_response)
        
        return np.mean(error**2, axis=1)[...,0]
    
    if is_cpsd(training_response):
        regularized_forces = frf_inverse @ training_response @ np.moveaxis(frf_inverse.conj(),-1,-2) 
        predicted_response = validation_frfs @ regularized_forces @ np.moveaxis(validation_frfs.conj(),-2,-1)
        
        if validation_response.ndim == 1: # loocv
            error = np.abs(validation_response[...,np.newaxis] - predicted_response[...,0])
        else: # k-fold
            error = np.abs(validation_response.diagonal(axis1=1,axis2=2) - predicted_response.diagonal(axis1=1,axis2=2))
        
        return np.mean(error**2, axis=1)

def leave_one_out_cv(frfs, response, 
                     low_regularization_limit=None,
                     high_regularization_limit=None, 
                     number_regularization_values=100,
                     error_function = mean_squared_error):
    """
    Estimates the error in a force estimation problem with leave one out cross
    validation. 

    Parameters
    ----------
    frfs : ndarray
        The full FRF array to use in the inverse problem. It should be sized 
        [number of lines, number of responses, number of references].
    response : ndarray
        The full response array to use in the inverse problem. It should be
        sized [number of lines, number of responses].
    low_regularization_limit : ndarray, optional
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the frf array (on a frequency-frequency basis). 
    high_regularization_limit : ndarray, optional
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the largest singular
        value of the frf array (on a frequency-frequency basis). 
    number_regularization_values : int, optional
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits. The default is 100.
    error_function : function, optional
        The function that is used to compute the error in the inverse 
        problem. The function should take the following parameters:
            - validation_frfs - ndarray that is sized [number of lines, number of validation responses, number of references]. 
            - validation_response - ndarray that is sized [number of lines, number of responses].
            - frf_inverse - ndarray that is sized [number of lines, number of references, number of training responses].
            - training_response - ndarray that is sized [number of lines, number of training responses, 1].
        The function should output:
            - error - a 1d array that is sized number of lines.
        The default function computes the residual squared error.
    
    Returns
    -------
    regularization_values : ndarray
        The regularization values for the inverse problem. It is sized
        [number_regularization_values, number of lines].
    error : ndarray
        The prediction error from the inverse problem. It is sized
        [number_regularization_values, number of responses, number of lines].
    """
    # Setting the regularization limits based on the full FRF array to have 
    # consistent parameters for all the subsamples
    if low_regularization_limit is None or high_regularization_limit is None:
        S = np.linalg.svd(frfs, full_matrices=False, compute_uv=False)
        if low_regularization_limit is None:
            low_regularization_limit = S[:,-1]
        if high_regularization_limit is None:
            high_regularization_limit = S[:,0]
        del S

    if np.any(low_regularization_limit==0):
        low_regularization_limit[np.where(low_regularization_limit==0)] = np.finfo(float).eps 
    if np.any(high_regularization_limit==0):
        high_regularization_limit[np.where(high_regularization_limit==0)] = np.finfo(float).eps

    regularization_values = np.zeros((number_regularization_values, frfs.shape[0]), dtype=float)
    error = np.zeros((number_regularization_values, frfs.shape[1], frfs.shape[0]), dtype=float)

    for ii in range(frfs.shape[1]):
        training_indices = np.array([idx for idx in range(frfs.shape[1]) if idx != ii])
        current_regularization_values, Uh, V, regularized_S = compute_regularized_svd_inv(frfs[:,training_indices,:], 
                                                                                          low_regularization_limit=low_regularization_limit, 
                                                                                          high_regularization_limit=high_regularization_limit, 
                                                                                          number_regularization_values=number_regularization_values)
        if ii==0:
            regularization_values = current_regularization_values
        else:
            if not np.all(current_regularization_values==regularization_values):
                raise ValueError('The regularization values are not the same for all the folds')
        
        for jj in range(number_regularization_values):
            regularized_frf_pinv = V@(regularized_S[jj,...,np.newaxis]*Uh) 
            if not is_cpsd(response):
                error[jj, ii, ...] = error_function(validation_frfs = frfs[:,ii,:][:,np.newaxis,:], 
                                                    validation_response = response[:,ii], 
                                                    frf_inverse = regularized_frf_pinv, 
                                                    training_response = response[:,training_indices])
            if is_cpsd(response):
                training_slice = np.ix_(np.arange(response.shape[0]), training_indices, training_indices)
                error[jj, ii, ...] = error_function(validation_frfs = frfs[:,ii,:][:,np.newaxis,:], 
                                                    validation_response = response[:,ii,ii], 
                                                    frf_inverse = regularized_frf_pinv, 
                                                    training_response = response[training_slice])

    return regularization_values, error

def k_fold_cv(frfs, response,
              number_folds=None, 
              random_seed=randbits(128),
              low_regularization_limit=None,
              high_regularization_limit=None, 
              number_regularization_values=100,
              error_function = mean_squared_error):
    """
    Estimates the error in a force estimation problem with K-fold cross
    validation. 

    Parameters
    ----------
    frfs : ndarray
        The full FRF array to use in the inverse problem. It should be sized 
        [number of lines, number of responses, number of references].
    response : ndarray
        The full response array to use in the inverse problem. It should be
        sized [number of lines, number of responses].
    number_folds : int
        The number of folds to use in the cross validation. The number of 
        response DOFs must be evenly divisible by the number of folds.
    random_seed : int
        The seed for the random number generator that selects response DOFs
        that are used in the different folds of the cross validation. The 
        default is a randomly selected integer.  
    low_regularization_limit : ndarray, optional
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the frf array (on a frequency-frequency basis). 
    high_regularization_limit : ndarray, optional
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the largest singular
        value of the frf array (on a frequency-frequency basis). 
    number_regularization_values : int, optional
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits. The default is 100.
    error_function : function, optional
        The function that is used to compute the error in the inverse 
        problem. The function should take the following parameters:
            - validation_frfs - ndarray that is sized [number of lines, number of validation responses, number of references]. 
            
            - validation_response - ndarray that is sized [number of lines, number of responses].
            
            - frf_inverse - ndarray that is sized [number of lines, number of references, number of training responses].
            
            - training_response - ndarray that is sized [number of lines, number of training responses, 1].
        The function should return a 1d array that is sized number of lines.
        The default function computes the mean squared error.
    
    Returns
    -------
    regularization_values : ndarray
        The regularization values for the inverse problem. It is sized
        [number_regularization_values, number of lines].
    error : ndarray
        The prediction error from the inverse problem. It is sized
        [number_regularization_values, number of folds, number of lines].

    Raises
    ------
    ValueError 
        If the number of response DOFs is not evenly divisible by the number
        of folds. 

    Notes
    -----
    The response DOFs for the cross validation folds are selected by a 
    random number generator. This can lead to different results for multiple
    function runs, since the default behavior is to use a different random
    number generator every time the function is ran. A random seed can be 
    supplied to avoid the variability in the response DOF selection. 
    However, different computers may still select different response DOFs 
    for the different folds (even with the same seed), for a variety of 
    reasons. 
    """
    fold_size = int(np.ceil(response.shape[1]/number_folds))
    if fold_size*number_folds != response.shape[1]:
        raise ValueError('The number of response DOFs must be evenly divisible by the number of folds')

    # Setting the regularization limits based on the full FRF array to have 
    # consistent parameters for all the subsamples
    if low_regularization_limit is None or high_regularization_limit is None:
        S = np.linalg.svd(frfs, full_matrices=False, compute_uv=False)
        if low_regularization_limit is None:
            low_regularization_limit = S[:,-1]
        if high_regularization_limit is None:
            high_regularization_limit = S[:,0]
        del S

    if np.any(low_regularization_limit==0):
        low_regularization_limit[np.where(low_regularization_limit==0)] = np.finfo(float).eps 
    if np.any(high_regularization_limit==0):
        high_regularization_limit[np.where(high_regularization_limit==0)] = np.finfo(float).eps

    rng = np.random.default_rng(random_seed)
    random_index = rng.choice(response.shape[1], size=fold_size*number_folds, replace=False)
    error = np.zeros((number_regularization_values, number_folds, frfs.shape[0]), dtype=float)

    for ii in range(number_folds):
        fold_indices = np.array([random_index[i] for i in range(random_index.shape[0]) if i not in np.arange(start=ii*fold_size, stop=ii*fold_size+fold_size)])
        fold_validation_indices = random_index[np.arange(start=ii*fold_size, stop=ii*fold_size+fold_size)]
        current_regularization_values, Uh, V, regularized_S = compute_regularized_svd_inv(frfs[:,fold_indices,:], 
                                                                                          low_regularization_limit=low_regularization_limit, 
                                                                                          high_regularization_limit=high_regularization_limit, 
                                                                                          number_regularization_values=number_regularization_values)
        
        if ii==0:
            regularization_values = current_regularization_values
        else:
            if not np.all(current_regularization_values==regularization_values):
                raise ValueError('The regularization values are not the same for all the folds')

        for jj in range(number_regularization_values):
            regularized_frf_pinv = V@(regularized_S[jj,...,np.newaxis]*Uh) 
            if not is_cpsd(response):
                error[jj, ii, ...] = error_function(validation_frfs = frfs[:,fold_validation_indices,:], 
                                                    validation_response = response[:,fold_validation_indices], 
                                                    frf_inverse = regularized_frf_pinv, 
                                                    training_response = response[:,fold_indices])[...,0]
            elif is_cpsd(response):
                validation_slice = np.ix_(np.arange(response.shape[0]), fold_validation_indices, fold_validation_indices)
                training_slice = np.ix_(np.arange(response.shape[0]), fold_indices, fold_indices)
                error[jj, ii, ...] = error_function(validation_frfs = frfs[:,fold_validation_indices,:], 
                                                    validation_response = response[validation_slice], 
                                                    frf_inverse = regularized_frf_pinv, 
                                                    training_response = response[training_slice])[...,0]

    return regularization_values, error

#%% Model Selection

def l_curve_optimal_regularization(regularization_values, penalty, residual,
                                   l_curve_type = 'forces',
                                   optimality_condition = 'curvature'):
    """
    Selects the optimal regularization parameter using L-curve methods.

    Parameters
    ----------
    regularization_values : ndarray
        The regularization values that were searched over. This should be sized
        [number of regularization values, number of frequency lines].
    penalty : ndarray
        The penalty from the regularized least squares problem. This should be sized
        [number of regularization values, number of frequency lines].
    residual : ndarray
        The residual from the regularized least squares problem (typically the 
        mean squared error). This should be sized 
        [number of regularization values, number of frequency lines].
    l_curve_type : str
        The type of L-curve that is used to find the "optimal regularization 
        parameter. The available types are:
            - forces (default) - This L-curve is constructed with the "size" 
            of the forces on the Y-axis and the regularization parameter on the 
            X-axis. 

            - standard - This L-curve is constructed with the residual squared 
            error on the X-axis and the "size" of the forces on the Y-axis. 
    optimality_condition : str
        The method that is used to find an "optimal" regularization parameter.
        The options are:
            - curvature (default) - This method searches for the regularization
            parameter that results in maximum curvature of the L-curve. It is 
            also referred to as the L-curve criterion. 

            - distance - This method searches for the regularization parameter that
            minimizes the distance between the L-curve and a "virtual origin". A 
            virtual origin is used, because the L-curve is scaled and offset to always 
            range from zero to one, in this case.
    
    Returns
    -------
    optimal_regularization : ndarray
        A vector of the optimal regularization values, as defined by the L-curve. 
        The length matches that number of frequency lines.

    Raises
    ------
    ValueError
        If the requested L-curve type is not available.
    ValueError
        If the requested optimality condition is not available.   
    
    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
        of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
        vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
        problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
        2000, pp. 119-142.  
    .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
        Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.
    """
    if not np.any(np.array(['standard', 'forces']) == l_curve_type):
        raise ValueError('The selected L-curve type is not available')
    if not np.any(np.array(['curvature', 'distance']) == optimality_condition):
        raise ValueError('The selected optimality condition is not available')

    if optimality_condition=='curvature':
        if l_curve_type=='standard':
            optimal_regularization, idx = broadcasting_l_curve_criterion(residual, penalty, regularization_values)
        elif l_curve_type=='forces':
            optimal_regularization, idx = broadcasting_l_curve_criterion(regularization_values, penalty, regularization_values)
    if optimality_condition=='distance':
        if l_curve_type=='standard':
            optimal_regularization, idx = broadcasting_l_curve_by_distance(residual, penalty, regularization_values)
        elif l_curve_type=='forces':
            optimal_regularization, idx = broadcasting_l_curve_by_distance(regularization_values, penalty, regularization_values)
        
    return optimal_regularization

def broadcasting_l_curve_criterion(x_axis, y_axis, regularization_values, return_curvature=False):
    """
    Finds the "optimal" regularization value from an L-curve via the 
    location where its curvature is at a maximum (the L-curve criterion).

    Parameters
    ----------
    x_axis : ndarray
        This is a vector that defines the X-axis of the L-curve. The variable 
        that is used for this depends on the type of L-curve that is being 
        used. This should be sized [number of regularization values, number of frequency lines].
    y_axis : ndarray
        This is a vector that defines the Y-axis of the L-curve. The variable 
        that is used for this depends on the type of L-curve that is being 
        used. This should be sized [number of regularization values, number of frequency lines].
    regularization_values : ndarray
        This is a vector of regularization values that were used in the linear
        regression problem that created the L-curve. This should be sized
        [number of regularization values, number of frequency lines].
    return_curvature : bool
        Whether or not to return the computed curvature function. The default is 
        False. 

    Returns
    -------
    optimal_regularization : ndarray
        This is the optimal regularization value based on the L-curve 
        criterion. It is sized [number of frequency lines]
    idx : ndarray
        The index that correspond to the optimal curvature. It is sized
        [number of frequency lines]
    curvature : ndarray, optional
        A vector of the curvature of the L-curve for the give sequence
        of regularization_values. This is not returned by default. It is
        sized [number of regularization values, number of frequency lines].
    
    Notes
    -----
    The code will automatically skip frequency lines where the regularization parameter 
    doesn't change (e.g., 0 Hz). 

    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
           of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
           vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
           problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
           2000, pp. 119-142.  
    """
    divisor = np.gradient(np.log(regularization_values), axis=0)

    nu_prime = np.gradient(np.log(y_axis[:, ~np.all(divisor==0, axis=0)]), axis=0) / divisor[:, ~np.all(divisor==0, axis=0)]
    nu_doubleprime = np.gradient(nu_prime, axis=0) / divisor[:, ~np.all(divisor==0, axis=0)]

    rho_prime = np.gradient(np.log(x_axis[:, ~np.all(divisor==0, axis=0)]), axis=0) / divisor[:, ~np.all(divisor==0, axis=0)]
    rho_doubleprime = np.gradient(rho_prime, axis=0) / divisor[:, ~np.all(divisor==0, axis=0)]

    curvature = np.zeros(regularization_values.shape, dtype=float)
    curvature[:, ~np.all(divisor==0, axis=0)] = (rho_prime*nu_doubleprime - nu_prime*rho_doubleprime) / np.power(nu_prime**2 + rho_doubleprime**2, 3/2)

    idx = np.abs(curvature).argmax(axis=0)

    if return_curvature:
        return regularization_values[idx, np.arange(regularization_values.shape[1])], idx, curvature
    else:
        return regularization_values[idx, np.arange(regularization_values.shape[1])], idx

def broadcasting_l_curve_by_distance(x_axis, y_axis, regularization_values):
    """
    Finds the "optimal" regularization from an L-curve by finding the parameter
    that puts the curve closest to the "virtual origin".

    Parameters
    ----------
    x_axis : ndarray
        This is a vector that defines the X-axis of the L-curve. The variable 
        that is used for this depends on the L-curve type this should be sized
        [number of regularization values, number of frequency lines].
    y_axis : ndarray
        This is a vector that defines the Y-axis of the L-curve. The variable 
        that is used for this depends on the L-curve type this should be sized 
        [number of regularization values, number of frequency lines]. 
    regularization_values : ndarray
        This is a vector of regularization values that were used in the linear
        regression problem that created the L-curve. This should be sized 
        [number of regularization values, number of frequency lines].

    Returns
    -------
    optimal_regularization : float
        This is the optimal regularization value based on the L-curve distance
        from the origin.
    idx : int
        The index that correspond to the optimal curvature.

    Notes
    -----
    This technique applies a scale and offset to the L-curve so the X and Y-axis
    always ranges from zero to one. This is required to obtain predictable 
    behavior from the method, but can also distort the shape of the curve.
    """
    rho = np.log(x_axis[:, ~np.all(regularization_values==0, axis=0)])
    rho = rho+np.sign(rho.min(axis = 0))*rho.min(axis = 0)
    rho = rho/rho.max(axis = 0)
    
    nu = np.log(y_axis[:, ~np.all(regularization_values==0, axis=0)])
    nu = nu+np.sign(nu.min(axis = 0))*nu.min(axis = 0)
    nu = nu/nu.max(axis = 0)

    idx = np.zeros(regularization_values.shape[1], dtype=int)
    idx[~np.all(regularization_values==0, axis=0)] = np.argmin(np.sqrt(nu**2 + rho**2), axis=0)

    return regularization_values[idx, np.arange(regularization_values.shape[1])], idx

def select_model_by_information_criterion(H, x, f, method):
    """
    Performs model selection with the desired information criterion.

    Parameters
    ----------
    H : ndarray
        The FRFs that were used to estimate the forces, sized: 
        [number_lines, number_responses, number_references].
    x : ndarray
        The responses that were used in the force estimation, sized:
        [number_lines, number_responses].
    f : ndarray
        The estimated forces from the model, sized:
        [number_lines, number_models, number_references]
    method : str
        The desired information criterion, the available options are:

            - 'BIC': the Bayesian information criterion
        
            - 'AIC': the Akaike information criterion
            
            - 'AICC': the corrected Akaike information criterion
    
    Returns
    -------
    selected_forces : ndarray
        The forces that were selected using the desired model, sized: 
        [number_lines, number_references]

    Raise
    -----
    NotImplementedError
        If the supplied responses are a CPSD matrix.
    """
    if is_cpsd(x):
        raise NotImplementedError('The information criterion code has not been implemented for power spectra')
    
    n = np.squeeze(x).shape[-1]
    p = f.shape[-1]
    residual = norm(np.squeeze(x[:, np.newaxis, ...] - H[:, np.newaxis, ...]@f[..., np.newaxis]), axis=-1)**2
    selected_force = np.zeros((f.shape[0], f.shape[-1]), dtype=complex)
    for ii in range(f.shape[0]):
        noise_variance = residual[ii, ...]/(n-p)
        number_active_parameters = np.count_nonzero(np.abs(f[ii, ...]), axis = -1)
        if method == 'BIC':
            criterion_factor = np.log(n)
            criterion_corrector = 0
        elif method == 'AIC':
            criterion_factor = 2
            criterion_corrector = 0
        elif method == 'AICC':
            criterion_factor = 2
            criterion_corrector = (2*number_active_parameters**2 + 2*number_active_parameters)/(n-number_active_parameters-1)
        criterion = n*np.log(2*np.pi*noise_variance) + residual[ii, ...]/noise_variance + criterion_factor * number_active_parameters + criterion_corrector
        selected_force[ii, ...] = f[ii, np.argmin(criterion), :]
    return selected_force

#%% Broadcasting Functions for Tikhonov regularization, saved since they might be useful, not directly used in anything
def compute_tikhonov_regularized_frf_pinv(frfs, low_regularization_limit=None, high_regularization_limit=None, number_regularization_values=100):
    """
    Computes the Tikhonov regularized FRF pseudo-inverse for many regularization parameters.

    Parameters
    ----------
    frfs : ndarray
        The array of FRFs to be inverted, should be organized
        [number of lines, number of responses, number of references].
    low_regularization_limit : ndarray
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the frf array (on a frequency-frequency basis). 
    high_regularization_limit : ndarray
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the largest singular
        value of the frf array (on a frequency-frequency basis). 
    number_regularization_values : int
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits. The default is 100. 
    
    Returns
    -------
    regularized_frf_pinv : ndarray
        Regularized pseudo-inverse of the FRF array. It is sized 
        [number of parameters, number of lines, number of responses, number of references].
    lambda_values : ndarray
        The regularization parameters that were used in the inverse. It is sized
        [number of parameters, number of lines].
    """
    U, S, Vh = np.linalg.svd(frfs, full_matrices=False, compute_uv=True)
    Uh = U.conj().transpose((0,2,1))
    V = Vh.conj().transpose((0,2,1))
    if np.any(low_regularization_limit) is None:
        low_regularization_limit = S[:, -1]
    if np.any(high_regularization_limit) is None:
        high_regularization_limit = S[:, 0]

    lambda_values = np.geomspace(low_regularization_limit, high_regularization_limit, num=number_regularization_values, axis=0)
    regularized_S = S[np.newaxis,...]/(S[np.newaxis,...]**2+lambda_values[...,np.newaxis])
    
    regularized_frf_pinv = np.zeros((number_regularization_values, frfs.shape[0], frfs.shape[2], frfs.shape[1]), dtype=complex)
    for ii in range(regularized_S.shape[0]):
        regularized_frf_pinv[ii,...] = V@(regularized_S[ii,...,np.newaxis]*Uh)
    
    return regularized_frf_pinv, lambda_values

def compute_regularized_forces(regularized_frf_inverse, response):
    """
    Computes the forces from a regularized inverse of an FRF matrix.

    Parameters
    ----------
    regularized_frf_inverse : ndarray
        Regularized pseudo-inverse of the FRF array. Should be sized 
        [number of parameters, number of lines, number of references, number of responses].
    response : ndarray
        The response to compute the forces from. Should be sized [number of lines, number of responses]
        for linear spectra or [number of lines, number of responses, number of responses] for power spectra.

    Returns
    -------
    regularized_forces : ndarray
        The regularized forces from the supplied parameters. It is sized
        [number of parameters, number of lines, number of references] for linear spectra
        or [number of parameters, number of lines, number of references, number of references]
        for power spectra.
    """
    if response.ndim == 2:
        regularized_forces = np.zeros((regularized_frf_inverse.shape[0], regularized_frf_inverse.shape[1], 
                                       regularized_frf_inverse.shape[2]), dtype=complex)
        for ii in range(regularized_frf_inverse.shape[0]):
            regularized_forces[ii,...] = (regularized_frf_inverse[ii,...]@response[...,np.newaxis])[...,0]
        return regularized_forces
    elif response.ndim == 3:
        regularized_forces = np.zeros((regularized_frf_inverse.shape[0], regularized_frf_inverse.shape[1], 
                                       regularized_frf_inverse.shape[2], regularized_frf_inverse.shape[2]), dtype=complex)
        for ii in range(regularized_frf_inverse.shape[0]):
            regularized_forces[ii,...] = regularized_frf_inverse[ii,...]@response@regularized_frf_inverse[ii,...].conj().transpose((0,2,1))
        return regularized_forces

#%% deprecated model selection functions 

"""
These functions are being kept for compatibility with the auto_truncation methods. They
will be eliminated as soon as the corresponding methods are updated to use the modern 
model selection functions. 
"""

def l_curve_selection(regularization_values, penalty, residual, forces_full_path,
                      l_curve_type = 'forces',
                      optimality_condition = 'curvature',
                      curvature_method = 'numerical'):
    """
    Selects the optimal regularization parameter and forces using L-curve methods

    Parameters
    ----------
    regularization_values : ndarray
        The regularization values that were searched over. This should be sized
        [number of frequency lines, number of regularization values]
    penalty : ndarray
        The penalty from the regularized least squares problem. This should be sized
        [number of frequency lines, number of regularization values]
    residual : ndarray
        The residual from the regularized least squares problem (typically the 
        mean squared error). This should be sized [number of frequency lines, number of regularization values]
    forces_full_path : ndarray
        The forces that were estimated in the regularized least squares problem.
        It should be sized such that the number of frequency lines is on the 
        first axis and the number of regularization values is on the second axis.
    l_curve_type : str
        The type of L-curve that is used to find the "optimal regularization 
        parameter. The available types are:
            - forces (default) - This L-curve is constructed with the "size" 
            of the forces on the Y-axis and the regularization parameter on the 
            X-axis. 
            - standard - This L-curve is constructed with the residual squared 
            error on the X-axis and the "size" of the forces on the Y-axis. 
    optimality_condition : str
        The method that is used to find an "optimal" regularization parameter.
        The options are:
            - curvature (default) - This method searches for the regularization
            parameter that results in maximum curvature of the L-curve. It is 
            also referred to as the L-curve criterion. 
            - distance - This method searches for the regularization parameter that
            minimizes the distance between the L-curve and a "virtual origin". A 
            virtual origin is used, because the L-curve is scaled and offset to always 
            range from zero to one, in this case.
    curvature_method : str
        The method that is used to compute the curvature of the L-curve, in the 
        case that the curvature is used to find the optimal regularization 
        parameter. The options are:
            - numerical (default) - this method computes the curvature of 
            the L-curve via numerical derivatives
            - cubic_spline - this method fits a cubic spline to the L-curve
            the computes the curvature from the cubic spline (this might 
            perform better if the L-curve isn't "smooth")      
    
    Raises
    ------
    ValueError
        If the requested L-curve type is not available.
    ValueError
        If the requested optimality condition is not available.
    
    Returns
    -------
    chosen_force : ndarray
        The force at the optimal regularization value, as defined by the L-curve. 
        It is sized [number of frequency lines, force array size]. 
    optimal_regularization : ndarray
        A vector of the optimal regularization values, as defined by the L-curve. 
        The length matches that number of frequency lines.   

    Notes
    -----
    This function can handle forces as either power spectra or linear spectra.

    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
        of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
        vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
        problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
        2000, pp. 119-142.  
    .. [3] M. Rezghi and S. Hosseini, "A new variant of L-curve for Tikhonov regularization,"
        Journal of Computational and Applied Mathematics, vol. 231, pp. 914-924, 2008.
    """
    if forces_full_path.ndim==4: #Forces are CPSDs
        chosen_force = np.zeros((forces_full_path.shape[0], forces_full_path.shape[-2], forces_full_path.shape[-1]), dtype=complex)
    elif forces_full_path.ndim==3: #Forces are linear spectra
        chosen_force = np.zeros((forces_full_path.shape[0], forces_full_path.shape[-1]), dtype=complex)
    else:
        raise ValueError('Cannot interpret the type of spectral quantity in the force')

    optimal_regularization = np.zeros(forces_full_path.shape[0], dtype=float)
    for ii in range(forces_full_path.shape[0]):
        if l_curve_type=='forces':
            if optimality_condition=='curvature':
                optimal_regularization[ii], idx = l_curve_criterion(regularization_values[ii, ...], penalty[ii, ...], regularization_values[ii, ...], method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii], idx = optimal_l_curve_by_distance(regularization_values[ii, ...], penalty[ii, ...], regularization_values[ii, ...])
            else:
                raise ValueError('The selected optimality_condition is not available')
        elif l_curve_type=='standard':
            if optimality_condition=='curvature':
                optimal_regularization[ii], idx = l_curve_criterion(residual[ii, ...], penalty[ii, ...], regularization_values[ii, ...], method = curvature_method)
            elif optimality_condition=='distance':
                optimal_regularization[ii], idx = optimal_l_curve_by_distance(residual[ii, ...], penalty[ii, ...], regularization_values[ii, ...])
            else:
                raise ValueError('The selected optimality_condition is not available')
        else:
            raise ValueError('The selected L-curve type is not available')
        chosen_force[ii, ...] = forces_full_path[ii, idx, ...]
    return chosen_force, optimal_regularization

def l_curve_criterion(x_axis, 
                      y_axis,
                      regularization_values,
                      method = 'numerical',
                      return_curvature = False):
    """
    Finds the "optimal" regularization value from an L-curve via the 
    location where its curvature is at a maximum (the L-curve criterion).

    Parameters
    ----------
    x_axis : ndarray
        This is a vector that defines the X-axis of the L-curve. The variable 
        that is used for this depends on the type of L-curve that is being 
        used.
    y_axis : ndarray
        This is a vector that defines the Y-axis of the L-curve. The variable 
        that is used for this depends on the type of L-curve that is being 
        used.
    regularization_values : ndarray
        This is a vector of regularization values that were used in the linear
        regression problem that created the L-curve. 
    method : str
        This is the method by which the curvature is computed, the 
        available methods are:
            - numerical (default) - this method computes the curvature of 
              the L-curve via numerical derivatives
            - cubic_spline - this method fits a cubic spline to the L-curve
              the computes the curvature from the cubic spline (this might 
              perform better if the L-curve isn't "smooth")

    Returns
    -------
    optimal_regularization : float
        This is the optimal regularization value based on the L-curve 
        criterion.
    idx : int
        The index that correspond to the optimal curvature.
    curvature : ndarray
        A vector of the curvature of the L-curve for the give sequence
        of regularization_values.
    
    References
    ----------
    .. [1] P.C. Hansen and D.P. O'Leary, "The Use of the L-Curve in the Regularization 
           of Discrete Ill-Posed Problems," SIAM Journal on Scientific Computing,
           vol. 14, no. 6, pp. 1487-1503, 1993.
    .. [2] P.C. Hansen, "The L-curve and its use in the numerical treatment of inverse
           problems," in Computational Inverse Problems in Electrocardiology," WIT Press, 
           2000, pp. 119-142.  
    """
    if np.all(x_axis==0) or np.all(y_axis==0):
        # This is done to handle the case where either the penalty or residual is all zero. 
        # Regularization is non-sensical in the case, so the minimum regularization value 
        # is selected.
        curvature = np.zeros(x_axis.shape, dtype=float)
        curvature[0] = 1
    else:
        # The variable names are cited match the literature
        if method == 'numerical':
            nu_prime = np.gradient(np.log(y_axis)) / np.gradient(np.log(regularization_values))
            nu_doubleprime = np.gradient(nu_prime) / np.gradient(np.log(regularization_values))

            rho_prime = np.gradient(np.log(x_axis)) / np.gradient(np.log(regularization_values))
            rho_doubleprime = np.gradient(rho_prime) / np.gradient(np.log(regularization_values))        
        elif method == 'cubic_spline':
            nu_cubic_spline = CubicSpline(np.log(regularization_values), np.log(y_axis))
            nu_prime = nu_cubic_spline.derivative(nu = 1)(np.log(regularization_values))
            nu_doubleprime = nu_cubic_spline.derivative(nu = 2)(np.log(regularization_values))

            rho_cubic_spline = CubicSpline(np.log(regularization_values), np.log(x_axis))
            rho_prime = rho_cubic_spline.derivative(nu = 1)(np.log(regularization_values))
            rho_doubleprime = rho_cubic_spline.derivative(nu = 2)(np.log(regularization_values))
        else:
            raise NameError('The selected L-curve criterion method is unavailable')

        curvature = (rho_prime*nu_doubleprime - nu_prime*rho_doubleprime) / np.power(nu_prime**2 + rho_doubleprime**2, 3/2)

    if return_curvature:
        return regularization_values[np.abs(curvature).argmax()], np.abs(curvature).argmax(), curvature
    else:
        return regularization_values[np.abs(curvature).argmax()], np.abs(curvature).argmax()

def optimal_l_curve_by_distance(x_axis, 
                                y_axis,
                                regularization_values):
    """
    Finds the "optimal" regularization from an L-curve by finding the parameter
    that puts the curve closest to the "virtual origin".

    Parameters
    ----------
    x_axis : ndarray
        This is a vector that defines the X-axis of the L-curve. The variable 
        that is used for this depends on the type of L-curve that is being 
        used.
    y_axis : ndarray
        This is a vector that defines the Y-axis of the L-curve. The variable 
        that is used for this depends on the type of L-curve that is being 
        used.
    regularization_values : ndarray
        This is a vector of regularization values that were used in the linear
        regression problem that created the L-curve. 

    Returns
    -------
    optimal_regularization : float
        This is the optimal regularization value based on the L-curve distance
        from the origin.
    idx : int
        The index that correspond to the optimal curvature.

    Notes
    -----
    This technique applies a scale and offset to the L-curve so the X and Y-axis
    always ranges from zero to one. This is required to obtain predictable 
    behavior from the method, but can also distort the shape of the curve.
    """
    if np.all(x_axis==0) or np.all(y_axis==0):
        # This is done to handle the case where either the penalty or residual is all zero. 
        # Regularization is non-sensical in the case, so the minimum regularization value 
        # is selected.
        rho = np.ones(x_axis.shape, dtype=float)
        rho[0] = 0
        nu = np.ones(y_axis.shape, dtype=float)
        nu[0] = 0
    else:
        rho = np.log(x_axis)
        rho = rho+np.sign(rho.min(axis = 0))*rho.min(axis = 0)
        rho = rho/rho.max(axis = 0)
        
        nu = np.log(y_axis)
        nu = nu+np.sign(nu.min(axis = 0))*nu.min(axis = 0)
        nu = nu/nu.max(axis = 0)

    return regularization_values[np.argmin(np.sqrt(nu**2 + rho**2), axis=0)], np.argmin(np.sqrt(nu**2 + rho**2), axis=0)

def compute_residual_penalty_for_l_curve(frfs, forces, response):
    """
    Computes the residual and penalty for an inverse force estimation problem 
    in preparation for model selection with the L-curve criterion. 

    Parameters
    ----------
    frfs : ndarray
        The FRFs that are used to compute the system response. Should be sized
        [number of lines, number of responses, number of references].
    forces : ndarray
        The estimated forces from the inverse problem. Should be sized
        [number of parameters, number of lines, number of references] for linear spectra
        or [number of parameters, number of lines, number of references, number of references]
        for power spectra
    response : ndarray
        The "truth" responses to compare the computed responses to. Should be
        sized [number of lines, number of responses] for linear spectra or 
        [number of lines, number of responses, number of responses] for power spectra.

    Returns
    -------
    residual : ndarray
        The residual error (i.e., sum squared error) between the computed and truth response. 
        It is sized [number of parameters, number of lines].
    penalty : ndarray
        The penalty to use in the L-curve parameter selection, which is the squared sum
        of the forces. It is sized [number of parameters, number of lines].

    Notes
    -----
    The returned residual and penalty are the squared norm of the residual error vector 
    and force vector, respectively.
    """
    # The loop is used to compute the residual to avoid memory inefficiency in a single 
    # line broadcasting calculation (where the intermediate matrices can get extremely large).
    if response.ndim == 2:
        residual = np.zeros((forces.shape[0], forces.shape[1]), dtype=float)
        for ii in range(forces.shape[0]):
            residual[ii,...] = norm(response - np.squeeze(frfs@forces[ii, ...][...,np.newaxis]), axis=-1, ord=2)**2
        penalty = norm(forces, axis=-1, ord=2)**2
    elif response.ndim == 3:
        residual = np.zeros((forces.shape[0], forces.shape[1]), dtype=float)
        for ii in range(forces.shape[0]):
            residual[ii,...] = norm(response - frfs@forces[ii, ...]@frfs.conj().transpose((0,2,1)), axis = (-2,-1), ord = 'fro')**2
        penalty = norm(forces, axis =(-2,-1), ord='fro')**2
    return residual, penalty

#%% obsolete auto-tikhonov functions

def _tikhonov_full_path(frf, response,
                       low_regularization_limit = None, 
                       high_regularization_limit = None,
                       number_regularization_values=100,
                       parallel = False,
                       num_jobs = -2):
    """
    Performs the inverse source estimation problem with Tikhonov regularization, 
    where the regularization parameter is automatically selected with L-curve 
    methods.

    Parameters
    ----------
    frf : ndarray
        The FRF data for the source estimation. It should be sized
        [number of lines, number of responses, number of references].
    response : ndarray
        The response data for the source estimation. This can either be 
        linear spectra or power spectrum and should be organized such that 
        the frequency lines are the first axis of the array. 
    low_regularization_limit : ndarray
        The low regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object. The default is the smallest singular
        value of the target frf array.
    high_regularization_limit : ndarray
        The high regularization limit to search through. This should be a 1d
        array with a length that matches the number of frequency lines in 
        the SourcePathReceiver object.
    number_regularization_values : int
        The number of regularization parameters to search over, where the 
        potential parameters are geometrically spaced between the low and high
        regularization limits.  
    l_curve_type : str
        The type of L-curve that is used to find the "optimal regularization 
        parameter. The available types are:
            - forces (default) - This L-curve is constructed with the "size" 
            of the forces on the Y-axis and the regularization parameter on the 
            X-axis. 
            - standard - This L-curve is constructed with the residual squared 
            error on the X-axis and the "size" of the forces on the Y-axis. 
    optimality_condition : str
        The method that is used to find an "optimal" regularization parameter.
        The options are:
            - curvature (default) - This method searches for the regularization
            parameter that results in maximum curvature of the L-curve. It is 
            also referred to as the L-curve criterion. 
            - distance - This method searches for the regularization parameter that
            minimizes the distance between the L-curve and a "virtual origin". A 
            virtual origin is used, because the L-curve is scaled and offset to always 
            range from zero to one, in this case.
    curvature_method : std
        The method that is used to compute the curvature of the L-curve, in the 
        case that the curvature is used to find the optimal regularization 
        parameter. The options are:
            - numerical (default) - this method computes the curvature of 
            the L-curve via numerical derivatives
            - cubic_spline - this method fits a cubic spline to the L-curve
            the computes the curvature from the cubic spline (this might 
            perform better if the L-curve isn't "smooth")
    use_transformation : bool
        Whether or not the response and reference transformation from the class 
        definition should be used (which is handled in the "linear_inverse_processing" 
        decorator function). The default is true. 
    response : ndarray
        The preprocessed response data. The preprocessing is handled by the decorator 
        function and object definition. This argument should not be supplied by the 
        user. 
    
    parallel : bool
        Whether or not to parallelize the computation using Joblib. The default is 
        False. 
    num_jobs : int
        The number of processors to use when parallelizing the code. The default is 
        -2, which uses all the available processors except one. Refer to the joblib
        documentation for more details. 

    Returns
    -------
    forces_full_path : ndarray
        The estimated forces over all frequencies and regularization parameters. It
        is sized [number of frequencies, number of regularization values, force array size], 
        where the force array could be a matrix depending on if the responses are linear
        spectra or CPSDs.
    regularization_values : ndarray
        The regularization values that were used in the force estimation. This variable
        is size [number of frequencies, number of regularization values].
    residual : ndarray
        The least squares residual (mean squared error) for all the estimated forces.
        This variable is sized [number of frequencies, number of regularization values].
    penalty : ndarray
        The square of the 2-norm of the forces (the frobenius norm is used when the responses
        are CPSDs).This variable is sized [number of frequencies, number of regularization values]. 

    Notes
    -----
    Parallelizing generally isn't faster for "small" inverse problems because of the 
    overhead involved in the parallelizing. Some experience has shown that the 
    parallelization adds ~1-1.5 minutes to the computation, but this will depend on 
    the specific computer that is being used.
    """
    if np.any(low_regularization_limit) or np.any(high_regularization_limit) is None:
        s = np.linalg.svd(frf, compute_uv=False)
    if np.any(low_regularization_limit) is None:
        low_regularization_limit = s[:, -1]
    if np.any(high_regularization_limit) is None:
        high_regularization_limit = s[:, 0]

    if parallel==True:
        tasks = [delayed(_tikhonov_full_path_single_frequency)(frf[ii, ...], 
                                                              response[ii, ...], 
                                                              low_regularization_limit=low_regularization_limit[ii], 
                                                              high_regularization_limit=high_regularization_limit[ii],
                                                              number_regularization_values=int(number_regularization_values)) for ii in range(frf.shape[0])]
        results = Parallel(n_jobs=int(num_jobs), prefer='threads')(tasks)
        forces_full_path, regularization_values, residual, penalty = zip(*results)
        # Making sure that the unpacked results are ndarrays
        forces_full_path = np.array(forces_full_path)
        regularization_values = np.array(regularization_values)
        residual = np.array(residual)
        penalty = np.array(penalty)
    elif parallel==False:
        if response.ndim==3: #Responses are CPSDs
            forces_full_path = np.zeros((frf.shape[0], number_regularization_values, frf.shape[2], frf.shape[2]), dtype=complex)
        else: # Responses are Linear Spectra
            forces_full_path = np.zeros((frf.shape[0], number_regularization_values, frf.shape[2]), dtype=complex)
        regularization_values = np.zeros((frf.shape[0], number_regularization_values), dtype=float) 
        residual = np.zeros((frf.shape[0], number_regularization_values), dtype=float) 
        penalty = np.zeros((frf.shape[0], number_regularization_values), dtype=float)
        for ii in range(frf.shape[0]):
            forces_full_path[ii, ...], regularization_values[ii, ...], residual[ii, ...], penalty[ii, ...] = _tikhonov_full_path_single_frequency(frf[ii, ...], 
                                                                                                                                                 response[ii, ...], 
                                                                                                                                                 low_regularization_limit=low_regularization_limit[ii], 
                                                                                                                                                 high_regularization_limit=high_regularization_limit[ii],
                                                                                                                                                 number_regularization_values=int(number_regularization_values))
    return forces_full_path, regularization_values, residual, penalty

def _tikhonov_full_path_single_frequency(H, x,  
                                        low_regularization_limit = None, 
                                        high_regularization_limit = None,
                                        number_regularization_values = 100):
    """
    Computes the Tikhonov "regularization path" between the high and low
    regularization parameters for the inverse source estimation problem at 
    a single frequency line. 

    Parameters
    ----------
    H : ndarray
        The FRF matrix for the inverse problem.
    x : ndarray
        The response for the inverse problem. A vector should be supplied 
        for a linear spectrum and a matrix should be supplied for a power 
        spectrum. Note that the function behaves slightly differently if a 
        vector or matrix is supplied for the response. 
    low_regularization_limit : float
        This is the smallest regularization value to be used in the 
        regression problem. The default is the smallest singular
        value of the FRF matrix. 
    high_regularization_limit : float
        This is the highest regularization value to be used in the 
        regression problem. The default is the largest singular value
        of the FRF matrix.
    number_regularization_values : int
        This is the number of values to put evaluate the L-curve
        over. The default is 100. 

    Returns
    -------
    forces : ndarray
        A matrix of forces found using the different regularization 
        values. Organized by [number of regularization values x number of forces]
    lambda_values : ndarray
        The sequence of regularization values used in developing the L-Curve
    residual : ndarray
        The residual error from the least squares problem for the different 
        regularization values. This is the square of the 2-norm of the error
        vector in the linear spectrum case or the square of the Frobenius norm 
        in the power spectrum case.
    penalty : np.ndarray
        The regularization penalty for the different regularization values. The 
        penalty is defined by the square of the 2-norm of the force vector in the 
        linear spectrum case or the square of the Frobenius norm in the power 
        spectrum case. 

    Notes
    -----
    The regularization values are spread over a geometric space that spans
    the low and high regularization limits. 

    This function works for situations where the response is either a linear 
    spectrum or a power spectrum. The difference is sensed based on the shape 
    of the response array (one dimension or two dimensions)
    """    
    lambda_values = np.geomspace(low_regularization_limit, high_regularization_limit, num = number_regularization_values)

    if x.ndim == 1:
        f = np.zeros([lambda_values.size, H.shape[1]], dtype=complex)
    elif x.ndim == 2:
        f = np.zeros([lambda_values.size, H.shape[1], H.shape[1]], dtype=complex)

    for ii, l in enumerate(lambda_values):
        H_pinv = pinv_by_tikhonov(H, regularization_parameter=l)
        if x.ndim == 1:
            f[ii, ...] = H_pinv@x
        if x.ndim == 2:
            f[ii, ...] = H_pinv@x@np.moveaxis(H_pinv.conj(), -1, -2)

    if x.ndim == 1:
        residual = norm(x - np.squeeze(H[np.newaxis, ...]@f[..., np.newaxis]), axis = -1, ord = 2)**2
        penalty = norm(f, axis = -1, ord = 2)**2
    elif x.ndim == 2:
        residual = norm(x - H[np.newaxis, ...]@f@np.moveaxis(H.conj(), -1, -2)[np.newaxis, ...], axis = (-2, -1), ord = 'fro')**2
        penalty = norm(f, axis = (-2, -1), ord = 'fro')**2

    return f, lambda_values, residual, penalty