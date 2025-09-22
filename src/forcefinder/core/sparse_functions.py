"""
Contains helper functions for estimating forces with the elastic net.

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
from joblib import Parallel, delayed

def elastic_net_full_path_all_frequencies_parallel(H, x, 
                                                   alpha, 
                                                   number_of_lambdas = 100,
                                                   max_number_of_iterations = 1e6,
                                                   break_tolerance = 1e-4):
    """
    Computes the elastic net forces via coordinate descent.

    Parameters
    ----------
    H : ndarray
        FRF matrix for a single frequency line. This must be a converted version
        of the complex valued matrix to a ring of real valued matrices (real 
        values in the top left and and bottom right corners, negative of the 
        imaginary values cast to real in the top right corner, and positive 
        imaginary values cast to real in the bottom left corner).
    x : ndarray
        Vector of (linear) responses for a single frequency line. This must be
        converted from a complex valued array to a vectorized real valued array
        that matches the FRF matrix (real values in the "top" half and imaginary
        values cast to real in the "bottom" half).
    alpha : float
        Alpha parameter for the elastic net. This controls the balance between the
        L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
        than 0 and less than 1. 
    number_of_lambdas : int   
        This parameter is supplied if the lambda_values are being determined by
        the code. The default is 100. 
    max_number_of_iterations : int
        This is the maximum number of iterations (cycles) in the coordinate descent
        optimization. The default is 1e5.
    break_tolerance : float
        This is the break tolerance for the coordinate descent optimization. The 
        default is 1e-5.

    Returns
    -------
    forces_full_path : ndarray
        Matrix of estimated forces for all the lambdas, organized 
        [force dof x lambda value]. 
    lambda_values : ndarray
        This is the regularization parameters that are used in the elastic net
        regularization.

    Notes
    -----
    This is designed to be used with a SourcePathReceiver object and does not do any 
    book keeping.

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani, M. Wainright, Statistical Learning with Sparsity:
           The Lasso with Generalizations. Boca Raton, Fl: CRC Press, 2015. 
    .. [2] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
           Linear Models via Coordinate Descent, Journal of Statistical Software, 
           Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01. 
    """
    x_ring = np.concatenate((np.real(x), np.imag(x)), axis=1)
    H_ring = np.block([[np.real(H), -np.imag(H)],
                       [np.imag(H), np.real(H)]])
    tasks = [delayed(elastic_net_full_path)(H_ring[ii, ...],
                                            x_ring[ii, ...], 
                                            alpha, 
                                            number_of_lambdas = number_of_lambdas,
                                            max_number_of_iterations = max_number_of_iterations,
                                            break_tolerance = break_tolerance) for ii in range(H_ring.shape[0])]
    results = Parallel(n_jobs=-2)(tasks)
    forces_ring, lambda_values = zip(*results)
    forces_ring = np.array(forces_ring)
    forces_full_path = forces_ring[:, :forces_ring.shape[1]//2, :] + forces_ring[:, forces_ring.shape[1]//2:, :]*1j
    return forces_full_path, np.array(lambda_values)

def elastic_net_full_path(H, x, 
                          alpha, 
                          number_of_lambdas = 100,
                          max_number_of_iterations = 1e6,
                          break_tolerance = 1e-4):
    """
    Computes the full regularization path for a single frequency line via the 
    elastic net.

    Parameters
    ----------
    H : ndarray
        FRF matrix for a single frequency line. This must be a converted version
        of the complex valued matrix to a ring of real valued matrices (real 
        values in the top left and and bottom right corners, negative of the 
        imaginary values cast to real in the top right corner, and positive 
        imaginary values cast to real in the bottom left corner).
    x : ndarray
        Vector of (linear) responses for a single frequency line. This must be
        converted from a complex valued array to a vectorized real valued array
        that matches the FRF matrix (real values in the "top" half and imaginary
        values cast to real in the "bottom" half).
    alpha : float
        Alpha parameter for the elastic net. This controls the balance between the
        L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
        than 0 and less than 1. 
    number_of_lambdas : int   
        This parameter is supplied if the lambda_values are being determined by
        the code. The default is 100. 
    max_number_of_iterations : int
        This is the maximum number of iterations (cycles) in the coordinate descent
        optimization. The default is 1e5.
    break_tolerance : float
        This is the break tolerance for the coordinate descent optimization. The 
        default is 1e-5.

    Returns
    -------
    forces_ring : ndarray
        Matrix of estimated forces for all the lambdas, organized 
        [force dof x lambda value]. It is a real valued array that is matches 
        the frfs and responses, where the real values are in the "top" half
        and the complex values are in the "bottom" half. 
    lambda_values : ndarray
        This is the regularization parameters that are used in the elastic net
        regularization.

    Note
    ----
    This compiles all of the deprecated elastic net functions into one large 
    function so it works well with the numba JIT compiler. 

    References
    ----------
    .. [1] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
           Linear Models via Coordinate Descent, Journal of Statistical Software, 
           Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01. 
    """
    # Have to make the lambdas here because numba doesn't seem to like nested functions
    lambda_max = np.max(np.abs(H.T@x))/(H.shape[0]*alpha)
    lambda_start = lambda_max/(number_of_lambdas*100)
    lambda_values = np.zeros(number_of_lambdas, dtype=np.float64)
    lambda_values[1:] = np.exp(np.linspace(start = np.log(lambda_start), stop = np.log(lambda_max), num = number_of_lambdas-1))
    
    # computing the full path of the forces
    forces_ring = np.zeros((H.shape[1], number_of_lambdas), dtype = np.float64)
    forces_ring[:, 0] = np.linalg.pinv(H)@x
    current_estimate = forces_ring[:, 0].copy()
    for kk, l in enumerate(lambda_values[1:]):
        gamma = l*alpha
        for ii in range(int(max_number_of_iterations)):
            previous_estimate = current_estimate.copy()
            for jj in range(H.shape[1]):
                partial_residual = x - H@current_estimate + H[:, jj]*current_estimate[jj]
                naive_update = np.dot(partial_residual, H[:, jj])/H.shape[0]
                # Soft threshold
                if np.abs(naive_update) <= gamma:
                    naive_update = 0
                else: 
                    naive_update = np.sign(naive_update)*(np.abs(naive_update)-gamma)
                # Shrinkage
                shrink_factor = (1/H.shape[0]) * np.dot(H[:, jj], H[:, jj]) + (1-alpha)*l
                current_estimate[jj] = naive_update / shrink_factor
            # Termination Tolerances
            iteration_change = (current_estimate - previous_estimate) / (1 + np.abs(previous_estimate))
            if np.linalg.norm(iteration_change, ord = np.inf) < break_tolerance:
                break
            if ii == int(max_number_of_iterations):
                print('The optimizer hit the maximum number of iterations and did not converge')
        forces_ring[:, kk+1] = current_estimate
    return forces_ring, lambda_values