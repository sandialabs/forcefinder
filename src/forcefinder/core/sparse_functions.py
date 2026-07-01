"""
Contains helper functions for sparse force estimation (not using all the 
available force DOFs).

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
from sdynpy.core.sdynpy_coordinate import coordinate_array
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

def forward_stepwise_force_dof_evaluation(spr_object, evaluation_function, num_forces=None, num_jobs=None, 
                                          use_transformation=False, **evaluation_kwargs):
    """
    Evaluates set of force DOFs for a given response through forward stepwise 
    selection algorithm where the cost function is defined by the user. 

    Parameters
    ----------
    spr_object : SourcePathReceiver
        A source path receiver that has the FRFs (at all possible force DOFs) and 
        training response.
    evaluation_function : callable
        A function to evaluate the a set of force DOFs with. This function should 
        take the FRF and response ordinate (with the frequency vector on the first
        axis), estimate the forces, then return some cost function that can be 
        used to compare one set of force DOFs against another. This function can 
        return multiple parameters, but the first parameter must be the cost 
        function that is used to compare the DOF sets. 
    num_forces : int, optional
        The maximum number of forces to include in a DOF set. I.E., this is the 
        maximum number of times through the stepwise algorithm. For example, if this
        is set to six, the stepwise algorithm will be cycled through six times, where
        a new "optimal" force DOF is added to the output at each step. The default 
        behavior is to evaluate all the DOFs, which returns the the force DOFs 
        (all the references in the FRFs) in order of the DOFs that minimized the 
        cost function the most. 
    num_jobs : int, optional
        The number of jobs to dispatch for the multi-processing.
    use_transformation : bool, optional
        Whether or not to use the transformations in the algorithm. 
    
    Returns
    -------
    ordered_force_dofs : sdpy.CoordinateArray
        The force DOFs (from the FRF matrix) that are ordered from the DOF that 
        minimized the cost function the most to the DOF that minimize the cost function 
        the least.
    tracking_error : ndarray
        A 1d array of the cost function value for a given force DOF set, as defined
        in the `evaluation_function`.
    extra_result : tuple
        Any extra return variables from the evaluation function. 

    Raises
    ------
    NotImplementedError
        If there is a non-identity reference transformation in the SPR object.

    Notes
    -----
    This function is a wrapper around the _forward_stepwise_force_evaluation_ that
    extracts the necessary data from the supplied SPR object. 

    The returned `ordered_force_dofs` should be viewed as the "optimal" set of 
    forces DOFs, depending on the desired number of forces. I.E., `ordered_force_dofs[:2]`
    is the optimal set of two force DOFs for the given force DOFs and training responses.
    The `tracking_error` should also be interpreted as the error for a set. I.E., 
    `tracking_error[1]` is the error that corresponds to dof set `ordered_force_dofs[:2]`.

    This function uses a greedy optimization approach, which means that the supplied
    force DOF sets are "approximately" optimal.
    """


    if use_transformation:
        # Checking that there is not a reference transformation
        num_reference_coord = spr_object._reference_coordinate_.shape[0]
        if spr_object._reference_transformation_array_.shape != (num_reference_coord, num_reference_coord):
            raise NotImplementedError('The greedy force DOF selection does not currently work with SPR objects that have non-identity reference transformations')
        else:
            if not np.all(spr_object._reference_transformation_array_ == np.eye(num_reference_coord)):
                raise NotImplementedError('The greedy force DOF selection does not currently work with SPR objects that have non-identity reference transformations')

        frf_ord = spr_object.transformed_training_frfs.ordinate.transpose(2,0,1)
        res_ord = spr_object.transformed_training_response.ordinate.transpose(2,0,1)
    elif not use_transformation:
        frf_ord = spr_object._training_frf_array_
        res_ord = spr_object._training_response_array_
    
    force_dof_index, tracking_error, *extra_result = _forward_stepwise_force_evaluation_(frf_ord, res_ord, evaluation_function, 
                                                                                  num_forces, num_jobs, **evaluation_kwargs)
    ordered_force_dofs = spr_object._reference_coordinate_[force_dof_index]

    if any(extra_result): # returning extra_result if something is in it
        return ordered_force_dofs, tracking_error, extra_result
    else:
        return ordered_force_dofs, tracking_error

def _forward_stepwise_force_evaluation_(frf_ord, res_ord, evaluation_function, num_forces=None, 
                                        num_jobs=-2, **evaluation_kwargs):
    """
    Evaluates set of force DOFs for a given response through forward stepwise 
    selection algorithm where the cost function is defined by the user. 

    Parameters
    ----------
    frf_ord : ndarray
        An ndarray of FRFs to run through the forward stepwise algorithm. The 
        FRFs should be sized [number of lines, number of responses, number of forces]
    res_ord : ndarray
        An ndarray of responses use in the forward stepwise algorithm. The 
        response can be either CPSDs or linear spectra. CPSDs should be sized 
        [number of lines, number of responses, number of forces] and linear spectra
        should be sized [number of lines, number of responses].
    evaluation_function : callable
        A function to evaluate the a set of force DOFs with. This function should 
        take the FRF and response ordinate (with the frequency vector on the first
        axis), estimate the forces, then return some cost function that can be 
        used to compare one set of force DOFs against another. This function can 
        return multiple parameters, but the first parameter must be the cost 
        function that is used to compare the DOF sets. 
    num_forces : int, optional
        The maximum number of forces to include in a DOF set. I.E., this is the 
        maximum number of times through the stepwise algorithm. For example, if this
        is set to six, the stepwise algorithm will be cycled through six times, where
        a new "optimal" force DOF is added to the output at each step. The default 
        behavior is to evaluate all the DOFs, which returns the the force DOFs 
        (all the references in the FRFs) in order of the DOFs that minimized the 
        cost function the most. 
    num_jobs : int, optional
        The number of jobs to dispatch for the multi-processing, which is used with
        the Joblib Parallel function. The default is -2.
    
    Returns
    -------
    best_ind : ndarray
        The indices that correspond to the column in the FRF matrix that minimized
        the cost function the most to the column that minimized the cost function 
        the least. 
    tracking_error : ndarray
        A 1d array of the cost function value for a given force DOF set, as defined
        in the `evaluation_function`.
    extra_result : tuple
        Any extra return variables from the evaluation function. 

    Notes
    -----
    The returned `best_ind` should be viewed as the indices that correspond to the
    "optimal" set of forces DOFs, depending on the desired number of forces. 
    I.E., `best_ind[:2]` corresponds to the optimal set of two force DOFs for the 
    given FRFs and training responses. The `tracking_error` should also be interpreted 
    as the error for a set. I.E., `tracking_error[1]` is the error that corresponds to 
    dof set that corresponds to `best_ind[:2]`.

    This function uses a greedy optimization approach, which means that the supplied
    force DOF sets are "approximately" optimal.
    """
    best_ind = np.array([], dtype=np.int64)
    if num_forces is None:
        num_forces = frf_ord.shape[-1]

    tracking_error = np.zeros(num_forces, dtype=np.float64)
    for jj in range(num_forces):
        outer_index = np.array([ind for ind in np.arange(frf_ord.shape[-1]) if ind not in best_ind])
        evaluation_result = Parallel(n_jobs=num_jobs)(delayed(evaluation_function)(res_ord=res_ord, 
                    frf_ord=frf_ord[:, :, np.append(best_ind, index)], **evaluation_kwargs) for index in outer_index)
        try: 
            error, *extra_result = zip(*evaluation_result)
        except TypeError:
            extra_result = []
            error = evaluation_result 
        best_ind = np.append(best_ind, outer_index[np.argmin(error)])
        tracking_error[jj] = np.min(error)
    try:
        return best_ind, tracking_error, extra_result
    except ValueError:
        return best_ind, tracking_error