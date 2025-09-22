# These are scripts for solving the sparse inverse problem

import numpy as np
import sdynpy as sdpy
from scipy.linalg import norm
#from numba import jit
from joblib import Parallel, delayed
import warnings

def force_estimation_elastic_net(H, x, 
                                 alpha,
                                 lambda_values = None,
                                 number_of_lambdas = 100,
                                 max_number_of_iterations = 1e6, 
                                 break_tolerance = 1e-4):
    """
    Estimates forces using elastic net penalized linear regression using
    a cyclical coordinate descent optimization algorithm. 

    Parameters
    ----------
    H : TransferFunctionArray
        The FRFs being used in the elastic net linear regression.
    x : SpectrumArray
        The responses being used in the elastic net linear regression.
    alpha : float
        Alpha parameter for the elastic net. This controls the balance between the
        L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
        than 0 and less than 1. 
    lambda_values : ndarray
        The option for the user to to supply their own regularization parameters is 
        not implemented yet.
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
    forces : SpectrumArray
        The estimated forces from the elastic net linear regression for all values. 
        The ordinate (and corresponding spectrum array) is sized
        [number of lambdas x number of forces x number of lines].
    lambda_values : ndarray
        The lambda values that were used in the regression problem. It is sized
        [number of lines x number of lambdas]. 
    
    Notes
    -----
    All of the load estimation is done in a single large function because it was 
    required to force compatibility with numba. 

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani, M. Wainright, Statistical Learning with Sparsity:
           The Lasso with Generalizations. Boca Raton, Fl: CRC Press, 2015. 
    .. [2] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
           Linear Models via Coordinate Descent, Journal of Statistical Software, 
           Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01. 
    """
    if lambda_values is not None:
        raise NotImplementedError('The option for a user to supply lambda values is not implemented yet')

    H_ord_ring, x_ord_ring, reference_dofs = organize_data_for_lasso_optimization(H, x)

    # Looping over the elastic net optimization for all frequencies
    forces_full_path_ring, lambda_values = elastic_net_full_path_all_frequencies_parallel(H_ord_ring, x_ord_ring, alpha, 
                                                                                          number_of_lambdas = number_of_lambdas,
                                                                                          max_number_of_iterations = max_number_of_iterations,
                                                                                          break_tolerance = break_tolerance)

    forces_full_path = forces_full_path_ring[:, :forces_full_path_ring.shape[1]//2, :] + forces_full_path_ring[:, forces_full_path_ring.shape[1]//2:, :]*1j

    forces = sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, H.flatten()[0].abscissa, forces_full_path, reference_dofs)

    return forces, lambda_values
    
def compute_model_information_criterion(H, x, f, method):
    """
    Computes the desired information criterion for the elastic net / LASSO
    model. 

    Parameters
    ----------
    H : TransferFunctionArray
        The FRFs that were used to estimate the forces.
    x : SpectrumArray
        The responses that were used in the force estimation.
    f : SpectrumArray
        The estimated forces from the elastic net / LASSO model.
    method : str
        The desired information criterion, the available options are:
            - 'BIC': the Bayesian information criterion
            - 'AIC': the Akaike information criterion
            - 'AICC': the corrected Akaike information criterion
    
    Returns
    -------
    criterion : ndarray
        The selected information criterion that is organized 
        [number of lines, number of regularization levels]
    """
    H = H.reshape_to_matrix()
    H = H[sdpy.coordinate.outer_product(H[:, 0].response_coordinate, f[0, :].response_coordinate)]
    x = x[(H[:,0].response_coordinate)[..., np.newaxis]]
    n = x.ordinate.shape[0]
    p = f.ordinate.shape[1]
    criterion = np.zeros((f.ordinate.shape[-1], f.ordinate.shape[0]), dtype = np.float64)

    for ii in range(f.ordinate.shape[-1]):
        residual = norm((x.ordinate[..., ii])[..., np.newaxis] - H.ordinate[..., ii]@np.moveaxis(f.ordinate[..., ii], 0, -1), axis = 0, ord = 2)**2
        noise_variance = residual/(n-p)
        number_active_parameters = np.count_nonzero(np.abs(f.ordinate[..., ii]), axis = 1)
        
        if method == 'BIC':
            criterion_factor = np.log(n)
            criterion_corrector = 0
        elif method == 'AIC':
            criterion_factor = 2
            criterion_corrector = 0
        elif method == 'AICC':
            criterion_factor = 2
            criterion_corrector = (2*number_active_parameters**2 + 2*number_active_parameters)/(n-number_active_parameters-1)
        criterion[ii, :] = n*np.log(2*np.pi*noise_variance) + residual/noise_variance + criterion_factor * number_active_parameters + criterion_corrector

    return criterion

def organize_data_for_inverse_solution(H, x):
    """
    Does the basic organization on the FRF matrix and response vector 
    for a linear inverse solution.

    Parameters
    ----------
    H : TransferFunctionArray
        The FRFs being used in the linear inverse solution.
    x : SpectrumArray or PowerSpectralDensityArray
        The responses being used in the linear inverse solution.
    
    Returns
    -------
    H : TransferFunctionArray
        The FRFs after being reshaped to ensure they are in a matrix
        (not flattened) format.
    x : SpectrumArray or PowerSpectralDensityArray
        The responses after being reorganized to match the response
        DOF ordering in the FRF matrix.

    Raises
    ------
    ValueError
        If the response and FRF abscissa do not match.
    ValueError
        If all the abscissa in the response array are not the same.
    """
    # Ensuring the abscissa matches for both the FRFs and responses
    abscissa = x.flatten()[0].abscissa
    if not np.allclose(abscissa, H.abscissa):
        raise ValueError('Transfer Function Abscissa do not match responses')
    if not np.allclose(abscissa, x.abscissa):
        raise ValueError('All response abscissa must be identical')
    
    # Organizing the FRFs and responses to ensure the response coordinates are the same
    H = H.reshape_to_matrix()
    if isinstance(x, sdpy.core.sdynpy_data.SpectrumArray):
        response_dofs = (H[:,0].response_coordinate)[..., np.newaxis]
        x = x[response_dofs]
    elif isinstance(x, sdpy.core.sdynpy_data.PowerSpectralDensityArray):
        response_dofs = sdpy.coordinate.outer_product(H[:,0].response_coordinate, H[:,0].response_coordinate)
        x = x[response_dofs]
    return H, x


def organize_data_for_lasso_optimization(H, x):
    """
    Organizes the data for a LASSO optimization. This reshapes 
    the variables into the correct shape, reorders the DOFs, 
    and organizes the complex valued matrices into "rings" of 
    real valued matrices. 

    Parameters
    ----------
    H : TransferFunctionArray
        The FRFs to be used in the LASSO penalized linear regression.
    x : SpectrumArray
        The responses to be used in the LASSO penalized linear regression.
    
    Returns
    -------
    H_ord_ring : ndarray
        The FRFs after being organized into a ring of real valued arrays.
        The real part of the FRFs are on the block diagonal. The negative
        of the imaginary part of the FRFs is in the upper right corner. The 
        unchanged imaginary part is in the lower left corner. 
    x_ord_ring : ndarray
        The responses after being organized into a ring of real valued
        arrays. The real part is in the "top half" of the vector and and 
        imaginary part is in the "bottom half". 
    reference_dofs : CoordinateArray
        A coordinate array of the references for labeling the forces at 
        the end of the process. 
    """
    # Getting the dof ordering correct
    H, x = organize_data_for_inverse_solution(H, x)
    reference_dofs = (H[0,:].reference_coordinate)[..., np.newaxis]
    x_ord = x.ordinate
    H_ord = H.ordinate

    # Organizing the data into "rings" 
    x_ord_ring = np.concatenate((np.real(x_ord), np.imag(x_ord)))
    H_ord = np.moveaxis(H_ord, -1, 0) # This is done to get the block function to work correctly
    H_ord_ring = np.block([[np.real(H_ord), -np.imag(H_ord)],
                           [np.imag(H_ord), np.real(H_ord)]])
    H_ord_ring = np.moveaxis(H_ord_ring, 0, -1) # Getting the FRFs shaped correctly 
    return H_ord_ring, x_ord_ring, reference_dofs

def elastic_net_full_path_all_frequencies_parallel(H_ring, x_ring, 
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
    #x_ring = np.concatenate((np.real(x), np.imag(x)), axis=1)
    #H_ring = np.block([[np.real(H), -np.imag(H)],
    #                [np.imag(H), np.real(H)]])
    results = Parallel(n_jobs=-2)(delayed(elastic_net_full_path)(H_ring[ii, ...],
                                                                 x_ring[ii, ...], 
                                                                 alpha, 
                                                                 number_of_lambdas = number_of_lambdas,
                                                                 max_number_of_iterations = max_number_of_iterations,
                                                                 break_tolerance = break_tolerance) for ii in range(H_ring.shape[0]))
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

def force_dof_selection_by_greedy(frf, response, evaluation_function, num_forces=None, num_jobs=None, response_normalization=False, **kwargs):

    if response_normalization:
        res_std = np.std(np.moveaxis(frf.ordinate, -1, 0), axis=-1)
        response_transform = np.broadcast_to(np.eye(res_std.shape[-1])[np.newaxis, ...], (res_std.shape[-2], res_std.shape[-1], res_std.shape[-1]))/res_std[..., np.newaxis]
        frf_ord = response_transform@np.moveaxis(frf.ordinate, -1, 0)
        if isinstance(response, sdpy.data.SpectrumArray):
            res_ord = response_transform@np.moveaxis(response.ordinate[:, np.newaxis, :], -1, 0)    
        elif isinstance(response, sdpy.data.PowerSpectralDensityArray):
            res_ord = response_transform@np.moveaxis(response.ordinate, -1, 0)@np.transpose(response_transform, (0, 2, 1))
    else:
        frf_ord = np.moveaxis(frf.ordinate, -1, 0)
        if isinstance(response, sdpy.data.SpectrumArray):
            res_ord = np.moveaxis(response.ordinate, -1, 0)[..., np.newaxis]    
        elif isinstance(response, sdpy.data.PowerSpectralDensityArray):
            res_ord = np.moveaxis(response.ordinate, -1, 0)

    force_dof_index, tracking_error, *extra_result = greedy_force_selection(frf_ord, res_ord, evaluation_function, num_forces, num_jobs, **kwargs)

    if any(extra_result): # returning extra_result if something is in it
        return sdpy.coordinate_array(string_array=frf[0, :].reference_coordinate.string_array()[force_dof_index]), tracking_error, extra_result
    else:
        return sdpy.coordinate_array(string_array=frf[0, :].reference_coordinate.string_array()[force_dof_index]), tracking_error

def greedy_force_selection(frf_ord, res_ord, evaluation_function, num_forces=None, num_jobs=None, **kwargs):
    best_ind = np.array([], dtype=np.int64)
    if num_forces is None:
        num_forces = frf_ord.shape[-1]

    if num_jobs is None:
        num_jobs = -2

    tracking_error_rms = np.zeros(num_forces, dtype=np.float64)
    for jj in range(num_forces):
        outer_index = np.array([ind for ind in np.arange(frf_ord.shape[-1]) if ind not in best_ind])
        evaluation_result = Parallel(n_jobs=num_jobs)(delayed(evaluation_function)(res_ord=res_ord, frf_ord=frf_ord[:, :, np.append(best_ind, index)], **kwargs) for index in outer_index)
        try: 
            error_rms, *extra_result = zip(*evaluation_result)
        except TypeError:
            extra_result = []
            error_rms = evaluation_result 
        best_ind = np.append(best_ind, outer_index[np.argmin(error_rms)])
        tracking_error_rms[jj] = np.min(error_rms)
    try:
        return best_ind, tracking_error_rms, extra_result
    except ValueError:
        return best_ind, tracking_error_rms

""" deprecated function in favor of the parallel elastic net function
#@jit(nopython = True)
def elastic_net_full_path_all_frequencies(H_ring, x_ring, 
                                          alpha, 
                                          number_of_lambdas = 100,
                                          max_number_of_iterations = 1e6,
                                          break_tolerance = 1e-4):
    
    Computes the full elastic net regularization path for all frequencies in
    a inverse source estimation problem. 

    Parameters
    ----------
    H_ring : ndarray
        FRF matrix organized into a 3D matrix. This must be a converted version
        of the complex valued matrix to a ring of real valued matrices (real 
        values in the top left and and bottom right corners, negative of the 
        imaginary values cast to real in the top right corner, and positive 
        imaginary values cast to real in the bottom left corner). It should be
        sized: [references, responses, frequencies].
    x_ring : ndarray
        Vector of (linear) responses for a single frequency line. This must be
        converted from a complex valued array to a vectorized real valued array
        that matches the FRF matrix (real values in the "top" half and imaginary
        values cast to real in the "bottom" half). It should be sized: [response, frequencies]
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
    
    forces_full_path_ring = np.zeros((number_of_lambdas, H_ring.shape[1], H_ring.shape[-1]), dtype = np.float64)
    lambda_values = np.zeros((H_ring.shape[-1], number_of_lambdas), dtype = np.float64)
    for ff in range(x_ring.shape[-1]):
        # Have to make the lambdas here because numba doesn't seem to like nested functions
        lambda_max = np.max(np.abs(H_ring[..., ff].T@x_ring[..., ff]))/(H_ring[..., ff].shape[0]*alpha)
        lambda_start = lambda_max/(number_of_lambdas*100)
        lambda_values[ff, 1:] = np.exp(np.linspace(start = np.log(lambda_start), stop = np.log(lambda_max), num = number_of_lambdas-1))
        # computing the full path of the forces
        forces_full_path_ring[0, :, ff] = np.linalg.pinv(H_ring[..., ff])@x_ring[..., ff]
        current_estimate = forces_full_path_ring[0, :, ff].copy()
        for kk, l in enumerate(lambda_values[ff, 1:]):
            gamma = l*alpha
            for ii in range(int(max_number_of_iterations)):
                previous_estimate = current_estimate.copy()
                for jj in range(H_ring[..., ff].shape[1]):
                    partial_residual = x_ring[..., ff] - H_ring[..., ff]@current_estimate + H_ring[:, jj, ff]*current_estimate[jj]
                    naive_update = np.dot(partial_residual, H_ring[:, jj, ff])/H_ring[..., ff].shape[0]
                    # Soft threshold
                    if np.abs(naive_update) <= gamma:
                        naive_update = 0
                    else: 
                        naive_update = np.sign(naive_update)*(np.abs(naive_update)-gamma)
                    # Shrinkage
                    shrink_factor = (1/H_ring[..., ff].shape[0]) * np.dot(H_ring[:, jj, ff], H_ring[:, jj, ff]) + (1-alpha)*l
                    current_estimate[jj] = naive_update / shrink_factor
                # Termination Tolerances
                iteration_change = (current_estimate - previous_estimate) / (1 + np.abs(previous_estimate))
                if np.linalg.norm(iteration_change, ord = np.inf) < break_tolerance:
                    break
                if ii == int(max_number_of_iterations):
                    print('The optimizer hit the maximum number of iterations and did not converge')
            forces_full_path_ring[kk+1, :, ff] = current_estimate
        if ff%100 == 0:
            print('Finished loop '+str(ff))
    return forces_full_path_ring, lambda_values
"""
"""deprecated function that creates the lambdas for the elastic net regularization
def create_lambdas_for_elastic_net(H, x, alpha, number_of_lambdas = 100):
    
    Calculates a range of lambda values for the elastic net regularization
    path. Starts at zero and ends at a value where most (if not all) the 
    estimated forces will be zero.

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

    Returns
    -------
    lambda_values : ndarray
        The vector of lambda values for the elastic net regularization. It starts
        at the smallest value and goes to the biggest value. 

    Raises
    ------
    ValueError
        If the alpha paramer is equal to zero.
    
    if alpha == 0:
        raise ValueError('alpha cannot be zero when using the elastic net')
    lambda_max = np.max(np.abs(H.T@x))/(H.shape[0]*alpha)
    # There's no actual way to start at zero, so this is how I'm doing it
    lambda_start = lambda_max/(number_of_lambdas*100)
    lambda_values = np.exp(np.linspace(start = np.log(lambda_start), stop = np.log(lambda_max), num = number_of_lambdas-1))
    lambda_values = np.insert(lambda_values, 0, 0)
    #lambda_values = np.geomspace(start = lambda_start, stop = lambda_max+lambda_start, num = number_of_lambdas) - lambda_start 
    return lambda_values
"""
""" deprecated function for coordinate descent at a single lambda 
def elastic_net_single_lambda(H, x, initial_f, 
                              lambda_value, alpha, 
                              max_number_of_iterations = int(1e5), 
                              break_tolerance = 1e-4):
    
    Estimates the forces for a single frequency line and lambda value using
    the elastic net with coordinate descent.

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
    lambda_value : ndarray
        The parameter for the elastic net regularization.
    alpha : float
        Alpha parameter for the elastic net. This controls the balance between the
        L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
        than 0 and less than 1. 
    max_number_of_iterations : int
        This is the maximum number of iterations (cycles) in the coordinate descent
        optimization. The default is 1e5.
    break_tolerance : float
        This is the break tolerance for the coordinate descent optimization. The 
        default is 1e-5.

    Returns
    -------
    current_estimate : ndarray
        Vector of estimated forces for all the lambdas, organized It is a real 
        valued array that is matches the frfs and responses, where the real 
        values are in the "top" half and the complex values are in the "bottom" 
        half. 

    Raises
    ------
    Warning
        If the optimizer hits the maximum number of iterations without reaching 
        the break tolerance. 

    References
    ----------
    .. [1] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
           Linear Models via Coordinate Descent, Journal of Statistical Software, 
           Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01.
     
    max_number_of_iterations = int(max_number_of_iterations)
    current_estimate = initial_f.copy()
    gamma = lambda_value*alpha
    for ii in range(max_number_of_iterations):
        previous_estimate = current_estimate.copy()
        for jj in range(H.shape[1]):
            partial_residual = x - H@current_estimate + H[:, jj]*current_estimate[jj]
            naive_update = np.dot(partial_residual, H[:, jj])/H.shape[0]
            # Soft threshold
            if np.abs(naive_update) <= gamma:
                naive_update = 0
            else: #standard soft threshold
                naive_update = np.sign(naive_update)*(np.abs(naive_update)-gamma)
            # Shrinkage
            shrink_factor = (1/H.shape[0]) * np.dot(H[:, jj], H[:, jj]) + (1-alpha)*lambda_value
            current_estimate[jj] = naive_update / shrink_factor
        # Termination Tolerances
        iteration_change = (current_estimate - previous_estimate) / (1 + np.abs(previous_estimate))
        if np.linalg.norm(iteration_change, ord = np.inf) < break_tolerance:
            break
        if ii == max_number_of_iterations:
            warnings.warn('The optimizer hit the maximum number of iterations and did not converge')
    return current_estimate
"""
""" deprecated elastic net full path function
def elastic_net_full_path_deprecated(H, x, 
                                     alpha, lambda_values = None, number_of_lambdas = 100, 
                                     max_number_of_iterations = 1e5, 
                                     break_tolerance = 1e-4):
    
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
    lambda_values : ndarray
        A vector of regularization parameters for the elastic net regularization. 
        This can be user supplied and the values should be organized from smallest
        to largest. The default (and likely best option) is for the code to 
        determine a good set of parameters on its own.
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
        This is returned if the option to have the code determine the lambda 
        values is used. 

    Note
    ----
    It is expected that the supplied lambda values will start at zero. This is
    not required, but the optimization may not find the most accurate value 
    for the forces (since the warm start will be zero instead of the standard
    pinv solution). 

    References
    ----------
    .. [1] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
           Linear Models via Coordinate Descent, Journal of Statistical Software, 
           Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01. 
    
    if lambda_values is None:
        lambda_output = True
        lambda_values = create_lambdas_for_elastic_net(H, x, alpha, 
                                                       number_of_lambdas = int(number_of_lambdas))

    forces_ring = np.zeros((H.shape[1], number_of_lambdas), dtype = np.float64)
    forces_ring[:, 0] = np.linalg.pinv(H)@x

    for ii, l in enumerate(lambda_values[1:]):
        forces_ring[:, ii+1] = elastic_net_single_lambda(H, x, forces_ring[:, ii],
                                                         l, alpha, 
                                                         max_number_of_iterations = int(max_number_of_iterations), 
                                                         break_tolerance = break_tolerance)
    if lambda_output:
        return forces_ring, lambda_values
    else:
        return forces_ring
"""
""" dprecated function for computing forces using the elastic net (replaced with a numba compatible function)
def force_estimation_elastic_net(H, x, 
                                 alpha,
                                 lambda_values = None,
                                 number_of_lambdas = 100,
                                 max_number_of_iterations = 1e6, 
                                 break_tolerance = 1e-4):
    
    Estimates forces using elastic net penalized linear regression using
    a cyclical coordinate descent optimization algorithm. 

    Parameters
    ----------
    H : TransferFunctionArray
        The FRFs being used in the elastic net linear regression.
    x : SpectrumArray
        The responses being used in the elastic net linear regression.
    alpha : float
        Alpha parameter for the elastic net. This controls the balance between the
        L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
        than 0 and less than 1. 
    lambda_values : ndarray
        The option for the user to to supply their own regularization parameters is 
        not implemented yet.
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
    forces : SpectrumArray
        The estimated forces from the elastic net linear regression for all values. 
        The ordinate (and corresponding spectrum array) is sized
        [number of lambdas x number of forces x number of lines].
    lambda_values : ndarray
        The lambda values that were used in the regression problem. It is sized
        [number of lines x number of lambdas]. 

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani, M. Wainright, Statistical Learning with Sparsity:
           The Lasso with Generalizations. Boca Raton, Fl: CRC Press, 2015. 
    .. [2] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
           Linear Models via Coordinate Descent, Journal of Statistical Software, 
           Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01. 
    
    if lambda_values is not None:
        raise NotImplementedError('The option for a user to supply lambda values is not implemented yet')

    H_ord_ring, x_ord_ring, reference_dofs = organize_data_for_lasso_optimization(H, x)

    # Looping over the elastic net optimization for all frequencies
    forces_full_path_ring = np.zeros((number_of_lambdas, H_ord_ring.shape[1], H_ord_ring.shape[-1]), dtype = np.float64)
    lambda_values = np.zeros((H_ord_ring.shape[-1], number_of_lambdas), dtype = np.float64)
    for ii in range(x_ord_ring.shape[-1]):
        temp_forces, lambda_values[ii, :] = elastic_net_full_path(H_ord_ring[..., ii], 
                                                                  x_ord_ring[..., ii], 
                                                                  alpha = alpha, 
                                                                  number_of_lambdas = number_of_lambdas,
                                                                  break_tolerance = break_tolerance,
                                                                  max_number_of_iterations = max_number_of_iterations)
        forces_full_path_ring[..., ii] = np.moveaxis(temp_forces, -1, 0)

    forces_full_path = forces_full_path_ring[:, :forces_full_path_ring.shape[1]//2, :] + forces_full_path_ring[:, forces_full_path_ring.shape[1]//2:, :]*1j

    forces = sdpy.data_array(sdpy.data.FunctionTypes.SPECTRUM, H.flatten()[0].abscissa, forces_full_path, reference_dofs)

    return forces, lambda_values
"""