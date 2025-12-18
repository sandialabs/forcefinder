forcefinder.core.sparse_functions
=================================

.. py:module:: forcefinder.core.sparse_functions

.. autoapi-nested-parse::

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



Functions
---------

.. autoapisummary::

   forcefinder.core.sparse_functions.elastic_net_full_path_all_frequencies_parallel
   forcefinder.core.sparse_functions.elastic_net_full_path


Module Contents
---------------

.. py:function:: elastic_net_full_path_all_frequencies_parallel(H, x, alpha, number_of_lambdas=100, max_number_of_iterations=1000000.0, break_tolerance=0.0001)

   Computes the elastic net forces via coordinate descent.

   :param H: FRF matrix for a single frequency line. This must be a converted version
             of the complex valued matrix to a ring of real valued matrices (real
             values in the top left and and bottom right corners, negative of the
             imaginary values cast to real in the top right corner, and positive
             imaginary values cast to real in the bottom left corner).
   :type H: ndarray
   :param x: Vector of (linear) responses for a single frequency line. This must be
             converted from a complex valued array to a vectorized real valued array
             that matches the FRF matrix (real values in the "top" half and imaginary
             values cast to real in the "bottom" half).
   :type x: ndarray
   :param alpha: Alpha parameter for the elastic net. This controls the balance between the
                 L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
                 than 0 and less than 1.
   :type alpha: float
   :param number_of_lambdas: This parameter is supplied if the lambda_values are being determined by
                             the code. The default is 100.
   :type number_of_lambdas: int
   :param max_number_of_iterations: This is the maximum number of iterations (cycles) in the coordinate descent
                                    optimization. The default is 1e5.
   :type max_number_of_iterations: int
   :param break_tolerance: This is the break tolerance for the coordinate descent optimization. The
                           default is 1e-5.
   :type break_tolerance: float

   :returns: * **forces_full_path** (*ndarray*) -- Matrix of estimated forces for all the lambdas, organized
               [force dof x lambda value].
             * **lambda_values** (*ndarray*) -- This is the regularization parameters that are used in the elastic net
               regularization.

   .. rubric:: Notes

   This is designed to be used with a SourcePathReceiver object and does not do any
   book keeping.

   .. rubric:: References

   .. [1] T. Hastie, R. Tibshirani, M. Wainright, Statistical Learning with Sparsity:
          The Lasso with Generalizations. Boca Raton, Fl: CRC Press, 2015.
   .. [2] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
          Linear Models via Coordinate Descent, Journal of Statistical Software,
          Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01.


.. py:function:: elastic_net_full_path(H, x, alpha, number_of_lambdas=100, max_number_of_iterations=1000000.0, break_tolerance=0.0001)

   Computes the full regularization path for a single frequency line via the
   elastic net.

   :param H: FRF matrix for a single frequency line. This must be a converted version
             of the complex valued matrix to a ring of real valued matrices (real
             values in the top left and and bottom right corners, negative of the
             imaginary values cast to real in the top right corner, and positive
             imaginary values cast to real in the bottom left corner).
   :type H: ndarray
   :param x: Vector of (linear) responses for a single frequency line. This must be
             converted from a complex valued array to a vectorized real valued array
             that matches the FRF matrix (real values in the "top" half and imaginary
             values cast to real in the "bottom" half).
   :type x: ndarray
   :param alpha: Alpha parameter for the elastic net. This controls the balance between the
                 L1 and L2 penalty (higher alpha weights the L1 more). It should be greater
                 than 0 and less than 1.
   :type alpha: float
   :param number_of_lambdas: This parameter is supplied if the lambda_values are being determined by
                             the code. The default is 100.
   :type number_of_lambdas: int
   :param max_number_of_iterations: This is the maximum number of iterations (cycles) in the coordinate descent
                                    optimization. The default is 1e5.
   :type max_number_of_iterations: int
   :param break_tolerance: This is the break tolerance for the coordinate descent optimization. The
                           default is 1e-5.
   :type break_tolerance: float

   :returns: * **forces_ring** (*ndarray*) -- Matrix of estimated forces for all the lambdas, organized
               [force dof x lambda value]. It is a real valued array that is matches
               the frfs and responses, where the real values are in the "top" half
               and the complex values are in the "bottom" half.
             * **lambda_values** (*ndarray*) -- This is the regularization parameters that are used in the elastic net
               regularization.

   .. rubric:: References

   .. [1] J.H. Friedman, T. Hastie, R. Tibshirani, Regularization Paths for Generalized
          Linear Models via Coordinate Descent, Journal of Statistical Software,
          Volume 33, Issue 1, 2010, Pages 1-22, https://doi.org/10.18637/jss.v033.i01.


