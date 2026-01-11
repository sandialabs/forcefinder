# Theory Guide
This section of the documentation introduces the mathematical theory behind the various methods in ForceFinder to explicitly describe what is happening "under the hood". It will go over the following:

- [Inverse Problem Form](inverse_problem_form) - The basic problem problem form that is used in the different SPR types
- [Regularization Methods](regularization_methods) - Theoretical information on the regularization techniques that are used in ForceFinder
- [Hyperparameter Tuning Methods](hyperparameter_tuning_methods) - Some basic information on the methods that are used to automatically select the hyperparameters in the different inverse methods (e.g., cross validation, L-curve, etc.)
- [Transformations](transformation_theory) - The theory for how transformations are applied in ForceFinder
- [Miscellaneous Techniques](miscellaneous_techniques) - Goes over the theory for miscellaneous techniques that have been implemented in ForceFinder, such as the buzz method and match trace updating
- [Error Metrics](error_metrics) - Provides the basic descriptions and theory for the different error metrics in ForceFinder.

```{tip}
References are included in the docstrings for several of the inverse methods so the interested practitioner can develop a deeper understanding for the theoretical underpinnings of the techniques. 
```
(theory_notation)=
## Notation
The documentation follows a consistent mathematical notation, the matrix and vector notation is:

| Quantity | Notation |
|:---:|:---:|
| Matrix Quantity | $\begin{bmatrix} \cdot \end{bmatrix}$ |
| Vector Quantity | $\begin{Bmatrix} \cdot \end{Bmatrix}$ |
| Pseudo-inverse | $\begin{bmatrix} \cdot \end{bmatrix}^+$ |
| Transpose | $\begin{bmatrix} \cdot \end{bmatrix}^T$ |
| Conjugate-transpose | $\begin{bmatrix} \cdot \end{bmatrix}^*$ |
| Transformed Quantity | $ \hat{\cdot} $ |
| Vector/Matrix Norm (n represents the norm order) | $ \lVert \cdot \rVert_n $ |
| Diagonal Operator | $ diag(\cdot) $ |
| Minimum Operator | $ min(\cdot) $ |


The symbols for the standard quantities in ISE are:
| Quantity | Notation |
|:---:|:---:|
| FRF Matrix | $\begin{bmatrix} H \end{bmatrix}$ |
| Diagonal Matrix of FRF Singular Values | $\begin{bmatrix} S \end{bmatrix}$ |
| Square Matrix of FRF Right Singular Vectors | $\begin{bmatrix} V \end{bmatrix}$ |
| Rectangular Matrix of FRF Left Singular Vectors | $\begin{bmatrix} U \end{bmatrix}$ |
| Response Vector (linear spectra and time trace) | $\begin{Bmatrix} X \end{Bmatrix}$ |
| Source Vector (linear spectra and time trace) | $\begin{Bmatrix} F \end{Bmatrix}$ |
| Response CPSD Matrix | $\begin{bmatrix} G_{xx} \end{bmatrix}$ |
| Source CPSD Matrix | $\begin{bmatrix} G_{ff} \end{bmatrix}$ |
| Phase | $ \phi $ |
| Coherence | $ \gamma $ |
| Response Transformation Matrix | $\begin{bmatrix} T_{res} \end{bmatrix}$ |
| Reference Transformation Matrix | $\begin{bmatrix} T_{ref} \end{bmatrix}$ |
| Regularization Parameter | $ \lambda $ |
| Optimization constraint | $ \tau $ |
| Spare Hyperparameter | $ \alpha $ |

```{note}
In general, the FRF and response quantities have an abscissa (time or frequency) dependency, but this has been left out for brevity
```
```{note}
The phase and coherence quantities may be computed between different response and reference DOFs, depending on the application, so these DOFs will be called out when necessary in the documentation. 
```