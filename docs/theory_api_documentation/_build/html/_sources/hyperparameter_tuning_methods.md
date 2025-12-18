# Hyperparameter Tuning Methods
Many of the advanced inverse methods in ForceFinder focus on using a specific form of regularization with an algorithm to select an "optimal" hyperparamter based on one or more metrics. For example, the `auto_tikhonov_by_l_curve` methods use Tikhonov regularization and select the regularization parameter with an L-curve. This section will describe the basic concepts for the hyperparameter tuning methods. Refer to the references in the docstrings for the different methods for details. 

## Manual Tuning
The `manual_inverse` inverse method allows the practitioner to manually tune the hyperparameters for the TSVD and Tikhonov regularization. There are two methods for tuning the hyperparameter in the TSVD: a condition number threshold or a discrete number of singular values. The inverse is exactly the same, regardless of the metric that is being tuned, since the condition number threshold is converted to a number of retained singular values in the FRF matrix inverse function. Only the regularization parameter (the $\lambda$ value) can be tuned for Tikhonov regularization. 

There are many strategies for manually tuning the hyperparameters and the practitioner should develop a method that suits the ISE problem at hand. Some common strategies include:

- Determine the effective rank of the `training_frfs` and adjust the hyperparameter accordingly
- Adjust the hyperparameter until the estimated sources have amplitudes that seem reasonable or are within the capabilities of the test system (for MIMO vibration testing)
- Determine the noise floor of the `training_frfs` and/or `training_response` and adjust the hyperparameter to make the ISE problem robust to that noise level 

```{tip}
The quantities of interest for various hyperparameter tuning strategies are typically frequency dependent, so the `manual_inverse` method has been designed to allow for frequency dependent hyperparameters.
```

## L-Curve Methods
L-curve methods, which are sometimes referred to as trade-off curves, have been implemented for the TSVD and Tikhonov regularization in the `auto_truncation_by_l_curve` and `auto_tikhonov_by_l_curve` inverse methods. These methods solve the ISE problem for many different regularization parameters and construct the so-called L-curve, which is named because of the characteristic shape of the curve. An example L-curve with the selected regularization parameter is shown in the image below:

```{figure} images/sample_l_curve.png
:alt: Example L-curve
:align: center
```

```{note}
The L-curve methods in ForceFinder construct a curve for every frequency line (and every [COLA segment](sec:cola_method)) in the ISE problem.
```

In general, the L-curve inverse methods define the optimal regularization as the knee of curve. The meaning of this point can be interpreted differently, depending on the type of L-curve that is being used. Two types of L-curves have been implemented in ForceFinder, where the labels here are the same as the optional kwargs in the inverse methods:

1. `standard` - This method constructs the L-curve with the "size" of the residual squared error on the X-axis and the size of the sources on the Y-axis. The size of the residual squared error is computed with $\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2$ and the size of the sources is computed with $\lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_2^2$ (the Frobenius norm is used for CPSDs). This is the classical version of the L-curve and assumes that the optimal regularization shrinks the sources as much as possible before there is significant error in the response reconstruction.

2. `forces` - This method constructs the L-curve with the regularization parameter on the X-axis and the size of the sources on the Y-axis (using the same definition as above). This version of the L-curve attempts to find the maximum regularization parameter that has a significant effect on the size of the sources. 

```{note}
By convention of the vector/matrix norm, these methods only consider the amplitude of the response and source spectra.
```

ForceFinder generates an L-curve and attempts to identify an optimal regularization at every frequency line. Given, the typical number of frequency lines, it is unreasonable to manually identify the knee of the L-curve. As such, automatic methods must be used and two methods are available in ForceFinder, where the labels here are the same as the optional kwargs in the inverse methods:

1. `curvature` - This method assumes that the knee of the L-curve is at the location with maximum curvature.
2. `distance` - This method assumes that the knee of the L-curve is at the location that is the closest to the origin of the plot of the L-curve. The distance method may be useful for situations where the L-curve is poorly defined and the curvature method might lead to erratic results. An example of one such situation is the TSVD where there are few singular values to search over. The distance is computed by first offsetting and scaling the X and Y axes of the curve, so the values fall between zero and one. Then the optimal regularization is determined by finding the point with the minimum distance to the [0,0] point. The plot below shows how the L-curve is changed when making the adjustments for the distance method. 

```{figure} images/sample_l_curve_distance.png
:alt: Example L-curves for the Curvature and Distance Methods
:align: center
```

```{warning}
The L-curve methods assume that the ISE problem follows the discrete picard condition and that an appropriate range of regularization parameters were searched over to generate a properly shaped curve. The `auto_truncation_by_l_curve` or `auto_tikhonov_by_l_curve` inverse methods do not check if these conditions are met. 
```

```{warning}
Experience has shown that the methods for constructing the L-curve will have erratic results for underdetermined problems (when there are fewer response DOFs than source DOFs), which can lead to the selection of sub-optimal regularization values. 
```

```{note}
By definition, L-curve methods will attempt to regularize the ISE problem as much as possible. As such they may over regularize the problem if it is already well posed. It is up to the practitioner to determine if these methods are appropriate for the problem at hand. 
```

```{tip}
The L-curve methods exhaustively search through a specified number of regularization parameters, where the default is 100 values. The computation time is directly related to the number of regularization parameters. As such, fewer regularization parameters will lead to reduced solve times. 
``` 

## Cross Validation
Cross validation (CV) has been implemented in ForceFinder for Tikhonov regularization in the `auto_tikhonov_by_cv_rse` method. As a general technique, CV attempts to select a regularization parameter, which estimates sources that minimize the corresponding response prediction error. This is done by iteratively solving the ISE problem with different regularization parameters and different DOFs (or DOF sets) left out of the source estimation. 

The prediction error, for the DOF(s) that were left out, is computed at every iteration and the optimal regularization parameter is the one that has the lowest average prediction error over all the held-out DOF(s). There are several sample splitting strategies in CV and two have been implemented in ForceFinder: leave one out CV (LOOCV) and k-fold CV. An example curve showing the prediction error vs. regularization parameter for an ISE problem with LOOCV is shown in the image below:

```{figure} images/sample_cv_curve.png
:alt: Example Cross Validation Prediction Error Curve
:align: center
```

```{note}
The CV methods in ForceFinder construct a prediction error curve for every frequency line (and every [COLA segment](sec:cola_method)) in the ISE problem.
```

```{warning}
CV assumes that the sample splitting strategies provide _independent_ datasets for each iteration, where the predicted response for the held-out DOF(s) can be treated as predictions on unseen data. For ISE problems, this translates to assuming that the held-out DOF(s) observe unique dynamics and/or unique sources of error compared to the DOFs that are being used for the source estimation. The CV may lead to sub-optimal regularization parameters and corresponding sources if this assumption is not true.
```

### Leave One Out Cross Validation
LOOCV uses a relatively basic sample splitting strategy, where the number of iterations matches the number of DOFs in the `training_response_coordinate`, holding a single DOF out to compute the prediction error in each iteration. An example of this scheme is shown in the table below. 

<div style="font-size: 0.7em">

| CV Iteration | DOF1 | DOF2 | DOF3 | DOF4 | DOF5 | DOF6 | DOF7 | DOF8 | DOF9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|
|2|🟩 Left in|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|
|3|🟩 Left in|🟩 Left in|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|
|4|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|
|5|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|
|6|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|
|7|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟩 Left in|🟩 Left in|
|8|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟩 Left in|
|9|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out| 

</div>

LOOCV can suffer from two main issues:
- LOOCV can be computationally expensive since it exhaustively searches over all the possible DOFs and regularization parameters. For example, LOOCV with 30 training DOFs while searching over 100 regularization parameters (the default in many ForceFinder methods) will result in 3,000 iterations for each frequency line in the ISE problem. 
- The sample splitting in LOOCV may not result in independent datasets since only one DOF is being changed for each iteration. As such, LOOCV may not appropriately assess the prediction error, leading it to pick a sub-optimal regularization parameter.

### K-Fold Cross Validation
In k-fold CV, the practitioner specifies that the `training_response_coordinate` should be evenly split into k "folds", where each fold is an iteration in the CV. As such, $\frac{n}{k}$ samples are held-out in each fold to compute the prediction error, if there are $n$ DOFs in the `training_response_coordinate`. An example of this sample splitting scheme is shown in the table below.

```{note}
The DOFs are randomly shuffled prior to the sample splitting to reduce any potential bias in how the `training_response_coordinate` is organized (e.g., DOFs on similar portions of the structure having similar node numbers). This shuffling is shown in the table below.
``` 

<div style="font-size: 0.7em">

| CV Iteration | DOF6 | DOF4 | DOF2 | DOF5 | DOF1 | DOF9 | DOF7 | DOF3 | DOF8 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|🟥 Held-out|🟥 Held-out|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|
|2|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟥 Held-out|🟥 Held-out|🟩 Left in|🟩 Left in|🟩 Left in|
|3|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟩 Left in|🟥 Held-out|🟥 Held-out|🟥 Held-out|

</div>

```{tip}
The number of folds should be selected to balance having a significant number of folds while also having a significant difference in data between the folds. These goals are in contrast with each other since more held-out DOFs will typically lead to greater independence between each fold, but this will obviously reduce the number of folds, which makes it difficult to assess the prediction error for each regularization parameter. 
```

### Assessing Prediction Error
The `auto_tikhonov_by_cv_rse` method assesses the prediction error via the mean squared error (MSE) in the predicted response for each CV iteration:

$$
mse = \frac{1}{n} \Sigma \lvert X^{predicted} - X^{training} \rvert^2
$$

Where $\lvert \cdot \rvert$ indicates the absolute value of the error and there are $n$ DOFs in the held-out sample. The prediction error for each regularization parameter is computed as the sum of the MSEs for each CV iteration. Although it is not shown here, only the errors in the PSDs are considered if the responses are CPSD matrices. 

```{note}
The `auto_tikhonov_by_cv_rse` methods only use the MSE as an error metric, but the `leave_one_out_cv` and `k_fold_cv` functions in the `auto_regularization` module can take a generic function to assess the prediction error. 
```

## Information Criterions
The Akaike information criterion (AIC), Akaike corrected information criterion (AICC), and Bayesian information criterion (BIC) have been implemented in ForceFinder for selecting the optimal regularization parameter in the `elastic_net_by_information_criterion` method. The definition for the different information criterions is:

$$
AIC = 2k - 2ln\begin{pmatrix}\hat{L}\end{pmatrix}
$$

$$
AICC = AIC + \frac{2k^2+2k}{n-k-1}
$$

$$
BIC = 2ln\begin{pmatrix}k\end{pmatrix} - 2ln\begin{pmatrix}\hat{L}\end{pmatrix}
$$

```{note}
The information criterions have only been implemented for linear spectra in ForceFinder and the `elastic_net_by_information_criterion` has only been implemented for the `LinearSourcePathReceiver`.
```

Where $k$ is the number of active (non-zero) source DOFs in the estimate, $n$ is the number of response DOFs in the source estimation, and $\hat{L}$ is the maximized value for the likelihood function of the source estimation. The likelihood value in these expressions is computed with:

$$
2ln\begin{pmatrix}\hat{L}\end{pmatrix} = ln\begin{pmatrix}2\pi\sigma\end{pmatrix}n+\frac{\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2}{\sigma}
$$

Where $\sigma$ is computed with:

$$
\sigma = \frac{\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2}{n-p}
$$

Where $p$ is the number of reference DOFs in the FRFs. The above expressions make it clear that the information criterions attempt to balance the number of active sources with the accuracy of the `reconstructed_training_response` (computed as a likelihood function). This effect makes the information criterions especially useful for inverse methods that promote sparse solutions, such as the elastic net. Conversely, the information criterions are not well suited to methods like the TSVD or Tikhonov regularization, where $k$ remains that same for the whole range of hyperparameters, meaning that the method will only consider the error in the `reconstructed_training_response`.  

```{note}
The information criterions could be applied to Tikhonov regularization by computing an "effective" number of source DOFs, but this has not been implemented. 
```

Like the L-curve and CV methods, the information criterion is computed for a range of regularization parameters at each frequency. The optimal regularization parameter is defined as the parameter that minimizes the information criterion (for the specific frequency). An example curve showing the AICC vs. regularization parameter is shown in the image below. This was computed from an ISE problem that that utilized the `elastic_net_by_information_criterion` inverse method. The sudden jumps in the curve are likely the result of a source being set to zero by the solver at that regularization parameter.

```{figure} images/sample_aicc_curve.png
:alt: Example Information Criterion Curve
:align: center
```