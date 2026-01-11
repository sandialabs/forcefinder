# Transformation Theory
Transformations have been implemented for every SPR object type in ForceFinder for two main purposes:

- Virtual point transformations (VPT), like what is common in TPA, substructuring, and MIMO vibration testing
- DOF weightings and normalizations

The most basic transformations are applied to time traces or linear spectra of the responses or sources, as shown below:

$$
\begin{Bmatrix} \hat{X} \end{Bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{Bmatrix} X \end{Bmatrix}
$$
$$
\begin{Bmatrix} \hat{F} \end{Bmatrix} = \begin{bmatrix} T_{ref} \end{bmatrix} \begin{Bmatrix} F \end{Bmatrix}
$$

Where $\begin{Bmatrix} \hat{X} \end{Bmatrix}$ is the transformed response and $\begin{Bmatrix} \hat{F} \end{Bmatrix}$ is the transformed source. The transformations convert the responses and sources from physical quantities to transformed quantities. The response transformation is applied to the FRFs by inserting the FRF equation of motion (EoM) into the response transformation equation:

$$
\begin{Bmatrix} \hat{X} \end{Bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{Bmatrix} X \end{Bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{bmatrix} H \end{bmatrix} \begin{Bmatrix} F \end{Bmatrix}
$$

Where the transformed FRFs can be explicitly extracted as:

$$
\begin{bmatrix} \hat{H} \end{bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{bmatrix} H \end{bmatrix}
$$

The source transformation transformation is applied by first inverting the source transformation equation to convert a transformed source to a physical quantity:

$$
\begin{Bmatrix} F \end{Bmatrix} = \begin{bmatrix} T_{ref} \end{bmatrix}^+ \begin{Bmatrix} \hat{F} \end{Bmatrix}
$$

This expression for the physical source is then applied to the FRF EoM:

$$
\begin{Bmatrix} X \end{Bmatrix} = \begin{bmatrix} H \end{bmatrix} \begin{Bmatrix} F \end{Bmatrix} = \begin{bmatrix} H \end{bmatrix} \begin{bmatrix} T_{ref} \end{bmatrix}^+ \begin{Bmatrix} \hat{F} \end{Bmatrix}
$$

Where the transformed FRFs are extracted as:

$$
\begin{bmatrix} \hat{H} \end{bmatrix} = \begin{bmatrix} H \end{bmatrix} \begin{bmatrix} T_{ref} \end{bmatrix}^+
$$

The expressions for the response and reference transformed FRFs can be combined, to simultaneously apply the response and reference transforms:

$$
\begin{bmatrix} \hat{H} \end{bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{bmatrix} H \end{bmatrix} \begin{bmatrix} T_{ref} \end{bmatrix}^+
$$

```{note}
Transformations have been implemented in ForceFinder so the `reference_transformation` and `response_transformation` are simultaneously applied or not applied at all. Practitioners will need to create separate SPR objects if they wish to apply transformations one at a time.
```

## Transformations on Power Spectra
The transformation equations can be easily adapted to power spectra by first defining how power spectra are computed from linear spectra (using transformed quantities):

$$
\begin{bmatrix} \hat{G}_{xx} \end{bmatrix} = \begin{Bmatrix} \hat{X} \end{Bmatrix} \begin{Bmatrix} \hat{X} \end{Bmatrix}^*
$$

The expression for the linear spectra transformation can then be inserted into the equation for computing the power spectra and manipulated to obtain the expression for transformed power spectra:

$$
\begin{gather*}
\begin{bmatrix} \hat{G}_{xx} \end{bmatrix} = \begin{pmatrix} \begin{bmatrix} T_{res} \end{bmatrix} \begin{Bmatrix} X \end{Bmatrix} \end{pmatrix} \begin{pmatrix}  \begin{bmatrix} T_{res} \end{bmatrix} \begin{Bmatrix} X \end{Bmatrix}\end{pmatrix}^*
\\
\downarrow
\\
\begin{bmatrix} \hat{G}_{xx} \end{bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{Bmatrix} X \end{Bmatrix} \begin{Bmatrix} X \end{Bmatrix}^* \begin{bmatrix} T_{res} \end{bmatrix}^*
\\
\downarrow
\\
\begin{bmatrix} \hat{G}_{xx} \end{bmatrix} = \begin{bmatrix} T_{res} \end{bmatrix} \begin{bmatrix} G_{xx} \end{bmatrix} \begin{bmatrix} T_{res} \end{bmatrix}^*
\end{gather*}
$$

The source power spectra use the same expression as the responses to convert from physical to transformed quantities:

$$
\begin{bmatrix} \hat{G}_{ff} \end{bmatrix} = \begin{bmatrix} T_{ref} \end{bmatrix} \begin{bmatrix} G_{ff} \end{bmatrix} \begin{bmatrix} T_{ref} \end{bmatrix}^*
$$

## DOF Weightings 
DOF weightings are implemented in ForceFinder through the transformations, where a default (identity) transformation would have weightings on the diagonal. In most cases, the weightings should be applied to the transformations with the `apply_response_weighting` or `apply_reference_weighting` method, which uses the following expression: 

$$
\begin{bmatrix} T^{weighted} \end{bmatrix} = diag\begin{pmatrix}\begin{Bmatrix} W_{transformed} \end{Bmatrix}\end{pmatrix} \begin{bmatrix} T^{original} \end{bmatrix} diag\begin{pmatrix}\begin{Bmatrix} W_{untransformed} \end{Bmatrix}\end{pmatrix}
$$

```{warning}
Unlike most methods in ForceFinder, the `apply_response_weighting` and `apply_reference_weighting` methods take `ndarrays` as arguments and don't check that the weighting organization matches the DOFs in the SPR object.
```
```{note}
Since the reference transformation is inverted when it is used in the inverse method, it may be appropriate to to invert the DOF weighting to achieve the desired effect.
```

A frequency dependent normalization can be applied to the `response_transformation` or `reference_transformation` using the `set_response_transformation_by_normalization` or `set_reference_transformation_by_normalization` methods. These methods set the DOF weightings by the inverse of the standard deviation of the accompanying row or column in the FRF matrix.