# Error Metrics
Several error metrics have been implemented in ForceFinder to evaluate the accuracy of the source estimation via the accuracy of the reconstructed response compared to a truth response. Most of these metrics compute a summary curve that attempts to represent the errors for all the DOFs in a single spectra or time trace. Additional methods are available to _plot_ the errors in the reconstructed responses, but they are not described here. 

Each of the error metric methods in ForceFinder have an optional kwarg that is called `channel_set`, which determines the DOFs for the truth and reconstructed responses that the metric is computed for. The options for this kwarg are:

- `training` - This computes the error metric between the `transformed_training_response` and `transformed_reconstructed_response` attributes of the SPR object.
- `validation` - This computes the error metric between the `validation_response` and `reconstructed_validation_response` attributes of the SPR object.
- `target` - This computes the error metric between the `target_response` and `reconstructed_target_response` attributes of the SPR object.

```{note}
The error metrics in ForceFinder are used to understand how well the estimated sources reconstruct the training/validation/target responses, which may not be indicative of the sources ability to predict responses at unseen locations or on unseen systems (in the case of component based TPA)
```
```{tip}
Unless otherwise noted, the error metrics are implemented as class methods in ForceFinder and are used with method call on the SPR object, such as: `spr_object.error_metric()`.
```

## Error Metrics for Spectral ISE Problems
The `LinearSourcePathReceiver` and `PowerSourcePathReceiver` use the same metrics, which evaluate the error in the PSDs for the different DOF sets. The responses for the `LinearSourcePathReceiver` must be converted from linear spectra to PSDs prior to computing the error metric. This is done with the following operation:

$$
G_{xx} = \frac{1}{\Delta f}\lvert X \rvert^2
$$

Where $G_{xx}$ is a PSD for a single DOF, $X$ is a spectra for a single DOF, and $\Delta f$ is the frequency resolution of the SPR object, which is given by the `abscissa_spacing` attribute.

```{note}
The error metrics for spectral ISE problems commonly use the ASD acronym, which stands for auto-spectral density and is equivalent to a PSD. The ASD acronym is used here to follow the convention for MIMO vibration testing standards. 
``` 
```{note}
All the equations for spectral ISE problems have a frequency dependency, but this has been left ouf for brevity.
```

(sec:global_asd_error)=
### Global ASD Error
The global ASD error, which is computed with the `global_asd_error` method, is a summary metric that is defined in MIL-STD 810. It sums the dB error for all the response DOFs while applying weights that are based on the relative response amplitudes. These weights are used to make the metric sensitive to errors in responses with large amplitudes and insensitive to errors in responses that have small amplitudes. As such, the global ASD error metric helps determine if the estimated sources apply sufficient vibration energy to a system in MIMO vibration testing, but may not be useful for a detailed investigation of the errors, since low responding DOFs may be ignored. 

The global ASD error is computed via a four step process:

1. A normalizing factor, $\eta$, is first computed, by taking the L2 norm of the truth response PSDs:

$$
\eta = \lVert diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{truth} \end{bmatrix}\end{pmatrix} \rVert_2
$$

2. A weighting vector, $\begin{Bmatrix}W\end{Bmatrix}$ is computed by dividing the PSD amplitude for each DOF by $\eta$:

$$
\begin{Bmatrix}W\end{Bmatrix} = \frac{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{truth} \end{bmatrix}\end{pmatrix}^2}{\eta^2}
$$

3. The dB error is computed for each DOF:

$$
\begin{Bmatrix}E_{dB}\end{Bmatrix} = 10*log_{10}\begin{pmatrix} \frac{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{reconstructed} \end{bmatrix}\end{pmatrix}}{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{truth} \end{bmatrix}\end{pmatrix}} \end{pmatrix}
$$

4. Finally, the global ASD error is computed by summing the element wise multiplication of $\begin{Bmatrix}E_{dB}\end{Bmatrix}$ and $\begin{Bmatrix}W\end{Bmatrix}$:

$$
E_{global} = \sum{\begin{Bmatrix}E_{dB}\end{Bmatrix}*\begin{Bmatrix}W\end{Bmatrix}}
$$

### Average ASD Error
The average ASD error, which is computed with the `average_asd_error` method, is a simple average of the dB error spectra for the all the response DOFs. The metric is computed with:

$$
E_{average} = 10*log_{10}\begin{pmatrix} \frac{1}{n}\sum{\frac{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{reconstructed} \end{bmatrix}\end{pmatrix}}{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{truth} \end{bmatrix}\end{pmatrix}}} \end{pmatrix}
$$

Where $n$ is the number of response DOFs for the metric computation.

```{note}
Decibel values are averaged on the corresponding linear values, which is why the average is done on the ratio of the reconstructed and truth PSDs.
```
```{note}
Many ISE problems are computed as least squares problems, which tend to result in an similar quantities of positive and negative errors. Consequently, the average ASD error may show less error than a subjective perception of the DOF by DOF error. However, it can be useful for quickly identifying large bias errors. 
```
(sec:rms_asd_error)=
### RMS ASD Error
The RMS ASD error, which is computed with a `rms_asd_error` method, is summary metric that computes the RMS value of the dB error spectra for all the response DOFs. This is done to have an error metric that treats the positive and negative errors the same, which may potentially be a better match to a subjective perception of the DOF by DOF error that the `average_asd_error`. The metric is computed with:

$$
\begin{Bmatrix}E_{dB}\end{Bmatrix} = 10*log_{10}\begin{pmatrix} \frac{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{reconstructed} \end{bmatrix}\end{pmatrix}}{diag \begin{pmatrix}\begin{bmatrix} G_{xx}^{truth} \end{bmatrix}\end{pmatrix}} \end{pmatrix}
$$

$$ 
E_{rms} = \sqrt{\frac{1}{n}\sum{\begin{Bmatrix}E_{dB}\end{Bmatrix}^2}}
$$

Where $\begin{Bmatrix} E_{dB} \end{Bmatrix}$ is the DOF by DOF dB error spectra and $n$ is the number of response DOFs for the metric computation.

(sec:transient_error_metrics)=
## Error Metrics for Transient Problems
Several error metrics have been implemented for transient problems, which attempt to evaluate the errors in response level, waveform shape, and spectral content. All of these metrics are time varying and are computed by splitting the full time trace into segments and computing the error on a segment-by-segment basis. The segmentation is specified with two parameters:

- Segment duration - This is the duration of the segment, which is specified as a time with the `frame_length` kwarg or an integer number of samples with the `samples_per_frame` kwarg.
- Overlap between segments - This is the overlap between the segments, which is specified as a percentage (in decimal format) with the `overlap` kwarg or as an integer number of samples with the `overlap_samples` kwarg.

(sec:global_rms_error)=
### Global RMS Error
The global RMS error, which is computed with the `global_rms_error` method, is a summary metric that is defined in MIL-STD 810. It sums the RMS errors for all the response DOFs while applying weights that are based on the relative response amplitudes. These weights are used to make the metric sensitive to errors in responses with large amplitudes and insensitive to errors in responses that have small amplitudes. As such, the global RMS error metric helps determine if the estimated sources apply sufficient vibration energy to a system in MIMO vibration testing, but may not be useful for a detailed investigation of the errors, since low responding DOFs may be ignored. 

The global RMS error is computed with the same four step process that is used for the [global ASD error](sec:global_asd_error), but applied to RMS levels vs. time instead of PSD amplitudes vs. frequency:

1. A normalizing factor, $\eta$, is first computed, by taking the L2 norm of the time varying RMS level for the truth response, $\begin{Bmatrix} {RMS}^{truth} \end{Bmatrix}$:

$$
\eta = \lVert \begin{Bmatrix} {RMS}^{truth} \end{Bmatrix} \rVert_2
$$

2. A weighting vector, $\begin{Bmatrix}W\end{Bmatrix}$ is computed by dividing the time varying RMS level for the truth response, $\begin{Bmatrix} {RMS}^{truth} \end{Bmatrix}$, by $\eta$:

$$
\begin{Bmatrix}W\end{Bmatrix} = \frac{\begin{Bmatrix} {RMS}^{truth} \end{Bmatrix}^2}{\eta^2}
$$

3. The dB error between the time varying RMS level for the truth response, $\begin{Bmatrix} RMS^{truth} \end{Bmatrix}$, and reconstructed response, $\begin{Bmatrix} RMS^{reconstructed} \end{Bmatrix}$, is computed for each DOF:

$$
\begin{Bmatrix}E_{dB}\end{Bmatrix} = 20*log_{10}\begin{pmatrix} \frac{\begin{Bmatrix} {RMS}^{reconstructed} \end{Bmatrix}}{\begin{Bmatrix} {RMS}^{truth} \end{Bmatrix}} \end{pmatrix}
$$

4. Finally, the global RMS error is computed by summing the element wise multiplication of $\begin{Bmatrix}E_{dB}\end{Bmatrix}$ and $\begin{Bmatrix}W\end{Bmatrix}$:

$$
E_{global} = \sum{\begin{Bmatrix}E_{dB}\end{Bmatrix}*\begin{Bmatrix}W\end{Bmatrix}}
$$

### Average RMS Error
The average RMS error, which is computed with the `average_rms_error` method, is a simple average of the dB RMS level error time traces for the all the response DOFs. The metric is computed for each time segment with the following expression:

$$
E_{average} = 20*log_{10}\begin{pmatrix} \frac{1}{n}\sum{\frac{\begin{Bmatrix} {RMS}^{reconstructed} \end{Bmatrix}}{\begin{Bmatrix} {RMS}^{truth} \end{Bmatrix}}} \end{pmatrix}
$$

Where $n$ is the number of response DOFs for the metric computation, $\begin{Bmatrix}RMS^{reconstructed}\end{Bmatrix}$ is the time varying RMS level for the reconstructed response, and $\begin{Bmatrix}RMS^{truth}\end{Bmatrix}$ is the time varying RMS level for the truth response.

```{note}
Decibel values are averaged on the corresponding linear values, which is why the average is done on the ratio of the reconstructed and truth RMS level errors.
```
```{note}
Many ISE problems are computed as least squares problems, which tend to result in an similar quantities of positive and negative errors. Consequently, the average RMS error may show less error than a subjective perception of the DOF by DOF error. However, it can be useful for quickly identifying large bias errors. 
```

(sec:time_varying_trac)=
### Time Varying TRAC
As the name implies, this metric computes a TRAC error time trace (based on the segmentation) for all the response DOFs and is computed with the `time_varying_trac` method. The TRAC error is computed for each DOF (at each segment) with:  

$$
{TRAC} = \frac{\begin{pmatrix}\begin{Bmatrix}x_n^{truth}\end{Bmatrix} \cdot \begin{Bmatrix}x_n^{reconstructed}\end{Bmatrix}\end{pmatrix}^2}{\begin{pmatrix}\begin{Bmatrix}x_n^{truth}\end{Bmatrix} \cdot \begin{Bmatrix}x_n^{truth}\end{Bmatrix}\end{pmatrix}*\begin{pmatrix}\begin{Bmatrix}x_n^{reconstructed}\end{Bmatrix} \cdot \begin{Bmatrix}x_n^{reconstructed}\end{Bmatrix}\end{pmatrix}}
$$

Where $n$ represents the response DOF index and the response vectors, $\begin{Bmatrix}x_n^{truth}\end{Bmatrix}$ and $\begin{Bmatrix}x_n^{reconstructed}\end{Bmatrix}$, are the time traces for a specific time segment and DOF (i.e., each entry in the vector is a different time sample). 

```{note}
The `time_varying_trac` method returns a SDynPy `TimeHistoryArray` with the time varying TRAC for each DOF and does not attempt to summarize the TRACs for the different DOFs into a single curve.
```

(sec:time_varying_level_error)=
### Time Varying Level Error
The time varying level error, which is computed with the `time_varying_level_error` method, computes the response level error in dB for all the DOFs rather than computing a single summary curve (like the `global_rms_error`, etc.). Currently, two types of levels are supported: the segment RMS level error and the segment maximum level error. 

(sec:spectrogram_error)=
### Spectrogram Error
The spectrogram error computes a short-time Fourier transform (STFT) PSD for all the DOFs, then computes the dB error between the truth and reconstructed STFTs. This metric is computed with the `compute_error_stft` function that is in the `transient_quality_metrics` module. The spectrogram error attempts to show the spectral errors as a function of time and can be useful to develop a thorough understanding of the errors in the ISE problem. 

It can be difficult to interpret the spectrograms to determine if the errors are significant to the overall response. This is because the dB error calculation makes it impossible to understand the response amplitude vs. time. For example, the spectrogram error could show high error but that error might occur at a time or frequency that has a low response amplitude, meaning that the error is insignificant to the overall response. An RMS level normalization, which is called with the `normalize_by_rms` kwarg, was added to the `compute_error_stft` function in an attempt to mitigate this issue. 

The normalization is computed for a specific DOF with:

$$
\eta = \frac{{RMS}^{truth}}{max\begin{pmatrix} {RMS}^{truth} \end{pmatrix}}
$$

Where $\eta$ is the time varying normalization factor (for a specific DOF), ${RMS}^{truth}$ is the RMS value vs. time (based on the segmentation), and $max\begin{pmatrix} {RMS}^{truth} \end{pmatrix}$ computes the maximum RMS value for the whole time trace. This normalization is applied to the spectrogram error for a specific DOF with:

$$
E_{STFT, normalized} = E_{STFT} * \eta
$$

Where the normalization for each segment is computed for the dB error spectrum for each time segment. 