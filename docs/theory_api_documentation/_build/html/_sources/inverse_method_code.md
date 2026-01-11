# Anatomy of an Inverse Method
All the inverse methods in ForceFinder follow a similar code layout to one another and knowing this layout can be helpful when reviewing the code or implementing a new inverse method. Although the layout is similar, there are slight differences for each SPR object type (based on differences in the `inverse_processing` decorator functions), so they will all be reviewed here. 

## LinearSourcePathReceiver Inverse Methods
The basic code layout for an inverse method in the `LinearSourcePathReceiver` is:

```{code-block} python
@linear_inverse_processing
def inverse_method(self, ..., use_transformation=True, 
                   response=None, frf=None):
    """
    Documentation string
    """
    force = inverse_method_sub_function(frf, response, ...)

    self.inverse_settings.update(...)

    return force
```

As previously described, the `linear_inverse_processing` decorator function gathers the response and FRF data from the SPR object and passes it to the `inverse_method` as the `response` and `frf` kwargs. Note that the decorator function will read the kwargs in the `inverse_method` function signature, and automatically apply transformations to the FRF and response data if the `use_transformation` kwarg is set to true. The `...` in the function signatures indicates method specific arguments that are used in the inverse method code. 

The sources are estimated with the `inverse_method_sub_function`, which is dependent on the actual inverse method and may include several sub functions, depending on what is convenient for a modular and self documenting code structure. The inverse method kwargs and any other pertinent settings (such as hyperparameters that were determined in the inverse method) are saved into the `inverse_settings` dictionary, which is an attribute of the SPR object that can be recalled outside of the inverse method. 

The `inverse_method` returns the estimated sources as an ndarray, which is labeled `force` and is organized [abscissa, force DOFs]. The returned value is read into the `linear_inverse_processing` decorator function and subsequently saved into the SPR object. The estimated sources will be converted from the transformed to physical quantity, depending on if the `use_transformation` kwarg is set to true or false. 

## PowerSourcePathReceiver Inverse Methods
The basic code layout for the `PowerSourcePathReceiver` is the same the `LinearSourcePathReceiver`, except it includes two additional kwargs:

```{code-block} python
@power_inverse_processing
def inverse_method(self, ..., use_transformation=True,
                   use_buzz=False, update_header=True, 
                   response=None, frf=None):
    """
    Documentation string
    """
    force = inverse_method_sub_function(frf, response, ...)

    self.inverse_settings.update(...)

    return force
```

The behavior of the code for the `PowerSourcePathReceiver` is exactly the the same as the `LinearSourcePathReceiver` except for the additional kwargs. These kwargs change the behavior of the inverse method in the following ways:
- `use_buzz` - If this lwarg is set to true, the `training_response` is updated with the [buzz method](sec:buzz_method), using the `buzz_cpsd` attribute of the SPR object, prior to the source estimation. This update is performed in the `power_inverse_processing` decorator function. 
- `update_header` - If this kwarg is true, the kwargs and other pertinent settings will be saved to the `inverse_settings` dictionary. Otherwise the settings will not be saved. This kwarg is used for Rattlesnake integration to prevent unexpected errors when recomputing the sources, since the ForceFinder control law uses additional header information that isn't captured in the standard `inverse_settings` dictionary. 

(sec:transient_inverse_code)=
## TransientSourcePathReceiver Inverse Methods
The layout for the `TransientSourcePathReceiver` is different than the other SPR object types since the inverse problem is done with [COLA segments](sec:cola_method), which requires looping and adds several optional kwargs: 

```{code-block} python
@transient_inverse_processing
def inverse_method(self, ..., cola_frame_length = None,
                   cola_window = ('tukey', 0.5),
                   cola_overlap_samples = None,
                   frf_interpolation_type = 'sinc',
                   transformation_interpolation_type = 'cubic',
                   use_transformation=True,
                   response_generator=None, frf=None 
                   reconstruction_generator=None):
    """
    Documentation string
    """
    frf_inverse = frf_inverse_sub_function(frf, ...)
    
    for segment_fft in response_generator:
        reconstructed_force = reconstruction_generator.send(frf_inverse@segment_fft)

    self.inverse_settings.update(...)

    return reconstructed_force
```

The extra optional kwargs for the inverse methods in the `TransientSourcePathReceiver` determine the behavior of the COLA processing in the `transient_inverse_processing` decorator function:
- `cola_window` - This is the window that is used when creating the COLA segments
- `cola_overlap_samples` - The number of overlapping samples between the cola segments (this must ensure a COLA condition)
- `frf_interpolation_type` - The type of interpolation that is used to make the frequency resolution in the FRFs match the zero padded response COLA segments
- `tranformation_interpolation_type` - The type of interpolation that is used to make the frequency resolution in the transformations match the zero padded response COLA segments, this is only used if the transformation is frequency dependent (which is detected by the shape of the array) 

```{warning}
Non-standard COLA settings can lead to unexpected performance issues in the source estimation. These parameters should only be modified from the defaults when there is a substantial reason to justify the change. 
```

The `transient_inverse_processing` decorator function also creates the `response_generator` and `reconstruction_generator` functions, which perform the following tasks:
- `response_generator` - Splits the response time data into the COLA segments, zero pads the segmented data, and converts the responses to the frequency domain. 
- `reconstruction_generator` - Converts the estimated sources from the frequency domain back to the time domain and performs the COLA processing to recompile the segmented source time traces into a single time trace.

The generator functions are used instead of one mass operation to prevent an excessive amount of data from being loaded into memory at one time, which could easily happen for long time traces or signals with high sampling rates. Computational efficiency is also why the FRF inverse is computed outside of the loop (to prevent recomputing the same inverse many times). However, the exact functions that occur outside or inside of the loop depend on the specific inverse method. 

```{tip}
Any inverse method that works for the `LinearSourcePathReceiver` will likely work for the `TransientSourcePathReceiver` since the inverse problem form (within the inverse method) is the same for both object types. Some minor changes may be necessary for computational efficiency and compatibility with the zero padding in the FRFs. 
```