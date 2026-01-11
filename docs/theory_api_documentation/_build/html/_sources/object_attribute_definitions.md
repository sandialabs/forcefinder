# SourcePathReceiver Object Attributes
The SPR objects in ForceFinder contain several attributes that are useful to review in the ISE process. The definitions for the different attributes are included in the glossary below. 
```{warning}
In general, the class attributes should be treated as read-only variables once the SPR object is created, unless they are being modified through a class method (e.g., how the `response_transformation` attribute is modified by the `apply_response_weighting` method). Technically, some attributes can be overwritten or modified, but this is considered bad practice. Further, some operations, which may seem logical for a SDynPy user will not work in ForceFinder. For example, the `spr_object.frfs.ordinate = data` command will not write or modify any data into the SPR object. 
```  

```{glossary}
abscissa
    The frequency vector for the frequency domain data in the SPR object.

abscissa_spacing
    The frequency resolution for the frequency domain data in the SPR object.

buzz_cpsd (`PowerSourcePathReceiver` only)
    The so-called buzz CPSD that is used to update the `training_response` when the buzz method is applied.

force
    The source, as a SDynPy object, for the SPR object.

frfs
    All the FRFs, as a SDynPy `TransferFunctionArray`, for the SPR object.

predicted_response
    The response that is computed by applying the `force` attribute to the `frfs` attribute. It is returned as a SDynPy object.

reconstructed_target_response
    The response that is computed by applying the `force` attribute to the `target_frfs` attribute. It is returned as a SDynPy object.

reconstructed_training_response
    The response that is computed by applying the `force` attribute to the `training_frfs` attribute. It is returned as a SDynPy object.

reconstructed_validation_response
    The response that is computed by applying the `force` attribute to the `validation_frfs` attribute. It is returned as a SDynPy object.

reference_coordinate
    A SDynPy `CoordinateArray` the describes the source DOFs in the SPR object.

reference_transformation
    The transformation that is applied to the `force` attribute and reference DOFs in the `training_frfs`. It is returned as a SDynPy `Matrix` object.

response_coordinate
    A SDynPy `CoordinateArray` the describes the response DOFs in the `frfs` attribute.

response_transformation
    The transformation that is applied to the `training_response` attribute and response DOFs in the `training_frfs`. It is returned as a SDynPy `Matrix` object.

target_frfs
    The frfs that were assigned at SPR object initialization by indexing `frfs` attribute with the `target_response_coordinate` attribute. Note that the data for this attribute is stored as a separate private attribute (from the other FRF attributes) in the object. It is returned as a SDynPy `TransferFunctionArray`.

target_response
    The responses in the SPR object, which have the `target_response_coordinate` as the response DOFs. Note that these responses can be assigned several ways and are stored as a separate private attribute (from the other response attributes) in the object. It is returned as a SDynPy object. 

target_response_coordinate
    The response DOFs in the `target_frfs` and `target_response` attributes. This DOF set is the concatenation of the `training_response_coordinate` and `validation_response_coordinate` attributes. It is returned as a SDynPy `CoordinateArray`.

time_abscissa (`TransientSourcePathReceiver` only)
    The time vector for the time domain data in the SPR object.

time_abscissa_spacing (`TransientSourcePathReceiver` only)
    The sampling time for the time domain data in the SPR object.

training_frfs
    The FRFs in the SPR object that are used for the ISE, which have the `training_response_coordinate` as the response DOFs. These FRFs can be supplied separately from the other FRFs in the SPR object and the data for this attribute is stored as a separate private attribute (from the other FRF attributes) in the object. It is returned as a SDynPy `TransferFunctionArray`.

training_response
    The responses in the SPR object that are used for the ISE, which have the `training_response_coordinate` as the response DOFs. These responses can be supplied separately from the other responses in the SPR object and the data for this attribute is stored as a separate private attribute (from the other response attributes) in the object. It is returned as a SDynPy object.

training_response_coordinate
    The response DOFs for the FRFs and responses in the SPR object that are used for the ISE. It is returned as a SDynPy `CoordinateArray`.

transformed_force
    The source for the SPR object with the `reference_transformation` applied. It is returned as a SDynPy object.

transformed_reconstructed_response
    The `reconstructed_training_response` attribute with the `response_transformation` applied. It is returned as a SDynPy object.

transformed_reference_coordinate
    The source DOFs that the `force` attribute is transformed into after the `reference_transformation` is applied. It is returned as a SDynPy `CoordinateArray`.

transformed_response_coordinate
    The response DOFs that the `training_response` attribute is transformed into after the `response_transformation` is applied. It is returned as a SDynPy `CoordinateArray`.

transformed_training_frfs
    The `training_frf` attribute with the `response_transformation` and `reference_transformation` applied. It is returned as SDynPy `TransferFunctionArray`.

transformed_training_response
    The `training_response` attribute with the `response_transformation` applied. It is returned as a SDynPy object. 

validation_frfs
    The frfs that were assigned at SPR object initialization by indexing `frfs` attribute indexed with the `validation_response_coordinate` attribute. Note that the data for this attribute is stored as a separate private attribute (from the other FRF attributes) in the object. It is returned as a SDynPy `TransferFunctionArray`.

validation_response
    The `target_response` attribute that has been indexed with the `validation_response_coordinate`. It is returned as a SDynPy object.

validation_response_coordinate
    The response DOFs that do not exist in the intersection between `target_response_coordinate` and `training_response_coordinate`.
```

## Required Attributes for Object Initialization
In most cases, SPR objects will need to be initialized with data for at least the FRFs and responses through the `frfs` attribute and either the `target_response` or `training_response` attributes. Additional data can be supplied at SPR at initialization, including the `force`, `training_response_coordinate`, `training_frfs`, `response_transformation`, and `reference_transformation` attributes. A `buzz_cpsd` can also be supplied, but only for the `PowerSourcePathReceiver`.

```{note}
Many of the SPR object attributes will be set to default values or duplicated from other attributes if specific data is not supplied at object initialization. For example, the `frfs` and `training_frfs` attributes will be the same if sufficient data is not supplied to determine a difference between the two. Similarly the `reference_transformation` and `response_transformation` attributes will default to identity if data is not supplied for them at initialization. 
``` 