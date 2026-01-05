# Automatic Bookkeeping, Sample Splitting, and Inverse Processing
In most cases, the data for the SPR object is passed to the initializer/constructor function as SDynPy objects, which allows ForceFinder to automatically organize the data. Once in the SPR object, the data is stored as NumPy arrays to reduce overhead and simplify the process for performing computations. As a result, the data is prepared for fundamental ISE operations upon object creation, meaning that the practitioner does not need to consider any bookkeeping operations. 

```{note}
Individual pieces of data (FRFs, response, etc.) are recalled as class attributes for the SPR object, which are returned as SDynPy arrays. For example, `spr_object.frfs` will return a SDynPy `TransferFunctionArray` of the FRFs in the SPR object.
```

(sec:sample_splitting)=
## Sample Splitting
The SPR object initializer function includes methods for splitting the response degrees of freedom (DOFs) into so-called "training" and "validation" DOFs. This allows the practitioner to split the response and FRF data so only the training data is used for the ISE and the validation data is held out for optional quality evaluations. The training and validation DOFs are concatenated to create a superset of "target" response DOFs, which are DOFs in the FRFs that have accompanying response data. The difference between the training and validation DOFs is intuited by the initializer function with one of two ways:

1. The practitioner can supply the target and training response data as separate SDynPy objects. The function will determine the validation DOFs based on the DOFs that are not in the intersection between the target and training data.
2. The user can supply the target response as a single SDynPy object and specify the training response DOFS with a SDynPy `CoordinateArray`. The function will split the supplied `target_response` into the training and validation responses accordingly. The validation DOFs are identified based on the DOFs that are not in the intersection between the target and training `CoordinateArrays`.

```{tip}
Accompanying response data is not required for all the response DOFs in the FRFs, meaning that responses can be predicted at locations where measured `target_response` data is unavailable.
```
```{tip}
The practitioner does not need to explicitly supply separate training and target response data or DOFs. The initializer will assume that the target and training DOFs are the same (i.e., there are not any validation DOFs) if it cannot intuit the sample split with the methods that are described above.
```
```{note}
The training and target data (for either the FRFs or responses) does not need to have the same ordinate, for cases where the data has been processed differently.
```

## Inverse Processing Decorator Functions
A so-called `inverse_processing` decorator function has been applied to every inverse method in ForceFinder (where there are different decorator functions for the different SPR types). These functions handle all the pre/post processing tasks that are common to every inverse method. These tasks include:
- Applying transformations
- Applying the buzz method (for `PowerSourcePathReceiver` objects)
- Applying constant overlap and add (COLA) processing for the `TransientSourcePathReceiver`
```{note}
Optional kwargs exist in the function signature for the inverse methods to enable/disable or modify default parameters for some of the pre/post processing in the `inverse_processing` decorator functions.
```

These `inverse_processing` decorator functions follow the same general process for each SPR type:
1. Collect the FRF and response data from the SPR object and preprocess it for the inverse method
2. Supply the preprocessed FRF and response data to the inverse method
3. Collect the estimated source data from the inverse method and convert it back to a physical quantity (if a transformation was applied)
4. Store the estimated sources (as a physical quantity) to the SPR object

```{note}
The `inverse_processing` decorator functions require that additional kwargs be added to the function signature for the inverse methods, as described in [Anatomy of an Inverse Method](inverse_method_code). Further, the use of the `inverse_processing` decorator functions should be transparent in most basic uses of ForceFinder. However, it is useful to understand the layout of the functions when reviewing code or implementing a new inverse method.
``` 