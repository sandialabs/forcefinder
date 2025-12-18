# ForceFinder Framework
This section of the documentation focuses on providing basic information for the object oriented framework that is leveraged in ForceFinder. In this framework, all the fundamental information for an ISE problem (FRFs, responses, etc.) are stored in a `SourcePathReceiver` object, which is referred to as an SPR object for short. The contents of this section include information on:

- [SPR Object types](spr_types) - The different SPR object types, which have been implemented for different types of responses
- [Bookkeeping features](automatic_bookkeeping) - Information on the automatic bookkeeping features in ForceFinder, which make it quick and easy to initialize and use an SPR object for an ISE problem
- [Anatomy of an inverse method](inverse_method_code) - General information on the basic code layout for the inverse methods in ForceFinder
- [SPR Attributes](object_attribute_definitions) - Information on the attributes that have been implemented for the SPR objects, so it is simple and quick to access a variety of data and information from the SPR objects 

## Link to SDynPy
The ForceFinder framework is largely built on top of the SDynPy package, similar to how Scikit-learn is built on top of NumPy. This integration makes it easy to use many bookkeeping, data analysis, and plotting features that have already been built into SDynPy. As a result, practitioners have access to a large suite of structural dynamics tools that are not directly included in ForceFinder. However, the practitioner does not need to use any SDynPy functions in a ISE problem, once the SPR object has been created. 