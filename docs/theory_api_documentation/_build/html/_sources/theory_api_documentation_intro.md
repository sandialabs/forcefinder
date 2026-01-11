# ForceFinder Theory and Programming Interface Documentation
```{figure} images/ForceFinder_Logo.png
:alt: ForceFinder Logo
:width: 300px
:align: center
```

ForceFinder is a Python project, which attempts to be a comprehensive tool for inverse source estimation (ISE) via frequency response function (FRF) matrix inversion. It is intended for use in force reconstruction, transfer path analysis (TPA), and multiple input / multiple output (MIMO) vibration control. ForceFinder leverages an object oriented framework, where all the components of the ISE problem (e.g., FRFs, responses, transformations, etc.) are stored in a single `SourcePathReceiver` (SPR) object. The main features of ForceFinder include:
- Automated bookkeeping
    - Organizes FRF and response DOFs to be consistent for the source estimation
    - Manages sample splitting for training and validation response DOFs
    - Organizes and applies response and reference transformation matrices, as appropriate
    - Ensures consistent abscissa quantities
    - Etc.
- Automatic record keeping for the meta-data that is related to the ISE problem
- Simple access to advanced inverse methods 

This guide attempts to provide in-depth documentation for the theory and programming interface that is used in ForceFinder. An end-end reading will prime the practitioner for in-depth use of the package. However, it may be useful to jump to the examples to quickly get up and running.

```{note}
ForceFinder is developed by ISE practitioners, for ISE practitioners. Attempts are made to provide theory documentation for all the inverse methods that exist in ForceFinder. However, functional and clean code is prioritized over theory documentation. Consequently, some of the methods might not be described here, depending on resource availability to generate the documentation. Please contact one of the ForceFinder developers if you have any questions about the theory for one of the inverse methods.
```

## Prerequisites 
ForceFinder leverages many of the SDynPy objects for the automated bookkeeping. As such, a basic familiarity of SDynPy is suggested to be proficient with the ForceFinder. While some of the examples include basic uses of SDynPy, in depth explanation of the SDynPy functions are not the focus of this documentation. See the [SDynPy Github](https://github.com/sandialabs/sdynpy) for more details on how to use SDynPy. 