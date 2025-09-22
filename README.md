![ForceFinder Logo](/logo/ForceFinder_Logo.png)
# ForceFinder Inverse Source Estimation
This repository houses ForceFinder, which is a Python project that extends SDynPy (Structural Dynamics Python Libraries) with a comprehensive tool for inverse source estimation (ISE) tasks via frequency response function (FRF) matrix inversion. ForceFinder leverages an object oriented framework, where all the components of the ISE problem (e.g., the FRFs, responses, transformations, etc.) are stored in a so-called "SourcePathReceiver" object.

The main features of ForceFinder are:
- Automated bookkeeping
    - Organizes FRF and response DOFs to be consistent for the source estimate
    - Manages sample splitting for training and validation response DOFs
    - Organizes and applies response and reference transformation matrices, as appropriate
    - Ensures consistent abscissa quantities
    - Etc.
- Automatic recording keeping for the meta-data related to the inverse problem
- Automatic regularization methods to mitigate overfitting 

The SourcePathReceiver object has also been built into a Rattlesnake control law for both random and transient vibration control, meaning that the same exact code can be used in offline predictions and online control during a multiple-input/multiple-output (MIMO) vibration test. Note that the control laws are included in this package for completeness, but are not imported to the Python environment when ForceFinder is used.

## Goals
The ForceFinder project has two main goals:

1. Advance the state of the art in ISE via FRF matrix inversion by creating a common framework to implement and review different ISE methods.
2. Make it easier for non-expert practitioners to use advanced ISE tools through the creation of a simple object-oriented framework where ISE methods can be used with a simple method call.

## Support
Please submit any bugs or feature requests into the Github issue tracker.

## Project status
ForceFinder is currently under development and the current version should be treated as an "alpha" release. Breaking changes may be pushed to the develop branch without notice. 

## Installation and Usage
It is suggested that ForceFinder be installed from a GIT repository on the users local machine because the package is in active development may see changes over the coming months. The process for doing this is:

1. Clone the ForceFinder repository to your local machine via "Clone with SSH"
2. PIP install ForceFinder with the following commands (from the command prompt or terminal):
    - cd local_ForceFinder_repository (this is the filepath to the local repository)
    - pip install -e . (this will pip install the whole repository, the -e flag lets python know that the package is a git repo so it will see changes as they are made to the repo)
3. Use ForceFinder with the following import "import forcefinder as ff" 

