[![Main Tests](https://github.com/sandialabs/forcefinder/actions/workflows/tests.yml/badge.svg)](https://github.com/sandialabs/forcefinder/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/sandialabs/forcefinder/graph/badge.svg?token=TxYDhM7NCq)](https://codecov.io/gh/sandialabs/forcefinder)

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

The SourcePathReceiver object has also been built into a Rattlesnake control law for random vibration control, meaning that the same exact code can be used in offline predictions and online control during a multiple-input/multiple-output (MIMO) vibration test. Note that the control law is included with this package for completeness, but are not imported to the Python environment when ForceFinder is used.

## Goals
The ForceFinder project has two main goals:

1. Advance the state of the art in ISE via FRF matrix inversion by creating a common framework to implement and review different ISE methods.
2. Make it easier for non-expert practitioners to use advanced ISE tools through the creation of a simple object-oriented framework where ISE methods can be used with a simple method call.

## Support
Please submit any bugs or feature requests into the Github issue tracker.

## Project status
ForceFinder is currently under development and the current version should be treated as an "alpha" release. Breaking changes may be pushed to the develop and main branches without notice. 

## Installation and Usage
ForceFinder is available on PyPi and can be installed with:

```
pip install forcefinder
```

However, the package is under active develop and may see changes to the develop branch, which are not immediately pushed to the main branch (which is what is published to PyPi). Further, the demo and benchmark portions of the package are not published to PyPi. As such, it may be preferable to install ForceFinder from a GIT repository on the users local machine. The process for doing this is:

```
1. Clone the ForceFinder repository to your local machine via "Clone with SSH"
2. PIP install ForceFinder with the following commands (from the command prompt or terminal):
    - cd local_ForceFinder_repository (this is the filepath to the local repository)
    - pip install -e . (this will pip install the whole repository, the -e flag lets python know that the package is a git repo so it will see changes as they are made to the repo)
```

Developers may want to install the package with extra dependencies that run the tests and build the documentation. This can be done by modifying the pip command from the above process to:

```
pip install -e .[dev]
```

Once installed, ForceFinder can be imported with the following alias:

```python3
import forcefinder as ff
```



