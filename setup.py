from setuptools import setup, find_packages

setup(
    name="forcefinder",
    version="0.1.1",
    author="Steven Carter",
    author_email="spcarte@sandia.gov",
    description="An advanced inverse source estimation package for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/forcefinder",
    packages=find_packages(where='src'),  # Tell setuptools to look in the src/ folder
    package_dir={"": "src"},  # Set the base folder for the packages to src/
    classifiers=["Programming Language :: Python :: 3.11",
                 "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                 "Natural Language :: English",
                 "Operating System :: Microsoft :: Windows :: Windows 10",
                 "Operating System :: Microsoft :: Windows :: Windows 11",
                 "Operating System :: MacOS :: MacOS X"
                 ],
    python_requires='>=3.9',
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.10.0",
        "sdynpy>=0.14.0",
        "joblib>=1.3.0",
        "matplotlib>=3.5.0",
        "cvxpy>=1.7",
        "pyqt5"
        ],
    extras_require={ # Extra packages that are useful for development
        "dev": [
            "sdynpy>=0.18.6",
            "pytest",
            "pytest-cov",
            "jupyter-book",
            "sphinx-autoapi"
        ]
    }
)


