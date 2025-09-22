from setuptools import setup, find_packages

setup(
    name="forcefinder",
    version="0.1.0",
    author="Steven Carter",
    author_email="spcarte@sandia.gov",
    description="An advanced inverse source estimation package for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://cee-gitlab.sandia.gov/spcarte/forcefinder",
    packages=find_packages(where='src'),  # Tell setuptools to look in the src/ folder
    package_dir={"": "src"},  # Set the base folder for the packages to src/
    classifiers=["Programming Language :: Python :: 3.11",
                 "License :: Other/Proprietary License",
                 'Natural Language :: English',
                 'Operating System :: Microsoft :: Windows :: Windows 10'
                 'Operating System :: Microsoft :: Windows :: Windows 11'
                 'Operating System :: MacOS :: MacOS X'
                 ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.10.0",
        "sdynpy>=0.14.0",
        "joblib>=1.3.0",
        "matplotlib>=3.5.0",
        "pyqt5"
        ],
)
