# Welcome to our Examples folder
This folder contains example scripts illustrating the use of the SolPOC package.
The scripts can be run directly after installing the package.
They allow:
1.	A better understanding of the package functionalities.
2.	A comparison of the optical calculations of SolPOC with a reference package (tmm-fast).
3.	Examples of using SolPOC in combination with optimization packages (SciPy and Nevergrad).
4.	Pre-configured scripts (template) corresponding to different use cases. 
5.	A curve-fitting script for comparison with an experimental signal.

These practical cases are useful for the SolPOC community and correspond to the examples presented in the User Guide.

## Description of the scripts
### A. `basic_example_solpoc`
This script shows how to use the basic functions of SolPOC.
It guides the user through the calculation of the reflectivity, transmissivity, and absorptivity of a four-layer thin film stack: Al2O3, Al, Al2O3, and SiO2 deposited on BK7 glass. This basic example is similar to the first chapters of the Jupyter Notebook.

#### B. `SolPOC_and_tmm_with_scipy`
This script illustrates how to use SolPOC with SciPy (tested with version 1.14.1) for thickness optimization, using the SciPy minimize function with the BFGS method.
Example provided: optimization of the thicknesses of three thin layers forming an anti-reflective coating on glass.

The glass is assumed to have a refractive index of n = 1.5. The three thin layers have refractive indices n = 1.7, n = 1.47, and n = 1.37, respectively.
All layers are considered perfectly transparent (k = 0).
The optical results and execution time are compared with the [tmm package version 0.2.1, available on GitHub](https://github.com/MLResearchAtOSRAM/tmm_fast/tree/v0.2.1)

Note: with this version of tmm, functions must be imported from the `tmm_fast.py` module file.

### C. `SolPOC_and_tmm_with_Nevergrad`
This script shows how to use SolPOC to define a cost function usable with Nevergrad, a black-box optimization package.
The same cost function was implemented with tmm (v0.2.1) and SolPOC to compare results for solving optical equations.

Note: to ensure similar comparison conditions between the two packages, the solar spectrum used by the cost function is loaded directly from the file Sol_Spec.npy.
### D. `template_XX`
All scripts started with `template_`... are various template demanded by the SolPOC user community.
They can act as tutorial for show how a specific probleme can be handle
These pre-configured scripts illustrate the use of SolPOC to optimize stacks for different use cases.
They intend to help non-coder users to apply SolPOC for some specific optimization problems.
Several examples specific to the solar thermal community are provided: optimization of an anti-reflective coating, optimization of a thermal absorber, etc.

### E. `fit_experimental_signal`
This script was requested by users of the SolPOC community. It allows performing inverse engineering: it enables the identification of the thicknesses of different thin layers in a real stack based on experimental measurements of its reflectivity and/or transmissivity (a process often called fitting).

The script is based on the `optimization_multiprocess.py` template script, just adding a new cost function for the fitting process (already include in the `solpoc` package), and reading the experimental signal from a text file (assuming the signal is similar to a RTA file : wavelength in nanometers, and 3 colonum for R, T and A).
By default, the text file must be placed into a folder named `Fit`