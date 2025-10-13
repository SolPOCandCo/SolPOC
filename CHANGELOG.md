# Changelog
## v0.9.7
This update was mainly carried out to address reviewers’ comments for the publication of a research article.

### 1. Package structure and management changes

- Added the script `basic_example_solpoc` to illustrate a simple example of how to use the package.

- Added two scripts comparing SolPOC with SciPy and Nevergrad against TMM (another optical package):
  1. `SolPOC_and_tmm_with_nevergrad.py`
  2. `SolPOC_and_tmm_with_scipy.py`

- Introduced a new variable `budget` to define the optimization method’s iteration budget. 
Previously, the `budget` was computed as `pop_size * nb_generation`. This approach, inherited from older code versions, was inaccurate and confusing.

- In `DEvol`, the crossover rate (Cr) was previously defined by the variable `mutation_rate`, which was incorrect and misleading.
DEvol now uses `crossover_rate` to define Cr, making the code more explicit.

- Removed result folders (code outputs ; when run an Template scripts, as an Template SolPOC create a folder with the results, txt files, png etc) in the Examples folder from the package to reduce clutter. The code output is now available through a [Zenodo repository](https://zenodo.org/records/17340621).

- Renamed "tutorial" scripts to "Template" to clarify that they are intended as usage examples only.
Similarly, renamed "optimisation_multiprocessing" scripts to "template_optimization" to:emphasize that these are example templates, not mandatory usage; reduce focus on the multiprocessing aspect.

- Replaced "from solpoc import *" with "import solpoc as sol" in all scripts for cleaner and safer imports.
This change is reverberate on the different scripts 

- Added the function get_parameters_stack to handle and auto-fill the parameters dictionary. This made several scripts lighter and easier to understand.

- Added the function run_main to manage the parameter dictionary, reducing redundancy in automation scripts.

- In the "optimization" Template, the function Simulation_amont_aval_txt (used for result processing) is now commented out by default.
It is no longer executed automatically. This function performs specific calculations related to solar flux distribution after spectral splitting.
This prevents standard users from generating overly specific outputs. However, it remains useful for ongoing research projects involving non-programmer users, who can easily uncomment the line to restore this functionality.

- The seed value can now be set to `None`, which is more intuitive for some users.
Previously, it had to be either commented out or explicitly assigned.

### 2. Added and modified functions

- Added a "cost" function for optimizing spectrally selective stacks using a TRT profile.

- Added a `get_parameters` function to manage and auto-fill the parameters dictionary.

- Added functions for material optimization in layer stacks:  
	1. `choose_materials`, `choose_material2`, `choose_materials3`  
	2. `print_material_probability`  
	3. added material "UM" to the material list  
	4. updated `individual_to_stack` and `Made_Stack` functions  

- Improved DEvol readability by adding sub-functions:  
	1. `apply_mutation_DE` to handle mutation strategies  
	2. `x_DEvol` to initialize the DEvol population  


## v0.9.6

Pinning the Scipy dependency at >=1.14.0 breaks support for python 3.9. Code was adjusted to be flexible around the `trapz` or `trapezoid` change, so that 3.9 can still be supported.

## v0.9.5
### Dependencies
- The package `solcore` is now a new dependency, in `function_SolPOC.py`: `from solcore.structure import Structure` and `from solcore.absorption_calculator import calculate_rat`
- Now SciPy is required to be at least [v1.14.0](https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html). The function `scipy.integrate.trapz` has since been renamed to `trapezoid`. Function calls have been renamed accordingly throughout the package. 

### New functions
- Created a new function, `RTA_curve_inco` to add the case of incoherency in thin layers stacks

- Created two different functions for `RTA_curve_inco`:
	1. `Stack_coherency`, which returns a boolean and a list to determine if a stack contains at least one incoherent thin layer
	2. `Made_SolCORE_Stack`, which creates the stack for the Solcore package, using incoherency. The stack is of type: `solcore.structure.Structure`.

- Created the Generate_perf_rh_txt function to write heliothermal efficiency rH, the solar absorptance AS, and the thermal emissivity E_BB of a spectral selective coating for thermal absorbers into a text file.

Here’s how the description of the added functions could be worded, with proper corrections and consistency:

- Added functions for the fitting process to calculate the differences between the model RTA and a signal (e.g., measured reflectivity):
	1. `evaluate_fit_R`: Calculates the difference between model reflectance and measured reflectance.
	2. `evaluate_fit_T`: Calculates the difference between model transmittance and measured transmittance.
	3. `evaluate_fit_T2face`: Calculates the difference between model transmittance and measured transmittance, considering the second face of the substrate.
	4. `evaluate_fit_RT`: Calculates the difference between `(model reflectance + transmittance) / 2` and `(measured reflectance + transmittance) / 2`.
	5. Added two functions to calculate the differences between two curves:
		1. `chi_square`: Calculates the chi-square difference between two curves.
		2. `normalized_mse`: Calculates the normalized mean squared error (MSE) between two curves.

### Changes to existing functions
- In the `RTA` function,
Use the new version of the `RTA` function
	- --> Reduces the calculation time (x2 - 4 time faster with 0.9.4 version)

- In the `Bruggeman` function,
Use the new version of the `Bruggmann` function, now vectorized
	- --> Strongly reduces the calculation time (x80 - 140 time faster with 0.9.4 version)

- In the `Generate_materials_txt` function,
  - Correct a bug in the `Generate_materials_txt` function,
	- --> `vf = Experience_results.get("vf") is null, vf is not present in Experience_results. Use vf = individual[len(Mat_Stack):,] instead`

- Removed the plot of the optical stack response, which was named "`OpticalStackRespond_plot`". This was an orthographical mistake, and this plot was already plotted

- In the `OpticalStackResponse_plot` function,
Renamed the plot file title
	- --> `OpticalStackResponse.png` becomes `Optical_Stack_Response.png`

- In the `Optimum_thickness_plot`,
Correct an orthographic mistake in the plot xlabel for the word "substrate"
	- --> Now "Number of layers, substrate to air"

- In the `Convergence_plots` function,
Divided the previous function in two : Convergence_plots and Convergence_plots_2
	- --> The graphic is now show with percentage of budget used, instead of the number of iteration
	- --> The new graphics are more readable

### Miscelaneous
- Modified folder creation to avoid having two folders with the same name.
	--> If two folders have the same name, they are now named -2, -3, etc.
- Add `y=1.05` as an argument in all plt.title. Example : `plt.title("Convergence Plots")` becomes `plt.title("Convergence Plots", y=1.05)`
	--> the title moved up for a better readability
