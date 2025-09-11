# Changelog
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
