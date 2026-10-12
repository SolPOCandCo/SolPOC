import numpy as np
import solpoc as sol
from plan_utils import generate_json

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
# ----------------------------------------------------------------------------#

priority = 1

# Comment to be written in the simulation text file
Comment = "Tutorial : selective coating for solar thermal systems, like CSP"
Mat_Stack = ["Fe", "W", "W-Al2O3", "Al2O3"]

# Choice of optimisation method
algo = sol.DEvol  # Callable. Name of the optimization method, callable
selection = (
    sol.selection_max
)  # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_rh  # Callable. Name of the cost function

# %% Important parameters
# Wavelength domain, here from 280 to 30µm with a 5 nm step 280-2.5µm, 2.5µm -> 30µm : 50 nm. Can be change!
Wl = sol.Wl_selectif()
# Open the solar spectrum
Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec("Materials/SolSpec.txt", "DC")
# Thickness of the substrate, in nm
Th_Substrate = 1e6  # Substrate thickness, in nm
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 300)  # in nm.
# volumic fraction of inclusion in host matrix, must be included in (0,1)
vf_range = (0, 1.0)
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0  # Incidence angle on the thin layers stack, in °

# %% Optional parameters, necessary for some cost function
C = 80  # Solar concentration. Data necessary for solar thermal application, like selective stack
T_air = (
    20 + 273
)  # Air temperature, in Kelvin. Data necessary for solar thermal application, like selective stack
T_abs = (
    300 + 273
)  # Thermal absorber temperature, in Kelvin. Data necessary for solar thermal application, like selective stack

# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual per iteration / generation
# crossover rate (1.0 = 100%) This is Cr for DEvol optimization method
crossover_rate = 0.5
f1, f2 = 0.9, 0.8  # Hyperparameter for mutation in DE
mutation_DE = "current_to_best"  # String. Mutaton method for DE optimization method

# %% Hyperparameters for optimisation methods
# Number of iteration.
budget = 500
nb_run = 4  # Number of run, the number of time were the probleme is solved
cpu_used = 4  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU your computer
seed = None  # Seed of the random number generator. Remplace None for use-it

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    generate_json(locals(), template_name="Selective Coating", priority=priority)
