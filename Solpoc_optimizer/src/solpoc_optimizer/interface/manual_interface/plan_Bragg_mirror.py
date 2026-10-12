import numpy as np
import solpoc as sol
from plan_utils import generate_json

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
# ----------------------------------------------------------------------------#

priority = 1

# Comment to be written in the simulation text file
Comment = "A 4 periodic layers of Bragg mirror, deposited on 1mm BK7 glass"
Mat_Stack = ["BK7", "SiO2", "TiO2", "SiO2", "TiO2", "SiO2", "TiO2", "SiO2", "TiO2"]
# or we can use : Mat_Stack = sol.write_stack_period(["BK7"], ["SiO2", "TiO2"], 4)

# Choice of optimisation method
algo = sol.DEvol  # Callable. Name of the optimization method, callable
selection = (
    sol.selection_max
)  # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_R_Brg  # Callable. Name of the cost function

# %% Important parameters
# Wavelength domain, here from 280 to 2500 nm with a 5 nm step. Can be change!
Wl = np.arange(400, 800, 5)  # np.arange(280, 2505, 5)
# Thickness of the substrate, in nm
Th_Substrate = 1e6  # Substrate thickness, in nm
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 200)  # in nm.
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0  # Incidence angle on the thin layers stack, in °

# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual per iteration / generation
# crossover rate (1.0 = 100%) This is Cr for DEvol optimization method
crossover_rate = 0.5
f1, f2 = 0.9, 0.8  # Hyperparameter for mutation in DE
mutation_DE = "current_to_best"  # String. Mutaton method for DE optimization method

# %% Hyperparameters for optimisation methods
# Number of iteration.
budget = 2000
nb_run = 8  # Number of run, the number of time were the probleme is solved
seed = 2905804230  # Seed of the random number generator. Remplace None for use-it

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    generate_json(locals(), template_name="Bragg Mirror", priority=priority)
