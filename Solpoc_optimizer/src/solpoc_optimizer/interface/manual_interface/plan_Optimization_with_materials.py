import numpy as np
import solpoc as sol
from plan_utils import generate_json

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
# ----------------------------------------------------------------------------#

priority = 1

# the thin layer stack.
Mat_Stack = [
    "BK7",
    "SiO2",
    "TiO2",
    "UM",
    "TiO2",
    "UM",
    "UM",
]  # Insert "UM" (Unknown Material) to specify a layer where the material can vary
# Liste of optional material of thin layer mark with "UM"
Mat_Option = ["SiO2", "ZnO", "TiO2"]  # Max actual length is 3.

algo = sol.DEvol  # Name of the optimization method
selection = (
    sol.selection_max
)  # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_R_s  # Callable. Name of the cost function

# %% Important parameters
Th_range = (50, 250)  # in nm.
Th_Substrate = 1e6  # Substrate thickness, in nm
Wl = np.arange(280, 2505, 20)  # np.arange(280, 2505, 5)
Ang = 0  # Incidence angle on the thin layers stack, in °
Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec("Materials/SolSpec.txt", "GT")

# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual in the initial population
# crossover rate (1.0 = 100%) This is Cr for DEvol optimization method
crossover_rate = 0.5
f1 = 1.0  # Hyperparameter for the mutation strategie
mutation_DE = "rand_1"  # Mutation strategie
budget = 1000  # budget, number of iteration
Mode_choose_material = "sigmoid"
seed = None
# seed = 2185585551

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    generate_json(
        locals(), template_name="Optimization with Materials", priority=priority
    )
