import numpy as np
import solpoc as sol
from plan_utils import generate_json

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
# ----------------------------------------------------------------------------#

priority = 1

# Comment to be written in the simulation text file
Comment = "Tutorial : anti-reflective coating for Si PV-Cell"
Mat_Stack = ["Si", "TiO2", "ZnO", "Al2O3"]

# Choice of optimisation method
algo = sol.DEvol  # Callable. Name of the optimization method, callable
selection = (
    sol.selection_max
)  # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_A_pv  # Callable. Name of the cost function

# %% Important parameters
# Wavelength domain, here from 280 to 2500 nm with a 5 nm step. Can be change!
Wl = np.arange(280, 2505, 5)  # np.arange(280, 2505, 5)
# Open the solar spectrum
Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec("Materials/SolSpec.txt", "GT")
# Thickness of the substrate, in nm
Th_Substrate = 1e6  # Substrate thickness, in nm
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 300)  # in nm.
# volumic fraction of inclusion in host matrix, must be included in (0,1)
vf_range = (0, 1.0)
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0  # Incidence angle on the thin layers stack, in °
Wl_PV, Signal_PV, name_PV = sol.open_Spec_Signal("Materials/PV_cells.txt", 1)

# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual per iteration / generation
# crossover rate (1.0 = 100%) This is Cr for DEvol optimization method
crossover_rate = 0.5
f1 = 1.0  # Hyperparameter for mutation in DE
mutation_DE = "rand_1"  # String. Mutaton method for DE optimization method

# %% Hyperparameters for optimisation methods
# Number of iteration.
budget = 1500
nb_run = 8  # Number of run, the number of time were the probleme is solved
cpu_used = 8  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU your computer
seed = None  # Seed of the random number generator. Remplace None for use-it

# ----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    generate_json(locals(), template_name="PV Cells", priority=priority)
