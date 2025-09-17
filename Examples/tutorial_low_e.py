# -*- coding: utf-8 -*-
"""
Created on 2025-09-11
SolPOC v 0.9.7
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import solpoc as sol
from datetime import datetime
from multiprocessing import Pool, cpu_count
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
#----------------------------------------------------------------------------#
# %%  Main : You can start to modified something
# Comment to be written in the simulation text file
Comment = "A low emissivity coating for building"
Mat_Stack = ["BK7", "Si3N4", "ZnO", "Ag", "ZnO", "Si3N4"]
# Choice of optimisation method
algo = sol.DEvol  # Callable. Name of the optimization method, callable
selection = sol.selection_max # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_low_e # Callable. Name of the cost function
# %% Important parameters
# Wavelength domain, here from 280 to 2500 nm with a 5 nm step. Can be change!
Wl = np.arange(280,1505,5) #np.arange(280, 2505, 5)
# Open the solar spectrum
Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec('Materials/SolSpec.txt', 'GT')
# Thickness of the substrate, in nm
Th_Substrate = 1e6  # Substrate thickness, in nm
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 200)  # in nm.
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0  # Incidence angle on the thin layers stack, in °
# Allows fixing the thickness of a layer that will not be optimized.
d_Stack_Opt = ["no", "no", 10, "no", "no"]# Set to "no" to leave it unset. For example, if there are three layers, it can be written ["no",40,"no"]. The code understands that only the middle layer is fixed
# Cuting Wavelenght. Data necessary for low-e, 
Lambda_cut_1 = 800 # nm
# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual per iteration / generation
crossover_rate = 0.5  # crossover rate (1.0 = 100%) This is Cr for DEvol optimization method
f1, f2 = 0.9, 0.8  # Hyperparameter for mutation in DE
mutation_DE = "current_to_best" # String. Mutaton method for DE optimization method
# %% Hyperparameters for optimisation methods
# Number of iteration. 
budget = 3000
nb_run = 8  # Number of run, the number of time were the probleme is solved
cpu_used = 8  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU your computer
seed = None # Seed of the random number generator. Remplace None for use-it 
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
"""_________________________________________________________________________"""
# Open and interpol the refractive index
n_Stack, k_Stack = sol.Made_Stack(Mat_Stack, Wl)
# Open and processing the reflectif index of materials used in the stack (Read the texte files in Materials/ )
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)  # Interpolate the solar spectrum
# parameters is a dictionary containing the problem variables
# This dictionary is given as input to certain functions
# => They then read the necessary variables  with the commande .get

parameters = sol.get_parameters(
    Mat_Stack = Mat_Stack,
    algo = algo,
    cost_function = cost_function,
    selection = selection,  
    Wl = Wl,
    Ang = Ang,
    Sol_Spec = Sol_Spec,
    name_Sol_Spec=name_Sol_Spec,
    n_Stack = n_Stack,
    k_Stack = k_Stack,
    Th_range = Th_range,
    Th_Substrate = Th_Substrate,
    d_Stack_Opt = d_Stack_Opt,
    Lambda_cut_1 = Lambda_cut_1,
    pop_size=pop_size,
    f1 = f1,
    f2 = f2, 
    mutation_DE = mutation_DE,
    crossover_rate = crossover_rate,
    budget = budget,
    nb_run = nb_run,
    seed  = seed,
)

# %%
# creation of a function for multiprocessing
def run_problem_solution(i):
    start_time = time.time()  # Time before the optimisation process
    # Create a dictionary for this particular run so we can update the seed with the specific seed for this run
    this_run_params = {}
    this_run_params.update(parameters)
    this_run_params['seed'] = parameters['seed_list'][i]
    # Run the optimisation process (algo), with an evaluate method, a selection method and the parameters dictionary.
    best_solution, dev, n_iter, seed = algo(cost_function, selection, this_run_params)
    # calculate the time used
    elapsed_time = time.time() - start_time
    # best solution is a stack. Evaluation of this stack
    if type(best_solution) != list:
        best_solution = best_solution.tolist()
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    perf = cost_function(best_solution, parameters)

    print("I finished case #", str(i+1), " in ", "{:.1f}".format(
            elapsed_time), " seconds.", " Best: ", "{:.4f}".format(perf), flush=True)

    return best_solution, perf, dev, n_iter, elapsed_time, seed

# %%
# Beginning of the main loop. The code must be in this loop to work in multiprocessing
if __name__ == "__main__":
    print("Start of the program")
    launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
    print("Launched at " + launch_time)
    print("Number of detected cores (logical/virtual, not necessarily physical): ", cpu_count())
    print("Number of used cores: ", cpu_used)
    
    sol.run_main(parameters)
    directory = parameters.get('directory')

    # creation of a pool
    mp_pool = Pool(cpu_used)
    # Solving each problem in the pool using multiprocessing
    results = mp_pool.map(run_problem_solution, range(nb_run))

    # Creation of empty lists, for use them later
    tab_best_solution = []
    tab_dev = []
    tab_perf = []
    tab_n_iter = []
    tab_temps = []
    tab_seed = []
    # sleep 1 ms
    time.sleep(1)
    # result is an array containing the solutions returned by my.pool. I extract them and place them in different arrays
    for i, (best_solution, perf, dev, n_iter, temps, seed) in enumerate(results):
        # I add the values to the tables
        tab_perf.append(perf)
        tab_dev.append(dev)
        tab_best_solution.append(best_solution)
        tab_n_iter.append(n_iter)
        tab_temps.append(temps)
        tab_seed.append(seed)

    Experience_results = {'tab_perf': tab_perf,
                          'tab_dev': tab_dev,
                          'tab_best_solution': tab_best_solution,
                          'tab_n_iter': tab_n_iter,
                          'tab_temps': tab_temps,
                          'tab_seed': tab_seed,
                          'Comment': Comment,
                          'language':'en',
                          'name_Sol_Spec': name_Sol_Spec,
                          'launch_time': launch_time,
                          'cpu_used': cpu_used,
                          'nb_run': nb_run, }  # End of the dict
    end_of_time = time.time()
    time_real = end_of_time - parameters["dawn_of_time"]
    parameters.update({"time_real": time_real})

    
    print("The total real time is: ",
              "{:.2f}".format(time_real), "seconds")
    print("The processor calculation real time is: ",
              "{:.2f}".format(sum(tab_temps)), "seconds")
    """_____________________Best results datas______________________"""

    sol.Explain_results(parameters, Experience_results)

    """_____________________Write results in a text file_________________"""
    
    sol.Convergences_txt(parameters, Experience_results, directory)
    sol.Generate_txt(parameters, Experience_results, directory)
    sol.Optimization_txt(parameters, Experience_results, directory)
    #sol.Simulation_amont_aval_txt(parameters, Experience_results, directory)
    sol.Generate_materials_txt(parameters, Experience_results, directory)

    """________________________Plot creation_________________________"""

    sol.Reflectivity_plot(parameters, Experience_results, directory)
    sol.Transmissivity_plot(parameters, Experience_results, directory)
    sol.OpticalStackResponse_plot(parameters, Experience_results, directory)
    sol.Convergence_plots(parameters, Experience_results, directory)
    sol.Convergence_plots_2(parameters, Experience_results, directory)
    sol.Consistency_curve_plot(parameters, Experience_results, directory)
    sol.Optimum_thickness_plot(parameters, Experience_results, directory)
    sol.Optimum_refractive_index_plot(parameters, Experience_results, directory)
    sol.Volumetric_parts_plot(parameters, Experience_results, directory)
    sol.Stack_plot(parameters, Experience_results, directory)

# %%
