# -*- coding: utf-8 -*-
"""
Created on 2025-09-17
SolPOC v 0.9.6
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
"""
This script performs a fitting procedure to determine the thicknesses of thin-film layers in a 
measured multilayer stack. The goal is to match the experimentally measured reflectance and transmittance 
with the values calculated by the code. Using a adapted cost function (labeled with : _fit),
the algorithm adjusts the layer thicknesses to minimize the difference (error) between the measured and 
calculated optical responses.
"""
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
#----------------------------------------------------------------------------#
# %%  Main : You can start to modified something
# Comment to be written in the simulation text file
Comment = "Test of fiting : found the thin layers thcinesses according to a reflectance and transmittance measurement" # Comment to be written in the simulation text file
# Write the thin layer stack, to substrat to ambiant 
Mat_Stack = ["BK7", "SiO2", "TiO2"]
cost_function = sol.evaluate_fit_RT # Callable. Cost function for the fitting, here R + T are fit
# %% Important parameters
Th_Substrate = 1e6  # Substrate thickness, in nm
Th_range = (0, 300) # Range of thickness (lower bound and upper bound), for the fitt process
Ang = 0  # Incidence angle on the thin layers stack, in °
# %% Signal to fit, in a "RTA" texte file. 
"""
/!\ Example: Here we load an RTA text file located in the "Fit" folder. The file contains experimental data. 
# The first call reads the wavelength (nm) and the reflectance (0–1 range),
# and the second call reads the wavelength (nm) and the transmittance (0–1 range).
"""
Wl_fit, Signal_fit, name_fit = sol.open_Spec_Signal('Fit/signal_fit.txt', 1) # Read the wavelenght (nm) and reflectance (-), in 0-1 range
Wl_fit, Signal_fit_2, name_fit = sol.open_Spec_Signal('Fit/signal_fit.txt', 2) # Read the wavelenght (nm) and transmittance (-) in 0-1 range
# %% Hyperparameters for optimization methods
# Number used to calculate the budget : nb_generation * pop_size 
budget = 2000 # 
nb_run = 8  # Number of run
cpu_used = 8  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU your computer
seed = None # Seed of the random number generator. Uncomment for fix the seed
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
"""_________________________________________________________________________"""
"""
Based on few experimented, DEvol (with "current to best" mutation strategies) is useful for "fitting" process. 
Is here an optimization : we reduce the difference between a reflectance and transmittance with an 
experimental reflectance and transmittance (in text file, placed in the Fit folder) 
If cost function near 0 : the difference is few, and the model reflectance is near the real reflectance. 
"""
algo = sol.DEvol  # Name of the optimization method
mutation_DE = "current_to_best"
pop_size = 30  # number of individual per iteration / generation
crossover_rate = 0.5  # crossover rate (1.0 = 100%)
f1, f2 = 0.9, 0.8  # Hyperparameter for DEvol

# String. Mutaton method for DEvol optimization method

selection = sol.selection_min # Callable. Name of the selection method : here selection_min
cost_function = sol.evaluate_fit_RT # Callable. Name of the cost function, here fit
Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec('Materials/SolSpec.txt', 'DC')# Open the solar spectrum
Wl = Wl_fit
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)  # Interpolate the solar spectrum
# Open and interpol the refractive index
n_Stack, k_Stack = sol. Made_Stack(Mat_Stack, Wl)
# parameters is a dictionary containing the problem variables
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
    Signal_fit = Signal_fit,
    Signal_fit_2 = Signal_fit_2,
    n_Stack = n_Stack,
    k_Stack = k_Stack,
    Th_range = Th_range,
    Th_Substrate = Th_Substrate,
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

# %%
# creation of a function for multiprocessing
def run_problem_solution(i):
    t1 = time.time()  # Time before the optimisation process
    # Line below to be uncommented to slightly desynchronize the cores, if the seed is generated by reading the clock.
    # time.sleep(np.random.random())
    # Create a dictionary for this particular run so we can update the seed with the specific seed for this run
    this_run_params = {}
    this_run_params.update(parameters)
    this_run_params['seed'] = parameters['seed_list'][i]
    # Run the optimisation process (algo), with an evaluate method, a selection method and the parameters dictionary.
    best_solution, dev, n_iter, seed = algo(
        cost_function, selection, this_run_params)
    # calculate the time used
    t2 = time.time()
    temps = t2 - t1
    # best solution is a stack. Evaluation of this stack
    if type(best_solution) != list:
        best_solution = best_solution.tolist()
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    perf = cost_function(best_solution, parameters)
    print("I finished case #", str(i+1), " in ", "{:.1f}".format(
            temps), " seconds.", " Best: ", "{:.4f}".format(perf), flush=True)

    return best_solution, perf, dev, n_iter, temps, seed

# %%
# Beginning of the main loop. The code must be in this loop to work in multiprocessing
if __name__ == "__main__":

    print("Start of the program")
    launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
    print("Launched at " + launch_time)
    print("Number of detected cores: ", cpu_count())
    print("Number of used cores: ", cpu_used)
    date_time = datetime.now().strftime("%Y-%m-%d-%Hh%M")
    dawn_of_time = time.time()
    
    sol.run_main(parameters)
    directory = parameters.get('directory')
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
                          'name_PV': "none",
                          'name_Th': "none",
                          'name_Sol_Spec': name_Sol_Spec,
                          'launch_time': launch_time,
                          'cpu_used': cpu_used,
                          'nb_run': nb_run, }  # End of the dict
    end_of_time = time.time()
    time_real = end_of_time - dawn_of_time
    parameters.update({"time_real": time_real})
    print("The total real time is: ",
              "{:.2f}".format(time_real), "seconds")
    print("The processor calculation real time is: ",
              "{:.2f}".format(sum(tab_temps)), "seconds")

    """_____________________Best results datas______________________"""

    sol.Explain_results_fit(parameters, Experience_results)

    """_____________________Write results in a text file_________________"""

    sol.Convergences_txt(parameters, Experience_results, directory)
    sol.Generate_txt(parameters, Experience_results, directory)

    """________________________Plot creation_________________________"""
    sol.Reflectivity_plot_fit(parameters, Experience_results, directory)
    sol.Transmissivity_plot_fit(parameters, Experience_results, directory)
    sol.Convergence_plots_2(parameters, Experience_results, directory)
    sol.Consistency_curve_plot(parameters, Experience_results, directory)
    sol.Optimum_thickness_plot(parameters, Experience_results, directory)
    sol.Volumetric_parts_plot(parameters, Experience_results, directory)
# %%
