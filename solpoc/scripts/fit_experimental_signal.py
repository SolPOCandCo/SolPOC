# -*- coding: utf-8 -*-
"""
Created on 2023-11-17
SolPOC v 0.9.6
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from solpoc import *
from datetime import datetime
from multiprocessing import Pool, cpu_count
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
#----------------------------------------------------------------------------#
# %%  Main : You can start to modified something
# Comment to be written in the simulation text file
Comment = "Test of fiting : found the thin layers thcinesses according to a reflectance and transmittance measurement" # Comment to be written in the simulation text file
# Write the thin layer stack, to substrat to ambiant 
Mat_Stack = ["BK7", "SiO2", "TiO2"]
# %% Important parameters
Th_Substrate = 1e6  # Substrate thickness, in nm
Th_range = (0, 300) # Range of thickness (lower bound and upper bound), for the fitt process
Ang = 0  # Incidence angle on the thin layers stack, in °
# %% Signal to fit, in a "RTA" texte file. 
Wl_fit, Signal_fit, name_fit = open_Spec_Signal('Fit/signal_fit.txt', 1) # Read the wavelenght (nm) and reflectance (-), in 0-1 range
Wl_fit, Signal_fit_2, name_fit = open_Spec_Signal('Fit/signal_fit.txt', 2) # Read the wavelenght (nm) and transmittance (-) in 0-1 range
# %% Hyperparameters for optimization methods
# Number used to calculate the budget : nb_generation * pop_size 
nb_generation = 50
nb_run = 8  # Number of run
cpu_used = 8  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU your computer
seed = 42 # Seed of the random number generator. Uncomment for fix the seed
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
"""_________________________________________________________________________"""
"""
Based on few experimented, DEvol (current to best) is useful for "fitting" process. 
Is here an optimization : we reduce the difference between a reflectance and transmittance with an 
experimental reflectance and transmittance (in text file, placed in the Fit folder) 
If cost function near 0 : the difference is few, and the model reflectance is near the real reflectance. 
"""
algo = DEvol  # Name of the optimization method
pop_size = 30  # number of individual per iteration / generation
crossover_rate = 0.5  # crossover rate (1.0 = 100%)
# volumic fraction of inclusion in host matrix, must be included in (0,1)
vf_range = (0.0, 1.0)
# Part of individuals selected to be the progenitors of next generations
evaluate_rate = 0.3
# chance of child gene muted during the birth. /!\ This is Cr for DEvol optimization method
mutation_rate = 0.5
# If a chromose mutates, the value change from random number include between + or - this values
mutation_delta = 15
f1, f2 = 0.9, 0.8  # Hyperparameter for DEvol
# String. Mutaton method for DEvol optimization method
mutation_DE = "current_to_best"
selection = selection_min # Callable. Name of the selection method : here selection_min
evaluate = evaluate_fit_RT # Callable. Name of the cost function, here fit
Wl_Sol, Sol_Spec, name_Sol_Spec = open_SolSpec('Materials/SolSpec.txt', 'DC')# Open the solar spectrum
Wl = Wl_fit
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)  # Interpolate the solar spectrum
# Open and interpol the refractive index
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
# parameters is a dictionary containing the problem variables
# => They then read the necessary variables  with the commande .get
parameters = {'Wl': Wl,  # I store a new variable called "Wl", and I give it Wl's value
              'Ang': Ang,
              'Sol_Spec': Sol_Spec,
              'name_Sol_Spec': name_Sol_Spec,
              'Th_range': Th_range,
              'Th_Substrate': Th_Substrate,
              'Signal_fit' : Signal_fit,
              'Signal_fit_2' : Signal_fit_2,
              'Mat_Stack': Mat_Stack,
              'n_range': (1.3, 3.0),
              'n_Stack': n_Stack,
              'k_Stack': k_Stack,
              'pop_size': pop_size,
              'algo': algo,
              'name_algo': algo.__name__,
              'evaluate': evaluate,
              'selection': selection,
              'name_selection': selection.__name__,
              'mutation_rate': mutation_rate,
              'mutation_delta': mutation_delta,
              'nb_generation': nb_generation, }  # End of the dict

# %%
# If nb_layer exists, then I optimize one or more theoretical thin layers
# I add values to the container (dictionary used to transmit variables)
language = "en"  # can change into fr to write the console information and the files in the folder in French
nb_layer = 0
if 'd_Stack_Opt' not in locals() or len(d_Stack_Opt) == 0:
    d_Stack_Opt = ["no"] * ((len(Mat_Stack) - 1) + nb_layer)
    parameters["d_Stack_Opt"] = d_Stack_Opt
else:
    parameters["d_Stack_Opt"] = d_Stack_Opt
if 'nb_layer' in locals() and nb_layer != 0:
    parameters["nb_layer"] = nb_layer
    parameters["n_range"] = n_range
# if the seed variable exists, i add it in the dictionary
# if not, define a seed

if 'seed' in locals():
    parameters["seed"] = seed
else:
    parameters["seed"] = get_seed_from_randint()
# Create seed list for multiprocessing
parameters['seed_list'] = get_seed_from_randint(
    nb_run,
    rng=np.random.RandomState(parameters['seed']))

if algo.__name__ == "DEvol":
    if 'mutation_DE' not in locals():
        mutation_DE = "current_to_best"

    parameters["mutation_DE"] = mutation_DE
    parameters["f1"] = f1
    parameters["f2"] = f2

if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
    parameters["vf_range"] = vf_range
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
        evaluate, selection, this_run_params)
    # calculate the time used
    t2 = time.time()
    temps = t2 - t1
    # best solution is a stack. Evaluation of this stack
    if type(best_solution) != list:
        best_solution = best_solution.tolist()
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    perf = evaluate(best_solution, parameters)
    if language == "fr":
        print("J'ai fini le cas n°", str(i+1), " en ", "{:.1f}".format(
            temps), " secondes.", " Meilleur : ", "{:.4f}".format(perf), flush=True)
    if language == "en":
        print("I finished case #", str(i+1), " in ", "{:.1f}".format(
            temps), " seconds.", " Best: ", "{:.4f}".format(perf), flush=True)

    return best_solution, perf, dev, n_iter, temps, seed

# %%
# Beginning of the main loop. The code must be in this loop to work in multiprocessing
if __name__ == "__main__":

    if language == "en":
        print("Start of the program")
        launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
        print("Launched at " + launch_time)
        print("Number of detected cores: ", cpu_count())
        print("Number of used cores: ", cpu_used)
    date_time = datetime.now().strftime("%Y-%m-%d-%Hh%M")
    dawn_of_time = time.time()

    # Writing of the backup folder, with current date/time
    directory = date_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    if language == "en":
        print("The folder '" + directory + "' has been created.")

    Mat_Stack_print = Mat_Stack

    if 'nb_layer' in locals():
        for i in range(nb_layer):
            Mat_Stack_print = Mat_Stack_print + ["X"]
    parameters.update({
        "Mat_Stack_print": Mat_Stack_print})

    if language == "en":
        print("The thin layer stack is : ", Mat_Stack_print)

    if language == "en":
        if 'nb_layer' in locals():
            nb_total_layer = len(Mat_Stack) + nb_layer
        else:
            nb_total_layer = len(Mat_Stack)

        print("The total number layer of thin layers is : " + str(nb_total_layer))
    parameters.update({
        "nb_total_layer": nb_total_layer
    })
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
                          'language': language,
                          'name_PV': "none",
                          'name_Th': "none",
                          'name_Sol_Spec': name_Sol_Spec,
                          'launch_time': launch_time,
                          'cpu_used': cpu_used,
                          'nb_run': nb_run, }  # End of the dict
    end_of_time = time.time()
    time_real = end_of_time - dawn_of_time
    parameters.update({"time_real": time_real})
    if language == "fr":
        print("Le temps réel total est de : ",
              "{:.2f}".format(time_real), "secondes")
        print("Le temps réel de calcul processeur est de : ",
              "{:.2f}".format(sum(tab_temps)), "secondes")

    """_____________________Best results datas______________________"""

    Explain_results_fit(parameters, Experience_results)

    """_____________________Write results in a text file_________________"""

    Convergences_txt(parameters, Experience_results, directory)
    Generate_txt(parameters, Experience_results, directory)

    """________________________Plot creation_________________________"""

    Reflectivity_plot_fit(parameters, Experience_results, directory)
    Transmissivity_plot_fit(parameters, Experience_results, directory)
    Convergence_plots_2(parameters, Experience_results, directory)
    Consistency_curve_plot(parameters, Experience_results, directory)
    Optimum_thickness_plot(parameters, Experience_results, directory)
    Volumetric_parts_plot(parameters, Experience_results, directory)
# %%
