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
Comment = "A sentence to be written in the final text file"
Mat_Stack = ["BK7", "TiO2", "SiO2"]
# Choice of optimisation method
algo = DEvol  # Name of the optimization method
# Callable. Name of the selection method : selection_max or selection_min
selection = selection_max
evaluate = evaluate_T_s # Callable. Name of the cost function
# %% Important parameters
# Wavelength domain, here from 320 to 2500 nm with a 5 nm step. Can be change!
# Wl_selectif() #/!\ Last value is not included in the array
Wl = np.arange(280,2505,5) #np.arange(280, 2505, 5)
# Thickness of the substrate, in nm
Th_Substrate = 1e6  # Substrate thickness, in nm
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 300)  # in nm.
# Range of refractive index (lower bound and upper bound), for the optimisation process
n_range = (1.3, 3.0)
# Range of volumic fraction (lower bound and upper bound), for the optimisation process
# volumic fraction of inclusion in host matrix, must be included in (0,1)
vf_range = (0, 1.0)
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0  # Incidence angle on the thin layers stack, in °
# %% Optional parameters
C = 80  # Solar concentration. Data necessary for solar thermal application, like selective stack
T_air = 20 + 273  # Air temperature, in Kelvin. Data necessary for solar thermal application, like selective stack
T_abs = 300 + 273  # Thermal absorber temperature, in Kelvin. Data necessary for solar thermal application, like selective stack
# Cuting Wavelenght. Data necessary for low-e, RTR or PV_CSP evaluates functions
Lambda_cut_1 = 500  # nm
Lambda_cut_2 = 1000  # nm
# Addition of theoretical thin layers with the variable nb_layer, whose thickness AND index must be optimized.
# Number of theoretical thin layers above the stack. This variable can be left undefined.
nb_layer = 0
# Allows fixing the thickness of a layer that will not be optimized.
d_Stack_Opt = []  # Set to "no" to leave it unset. For example, if there are three layers, it can be written ["no",40,"no"]. The code understands that only the middle layer is fixed
# Open the solar spectrum
Wl_Sol, Sol_Spec, name_Sol_Spec = open_SolSpec('Materials/SolSpec.txt', 'GT')
# Open a file with PV cell shape
Wl_PV, Signal_PV, name_PV = open_Spec_Signal('Materials/PV_cells.txt', 1)
# Open a file with thermal absorber shape
Wl_Th, Signal_Th, name_Th = open_Spec_Signal('Materials/Thermal_absorber.txt', 1)
# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual per iteration / generation
crossover_rate = 0.5  # crossover rate (1.0 = 100%)
# Part of individuals selected to be the progenitors of next generations
evaluate_rate = 0.3
# chance of child gene muted during the birth. /!\ This is Cr for DEvol optimization method
mutation_rate = 0.5
# If a chromose mutates, the value change from random number include between + or - this values
mutation_delta = 15
f1, f2 = 0.9, 0.8  # Hyperparameter for DEvol
# String. Mutaton method for DEvol optimization method
mutation_DE = "current_to_best"
# Number of generation/iteration. For DEvol is also used to calculate the budget (nb_generation * pop_size)
nb_generation = 50
# accurency for stop the optimisation processs for some optimization method, as optimiza_agn or strangle
precision_AlgoG = 1e-5
nb_run = 8  # Number of run
cpu_used = 8  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU your computer
#­seed = 415  # Seed of the random number generator. Uncomment for fix the seed
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
"""_________________________________________________________________________"""
# Open and interpol the refractive index
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
# Open and processing the reflectif index of materials used in the stack (Read the texte files in Materials/ )
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)  # Interpolate the solar spectrum
Signal_PV = np.interp(Wl, Wl_PV, Signal_PV)  # Interpolate the signal
Signal_Th = np.interp(Wl, Wl_Th, Signal_Th)  # Interpolate the signal
# parameters is a dictionary containing the problem variables
# This dictionary is given as input to certain functions
# => They then read the necessary variables  with the commande .get
parameters = {'Wl': Wl,  # I store a new variable called "Wl", and I give it Wl's value
              'Ang': Ang,
              'C': C,
              'T_abs': T_abs,
              'T_air': T_air,
              'Sol_Spec': Sol_Spec,
              'name_Sol_Spec': name_Sol_Spec,
              'Th_range': Th_range,
              'Th_Substrate': Th_Substrate,
              'Signal_PV': Signal_PV,
              'Signal_Th': Signal_Th,
              'Mat_Stack': Mat_Stack,
              'Lambda_cut_1': Lambda_cut_1,
              'Lambda_cut_2': Lambda_cut_2,
              'n_range': n_range,
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
# If I optimized an antireflective coating for PV, I need the PV signal shape
if evaluate.__name__ == "evaluate_T_pv" or evaluate.__name__ == "evaluate_A_pv":
    parameters["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec

if evaluate.__name__ == "evaluate_T_vis":
    # Open a file with Human eye response
    # eye is written fully for not misunderstood with the e for emissivity
    Wl_H_eye, Signal_H_eye, name_H_eye = open_Spec_Signal(
        'Materials/Human_eye.txt', 1)
    # Interpolate the signal
    Signal_H_eye = np.interp(Wl, Wl_H_eye, Signal_H_eye)

    Sol_Spec = Signal_H_eye
    parameters["Sol_Spec"] = Sol_Spec
    parameters["Sol_Spec_with_Human_eye"] = Signal_H_eye

# If I optimized an antireflective coating for PV, I need the PV signal shape
if evaluate.__name__ == "evaluate_rh":
    if 'C' in locals():
        parameters["C"] = C
    else:
        parameters["C"] = 80
    if 'T_air' in locals():
        parameters["T_air"] = T_air
    else:
        parameters["T_air"] = 293
    if 'T_abs' in locals():
        parameters["T_abs"] = T_abs
    else:
        parameters["T_abs"] = 300 + 273

# Optimize a PV/CSP coating not with a RTR shape, but with a net energy balance
if evaluate.__name__ == "evaluate_netW_PV_CSP":
    if 'poids_PV' in locals():
        parameters['poids_PV'] = poids_PV
    else:
        poids_PV = 3.0
        parameters['poids_PV'] = poids_PV
    # Interpolation
    # Update the PV cells and the absorber signal within the parameters dict
    parameters["Signal_PV"] = Signal_PV
    parameters["Signal_Th"] = Signal_Th

if algo.__name__ == "DEvol":
    if 'mutation_DE' not in locals():
        mutation_DE = "current_to_best"

    parameters["mutation_DE"] = mutation_DE
    parameters["f1"] = f1
    parameters["f2"] = f2

if algo.__name__ == "optimize_ga":
    if 'precision_AlgoG' not in locals():
        precision_AlgoG = 1e-5
    if 'mutation_delta' not in locals():
        mutation_delta = 15
    if 'crossover_rate' not in locals():
        crossover_rate = 0.9
    if 'evaluate_rate' not in locals():
        evaluate_rate = 0.3

    parameters["Precision_AlgoG"] = precision_AlgoG
    parameters["mutation_delta"] = mutation_delta
    parameters['crossover_rate'] = crossover_rate
    parameters['evaluate_rate'] = evaluate_rate
    parameters["Mod_Algo"] = "for"

if algo.__name__ == "optimize_strangle":
    parameters["Precision_AlgoG"] = precision_AlgoG
    parameters["Mod_Algo"] = "for"
    parameters['evaluate_rate'] = evaluate_rate

if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
    parameters["vf_range"] = vf_range

if 'Lambda_cut_1' not in locals():
    Lambda_cut_1 = 0  # nm
if 'Lambda_cut_2' not in locals():
    Lambda_cut_2 = 0  # nm
parameters["Lambda_cut_1"] = Lambda_cut_1
parameters["Lambda_cut_2"] = Lambda_cut_2
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
    if language == "fr":
        print("Début du programme")
        launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
        print("Lancement à " + launch_time)
        print("Nombre de coeur détecté : ", cpu_count())
        print("Nombre de coeur utilisé : ", cpu_used)

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
    base_directory = directory
    count = 2
    while os.path.exists(directory):
        directory = f"{base_directory}-{count}"
        count += 1
    os.makedirs(directory)
    if language == "fr":
        print("Le dossier '" + directory + "' a été créé.")
    if language == "en":
        print("The folder '" + directory + "' has been created.")

    Mat_Stack_print = Mat_Stack

    if 'nb_layer' in locals():
        for i in range(nb_layer):
            Mat_Stack_print = Mat_Stack_print + ["X"]
    parameters.update({
        "Mat_Stack_print": Mat_Stack_print})
    if language == "fr":
        print("Le stack est : ", Mat_Stack_print)
    if language == "en":
        print("The thin layer stack is : ", Mat_Stack_print)

    if language == "fr":
        if 'nb_layer' in locals():
            nb_total_layer = len(Mat_Stack) + nb_layer
        else:
            nb_total_layer = len(Mat_Stack)
        print("Le nombre total de couches minces est de : " + str(nb_total_layer))

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
                          'name_PV': name_PV,
                          'name_Th': name_Th,
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
    if language == "en":
        print("The total real time is: ",
              "{:.2f}".format(time_real), "seconds")
        print("The processor calculation real time is: ",
              "{:.2f}".format(sum(tab_temps)), "seconds")
    """_____________________Best results datas______________________"""

    Explain_results(parameters, Experience_results)

    """_____________________Write results in a text file_________________"""

    Convergences_txt(parameters, Experience_results, directory)
    Generate_txt(parameters, Experience_results, directory)
    Optimization_txt(parameters, Experience_results, directory)
    Simulation_amont_aval_txt(parameters, Experience_results, directory)
    Generate_materials_txt(parameters, Experience_results, directory)

    """________________________Plot creation_________________________"""

    Reflectivity_plot(parameters, Experience_results, directory)
    Transmissivity_plot(parameters, Experience_results, directory)
    OpticalStackResponse_plot(parameters, Experience_results, directory)
    Convergence_plots(parameters, Experience_results, directory)
    Convergence_plots_2(parameters, Experience_results, directory)
    Consistency_curve_plot(parameters, Experience_results, directory)
    Optimum_thickness_plot(parameters, Experience_results, directory)
    Optimum_refractive_index_plot(parameters, Experience_results, directory)
    Volumetric_parts_plot(parameters, Experience_results, directory)
    Stack_plot(parameters, Experience_results, directory)

# %%
