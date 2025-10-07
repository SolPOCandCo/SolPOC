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
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
#----------------------------------------------------------------------------#
# %%  Main : You can start to modified something
# Comment to be written in the simulation text file
Comment = "A 4 periodic layers of Bragg mirror, deposited on 1mm BK7 glass"
Mat_Stack = ["BK7", "SiO2", "TiO2", "SiO2", "TiO2", "SiO2", "TiO2", "SiO2", "TiO2"] # or we can use : Mat_Stack = sol.write_stack_period(["BK7"], ["SiO2", "TiO2"], 4)
# Choice of optimisation method
algo = sol.DEvol  # Callable. Name of the optimization method, callable
selection = sol.selection_max # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_R_Brg # Callable. Name of the cost function
# %% Important parameters
# Wavelength domain, here from 280 to 2500 nm with a 5 nm step. Can be change!
Wl = np.arange(400, 800,5) #np.arange(280, 2505, 5)
# Thickness of the substrate, in nm
Th_Substrate = 1e6  # Substrate thickness, in nm
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 200)  # in nm.
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0  # Incidence angle on the thin layers stack, in °
# %% Hyperparameters for optimisation methods
pop_size = 30  # number of individual per iteration / generation
crossover_rate = 0.5  # crossover rate (1.0 = 100%) This is Cr for DEvol optimization method
f1, f2 = 0.9, 0.8  # Hyperparameter for mutation in DE
mutation_DE = "current_to_best" # String. Mutaton method for DE optimization method
# %% Hyperparameters for optimisation methods
# Number of iteration. 
budget = 2000
nb_run = 8  # Number of run, the number of time were the probleme is solved
seed = 2905804230 # Seed of the random number generator. Remplace None for use-it 
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
"""_________________________________________________________________________"""
# Open and interpol the refractive index
n_Stack, k_Stack = sol.Made_Stack(Mat_Stack, Wl)
# Open and processing the reflectif index of materials used in the stack (Read the texte files in Materials/ )# Interpolate the solar spectrum
# parameters is a dictionary containing the problem variables
# This dictionary is given as input to certain functions
# => They then read the necessary variables  with the commande .get

parameters = sol.get_parameters(
    Mat_Stack=Mat_Stack,
    algo=algo,
    cost_function=cost_function,
    selection=selection,  
    Wl=Wl,
    Ang=Ang,
    n_Stack=n_Stack,
    k_Stack=k_Stack,
    Th_range=Th_range,
    Th_Substrate = Th_Substrate,
    pop_size=pop_size,
    f1 = f1,
    f2 = f2, 
    mutation_DE = mutation_DE,
    crossover_rate=crossover_rate,
    budget=budget,
    nb_run=nb_run,
    seed  = seed
)

# %%

print("Start of the program")
launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")

results = []
for i, seed in enumerate(parameters['seed_list']):
    t1 = time.time()
    
    # Préparer les paramètres pour cette exécution
    this_run_params = parameters.copy()
    this_run_params['seed'] = seed
    
    # Lancer l'optimisation
    best_solution, dev, n_iter, seed_used = algo(cost_function, selection, this_run_params)
    
    t2 = time.time()
    temps = t2 - t1
    
    # S'assurer que best_solution et dev sont des arrays
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    
    perf = cost_function(best_solution, parameters)
    
    print(f"I finished case #{i+1} in {temps:.1f} seconds. Best: {perf:.4f}", flush=True)
    
    results.append((best_solution, perf, dev, n_iter, temps, seed_used))


best_result = max(results, key=lambda x: x[1])  # x[1] = perf
best_solution_overall = best_result[0]

# Calculs des performances solaires
R, T, A = sol.RTA_curve(best_solution_overall, parameters)

# Plot the reflectivity
fig, ax1 = plt.subplots()
# --- Tracé de la réflectivité ---
ax1.plot(Wl, R, color='black')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Reflectivity (-)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, 1)
ax1.grid(True)  # Ajout de la grille
# Mise en page et affichage
fig.tight_layout()
plt.title("Optimum Reflectivity", y=1.05)
#plt.savefig("Optimum_Reflectivity.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot the consistency curve
tab_perf = [res[1] for res in results]
tab_perf_sorted = tab_perf.copy()
tab_perf_sorted.sort(reverse=True)
# Création du graphique
fig, ax = plt.subplots()
color = 'black'
# Ajustement de l'axe y si toutes les valeurs sont très proches
if max(tab_perf_sorted) - min(tab_perf_sorted) < 1e-4:
    mean_val = np.mean(tab_perf_sorted)
    ax.set_ylim(mean_val - 0.0005, mean_val + 0.0005)
ax.plot(tab_perf_sorted, linestyle='dotted', marker='o', color=color)
ax.set_xlabel('Best cases (left) to worse (right)')
ax.set_ylabel('Cost function (-)', color=color)
ax.tick_params(axis='y', labelcolor=color)
plt.title("Consistency Curve", y=1.05)
plt.savefig("ConsistencyCurve.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot the stack, using Stack_plot from Python package

Experience_results = {'d_Stack': best_solution_overall,}
# Définir le répertoire où enregistrer les fichiers
try:
    # Si le script est exécuté depuis un fichier .py
    directory = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Si on est dans un notebook (__file__ n'existe pas)
    directory = os.getcwd()
sol.Stack_plot(parameters, Experience_results, directory)