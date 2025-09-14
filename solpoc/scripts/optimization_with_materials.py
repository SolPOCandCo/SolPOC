# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:14:27 2025
@author: agrosjean
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import solpoc as sol # run v 0.9.4

# the thin layer stack. 
Mat_Stack = ["BK7", "SiO2", "TiO2", "UM", "TiO2", "UM", "UM"] # Insert "UM" (Unknown Material) to specify a layer where the material can vary
# Liste of optional material  of thin layer mark with "UM"
Mat_Option = ["SiO2", "ZnO", "TiO2"] # Max actual length is 3. 
algo = sol.DEvol # Name of the optimization method
selection = sol.selection_max # Callable. Name of the selection method : selection_max or selection_min
cost_function = sol.evaluate_R_s # Callable. Name of the cost function
Th_range = (50, 250)  # in nm.
Th_Substrate = 1e6  # Substrate thickness, in nm
Wl = np.arange(280,2505,20) #np.arange(280, 2505, 5)
Ang = 0  # Incidence angle on the thin layers stack, in °
Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec('Materials/SolSpec.txt', 'GT')
pop_size = 30  # number of individual in the initial population 
crossover_rate = 0.5
f1 = 1.0  # Hyperparameter for the mutation strategie
mutation_DE = "rand_1" # Mutaiton strategie 
budget = 500 # budget, number of iteration  
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
    Mat_Stack=Mat_Stack,
    Mat_Option = Mat_Option,
    algo=algo,
    cost_function=cost_function,
    selection=selection,
    Th_range=Th_range,
    Th_Substrate = Th_Substrate,
    Wl=Wl,
    Ang=Ang,
    Sol_Spec=Sol_Spec,
    name_Sol_Spec=name_Sol_Spec,
    pop_size=pop_size,
    crossover_rate=crossover_rate,
    f1 = f1,
    mutation_DE = mutation_DE,
    budget=budget,
    n_Stack=n_Stack,
    k_Stack=k_Stack,
)

t1 = time.time()
best_solution, dev, n_iter, seed = algo(cost_function, selection, parameters)
t2 = time.time()
print("The total time is: ", "{:.2f}".format(t2 -t1), "seconds")
d_Stack, x = best_solution[:-len(Mat_Stack)], best_solution[-len(Mat_Stack):]
print("Best thin layer stack, in nm :",[f"{x:.1f} nm" for x in d_Stack])
Rs, Ts, As = sol.evaluate_RTA_s(best_solution, parameters) 
print(f"Solar reflectance : {Rs:.4f}")
print(sol.fill_material_stack(Mat_Stack, x, Mat_Option))
sol.print_material_probabilities(Mat_Stack, x, Mat_Option, n_trials=1000)

tab_results = []  # (Rs, best_solution, mat_stack_utilisé)

limit_2 = 5 
for i in range(limit_2):
    print("Launch of second optimization : ", str(i+1) + "/" + str(limit_2))
    # Génère un nouvel empilement de matériaux pour cette itération
    current_stack = sol.fill_material_stack(Mat_Stack, x, Mat_Option)
    parameters["Mat_Stack"] = current_stack

    # Lance l'optimisation
    best_solution_2, dev, n_iter, seed = algo(cost_function, selection, parameters)

    # Évalue la solution trouvée
    Rs, Ts, As = sol.evaluate_RTA_s(best_solution_2, parameters)

    # Sauvegarde Rs, la solution et l'empilement utilisé
    tab_results.append((Rs, best_solution_2, current_stack))

# Recherche le meilleur résultat selon Rs
best_Rs, best_sol, best_stack = max(tab_results, key=lambda item: item[0])

# Affiche les résultats de la meilleure solution
d_Stack_best, x_best = best_sol[:-len(Mat_Stack)], best_sol[-len(Mat_Stack):]

print("\n--- BEST SOLUTION AFTER SECOND OPTIMIZATION ---")
print("Material stack used:", best_stack)
print("Best thin layer stack, in nm :", [f"{val:.1f} nm" for val in d_Stack_best])
print(f"Solar reflectance : {best_Rs:.4f}")



