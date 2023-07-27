# -*- coding: utf-8 -*-
"""
Created on 27072023
COPS v 0.9.0
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
# %% 
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from functions_COPS import *
from datetime import datetime
from multiprocessing import Pool, cpu_count

# %%  Main : You can start to modified something
Comment = "A sentence to be written in the final text file" # Comment to be written in the simulation text file
Mat_Stack = Write_Stack_Periode(["BK7"], ["TiO2", "SiO2"], 2)
# Choice of optimization method
algo = DEvol # Name of the optimization methode 
selection = selection_max # Callable. Name of the selection methode : selection_max or selection_min
evaluate = evaluate_R_Brg# Callable. Name of the cost function
# %% Important parameters 
# Wavelenght domain, here from 320 to 2500 nm wit a 5 nm step. Can be change!   
Wl = np.arange(400 , 800, 5) # /!\ Last value is not include in the array
# Thickness of the substrack, in nm 
Th_Substrack = 1e6 # Substrat thickness, in nm 
# Range of thickness (lower bound and upper bound), for the optimisation process
Th_range = (0, 200) # in nm.
# Range of refractive index (lower bound and upper bound), for the optimisation process
n_range = (1.3 , 3.0) 
# Range of volumic fraction (lower bound and upper bound), for the optimisation process
vf_range = (0 , 1.0) #  volumic fraction of inclusion in host matrix, must be include in (0,1)
# Incidance angle of the thina layer stack. 0 degres is for normal incidence angle
Ang = 0 # Incidence angle on the thin layers stack, in °
#%% Optional parameters
C = 80 # Solar concentration. Data necessary for solar thermal application, like selective stack 
T_air = 20 + 273 # Air temperature, in Kelvin. Data necessary for solar thermal application, like selective stack 
T_abs = 300 + 273 # Thermal absorber temperature, in Kelvin. Data necessary for solar thermal application, like selective stack 
# Cuting Wavelenght. Data necessary for low-e, RTR or PV_CSP evaluates functions
Lambda_cut_1 = 800 # nm 
Lambda_cut_2 = 1000 # nm 
# Addition of theoretical thin layers with the variable nb_layer, whose thickness AND index must be optimized.
nb_layer = 0 # Number of theoretical thin layers above the stack. This variable can be left undefined.
# Allows fixing the thickness of a layer that will not be optimized. d
d_Stack_Opt = [] #Set to "no" to leave it unset. For example, if there are three layers, it can be written ["no",40,"no"]. The code understands that only the middle layer is fixed
# Open the solar spectrum 
Wl_sol , Sol_Spec , name_SolSpec = open_SolSpec('Materials/SolSpec.txt','GT')
# Open a file with PV cell shape
Wl_PV , Signal_PV , name_PV = open_Spec_Signal('Materials/PV_cells.txt', 1)
# Open a file with thermal absorber shape
Wl_Th , Signal_Th , name_Th = open_Spec_Signal('Materials/Thermal_absorber.txt', 1)
#%% Hyperparameters for optimisation methods
pop_size = 30 # number of individual per iteration / generation 
crossover_rate = 0.5 # crossover rate (1.0 = 100%)
evaluate_rate = 0.3 # Part of individuals selected to be the progenitors of next generations
mutation_rate = 0.5 # chance of child gene muted during the birth. /!\ This is Cr for DEvol optimization methode
mutation_delta = 15 # If a chromose mutate, le value change form random number include between + or - this values
f1, f2 = 0.9, 0.8  # Hyperparameter for DEvol 
mutation_DE = "current_to_best" # String. Mutaton methode for DEvol optimization method
nb_generation = 300 # Number of generation/iteration. For DEvol is also used to calculate the budget (nb_generation * pop_size)
precision_AlgoG = 1e-5 # accurency for stop the optimisation processs for some optimization methode, as optimiza_agn or strangle
nb_run = 10 # Number of run
cpu_used = 10  # Number of CPU used. /!\ be "raisonable", regarding the real number of CPU our computer
#seed = 45 # Seed of the random number generator. Uncommet for fix the seed
#%% You should stop modifying anything :) 
"""_________________________________________________________________________"""
# Open and interpol the refractive index
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
# Open and processing the reflectif index of materials used in the stack (Read the texte files in Materials/ )
Sol_Spec = np.interp(Wl, Wl_sol, Sol_Spec) # Interpolate the solar spectrum 
Signal_PV = np.interp(Wl, Wl_PV, Signal_PV) # Interpolate the signal
Signal_Th = np.interp(Wl, Wl_Th, Signal_Th) # Interpolate the signal 
# The container is a dictionary containing the problem variables
# The container is given as input to certain functions
# => They then read the necessary variables  with the commande .get
conteneur = {'Wl': Wl, # Je stocke une variable nommée "Wl", et lui donne la valeur de Wl
            'Ang': Ang, 
            'Ep_Substrack' : Th_Substrack,
            'Ep_plage' : Th_range,
            'Mat_Stack' : Mat_Stack,
            'SolSpec' : Sol_Spec,
            'Lambda_cut_min' : Lambda_cut_1,
            'Lambda_cut' : Lambda_cut_2,
            'n_Stack' : n_Stack,
            'k_Stack' : k_Stack,
            'pop_size': pop_size,
            'name_algo' : algo.__name__, 
            'name_selection' : selection.__name__, 
            'mutation_rate': mutation_rate,
            'nb_generation' :nb_generation,} # End of the dict

#%%
# If nb_layer exists, then I optimize one or more theoretical thin layers
# I add values to the container (dictionary used to transmit variables) 
language = "en" # can change into fr for write the console information and the files in the folder in english

if 'd_Stack_Opt' not in locals() or len(d_Stack_Opt) == 0:
    d_Stack_Opt =  ["no"] * ((len(Mat_Stack) - 1) + nb_layer)
    conteneur["d_Stack_Opt"] = d_Stack_Opt
else:
    conteneur["d_Stack_Opt"] = d_Stack_Opt
if 'nb_layer' in locals() and nb_layer != 0:
    conteneur["nb_layer"] = nb_layer
    conteneur["n_plage"] = n_range
# si la variale seed existe, je la rajoute dans le dictionnaire. 
if 'seed' in locals():
    conteneur["seed"] = seed
    
# If I optimized a antireflective coating for PV, I need the Pv signal shape
if evaluate.__name__ == "evaluate_T_pv" or evaluate.__name__ == "evaluate_A_pv":
    conteneur["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec
    
if evaluate.__name__ == "evaluate_T_Human_eye":
    # Open a file with Human eye response 
    # eye is written fully for not misunderstood with the e for emissivity
    Wl_H_eye , Signal_H_eye , name_H_eye = open_Spec_Signal('Materials/Human_eye.txt', 1)
    Signal_H_eye = np.interp(Wl, Wl_H_eye, Signal_H_eye) # Interpolate the signal
    
    Sol_Spec = Signal_H_eye 
    conteneur["Sol_Spec_with_Human_eye"] = Signal_H_eye 
    
# If I optimized a antireflective coating for PV, I need the Pv signal shape
if evaluate.__name__ == "evaluate_rh":
    if 'C' in locals(): 
        conteneur["C"] = C
    else : 
        conteneur["C"] = 80
    if 'T_air' in locals():
        conteneur["T_air"] = T_air
    else:
        conteneur["T_air"] = 293
    if 'T_abs' in locals():
        conteneur["T_abs"] = T_abs
    else:
        conteneur["T_abs"] = 300 + 273

# Optimize a PV/CSP coating not with a RTR shape, but with a net energy balance
if evaluate.__name__ == "evaluate_netW_PV_CSP":
    if 'poids_PV' in locals():
        conteneur['poids_PV'] = poids_PV
    else : 
        poids_PV = 3.0
        conteneur['poids_PV'] = poids_PV
    # Interpolation 
    # Put the PV cells and the absorber signal in the conteneur
    conteneur["Signal_PV"] = Signal_PV
    conteneur["Signal_Th"] = Signal_Th 
    
if algo.__name__ == "DEvol":
    if 'mutation_DE' not in locals():
        mutation_DE = "current_to_best" 
    
    conteneur["mutation_DE"] = mutation_DE
    conteneur["f1"] = f1
    conteneur["f2"] = f2
    
if algo.__name__ == "optimize_ga":
    if 'precision_AlgoG' not in locals():
        precision_AlgoG = 1e-5
    if 'mutation_delta' not in locals():
        mutation_delta = 15
    if 'crossover_rate' not in locals():
        crossover_rate = 0.9
    if 'evaluate_rate' not in locals():
        evaluate_rate = 0.3

    conteneur["Precision_AlgoG"] = precision_AlgoG
    conteneur["mutation_delta"] = mutation_delta
    conteneur['crossover_rate']= crossover_rate
    conteneur['evaluate_rate'] = evaluate_rate
    conteneur["Mod_Algo"] = "for"   

if algo.__name__ == "optimize_strangle":
    conteneur["Precision_AlgoG"] = precision_AlgoG
    conteneur["Mod_Algo"] = "for"
    conteneur['evaluate_rate'] = evaluate_rate

if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
    conteneur["vf_plage"] = vf_range
    
if 'Lambda_cut_1' not in locals():
    Lambda_cut_1 = 0 # nm 
if 'Lambda_cut_2' not in locals():
    Lambda_cut_2 = 0 # nm 
    
#%%
# creation of a function for multiprocessing
def run_problem_solution(i):
    t1 = time.time() # Time before the optimisation process
    # Line below to be uncommented to slightly desynchronize the cores, if the seed is generated by reading the clock.
    # time.sleep(np.random.random())
    # Run the optimisation process (algo), with an evaluate methode, a selection methode and the container
    best_solution, dev, nb_run, seed = algo(evaluate, selection, conteneur)
    # calcul the time used
    t2 = time.time()
    temps = t2 - t1
    # best solution is a stack. Evaluation of this stack 
    if type(best_solution) != list:
        best_solution = best_solution.tolist()
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    perf = evaluate(best_solution, conteneur)
    if language== "fr":
        print("J'ai fini le cas n°", str(i+1), " en ", "{:.1f}".format(temps), " secondes."," Meilleur : ", "{:.4f}".format(perf),flush=True)
    if language== "en":
        print("I finished case #", str(i+1), " in ", "{:.1f}".format(temps), " seconds.", " Best: ", "{:.4f}".format(perf), flush=True)
                  
    return best_solution, perf, dev, nb_run, temps, seed
#%%
# Début de la boucle main. Le code doit être dans cette boucle pour fonctionner en multiprocessing 
if __name__=="__main__":
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
    
    # Ecriture du dossier de sauvegarde, à la date et heure du jour 
    directory = date_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    if language == "fr":
        print("Le dossier '" + directory + "' a été créé.")
    if language == "en": 
        print("The folder '" + directory + "' has been created.")
    
    Mat_Stack_print = Mat_Stack
    
    if 'nb_layer' in locals():
        for i in range(nb_layer):
            Mat_Stack_print = Mat_Stack_print + ["X"]
    
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
        
    # creation of a pool 
    mp_pool = Pool(cpu_used)
    # Solving each problem in the pool using multiprocessing
    results = mp_pool.map(run_problem_solution, range(nb_run))
    
    # Creation of empty list, for use then later  
    tab_best_solution = []
    tab_dev = []
    tab_perf = []
    tab_nb_run = []
    tab_temps = []
    tab_seed = []
    # sleep 1 ms
    time.sleep(1) 
    # result is an array containing the solutions returned by my.pool. I extract them and place them in different arrays
    for i, (best_solution, perf, dev, nb_run, temps, seed) in enumerate(results):
        # I add the values to the tables 
        tab_perf.append(perf)
        tab_dev.append(dev)
        tab_best_solution.append(best_solution)
        tab_nb_run.append(nb_run) 
        tab_temps.append(temps)
        tab_seed.append(seed)

    end_of_time = time.time()
    time_real = end_of_time - dawn_of_time
    if language == "fr": 
        print("Le temps réel total est de : ", "{:.2f}".format(time_real), "secondes")
        print("Le temps réel de calcul processeur est de : ", "{:.2f}".format(sum(tab_temps)), "secondes")
    if language == "en":   
        print("The total real time is: ", "{:.2f}".format(time_real), "seconds")
        print("The processor calculation real time is: ", "{:.2f}".format(sum(tab_temps)), "seconds")
    """___________________Données des meilleurs résultats______________________"""
    # Go to find the best in all the result
    if selection.__name__ == "selection_max":
        max_value = max(tab_perf) # cherche le max
    if selection.__name__ == "selection_min":
        max_value = min(tab_perf) # cherche le max    
    max_index = tab_perf.index(max_value) # cherche l'index du max (où il est)

    # I've just found my maximum, out of all my runs. It's the best of the best! Congratulations! 
    
    # Calcul of Rs, Ts, As du max (solar performances)
    Rs, Ts, As = evaluate_RTA_s(tab_best_solution[max_index], conteneur) 
    # Calcul le R, T, A (Reflectivity and other, for plot a curve)
    R, T, A = RTA_curve(tab_best_solution[max_index], conteneur)
    # I set at least one value other than 0 to avoid errors when calculating the integral.
    
    if all(value == 0 for value in T):
        T[0] = 10**-301
    if all(value == 0 for value in R):
         R[0] = 10**-301
    if all(value == 0 for value in A):
        A[0] = 10**-301
    
    # Upstream
    # Opening the solar spectrum
    # Reminder: GT spectrum => Global spectrum, i.e., the spectrum of the sun + reflection from the environment
    # GT Spectrum = Direct Spectrum (DC) + Diffuse Spectrum
    # This is the spectrum seen by the surface
    Wl_sol, Sol_Spec, name_SolSpec = open_SolSpec('Materials/SolSpec.txt', 'GT')
    Sol_Spec = np.interp(Wl, Wl_sol, Sol_Spec)
    # Integration of the solar spectrum, raw in W/m2
    Sol_Spec_int = trapz(Sol_Spec, Wl)
    # Writing the solar spectrum modified by the treatment's transmittance
    Sol_Spec_mod_T = T * Sol_Spec
    Sol_Spec_mod_T_int = trapz(Sol_Spec_mod_T, Wl)  # integration of the T-modified solar spectrum, result in W/m2
    # Integration of the solar spectrum modified by the treatment's reflectance, according to the spectrum
    Sol_Spec_mod_R = R * Sol_Spec
    Sol_Spec_mod_R_int = trapz(Sol_Spec_mod_R, Wl)  # integration of the R-modified solar spectrum, result in W/m2
    # Integration of the solar spectrum modified by the treatment's absorbance, according to the spectrum
    Sol_Spec_mod_A = A * Sol_Spec
    Sol_Spec_mod_A_int = trapz(Sol_Spec_mod_A, Wl)  # integration of the A-modified solar spectrum, result in W/m2
    # Calculation of the upstream solar efficiency, for example, the efficiency of the PV solar cell with the modified spectrum
    Ps_amont = SolarProperties(Wl, Signal_PV, Sol_Spec_mod_T)
    # Calculation of the upstream treatment solar efficiency with an unmodified spectrum
    Ps_amont_ref = SolarProperties(Wl, Signal_PV, Sol_Spec)
    # Calculation of the integration of the useful upstream solar spectrum
    Sol_Spec_mod_amont = Sol_Spec * Signal_PV
    Sol_Spec_mod_amont_int = trapz(Sol_Spec_mod_amont, Wl)
    # Calculation of the integration of the useful upstream solar spectrum with T-modified spectrum
    Sol_Spec_mod_T_amont = Sol_Spec_mod_T * Signal_PV
    Sol_Spec_mod_T_amont_int = trapz(Sol_Spec_mod_T_amont, Wl)

    # Downstream
    # Opening the solar spectrum, which may be different from the first one depending on the cases
    # Reminder: DC spectrum => Direct spectrum, i.e., only the spectrum of the sun, concentrable by an optical system
    Wl_sol_2, Sol_Spec_2, name_SolSpec_2 = open_SolSpec('Materials/SolSpec.txt', 'DC')
    Sol_Spec_2 = np.interp(Wl, Wl_sol_2, Sol_Spec_2)
    # Integration of the solar spectrum, raw in W/m2
    Sol_Spec_int_2 = trapz(Sol_Spec_2, Wl)
    # Writing the solar spectrum modified by the treatment's reflectance
    Sol_Spec_mod_R_2 = R * Sol_Spec_2
    Sol_Spec_mod_R_int_2 = trapz(Sol_Spec_mod_R_2, Wl)  # integration of the R-modified solar spectrum, result in W/m2
    # Calculation of the downstream solar efficiency, for example, the efficiency of the thermal absorber
    Ps_aval = SolarProperties(Wl, Signal_Th, Sol_Spec_mod_R_2)
    # Calculation of the downstream treatment solar efficiency with an unmodified spectrum
    Ps_aval_ref = SolarProperties(Wl, Signal_Th, Sol_Spec_2)
    # Calculation of the integration of the useful downstream solar spectrum
    Sol_Spec_mod_aval = Sol_Spec_2 * Signal_Th
    Sol_Spec_mod_aval_int = trapz(Sol_Spec_mod_aval, Wl)
    # Calculation of the integration of the useful downstream solar spectrum
    Sol_Spec_mod_R_aval = Sol_Spec_mod_R_2 * Signal_Th
    Sol_Spec_mod_R_aval_int = trapz(Sol_Spec_mod_R_aval, Wl)
    
    """________________________Plot creation_________________________"""

    # Reflectivity plot
    fig, ax1 = plt.subplots()
    color = 'black' # Couleurs de base possibles: b g r c m y k w
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectivity (-)', color=color)
    if evaluate.__name__ == 'evaluate_rh':
        ax1.set_xscale('log')
    ax1.plot(Wl, R, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Code line to change y-axis, for reflectance # Ligne de code pour changer l'axe de y, pour la réflectance
    # Disabled for automatic scaling  # Désactivé pour échelle automatique
    
    ax1.set_ylim(0, 1) # changer l'échelle de l'axe y
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
    ax2.plot(Wl, Sol_Spec, color=color)
    if evaluate.__name__ == 'evaluate_rh':
        BB_shape = BB(T_abs, Wl)
        ## BB_shape est la forme du corps noir. En fonction de la température, l'irradiance du corps noir peut être tres supérieur
        # au spectre solair. Pour ce graphiquie, je met donc le corps noir à la meme hauteur 
        BB_shape =BB_shape*(max(Sol_Spec)/max(BB_shape))
        ax2.plot(Wl, BB_shape, color='orange', linestyle = 'dashed')
    
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  
    ax2.set_ylim(0, 2) # changer l'échelle de l'axe y
    plt.title("Optimum Reflectivity")
    # Save the plot. Sauvegarde de la figure
    plt.savefig(directory + "/" + "Optimum_Reflectivity.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    # Plot transmissivity # Graph de la transmittance
    fig, ax1 = plt.subplots()
    color = 'black' # Couleurs de base possibles: b g r c m y k w
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transmissivity (-)', color=color)
    if evaluate.__name__ == 'evaluate_rh':
        ax1.set_xscale('log')
    ax1.plot(Wl, T, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Ligne de code pour changer l'axe de y, pour la réflectance
    # Désactivé pour échelle automatique
    ax1.set_ylim(0, 1) # changer l'échelle de l'axe y
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
    ax2.plot(Wl, Sol_Spec, color=color)
    if evaluate.__name__ == 'evaluate_rh':
        BB_shape = BB(T_abs, Wl)
        ## BB_shape est la forme du corps noir. En fonction de la température, l'irradiance du corps noir peut être tres supérieur
        # au spectre solair. Pour ce graphiquie, je met donc le corps noir à la meme hauteur 
        BB_shape =BB_shape*(max(Sol_Spec)/max(BB_shape))
        ax2.plot(Wl, BB_shape, color='orange', linestyle = 'dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  
    ax2.set_ylim(0, 2) # changer l'échelle de l'axe y
    plt.title("Optimum Transmissivity")
    plt.savefig(directory + "/" + "Optimum_Transmissivity.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    # convergences plots
    if (nb_run > 2): 
        tab_perf_save = tab_perf.copy()
        tab_dev_save = tab_dev.copy()
        # Je cherche l'index max dans la table de performance
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_1 = tab_dev_save[max_index] # Je cherche le dev associé
        del tab_perf_save[max_index], tab_dev_save[max_index]
        
        # Je recherche le max, qui est alors le second
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_2 = tab_dev_save[max_index]
        del tab_perf_save[max_index], tab_dev_save[max_index]
    
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_3 = tab_dev_save[max_index]
        del tab_perf_save, tab_dev_save# suprime toute la variable 
        
        if  algo.__name__ == "DEvol":
            if  selection.__name__ == "selection_max":
                dev_1 = [1- x  for x in dev_1]
                dev_2 = [1- x  for x in dev_2]
                dev_3 = [1- x  for x in dev_3]
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('Cost function (-)', color='black')
        ax1.plot(dev_1, color='black', label = "Best")
        ax1.plot(dev_2, color='red', label = "Second")
        ax1.plot(dev_3, color='green', label = "Third")
        plt.legend()
        ax1.tick_params(axis='y', labelcolor='black')
        plt.title("Convergence Plots")
        plt.savefig(directory + "/" + "ConvergencePlots.png", dpi = 300, bbox_inches='tight')
        plt.show()
    
    # Je copie ma table de performance
    if (nb_run > 5): 
        tab_perf_save = tab_perf.copy()
        tab_dev_save = tab_dev.copy()
        # Je cherche l'index max dans la table de performance
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_1 = tab_dev_save[max_index] # Je cherche le dev associé
        del tab_perf_save[max_index], tab_dev_save[max_index]
        
        # Je recherche le max, qui est alors le second
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_2 = tab_dev_save[max_index]
        del tab_perf_save[max_index], tab_dev_save[max_index]
    
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_3 = tab_dev_save[max_index]
        del tab_perf_save[max_index], tab_dev_save[max_index]
        
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_4 = tab_dev_save[max_index]
        del tab_perf_save[max_index], tab_dev_save[max_index]
        
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_5 = tab_dev_save[max_index]
        del tab_perf_save[max_index], tab_dev_save[max_index]
        
        max_index = tab_perf_save.index(max(tab_perf_save))
        dev_6 = tab_dev_save[max_index]
        del tab_perf_save, tab_dev_save# suprime toute la variable  
        
        if  algo.__name__ == "DEvol":
            if  selection.__name__ == "selection_max":
                dev_1 = [1- x  for x in dev_1]
                dev_2 = [1- x  for x in dev_2]
                dev_3 = [1- x  for x in dev_3]
                dev_4 = [1- x  for x in dev_4]
                dev_5 = [1- x  for x in dev_5]
                dev_6 = [1- x  for x in dev_6]
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('Cost function (-)', color='black')
        ax1.plot(dev_1, color='black', label = "1st")
        ax1.plot(dev_2, color='red', label = "2nd")
        ax1.plot(dev_3, color='green', label = "3rd")
        ax1.plot(dev_4, color='blue', label = "4th")
        ax1.plot(dev_5, color='orange', label = "5th")
        ax1.plot(dev_6, color='purple', label = "6th")
        plt.legend()
        ax1.tick_params(axis='y', labelcolor='black')
        plt.title("Convergence Plots")
        plt.savefig(directory + "/" + "ConvergencePlots2.png", dpi = 300, bbox_inches='tight')
        plt.show()
    
    # Graph de la convergence du problème
    tab_perf_sorted = tab_perf.copy()
    tab_perf_sorted.sort(reverse = True)
    fig, ax1 = plt.subplots()
    color = 'black' # Couleurs de base possibles: b g r c m y k w
    if max(tab_perf_sorted) - min(tab_perf_sorted) < 1e-4:
        ax1.set_ylim(np.mean(tab_perf_sorted) - 0.0005, np.mean(tab_perf_sorted) + 0.0005) # changer l'échelle de l'axe y
    ax1.set_xlabel('Best cases (left) to worse (right)')
    ax1.set_ylabel('Cost function (-)', color=color)
    ax1.plot(tab_perf_sorted, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.title("Consistency Curve")
    plt.savefig(directory + "/" + "ConsistencyCurve.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    # Plot of thickness # Graph des épaisseurs
    ep = tab_best_solution[max_index]
    if 'nb_layer' in locals() and nb_layer != 0:
        ep = np.delete(ep, np.s_[(nb_layer + len(Mat_Stack)):])
        
    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2: 
        vf = []
        vf = ep[len(Mat_Stack):len(ep)]
        ep = np.delete(ep, np.s_[(len(Mat_Stack)):len(ep)])
 
    #del epaisseur[0]
    lower = Th_range[0]
    upper = Th_range[1]
    fig, ax = plt.subplots()
    ax.scatter(range(1, len(ep)), ep[1:])
    ax.axhline(lower, color='r')
    ax.axhline(upper, color='g')
    ax.set_xticks(range(1, len(ep)))
    ax.set_xticklabels([str(i) for i in range(1, len(ep))])
    for i, val in enumerate(ep[1:]):
        ax.annotate(str("{:.0f}".format(val)), xy=(i + 1, val), xytext=(i + 1.1, val + 1.1 ))
    plt.xlabel("Number of layers, subtrate to air")
    plt.ylabel("Thickness (nm)")
    plt.title("Optimum Thickness ")
    plt.savefig(directory + "/" + "Optimum_Thickness_Stack.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    if 'nb_layer' in conteneur:
        # Plot of refractif index Graph des indices
        n_list = tab_best_solution[max_index]
        for i in range(nb_layer + len(Mat_Stack)-1):
            n_list = np.delete(n_list, 0)
        #del epaisseur[0]
        lower = n_range[0]
        upper = n_range[1]
        fig, ax = plt.subplots()
        ax.scatter(range(1, len(n_list)), n_list[1:])
        ax.axhline(lower, color='r')
        ax.axhline(upper, color='g')
        ax.set_xticks(range(1, len(n_list)))
        ax.set_xticklabels([str(i) for i in range(1, len(n_list))])
        # Mettre les étiquettes 
        for i, val in enumerate(n_list[1:]):
            ax.annotate(str("{:.2f}".format(val)), xy=(i +1 , val), xytext=(i+1.05, val +0.05))
        # Fixe les limites sur l'axe y : ici de 1 à 3 
        ax.set_ylim((min(n_range)-0.5), (max(n_range)+0.5)) # changer l'échelle de l'axe y
        plt.xlabel("Number of layers, substrat to air")
        plt.ylabel("Refractive Index (-)")
        plt.title("Optimum Refractive Index ")
        plt.savefig(directory + "/" + "Optimum_RefractiveIndex_Stack.png", dpi = 300, bbox_inches='tight')
        plt.show()

    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        # Graph des fractions volumiques
        lower = vf_range[0]
        upper = vf_range[1]
        fig, ax = plt.subplots()
        ax.scatter(range(1, len(vf)), vf[1:])
        ax.axhline(lower, color='r')
        ax.axhline(upper, color='g')
        ax.set_xticks(range(1, len(vf)))
        ax.set_xticklabels([str(i) for i in range(1, len(vf))])
        # Mettre les étiquettes 
        for i, val in enumerate(vf[1:]):
            ax.annotate(str("{:.3f}".format(val)), xy=(i +1 , val), xytext=(i+1.05, val +0.05))
        # Fixe les limites sur l'axe y : ici de 1 à 3 
        ax.set_ylim((min(vf_range)), (max(vf_range))) # changer l'échelle de l'axe y
        plt.xlabel("Number of layers, substrat to air")
        plt.ylabel("Volumic Fraction (-)")
        plt.title("Volumic Fraction ")
        plt.savefig(directory + "/" + "Optimum_VolumicFraction.png", dpi = 300, bbox_inches='tight')
        plt.show()
    
    """_____________________Write results in a texte file_________________"""
    
    # Mon but est de récupérer quelque valeurs(via equidistant_value) de la convergence de l'algo (contenu dans tab_dev)
    tab_perf_dev = []
    for i in range(len(tab_dev)):
        # Je parcour tab_dev
        data_dev = []
        data_dev = tab_dev[i]
        # Je prend quelque valeurs (de base 5) équidistante
        data_dev = valeurs_equidistantes(data_dev, 5)
        # J'inverse ma liste, car de base la 1er valeur est celle au début du problème, et la derniere correspond
        # a la derniere fonction coût calculé, donc normalement mon best pour ce run
        data_dev.reverse()
        # Si j'ai lancé DEvol en mode sélection_max, les valeurs présente sont en réalité 1 - fcout
        if  algo.__name__ == "DEvol":
            if  selection.__name__ == "selection_max":
                data_dev = [1- x  for x in data_dev]
        tab_perf_dev.append(data_dev)
    # je met la list de list en array
    tab_perf_dev = np.array(tab_perf_dev, dtype=float)
    # Ecriture de tab_perf_dev dans un fichier txt
    np.savetxt(directory + '/Convergence.txt', tab_perf_dev, fmt='%.18e', delimiter='  ')
    
    tab_perf_dev = []
    for i in range(len(tab_dev)):
        # Je parcour tab_dev
        data_dev = []
        data_dev = tab_dev[i]
        # Je prend quelque valeurs (de base 25) équidistante
        data_dev = valeurs_equidistantes(data_dev, 25)
        # J'inverse ma liste, car de base la 1er valeur est celle au début du problème, et la derniere correspond
        # a la derniere fonction coût calculé, donc normalement mon best pour ce run
        data_dev.reverse()
        # Si j'ai lancé DEvol en mode sélection_max, les valeurs présente sont en réalité 1 - fcout
        if  algo.__name__ == "DEvol":
            if  selection.__name__ == "selection_max":
                data_dev = [1- x  for x in data_dev]
        tab_perf_dev.append(data_dev)
    # je met la list de list en array
    tab_perf_dev = np.array(tab_perf_dev, dtype=float)
    # Ecriture de tab_perf_dev dans un fichier txt
    np.savetxt(directory + '/Convergence_25.txt', tab_perf_dev, fmt='%.18e', delimiter='  ')
    
    filename = directory + "/performance.txt"
    with open(filename, "w") as file:
        for value in tab_perf:
            file.write(str(value) + "\n")

    filename = directory + "/seed.txt"
    with open(filename, "w") as file:
        for value in tab_seed:
            file.write(str(value) + "\n")                   
    
    filename = directory + "/time.txt"
    with open(filename, "w") as file:
        for value in tab_temps:
            file.write(str(value) + "\n")
    
    filename = directory + "/Stacks.txt"
    with open(filename, "w") as file:
        for value in tab_best_solution:
            np.savetxt(file, value.reshape(1, -1), fmt='%.18e', delimiter=' ')
            
    filename = directory + "/Convergence.txt"
    with open(filename, "w") as file:
        for value in tab_dev:
            np.savetxt(file, value.reshape(1, -1), fmt='%.18e', delimiter=' ')            
            
    filename = directory + "/Sol_Spec_mod_R.txt"
    with open(filename, "w") as file:
        for i in range(len(Wl)):
            file.write(str(Wl[i]) + "\t" + str(Sol_Spec_mod_R[i]) + "\n")
            
    filename = directory + "/Sol_Spec_mod_T.txt"
    with open(filename, "w") as file:
        for i in range(len(Wl)):
            file.write(str(Wl[i]) + "\t" + str(Sol_Spec_mod_T[i]) + "\n")               
    
    if language == "fr":
        print("Les résultats ont été écrits dans le dossier")
    if language == "en":
        print("The results were written in the folders")
    
    filename = directory + "/RTA.txt"
    with open(filename, "w") as file:
        for i in range(len(A)):
            file.write(str(Wl[i]) + "\t" + str(R[i]) + "\t" + str(T[i]) + "\t" + str(A[i]) + "\n")
    
    if language == "fr": 
        print("Les données RTA du meilleur empillement ont été écrites dans cet ordre")
    if language == "en": 
        print("The RTA data for the best stack were written in the folder")
    
    filename = directory + "/Optimization.txt"
    script_name = os.path.basename(__file__)
    if language == "fr": 
        with open(filename, "w") as file:
            file.write("Le nom du fichier est : " + str(script_name) + "\n")
            file.write("Heure de lancement " + str(launch_time) + "\n")
            file.write(str(Comment) + "\n")
            file.write("_____________________________________________" +  "\n")
            file.write("Le nom de la fonction d'optimisation est : " + str(algo.__name__) + "\n")
            file.write("Le nom de la fonction d'évaluation est : " + str(evaluate.__name__) + "\n")
            file.write("Le nom de la fonction de sélection est : " + str(selection.__name__) + "\n")
            file.write("Si optimisation par DE, la mutation est : " + mutation_DE + "\n")
            file.write("\n")
            file.write("L'emplacement et le nom du spectre solaire est :"  + str(name_SolSpec) + "\n")
            file.write("La valeur d'irradiance : " + str("{:.1f}".format(trapz(Sol_Spec, Wl))) + " W/m²" + "\n")
            file.write("\n")
            file.write("Nom du dossier :\t" + str(directory) + "\n")
            file.write("Matériaux de l'empillement\t" + str(Mat_Stack_print) + "\n")
            file.write("Le nombre de couche minces est \t" + str(nb_total_layer) + "\n")
            file.write("Domaine des longueurs d'ondes \t" + str(min(Wl)) + " nm à " + str(max(Wl)) + " nm, pas de " + str(Wl[1]-Wl[0])+ " nm"+ "\n")
            file.write("Epaisseur du substrat, en nm \t" + str(Ep_Substrack) + "\n")
            file.write("Plage des épaisseur des couches minces\t" + str(Th_range[0]) + " à " + str(Th_range[1]) + " nm" + "\n")
            file.write("Plage des indices des couches minces\t" + str(n_range[0]) + " à " + str(Range_n[1]) + "\n")
            file.write("Angle d'incidence sur le stack\t" + str(Ang) + "°" + "\n")
            file.write("Le taux de concentration est\t" + str(C) + "\n")
            file.write("La température de l'air est\t" + str(T_air) + " K" + "\n")
            file.write("La température de l'absorbeur' est\t" + str(T_abs) + " K" + "\n")
            if evaluate.__name__ == "evaluate_low_e" or evaluate.__name__ == "evaluate_RTR":
                file.write("Pour les profils d'optimisaiton low-e et RTR " + "\n")
                file.write("La longueur d'onde de coupure UV est \t" + str(Lambda_cut_1) + " nm" + "\n")
                file.write("La longueur d'onde de coupure IR est \t" + str(Lambda_cut_2) + " nm" + "\n")
            if evaluate.__name__ == "evaluate_netW_PV_CSP" : 
                file.write("Pour les profils d'optimisaiton evaluate_netW_PV_CSP" + "\n")
                file.write("Le coût fictif du PV est \t" + str(poids_PV) + "\n")
            file.write("Taille de la population\t" + str(pop_size) + "\n")
            file.write("Taux de crossover\t" + str(crossover_rate) + "\n")
            file.write("Taux d'évaluation\t" + str(evaluate_rate) + "\n")
            file.write("Taux de mutation\t" + str(mutation_rate) + "\n")
            file.write("Valeurs de f1 et f2\t" + str(f1) + " & " + str(f2) + "\n")
            file.write("Etendue de la mutation\t" + str(mutation_delta) + "\n")
            file.write("Precision de l'algo en auto\t" + str(precision_AlgoG) + "\n")
            file.write("Nombre de génération\t" + str(nb_generation) + "\n")
            file.write("Nb de Lancement\t" + str(nb_run) + "\n")
            file.write("Nb de processeur disponible\t" +str(cpu_count()) + "\n")
            file.write("Nb de processeur utilisé\t" +str(cpu_used) + "\n")
            file.write("Le temps réel d'éxécution (en s) total est de :\t" + str("{:.2f}".format(time_real))  + "\n")
            file.write("La somme du temps de calcul (en s) processeur est de :\t" + str("{:.2f}".format(sum(tab_temps)) +  "\n"))
        
        print("Les noms et valeurs des variables de la simulation ont été écrites")
    
    if language == "en":
        with open(filename, "w") as file:
            file.write("The filename is: " + str(script_name) + "\n")
            file.write("Launch time: " + str(launch_time) + "\n")
            file.write(str(Comment) + "\n")
            file.write("_____________________________________________" + "\n")
            file.write("The name of the optimization function is: " + str(algo.__name__) + "\n")
            file.write("The name of the evaluation function is: " + str(evaluate.__name__) + "\n")
            file.write("The name of the selection function is: " + str(selection.__name__) + "\n")
            file.write("If optimizing with DE, the mutation is: " + mutation_DE + "\n")
            file.write("\n")
            file.write("The location and name of the solar spectrum is: " + str(name_SolSpec) + "\n")
            file.write("The irradiance value: " + str("{:.1f}".format(trapz(Sol_Spec, Wl))) + " W/m²" + "\n")
            file.write("\n")
            file.write("Folder name: " + str(directory) + "\n")
            file.write("Materials in the stack: " + str(Mat_Stack_print) + "\n")
            file.write("The number of thin layers: " + str(nb_total_layer) + "\n")
            file.write("Wavelength range: " + str(min(Wl)) + " nm to " + str(max(Wl)) + " nm, step of " + str(Wl[1]-Wl[0]) + " nm" + "\n")
            file.write("Substrate thickness, in nm: " + str(Th_Substrack) + "\n")
            file.write("Range of thin layer thickness: " + str(Th_range[0]) + " to " + str(Th_range[1]) + " nm" + "\n")
            file.write("Range of thin layer indices: " + str(n_range[0]) + " to " + str(n_range[1]) + "\n")
            file.write("Incident angle on the stack: " + str(Ang) + "°" + "\n")
            file.write("Concentration ratio: " + str(C) + "\n")
            file.write("Air temperature: " + str(T_air) + " K" + "\n")
            file.write("Absorber temperature: " + str(T_abs) + " K" + "\n")
            if evaluate.__name__ == "evaluate_low_e" or evaluate.__name__ == "evaluate_RTR":
                file.write("For low-e and RTR optimization profiles" + "\n")
                file.write("UV cutoff wavelength: " + str(Lambda_cut_1) + " nm" + "\n")
                file.write("IR cutoff wavelength: " + str(Lambda_cut_2) + " nm" + "\n")
            if evaluate.__name__ == "evaluate_netW_PV_CSP":
                file.write("For evaluate_netW_PV_CSP optimization profile" + "\n")
                file.write("PV fictitious cost: " + str(poids_PV) + "\n")
            file.write("Population size: " + str(pop_size) + "\n")
            file.write("Crossover rate: " + str(crossover_rate) + "\n")
            file.write("Evaluation rate: " + str(evaluate_rate) + "\n")
            file.write("Mutation rate: " + str(mutation_rate) + "\n")
            file.write("Values of f1 and f2: " + str(f1) + " & " + str(f2) + "\n")
            file.write("Mutation range: " + str(mutation_delta) + "\n")
            file.write("Precision of the algorithm in auto: " + str(precision_AlgoG) + "\n")
            file.write("Number of generations: " + str(nb_generation) + "\n")
            file.write("Number of run: " + str(nb_run) + "\n")
            file.write("Number of available CPU: " + str(cpu_count()) + "\n")
            file.write("Number of used CPU: " + str(cpu_used) + "\n")
            file.write("Total execution time (in s): " + str("{:.2f}".format(time_real)) + "\n")
            file.write("Sum of processor computation time (in s): " + str("{:.2f}".format(sum(tab_temps)) + "\n"))
        
        print("The names and values of the simulation variables have been written.")

    if evaluate.__name__ == "evaluate_netW_PV_CSP" or evaluate.__name__ == "evaluate_RTR" or evaluate.__name__ == "evaluate_low_e":
    
        filename = directory + "/simulation_amont_aval.txt"
        script_name = os.path.basename(__file__)
        with open(filename, "w") as file:
            file.write("Le nom du fichier est : " + str(script_name) + "\n")
            file.write("Heure de lancement " + str(launch_time) + "\n")
            file.write(str(Comment) + "\n")
            file.write("_____________________________________________" +  "\n")
            file.write("Le nom du fichier amont et le n° de la colone est : " + name_PV + "\n")
            file.write("Le nom du fichier avant et le n° de la colone est : " + name_Th + "\n")
            file.write("Le nom du spectre solaire utilisé pour l'optimisation ': " + name_SolSpec + "\n")
            file.write("L'intégration de ce spectre solaire (en W/m2) est " + str("{:.2f}".format(Sol_Spec_int)) + "\n")
            file.write("La puissance transmise par le traitement du spectre solaire incident (en W/m2) est " + str("{:.2f}".format(Sol_Spec_mod_T_int)) + "\n")
            file.write("La puissance réfléchie par le traitement du spectre solaire incident (en W/m2) est " + str("{:.2f}".format(Sol_Spec_mod_R_int)) + "\n")
            file.write("La puissance absorbée par le traitement du spectre solaire incident (en W/m2) est " + str("{:.2f}".format(Sol_Spec_mod_A_int)) + "\n")
            if Lambda_cut_1 != 0 and Lambda_cut_2 != 0: 
                Wl_1 = np.arange(min(Wl),Lambda_cut_1,(Wl[1]-Wl[0]))
                Wl_2 = np.arange(Lambda_cut_1, Lambda_cut_2, (Wl[1]-Wl[0]))
                Wl_3 = np.arange(Lambda_cut_2, max(Wl)+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
                # P_low_e = np.concatenate([R[0:len(Wl_1)],T[len(Wl_1):(len(Wl_2)+len(Wl_1)-1)], R[(len(Wl_2)+len(Wl_1)-1):]])
                file.write("\n")
                # Partie avec le spectre GT
                file.write("Calcul avec le spectre': " + name_SolSpec + "\n")
                # a = trapz(Sol_Spec[0:len(Wl_1)]* R[0:len(Wl_1)], Wl_1)
                # file.write("La puissance solaire réfléchie du début du spectre à Lambda_cut_UV (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                a = trapz(Sol_Spec[len(Wl_1):(len(Wl_2)+len(Wl_1))]* T[len(Wl_1):(len(Wl_2)+len(Wl_1))], Wl_2)
                file.write("La puissance solaire transmise de Lambda_UV à Lambda_IR (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                # a = trapz(Sol_Spec[(len(Wl_2)+len(Wl_1)):]* R[(len(Wl_2)+len(Wl_1)):], Wl_3)
                # file.write("La puissance solaire réfléchie à partir de Lambda_IR (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                # Partie avec le spectre DC
                file.write("Calcul avec le spectre': " + name_SolSpec_2 + "\n")
                a = trapz(Sol_Spec_2[0:len(Wl_1)]* R[0:len(Wl_1)], Wl_1)
                file.write("La puissance solaire réfléchie du début du spectre à Lambda_cut_UV (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                # a = trapz(Sol_Spec_2[len(Wl_1):(len(Wl_2)+len(Wl_1))]* T[len(Wl_1):(len(Wl_2)+len(Wl_1))], Wl_2)
                # file.write("La puissance solaire transmise de Lambda_UV à Lambda_IR (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                a = trapz(Sol_Spec_2[(len(Wl_2)+len(Wl_1)):]* R[(len(Wl_2)+len(Wl_1)):], Wl_3)
                file.write("La puissance solaire réfléchie à partir de Lambda_IR (en W/m2) est " + str("{:.2f}".format(a)) + "\n")            
                del a, Wl_1, Wl_2, Wl_3
                
                file.write("\n")
                file.write("En amont (partie cellule PV sur un système solaire PV/CSP) : " +  "\n")
                file.write("Le nom du spectre solaire est': " + name_SolSpec + "\n")
                file.write("L'intégration de ce spectre solaire (en W/m2) est " + str("{:.2f}".format(Sol_Spec_int)) + "\n")
                file.write("La puissance transmise par le traitement du spectre solaire GT (en W/m2) est " + str("{:.2f}".format(Sol_Spec_mod_T_int)) + "\n")
                file.write("L'efficacité (%) de la cellule avec le spectre solaire non modifié (sans traitement) est " + str("{:.3f}".format(Ps_amont_ref)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " + str("{:.2f}".format(Sol_Spec_mod_amont_int)) + "\n")
                file.write("L'efficacité (%) de la cellule avec le spectre solaire modifié (avec traitement) est " + str("{:.3f}".format(Ps_amont)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " + str("{:.2f}".format(Sol_Spec_mod_T_amont_int)) + "\n")
                file.write("\n")
                file.write("En aval (partie absorbeur thermique sur un système solaire PV/CSP) : " +  "\n")
                file.write("Le nom du spectre solaire est : " + name_SolSpec_2 + "\n")
                file.write("L'intégration de ce spectre solaire (en W/m2) est " + str("{:.2f}".format(Sol_Spec_int_2)) + "\n")
                file.write("La puissance réfléchie par le traitement du spectre solaire DC (en W/m2) est " + str("{:.2f}".format(Sol_Spec_mod_R_int_2)) + "\n")
                file.write("L'efficacité (%) du traitement absorbant avec le spectre solaire non modifié (sans traitement) est " + str("{:.3f}".format(Ps_aval_ref)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " + str("{:.2f}".format(Sol_Spec_mod_aval_int)) + "\n")
                file.write("L'efficacité (%) du traitement absorbant avec le spectre solaire modifié (avec traitement) est " + str("{:.3f}".format(Ps_aval)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " + str("{:.2f}".format(Sol_Spec_mod_R_aval_int)) + "\n")
    
            print("Le fichier simulation_amont_aval.txt est écrit")
            
    # Write the refractive index of materials used in a texte files 
    result = eliminate_duplicates(Mat_Stack)
    n_Stack_w, k_Stack_w = Made_Stack(result[0], Wl)
    
    for i in range(len(result[0])):
        name = result[0][i]
        n = n_Stack_w[:,i]
        k = k_Stack_w[:,i]
        # Trace le graphique
        plt.title("RefractiveIndex of " + name)
        plt.plot(Wl, n, label = "n extroplated")
        plt.plot(Wl, k, label = "k extroplated")
    # =============================================================================
    #     # Ouvre le fichier originel
    #     Wl_2, n_2, k_2 = open_material(result[0][i])
    #     plt.plot(Wl_2, n_2, 'o', label = "n data")
    #     plt.plot(Wl_2, k_2, 'o', label = "k data")
    # =============================================================================
        plt.xlabel("Wavelenght (nm)")
        plt.legend()
        plt.ylabel("Value of refractiv index (-)")
        plt.savefig(directory + "/" + "RefractiveIndex_"+ name + ".png")
        plt.close()
        #filename = "/" + str(name) + ".txt"
        filename = directory + "/" + "RefractiveIndex_" + str(name) + ".txt"
        with open(filename, "w") as file:
            for i in range(len(n)):
                file.write(str(Wl[i]) + "\t" + str(n[i]) + "\t" + str(k[i])+ "\n")
                
    if language == "fr":            
        print("L'indice de réfraction de chaque matériaux a été écrits dans un fichier à leurs noms")
    if language == "en":            
        print("The refractive index of each material have been written in a file with their names")
