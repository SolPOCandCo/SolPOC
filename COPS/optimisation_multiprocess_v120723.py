# -*- coding: utf-8 -*-
"""
Created on 12 07 2023
@authors: A.Grosjean (main, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
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

# %%  Main 
Comment = "Essais de la fonction d'évaluation evaluate_netW_PV_CSP " # Commentaire qui seras inscrit dans le fichier texte final
Mat_Stack = Ecrit_Stack_Periode(["BK7"], ["TiO2_I", "SiO2_I"], 6) # Ecriture du stack. 
# Ajout de couches minces théorique avec la variable nb_layer, dont l'épaisseur ET l'indice doivent être optimisées
# nb_layer = 0 # Nombre de couche mince théorique par dessus le stack. cette variable peut de pas être définie
d_Stack_Opt =  ["no"] # Permet de fixer une épaisseur d'une couche qui ne seras pas optimiser. Mettre "no" pour ne rien fixer. Si on as par exemple trois couches, on peut écrire [ ,40, ]. Le code comprend que seule la couche du milieu est fixe. 
# Choix de méthodes d'optimisation  
algo = DEvol # On doit écrire le nom de la fonction d'optimisation 
selection = selection_max # On doit écrire le nom de la fonction de sélection
evaluate = evaluate_netW_PV_CSP # On doit écrire le nom de la fonction coût
mutation_DE = "current_to_best" # On doit écrire une chaine de caractère avec nom de la méthode de mutation pour DE
#%%
"""_________________________________________________________________________"""
# Domaine des longueurs d'ondes. Ici de 280 à 2500 nm par pas de 5 nm  
Wl = np.arange(280 , 2505, 5) #  #/!\. La valeur extréme supérieur est exclus. Il existe une fonction Wl_selectif() qui permet Wl de 280 à 2500 par pas de 5 nm, puit de 2500 à 30µm par pas de 50 nm# 
# Epaisseur du substrat, en nm 
Ep_Substrack = 1e6 # en nm 
# Plage des épaisseurs des couches minces, en nm
Plage_ep = (0, 350) # en nm. Plage indentique pour chaque couches minces pour générée la 1ere population 
# Plage des indices des couches minces
Plage_n = (1.3 , 3.0) # indice de réfraction, sans unité. Plage indentique pour chaque couches minces pour générée la 1ere population 
# Plage des fractions volumiques
Plage_vf = (0 , 1.0) #  volumic fraction of inclusion in host matrix
# Angle d'incidence du rayonnement sur le stack de couche mince 
Ang = 0 # en °
C = 80 # Concentration solaire
Tair = 20 + 273 # Température de l'air (Tair) et de l'absorbeur en Kelvin
Tabs = 300 + 273 # Température de l'absorbeur en Kelvin
# Longueur d'onde de coupure, uniquement pour les verre low-e ou le PV-CSP (fonction cout RTR)
Lambda_cut_UV = 500 # nm 
Lambda_cut_IR = 1000 # nm 
# Ouverture et traitement des matériaux réel 
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
# Ouverture du spectre solaire et traitement des matériaux réel 
Wl_sol , Sol_Spec , name_SolSpec = open_SolSpec('Materials/SolSpec.txt', 'GT')
Sol_Spec = np.interp(Wl, Wl_sol, Sol_Spec) # Interpolation du spectre
# Ouverture du fichier des cellules PV
Wl_PV , Signal_PV , name_PV = open_Spec_Signal('Materials/PV_cells.txt', 1)
Signal_PV = np.interp(Wl, Wl_PV, Signal_PV) # Interpolation du spectre
# Ouverture du fichier de l'absorbeur thermique /!\ en absorptance
Wl_Th , Signal_Th , name_Th = open_Spec_Signal('Materials/Thermal_absorber.txt', 1)
Signal_Th = np.interp(Wl, Wl_Th, Signal_Th) # Interpolation du spectre
# Algo génétique ou DE, hyperparamètres à changer
pop_size = 30 # nombre d'individus par génération 
crossover_rate = 0.9 # chance que deux parents échangent leurs gènes (1.0 = 100%)
evaluate_rate = 0.3 # part individus sélectionnés pour être les géniteurs des générations suivante
mutation_rate = 0.5 # chance qu'un gène d'un enfants mute à sa naissance. /!\ Cr pour DE
mutation_delta = 15 # Si un gène mute, le gène varie de + ou - un nombre aléatorie compris entre 0 et cette valeur. Mutation absolue fixe
f1, f2 = 0.9, 0.8  # Hyperparamétre pour DEvol
nb_generation = 200 # Nombre de génération. Pour DE sert aussi à calculer le budget, via nb_generation * pop_size
precision_AlgoG = 10**-5 # Précision sur le paramètres pour stoper l'optimisation
nb_lancement = 16 # Nombre de lancement 
cpu_used = 8  # Nombre de processeur utilisé /!\ à rester "raisonable"
#seed = 45 # Variable optionelle. Fixation du seed (graine du générateur de nombre aléatoire)
#%%
"""_________________________________________________________________________"""
# le dictionnaire parameters est un dictionnaire qui contient les variables du problèmes
# On donne le dictionnaire parameters comme entrée dans certaines fonctions
# => Elles vont ensuite chercher les variables nécessaires  
parameters = {'Wl': Wl, # Je stocke une variable nommée "Wl", et lui donne la valeur de Wl
            'Ang': Ang, 
            'C' : C,
            'T_air' : Tair,
            'T_abs' : Tabs,
            'Ep_Substrack' : Ep_Substrack,
            'Ep_plage' : Plage_ep,
            'd_Stack_Opt' : d_Stack_Opt,
            'Mat_Stack' : Mat_Stack,
            'SolSpec' : Sol_Spec,
            'Lambda_cut_min' : Lambda_cut_UV,
            'Lambda_cut' : Lambda_cut_IR,
            'n_Stack' : n_Stack,
            'k_Stack' : k_Stack,
            'pop_size': pop_size,
            'name_algo' : algo.__name__, 
            'name_selection' : selection.__name__, 
            'mutation_DE' : mutation_DE, 
            'crossover_rate' : crossover_rate,
            'evaluate_rate' : evaluate_rate,
            'mutation_rate': mutation_rate,
            'f1': f1,
            'f2': f2,
            'mutation_delta': mutation_delta,
            'Precision_AlgoG': precision_AlgoG,
            'nb_generation' :nb_generation,
            'Mod_Algo' : ("for"),} # Fin du dico 

#%%
# Si nb_layer exite, alors j'optimise une ou plusieurs couches minces théorique
# Je rajoute les valeurs dans le dictionnaire parameters (dictionnaire qui sert à transmettre les variables) 
if 'nb_layer' in locals():
    parameters["nb_layer"] = nb_layer
    parameters["n_plage"] = Plage_n
# si la variale seed existe, je la rajoute dans le dictionnaire. 
if 'seed' in locals():
    parameters["seed"] = seed
# Optimize a PV/CSP coating not with a RTR shape, but with a net energy balance
if evaluate.__name__ == "evaluate_netW_PV_CSP":
    if 'poids_PV' in locals():
        parameters['poids_PV'] = poids_PV
    else : 
        poids_PV = 3.0
        parameters['poids_PV'] = poids_PV
    # Interpolation 
    # Update the PV celle and the selective coating within the parameters dict
    parameters["Signal_PV"] = Signal_PV
    parameters["Signal_Th"] = Signal_Th 
        
if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
    parameters["vf_plage"] = Plage_vf
    
if 'Lambda_cut_UV' not in locals():
    Lambda_cut_UV = 0 # nm 
if 'Lambda_cut_IR' not in locals():
    Lambda_cut_IR = 0 # nm 
    
#%%
# création d'une fonction pour faire du multiprocessing
def run_problem_solution(i):
    t1 = time.time() # Temps avant l'exécution de la fonction
        # Ligne en dessous à activé pour désynchroniser légèrement les coeurs, si le seed est génèrer par lecture de l'horloge
    # time.sleep(np.random.random())
    # Lancement de l'algo d'optmisation (algo), selon une fonction coût (evaluate), une sélection (selection) en donnant toutes les infos nécessaires via le dictionnaire parameters
    best_solution, dev, nb_run, seed = algo(evaluate, selection, parameters)
    # calcul du temps utilisé
    t2 = time.time()
    temps = t2 - t1
    # best solution est un empillement. J'évalue cette empillement 
    if type(best_solution) != list:
        best_solution = best_solution.tolist()
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    perf = evaluate(best_solution, parameters)
    print("J'ai fini le cas n°", str(i+1), " en ", 
          "{:.1f}".format(temps), " secondes.",
          " Meilleur : ", "{:.4f}".format(perf),
          flush=True)
    return best_solution, perf, dev, nb_run, temps, seed
#%%
# Début de la boucle main. Le code doit être dans cette boucle pour fonctionner en multiprocessing 
if __name__=="__main__":
    
    print("Début du programme")
    launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
    print("Lancement à " + launch_time)
    print("Nombre de coeur détecté : ", cpu_count())
    print("Nombre de coeur utilisé : ", cpu_used)
    
    date_time = datetime.now().strftime("%Y-%m-%d-%Hh%M")
    dawn_of_time = time.time()
    
    # Ecriture du dossier de sauvegarde, à la date et heure du jour 
    directory = date_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("Le dossier '" + directory + "' a été créé.")
    
    Mat_Stack_print = Mat_Stack
    
    if 'nb_layer' in locals():
        for i in range(nb_layer):
            Mat_Stack_print = Mat_Stack_print + ["X"]
        
    print("Le stack est : ", Mat_Stack_print)
    
    if 'nb_layer' in locals():
        nb_total_layer = len(Mat_Stack) + nb_layer
        print("Le nombre total de couches minces est de : " + str(nb_total_layer))
    else: 
        nb_total_layer = len(Mat_Stack)
        print("Le nombre total de couches minces est de : " + str(len(Mat_Stack)))

    # Je créer un pool de problème réssoudre
    mp_pool = Pool(cpu_used)
    # Je ressous chaque problème du pool avec multiprocessing
    results = mp_pool.map(run_problem_solution, range(nb_lancement))
    
    # /!\ Dans Python, il faut créer des listes vides avant de rajouter des choses dedans 
    tab_best_solution = []
    tab_dev = []
    tab_perf = []
    tab_nb_run = []
    tab_temps = []
    tab_seed = []
    # for i in range(nb_lancement): # je lance nb_lancement fois le problèmes
    
    time.sleep(1) 
    # result est un tableau qui contient les solutions renvoyé par my.pool. Je les extrait pour les placé dans des tableaux différents
    for i, (best_solution, perf, dev, nb_run, temps, seed) in enumerate(results):
        # je rajoute les valeurs dans les tableaux 
        tab_perf.append(perf)
        tab_dev.append(dev)
        tab_best_solution.append(best_solution)
        tab_nb_run.append(nb_run) 
        tab_temps.append(temps)
        tab_seed.append(seed)
        #print(f"Cas n° {i+1}, fini en {temps:.1f} secondes.", " Meilleur : ", "{:.4f}".format(perf))

    end_of_time = time.time()
    time_real = end_of_time - dawn_of_time
    
    print("Le temps réel total est de : ", "{:.2f}".format(time_real), "secondes")
    print("Le temps réel de calcul processeur est de : ", "{:.2f}".format(sum(tab_temps)), "secondes")
    
    """___________________Données des meilleurs résultats______________________"""
    # Va chercher la meilleur valeur dans le tableau de toutes les performances
    max_value = max(tab_perf) # cherche le max
    max_index = tab_perf.index(max_value) # cherche l'index du max (où il est)
    
    # Je viens de trouver mon max, de tout mes run. C'est le meilleur des meilleurs ! Bravo ! 
    
    # Calcul Rs, Ts, As du max (performance solaire)
    Rs, Ts, As = evaluate_RTA_s(tab_best_solution[max_index], parameters) 
    # Calcul le R, T, A (Reflectance et cie, pour tracer une courbe)
    R, T, A = RTA_curve(tab_best_solution[max_index], parameters)
    # Je met au moin une valeur différente de 0 pour éviter une erreur lors du calcul de l'intégral
    if all(value == 0 for value in T):
        T[0] = 10**-301
    if all(value == 0 for value in R):
         R[0] = 10**-301
    if all(value == 0 for value in A):
        A[0] = 10**-301
    
    # En amont
    # Ouverture du spectre solaire 
    # Rappel : spectre GT => Spectre global, c-a-d le spectre du soleil + reflexion de l'environement 
    # Spectre GT  = Spectre DC (Direct) + Spectre Diffus
    # C'est le spectre que voit la surface
    Wl_sol , Sol_Spec , name_SolSpec = open_SolSpec('Materials/SolSpec.txt', 'GT')
    Sol_Spec = np.interp(Wl, Wl_sol, Sol_Spec)
    # Intégration du spectre solaire, brut en W/m2
    Sol_Spec_int = trapz(Sol_Spec, Wl)
    # Ecriture du spectre solaire modifié par la transmittance du traitement
    Sol_Spec_mod_T = T*Sol_Spec
    Sol_Spec_mod_T_int = trapz(Sol_Spec_mod_T, Wl)  # intégration du spectre solaire T, résultat en W/m2
    # Intégration du spectre solaire modifié par la réflectance du traitement, selon le spectre 
    Sol_Spec_mod_R = R*Sol_Spec
    Sol_Spec_mod_R_int = trapz(Sol_Spec_mod_R, Wl)  # intégration du spectre solaire R, résultat en W/m2
    # Intégration du spectre solaire modifié par l'absoprtance du traitement, selon le spectre 
    Sol_Spec_mod_A = A*Sol_Spec
    Sol_Spec_mod_A_int = trapz(Sol_Spec_mod_A, Wl)  # intégration du spectre solaire R, résultat en W/m2
    # Calcul de l'efficacité solaire en amont, par exemple l'efficacité de la cellule solaire Pv avec le spectre modifié
    Ps_amont = SolarProperties(Wl, Signal_PV, Sol_Spec_mod_T)
    # Calcul de l'efficacité solaire du traitement en amont avec un spectre non modifié
    Ps_amont_ref = SolarProperties(Wl, Signal_PV, Sol_Spec)
    # Calcul de l'intégration du spectr solaire utile en amont
    Sol_Spec_mod_amont = Sol_Spec * Signal_PV
    Sol_Spec_mod_amont_int = trapz(Sol_Spec_mod_amont, Wl)
    # Calcul de l'intégration du spectr solaire utile en amont
    Sol_Spec_mod_T_amont = Sol_Spec_mod_T * Signal_PV
    Sol_Spec_mod_T_amont_int = trapz(Sol_Spec_mod_T_amont, Wl)
    
    # En aval
    # Ouverture du spectre solaire, qui peut être différent du 1er en fonction des cas
    # Rappel : spectre DC => Spectre direct c-a-d uniquement le spectre du soleil, concentrable par un systeme optique 
    Wl_sol_2 , Sol_Spec_2 , name_SolSpec_2 = open_SolSpec('Materials/SolSpec.txt', 'DC')
    Sol_Spec_2 = np.interp(Wl, Wl_sol_2, Sol_Spec_2)
    # Intégration du spectre solaire, brut en W/m2
    Sol_Spec_int_2 = trapz(Sol_Spec_2, Wl)
    # Ecriture du spectre solaire modifié par la réflectance du traitement
    Sol_Spec_mod_R_2 = R*Sol_Spec_2
    Sol_Spec_mod_R_int_2 = trapz(Sol_Spec_mod_R_2, Wl) # intégration du spectre solaire R, résultat en W/m2
    # Calcul de l'efficacité solaire en aval, par exemple l'efficacité de l'absorbeur thermique
    Ps_aval = SolarProperties(Wl, Signal_Th, Sol_Spec_mod_R_2)
    # Calcul de l'efficacité solaire du traitement en aval avec un spectre non modifié
    Ps_aval_ref = SolarProperties(Wl, Signal_Th, Sol_Spec_2)
    # Calcul de l'intégration du spectr solaire utile en aval
    Sol_Spec_mod_aval = Sol_Spec_2 * Signal_Th
    Sol_Spec_mod_aval_int = trapz(Sol_Spec_mod_aval, Wl)
    # Calcul de l'intégration du spectr solaire utile en aval
    Sol_Spec_mod_R_aval = Sol_Spec_mod_R_2 * Signal_Th
    Sol_Spec_mod_R_aval_int = trapz(Sol_Spec_mod_R_aval, Wl)
    
    """________________________Ecriture des graphiques_________________________"""

    # Graph de réflectance 
    fig, ax1 = plt.subplots()
    color = 'black' # Couleurs de base possibles: b g r c m y k w
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectivity (-)', color=color)
    if evaluate.__name__ == 'evaluate_rh':
        ax1.set_xscale('log')
    ax1.plot(Wl, R, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Ligne de code pour changer l'axe de y, pour la réflectance
    # Désactivé pour échelle automatique
    
    ax1.set_ylim(0, 1) # changer l'échelle de l'axe y
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
    ax2.plot(Wl, Sol_Spec, color=color)
    if evaluate.__name__ == 'evaluate_rh':
        BB_shape = BB(Tabs, Wl)
        ## BB_shape est la forme du corps noir. En fonction de la température, l'irradiance du corps noir peut être tres supérieur
        # au spectre solair. Pour ce graphiquie, je met donc le corps noir à la meme hauteur 
        BB_shape =BB_shape*(max(Sol_Spec)/max(BB_shape))
        ax2.plot(Wl, BB_shape, color='orange', linestyle = 'dashed')
    
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  
    ax2.set_ylim(0, 2) # changer l'échelle de l'axe y
    plt.title("Optimum reflectivity")
    # Sauvegarde de la figure
    plt.savefig(directory + "/" + "Reflectance.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    # Graph de la transmittance
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
        BB_shape = BB(Tabs, Wl)
        ## BB_shape est la forme du corps noir. En fonction de la température, l'irradiance du corps noir peut être tres supérieur
        # au spectre solair. Pour ce graphiquie, je met donc le corps noir à la meme hauteur 
        BB_shape =BB_shape*(max(Sol_Spec)/max(BB_shape))
        ax2.plot(Wl, BB_shape, color='orange', linestyle = 'dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  
    ax2.set_ylim(0, 2) # changer l'échelle de l'axe y
    plt.title("Optimum transmissivity")
    
    plt.savefig(directory + "/" + "Transmittance.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    # Graph de la convergence
    # Je copie ma table de performance
    if (nb_lancement > 2): 
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
        plt.title("Convergence plots")
        plt.savefig(directory + "/" + "ConvergencePlots.png", dpi = 300, bbox_inches='tight')
        plt.show()
    
    # Je copie ma table de performance
    if (nb_lancement > 5): 
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
        plt.title("Convergence plots")
        plt.savefig(directory + "/" + "ConvergencePlots2.png", dpi = 300, bbox_inches='tight')
        plt.show()
    
    # Graph de la convergence du problème
    tab_perf_sorted = tab_perf.copy()
    tab_perf_sorted.sort(reverse = True)
    fig, ax1 = plt.subplots()
    color = 'black' # Couleurs de base possibles: b g r c m y k w
    ax1.set_xlabel('Best cases (left) to worse (right)')
    ax1.set_ylabel('Cost function (-)', color=color)
    ax1.plot(tab_perf_sorted, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.title("Consistency curve")
    plt.savefig(directory + "/" + "Consistency curve.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    # Graph des épaisseurs
    ep = tab_best_solution[max_index]
    if 'nb_layer' in locals():
        ep = np.delete(ep, np.s_[(nb_layer + len(Mat_Stack)):])
    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2: 
        vf = []
        vf = ep[len(Mat_Stack):len(ep)]
        ep = np.delete(ep, np.s_[(len(Mat_Stack)):len(ep)])
 
    #del epaisseur[0]
    lower = Plage_ep[0]
    upper = Plage_ep[1]
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
    plt.title("Thickness ")
    plt.savefig(directory + "/" + "Thickness.png", dpi = 300, bbox_inches='tight')
    plt.show()
    
    if 'nb_layer' in parameters:
        # Graph des indices
        n_list = tab_best_solution[max_index]
        for i in range(nb_layer + len(Mat_Stack)-1):
            n_list = np.delete(n_list, 0)
        #del epaisseur[0]
        lower = Plage_n[0]
        upper = Plage_n[1]
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
        ax.set_ylim((min(Plage_n)-0.5), (max(Plage_n)+0.5)) # changer l'échelle de l'axe y
        plt.xlabel("Number of layers, substrat to air")
        plt.ylabel("Refractive_Index (-)")
        plt.title("Refractive_Index ")
        plt.savefig(directory + "/" + "Refractive_Index.png", dpi = 300, bbox_inches='tight')
        plt.show()

    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        # Graph des fractions volumiques
        lower = Plage_vf[0]
        upper = Plage_vf[1]
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
        ax.set_ylim((min(Plage_vf)), (max(Plage_vf))) # changer l'échelle de l'axe y
        plt.xlabel("Number of layers, substrat to air")
        plt.ylabel("Volumic_Fraction (-)")
        plt.title("Volumic_Fraction ")
        plt.savefig(directory + "/" + "Volumic_Fraction.png", dpi = 300, bbox_inches='tight')
        plt.show()
    
    """_____________________Ecriture des résultats dans des txt_________________"""
    
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
    np.savetxt(directory + '/performance_dev.txt', tab_perf_dev, fmt='%.18e', delimiter='  ')
    
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
    np.savetxt(directory + '/performance_dev_2.txt', tab_perf_dev, fmt='%.18e', delimiter='  ')
    
    filename = directory + "/performance.txt"
    with open(filename, "w") as file:
        for value in tab_perf:
            file.write(str(value) + "\n")

    filename = directory + "/seed.txt"
    with open(filename, "w") as file:
        for value in tab_seed:
            file.write(str(value) + "\n")                   
    
    filename = directory + "/temps.txt"
    with open(filename, "w") as file:
        for value in tab_temps:
            file.write(str(value) + "\n")
    
    filename = directory + "/empillement.txt"
    with open(filename, "w") as file:
        for value in tab_best_solution:
            np.savetxt(file, value.reshape(1, -1), fmt='%.18e', delimiter=' ')
            
    filename = directory + "/dev.txt"
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
            
    print("Les résultats ont été écrits dans le dossier")
    
    filename = directory + "/RTA.txt"
    with open(filename, "w") as file:
        for i in range(len(A)):
            file.write(str(Wl[i]) + "\t" + str(R[i]) + "\t" + str(T[i]) + "\t" + str(A[i]) + "\n")
            
    print("Les données RTA du meilleur empillement ont été écrites dans cet ordre")
    
    filename = directory + "/simulation.txt"
    script_name = os.path.basename(__file__)
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
        file.write("Plage des épaisseur des couches minces\t" + str(Plage_ep[0]) + " à " + str(Plage_ep[1]) + " nm" + "\n")
        file.write("Plage des indices des couches minces\t" + str(Plage_n[0]) + " à " + str(Plage_n[1]) + "\n")
        file.write("Angle d'incidence sur le stack\t" + str(Ang) + "°" + "\n")
        file.write("Le taux de concentration est\t" + str(C) + "\n")
        file.write("La température de l'air est\t" + str(Tair) + " K" + "\n")
        file.write("La température de l'absorbeur' est\t" + str(Tabs) + " K" + "\n")
        if evaluate.__name__ == "evaluate_low_e" or evaluate.__name__ == "evaluate_RTR":
            file.write("Pour les profils d'optimisaiton low-e et RTR " + "\n")
            file.write("La longueur d'onde de coupure UV est \t" + str(Lambda_cut_UV) + " nm" + "\n")
            file.write("La longueur d'onde de coupure IR est \t" + str(Lambda_cut_IR) + " nm" + "\n")
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
        file.write("Nb de Lancement\t" + str(nb_lancement) + "\n")
        file.write("Nb de processeur disponible\t" +str(cpu_count()) + "\n")
        file.write("Nb de processeur utilisé\t" +str(cpu_used) + "\n")
        file.write("Le temps réel d'éxécution (en s) total est de :\t" + str("{:.2f}".format(time_real))  + "\n")
        file.write("La somme du temps de calcul (en s) processeur est de :\t" + str("{:.2f}".format(sum(tab_temps)) +  "\n"))
    
    print("Les noms et valeurs des variables de la simulation ont été écrites")
    
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
            if Lambda_cut_UV != 0 and Lambda_cut_IR != 0: 
                Wl_1 = np.arange(min(Wl),Lambda_cut_UV,(Wl[1]-Wl[0]))
                Wl_2 = np.arange(Lambda_cut_UV, Lambda_cut_IR, (Wl[1]-Wl[0]))
                Wl_3 = np.arange(Lambda_cut_IR, max(Wl)+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
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
    # J'écrit les indices des matériaux dans un fichier texte
    result = eliminate_duplicates(Mat_Stack)
    n_Stack_w, k_Stack_w = Made_Stack(result[0], Wl)
    
    for i in range(len(result[0])):
        name = result[0][i]
        n = n_Stack_w[:,i]
        k = k_Stack_w[:,i]
        # Trace le graphique
        plt.title("N et k de " + name)
        plt.plot(Wl, n, label = "n extrapolé")
        plt.plot(Wl, k, label = "k extrapolé")
    # =============================================================================
    #     # Ouvre le fichier originel
    #     Wl_2, n_2, k_2 = open_material(result[0][i])
    #     plt.plot(Wl_2, n_2, 'o', label = "n data")
    #     plt.plot(Wl_2, k_2, 'o', label = "k data")
    # =============================================================================
        plt.xlabel("Longueur d'onde (nm)")
        plt.legend()
        plt.ylabel("Valeur de n et k (-)")
        plt.savefig(directory + "/" + "refractive_index"+ name + ".png")
        plt.close()
        #filename = "/" + str(name) + ".txt"
        filename = directory + "/" + str(name) + ".txt"
        with open(filename, "w") as file:
            for i in range(len(n)):
                file.write(str(Wl[i]) + "\t" + str(n[i]) + "\t" + str(k[i])+ "\n")
                
    print("Les n,k de chaque matériaux ont été écrits dans un fichier à leurs noms")
