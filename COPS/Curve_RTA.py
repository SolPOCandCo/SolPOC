# -*- coding: utf-8 -*-
"""
Created on 12 07 2023
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
import numpy as np
import time
from functions_COPS import *
import random 
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Date: 17 07 2023
Comment = " On peut écrire ici un commentaire, qui seras imprimé dans le fichier txt"
#Mat_Stack = Ecrit_Stack_Periode(["Al_Rakic"], ["SiO2_I", "TiO2_I"], 4)
Mat_Stack = ("BK7", "TiO2_I", "ZnO-Al_Rakic", "Ag_Babar", "ZnO-Al_Rakic", "TiO2_I", "SiO2_I")
# Epaisseur des couche mince en nm 
d_Stack = [1000000, 30, 5, 8, 5, 22, 67]
# volume fraction : loi de mélange (EMA)
vf = [0, 0, 0.7, 0, 0.5, 0, 0]
# Domaine des longueurs d'ondes. Ici de 280 à 2500 nm par pas de 5 nm  
Wl = np.arange(280,2505,5) # /!\. La valeur extréme supérieur est exclus
# Angle d'incidence du rayonnement sur le stack
Ang = 0 # en °
# Ouverture et traitement des matériaux
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
n_Stack, k_Stack = Made_Stack_vf(n_Stack, k_Stack, vf)
# Ouverture du spectre solaire
Wl_sol , Sol_Spec , name_SolSpec= open_SolSpec('Materials/SolSpec.txt')
Sol_Spec = np.interp(Wl, Wl_sol, Sol_Spec)

"""_________________________________________________________________________"""
# parameters est un dictionnaire qui contient les variables du problèmes
# On donne le dictionnaire parameters comme entrée dans certaines fonctions
# => Elles vont ensuite chercher les variables nécessaires   

parameters = {'Wl': Wl, # Je stocke une variable nommée "Wl", et lui donne la valeur de Wl
            'Ang': Ang, 
            'Ep_Substrack' : d_Stack[0],
            'Mat_Stack' : Mat_Stack,
            'SolSpec' : Sol_Spec,
            'n_Stack' : n_Stack,
            'k_Stack' : k_Stack,}

# /!\ Dans Python, il faut créer des listes vides avant de rajouter des choses dedans 

print("Début du programme")
launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
print("Lancement à " + launch_time)

date_time = datetime.now().strftime("%Y-%m-%d-%Hh%M")
directory = date_time
if not os.path.exists(directory):
    os.makedirs(directory)
print("Le dossier '" + directory + "' a été créé.")

"""___________________Données des meilleurs résultats______________________"""

t1 = time.time()
# Calcul Rs, Ts, As du max
Rs, Ts, As = evaluate_RTA_s(d_Stack, parameters) 
# Calcul le R, T, A
R, T, A = RTA_curve(d_Stack, parameters)

# Ecriture du spectre solaire modifié
Sol_Spec_mod = R*Sol_Spec
t2 = time.time()

"""________________________Ecriture des graphiquyes_________________________"""

# Graph de réflectance 
fig, ax1 = plt.subplots()
color = 'black' # Couleurs de base possibles: b g r c m y k w
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Reflectivity (-)', color=color)
ax1.plot(Wl, R, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
ax2.plot(Wl, Sol_Spec, color=color)
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
ax1.plot(Wl, T, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
ax2.plot(Wl, Sol_Spec, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
ax2.set_ylim(0, 2) # changer l'échelle de l'axe y
plt.title("Optimum transmissivity")
plt.savefig(directory + "/" + "Transmittance.png", dpi = 300, bbox_inches='tight')
plt.show()

"""_____________________Ecriture des résultats dans des txt_________________"""

filename = directory + "/performance.txt"
with open(filename, "w") as file:
    file.write("Solar Reflectance " + str(Rs) + "\n")
    file.write("Solar Transmittance " + str(Ts) + "\n")
    file.write("Solar Absorptance " + str(As) + "\n")

filename = directory + "/temps.txt"
with open(filename, "w") as file:
    file.write(" Temps " + str((t2 - t1)) + "\n")

filename = directory + "/empillement.txt"
with open(filename, "w") as file:
    file.write(str(d_Stack) + "\n")
        
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
    file.write("Matériaux de l'empillement\t" + str(Mat_Stack) + "\n")
    file.write("Domaine des longueurs d'ondes \t" + str(min(Wl)) + " nm à " + str(max(Wl)) + " nm, pas de " + str(Wl[1]-Wl[0])+ " nm"+ "\n")
    file.write("Epaisseur du substrat\t" + str(d_Stack[0]) + "nm" +  "\n")
    file.write("Epaisseur des couches minces\t" + str(d_Stack[1:]) + " nm" + "\n")
    file.write("Angle d'incidence sur le stack\t" + str(Ang) + "°" + "\n")

print("Les noms et valeurs des variables de la simulation ont été écrites")

# J'écrit les indices des matériaux dans un fichier texte
result = eliminate_duplicates(Mat_Stack)
n_Stack_w, k_Stack_w = Made_Stack(result[0], Wl)

for i in range(len(result[0])):
    name = result[0][i]
    n = n_Stack_w[:,i]
    k = k_Stack_w[:,i]
    # Trace le graphique
    plt.title("N et k de" + name)
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
    plt.savefig(directory + "/" + "refractive_index"+ name + ".png", dpi = 300, bbox_inches='tight')
    plt.close()
    #filename = "/" + str(name) + ".txt"
    filename = directory + "/" + str(name) + ".txt"
    with open(filename, "w") as file:
        for i in range(len(n)):
            file.write(str(Wl[i]) + "\t" + str(n[i]) + "\t" + str(k[i])+ "\n")
            
print("Les n,k de chaque matériaux ont été écrits dans un fichier à leurs noms")





    
