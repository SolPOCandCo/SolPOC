# -*- coding: utf-8 -*-
"""
Created on 12 07 2023
SolPOC v 0.9.6
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
import numpy as np
import time
from solpoc import *
import os
from datetime import datetime
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
#----------------------------------------------------------------------------#
# %%  Main : You can start to modified something
# Comment to be written in the simulation text file
Comment = "We can write a comment which will be printed into the txt file"
# Write the thin layer stack, to subtrat to ambiant 
Mat_Stack = ["BK7", "SiO2", "ZnO-Al_Rakic", "Fe", "TiO2", "W-Al2O3"]
# Thin layers thickness in nm  /!\ First thickness is for substrat
d_Stack = [1e6, 100, 50, 25, 100, 75] #Rmax 15
# volume fraction for the Effective Medium Approximation - Theory (mixing law)
vf = [0, 0, 0.4, 0, 0, 0.7]
# Wavelengths domain. From 280 to 2500 nm with a 5 nm interval 
Wl = np.arange(280,2505,5) # /!\. The extreme upper value is excluded
# Angle of Incidence (AOI) of the radiation on the stack. 0 degrees is for normal incidence angle
Ang = 0 # in °
# Materials opening and treatment
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
n_Stack, k_Stack = Made_Stack_vf(n_Stack, k_Stack, vf)
# Solar spectrum opening, here ASTM Global Tilt (GT) Write DC for Direct and Circumsolar
Wl_Sol , Sol_Spec , name_Sol_Spec= open_SolSpec('Materials/SolSpec.txt', 'GT')
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
"""_________________________________________________________________________"""
# parameters is a dictionary which includes the problem variables
# We set parameters dictionary as an input for some functions
# => Then, they go chekc necesary variables   

parameters = {'Wl': Wl, # I store a variable named "Wl", and give it Wl value
            'Ang': Ang,
            'd_Stack' : d_Stack,
            'vf' : vf,
            'Wl' : Wl,
            'Th_Substrate' : d_Stack[0],
            'Mat_Stack' : Mat_Stack,
            'Sol_Spec' : Sol_Spec,
            'n_Stack' : n_Stack,
            'k_Stack' : k_Stack,
            'coherency_limit' : 2000,}

Experience_results = ({
    'd_Stack' : d_Stack,
    'vf' : vf,})

print("Start of the program")
launch_time = datetime.now().strftime("%Hh-%Mm-%Ss")
print("Launch at " + launch_time)

date_time = datetime.now().strftime("%Y-%m-%d-%Hh%M")
directory = date_time
if not os.path.exists(directory):
    os.makedirs(directory)
print("The '" + directory + "' directory has been created")

"""___________________Best results datas______________________"""

t1 = time.time()

# Calculate the R, T, A
R, T, A = RTA_curve_inco(d_Stack, parameters)
# Calculate Rs, Ts, As of the maximum
Rs, Ts, As = evaluate_RTA_s(d_Stack, parameters) 

# Writing of the modified solar spectrum
Sol_Spec_mod = R*Sol_Spec
t2 = time.time()

"""________________________Graphs writing_________________________"""

Stack_plot(parameters, Experience_results, directory)

"""________________Results writing in txt files___________________"""

# Reflectance graph 
fig, ax1 = plt.subplots()
color = 'black' # Basic colors available: b g r c m y k w
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
ax2.set_ylim(0, 2) # Change y-axis scale
plt.title("Optimum reflectivity")
# Figure save
plt.savefig(directory + "/" + "Reflectance.png", dpi = 300, bbox_inches='tight')
plt.show()

# Transmitance graph
fig, ax1 = plt.subplots()
color = 'black' # Basic colors available: b g r c m y k w
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
ax2.set_ylim(0, 2) # Change y-axis scale
plt.title("Optimum transmissivity")
plt.savefig(directory + "/" + "Transmittance.png", dpi = 300, bbox_inches='tight')
plt.show()

filename = directory + "/RTA.txt"
with open(filename, "w") as file:
    for i in range(len(A)):
        file.write(str(Wl[i]) + "\t" + str(R[i]) + "\t" + str(T[i]) + "\t" + str(A[i]) + "\n")
        
        
filename = directory + "/Stacks.txt"
with open(filename, "w") as file:
    file.write("Material in the thin layers stack" + "\n")
    file.write(str(Mat_Stack) +  "\n")
    file.write("Thin layers thicknesses" + "\n")
    file.write(str(d_Stack))

print("Reflectivity, transmissivity and absorptivity (RTA) is saved")

# J'écrit les indices des matériaux dans un fichier texte
result = eliminate_duplicates(Mat_Stack)
n_Stack_w, k_Stack_w = Made_Stack(result[0], Wl)

for i in range(len(result[0])):
    name = result[0][i]
    n = n_Stack_w[:,i]
    k = k_Stack_w[:,i]
    # Trace le graphique
    plt.title("N and k from" + name)
    plt.plot(Wl, n, label = "n extrapoled")
    plt.plot(Wl, k, label = "k extrapoled")
# =============================================================================
#     # Ouvre le fichier originel
#     Wl_2, n_2, k_2 = open_material(result[0][i])
#     plt.plot(Wl_2, n_2, 'o', label = "n data")
#     plt.plot(Wl_2, k_2, 'o', label = "k data")
# =============================================================================
    plt.xlabel("Wavelenght (nm)")
    plt.legend()
    plt.ylabel("Value of n and k (-) ")
    plt.savefig(directory + "/" + "refractive_index"+ name + ".png")
    plt.close()
    #filename = "/" + str(name) + ".txt"
    filename = directory + "/" + str(name) + ".txt"
    with open(filename, "w") as file:
        for i in range(len(n)):
            file.write(str(Wl[i]) + "\t" + str(n[i]) + "\t" + str(k[i])+ "\n")
            
print("Refractive index of each material is saved")