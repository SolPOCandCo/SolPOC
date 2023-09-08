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
Comment = "We can write a comment which will be printed into the txt file"
#Mat_Stack = write_stack_period(["Al_Rakic"], ["SiO2_I", "TiO2_I"], 4)
Mat_Stack = ("BK7", "TiO2_I", "ZnO-Al_Rakic", "Ag_Babar", "ZnO-Al_Rakic", "TiO2_I", "SiO2_I")
# Thin layers thickness in nm 
d_Stack = [1000000, 30, 5, 8, 5, 22, 67]
# volume fraction : mixing law
vf = [0, 0, 0.7, 0, 0.5, 0, 0]
# Wavelengths domain. From 280 to 2500 nm with a 5 nm interval 
Wl = np.arange(280,2505,5) # /!\. The extreme upper value is excluded
# Incidence angle of the radiation on the stack
Ang = 0 # in °
# Materials opening and treatment
n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
n_Stack, k_Stack = Made_Stack_vf(n_Stack, k_Stack, vf)
# Solar spectrum opening
Wl_Sol , Sol_Spec , name_Sol_Spec= open_SolSpec('Materials/SolSpec.txt')
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)

"""_________________________________________________________________________"""
# parameters is a dictionary which includes the problem variables
# We set parameters dictionary as an input for some functions
# => Then, they go chekc necesary variables   

parameters = {'Wl': Wl, # I store a variable named "Wl", and give it Wl value
            'Ang': Ang, 
            'Th_Substrate' : d_Stack[0],
            'Mat_Stack' : Mat_Stack,
            'Sol_Spec' : Sol_Spec,
            'n_Stack' : n_Stack,
            'k_Stack' : k_Stack,}

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
# Calculate Rs, Ts, As of the maximum
Rs, Ts, As = evaluate_RTA_s(d_Stack, parameters) 
# Calculate the R, T, A
R, T, A = RTA_curve(d_Stack, parameters)

# Writing of the modified solar spectrum
Sol_Spec_mod = R*Sol_Spec
t2 = time.time()

"""________________________Graphs writing_________________________"""

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

"""_____________________Results writing in txt files_________________"""

filename = directory + "/performance.txt"
with open(filename, "w") as file:
    file.write("Solar Reflectance " + str(Rs) + "\n")
    file.write("Solar Transmittance " + str(Ts) + "\n")
    file.write("Solar Absorptance " + str(As) + "\n")

filename = directory + "/temps.txt"
with open(filename, "w") as file:
    file.write(" Temps " + str((t2 - t1)) + "\n")

filename = directory + "/StacksThicknesses.txt"
with open(filename, "w") as file:
    file.write(str(d_Stack) + "\n")
        
print("Results have been successfully writed in the directory")

filename = directory + "/RTA.txt"
with open(filename, "w") as file:
    for i in range(len(A)):
        file.write(str(Wl[i]) + "\t" + str(R[i]) + "\t" + str(T[i]) + "\t" + str(A[i]) + "\n")
        
print("RTA datas of the best stack have been writed in this order")

filename = directory + "/simulation.txt"
script_name = os.path.basename(__file__)
with open(filename, "w") as file:
    file.write("File name is : " + str(script_name) + "\n")
    file.write("Launch time is " + str(launch_time) + "\n")
    file.write(str(Comment) + "\n")
    file.write("_____________________________________________" +  "\n")
    file.write("Stack materials\t" + str(Mat_Stack) + "\n")
    file.write("Wavelengths domain\t" + str(min(Wl)) + " nm at " + str(max(Wl)) + " nm, interval of " + str(Wl[1]-Wl[0])+ " nm"+ "\n")
    file.write("Substrate thickness\t" + str(d_Stack[0]) + "nm" +  "\n")
    file.write("Thin layers thickness\t" + str(d_Stack[1:]) + " nm" + "\n")
    file.write("Incidence angle on the stack\t" + str(Ang) + "°" + "\n")

print("Variables names and values pf the simulation have been writed")

# I write materials indexes in a text file
result = eliminate_duplicates(Mat_Stack)
n_Stack_w, k_Stack_w = Made_Stack(result[0], Wl)

for i in range(len(result[0])):
    name = result[0][i]
    n = n_Stack_w[:,i]
    k = k_Stack_w[:,i]
    # plot the graph
    plt.title("N et k de" + name)
    plt.plot(Wl, n, label = "n extrapolé")
    plt.plot(Wl, k, label = "k extrapolé")
# =============================================================================
#     # Open the initial file
#     Wl_2, n_2, k_2 = open_material(result[0][i])
#     plt.plot(Wl_2, n_2, 'o', label = "n data")
#     plt.plot(Wl_2, k_2, 'o', label = "k data")
# =============================================================================
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.ylabel("n and k values (-)")
    plt.savefig(directory + "/" + "refractive_index"+ name + ".png", dpi = 300, bbox_inches='tight')
    plt.close()
    #filename = "/" + str(name) + ".txt"
    filename = directory + "/" + str(name) + ".txt"
    with open(filename, "w") as file:
        for i in range(len(n)):
            file.write(str(Wl[i]) + "\t" + str(n[i]) + "\t" + str(k[i])+ "\n")
            
print("The n,k of each material have been writd in a file named as them")





    
