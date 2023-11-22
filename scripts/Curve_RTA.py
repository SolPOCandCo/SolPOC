# -*- coding: utf-8 -*-
"""
Created on 12 07 2023
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
import numpy as np
import time
from solpoc import *
import random 
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

# Date: 17 07 2023
Comment = "We can write a comment which will be printed into the txt file"
#Mat_Stack = write_stack_period(["Al_Rakic"], ["SiO2_I", "TiO2_I"], 4)
Mat_Stack = ["BK7", "SiO2", "ZnO-Al_Rakic", "Fe", "TiO2", "W-Al2O3"]

# Thin layers thickness in nm 
d_Stack = [1000000, 100, 50, 25, 100, 75] #Rmax 15

# volume fraction : mixing law
vf = [0, 0, 0.4, 0, 0, 0.7]
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
            'd_Stack' : d_Stack,
            'vf' : vf,
            'Wl' : Wl,
            'Th_Substrate' : d_Stack[0],
            'Mat_Stack' : Mat_Stack,
            'Sol_Spec' : Sol_Spec,
            'n_Stack' : n_Stack,
            'k_Stack' : k_Stack,}

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
# Calculate Rs, Ts, As of the maximum
Rs, Ts, As = evaluate_RTA_s(d_Stack, parameters) 
# Calculate the R, T, A
R, T, A = RTA_curve(d_Stack, parameters)

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