# -*- coding: utf-8 -*-
"""
Created on 2025-09-05
SolPOC v 0.9.6
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""
import numpy as np
import solpoc as sol

print("Start of the program")
"""
This basic example shows you how to use the elementary functions of SolPOC.
It guides you through the calculation of the reflectivity, transmissivity, and absorptivity of a 4-layer 
thin film stack (Al₂O₃, Al, Al₂O₃, and SiO₂) on BK7 glass.
"""
# Wavelength domain, in nanometers: from 280 to 2500 nm with a 5 nm interval
Wl = np.arange(280, 2505, 5)
"""
Describe the thin layer stack. The first material (index 0 in the list) is the substrate. 
In SolPOC, it is unnecessary to add air (n=1) above the stack.
SolPOC will automatically search its library for materials identified by a string.
If the material is not found, it will try to read it locally from refractive index data 
provided as a text file placed in the "Material" folder.
"""
Mat_Stack = ["BK7", "Al2O3", "Al", "Al2O3", "SiO2"]
# Build the refractive index matrices
n_Stack, k_Stack = sol.Made_Stack(Mat_Stack, Wl)

# Thickness of each thin layer, in nm. BK7 is the substrate, so its thickness is 1e6 nm (1 mm)
# So we have: 1 mm of BK7, 50 nm of Al2O3, 300 nm of Al, 50 nm of Al2O3, 200 nm of SiO2
d_Stack = [1e6, 30, 50, 30, 200]
# Reshape the thickness list
d_Stack = np.array([d_Stack])

# Incidence angle of light in the stack, in degrees
Ang = 0  
# Calculate the spectral reflectivity (R), transmissivity (T), and absorptivity (A) over all wavelengths for a single incidence angle
R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

print("End of the program :)")

