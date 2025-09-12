# -*- coding: utf-8 -*-
"""
SolPOC v0.9.7
"""
import numpy as np
import solpoc as sol

print("Start of the program")

# Wavelength domain, in nanometers: from 280 to 2500 nm with a 5 nm interval
Wl = np.arange(280, 2505, 5)

# Describe the thin layer stack. In SolPOC, it is unnecessary to add air (n=1) above the stack
Mat_Stack = ["BK7", "Al2O3", "Al", "Al2O3", "SiO2"]

# Incidence angle of light in the stack, in degrees
Ang = 0  

# Thickness of each thin layer, in nm. BK7 is the substrate, so its thickness is 1e6 nm (1 mm)
d_Stack = [1e6, 50, 300, 50, 200]
# So we have:
# 1 mm of BK7, 50 nm of Al2O3, 300 nm of Al, 50 nm of Al2O3, 200 nm of SiO2

d_Stack = np.array([d_Stack])

# Build the refractive index matrices
n_Stack, k_Stack = sol.Made_Stack(Mat_Stack, Wl)

# Calculate the spectral reflectivity, transmissivity, and absorptivity over all wavelengths for a single incidence angle
R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

print("End of the program")

