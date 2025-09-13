# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 10:41:51 2025
@author: agrosjean
"""

import numpy as np
from scipy.optimize import minimize
import solpoc as sol
import time 

# Objective function with an additional argument n
def square(x, n):
    return np.sum((x - 0.5)**n)

# Starting point
x0 = np.array([0.0, 1.0, 2.0])

# Pass n=2 as an extra argument via 'args'
result = minimize(square, x0, args=(2,), method='BFGS')

print("Scipy basic example:")
print("Success:", result.success)
print("Optimal x found:", result.x)
print("Minimum value:", result.fun)


# Objective function for the anti-reflection coating
def AR_coating(d_Stack, n_Stack, k_Stack, Wl, Ang, Sol_Spec):
    # Convert d_Stack (received as ndarray) to list
    d_list = list(d_Stack)

    # Add the substrate thickness (BK7, 1e6 nm) at the beginning
    d_full = [1e6] + d_list

    # Make it a 2D array (shape (1, n_layers))
    d_Stack = np.array([d_full])
    
    # Calculate spectral reflectivity, transmissivity, and absorptivity
    # over all wavelengths for a single incidence angle
    R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calculate the solar reflectivity
    Rs = sol.SolarProperties(Wl, R, Sol_Spec)
  
    return Rs

# Wavelength domain, in nanometers: from 400 to 800 nm with 5 nm step
Wl = np.arange(280, 2505, 5)

# Describe the thin layer stack.
# In SolPOC, it is not necessary to add air (n=1) above the stack
Mat_Stack = ["n15", "n17", "n147", "n137"]

# Incidence angle of light in the stack, in degrees
Ang = 0  

# Get refractive index (n) and extinction coefficient (k)
n_Stack, k_Stack = sol.Made_Stack(Mat_Stack, Wl)

Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec('Materials/SolSpec.txt', 'GT')
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)

np.random.seed(512)
# Starting point for the layer thicknesses to optimize (in nm)
x_AR = np.random.uniform(low=0, high=300, size=3)

t1 = time.time()
# Run the optimization
result_AR = minimize(
    AR_coating, 
    x_AR, 
    args=(n_Stack, k_Stack, Wl, Ang, Sol_Spec), 
    method= 'BFGS'
)
t2 = time.time()
print("SolPOC basic solar antireflectif coating : ")
print("Time calcul", str(t2-t1), "seconde")
print("Success:", result_AR.success)
print("Optimized thicknesses [nm]:", result_AR.x)
print("Average solar reflectivity:", result_AR.fun)
