# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 12:14:57 2025

@author: agrosjean
"""

import numpy as np
from scipy.optimize import minimize
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
import solpoc as sol
import time

# Objective function for the anti-reflection coating
def AR_coating_tmm_nm(d_list, M, theta, wl, Sol_Spec):
    """
    Objective function for the anti-reflection coating using TMM (thicknesses in nanometers)
    This version expects layer thicknesses in nanometers for better numerical stability.
    The substrate and air are treated as semi-infinite layers.
    
    Parameters:
       d_list: 1D array of thin layer thicknesses to optimize (excluding substrate and air), in nm ! 
       M: refractive index matrix (num_samples, num_layers, num_wavelengths)
       theta: incidence angles (radians)
       wl: wavelengths (meters)
       Sol_Spec: solar spectrum interpolated on wl
    """

    T = np.zeros((1, 5))

    # Assign optimized thicknesses
    T[:, 1:-1] = d_list *1e-9 # assuming d_list shape = (num_layers-2,)

    # Substrate and air are semi-infinite
    T[:, 0] = 1e6
    T[:, -1] = np.inf
    
    # Compute TMM for s- and p-polarizations
    O_s = tmm('s', M, T, theta, wl, device='cpu')
    O_p = tmm('p', M, T, theta, wl, device='cpu')

    # Average reflectivity
    R_s = O_s['R'][0, 0, :]
    R_p = O_p['R'][0, 0, :]
    R = (R_s + R_p) / 2

    # Compute solar reflectivity
    
    Rs = sol.SolarProperties(wl, R, Sol_Spec)
  
    return Rs

# Objective function for the anti-reflection coating
def AR_coating_tmm(d_list, M, theta, wl, Sol_Spec):
    """
    Objective function for the anti-reflection coating using TMM package
    This version expects layer thicknesses in meters (no unit conversion), as expected by tmm.
    The function constructs the thickness matrix T for TMM, assigning:
       - Optimized thicknesses to the thin layers (d_list)
       - Semi-infinite values for the substrate (T[:, 0]) and air (T[:, -1])
    It then computes the reflectivity for both s- and p-polarizations,
    averages them, and calculates the solar reflectivity by weighting
    the spectrum with the solar irradiance (Sol_Spec).
    # 
    Parameters:
       d_list: 1D array of thin layer thicknesses to optimize (excluding substrate and air)
       M: refractive index matrix (num_samples, num_layers, num_wavelengths)
       theta: incidence angles (radians)
       wl: wavelengths (meters)
       Sol_Spec: solar spectrum interpolated on wl
    """

    T = np.zeros((1, 5))

    # Assign optimized thicknesses
    T[:, 1:-1] = d_list  # assuming d_list shape = (num_layers-2,)

    # Substrate and air are semi-infinite
    T[:, 0] = 1e6
    T[:, -1] = np.inf
    
    # Compute TMM for s- and p-polarizations
    O_s = tmm('s', M, T, theta, wl, device='cpu')
    O_p = tmm('p', M, T, theta, wl, device='cpu')

    # Average reflectivity
    R_s = O_s['R'][0, 0, :]
    R_p = O_p['R'][0, 0, :]
    R = (R_s + R_p) / 2

    # Compute solar reflectivity
    
    Rs = sol.SolarProperties(wl, R, Sol_Spec)
  
    return Rs

# ------------------------ Parameters ------------------------
wl = np.arange(280, 2505, 5) * 1e-9
theta = np.array([0]) * (np.pi/180)        # incidence angle in radians
num_layers = 5                              # includes substrate and air

# Refractive indices for each layer
M = np.ones((1, num_layers, wl.shape[0]))
M[:, 0, :] = 1.5  # substrate
M[:, 1, :] = 1.7  # layer 1
M[:, 2, :] = 1.47  # layer 2
M[:, 3, :] = 1.37 # layer 3

# Solar spectrum
Wl_Sol, Sol_Spec_data, name_Sol_Spec = sol.open_SolSpec('Materials/SolSpec.txt', 'GT')
Sol_Spec = np.interp(wl* 1e9, Wl_Sol, Sol_Spec_data)

# ------------------------ Initial thicknesses ------------------------
"""
Depending of the method, It can be s better to keep the thicknesses in nanometers for the optimization.
When using meters, the values are very small (~1e-7), which can cause
numerical instability in gradient-based optimizers like BFGS.
This is why SolPOC uses nanometers for layer thicknesses.
"""
max_t = 300  # 300 nm
min_t = 0    # 0 nm

# Only thin layers are optimized (exclude substrate and air)
np.random.seed(512)
x_AR = np.random.uniform(min_t, max_t, num_layers - 2)  

# ------------------------ Optimization ------------------------
t1 = time.time()
result_AR = minimize(
    AR_coating_tmm_nm, 
    x_AR, 
    args=(M, theta, wl, Sol_Spec), 
    method='BFGS'
)
t2 = time.time()

print("TMM basic solar anti-reflective coating:")
print("Time calcul", str(t2-t1), "seconde")
print("Success:", result_AR.success)
print("Optimized thicknesses [m]:", result_AR.x)
print("Average solar reflectivity:", result_AR.fun)

# ------------------------ Initial thicknesses ------------------------
max_t = 300*1e-9  # 300 nm
min_t = 0 * 1e-9   # 0 nm

# Only thin layers are optimized (exclude substrate and air)
np.random.seed(512)
x_AR = np.random.uniform(min_t, max_t, num_layers - 2)  
# ------------------------ Optimization ------------------------
t1 = time.time()
result_AR = minimize(
    AR_coating_tmm, 
    x_AR, 
    args=(M, theta, wl, Sol_Spec), 
    method='BFGS'
)
t2 = time.time()

print("TMM basic solar anti-reflective coating:")
print("Time calcul", str(t2-t1), "seconde")
print("Success:", result_AR.success)
print("Optimized thicknesses [m]:", result_AR.x)
print("Average solar reflectivity:", result_AR.fun)
