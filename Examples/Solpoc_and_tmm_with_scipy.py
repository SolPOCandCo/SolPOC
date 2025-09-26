# -*- coding: utf-8 -*-
"""
Created on 2025-09-13
SolPOC v 0.9.6
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
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

# Objective function for the anti-reflection coating
def AR_coating_solpoc(d_Stack, n_Stack, k_Stack, Wl, Ang, Sol_Spec):
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
    AR_coating_solpoc, 
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
