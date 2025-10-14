# -*- coding: utf-8 -*-
"""
Created on 2025-09-05
SolPOC v 0.9.7
@authors: A.Grosjean (main author, EPF, France), A.Soum-Glaude (PROMES-CNRS, France), A.Moreau (UGA, France) & P.Bennet (UGA, France)
contact : antoine.grosjean@epf.fr
"""

import numpy as np
import time
from solpoc import open_SolSpec, RTA, SolarProperties
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
import nevergrad as ng

def DBR_solar_tmm(d_Stack):
    """
    Compute the solar performance of a Distributed Bragg Reflector (DBR) thin-film stack 
    using the tmm-fast package 
    
    This function:
        - Builds an optical stack made of alternating low-index (n = 1.46) and high-index (n = 2.56) layers.
        - Adds semi-infinite air layers (n = 1.0) on top and bottom of the stack.
        - Defines the thickness of each layer from the input array `d_Stack`.
        - The incidence is normal 
        - Computes the spectral transmittance for both s and p polarizations.
        - Evaluates the solar-weighted transmission using the ASTM G173-03 solar spectrum (stored in 'Sol_Spec.npy') using SolPOC function according to the SolarPaces guidelines (280–2500 nm, 5 nm steps).
    
    Args:
        d_Stack (array-like): Thicknesses (in nanometers) of the DBR thin-film layers.
                              Odd indices use n = 1.46 and even indices use n = 2.56.
    
    Returns:
        float: Solar performance metric computed by the `SolarProperties` function,
               based on the calculated transmission spectrum.
    
    Notes:
        - Wavelengths are sampled from 280 to 2500 nm.
        - The optimizer is set to minimize the cost function. Since we aim to maximize reflectance,
          the cost is defined as the transmittance.
        - Air (n = 1.0) is automatically added on both sides of the stack.
        - The `tmm` function is used to calculate transmittance for both polarizations.
    """
    wl = np.linspace(280, 2500, 445) * (10**(-9))
    # See above for download the a solar spectra with SolPOC
    Sol_Spec = np.load('Sol_Spec.npy') # ASTM G173-03 DC solar spectra, with a 5 nm step
    theta = np.linspace(0, 0, 1) * (np.pi/180)
    mode = 'T'
    num_layers = len(d_Stack) + 2 # need to add 1 "layer" in tmm, for include the air above and upper the thin layer stack
    num_stacks = 1 
    #create m
    M = np.ones((num_stacks, num_layers, wl.shape[0]))
    # Add air, n = 1
    M[:, 0, :] = 1.0

    # Thin layer stack, alternation of low and high refractive index 
    for i in range(1, M.shape[1]-1): 
        if i % 2 == 1:
            M[:, i, :] = 1.46  
        else:
            M[:, i, :] = 2.56
    # Add air, n = 1
    M[:, -1, :] = 1.0
    #print(M[0,:,0])
    #create T : the thin layers thicknesses
    T = np.zeros((num_stacks, num_layers))
    T[:, 0] = np.inf      # Air = inf
    
    # Thin layer thickness
    for i in range(1, M.shape[1]-1):
        if i % 2 == 1:  # indices impairs → n=2.4
            T[:, i] = d_Stack[i-1] * (10**(-9))
        else:           # indices pairs → n=1.5
            T[:, i] = d_Stack[i-1] * (10**(-9))
    T[:, -1] = np.inf     # Air = inf
    # In TMM-fast, the transmission must be computed separately for each polarization.
    # Here we compute 's' and 'p' polarizations individually and then average them
    # to get the total spectral transmittance.
    O_s = tmm('s', M, T, theta, wl, device='cpu')
    O_p = tmm('p', M, T, theta, wl, device='cpu')
    T_spec = (O_s['T'] + O_p['T']) / 2
    # We aim to maximize the reflectance of the stack. 
    # Since the optimizer minimizes a cost function, we define the cost as the solar-weighted transmittance.
    # Minimizing `f_cout` therefore corresponds to maximizing the reflectance.
    f_cout = SolarProperties(wl, T_spec[0, 0, :], Sol_Spec)

    return f_cout


def DBR_solar_SolPOC(d_Stack):
    """
    Compute the solar performance of a Distributed Bragg Reflector (DBR) thin-film stack 
    using using the SolPOC framework.
    
    This function:
        - Builds an optical stack made of alternating low-index (n = 1.46) and high-index (n = 2.56) layers.
        - Adds a air layers (n = 1.0) at bottom of the stack.
        - Defines the thickness of each layer from the input array `d_Stack`.
        - The incidence is normal 
        - Computes the spectral transmittance at s and p polarizations.
        - Evaluates the solar-weighted transmission using the ASTM G173-03 solar spectrum (stored in 'Sol_Spec.npy') using SolPOC function according to the SolarPaces guidelines (280–2500 nm, 5 nm steps).
    Args:
        d_Stack (array-like): Thicknesses (in nanometers) of the thin-film layers.
    
    Returns:
        float: Solar-weighted transmittance, used as a cost function for optimization.
    
    Notes:
        - Wavelengths are sampled from 280 to 2500 nm.
        - The optimizer is set to minimize the cost function. Since we aim to maximize reflectance,
          the cost is defined as the transmittance.
         - Air (n = 1.0) is AUTOMATICALLY upper the stack in the RTA function.
         - Air (n = 1.0) is MUST BE added above sides of the stack.
        - `n_Stack` and `k_Stack` represent the optical properties (real and imaginary refractive indices)
          of each layer in the stack.
    """
    
    # Wavelenths according the SolarPaces guidelines. Do not change
    Wl = np.arange(280,2505,5)
    # ASTM G173-03 DC solar spectra, with a 5 nm step
    Sol_Spec = np.load('Sol_Spec.npy')
    # Working the proper shape
    d_Stack = d_Stack.reshape(1,d_Stack.shape[0])
    d_Stack = np.insert(d_Stack, 0, 1000, axis=1) # We 
    # Creation onf n_Stack and k_Stack, array with n (real part of refractie index) et k (complexe part)
    n_Stack, k_Stack = make_DBR_stack(d_Stack.shape[1], Wl)
    Ang = 0  # Incidence angle
    # Plot the R, T, A curve
    # n_Stack and k_Stack are ncessary for 
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang);
    # We aim to maximize the reflectance of the stack. 
    # Since the optimizer minimizes a cost function, we define the cost as the solar-weighted transmittance.
    # Minimizing `f_cout` therefore corresponds to maximizing the reflectance.
    f_cout = SolarProperties(Wl, T, Sol_Spec)

    return f_cout

def make_DBR_stack(d_stack, Wl):
    """
    Create a Distributed Bragg Reflector (DBR) stack of alternating SiO2 and TiO2 layers.
    The function builds refractive index (n) and extinction coefficient (k) stacks 
    for a given number of layers, alternating between:
        - SiO2 like (n = 1.46, k = 0)
        - TiO2 like (n = 2.56, k = 0)
    The first layer (substrate) is set to n = 1.5 and k = 0.
    Args:
        len_stack (int): Number of alternating layers (SiO2/TiO2).
        Wl (array-like): Wavelength grid (used to define the shape of the output arrays).
    Returns:
        tuple of np.ndarray:
            - n_Stack (ndarray): Refractive indices for each layer at each wavelength (shape: [len(Wl), len_stack+1]).
            - k_Stack (ndarray): Extinction coefficients for each layer at each wavelength (same shape).
    """
    
    n_Stack = np.zeros((len(Wl), d_stack))
    k_Stack = np.zeros((len(Wl), d_stack))
    
    # First layer (add a thin layer of air)
    n_Stack[:, 0] = 1.0
    k_Stack[:, 0] = 0

    # Alternate SiO2 like and TiO2 like layers
    # In SolPOC, is not necessary to include a layer of air above the thin layer stack, it's include automaticly in the optical properties calculation, in RTA function 
    for i in range(1, d_stack):
        if i % 2 != 0:  # Odd index: SiO2
            n_Stack[:, i] = 1.46
            k_Stack[:, i] = 0
        else:            # Even index: TiO2
            n_Stack[:, i] = 2.56
            k_Stack[:, i] = 0
    
    return n_Stack, k_Stack


"""
For create the Sol_Spec.npy, just one time the following line : 
Think to update the folder for the np.save ! 
    
Wl_Sol, Sol_Spec, name_Sol_Spec = open_SolSpec('SolSpec.txt')
Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec) 
np.save("./Sol_Spec.npy", Sol_Spec)
"""
num_layer = 4
Wl = np.arange(280,2505,5) # creat Wl
budget = 100

print("Launch of the programm with tmm package")
np.random.seed(42)
t1 = time.time() # Time before the optimisation process
# run nevergrad
instrum = ng.p.Instrumentation(ng.p.Array(shape=(num_layer,)).set_bounds(lower=0, upper=300))
optimizer = ng.optimizers.DE(parametrization=instrum, budget=budget) # We use DE optimizer, in nevergrad
recommendation = optimizer.minimize(DBR_solar_tmm)  # best value
# calcul the time used
t2 = time.time()
temps = t2 - t1
print("Time calculation for tmm", t2 -t1, " s")
print("Value of cost function for tmm:", DBR_solar_tmm(recommendation.value[0][0]))    
print("Solution (Thin layers thicknesses, in nm) for tmm " ,recommendation.value[0][0])

# run nevergrad
print("\n")
print("Launch of the programm with SolPOC package")
np.random.seed(42)
t1 = time.time() # Time before the optimisation process
# run nevergrad
instrum = ng.p.Instrumentation(ng.p.Array(shape=(num_layer,)).set_bounds(lower=0, upper=300))
optimizer = ng.optimizers.DE(parametrization=instrum, budget=budget)
recommendation = optimizer.minimize(DBR_solar_SolPOC)  # best value
# calcul the time used
t2 = time.time()
temps = t2 - t1
print("Time calculation for SolPOC:", t2 -t1, " s")
print("Value of cost function for SolPOC:", DBR_solar_SolPOC(recommendation.value[0][0]))    
print("Solution (Thin layers thicknesses, in nm) with SolPOC " ,recommendation.value[0][0])
