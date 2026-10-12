# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:00:55 2026

@author: agrosjean
"""
import solpoc as sol
import numpy as np

def evaluate_R_s_AOI_weighted(individual, parameters):
    """
    Calculates the weighted average solar reflectance of an individual
    for different angles of incidence.

    Parameters
    ----------
    individual : array
        Output of optimisation method (list of thickness in nm)

    parameters : dict
        Dictionary containing all parameters, including:
        - 'Ang_list': list of angles (deg)
        - 'weights': list of weights associated to each angle

    Returns
    -------
    R_s : float
        Weighted mean solar reflectance
    """

    Wl = parameters.get('Wl')
    Ang_list = parameters.get('Ang_list', [0, 10, 20, 30, 40])
    weights = parameters.get('weights', [1]*len(Ang_list))  # poids égaux par défaut

    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')

    # Transformation individu -> stack
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)

    # Vérification sécurité
    if len(Ang_list) != len(weights):
        raise ValueError("Ang_list and weights must have the same length")

    R_s_list = []

    for Ang, w in zip(Ang_list, weights):
        R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
        R_val = sol.SolarProperties(Wl, R, Sol_Spec)
        R_s_list.append(R_val * w)

    # Moyenne pondérée normalisée
    R_s = np.sum(R_s_list) / np.sum(weights)

    return R_s

def evaluate_RTR_AOI_weighted(individual, parameters):
    """
Calculates the performance according an RTR shape.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contain all parameters. 

Returns
-------
P_RTR: Int (float)
    Performance according an RTR shape.
    """
    Wl = parameters.get('Wl')
    
    Ang_list = parameters.get('Ang_list', [0, 10, 20, 30, 40])
    weights = parameters.get('weights', [1]*len(Ang_list))  # poids égaux par défaut

    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # The profile is reflective from 0 to Lambda_cut_1
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    # The profile is transparent from Lambda_cut_1 to Lambda_cut_1
    Lambda_cut_2 = parameters.get('Lambda_cut_2')
    # Treatment of the optimization of the n(s)
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack ?
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    Wl_1 = np.arange(min(Wl), Lambda_cut_1+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    Wl_2 = np.arange(Lambda_cut_1, Lambda_cut_2+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    
    d_Stack = d_Stack.reshape(1, len(individual))
    
    P_list = []

    # 🔥 BOUCLE MULTI-ANGLE (LE point clé)
    for Ang, w in zip(Ang_list, weights):

        R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
        P_low_e = np.concatenate([R[0:len(Wl_1)], T[len(Wl_1):(
            len(Wl_2)+len(Wl_1)-1)], R[(len(Wl_2)+len(Wl_1)-1):]])
        P_RTR = sol.SolarProperties(Wl, P_low_e, Sol_Spec)

        P_list.append(P_RTR * w)

    # Moyenne pondérée
    P_RTR = np.sum(P_list) / np.sum(weights)

    return P_RTR

def evaluate_TRT_AOI_weighted(individual, parameters):
    """
Calculates the performance according an RTR shape.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contain all parameters. 

Returns
-------
P_TRT: Int (float)
    Performance according an TRT shape.
    """
    Wl = parameters.get('Wl')
    
    Ang_list = parameters.get('Ang_list', [0, 10, 20, 30, 40])
    weights = parameters.get('weights', [1]*len(Ang_list))  # poids égaux par défaut

    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # The profile is reflective from 0 to Lambda_cut_1
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    # The profile is transparent from Lambda_cut_1 to Lambda_cut_1
    Lambda_cut_2 = parameters.get('Lambda_cut_2')
    # Treatment of the optimization of the n(s)
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack ?
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    Wl_1 = np.arange(min(Wl), Lambda_cut_1+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    Wl_2 = np.arange(Lambda_cut_1, Lambda_cut_2+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    
    d_Stack = d_Stack.reshape(1, len(individual))
    
    P_list = []

    # 🔥 BOUCLE MULTI-ANGLE (LE point clé)
    for Ang, w in zip(Ang_list, weights):

        R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
        P_low_e = np.concatenate([T[0:len(Wl_1)], R[len(Wl_1):(
            len(Wl_2)+len(Wl_1)-1)], T[(len(Wl_2)+len(Wl_1)-1):]])
        P_TRT = sol.SolarProperties(Wl, P_low_e, Sol_Spec)

        P_list.append(P_TRT * w)

    # Moyenne pondérée
    P_TRT = np.sum(P_list) / np.sum(weights)

    return P_TRT

def evaluate_T_in_RTR(individual, parameters):
    """
Calculates the transmitted solar flux between Lambda_cut_1 and Lambda_cut_2.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contain all parameters. 

Returns
-------
P_T: Int (float)
    Transmitted solar flux between Lambda_cut_1 and Lambda_cut_2.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    Lambda_cut_2 = parameters.get('Lambda_cut_2')
    Mat_Stack = parameters.get('Mat_Stack')

    """
    Why Individual_to_Stack ?
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)

    # Selection of the wavelength range of interest
    Wl_T = np.arange(Lambda_cut_1, Lambda_cut_2+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))

    # Calculation of the RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # Extraction of transmission in the selected range
    idx_1 = np.where(Wl >= Lambda_cut_1)[0][0]
    idx_2 = np.where(Wl <= Lambda_cut_2)[0][-1] + 1
    T_range = T[idx_1:idx_2]

    # Matching wavelength and spectrum range
    Wl_range = Wl[idx_1:idx_2]
    Sol_range = Sol_Spec[idx_1:idx_2]

    # Calculation of transmitted solar flux
    P_T = sol.SolarProperties(Wl_range, T_range, Sol_range)

    return P_T

def evaluate_R_in_TRT(individual, parameters):
    """
Calculates the reflected solar flux between Lambda_cut_1 and Lambda_cut_2.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contain all parameters. 

Returns
-------
P_R: Int (float)
    Reflected solar flux between Lambda_cut_1 and Lambda_cut_2.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    Lambda_cut_2 = parameters.get('Lambda_cut_2')
    Mat_Stack = parameters.get('Mat_Stack')

    """
    Why Individual_to_Stack ?
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)

    # Selection of the wavelength range of interest
    Wl_R = np.arange(Lambda_cut_1, Lambda_cut_2+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))

    # Calculation of the RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # Extraction of reflection in the selected range
    idx_1 = np.where(Wl >= Lambda_cut_1)[0][0]
    idx_2 = np.where(Wl <= Lambda_cut_2)[0][-1] + 1
    R_range = R[idx_1:idx_2]

    # Matching wavelength and spectrum range
    Wl_range = Wl[idx_1:idx_2]
    Sol_range = Sol_Spec[idx_1:idx_2]

    # Calculation of reflected solar flux
    P_R = sol.SolarProperties(Wl_range, R_range, Sol_range)

    return P_R

def evaluate_Rs_Tw(individual, parameters):
    """
Calculates normalized solar performance as:
- Reflectance over the full solar spectrum
- Transmittance between Lambda_cut_1 and Lambda_cut_2
Normalized by total solar flux to be between 0 and 1.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    Lambda_cut_2 = parameters.get('Lambda_cut_2')
    Mat_Stack = parameters.get('Mat_Stack')

    # Transform individual to stack
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)

    # Calculation of the RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # --- Flux réfléchi sur tout le spectre
    flux_R_total = np.trapz(R * Sol_Spec, Wl)

    # --- Flux transmis entre Lambda_cut_1 et Lambda_cut_2
    idx_1 = np.where(Wl >= Lambda_cut_1)[0][0]
    idx_2 = np.where(Wl <= Lambda_cut_2)[0][-1] + 1
    flux_T_range = np.trapz(T[idx_1:idx_2] * Sol_Spec[idx_1:idx_2], Wl[idx_1:idx_2])

    # --- Flux solaire total pour normalisation
    flux_total = np.trapz(Sol_Spec, Wl)

    # --- Performance normalisée
    P = (flux_R_total + flux_T_range) / flux_total
    return P

def evaluate_Ts_Rw(individual, parameters):
    """
Calculates normalized solar performance as:
- Transmittance over the full solar spectrum
- Reflectance between Lambda_cut_1 and Lambda_cut_2
Normalized by total solar flux to be between 0 and 1.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    Lambda_cut_2 = parameters.get('Lambda_cut_2')
    Mat_Stack = parameters.get('Mat_Stack')

    # Transform individual to stack
    d_Stack, n_Stack, k_Stack = sol.Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)

    # Calculation of the RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = sol.RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # --- Flux transmis sur tout le spectre
    flux_T_total = np.trapz(T * Sol_Spec, Wl)

    # --- Flux réfléchi entre Lambda_cut_1 et Lambda_cut_2
    idx_1 = np.where(Wl >= Lambda_cut_1)[0][0]
    idx_2 = np.where(Wl <= Lambda_cut_2)[0][-1] + 1
    flux_R_range = np.trapz(R[idx_1:idx_2] * Sol_Spec[idx_1:idx_2], Wl[idx_1:idx_2])

    # --- Flux solaire total pour normalisation
    flux_total = np.trapz(Sol_Spec, Wl)

    # --- Performance normalisée
    P = (flux_T_total + flux_R_range) / flux_total
    return P