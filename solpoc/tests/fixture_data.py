import numpy as np
import solpoc


def get_base_parameters():
    parameters = {'seed': 3245298}
    # Wavelength domain, here from 320 to 2500 nm with a 5 nm step. Can be change!
    Wl = np.arange(280, 2505, 5)  # /!\ Last value is not included in the array
# Thickness of the substrate, in nm
    # Open the solar spectrum
    Wl_Sol, Sol_Spec, name_Sol_Spec = solpoc.open_SolSpec(
        'Materials/SolSpec.txt', 'GT')
    # Open a file with PV cell shape
    Wl_PV, Signal_PV, name_PV = solpoc.open_Spec_Signal(
        'Materials/PV_cells.txt', 1)
    # Open a file with thermal absorber shape
    Wl_Th, Signal_Th, name_Th = solpoc.open_Spec_Signal(
        'Materials/Thermal_absorber.txt', 1)
    # Open and interpol the refractive index
    Mat_Stack = ["BK7", "TiO2", "SiO2"]
    n_Stack, k_Stack = solpoc.Made_Stack(Mat_Stack, Wl)
    # Open and processing the reflectif index of materials used in the stack (Read the texte files in Materials/ )
    # Interpolate the solar spectrum
    Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)
    Signal_PV = np.interp(Wl, Wl_PV, Signal_PV)  # Interpolate the signal
    Signal_Th = np.interp(Wl, Wl_Th, Signal_Th)  # Interpolate the signal
    parameters.update({
        'Wl':  Wl,  # Thickness of the substrate, in nm
        'Ang': 0,  # Incidence angle on the thin layers stack, in °
        'C': 80,  # Solar concentration. Data necessary for solar thermal application, like selective stack
        'T_abs': 300 + 273,  # Thermal absorber temperature, in Kelvin. Data necessary for solar thermal application, like selective stack
        # Air temperature, in Kelvin. Data necessary for solar thermal application, like selective stack
        'T_air': 20 + 273,
        'Sol_Spec': Sol_Spec,
        'name_Sol_Spec': name_Sol_Spec,
        'Th_range': (0, 200),  # in nm.,
        'Th_Substrate':  1e6,  # Substrate thickness, in nm
        'Signal_PV': Signal_PV,
        'Signal_Th': Signal_Th,
        'Mat_Stack': Mat_Stack,
        # Cuting Wavelenght. Data necessary for low-e, RTR or PV_CSP evaluates functions
        'Lambda_cut_1': 800,  # nm
        'Lambda_cut_2': 1000,  # nm
        # Range of refractive index (lower bound and upper bound), for the optimisation process
        'n_range': (1.3, 3.0),
        'n_Stack': n_Stack,
        'k_Stack': k_Stack,
    })

    return parameters


def get_DEvol_parameters():
    parameters = {}
    # Hyperparameters for optimisation methods
    # Choice of optimisation method
    algo = solpoc.DEvol  # Name of the optimization method
    # Callable. Name of the selection method : selection_max or selection_min
    selection = solpoc.selection_max
    evaluate = solpoc.evaluate_T_s  # Callable. Name of the cost function
    parameters.update({
        'pop_size': 30,  # number of individual per iteration / generation ,
        'algo': algo,  # Name of the optimization method ,
        'name_algo': algo.__name__,
        'evaluate': evaluate,  # Callable. Name of the cost function,
        # Callable. Name of the selection method : selection_max or selection_min,
        'selection': selection,
        'name_selection': selection.__name__,
        # chance of child gene muted during the birth. /!\ This is Cr for DEvol optimization method
        'mutation_rate': 0.5,
        # If a chromosome mutates, the value change from random number include between + or - this values
        'mutation_delta': 15,
        # Number of generation/iteration. For DEvol is also used to calculate the budget (nb_generation * pop_size)
        'nb_generation': 4,
        'mutation_DE': "current_to_best",
        'f1': 0.9, 'f2': 0.8,  # Hyperparameter for DEvol
    })
    return parameters
