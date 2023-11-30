import solpoc
import numpy as np
from pytest import fixture

def test_RTA3C():
    # Test case taken from function docstring

    # Write these variables :
    l = np.arange(600,750,100) # We can notice that two wavelengths are calculated : 600 and 700 nm.
    d = np.array([[1000000, 150, 180]])  # 1 mm of substrate, 150 nm of n°1 layer, 180 of n°2 and empty space (n=1, k=0).
    n = np.array([[1.5, 1.23,1.14],[1.5, 1.2,1.1]])
    k = np.array([[0, 0.1,0.05], [0, 0.1, 0.05]])
    Ang = 0

    Refl, Trans, Abs = solpoc.RTA3C(l, d, n, k, Ang)

    assert np.allclose(Refl, np.array([0.00767694, 0.00903544]))
    assert np.allclose(Trans, np.array([0.60022613, 0.64313401]))
    assert np.allclose(Abs, np.array([0.39209693, 0.34783055]))


def test_get_seed_from_randint():
    # no args
    seed = solpoc.get_seed_from_randint()
    assert seed.dtype == 'uint32'
    # with size argument
    seed_list = solpoc.get_seed_from_randint(size=5)
    assert seed_list.shape[0] == 5
    assert seed_list.dtype == 'uint32'
    # with rng input
    rng = np.random.RandomState(24)
    seed = solpoc.get_seed_from_randint(rng=rng)
    assert seed.dtype == 'uint32'


def test_get_seed_for_run():
    out_call1 = solpoc.get_seed_for_run(0, 1, 24)
    out_call2 = solpoc.get_seed_for_run(0, 1, 24)
    assert np.all(np.equal(out_call1, out_call2))

    out_call1 = solpoc.get_seed_for_run(5, 10, 24)
    out_call2 = solpoc.get_seed_for_run(5, 10, 24)
    assert np.all(np.equal(out_call1, out_call2))


@fixture()
def base_parameters():
    parameters = {'seed' : 3245298}
    # Wavelength domain, here from 320 to 2500 nm with a 5 nm step. Can be change!   
    Wl = np.arange(280 , 2505, 5) # /!\ Last value is not included in the array
# Thickness of the substrate, in nm 
    # Open the solar spectrum
    Wl_Sol , Sol_Spec , name_Sol_Spec = solpoc.open_SolSpec('Materials/SolSpec.txt','GT')
    # Open a file with PV cell shape
    Wl_PV , Signal_PV , name_PV = solpoc.open_Spec_Signal('Materials/PV_cells.txt', 1)
    # Open a file with thermal absorber shape
    Wl_Th , Signal_Th , name_Th = solpoc.open_Spec_Signal('Materials/Thermal_absorber.txt', 1)
    # Open and interpol the refractive index
    Mat_Stack = ["BK7", "TiO2", "SiO2"]
    n_Stack, k_Stack = solpoc.Made_Stack(Mat_Stack, Wl)
    # Open and processing the reflectif index of materials used in the stack (Read the texte files in Materials/ )
    Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec) # Interpolate the solar spectrum 
    Signal_PV = np.interp(Wl, Wl_PV, Signal_PV) # Interpolate the signal
    Signal_Th = np.interp(Wl, Wl_Th, Signal_Th) # Interpolate the signal 
    parameters.update({
        'Wl':  Wl, # Thickness of the substrate, in nm 
        'Ang': 0, # Incidence angle on the thin layers stack, in °
        'C' : 80, # Solar concentration. Data necessary for solar thermal application, like selective stack
        'T_abs' : 300 + 273, # Thermal absorber temperature, in Kelvin. Data necessary for solar thermal application, like selective stack 
        'T_air' : 20 + 273, # Air temperature, in Kelvin. Data necessary for solar thermal application, like selective stack
        'Sol_Spec' : Sol_Spec,
        'name_Sol_Spec' : name_Sol_Spec,
        'Th_range' : (0, 200), # in nm.,
        'Th_Substrate' :  1e6, # Substrate thickness, in nm 
        'Signal_PV' : Signal_PV,
        'Signal_Th' : Signal_Th,
        'Mat_Stack' : Mat_Stack,
        # Cuting Wavelenght. Data necessary for low-e, RTR or PV_CSP evaluates functions
        'Lambda_cut_1' : 800, # nm
        'Lambda_cut_2' : 1000, # nm
        # Range of refractive index (lower bound and upper bound), for the optimisation process
        'n_range' : (1.3 , 3.0),
        'n_Stack' : n_Stack,
        'k_Stack' : k_Stack,
        })
    

    return parameters

@fixture
def DEvol_parameters():
    parameters = {}
    # Hyperparameters for optimisation methods
    # Choice of optimisation method
    algo = solpoc.DEvol # Name of the optimization method 
    selection = solpoc.selection_max # Callable. Name of the selection method : selection_max or selection_min
    evaluate = solpoc.evaluate_T_s # Callable. Name of the cost function
    parameters.update({
        'pop_size': 30, # number of individual per iteration / generation ,
        'algo' : algo, # Name of the optimization method ,
        'name_algo' : algo.__name__,
        'evaluate' : evaluate, # Callable. Name of the cost function,
        'selection' : selection, # Callable. Name of the selection method : selection_max or selection_min,
        'name_selection' : selection.__name__, 
        'mutation_rate': 0.5, # chance of child gene muted during the birth. /!\ This is Cr for DEvol optimization method
        'mutation_delta' : 15, # If a chromosome mutates, the value change from random number include between + or - this values
        'nb_generation' :4, # Number of generation/iteration. For DEvol is also used to calculate the budget (nb_generation * pop_size)
        'mutation_DE': "current_to_best",
        'f1':0.9, 'f2':0.8,  # Hyperparameter for DEvol 
    })
    return parameters

def test_DEvol(base_parameters, DEvol_parameters):
    # only tests if it is running
    parameters = base_parameters
    parameters.update(DEvol_parameters)

    best_solution, dev, n_iter, seed = parameters['algo'](
        parameters['evaluate'], 
        parameters['selection'], 
        parameters)

    best_solution = np.array(best_solution)
    perf = parameters['evaluate'](best_solution, parameters)

# TODO complete with tests for other functions in file