from importlib import metadata
try:
    __version__ = metadata.version(__package__)
    # Note: the metadata method returns an object representing an email-like message
    # payload corresponds to it's content, which is equal to the PKG_INFO file 
    # within the egg-info folder
    _metadata = metadata.metadata(__package__).get_payload()
    # there is a blank line separating the readme part from the "header" of the PKG-INFO
    # so we split the content at that line and get the second part
    _summary = _metadata.split('\n\n',1)[1]  # get readme part of the metadata

    __doc__ = f"""
    Version {__version__}  

    {_summary}
    """
except metadata.PackageNotFoundError:
    pass


from .functions_SolPOC import acceptance_probability
from .functions_SolPOC import BB
from .functions_SolPOC import Bruggeman
from .functions_SolPOC import chi_square
from .functions_SolPOC import children_strangle
from .functions_SolPOC import Consistency_curve_plot
from .functions_SolPOC import Convergence_plots
from .functions_SolPOC import Convergence_plots_2
from .functions_SolPOC import Convergences_txt
from .functions_SolPOC import crossover
from .functions_SolPOC import DEvol
from .functions_SolPOC import DEvol_Video
from .functions_SolPOC import E_BB
from .functions_SolPOC import eliminate_duplicates
from .functions_SolPOC import equidistant_values
from .functions_SolPOC import evaluate_A_pv
from .functions_SolPOC import evaluate_A_s
from .functions_SolPOC import evaluate_EBB
from .functions_SolPOC import evaluate_example
from .functions_SolPOC import evaluate_fit_R
from .functions_SolPOC import evaluate_fit_RT
from .functions_SolPOC import evaluate_fit_T
from .functions_SolPOC import evaluate_fit_T2face
from .functions_SolPOC import evaluate_low_e
from .functions_SolPOC import evaluate_netW_PV_CSP
from .functions_SolPOC import evaluate_R
from .functions_SolPOC import evaluate_R_Brg
from .functions_SolPOC import evaluate_R_s
from .functions_SolPOC import evaluate_rh
from .functions_SolPOC import evaluate_RTA_s
from .functions_SolPOC import evaluate_RTR
from .functions_SolPOC import evaluate_T
from .functions_SolPOC import evaluate_T_pv
from .functions_SolPOC import evaluate_T_s
from .functions_SolPOC import evaluate_T_vis
from .functions_SolPOC import Explain_results
from .functions_SolPOC import Explain_results_fit
from .functions_SolPOC import Generate_materials_txt
from .functions_SolPOC import generate_mutant
from .functions_SolPOC import generate_neighbor
from .functions_SolPOC import Generate_perf_rh_txt
from .functions_SolPOC import generate_population
from .functions_SolPOC import Generate_txt
from .functions_SolPOC import get_seed_from_randint
from .functions_SolPOC import helio_th
from .functions_SolPOC import Individual_to_Stack
from .functions_SolPOC import interpolate_with_extrapolation
from .functions_SolPOC import Made_SolCORE_Stack
from .functions_SolPOC import Made_Stack
from .functions_SolPOC import Made_Stack_vf
from .functions_SolPOC import mutation
from .functions_SolPOC import nb_compo
from .functions_SolPOC import normalized_mse
from .functions_SolPOC import One_plus_One_ES
from .functions_SolPOC import open_material
from .functions_SolPOC import open_SolSpec
from .functions_SolPOC import open_Spec_Signal
from .functions_SolPOC import OpticalStackResponse_plot
from .functions_SolPOC import Optimization_txt
from .functions_SolPOC import optimize_ga
from .functions_SolPOC import optimize_strangle
from .functions_SolPOC import Optimum_refractive_index_plot
from .functions_SolPOC import Optimum_thickness_plot
from .functions_SolPOC import PSO
from .functions_SolPOC import Reflectivity_plot
from .functions_SolPOC import Reflectivity_plot_fit
from .functions_SolPOC import RTA
from .functions_SolPOC import RTA3C
from .functions_SolPOC import RTA_curve
from .functions_SolPOC import RTA_curve_inco
from .functions_SolPOC import selection_max
from .functions_SolPOC import selection_min
from .functions_SolPOC import simulated_annealing
from .functions_SolPOC import Simulation_amont_aval_txt
from .functions_SolPOC import SolarProperties
from .functions_SolPOC import Stack_coherency
from .functions_SolPOC import Stack_plot
from .functions_SolPOC import Transmissivity_plot
from .functions_SolPOC import Transmissivity_plot_fit
from .functions_SolPOC import valeurs_equidistantes
from .functions_SolPOC import Volumetric_parts_plot
from .functions_SolPOC import Wl_selectif
from .functions_SolPOC import write_stack_period 

from .cli import init