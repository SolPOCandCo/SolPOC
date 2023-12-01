# test for multiprocessing

import numpy as np
import solpoc
from .fixture_data import get_DEvol_parameters, get_base_parameters

from multiprocessing import Pool

# elements that go through the multiproc pool need to be declared at top level
mp_parameters = {}
mp_parameters.update(get_base_parameters())
mp_parameters.update(get_DEvol_parameters())
nb_run = 10

# Test 1: test seed control of the parallel runs
mp_parameters['seed_list'] = solpoc.get_seed_from_randint(
    nb_run,
    rng=np.random.RandomState(mp_parameters['seed']))


def single_run(i):
    this_run_params = {}
    this_run_params.update(mp_parameters)
    this_run_params['seed'] = mp_parameters['seed_list'][i]
    return this_run_params['seed']


def test_multiprocessesing_seeds():
    global mp_parameters, nb_run

    # ensure same list is generated
    initial_rng = np.random.RandomState(mp_parameters['seed'])
    seed_list_initial = solpoc.get_seed_from_randint(nb_run, initial_rng)

    assert np.all(np.equal(seed_list_initial, mp_parameters['seed_list']))

    # ensure seeds reach the different processes
    # we make them return the seed they got,
    # then check if we get the same seeds in the same order
    pool = Pool(2)
    seed_list_after_map = pool.map(single_run, range(nb_run))
    assert np.all(np.equal(
        seed_list_initial, seed_list_after_map))

# Test 2: test launching one optim algo in parallel with the controled seeds
# ensure results are reproduced


def one_DEvol_run(i):
    this_run_params = {}
    this_run_params.update(mp_parameters)
    this_run_params['seed'] = mp_parameters['seed_list'][i]

    best_solution, dev, n_iter, seed = this_run_params['algo'](
        this_run_params['evaluate'],
        this_run_params['selection'],
        this_run_params)

    best_solution = np.array(best_solution)

    return best_solution


def test_multiprocessesing_reprod_results():
    # ensure same solution is found when executing pool map on the same inputs
    global mp_parameters, nb_run

    pool = Pool(2)
    solution1 = pool.map(one_DEvol_run, range(nb_run))
    solution2 = pool.map(one_DEvol_run, range(nb_run))

    assert np.all(np.equal(
        solution1, solution2))
