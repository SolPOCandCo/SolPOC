import solpoc
import numpy as np
from pytest import fixture

from .fixture_data import get_DEvol_parameters, get_base_parameters


@fixture()
def base_parameters():
    return get_base_parameters()


@fixture
def DEvol_parameters():
    return get_DEvol_parameters()


def test_RTA3C():
    # Test case taken from function docstring

    # Write these variables :
    # We can notice that two wavelengths are calculated : 600 and 700 nm.
    l = np.arange(600, 750, 100)
    # 1 mm of substrate, 150 nm of n°1 layer, 180 of n°2 and empty space (n=1, k=0).
    d = np.array([[1000000, 150, 180]])
    n = np.array([[1.5, 1.23, 1.14], [1.5, 1.2, 1.1]])
    k = np.array([[0, 0.1, 0.05], [0, 0.1, 0.05]])
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
