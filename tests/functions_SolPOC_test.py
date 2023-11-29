import solpoc
import numpy as np

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

# TODO complete with tests for other functions in file