# -*- coding: utf-8 -*-
"""
Created on 27 07 2023
SolPOC v 0.9.6
@author: A.Grosjean, A.Soum-Glaude, A.Moreau & P.Bennet
Stack_plot function by Titouan Fevrier
contact : antoine.grosjean@epf.fr

A specific presentation of the docstrings has been added to allow Sphinx software to generate a full documentation of this document

List of main functions used and developed for SolPOC. For use them without any implementation work see the other Python script
"""

import importlib
from importlib import resources as impresources
import numpy as np
import math
# trapz renamed as trapezoid since scipy 1.14.0
try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid
from scipy.interpolate import interp1d
import random
import os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from solcore.structure import Structure
from solcore.absorption_calculator import calculate_rat


def RTA3C(Wl, d, n, k, Ang=0):
    """
Exemple of RTA.
\nRTA is the key function : it's allow us to calcul reflectivity (R) transmissivity (T) and absorptivity(A). 
This function is an exemple working only with 3 thin layers on the substrat. 
This tutorial functions is not used in the code.

The RTA function calculates the reflectivity, the transmissivity, and the absorptivity of a group of thin layers applied on a substrate
as reflectivity + transmissivity + absorptivity = 1.
\nThe number after 'RTA' means the amount of thin layers applied on the substrate (we count the substrate in the amount). The RTA 3C code is logically for 1 substrate and 2 thin layers maximum

The numpy (np) utilisation is a time saver on the launch !
=> the time saved compared to the launch of an RTA function for one wavelength in a for loop is about 100 times faster !

-Input arguments :
\nl : The wavelength in nm, has a vector type.
\nd : Substrate and thin layers thickness in nm.
\nn : Real parts of materials refraction indexes. n is a 2D table with the thin layers indexes in the columns and the wavelengths in the rows.
\nk : Complex parts of material refraction indexes. k is a 2D table with the thin layers extinction coefficients in the columns and the wavelengths in the rows.
\nAng : Incidence angle of the radiation in degrees.

-Output :
\nRefl is a column vector which includes the reflectivity.
    Column indexes corresponds to wavelengths. 
\nTrans is a column vector which includes the transmissivity.
    Column indexes corresponds to wavelengts'
\nAbs is a column vector which includes the absorptivity.
    Column indexes corresponds to wavelengths.

-Test :
\n# Write these variables :
\nl = np.arange(600,750,100). We can notice that two wavelengths are calculated : 600 and 700 nm.
\nd = np.array([[1000000, 150, 180]]). 1 mm of substrate, 150 nm of n°1 layer, 180 of n°2 and empty space (n=1, k=0).
\nn = np.array([1.5, 1.23,1.14],[1.5, 1.2,1.1]]).
\nk = np.array([[0, 0.1,0.05], [0, 0.1, 0.05]]).
\nAng = 0.

# Run function
\nRefl, Trans, Abs = RTA3C(l, d, n, k, Ang).

For the indexes notation n and k, understand that:
\n@ 600 nm, n = 1.5 for the substrate, n = 1.23 for the layer n°1 and n = 1.14 for the layer n°2.
\n@ 700 nm, n = 1.5 for the substrate, n = 1.20 for the layer n°1 and n = 1.1 for the layer n°2.
\n@ 600 nm, k = 0 for the substrate, k = 0.1 for the layer n°1 and k = 0.05 for the layer n°2.
\n@ 700 nm, k = 0 for the substrate, k = 0.1 for the layer n°1 and k = 0.05 for the layer n°2.

-We can get :
\nRefl = array([0.00767694, 0.00903544]).
\nTrans = array([0.60022613, 0.64313401]).
\nAbs = array([0.39209693, 0.34783055]).
\n=> The reflectivity is 0.00767694 (number between 0 and 1) at 600 nm and 0.00903544 at 700 nm.
    """
    # Add an air layer on top
    # replacement of 2 by len(l)
    n = np.append(n, np.ones((len(Wl), 1)), axis=1)
    # replacement of 2 by len(l)
    k = np.append(k, np.zeros((len(Wl), 1)), axis=1)
    d = np.append(d, np.zeros((1, 1)), axis=1)

    # Incidence angle of solar radiation on the absorber = normal incidence
    Phi0 = Ang*math.pi/180

    # Ambient medium = vaccum
    n0 = 1
    k0 = 0
    N0 = n0 + 1j*k0
    q0PolaS = N0*np.cos(Phi0)
    q0PolaP = N0/np.cos(Phi0)

    # Substrate
    # I take the 1st column which includes the n of the substrate for wavelengths
    nS = n[:, 0]
    kS = k[:, 0]
    Ns = nS + 1j*kS
    PhiS = np.arcsin(N0*np.sin(Phi0)/Ns)
    qSPolaS = Ns*np.cos(PhiS)
    qSPolaP = Ns/np.cos(PhiS)  # Ok here

    # Multilayers (layer 1 is the closest one to the substrate)
    nj = np.delete(n, 0, axis=1)
    kj = np.delete(k, 0, axis=1)
    dj = np.delete(d, 0, axis=1)

    numlayers = nj.shape[1]  # nj is just a table
    Nj = np.zeros((numlayers, 1, len(Wl)), dtype=complex)  # OK

    # was a column in Scilab, row here
    Phij = np.zeros((numlayers, 1, len(Wl)), dtype=complex)
    qjPolaS = np.zeros((numlayers, 1, len(Wl)), dtype=complex)
    qjPolaP = np.zeros((numlayers, 1, len(Wl)), dtype=complex)
    thetaj = np.zeros((numlayers, 1, len(Wl)), dtype=complex)
    MpolaS = np.zeros((2, 2*numlayers, len(Wl)), dtype=complex)
    MpolaP = np.zeros((2, 2*numlayers, len(Wl)), dtype=complex)
    Ms = np.zeros((2, 2, len(Wl)), dtype=complex)
    Mp = np.zeros((2, 2, len(Wl)), dtype=complex)

    sous_tableaux = np.split(nj, nj.shape[1], axis=1)
    # el.reshape(1,2) becomes el.reshape(1,len(l))
    nj = np.array([el.reshape(1, len(Wl)) for el in sous_tableaux])
    sous_tableaux = np.split(kj, kj.shape[1], axis=1)
    kj = np.array([el.reshape(1, len(Wl)) for el in sous_tableaux])

    dj = np.squeeze(dj)
    # Note  : invert table with numpy.transpose()
    for LayerJ in range(numlayers):
        Nj[LayerJ] = nj[LayerJ] + 1j * kj[LayerJ]
        Phij[LayerJ] = np.arcsin(N0 * np.sin(Phi0) / Nj[LayerJ])
        qjPolaS[LayerJ] = Nj[LayerJ] * np.cos(Phij[LayerJ])
        qjPolaP[LayerJ] = Nj[LayerJ] / np.cos(Phij[LayerJ])
        thetaj[LayerJ] = (2 * np.pi / Wl) * dj[LayerJ] * \
            Nj[LayerJ] * np.cos(Phij[LayerJ])  # OK

        # Characteristic matrix of layer j
        """ Calcul of MpolaS"""
        MpolaS[0, 2*LayerJ] = np.cos(thetaj[LayerJ]
                                     )  # In Scilab MpolaS(1,2*LayerJ-1)
        MpolaS[0, 2*LayerJ+1] = -1j/qjPolaS[LayerJ] * \
            np.sin(thetaj[LayerJ])  # In Scilab MpolaS(1,2*LayerJ)
        MpolaS[1, 2*LayerJ] = -1j*qjPolaS[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaS[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        """ Calculation of MpolaP"""
        MpolaP[0, 2*LayerJ] = np.cos(thetaj[LayerJ])
        MpolaP[0, 2*LayerJ+1] = -1j/qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ] = -1j*qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        # print(MpolaS)

    # Global characteristic (transfer) matrix [Furman92, Andersson80]
    if numlayers == 1:  # Substrate only
        M1s = np.array([[MpolaS[0, 0], MpolaS[0, 1]],
                       [MpolaS[1, 0], MpolaS[1, 1]]])
        M1p = np.array([[MpolaP[0, 0], MpolaP[0, 1]],
                       [MpolaP[1, 0], MpolaP[1, 1]]])
        Ms = M1s
        Mp = M1p
    elif numlayers == 2:  # Substrate + 1 layer
        M1s = np.array([[MpolaS[0, 0], MpolaS[0, 1]],
                       [MpolaS[1, 0], MpolaS[1, 1]]])
        M2s = np.array([[MpolaS[0, 2], MpolaS[0, 3]],
                       [MpolaS[1, 2], MpolaS[1, 3]]])
        M1p = np.array([[MpolaP[0, 0], MpolaP[0, 1]],
                       [MpolaP[1, 0], MpolaP[1, 1]]])
        M2p = np.array([[MpolaP[0, 2], MpolaP[0, 3]],
                       [MpolaP[1, 2], MpolaP[1, 3]]])
        # Matrix multiplication with conservation of the 3rd axis (z in an orthonormal coordonate system, named 1 here) constant
        Ms = np.einsum('nkl,kml->nml', M2s, M1s)
        Mp = np.einsum('nkl,kml->nml', M2p, M1p)
    elif numlayers == 3:  # Substrate + 2 layers
        M1s = np.array([[MpolaS[0, 0], MpolaS[0, 1]],
                       [MpolaS[1, 0], MpolaS[1, 1]]])
        M2s = np.array([[MpolaS[0, 2], MpolaS[0, 3]],
                       [MpolaS[1, 2], MpolaS[1, 3]]])
        M3s = np.array([[MpolaS[0, 4], MpolaS[0, 5]],
                       [MpolaS[1, 4], MpolaS[1, 5]]])
        M1p = np.array([[MpolaP[0, 0], MpolaP[0, 1]],
                       [MpolaP[1, 0], MpolaP[1, 1]]])
        M2p = np.array([[MpolaP[0, 2], MpolaP[0, 3]],
                       [MpolaP[1, 2], MpolaP[1, 3]]])
        M3p = np.array([[MpolaP[0, 4], MpolaP[0, 5]],
                       [MpolaP[1, 4], MpolaP[1, 5]]])
        Ms = np.einsum('nkl,kml->nml', M3s,
                       np.einsum('nkl,kml->nml', M2s, M1s))
        Mp = np.einsum('nkl,kml->nml', M3p,
                       np.einsum('nkl,kml->nml', M2p, M1p))

    # Matrix element
    m11s = Ms[0, 0]
    m12s = Ms[0, 1]
    m21s = Ms[1, 0]
    m22s = Ms[1, 1]

    m11p = Mp[0, 0]
    m12p = Mp[0, 1]
    m21p = Mp[1, 0]
    m22p = Mp[1, 1]

    # Fresnel total reflexion and transmission coefficient
    rs = (q0PolaS*m11s-qSPolaS*m22s+q0PolaS*qSPolaS*m12s-m21s) / \
        (q0PolaS*m11s+qSPolaS*m22s+q0PolaS*qSPolaS*m12s+m21s)
    rp = (q0PolaP*m11p-qSPolaP*m22p+q0PolaP*qSPolaP*m12p-m21p) / \
        (q0PolaP*m11p+qSPolaP*m22p+q0PolaP*qSPolaP*m12p+m21p)
    ts = 2*q0PolaS/(q0PolaS*m11s+qSPolaS*m22s+q0PolaS*qSPolaS*m12s+m21s)
    tp = 2*q0PolaP/(q0PolaP*m11p+qSPolaP*m22p+q0PolaP*qSPolaP*m12p+m21p)

    # Power transmittance
    Rs = (np.real(rs)) ** 2 + (np.imag(rs)) ** 2
    Rp = (np.real(rp)) ** 2 + (np.imag(rp)) ** 2
    # this stands only when the incident light is unpolarized (ambient)
    Refl = (Rs + Rp) / 2

    # Power transmittance
    # Transmittance of the multilayer stack only (substrate transmittance is not taken into account !)
    Ts = np.real(qSPolaS) / np.real(q0PolaS) * \
        ((np.real(ts) ** 2) + (np.imag(ts) ** 2))
    Tp = np.real(qSPolaP) / np.real(q0PolaP) * \
        ((np.real(tp) ** 2) + (np.imag(tp) ** 2))
    # This stands only when the incident light is unpolarized (ambient)
    TransMultilayer = (Ts + Tp) / 2

    # Transmittance of the substrate
    d = np.squeeze(d)
    TransSub = np.exp((-4*np.pi*kS*d[0])/Wl)

    # Transmittance of the substrate + multilayer stack
    Trans = TransMultilayer * TransSub

    # Power absorptance
    Abs = 1 - Refl - Trans
    return Refl, Trans, Abs


def RTA(Wl, d, n, k, Ang=0):
    """
See the function RTA3C for an example/tutoral and the version of the function writen for 3 layers (2 thin layers + the substrat). 
RTA calculates the reflectivity, transmissivity and absorptivity using Abélès matrix formalism.
\nThe Abélès matrix formalism provides the best ratio accurency/speed for stack below 100 thin layers. 
\nThe present version of RTA works for a infinite number of thin layer, but we not recommand to go over 100.

Parameters
----------
l : array
    Wavelength, in nanometer
d : array
    Tickness of stack, including the substrat
n : array 
    Real part of the refractive index of the layers.
k : array
    Extinction coefficient of the layers.
Ang : int, optional
    Incidence angle (in degrees) of the light one the optical stack. The default is 0 degree, so light perpendicular at the substrat.

Returns
-------
Refl : array
    The stack reflectivity, for each wavelength.
Trans : float
    The stack transmissivity, for each wavelength.
Abs : float
    The stack absorptivituy, for each wavelength.
    """

    # Convertir les angles en radians
    Phi0 = Ang * math.pi / 180

    # Définir les constantes initiales
    num_wl = len(Wl)
    N0 = 1 + 1j * 0
    sin_Phi0 = np.sin(Phi0)
    cos_Phi0 = np.cos(Phi0)
    q0PolaS = N0 * cos_Phi0
    q0PolaP = N0 / cos_Phi0

    # Ajouter les couches d'air et de substrat
    n = np.hstack((n, np.ones((num_wl, 1))))
    k = np.hstack((k, np.zeros((num_wl, 1))))
    d = np.hstack((d, np.zeros((1, 1))))

    # Calcul des valeurs pour le substrat
    nS = n[:, 0]
    kS = k[:, 0]
    Ns = nS + 1j * kS
    sin_PhiS = sin_Phi0 / Ns
    cos_PhiS = np.sqrt(1 - sin_PhiS**2)
    qSPolaS = Ns * cos_PhiS
    qSPolaP = Ns / cos_PhiS

    # Supprimer la première couche (couche d'air)
    nj = n[:, 1:]
    kj = k[:, 1:]
    dj = d[:, 1:].squeeze()

    numlayers = nj.shape[1]
    Nj = nj + 1j * kj
    sin_Phij = sin_Phi0 / Nj
    cos_Phij = np.sqrt(1 - sin_Phij**2)
    qjPolaS = Nj * cos_Phij
    qjPolaP = Nj / cos_Phij
    thetaj = (2 * np.pi / Wl[:, np.newaxis]) * dj * Nj * cos_Phij

    # Pré-calcul des valeurs trigonométriques
    cos_thetaj = np.cos(thetaj)
    sin_thetaj = np.sin(thetaj)

    # Initialisation des matrices de transfert
    MpolaS = np.zeros((2, 2 * numlayers, num_wl), dtype=complex)
    MpolaP = np.zeros((2, 2 * numlayers, num_wl), dtype=complex)

    # Remplissage des matrices de transfert
    MpolaS[0, 0::2] = cos_thetaj.T
    MpolaS[0, 1::2] = (-1j / qjPolaS * sin_thetaj).T
    MpolaS[1, 0::2] = (-1j * qjPolaS * sin_thetaj).T
    MpolaS[1, 1::2] = cos_thetaj.T

    MpolaP[0, 0::2] = cos_thetaj.T
    MpolaP[0, 1::2] = (-1j / qjPolaP * sin_thetaj).T
    MpolaP[1, 0::2] = (-1j * qjPolaP * sin_thetaj).T
    MpolaP[1, 1::2] = cos_thetaj.T

    # Initialisation des matrices M1s et M1p
    M1s = MpolaS[:, :2]
    M1p = MpolaP[:, :2]

    # Calcul des matrices Ms et Mp pour tous les couches
    Ms = M1s
    Mp = M1p
    for i in range(1, numlayers):
        Mi_s = MpolaS[:, i*2:i*2 + 2]
        Mi_p = MpolaP[:, i*2:i*2 + 2]
        Ms = np.einsum('ijm,jkm->ikm', Mi_s, Ms)
        Mp = np.einsum('ijm,jkm->ikm', Mi_p, Mp)

    # Extraction des éléments des matrices finales
    m11s, m12s, m21s, m22s = Ms[0, 0], Ms[0, 1], Ms[1, 0], Ms[1, 1]
    m11p, m12p, m21p, m22p = Mp[0, 0], Mp[0, 1], Mp[1, 0], Mp[1, 1]

    # Calcul des coefficients de réflexion et transmission
    denom_s = (q0PolaS * m11s + qSPolaS * m22s +
               q0PolaS * qSPolaS * m12s + m21s)
    denom_p = (q0PolaP * m11p + qSPolaP * m22p +
               q0PolaP * qSPolaP * m12p + m21p)
    rs = (q0PolaS * m11s - qSPolaS * m22s +
          q0PolaS * qSPolaS * m12s - m21s) / denom_s
    rp = (q0PolaP * m11p - qSPolaP * m22p +
          q0PolaP * qSPolaP * m12p - m21p) / denom_p
    ts = 2 * q0PolaS / denom_s
    tp = 2 * q0PolaP / denom_p

    # Calcul des intensités de réflexion et de transmission
    Rs = np.abs(rs)**2
    Rp = np.abs(rp)**2
    Refl = (Rs + Rp) / 2

    Ts = np.real(qSPolaS) / np.real(q0PolaS) * np.abs(ts)**2
    Tp = np.real(qSPolaP) / np.real(q0PolaP) * np.abs(tp)**2
    TransMultilayer = (Ts + Tp) / 2

    # Transmission à travers le substrat
    TransSub = np.exp((-4 * np.pi * kS * d[0, 0]) / Wl)

    # Transmission totale
    Trans = TransMultilayer * TransSub

    # Calcul de l'absorption
    Abs = 1 - Refl - Trans

    return Refl, Trans, Abs


def nb_compo(Mat_Stack):
    """
Gives back the amount of composite thin layers (made up of two materials). As a cermet or a porous material,
a composite thin layer includes the dash - in it string.
\nExemple :
\n'W-Al2O3' => composite layer of Al2O3 matrix with W inclusion (cermet type).
\n' air-SiO2' =>  composite layer of SiO2 matrix with air inclusion, (porous type).
    """
    nb = 0
    for i in Mat_Stack:
        if "-" in i:
            nb += 1
    return nb


def interpolate_with_extrapolation(x_new, x, y):
    """
This fonction provides linear extrapolation for refractive index data.
Extrapolation is necessary, because refractive index may cannot covert the wavelenght domain used.

Parameters
----------
x : Numpy array of float
    Here, x represents the wavelegenth domain present into the materials files.
y : Numpy array of float
    Here, y represents the refractive index (n or k) present into the materials files.
x_new : Numpy array of float
    The new wavelenght domain where the refractive index (y) must be extrapoled.

Returns
-------
TYPE
    Numpy array of float.\n
    Here, y represents the refractive index (n or k) extrapoled.\n
    y_new has the same dimension than x_new.

Exemple :

# Original data
\nWl_mat = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800])
\nn_mat = np.array([1.75, 1.640625, 1.5625, 1.515625, 1.5, 1.515625, 1.5625, 1.640625, 1.75])

# Wavelength domain used
\nWl = np.arange(200, 1001, 50)

# Interpolation with linear extrapolation 
\nn_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)

n_map : array([2.1875  , 2.078125, 1.96875 , 1.859375, 1.75    , 1.640625,
    1.5625  , 1.515625, 1.5     , 1.515625, 1.5625  , 1.640625,\n
    1.75    , 1.859375, 1.96875 , 2.078125, 2.1875  ])
    """
    interp_function = interp1d(x, y, kind='linear', fill_value='extrapolate')
    y_new = interp_function(x_new)
    return y_new


def Made_Stack(Mat_Stack, Wl):
    """
This key fonction strat with a list a material with describe the stack.  
It returns two table numpy array, on for the real part of the refractive index (n), and the other for the imaginary part (k). 

Parameters
----------
Mat_Stack : List of string, which contain each material of the stack. See as exemple the function write_stack_period.
\nExemple : Mat_Stack = ['BK7', 'TiO2', 'SiO2'] is a stack of TiO2 and SiO2 deposited on BK7 subtrate. 
The SiO2 layer is in contact with the air.

Wl : numpy array
    The list of the wavelenght.

Returns
-------
n_Stack : numpy array
    Each line is a different wavelenght.\n
    Each colonne is a different material.
k_Stack : nympy array
    Same than for n_Stack, but for k (the imaginary part).


Exemple 1 :
\nMat_Stack = ['BK7', 'TiO2', 'SiO2']
\nWl = [400, 450, 500]

n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
\nn_Stack : array([[1.5309    , 2.84076063, 1.48408   ],
\n[1.5253    , 2.78945014, 1.479844  ],
\n[1.5214    , 2.69067759, 1.476849  ]])
\nThe value must be understand like : 
                ____BK7 | TiO2 | SiO2
400 nm|1.5309|2.84076|1.48408
\n450 nm|1.5253|2.78945|1.47984
\n500 nm|1.5214|2.69068|1.47685
\nAs exemple, the value 1.5214 is the real part of the refractive index of BK7, at 500 nm.
    """
    # Creation of the Stack
    # I search if the name of a thin layer material is separated by a dash -
    # If yes, it's a composite material
    no_dash = True
    for s in Mat_Stack:
        if "-" in s:
            no_dash = False
            break

    if no_dash:  # If no_dash is true, I enter the loop
        n_Stack = np.zeros((len(Wl), len(Mat_Stack)))
        k_Stack = np.zeros((len(Wl), len(Mat_Stack)))
        for i in range(len(Mat_Stack)):
            Wl_mat, n_mat, k_mat = open_material(Mat_Stack[i])
            # Interpolation
            n_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)
            k_mat = interpolate_with_extrapolation(Wl, Wl_mat, k_mat)
            n_Stack[:, i] = n_mat[:,]
            k_Stack[:, i] = k_mat[:,]
            for i in range(len(k_Stack)):
                for j in range(len(k_Stack[0])):
                    if (k_Stack[i][j] < 0):
                        k_Stack[i][j] = 0
        return n_Stack, k_Stack

    else:  # Else, there must be a dash, so two materials
        n_Stack = np.zeros((len(Wl), len(Mat_Stack), 2))
        k_Stack = np.zeros((len(Wl), len(Mat_Stack), 2))
        for i in range(len(Mat_Stack)):
            # I open the first material
            list_mat = []
            list_mat = Mat_Stack[i].split("-")
            if len(list_mat) == 1:
                # the list includes one material. I charge it as usual
                # Row: wavelenght, column : material indexes
                Wl_mat, n_mat, k_mat = open_material(Mat_Stack[i])
                # Interpolation
                n_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)
                k_mat = interpolate_with_extrapolation(Wl, Wl_mat, k_mat)
                n_Stack[:, i, 0] = n_mat[:,]
                k_Stack[:, i, 0] = k_mat[:,]
            if len(list_mat) == 2:
                # the list includes two materials. I place the second on the z=2 axis
                Wl_mat, n_mat, k_mat = open_material(list_mat[0])
                # Interpolation
                n_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)
                k_mat = interpolate_with_extrapolation(Wl, Wl_mat, k_mat)
                n_Stack[:, i, 0] = n_mat[:,]
                k_Stack[:, i, 0] = k_mat[:,]
                # Opening of the second material
                Wl_mat, n_mat, k_mat = open_material(list_mat[1])
                # Interpolation
                n_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)
                k_mat = interpolate_with_extrapolation(Wl, Wl_mat, k_mat)
                n_Stack[:, i, 1] = n_mat[:,]
                k_Stack[:, i, 1] = k_mat[:,]
            for i in range(k_Stack.shape[0]):
                for j in range(k_Stack.shape[1]):
                    for k in range(k_Stack.shape[2]):
                        if (k_Stack[i][j][k] < 0):
                            k_Stack[i][j][k] = 0
    return n_Stack, k_Stack


def Made_Stack_vf(n_Stack, k_Stack, vf=[0]):
    """
n_Stack_vf, or k_stack_vf means an n and k calculated by a Bruggeman function (EMA mixing law).
\nThese are the values to be injected into RTA:
\nIf vf = 0 for all materials, then n_Stack_vf = n_Stack (idem for k).
\nOtherwise, calculate.

Parameters
----------
n_Stack : array, in  3 dimensional
    The real part of the refractive index
k_Stack : array, in 3 dimensional
    The complexe part of the refractive index
vf : list of int, optional
    The volumic fraction of all thin layer, as a list. The default is [0].

Returns
-------
n_Stack : float array
    Real part of the refractive index of the stack.
k_Stack : float array
    Extinction coefficient of the stack.
n_Stack_vf : float array
    Real part of the refractive index of the stack with volume fractions.
k_Stack_vf : float array
    Extinction coefficient of the stack with volume fractions.
    """
    if all(elem == 0 for elem in vf):  # (vf ==0).all():
        """ All ths vf(s) = 0. It's not necessary to launch Bruggman. 
        """
        return n_Stack, k_Stack
    else:
        """ vf.all == [0] is not True. At least vf exists, it means that only one layer of the stack 
        is made of two materials. n_Stack and k_Stack are 3D table.For exemple, for "W-Al2O3",
        W datas are in range [:,:,0] and Al2O3 datas are in range [:,:,1]
        """
        n_Stack_vf = np.empty((n_Stack.shape[0], np.shape(n_Stack)[1]))
        k_Stack_vf = np.empty((k_Stack.shape[0], np.shape(k_Stack)[1]))
        # old version
        # n_Stack_vf = []
        # k_Stack_vf = []
        # n_Stack_vf = np.array(n_Stack_vf)
        # k_Stack_vf = np.array(k_Stack_vf)
        for i in range(np.shape(k_Stack)[1]):
            #  I check every layer and recup the n and the k of the matrix (M) and of the inclusions (I)
            # If the layer is made of one material only, the datas are in nI and kI
            # => nM and km are full of zeros in this case
            nM = np.copy(n_Stack[:, i, 1])
            kM = np.copy(k_Stack[:, i, 1])
            nI = np.copy(n_Stack[:, i, 0])
            kI = np.copy(k_Stack[:, i, 0])
            # kM= k_Stack[:,i,1].copy()
            # nI= n_Stack[:,i,0].copy()
            # kI= k_Stack[:,i,0].copy()
            n_Stack_vf[:, i], k_Stack_vf[:, i] = Bruggeman(
                nM, kM, nI, kI, vf[i])
            # n_Stack_vf[:,i], k_Stack_vf[:,i] = Bruggeman_np(nM, kM, nI, kI, vf[i])
        return n_Stack_vf, k_Stack_vf


def Bruggeman(nM, kM, nI, kI, VF):
    """
Bruggemann function. 
Allow us to calculte the complex refractive index of a mixture of two materials, using an EMA (Effective Medium Approximation)

Parameters
----------
nM : array
Real part of refractive index of host Matrix (M is for Matrix)
\nkM : array
Complex part of refractive index of host Matrix (M is for Matrix)
\nnI : array
Real part of refractive index of inclusion (I is for Inclusion)
\nkI : TYPE
Complex part of refractive index of inclusion (I is for Inclusion)
\nVF : int
Volumic Fraction of inclusions in host matrix. Number between 0 et 1 (0 and 100%)     

Returns
-------
nEffective : array
Real part of the refractive index of the effective medium : the mixture of the host Matrix and the embedded particules
\nkEffective : array
Complex part of the refractive index of the effective medium : the mixture of the host Matrix and the embedded particules

Noted than If vf = 0 :
\nnEffective = nM and kEffective = kM
\nNoted than If vf = 1.0 : 
\nnEffective = nI and kEffective = kI
    """
    if VF == 0:
        return nI, kI

    eM = (nM + 1j * kM) ** 2
    eI = (nI + 1j * kI) ** 2
    y = 2

    # Calculating coefficients for the quadratic equation
    a = -y
    b = (VF * y + VF - 1) * eI - (VF * y + VF - y) * eM
    c = eM * eI

    # Solving the quadratic equation using NumPy's vectorized roots
    discriminant = np.sqrt(b**2 - 4 * a * c)
    e1 = (-b + discriminant) / (2 * a)
    e2 = (-b - discriminant) / (2 * a)

    # Choosing the appropriate roots
    positive_imaginary = np.imag(e1) > 0
    positive_real = np.real(e1) > 0

    Neffective = np.where(positive_imaginary, np.sqrt(e1),
                          np.where(np.imag(e2) > 0, np.sqrt(e2),
                          np.where(positive_real, np.sqrt(e1), np.sqrt(e2))))

    # Extracting the real and imaginary parts
    nEffective = np.real(Neffective)
    kEffective = np.imag(Neffective)

    return nEffective, kEffective


def BB(T, Wl):
    """
    Parameters
    ----------
    T : Int
        Black Body Temperature , in Kelvin
    Wl : 1D vector, array of int
        Wavelenght, in nm

    Returns
    -------
    BB_spec : Array of float
        Black Body Luminance, in W.m^-3. 
        Note the pi factor in the equation to transform W.m^-3.sr^-1 into W.m^-3
    """
    h = 6.62607015e-34  # Planck constant
    c = 299792458  # Speed of light
    k = 1.380649e-23  # Boltzmann constant
    BB_spec = 2 * np.pi * h * c**2 / \
        (Wl * 1e-9)**5 / (np.exp(h * c / (Wl * 1e-9 * k * T)) - 1) * 1e-9
    return BB_spec


def SolarProperties(Wl, R, SolSpec):
    """
Parameters
----------
R : array
    Stack Optical Properties, for different Wavelengths, properly interpoled. 
    Not than R is not necessary the reflectivity, can be transmissivity or absorptivity.
Wl : array
    Wavelength, in nm.
SolSpec : Vector.
    SolarSpectrum used, properly intepoled in W.m^2.nm^-1. 
    R and SolSpec must have the same length.

Returns
-------
R_s : float
    Solar Properties, accorting to the Strack Optical Properties.
    \n=> not necessary Solar Reflectance. 
    """
    if len(Wl) != len(R) or len(Wl) != len(SolSpec) or len(R) != len(SolSpec):
        raise ValueError(
            "Vectors l, R, and SolSpec must have the same length.")
    try:
        R_Spec = []
        R_Spec = R * SolSpec
        SolSpec_int = trapezoid(SolSpec, Wl)
        R_int = trapezoid(R_Spec, Wl)
        R_s = R_int / SolSpec_int
    except:
        raise ValueError("Vectors l, R, and SolSpec must be a numpy array.")

    return R_s


def E_BB(l, A, BB):
    """
Parameters
----------
A : Arroy of float
    Strack Absortivity Properties, for different Wavelength, properly intepoled
L : Vector
    Wavelength, in nm
BB : Arroy of float
    Black Body Luminance, in W.m^-3, properly intepoled

A and BB must have the same length !

Returns
-------
E_BB : float
Thermal emissivty, according to the black body irradiance BB
    """
    if len(l) != len(A) or len(l) != len(BB) or len(A) != len(BB):
        raise ValueError(
            "Vectors l, R, and SolSpec must have the same length.")
    try:
        E_BB_Spec = []
        E_BB_Spec = A * BB
        BB_Spec_int = trapezoid(BB, l)
        E_int = trapezoid(E_BB_Spec, l)
        E_BB = E_int / BB_Spec_int
    except:
        raise ValueError("Vectors l, R, and SolSpec must be a numpy array.")

    return E_BB


def helio_th(A_s, E_BB, T_stack, T_air, C, I,  r_Opt=0.7, FFabs=1):
    """
Calculates the heliothermal efficiency.

Parameters
----------
A_s : Float
    Solar Absoptivity, calculated previsouly.
E_BB : Float
    Thermal emissivity, calculated previsouly according to the temperature T_Stack.
T_stack : Float
    Temperature, in Kelvin, of the stack. 
T_air : Float
    Temperature, in Kevlin, of the environement.
C : Float
    Solar concentration.
I : Float
    Solar irradiance on the stack, in W/m2. Normaly calculatated with integration of the solar spectrum.
r_Opt : TYPE, optional
    Optical performance of the optical concentrator, used with the solar selective stack. The default value is 0.7.
FFabs : Float, optional
    Ratio between the absorting surface and the emissivity surface. The default value is 1.

Returns
-------
rHelio : Float
    Heliothermal efficiency, the conversion between photo to thermal, according the operating conditions (T_stack, T_air, C, I etc).
    """
    sigma = 5.67037321e-8
    rHelio = A_s - E_BB * FFabs * sigma * \
        (T_stack**4 - T_air**4)/(r_Opt * C * I)
    return rHelio


def open_material(name):
    """
Opens a text file which contains refractive index from Materials folder.
\nExemple Wl, n, k = open_material("Ag") opens a file named Ag.txt, into the Materials.

name : string
    The name of a text file which contains Wavelength, and refractive index values.\n
    The texte file must be formated according the following :\n
    1 row : wavelenght in nm\n
    2 row : real part of refractive index\n
    3 row : imaginary part of the refractive index\n
    rows seperated by a tabulation

Returns
-------
Wl : numpy array 
    Wavelength, in nanometer (nm) .
n : numpy array
    Real part of the Refractive Index.
k : numpy array
    Complex part of the Refractive Index.
    """

    assert isinstance(
        name, str), f"Argument 'name' must be a string but had type {type(name)}"
    # Initialise an empty table
    tableau3D = []
    name = "Materials/" + name + ".txt"
    try:
        lines = _flexible_open_resource(name)

        # Make an iteration on the lines
        nb_line = len(lines)
        for i in range(nb_line):
            values = lines[i].split("\t")
            values[2] = values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {name} not found")
    # Transform the list into a numpy table
    tableau3D = np.array(tableau3D)

    Wl = []
    Wl = tableau3D[:, 0]
    n = []
    n = tableau3D[:, 1]
    k = []
    k = tableau3D[:, 2]

    return Wl, n, k


_valid_materials = list(map(lambda p: p.name,
                            impresources.files('solpoc').joinpath('Materials').glob("*.txt")))


def _flexible_open_resource(filepath, resource_dir="Materials"):
    name = os.path.basename(filepath)
    user_dir = os.path.dirname(os.path.abspath(filepath))
    pkg_path = impresources.files('solpoc')
    # First try finding the resource file within the package install folder
    try:
        pkg_filepath = pkg_path.joinpath(resource_dir, name)
        with pkg_filepath.open("r") as fp:
            return fp.readlines()
    except FileNotFoundError:
        try:
            with open(filepath, "r") as fp:
                return fp.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{name} as not found neither in your local directory '{user_dir}' nor in the package install directory. "
                f"Please supply a valid name from {_valid_materials},"
                f" OR create the file {filepath} within {user_dir}.")


def open_SolSpec(name='Materials/SolSpec.txt', type_spec="DC"):
    """
Name : string
    Opens txt file with the solar spectra data. Normal name is SolSpec.txt in material files.\n
    SolSpec is a table, with one wavelength per line.  
type_spec : string 
    Optional is for the type for solar spectra.
DC : Direct and Circumsolar
    The sun irradiance which come from directly from the sun and his corona.
GT Global Tilt
    The sun irradiance which comme from the sun and the near environement (reflexion / albedo).
Extr
    Extra-terra solar spectrum.
    """

    # Initialise an empty table
    tableau3D = []
    try:
        lines = _flexible_open_resource(name)

        # Make an iteration on the lines
        nb_line = len(lines)
        for i in range(nb_line):
            values = lines[i].split("\t")
            values[2] = values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {name} not found")
    # Transform the list into a numpy table
    tableau3D = np.array(tableau3D)
    # Extract wished datas
    Wl = []
    Wl = tableau3D[:, 0]
    spec = []
    if type_spec == "DC":
        spec = tableau3D[:, 1]
    if type_spec == "Extr":
        spec = tableau3D[:, 2]
    if type_spec == "GT":
        spec = tableau3D[:, 3]

    # Upadted on 05/05/2023. I'm adding the solar spectrum in the name, to directly have the spectrum type
    name = name + " type_de_spectre:" + type_spec

    return Wl, spec, name


def open_Spec_Signal(name, nb_col):
    """
Opens a spectral respond into a file.

Parameters
----------
name
    The name of a file.
nb_col : Int
    The number of read column in the file.

Returns
-------
Wl : array
    Wavelenght, must be in nm into the file.
spec : array
    The value present in the file, according to the Wavelength.
name_f : string
    The name of the file opened, with the number of the column used.\n  
    As :  name + " ,col n° " + str(nb_col)
    """

    # Initialise an empty table
    tableau3D = []
    try:
        lines = _flexible_open_resource(name)

        # Make an iteration on the lines
        nb_line = len(lines)
        for i in range(nb_line):
            values = lines[i].split("\t")
            values[2] = values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {name} not found")
    # Transform the list into a numpy table
    tableau3D = np.array(tableau3D)
    # Extract wished datas
    Wl = []
    Wl = tableau3D[:, 0]
    spec = []
    spec = tableau3D[:, nb_col]
    name_f = name + " ,col n° " + str(nb_col)
    return Wl, spec, name_f


def eliminate_duplicates(lst):
    """
Removes duplicates from a list.
\n-Exemple :
\nA = [1, 2, 3, 3, 4, 4, 5].
\nB, C = eleminate_duplicates(A).
\nB = [1, 2, 3, 4, 5].
\nC = [3, 5].

Parameters
----------
lst : List of values.

Returns
-------
unique_elements : List
    List of values without duplica.
indices_removed : 
    List of indice removed values in the list.
    """
    unique_elements = []
    indices_removed = []

    for i, element in enumerate(lst):
        if element not in unique_elements:
            unique_elements.append(element)
        else:
            indices_removed.append(i)

    return unique_elements, indices_removed


def write_stack_period(Subtrat, Mat_Periode, nb_periode):
    """
Builds a stack by repeating a material period multiple times on top of a substrate.
    Exemple 1 :\n 
    Mat_Stack = write_stack_period(["BK7"], ["TiO2_I", "SiO2_I"], 3)\n
    Mat_Stack :  ['BK7', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I']\n
    Exemple 2:\n
    Mat_Stack = write_stack_period(["BK7", "TiO2", "Al2O3",], ["TiO2_I", "SiO2_I"], 2)\n
    Mat_Stack  : ['BK7', 'TiO2', 'Al2O3', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I']\n

Parameters
----------
Subtrat : List of string
    Each elements of this list is a string as valid material (a material with an associatd text file in Material/).
Mat_Periode : List of string
    A list of strings. Each elements of this list is a string as valid material.
nb_periode : Int 
    The number of time were the Mat_Periode must be repeted.

Returns
-------
Subtrat : List of string
    List which contains the different substrates.
    """
    for i in range(nb_periode):
        Subtrat += Mat_Periode
    return Subtrat


def equidistant_values(lst):
    """
Returns a small list of y equidistant values from a large list lst.

Parameters
----------
lst : List
    Returns a small list of y equidistant values from a large list lst.

Returns
-------
result : list
    Returns a small list of y equidistant values from a large list lst.
    """
    # Enables to return a small list of y equidistant values from a long list
    x = 5
    n = len(lst)
    # I substract 1 from the interval to avoid the out of range error
    interval = (n // (x - 1))-1
    result = [lst[i*interval] for i in range(x)]
    return result


def valeurs_equidistantes(liste, n=5):
    """
From a large list, returns a small list with equidistant values.

Parameters
----------
liste : list or array
    A large list.
n : Int number, optional
    Total of elements in the list. The default value is 5.

Returns
-------
petite_liste : list or array
    Returns a small list of y equidistant values from a large list.
    """
    # Determine the distance between every value
    distance = len(liste) / (n - 1)
    # Initialise the short list
    petite_liste = [liste[0]]
    # Add equidistant values to the short list
    for i in range(1, n - 1):
        index = int(i * distance)
        petite_liste.append(liste[index])
    petite_liste.append(liste[-1])
    # Return the short list
    return petite_liste


def Wl_selectif():
    """
Give a vector of Wavelength (in nm), optimized for selective coating optimisation/calculation of performances.
    280 to 2500 nm (solar domain) with a 5 nm step for the calculation for performances.\n
    2500 nm to 30µm (IR domain) with a 50 nm step for the calculation of thermal emissivity (named E_BB in this code).

Returns
-------
Wl : array
    Wavelenght, in nm.
    """
    Wl_1 = np.arange(280, 2500, 5)
    Wl_2 = np.arange(2500, 30050, 50)
    Wl = np.concatenate((Wl_1, Wl_2))
    return Wl


def evaluate_example(individual):
    """
Example of an evaluate function. The individual is a list. 
\nThe sum of squares of each term in the list is sought. 
\nExample of evaluate function (= cost function) for an optimisation method.

Parameters
----------
individual : List
    A list representing an individual solution.

Returns
-------
score : Float
    The score calculated based on the squares of the values in the individual.
    """

    # Cenvert a list in an array np.array(population[1])
    score = 0
    for sub_list in individual:
        score += sub_list*sub_list
    return score


def Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack, parameters):
    """
To understand Individual_to_Stack work, we sudject to run the main script with the following : 
\nMat_Stack : ['BK7', 'W-Al2O3', 'SiO2']
\nWl = np.arange(400 , 600, 50)

Now note than the first thin layer is a composite layer, made of W and Al2O3 (BK7 is the stack):
We need the refractive index of W AND Al2O3 for the layer 1, and we need to optimise the tickness AND volumic fraction in the W-Al2O3 layer.
See EMA or Brugeman function for definition of volumuc fraction.
Each individual is now an array of lenght 6, as exemple : 
\nindividual : [1.00000000e+06, 40, 125, 0, 0.3, 0]
\nThe [1.00000000e+06, 40, 125] part of the list contain the thickness, in nm
\nThe [0, 0.3, 0] part of the list contain the volumic fraction, between 0 and 1
\nk_Stack and n_Stack are array of float, of size (4, 3, 2), noted here (x, y, z) dimension
\nx dimension is for wavelenght
\ny dimension is for each layer
\nz dimension is for stored the value of W AND Al2O3 with the same x and y dimension

As exemple : 
n_Stack :\n
    array([[[1.5309    , 0.        ],
    [3.39      , 1.66518263],
    [1.48408   , 0.        ]],

    [[1.5253    , 0.        ],
    [3.30888889, 1.65954554],
    [1.479844  , 0.        ]],

    [[1.5214    , 0.        ],
    [3.39607843, 1.65544143],
    [1.476849  , 0.        ]],

    [1.5185    , 0.        ],
    [3.5       , 1.65232045],
    [1.474652  , 0.        ]]])

The purpose of Individual_to_Stack is to transform in such case the individual, n_Stack and k_Stack

Parameters
----------
individual : array
    \nindividual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.
\nn_Stack : array 
    The real part of refractive index. 
    Can be of size (x, y, 2), with x the len of wavelenght and y the number of layer.
k_Stack : array
    The complex part of refractive index. 
    Can be of size (x, y, 2), with x the len of wavelenght and y the number of layer.
Mat_Stack : List of string
    List of materials.
parameters : Dict
    Dictionary which contain all parameters. 

Raises
------
ValueError
    It is not possible to work with theoretical and composite layers at the same time.

Returns
-------
d_Stack : 
    List of only thickness in nm
n_Stack : array
    The real part of refractive index. 
    Must be size of (x, y) with x the len of wavelenght and y the number of layer    
k_Stack : array
    The comlex part of refractive index. 
    Must be size of (x, y) with x the len of wavelenght and y the number of layer  
    """

    # Add in work with vf(s)
    if 'nb_layer' in parameters:
        if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
            raise ValueError(
                "It is not possible to work with theoretical and composite layers at the same time.")

    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        vf = []
        vf = individual[len(Mat_Stack):len(individual)]
        individual_list = individual.tolist()  # Conversion is a list
        del individual_list[len(Mat_Stack):len(individual)]
        individual = np.array(individual_list)  # Conversion in a table
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
        vf = np.array(vf)
        n_Stack, k_Stack = Made_Stack_vf(n_Stack, k_Stack, vf)

    if 'nb_layer' in parameters:
        nb_layer = parameters.get('nb_layer')
        for i in range(nb_layer):
            # I check the value of the layer's index
            n = individual[nb_layer + len(Mat_Stack)]
            # I add the layer of n index and k = 0 to the Stack
            n_Stack = np.insert(n_Stack, len(Mat_Stack) + i, n, axis=1)
            k_Stack = np.insert(k_Stack, len(Mat_Stack) + i, 0, axis=1)
            index_to_remove = np.where(individual == n)[0][0]
            individual = np.delete(individual, index_to_remove)
        # As I did in previous versions, I transform d_Strack into an array
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
    else:
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))

    return d_Stack, n_Stack, k_Stack


def evaluate_R(individual, parameters):
    """
Cost function for the average reflectivity at one or several wavelength.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo).\n
    individual describe a stack of thin layers, substrat included. Each number are thickness in nm.\n
    Exemple : [1000000, 100, 50, 120, 70] is a stack of 4 thin layers, respectivly of 100 nm, 50 nm, 120 nm and 70 nm.\n
    The 70 nm thick layer is in contact with air.\n
    The 100 nm thick layer is in contact with the substrat, here 1 mm thick.\n
    1 individual = 1 stack = 1 possible solution.\n
    List of thickness in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contain all parameters. 

Returns
-------
R_mean : Int (float)
    The average reflectance.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')
    # Creation of
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    R_mean = np.mean(R)
    return R_mean


def evaluate_T(individual, parameters):
    """
Cost function for the average transmissivity at one or several wavelength.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contains all parameters. 

Returns
-------
T_mean: Int (float)
    The average transmittance.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')

    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # change
    T_mean = np.mean(T)
    return T_mean


def evaluate_R_s(individual, parameters):
    """
Calculates the solar reflectance of an individual.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contains all parameters. 

Returns
-------
R_s : Int (float)
    The solar reflectance.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation
    R_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    R_s = SolarProperties(Wl, R, Sol_Spec)
    return R_s


def evaluate_T_s(individual, parameters):
    """
Calculates the solar transmittance of an individual.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contains all parameters.

Returns
-------
T_s : Int (float)
    The solar transmittance.
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')  # , 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    T_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_s = SolarProperties(Wl, T, Sol_Spec)
    return T_s


def evaluate_A_s(individual, parameters):
    """
Calculates the solar absoptance of an individual.
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.
parameters : Dict
    Dictionary which contain all parameters.

Returns
-------
A_s : Int (float)
    The solar absoptance.
    """

    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')  # , 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    A_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    A_s = SolarProperties(Wl, A, Sol_Spec)
    return A_s


def evaluate_R_Brg(individual, parameters):
    """
Cost function for a Bragg mirror.
\nMaximises the average reflectivity between 500 to 650 nm (default value).
\n1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo).\n
    individual describes a stack of thin layers, substrat included. Each number are thickness in nm.\n
    Exemple : [1000000, 100, 50, 120, 70] is a stack of 4 thin layers, respectivly of 100 nm, 50 nm, 120 nm and 70 nm.\n
    The 70 nm thick layer is in contact with air.\n
    The 100 nm thick layer is in contact with the substrat, here 1 mm thcik.\n
    1 individual = 1 stack = 1 possible solution.\n
    List of thicknesses in nm, witch can be added with volumic fraction or refractive index.\n

parameters : Dict
    Dictionary witch contain all parameters. 

Returns
-------
R_mean : Int (float)
    The average reflectance.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')
    # Wl_targ = 550
    # Creation of
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Wavelenght array where the reflectivity of the Bragg mirror must be maximised
    Wl_2 = np.arange(500, 655, 5)
    R_Bragg = 0
    for i in range(len(Wl_2)):
        index = np.where(Wl == Wl_2[i])
        value_index = index[0][0]
        R_Bragg = R_Bragg + R[value_index]

    return R_Bragg * 1/(len(Wl_2))


def evaluate_T_pv(individual, parameters):
    """
Calculates the solar transmissivity WITH a PV cells signal
with the following line code in the main script.
\nif evaluate.__name__ == "evaluate_T_PV":
    parameters["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec

1 individual = 1 output of one optimization function = 1 possible solution
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contains all parameters. 

Returns
-------
T_PV: Int (float)
    Solar transmissivity WITH a PV cells signal.
    """

    Wl = parameters.get('Wl')  # ,
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec_with_PV')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    T_PV = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Sol_Spec is Sol_Spec_with_PV
    T_PV = SolarProperties(Wl, T, Sol_Spec)
    return T_PV


def evaluate_A_pv(individual, parameters):
    """
Calculates the solar absoptivity WITH a PV cells signal with the following ligne code in the main script:\n
if evaluate.__name__ == "evaluate_T_PV":\n  
    parameters["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec  
1 individual = 1 output of one optimization function = 1 possible solution.
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, which can be added with volumic fraction or refractive index.

Parameters : Dict
    Dictionary which contain all parameters.

Returns
-------
T_PV: Int (float)
    Solar transmissivity WITH a PV cells signal.
    """

    Wl = parameters.get('Wl')  # ,
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    # Sol_Spec_with_pv = Sol_Spec * Signal_PV
    Sol_Spec = parameters.get('Sol_Spec_with_PV')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    T_PV = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Sol_Spec is Sol_Spec_with_PV
    A_PV = SolarProperties(Wl, A, Sol_Spec)
    return A_PV


def evaluate_T_vis(individual, parameters):
    """
Calculates the optical transmittance with a human eye input.
\nThe solar spectrum (Sol_Spec) has been replaced by a human eye sensivity to wavelenght during the process.
\nSee the following code lines in the main script.

Wl_H_eye , Signal_H_eye , name_H_eye = open_Spec_Signal('Materials/Human_eye.txt', 1)
\nSignal_H_eye = np.interp(Wl, Wl_H_eye, Signal_H_eye) # Interpolate the signal

\nparameters["Sol_Spec_with_Human_eye"] = Signal_H_eye 

\n1 individual = 1 output of one optimization function = 1 possible solution
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contains all parameters.

Returns
-------
T_PV: Int (float)
    Solar transmissivity WITH a PV cells signal.
    """

    Wl = parameters.get('Wl')  # ,
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec_Heye = parameters.get('Sol_Spec_with_Human_eye')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    T_vis = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_vis = SolarProperties(Wl, T, Sol_Spec_Heye)
    return T_vis


def evaluate_low_e(individual, parameters):
    """
Calculates the low_e performances.
\n1 individual = 1 output of one optimization function = 1 possible solution
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contain all parameters.

Returns
-------
P_low_e: Int (float)
    Low_e performances.
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')  # , 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # The profile is reflective from 0 to Lambda_cut_1
    # The profil is transparent from Lambda_cut_1 to + inf
    Lambda_cut_1 = parameters.get('Lambda_cut_2')
    d_Stack = np.array(individual)
    # Calculation of the domains
    Wl_1 = np.arange(min(Wl), Lambda_cut_1, (Wl[1]-Wl[0]))
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calculation
    # Transmitted solar flux on the Wl-1 part
    P_low_e = np.concatenate([T[0:len(Wl_1)], R[len(Wl_1):]])
    P_low_e = SolarProperties(Wl, P_low_e, Sol_Spec)

    return P_low_e


def evaluate_rh(individual, parameters):
    """
Calculates the heliothermal efficiency.
\n1 individual = 1 output of one optimization function = 1 possible solution
\nindividual : array
    individual is an output of optimisation method (algo). 
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contain all parameters.

Returns
-------
rH: Int (float)
    Heliothermal efficiency.
    """

    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    C = parameters.get('C')
    T_air = parameters.get('T_air')
    T_abs = parameters.get('T_abs')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # Integration of solar spectrum, raw en W/m2
    I = trapezoid(Sol_Spec, Wl)
    # Creation of the stack
    d_Stack = np.array(individual)
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calculation of the solar absorption
    A_s = 0
    A_s = SolarProperties(Wl, A, Sol_Spec)
    # Calculation of the balck body
    BB_shape = BB(T_abs, Wl)
    # calculation of the emittance of the surface
    E_BB_T_abs = E_BB(Wl, A, BB_shape)

    # Calculation of the solar thermal yield. Argument of the function helio_th(A_s, E_BB, T_stack, T_air, C, I,  r_Opt = 0.7, FFabs=1):
    rH = helio_th(A_s, E_BB_T_abs, T_abs, T_air, C, I,  r_Opt=0.7, FFabs=1)

    return rH


def evaluate_RTR(individual, parameters):
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
    Ang = parameters.get('Ang')
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
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    Wl_1 = np.arange(min(Wl), Lambda_cut_1+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    Wl_2 = np.arange(Lambda_cut_1, Lambda_cut_2+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    # Calculation of the RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    P_low_e = np.concatenate([R[0:len(Wl_1)], T[len(Wl_1):(
        len(Wl_2)+len(Wl_1)-1)], R[(len(Wl_2)+len(Wl_1)-1):]])
    P_RTR = SolarProperties(Wl, P_low_e, Sol_Spec)

    return P_RTR


def evaluate_netW_PV_CSP(individual, parameters):
    """
Calculates the performance according an RTR shape.
\n1 individual = 1 output of one optimization function = 1 possible solution
\nindividual : array
    individual is an output of optimisation method (algo).
    List of thicknesses in nm, which can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary which contain all parameters.

Returns
-------
P_RTR: Int (float)
    Performance according an RTR shape.
    """

    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')  # , 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')

    # Treatment of the optimization of the n(s)
    Mat_Stack = parameters.get('Mat_Stack')

    """Get the "cost of PV". We need to give more importance to the PV part. Without that, the optimization process not provide
    a RTR like coating, but a near perfect mirror
    Without cost of PV the best coating a dielectric mirror, witch reflected all the sun light without transmited solar flux to the PV cells
    """
    # PV part
    poids_PV = parameters.get('poids_PV')
    Signal_PV = parameters.get('Signal_PV')
    # Thermal part
    Signal_Th = parameters.get('Signal_Th')

    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # I calculate Rs
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Intégration du spectre solaire, brut en W/m2
    Sol_Spec_PV = Sol_Spec * Signal_PV
    Sol_Spec_PV_int = trapezoid(Sol_Spec_PV, Wl) * poids_PV
    Sol_Spec_Th = Sol_Spec * Signal_Th
    Sol_Spec_Th_int = trapezoid(Sol_Spec_Th, Wl)

    # Integration of the absorbed power by the PV
    Sol_Spec_T_PV = Sol_Spec * T * Signal_PV
    Sol_Spec_T_PV_int = trapezoid(Sol_Spec_T_PV, Wl) * poids_PV

    # Integration of the absorbed power by the PV
    Sol_Spec_R_Th = Sol_Spec * R * Signal_Th
    Sol_Spec_R_Th_int = trapezoid(Sol_Spec_R_Th, Wl)

    net_PV_CSP = (Sol_Spec_T_PV_int + Sol_Spec_R_Th_int) / \
        (Sol_Spec_PV_int + Sol_Spec_Th_int)
    return net_PV_CSP


def evaluate_RTA_s(individual, parameters):
    """
Calculates the solar reflectance, the solar transmittance and the solar absoptance 
for a full spectrum.

Parameters
----------
individual : array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractive index.

parameters : Dict
    Dictionary witch contain all parameters.

Returns
-------
R_s : Float
    Solar reflectance.
T_s : Float
    Solar transmittance.
A_s : Float
    Solar absorptance.
    """
    # Calculates the solar reflectance, solar transmittance and the absorptance
    # Every individual is a list of thickness.
    # I set the variables Wl, Ang, n_Stack, k_Stack and SolSpec in global

    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')  # , 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')

    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    if 'coherency_limit' in parameters:
        coherency_limit = parameters.get('coherency_limit')
    else:
        coherency_limit = 2e4

    # Check if one or several layer are coherent
    # uncoherent if thickness up to 2500 nm or if presence of air or vaccum in the stack
    cl, coherency = Stack_coherency(
        d_Stack.flatten().tolist(), Mat_Stack, coherency_limit)  # d_Stack is array

    # if coherency is True, I use my own function (much more faster)
    if coherency:
        R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    else:  # if coherency is True, I use my own function (faster)
        # Fonction calculate_rat from solcore
        stack = Made_SolCORE_Stack(d_Stack, Wl, n_Stack, k_Stack)
        rat_data = calculate_rat(
            stack, angle=Ang, wavelength=Wl, coherent=False, coherency_list=cl)
        size = np.shape(rat_data["A_per_layer"])
        R = rat_data["A_per_layer"][0]
        T = rat_data["A_per_layer"][size[0]-2]
        A = 1 - R - T

    R_s, T_s, A_s = 0, 0, 0

    if all(value == 0 for value in T):
        T[0] = 10**-301
    if all(value == 0 for value in R):
        R[0] = 10**-301
    if all(value == 0 for value in A):
        A[0] = 10**-301

    R_s = SolarProperties(Wl, R, Sol_Spec)
    T_s = SolarProperties(Wl, T, Sol_Spec)
    A_s = SolarProperties(Wl, A, Sol_Spec)
    return R_s, T_s, A_s


def evaluate_EBB(individual, parameters):
    """
    Calculate the thermal emissivity
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    EBB_T_abs: Int (float)
        Thermal emissiviy according the absorber (abs) temperature (T)
    """

    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    T_air = parameters.get('T_air')
    T_abs = parameters.get('T_abs')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    # Creation of the stack
    d_Stack = np.array(individual)
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    BB_shape = BB(T_abs, Wl)
    # calculation of the emittance of the surface
    E_BB_T_abs = E_BB(Wl, A, BB_shape)

    return E_BB_T_abs


def evaluate_fit_R(individual, parameters):
    """
    Calculate the stack thin layers thickness for fit a experimental reflectivity signal
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    cost: Int (float)
        Normalize the MSE to be between 0 and 1 
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    signal = parameters.get('Signal_fit')

    # Creation of the stack
    d_Stack = np.array(individual)
    Mat_Stack = parameters.get('Mat_Stack')
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # Normalize the MSE to be between 0 and 1
    #  If R and signal are identical, the cost will be 0, and the more they differ, the closer the cost will be to 1.
    cost = normalized_mse(R, signal)

    return cost


def evaluate_fit_T(individual, parameters):
    """
    Calculate the stack thin layers thickness for fit a experimental transmissivity signal
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    cost: Int (float)
        Normalize the MSE to be between 0 and 1 
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    signal = parameters.get('Signal_fit')

    # Creation of the stack
    d_Stack = np.array(individual)
    Mat_Stack = parameters.get('Mat_Stack')
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # Normalize the MSE to be between 0 and 1
    #  If T and signal are identical, the cost will be 0, and the more they differ, the closer the cost will be to 1.
    cost = normalized_mse(T, signal)

    return cost


def evaluate_fit_T2face(individual, parameters):
    """
    Calculate the stack thin layers thickness for fit a experimental transmissivity signal
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    cost: Int (float)
        Normalize the MSE to be between 0 and 1 
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    signal = parameters.get('Signal_fit')

    # Creation of the stack
    d_Stack = np.array(individual)

    # Calculation of the RTA
    d_Stack[0] = 1000
    d_Stack[1] = 2e6
    R, T, A = RTA_curve_inco(d_Stack, parameters)

    # Normalize the MSE to be between 0 and 1
    #  If T and signal are identical, the cost will be 0, and the more they differ, the closer the cost will be to 1.
    cost = normalized_mse(T, signal)

    return cost


def evaluate_fit_RT(individual, parameters):
    """
    Calculate the stack thin layers thickness for fit a experimental transmissivity signal
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    cost: Int (float)
        Normalize the MSE to be between 0 and 1 
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    signal = parameters.get('Signal_fit')  # Spectral reflectance
    signal_2 = parameters.get('Signal_fit_2')  # Spectral transmittance

    if signal.shape != signal_2.shape:
        raise ValueError("signal and signal_2 must have the same dimensions")

    if np.array_equal(signal, signal_2):
        raise ValueError("signal and signal_2 are egal. Check your data")

    # Creation of the stack
    d_Stack = np.array(individual)
    Mat_Stack = parameters.get('Mat_Stack')
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)

    # Normalize the MSE to be between 0 and 1
    # cost for R
    # cost_R = chi_square(R, signal)
    cost_R = normalized_mse(R, signal)
    # cost for T
    # cost_T = chi_square(T, signal_2)
    cost_T = normalized_mse(T, signal_2)

    # Total cost average of R and T
    return (cost_R + cost_T)/2


def normalized_mse(R, signal):
    """
Parameters
----------
R : Array of floats
    Optical properties (reflectance or transmisstance) calculate by the code
signal : Array of floats
    Reflectance mesured on real sample.

Returns
-------
normalized_error : float
    Normalize the MSE to be between 0 and 1. 0 mean the reflectance calculate fit with the mesured
    """
    # Check that R and signal have the same dimensions
    if R.shape != signal.shape:
        raise ValueError(
            "Optical properties and signal must have the same dimensions")

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((R - signal) ** 2)

    # Calculate the range of possible values (max difference in signal)
    range_max = np.max(signal) - np.min(signal)

    # Normalize the MSE to be between 0 and 1
    normalized_error = mse / (range_max ** 2)

    return normalized_error


def chi_square(R, signal):
    """
Parameters
----------
R : Array of floats
    Optical properties (reflectance or transmisstance) calculate by the code
signal : Array of floats
    Reflectance mesured on real sample.

Returns
-------
normalized_error : float
    Normalize the chi_square to be between 0 and 1. 0 mean the reflectance calculate fit with the mesured
    """
    if R.shape != signal.shape:
        raise ValueError("R and signal must have the same dimensions")

    # Prevent division by zero by adding a small epsilon to signal
    epsilon = 1e-10
    chi_squared = np.sum(((R - signal) ** 2) / (signal + epsilon))

    # Calculate the maximum possible chi-square (assuming max deviation)
    max_deviation = np.sum(
        ((np.max(signal) - signal) ** 2) / (signal + epsilon))

    # Normalize chi-square to be between 0 and 1
    normalized_chi_sq = chi_squared / max_deviation if max_deviation != 0 else 0

    return normalized_chi_sq


def Stack_coherency(d_Stack, Mat_Stack, coherency_limit):
    """
Parameters
----------
d_Stack : List of floats
    List of thicknesses. 
Mat_Stack : List of strings
    List of materials.

Returns
-------
cl : List of string
    List of "i" or "c" to indicate if a layer is coherent or incoherent.
coherency : Boolean 
    True of all layers in the stack are coherent
    False if at least one or several layers in the stack are incoherent 
    """

    # Liste of coherency for function calculate_rat() from solcore package
    cl = []
    coherency = True  # All thin layers are coherent
    for i in range(len(d_Stack)):
        # If thicknesses up to 1000 nm : non coherency.
        # If air or vaccum in the stack : non coherency
        if d_Stack[i] > coherency_limit or Mat_Stack[i] in ["air", "vaccum", "Air", "Vaccum"]:
            coherency = False
            # add ["i"] if a layers is incoherent
            cl = cl + ["i"]
            coherency = False
            # If the layers #0 (the substrat) is no coherent SolPOC can solve
            if i == 0:
                coherency = True
        else:
            # add ["c"] if a layers is coherent
            cl = cl + ["c"]
    cl.reverse()  # must reverse the list for function calculate_rat() from solcore

    return cl, coherency


def Made_SolCORE_Stack(d_Stack, Wl, n_Stack, k_Stack):
    """
This fonction created a stack used for Solcore package
Solcore package is used if one layers in incoherent in the stack 

Parameters
----------
d_Stack : List of floats
    List of thicknesses. 
Wl : numpy array
    The list of the wavelenght.
n_Stack : array, in  3 dimensional
    The real part of the refractive index
k_Stack : array, in 3 dimensional
    The complexe part of the refractive index

Returns
-------
stack : solcore.structure.Structure
    Specific type used by Solcore package

    """
    # build the stack from solcore
    stack = Structure([])
    d_Stack = d_Stack.flatten().tolist()
    for i in range(len(d_Stack)-1, -1, -1):  # je ballaye la liste à l'envers
        stack.append([d_Stack[i], Wl, n_Stack[:, i], k_Stack[:, i]])

    return stack


def RTA_curve(individual, parameters):
    """
Parameters
----------
individual : numpy array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractif index.
parameters : Dict
    Dictionary with contain all "global" variables.

Returns
-------
R : List
    Reflectance of the stack, according the wavelenght list in the parameters.
T : List
    Transmittance of the stack, according the wavelenght list in the parameters.
A : List
    Absoptance of the stack, according the wavelenght list in the parameters.
    """
    Wl = parameters.get('Wl')  # , np.arange(280,2505,5))
    Ang = parameters.get('Ang')  # , 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')

    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    return R, T, A


def RTA_curve_inco(individual, parameters):
    """
Parameters
----------
individual : numpy array
    individual is an output of optimisation method (algo). 
    List of thickness in nm, witch can be added with volumic fraction or refractif index.
parameters : Dict
    Dictionary with contain all "global" variables.

Returns
-------
R : List
    Reflectance of the stack, according the wavelenght list in the parameters.
T : List
    Transmittance of the stack, according the wavelenght list in the parameters.
A : List
    Absoptance of the stack, according the wavelenght list in the parameters.
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')

    if 'coherency_limit' in parameters:
        coherency_limit = parameters.get('coherency_limit')
    else:
        coherency_limit = 2000

    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Check if one or several layer are coherent
    # uncoherent if thickness up to 2500 nm or if presence of air or vaccum in the stack
    cl, coherency = Stack_coherency(
        d_Stack.flatten().tolist(), Mat_Stack, coherency_limit)  # d_Stack is array

    # if coherency is True, I use my own function (much more faster)
    if coherency:
        print(" All layers are coherent")
        R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    else:  # if coherency is True, I use my own function (faster)

        print(" At least one layer is incoherent")
        # Fonction calculate_rat from solcore
        stack = Made_SolCORE_Stack(d_Stack, Wl, n_Stack, k_Stack)
        rat_data = calculate_rat(
            stack, angle=Ang, wavelength=Wl, coherent=False, coherency_list=cl)
        size = np.shape(rat_data["A_per_layer"])
        R = rat_data["A_per_layer"][0]
        T = rat_data["A_per_layer"][size[0]-2]
        A = 1 - R - T

    return R, T, A


def get_seed_from_randint(size=None, rng=None):
    """Uses numpy randint to generate integer seeds between uint32 min and uint32 max (i.e. 0 and 2^32 - 1).

    Args:
        size (int, optional): number of integer seeds to generate. If None, a single scalar is returned. Defaults to None.
        rng (numpy RNG, optional): If given, sets the numpy RNG onto which randint is called. If left to be None, np.random.randint is used. Defaults to None.

    Returns:
        int or ndarray(uint32): the seed (or array of seeds) generated
    """
    if rng is None:
        rng = np.random
    return rng.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max,
                       size=size, dtype=np.uint32)


def generate_population(chromosome_size, parameters):
    """
See : function optimize_gn.
\nThis function generates the 1st generation for the genetic optimization process. 
\nThat is, a series of thin film stacks, each thickness of which is within the range for genetic algo, optimize_gn'.

Parameters
----------
chromosome_size : Int 
    The lenght of individual, so the number of chromosone 
parameters : Dict
    Dictionary which contains necessary settings to generate a population.

Returns
-------
population : numpy array
    Contains generated population.
    """
    pop_size = parameters.get('pop_size')
    Th_range = parameters.get('Th_range')
    Th_Substrate = parameters.get('Th_Substrate')
    # I search Je vais chercher d_Stack_Opt
    d_Stack_Opt = parameters.get('d_Stack_Opt')

    # If d_Stack_Opt doesn't exist in parameters, he is already created but he has a NoneType
    # That means that all thicknesses must be optimized

    if isinstance(d_Stack_Opt, type(None)):
        d_Stack_Opt = ["no"] * chromosome_size

    population = []
    for i in range(pop_size):
        # 0 and 200 are, in nm, the ranges of thickness of the substrate
        individual = [Th_Substrate]
        for j in range(chromosome_size):
            if isinstance(d_Stack_Opt[j], (int, float)):
                individual += [d_Stack_Opt[j]]
            else:
                individual += [np.random.randint(Th_range[0], Th_range[1])]
        population.append(individual)

    return population


def selection_min(population, evaluate, evaluate_rate, parameters):
    """
Parameters
----------
population : List of array 
    Population is a list of the different indivudals.\n
    Each individual is a stack, so a list of ticknesses.
evaluate : callable
    The name of an evaluation fonction (the cost function), defined previously. 
evaluate_rate : Float
    Rate of individuals in the population that will be conserved as parents in the next generation.
parameters : Dict
    "parameters" is a dictionary which contains all "global" variables.

Returns
-------
parents : List
Selected individuals to become parents after the reproduction process.
\nUses the evaluate function to calculate individuals' performances according to a function.
\nIn the evaluate program, if the function callvalue is evaluate_R_s, the code replaces "evaluate" by "evaluate_R_s".
\n=> the function name is adaptative ! 

Selects according to the minimum.
    """
    scores = [evaluate(individual, parameters) for individual in population]
    parents = []
    for i in range(int(len(population)*evaluate_rate)):
        parent1 = population[scores.index(min(scores))]
        scores.pop(scores.index(min(scores)))
        parents.append(parent1)
    return parents


def selection_max(population, evaluate, evaluate_rate, parameters):
    """
    Selects the maximum.
    """
    scores = [evaluate(individual, parameters) for individual in population]
    parents = []
    for i in range(int(len(population)*evaluate_rate)):
        parent1 = population[scores.index(max(scores))]
        scores.pop(scores.index(max(scores)))
        parents.append(parent1)
    return parents

# New crossover version, by mask. # We totally mix the genes


def crossover(parents, crossover_rate, pop_size):
    """
See : optimize_gn.
    """
    children = []
    for i in range((pop_size-len(parents))//2):  # We make two child for each parents
        parent1 = parents[np.random.randint(0, len(parents)-1)]
        parent2 = parents[np.random.randint(0, len(parents)-1)]
        if np.random.uniform(0, 1) < crossover_rate:
            mask = [np.random.choice([0, 1]) for _ in range(len(parent1))]
            child1 = [parent1[i] if mask[i] == 0 else parent2[i]
                      for i in range(len(parent1))]
            child2 = [parent2[i] if mask[i] == 0 else parent1[i]
                      for i in range(len(parent1))]
            children.append(child1)
            children.append(child2)
        else:
            children.append(parent1)
            children.append(parent2)
    return children

# New version of the mutation
# Each gene of the child has mutatin_rate chance of mutate


def mutation(children, mutation_rate, mutation_delta, d_Stack_Opt):
    """
    See : optimize_gn.

    This function enables the mutation of the childs (the new stacks), during their births.\n
    During his birth, a child has a % of chance (mutatin_rate) to mutate.\n
    Some thicknesses vary about +/- mutation_delta.\n
    Addition of an if loop to avoid a negative thickness.
    """
    for i in range(1, len(children)):
        for j in range(np.shape(children)[1] - 1):
            if np.random.uniform(0, 1) < mutation_rate:
                # Check if d_Stack_Opt[j] is an int or float
                if isinstance(d_Stack_Opt[j], (int, float)):
                    # If it is, use the value from d_Stack_Opt to mutate the child
                    children[i][j + 1] = d_Stack_Opt[j]
                else:
                    # Otherwise, perform random mutation
                    children[i][j +
                                1] += np.random.uniform(-mutation_delta, mutation_delta)
                    if children[i][j + 1] <= 0:
                        children[i][j + 1] = 0
    return children


def optimize_ga(evaluate, selection, parameters):
    """
Parameters
----------
evaluate : String
    Name of the evaluation fonction.
selection : String
    Name of the selection fonction.

Returns
-------
best_solution : numpy array
    The best stack of thin film (a list a thickness = individual) which provides the high cost function.
dev : numpy array
    The value of the best solution during the optimisation process.
nb_run : Int 
    The number of epoch.
seed : Int
    Value of the seed, used in the random number generator.
    """
    Mat_Stack = parameters.get('Mat_Stack')
    mod = parameters.get('Mod_Algo')
    pop_size = parameters.get('pop_size')
    crossover_rate = parameters.get('crossover_rate')
    evaluate_rate = parameters.get('evaluate_rate')
    mutation_rate = parameters.get('mutation_rate')
    mutation_delta = parameters.get('mutation_delta')
    Precision_AlgoG = parameters.get('Precision_AlgoG')
    nb_generation = parameters.get('nb_generation')
    d_Stack_Opt = parameters.get('d_Stack_Opt')

    # Seed
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else:
        seed = random.randint(1, 2**31)
        np.random.seed(seed)

    np.random.seed(np.random.randint(1, 2**31))

    # Settings of the optimization
    population = np.zeros(0)
    dev = float("inf")
    dev_tab = []
    nb_run = 0
    chromosome_size = len(Mat_Stack) - 1  # Number of thin layers
    population = generate_population(chromosome_size, parameters)

    if mod == "for":
        """
        The "for" mod launches the genetic algorithm for an accurate number of generations'
        """
        for i in range(nb_generation):
            parents = selection(population, evaluate,
                                evaluate_rate, parameters)
            children = crossover(parents, crossover_rate, pop_size)
            children = mutation(children, mutation_rate,
                                mutation_delta, d_Stack_Opt)
            population = parents + children
            scores = [evaluate(individual, parameters)
                      for individual in population]
            dev = np.std(scores)
            dev_tab.append(dev)
            nb_run = nb_run + 1
            # Final optimization test
    else:
        """
        The "while" mod (if we don't write for) launches the genetic algorithm for an infinite number of generations, while the algorithm hasn't talked
        """
        while dev > Precision_AlgoG:
            parents = selection(population, evaluate, evaluate_rate)
            children = crossover(parents, crossover_rate, pop_size)
            children = mutation(children, mutation_rate, mutation_delta)
            population = parents + children
            # Final test optimization
            scores = [evaluate(individual, parameters)
                      for individual in population]
            dev = np.std(scores)
            dev_tab.append(dev)
            nb_run = nb_run + 1
    # End of the optimization
    scores = [evaluate(individual, parameters) for individual in population]
    # dev = np.std(scores)
    # dev = "{:.2e}".format(dev)

    # /!\ Can be a problem because we keep the minimum of the best scores here.
    # But we can optimize by looking for the maximum.
    # But if the optimization is good, the minimum of the best scores should be equivalent to te maximum

    best_solution = population[scores.index(max(scores))]
    return best_solution, dev_tab, nb_run, seed


def optimize_strangle(evaluate, selection, parameters):
    """
Parameters
----------
evaluate : String
    Name of the evaluation function.
selection : String
    Name of the selection function.

Returns
-------
best_solution : numpy array
    The best stack of thin film (a list a thickness = individual) which provides the high cost function. 
dev : numpy array
    The value of the best solution during the optimization process.
nb_run : Int 
    The number of epoch.
seed : Int
    Value of the seed, used in the random number generator.
    """
    # I search for the variables in the settings
    Mat_Stack = parameters.get('Mat_Stack')
    mod = parameters.get('Mod_Algo')
    # Settings of the optimization
    pop_size = parameters.get('pop_size')
    evaluate_rate = parameters.get('evaluate_rate')
    Precision_AlgoG = parameters.get('Precision_AlgoG')
    nb_generation = parameters.get('nb_generation')

    # Option 1
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else:
        seed = random.randint(1, 2**32 - 1)
        np.random.seed(seed)

    # Launch of the problem
    population = np.zeros(0)
    dev = float("inf")
    tab_dev = []
    chromosome_size = len(Mat_Stack) - 1  # Number of thin layers
    population = generate_population(chromosome_size, parameters)
    if mod == "for":
        nb_run = 0
        for i in range(nb_generation):
            parents = selection(population, evaluate,
                                evaluate_rate, parameters)
            children = children_strangle(pop_size, parents, chromosome_size)
            population = parents + children
            scores = [evaluate(individual, parameters)
                      for individual in parents]
            dev = np.std(scores)
            tab_dev.append(dev)
            nb_run = nb_run + 1
            # Final test optimization
    else:
        dev = float("inf")
        while dev > Precision_AlgoG:
            parents = selection(population, evaluate,
                                evaluate_rate, parameters)
            children = children_strangle(pop_size, parents, chromosome_size)
            population = parents + children
            # Final test optimization
            scores = [evaluate(individual, parameters)
                      for individual in parents]
            dev = np.std(scores)
            tab_dev.append(dev)
            nb_run = nb_run + 1
    # End of the optimization
    scores = [evaluate(individual, parameters) for individual in population]
    dev = np.std(scores)
    dev = "{:.2e}".format(dev)

    # /!\ Can be a problem because we keep the minimum of the best scores here.
    # But we can optimize by looking for the maximum.
    # But if the optimization is good, the minimum of the best scores should be equivalent to te maximum

    best_solution = population[scores.index(max(scores))]

    return best_solution, tab_dev, nb_run, seed


def children_strangle(pop_size, parents, chromosome_size):
    """
See : the function optimize_strangle.

Parameters
----------
pop_size : Int
    The number of individuals in the population.
parents : List of array 
    List of individuals selected for being the parent of the next generation.
chromosome_size : Int
    The length of each individual.

Returns
-------
children : List of array 
    List of individuals born from the parent.
    """
    children = []
    for i in range(pop_size-len(parents)):
        # 0 and 200 are, in nm, the ranges of the substrate
        individual = [1000000]
        for j in range(chromosome_size):
            min_values = min([sublist[j+1] for sublist in parents])
            max_values = max([sublist[j+1] for sublist in parents])
            individual = individual + \
                [np.random.randint(min_values, max_values+1)]
        children.append(individual)
    return children


def DEvol(f_cout, f_selection, parameters):
    """
Main author : A.Moreau, Photon team, University of Clermont Auvergne, France and Antoine Grosjean
"This DE is a current to best. Hypertuned on the chirped problem.
Abrupt elimination of individuals not respecting the bounds
(compare with what happens if you just put back to the edge
could be a good idea on some problems)"

Parameters
----------

evaluate : Callable 
\nevaluation fonction, give in evaluate
\nselection : Callable
\nselection fonction, give in selection

Returns
-------
best_solution : numpy array
    The best stack of thin film (a list a thickness = individual) which provide the high cost function 
dev : numpy array
    The value of the best solution during the optimisation process
nb_run : Int 
    The number of epoch
seed : Int
    Value of the seed, used in the random number generator
    """
    selection = f_selection.__name__,

    # DE settings - potential settings of the function
    # cr=0.5; # Probability to give parents settings to his child.
    cr = parameters.get('mutation_rate')
    f1 = parameters.get('f1')  # f1=0.9;
    f2 = parameters.get('f2')  # f2=0.8;

    # Following seed problem when using the code, the seed can be manually targeting

    # Option 1
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else:
        seed = random.randint(1, 2**32 - 1)
        np.random.seed(seed)

    # Calculation of the budget :
    pop_size = parameters.get('pop_size')
    nb_generation = parameters.get('nb_generation')
    budget = pop_size * nb_generation

    # I give the population value
    population = pop_size

    # calculation of the X_min(s) and the X_max(s)
    Th_range = parameters.get('Th_range')
    vf_range = parameters.get('vf_range')
    Th_Substrate = parameters.get('Th_Substrate')
    Mat_Stack = parameters.get('Mat_Stack')
    if 'nb_layer' in parameters:
        nb_layer = parameters.get('nb_layer')
    else:
        nb_layer = 0

    chromosome_size = len(Mat_Stack) + nb_layer - 1  # Number of thin layers

    d_Stack_Opt = parameters.get('d_Stack_Opt')
    if isinstance(d_Stack_Opt, type(None)):
        d_Stack_Opt = ["no"] * chromosome_size

    X_min = [Th_Substrate]
    X_max = [Th_Substrate]
    for i in range(chromosome_size):
        if isinstance(d_Stack_Opt[i], (int, float)):
            X_min += [d_Stack_Opt[i]]
            X_max += [d_Stack_Opt[i]]
        else:
            X_min += [Th_range[0]]
            X_max += [Th_range[1]]

    if 'n_range' in parameters:
        n_range = parameters.get('n_range')
        for i in range(nb_layer):
            X_min += [n_range[0]]
            X_max += [n_range[1]]

    if 'vf_range' in parameters:
        vf_range = parameters.get('vf_range')
        for i in range(len(Mat_Stack)):
            if "-" in Mat_Stack[i]:
                X_min += [vf_range[0]]
                X_max += [vf_range[1]]
            else:
                X_min += [vf_range[0]]
                X_max += [vf_range[0]]

    """
    idea ; I check all the Mat_stack list. I create an X_max_2. When I found a - 
    I broadcast between vf[0] and vf[1] between X_max and X min, else Xmin_2 = Xmax_2 = 0 
    If at the end I have only 0, I do nothing. If I have other values than = 0 , I add the vector'
    for s in Mat_Stack: 
        if "-" in s:
            no_dash = False
            break
            
    if no_dash: # If no_dask is true, i go into the loop
        for i in range(chromosome_size):
            X_min += [vf_range[0]]
            X_max += [vf_range[1]]*
    """

    # I put the lists in array
    X_min = np.array(X_min)
    X_max = np.array(X_max)

    # End of the code lines of COPS

    n = X_min.size

    # Initialization of the population
    omega = np.zeros((population, n))
    cost = np.zeros(population)
    # Random draw in the range between X_min and X_max.
    for k in range(0, population):
        omega[k] = X_min+(X_max-X_min)*np.random.random(n)
        # Change, because I usually want to maximize. In the other algorithms, a function
        # selection is used.
        if selection[0] == "selection_min":
            cost[k] = f_cout(omega[k], parameters)
        elif selection[0] == "selection_max":
            cost[k] = 1-f_cout(omega[k], parameters)

    # Who's the best ?
    who = np.argmin(cost)
    best = omega[who]
    # Initializations
    evaluation = population
    convergence = []
    generation = 0
    convergence.append(cost[who])

    mutation_DE = parameters.get('mutation_DE')

    # DE loop
    while evaluation < budget-population:
        for k in range(0, population):
            crossover = (np.random.random(n) < cr)
            # *crossover+(1-crossover)*omega[k] : crossover step

            if mutation_DE == "current_to_best":
                # current to best
                # y(x) = x + F1 (a-b) + F2(best - x)
                X = (omega[k] + f1*(omega[np.random.randint(population)] - omega[np.random.randint(
                    population)])+f2*(best-omega[k]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_to_best":
                # rand to best
                # y = c + F1 *(a-b) + F2(best - c)
                X = (omega[np.random.randint(population)] + f1*(omega[np.random.randint(population)]-omega[np.random.randint(
                    population)])+f2*(best-omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_1":
                # best 1
                X = (best - f1*(omega[np.random.randint(population)] -
                     omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_2":
                # best 2
                X = (best + f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)] +
                     omega[np.random.randint(population)] - omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_1":
                # rand
                a = omega[np.random.randint(population)]
                b = omega[np.random.randint(population)]
                c = omega[np.random.randint(population)]
                X = (a + f1*(b - c))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_2":
                # rand 2
                a = omega[np.random.randint(population)]
                b = omega[np.random.randint(population)]
                c = omega[np.random.randint(population)]
                d = omega[np.random.randint(population)]
                X = (a + f1*(a - b + c - d))*crossover+(1-crossover)*omega[k]

            if np.prod((X >= X_min)*(X <= X_max)):
                if selection[0] == "selection_min":
                    tmp = f_cout(X, parameters)
                elif selection[0] == "selection_max":
                    tmp = 1-f_cout(X, parameters)
                evaluation = evaluation+1
                if (tmp < cost[k]):
                    cost[k] = tmp
                    omega[k] = X

        generation = generation+1
        # print('generation:',generation,'evaluations:',evaluation)
        #
        who = np.argmin(cost)
        best = omega[who]
        convergence.append(cost[who])

    convergence = convergence[0:generation+1]

    return [best, convergence, budget, seed]


def DEvol_Video(f_cout, f_selection, parameters):
    """ 
Sub version of DE.
Used by the main author of COPS for provide video of the optimization process. 
The stack tickness is save during the process
    """
    selection = f_selection.__name__,

# DE settings - pontential settings of the function
    cr = parameters.get('mutation_rate')
    # cr=0.5; # Probability to give parents settings to his child.
    f1 = parameters.get('f1')
    f2 = parameters.get('f2')
    """
    After some problems with Colossus server, I fix the seed into the function.
    Choose one of the different options
    """
    # Option 1
    seed = parameters.get('seed')
    np.random.seed(seed)

    # Calculation of the budget :
    pop_size = parameters.get('pop_size')
    nb_generation = parameters.get('nb_generation')
    # print(nb_generation)
    budget = pop_size * nb_generation
   # print(budget)

    # I give the population value
    population = pop_size

    # calculation of the X_min(s) and X_max(s)
    Th_range = parameters.get('Th_range')
    vf_range = parameters.get('vf_range')
    Th_Substrate = parameters.get('Th_Substrate')
    Mat_Stack = parameters.get('Mat_Stack')
    if 'nb_layer' in parameters:
        nb_layer = parameters.get('nb_layer')
    else:
        nb_layer = 0

    chromosome_size = len(Mat_Stack) + nb_layer - 1  # Number of thin layers

    X_min = [Th_Substrate]
    X_max = [Th_Substrate]
    for i in range(chromosome_size):
        X_min += [Th_range[0]]
        X_max += [Th_range[1]]

    if 'n_range' in parameters:
        n_range = parameters.get('n_range')
        for i in range(nb_layer):
            X_min += [n_range[0]]
            X_max += [n_range[1]]

    if 'vf_range' in parameters:
        vf_range = parameters.get('vf_range')
        for i in range(len(Mat_Stack)):
            if "-" in Mat_Stack[i]:
                X_min += [vf_range[0]]
                X_max += [vf_range[1]]
            else:
                X_min += [vf_range[0]]
                X_max += [vf_range[0]]

    # I put the lists in array
    X_min = np.array(X_min)
    X_max = np.array(X_max)

    # End of the code line of COPS

    n = X_min.size

    # Initialization of the population
    omega = np.zeros((population, n))
    cost = np.zeros(population)
    # Random draw in the range between X_min and X_max.
    for k in range(0, population):
        omega[k] = X_min+(X_max-X_min)*np.random.random(n)
        # Change, because I usually want to maximize. In the other algorithm, a function
        # selection is used.
        if selection[0] == "selection_min":
            cost[k] = f_cout(omega[k], parameters)
        elif selection[0] == "selection_max":
            cost[k] = 1-f_cout(omega[k], parameters)

    # Who's the best ?
    who = np.argmin(cost)
    best = omega[who]

    """ 
    Bug fix on the 27/06/2023. We shouldn't use .append but .copy
    Reprodiuce the bug
    
    best_tab= []
    best_tab.append(best)
    """
    # print(best)
    best_tab = np.copy(best)
    # print(best_tab)

    # Initializations
    evaluation = population
    convergence = []
    generation = 0
    convergence.append(cost[who])

    mutation_DE = parameters.get('mutation_DE')

    # DE loop
    while evaluation < budget-population:
        for k in range(0, population):
            crossover = (np.random.random(n) < cr)

            if mutation_DE == "current_to_best":
                # current to best
                # y(x) = x + F1 (a-b) + F2(best - x)
                X = (omega[k] + f1*(omega[np.random.randint(population)] - omega[np.random.randint(
                    population)])+f2*(best-omega[k]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_to_best":
                # rand to best
                # y = c + F1 *(a-b) + F2(best - c)
                X = (omega[np.random.randint(population)] + f1*(omega[np.random.randint(population)]-omega[np.random.randint(
                    population)])+f2*(best-omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_1":
                # best 1
                X = (best - f1*(omega[np.random.randint(population)] -
                     omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_2":
                # best 2
                X = (best + f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)] +
                     omega[np.random.randint(population)] - omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_1":
                # rand
                a = omega[np.random.randint(population)]
                b = omega[np.random.randint(population)]
                c = omega[np.random.randint(population)]
                X = (a + f1*(b - c))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_2":
                # rand 2
                a = omega[np.random.randint(population)]
                b = omega[np.random.randint(population)]
                c = omega[np.random.randint(population)]
                d = omega[np.random.randint(population)]
                X = (a + f1*(a - b + c - d))*crossover+(1-crossover)*omega[k]

            # *crossover+(1-crossover)*omega[k] : etape de crossover

            if np.prod((X >= X_min)*(X <= X_max)):
                if selection[0] == "selection_min":
                    tmp = f_cout(X, parameters)
                elif selection[0] == "selection_max":
                    tmp = 1 - f_cout(X, parameters)
                evaluation = evaluation+1
                if (tmp < cost[k]):
                    cost[k] = tmp
                    omega[k] = X

        generation = generation+1
        # print('generation:',generation,'evaluations:',evaluation)
        #
        who = np.argmin(cost)
        best = omega[who]
        """ 
        Old buggy version : 
        #best_tab.append(best)
        Bug fix.
        """
        best_tab = np.vstack((best_tab, best))
        # np.append(best_tab, best)
        convergence.append(cost[who])

    convergence = convergence[0:generation+1]

    return [best, best_tab, convergence, budget, seed]


class Particle():
    """
Class Particle, for PSO (Particle Swarm Optimization) optimization.
\nSee function "PSO()".
    """

    def __init__(self, position, velocity, score_ref=0):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = score_ref


def PSO(evaluate, selection, parameters):
    """
PSO : particle swarm optimization 
Need to work with class Particle()

The proposed parameters for PSO are defaults values. They are NOT optimized for coatings optimization:
    inertia_weight = 0.8
    cognitive_weight = 1.5
    social_weight = 1.5

Parameters
----------
evaluate : Callable 
    Evaluation fonction, give in evaluate.
selection : Callable
    Selection fonction, give in selection.

Returns
-------
best_solution : numpy array
    The best stack of thin film (a list a thickness = individual) which provide the high cost function. 
dev : numpy array
    The value of the best solution during the optimisation process.
nb_run : Int 
    The number of epoch.
seed : Int
    Value of the seed, used in the random number generator.

Need  generate_neighbor() and acceptance_probability() functions.
    """

    # Stack : refractive index of the materials. Each colonne is a different layer. Each lign is a different wavelenght. Z axe (if present) is for mixture material
    Mat_Stack = parameters.get('Mat_Stack')
    # number of particules is storage in pop_size
    num_particles = parameters.get('pop_size')
    # number of particules is storage in nb_generation
    num_iterations = parameters.get('nb_generation')
    Th_Substrate = parameters.get('Th_Substrate')

    selection = selection.__name__,

    # Parameters just for PSO. They are NOT optimized for coatings optimization or photonics
    inertia_weight = 0.8
    cognitive_weight = 1.5
    social_weight = 1.5

    # Fixation of the seed
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else:
        seed = random.randint(1, 2**32 - 1)
        np.random.seed(seed)

    # Creation of lower_bound and upper bound
    Th_range = parameters.get('Th_range')
    chromosome_size = len(Mat_Stack) - 1  # Number of thin layers
    # Define lower bounds for each dimension
    lower_bound = np.array([Th_range[0]] * chromosome_size)
    # I add the thickness of the substrate in the bounds
    lower_bound = np.insert(lower_bound, 0, Th_Substrate)
    # Define upper bounds for each dimension
    upper_bound = np.array([Th_range[1]] * chromosome_size)
    # I add the thickness of the substrate in the bounds
    upper_bound = np.insert(upper_bound, 0, Th_Substrate)

    # Start
    num_dimensions = len(lower_bound)
    particles = []
    convergence = []  # List of best values durint the optimization process
    # Initialization

    if selection[0] == "selection_min":
        global_best_position = np.zeros(num_dimensions)
        global_best_score = float('inf')
        score_ref = float('inf')

    elif selection[0] == "selection_max":
        global_best_position = np.array(0 * num_dimensions)
        global_best_score = 0
        score_ref = 0

    # Initialization
    for _ in range(num_particles):
        position = np.random.uniform(lower_bound, upper_bound)
        velocity = np.random.uniform(lower_bound * 0.1, upper_bound * 0.1)
        particle = Particle(position, velocity, score_ref)
        particles.append(particle)

        # Update of the global best score
        if selection[0] == "selection_min":
            score = evaluate(position, parameters)
            if score < global_best_score:
                global_best_score = score
                global_best_position = position

            # Update of the personal best score of the particle
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = position

        elif selection[0] == "selection_max":
            score = evaluate(position, parameters)
            if score > global_best_score:
                global_best_score = score
                global_best_position = position

            # Update of the personal best score of the particle
            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = position

    convergence.append(global_best_score)  # First best values

    # Optimization
    for _ in range(num_iterations):
        for particle in particles:
            # update of the velocity and the position
            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_weight * np.random.rand() * (particle.best_position - particle.position) +
                                 social_weight * np.random.rand() * (global_best_position - particle.position))
            particle.position = np.clip(
                particle.position + particle.velocity, lower_bound, upper_bound)

            # Update of the global best score
            if selection[0] == "selection_min":
                score = evaluate(particle.position, parameters)
                # Update of the personal and the global best score
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position

                if score < global_best_score:
                    global_best_score = score
                    # Adding the newest best values
                    convergence.append(global_best_score)
                    global_best_position = particle.position

            if selection[0] == "selection_max":
                score = evaluate(particle.position, parameters)
                # Update of the personal and the global best score
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position

                if score > global_best_score:
                    global_best_score = score
                    # Adding the newest best values
                    convergence.append(global_best_score)
                    global_best_position = particle.position

    # global_best_score : score (cost function) of the best position
    return [global_best_position, convergence, num_iterations, seed]


def generate_neighbor(solution, parameters):
    """
Generates a neighboring solution for the simulated annealing algorithm.

Parameters
----------
solution : array
    The current solution represented as a list.
parameters : Dict
    A dictionary containing relevant parameters and constraints.

Returns
-------
neighbor : array
    The generated neighboring solution obtained by modifying a randomly selected value in the solution.
    """
    Th_range = parameters.get('Th_range')

    neighbor = solution.copy()
    # random.randint start at 1 and not 0, because the 1st value is the substrat thickness, witch cannot be modified
    index = random.randint(1, len(neighbor) - 1)
    # Choose a random value between -1 and 1 for the selected sublist
    neighbor[index] = random.uniform(Th_range[0], Th_range[1])
    return neighbor


def acceptance_probability(current_score, new_score, temperature):
    """
Calculates the acceptance probability for the simulated annealing algorithm.

Args:
    current_score: The current score or energy.\n
    new_score: The score or energy of the new state.\n
    temperature: The current temperature of the system.

Returns:
    The acceptance probability based on the current and new scores and the temperature.
    """
    if new_score < current_score:
        return 1.0
    return math.exp((current_score - new_score) / temperature)


def simulated_annealing(evaluate, selection, parameters):
    """
Parameters
----------
evaluate : Callable 
    Evaluation function, give in evaluate.
selection : Callable
    Selection function, give in selection.

Returns
-------
best_solution : numpy array
    The best stack of thin film (a list a thickness = individual) which provides the high cost function.
dev : numpy array
    The value of the best solution during the optimization process.
nb_run : Int 
    The number of epoch.
seed : Int
    Value of the seed, used in the random number generator.

Need generate_neighbor() and acceptance_probability() functions.
    """
    # Stack : refractive index of the materials. Each colonne is a different layer. Each lign is a different wavelenght. Z axe (if present) is for mixture material
    Mat_Stack = parameters.get('Mat_Stack')
    Th_Substrate = parameters.get('Th_Substrate')
    # number of iteration of the annealing
    nb_generation = parameters.get('nb_generation')
    pop_size = parameters.get('pop_size')
    budget = pop_size * nb_generation
    Th_range = parameters.get('Th_range')

    # Get the name of the selection function
    selection = selection.__name__,

    # Settings of the simulated annealing
    initial_temperature = 4500.0
    cooling_rate = 0.95
    current_temperature = initial_temperature

    # Seed fixation
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else:
        seed = random.randint(1, 2**32 - 1)
        np.random.seed(seed)

    # Creation of lower_bound and upper bound

    chromosome_size = len(Mat_Stack) - 1  # Number of thin layers
    # Generation of the initial solution
    current_solution = [random.uniform(Th_range[0], Th_range[1]) for _ in range(
        chromosome_size)]  # Generate a random solution
    # I add the thickness of the substrate between bounds
    current_solution = np.insert(current_solution, 0, Th_Substrate)

    # Initialization
    best_solution = current_solution.copy()
    best_score = evaluate(best_solution, parameters)

    convergence = []  # List of best values durint the optimization process
    convergence.append(best_score)  # First best values
    i = 0

    # Start of annealing
    while i < budget:

        neighbor_solution = generate_neighbor(current_solution, parameters)

        # Evaluate the score of the neighbor according of
        if selection[0] == "selection_min":
            neighbor_score = evaluate(neighbor_solution, parameters)
        elif selection[0] == "selection_max":
            neighbor_score = 1 - evaluate(neighbor_solution, parameters)

        neighbor_score = evaluate(neighbor_solution, parameters)

        if acceptance_probability(evaluate(current_solution, parameters), neighbor_score, current_temperature) > random.uniform(0, 1):
            current_solution = neighbor_solution
            # Keeping the current solution, depending of the selection method (min or max)
            if selection[0] == "selection_max" and neighbor_score > best_score:
                best_solution = current_solution
                best_score = neighbor_score
                convergence.append(best_score)

            if selection[0] == "selection_min" and neighbor_score < best_score:
                best_solution = current_solution
                best_score = neighbor_score
                convergence.append(best_score)
        i = i + 1
        current_temperature *= cooling_rate

    # best_score : score (cost function) of the best solution
    return [best_solution, convergence, nb_generation, seed]


def generate_mutant(solution, step_size, Th_range):
    """
Function for One_plus_One optimisation method.

Parameters
----------
solution : List
    Initial solution.
step_size : Float
    Disruption amplitude.

Returns
-------
mutant : List
    List based on solution with random values added.
    """
    if step_size < 1:
        step_size = 1
    if step_size > Th_range[1]:
        step_size = Th_range[1]
    # Modification of the mutant start at 1 and not 0, because the 1st value is the substrat thickness, witch cannot be modified
    mutant = solution.copy()  # Copy of the initial solution
    for i in range(len(solution)-1):
        mutant[i+1] = np.random.normal(solution[i+1], step_size)
    # Modification if the mutant is below or upper the limite
    for i in range(1, len(mutant)):
        mutant[i] = max(min(mutant[i], Th_range[1]), Th_range[0])

    # return mutant.tolist()
    return mutant


def One_plus_One_ES(evaluate, selection, parameters):
    """
The algorithm mentioned here is referred to as One_plus_One instead of (1+1) 
because using (1+1) as a name for a function is not recommended. 
\nHowever, it is important to note that the presented algorithm may not be the (1+1)_ES version.
\nAlthough the algorithm presented here is (1+1)_ES, we cannot confirm with certainty 
that it is the exact (1+1)_ES implementation based on information at our disposal. 
\nSee P.Bennet thesis and/or Nikolaus Hansen and al. Comparing results of 31 algorithms from the black-box optimization
benchmarking BBOB-2009 | Proceedings of the 12th annual conference companion on Genetic
and evolutionary computation. 2010.

Parameters
----------
evaluate : Callable 
    evaluation fonction, give in evaluate
selection : Callable
    selection fonction, give in selection

Returns
-------
best_solution : numpy array
    The best stack of thn film (a list a thickness = individual) whitch provide the high cost function.
dev : numpy array
    The value of the best solution during the optimisation process.
nb_run : Int 
    The number of epoch.
seed : Int
    Value of the seed, used in the random number generator.
    """

    # Stack : refractive index of the materials. Each colonne is a different layer. Each lign is a different wavelenght. Z axe (if present) is for mixture material
    Mat_Stack = parameters.get('Mat_Stack')
    # Interation
    pop_size = parameters.get('pop_size')
    nb_generation = parameters.get('nb_generation')
    # print(nb_generation)
    num_iterations = pop_size * nb_generation
    Th_Substrate = parameters.get('Th_Substrate')
    Th_range = parameters.get('Th_range')

    # Step size scaling factor
    step_size_factor = 0.99

    # Get the selection function name
    selection = selection.__name__,

    # Parameter for (1+1)-ES
    initial_step_size = 10  # Taille de pas initiale

    # Fixation of the seed
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else:
        seed = random.randint(1, 2**32 - 1)
        np.random.seed(seed)

    # Creation of the solution

    chromosome_size = len(Mat_Stack) - 1  # Number of thin layers
    # Generation of the initial solution
    initial_solution = [np.random.uniform(Th_range[0], Th_range[1]) for _ in range(
        chromosome_size)]  # Generate a random solution
    # I add the thickness of the substrate between bounds
    initial_solution = np.insert(initial_solution, 0, Th_Substrate)

    current_solution = initial_solution
    current_step_size = initial_step_size

    current_score = evaluate(current_solution, parameters)

    convergence = []  # List of best values durint the optimization process
    convergence.append(current_score)

    for _ in range(num_iterations):
        mutant_solution = generate_mutant(
            current_solution, current_step_size, Th_range)
        mutant_score = evaluate(mutant_solution, parameters)

        if selection[0] == "selection_max" and mutant_score > current_score:
            current_solution = mutant_solution
            current_score = mutant_score
            convergence.append(current_score)
            current_step_size *= step_size_factor
        else:
            current_step_size /= step_size_factor

        if selection[0] == "selection_min" and mutant_score < current_score:
            current_solution = mutant_solution
            current_score = mutant_score
            convergence.append(current_score)
            current_step_size *= step_size_factor
        else:
            current_step_size /= step_size_factor

        if current_step_size > 50:
            current_step_size = 50
        if current_step_size < 0:
            current_step_size = 5

    return [current_solution, convergence, num_iterations, seed]


def Reflectivity_plot(parameters, Experience_results, directory):
    if 'evaluate' in parameters:
        evaluate = parameters.get("evaluate")
    Wl = parameters.get("Wl")
    R = Experience_results.get("R")
    Sol_Spec = parameters.get("Sol_Spec")
    if 'T_abs' in parameters:
        T_abs = parameters.get("T_abs")

    # Reflectivity plot
    fig, ax1 = plt.subplots()
    color = 'black'  # Basic colors availables : b g r c m y k w
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectivity (-)', color=color)
    if 'evaluate' in parameters and evaluate.__name__ == 'evaluate_rh':
        ax1.set_xscale('log')
    ax1.plot(Wl, R, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Code line to change y-axis, for reflectance
    # Disabled for automatic scaling

    ax1.set_ylim(0, 1)  # Change y-axis' scale
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
    ax2.plot(Wl, Sol_Spec, color=color)
    if 'evaluate' in parameters and evaluate.__name__ == 'evaluate_rh':
        BB_shape = BB(T_abs, Wl)
        # BB_shape is the black body's shape. According to the temperature, the black body's irradiance can be very higher
        # than solar spectrum. That's why I put the black body at the same height for this chart
        BB_shape = BB_shape*(max(Sol_Spec)/max(BB_shape))
        ax2.plot(Wl, BB_shape, color='orange', linestyle='dashed')

    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    ax2.set_ylim(0, 2)  # Change y-axis' scale
    plt.title("Optimum Reflectivity", y=1.05)
    # Save the plot.
    plt.savefig(directory + "/" + "Optimum_Reflectivity.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def Transmissivity_plot(parameters, Experience_results, directory):
    if 'evaluate' in parameters:
        evaluate = parameters.get("evaluate")
    Wl = parameters.get("Wl")
    T = Experience_results.get("T")
    Sol_Spec = parameters.get("Sol_Spec")
    if 'T_abs' in parameters:
        T_abs = parameters.get("T_abs")

    # Plot transmissivity
    fig, ax1 = plt.subplots()
    color = 'black'  # Basic colors availables : b g r c m y k w
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transmissivity (-)', color=color)
    if 'evaluate' in parameters and evaluate.__name__ == 'evaluate_rh':
        ax1.set_xscale('log')
    ax1.plot(Wl, T, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Line to change y-axis (for the reflectance)
    # Disabled for automatic scale
    ax1.set_ylim(0, 1)  # Change y-axis' scale
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Solar Spectrum (W/m²nm⁻¹)', color=color)
    ax2.plot(Wl, Sol_Spec, color=color)
    if 'evaluate' in parameters and evaluate.__name__ == 'evaluate_rh':
        BB_shape = BB(T_abs, Wl)
        # BB_shape is the black body's shape. According to the temperature, the black body's irradiance can be very higher
        # than solar spectrum. That's why I put the black body at the same height for this chart
        BB_shape = BB_shape*(max(Sol_Spec)/max(BB_shape))
        ax2.plot(Wl, BB_shape, color='orange', linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    ax2.set_ylim(0, 2)  # Change y-axis' scale
    plt.title("Optimum Transmissivity", y=1.05)
    plt.savefig(directory + "/" + "Optimum_Transmissivity.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def OpticalStackResponse_plot(parameters, Experience_results, directory):
    Wl = parameters.get("Wl")
    R = Experience_results.get("R")
    T = Experience_results.get("T")
    A = Experience_results.get("A")

    fig, ax1 = plt.subplots()
    color = 'black'  # Basic colors available: b g r c m y k w
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel(
        'Reflectivity, transmissivity or absorptivity (-)', color=color)
    ax1.plot(Wl, R, 'black', label='Reflectivity')
    ax1.plot(Wl, T, 'blue', label='Transmissivity')
    ax1.plot(Wl, A, 'red', label='Absorptivity')
    ax1.tick_params(axis='y', labelcolor=color)
    # plt.title("Optical stack respond")
    ax1.set_ylim(0, 1)  # Change y-axis scale
    ax1.grid(True)
    plt.legend()
    plt.savefig(directory + "/" + "Optical_Stack_Response.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def Convergence_plots(parameters, Experience_results, directory):
    tab_dev = Experience_results.get("tab_dev")
    algo = parameters.get("algo")
    selection = parameters.get("selection")

    # normalized_data Type Array of float (nb_run, x)
    normalized_data = tab_dev

    if algo.__name__ == "DEvol":
        if selection.__name__ == "selection_max":
            normalized__data = [1 - x for x in normalized_data]

    # Interpolation de chaque ligne sur 1000 points
    num_points = 1000
    interpolated_data = []

    for row in normalized_data:
        # Filtrer les points non-NaN
        valid_indices = ~np.isnan(row)
        valid_points = np.arange(len(row))[valid_indices]
        valid_values = row[valid_indices]

        # Vérifier s'il y a suffisamment de points pour l'interpolation
        if len(valid_points) < 2:
            # Si moins de 2 points valides, remplir avec NaN
            interpolated_row = np.full(num_points, np.nan)
        else:
            # Créer la fonction d'interpolation en utilisant seulement les points valides
            interp_func = interp1d(
                valid_points, valid_values, kind='linear', fill_value='extrapolate')

            # Points où nous voulons interpoler
            new_points = np.linspace(
                valid_points[0], valid_points[-1], num_points)

            # Interpolation
            interpolated_row = interp_func(new_points)

        interpolated_data.append(interpolated_row)

    # Convertir en tableau numpy
    interpolated_data = np.array(interpolated_data)

    if algo.__name__ == "DEvol":
        if selection.__name__ == "selection_max":
            interpolated_data = [1 - x for x in interpolated_data]

    # Convertir en tableau numpy
    tab_dev = np.array(interpolated_data)

    # Trouver les indices des lignes ayant les meilleures valeurs max dans la dernière colonne
    # Indices des 6 plus grandes valeurs dans la dernière colonne
    best_indices = np.argsort(tab_dev[:, -1])[-3:]

    # Trier les indices pour obtenir les meilleurs en ordre croissant
    best_indices = best_indices[np.argsort(tab_dev[best_indices, -1])[::-1]]

    # Préparer les couleurs et labels dans le bon ordre
    colors = ['black', 'red', 'green']
    labels = ['1st', '2nd', '3rd']

    # Tracer les courbes correspondantes
    fig, ax = plt.subplots()

    for i, idx in enumerate(best_indices):
        ax.plot(np.linspace(0, 100, num_points),
                interpolated_data[idx], color=colors[i], label=labels[i])
    labels.reverse()

    ax.set_ylabel('Cost function (-)')  # Nommer l'axe des ordonnées
    # Nommer l'axe des abscisses (étapes)
    ax.set_xlabel('Percentage of budget (%)')
    ax.legend()  # Afficher la légende
    plt.title("Convergence Plots", y=1.05)
    plt.savefig(directory + "/" + "Convergence_Plots.png",
                dpi=300, bbox_inches='tight')
    plt.show()  # Afficher le graphique


def Convergence_plots_2(parameters, Experience_results, directory):
    tab_dev = Experience_results.get("tab_dev")
    algo = parameters.get("algo")
    selection = parameters.get("selection")

    # normalized_data Type Array of float (nb_run, x)
    normalized_data = tab_dev

    if algo.__name__ == "DEvol":
        if selection.__name__ == "selection_max":
            normalized__data = [1 - x for x in normalized_data]

    # Interpolation de chaque ligne sur 1000 points
    num_points = 1000
    interpolated_data = []

    for row in normalized_data:
        # Filtrer les points non-NaN
        valid_indices = ~np.isnan(row)
        valid_points = np.arange(len(row))[valid_indices]
        valid_values = row[valid_indices]

        # Vérifier s'il y a suffisamment de points pour l'interpolation
        if len(valid_points) < 2:
            # Si moins de 2 points valides, remplir avec NaN
            interpolated_row = np.full(num_points, np.nan)
        else:
            # Créer la fonction d'interpolation en utilisant seulement les points valides
            interp_func = interp1d(
                valid_points, valid_values, kind='linear', fill_value='extrapolate')

            # Points où nous voulons interpoler
            new_points = np.linspace(
                valid_points[0], valid_points[-1], num_points)

            # Interpolation
            interpolated_row = interp_func(new_points)

        interpolated_data.append(interpolated_row)

    # Convertir en tableau numpy
    interpolated_data = np.array(interpolated_data)

    if algo.__name__ == "DEvol":
        if selection.__name__ == "selection_max":
            interpolated_data = [1 - x for x in interpolated_data]

    # Convertir en tableau numpy
    tab_dev = np.array(interpolated_data)

    # Trouver les indices des lignes ayant les meilleures valeurs max dans la dernière colonne
    # Indices des 6 plus grandes valeurs dans la dernière colonne
    best_indices = np.argsort(tab_dev[:, -1])[-6:]

    # Trier les indices pour obtenir les meilleurs en ordre croissant
    best_indices = best_indices[np.argsort(tab_dev[best_indices, -1])[::-1]]
    # Préparer les couleurs et labels dans le bon ordre
    colors = ['black', 'red', 'green', 'blue', 'orange', 'purple']
    labels = ['1st', '2nd', '3rd', '4th', '5th', '6th']

    # Tracer les courbes correspondantes
    fig, ax = plt.subplots()

    for i, idx in enumerate(best_indices):
        ax.plot(np.linspace(0, 100, num_points),
                interpolated_data[idx], color=colors[i], label=labels[i])

    ax.set_ylabel('Cost function (-)')  # Nommer l'axe des ordonnées
    # Nommer l'axe des abscisses (étapes)
    ax.set_xlabel('Percentage of budget (%)')
    ax.legend()  # Afficher la légende
    plt.title("Convergence Plots", y=1.05)
    plt.savefig(directory + "/" + "Convergence_Plots_2.png",
                dpi=300, bbox_inches='tight')
    plt.show()  # Afficher le graphique


# parameters is not used but we put it in arguments to follow the same calling method than other function to be easier to understand and master
def Consistency_curve_plot(parameters, Experience_results, directory):
    tab_perf = Experience_results["tab_perf"]
    selection = parameters.get("selection")

    # Problem's convergence plot
    tab_perf_sorted = tab_perf.copy()
    if selection.__name__ == "selection_max":
        tab_perf_sorted.sort(reverse=True)
    if selection.__name__ == "selection_min":
        tab_perf_sorted.sort(reverse=False)

    fig, ax1 = plt.subplots()
    color = 'black'  # Basic colors availables : b g r c m y k w
    if max(tab_perf_sorted) - min(tab_perf_sorted) < 1e-4:
        ax1.set_ylim(np.mean(tab_perf_sorted) - 0.0005,
                     np.mean(tab_perf_sorted) + 0.0005)  # Change y-axis' scale

    ax1.set_xlabel('Best cases (left) to worse (right)')

    ax1.set_ylabel('Cost function (-)', color=color)
    ax1.plot(tab_perf_sorted, linestyle='dotted', marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.title("Consistency Curve", y=1.05)
    plt.savefig(directory + "/" + "ConsistencyCurve.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def Optimum_thickness_plot(parameters, Experience_results, directory):
    tab_best_solution = Experience_results.get("tab_best_solution")
    max_index = Experience_results.get("max_index")
    Mat_Stack = parameters.get("Mat_Stack")
    n_Stack = parameters.get("n_Stack")
    Th_range = parameters.get("Th_range")
    # Plot of thickness
    ep = tab_best_solution[max_index]
    if 'nb_layer' in locals() and parameters['nb_layer'] != 0:
        ep = np.delete(ep, np.s_[(parameters['nb_layer'] + len(Mat_Stack)):])

    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        vf = []
        vf = ep[len(Mat_Stack):len(ep)]
        ep = np.delete(ep, np.s_[(len(Mat_Stack)):len(ep)])
        Experience_results.update({'vf': vf})

    # del epaisseur[0]
    lower = Th_range[0]
    upper = Th_range[1]
    fig, ax = plt.subplots()
    ax.scatter(range(1, len(ep)), ep[1:])
    ax.axhline(lower, color='r')
    ax.axhline(upper, color='g')
    ax.set_xticks(range(1, len(ep)))
    ax.set_xticklabels([str(i) for i in range(1, len(ep))])
    for i, val in enumerate(ep[1:]):
        ax.annotate(str("{:.0f}".format(val)), xy=(
            i + 1, val), xytext=(i + 1.1, val + 1.1))
    plt.xlabel("Number of layers, substrate to air")
    plt.ylabel("Thickness (nm)")
    plt.title("Optimum Thickness ", y=1.05)
    plt.savefig(directory + "/" + "Optimum_Thickness_Stack.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def Optimum_refractive_index_plot(parameters, Experience_results, directory):
    tab_best_solution = Experience_results.get("tab_best_solution")
    max_index = Experience_results.get("max_index")
    Mat_Stack = parameters.get("Mat_Stack")
    n_range = parameters.get("n_range")

    if 'nb_layer' in parameters:
        # Plot of refractive index
        n_list = tab_best_solution[max_index]
        for i in range(parameters['nb_layer'] + len(Mat_Stack)-1):
            n_list = np.delete(n_list, 0)
        # del epaisseur[0]
        lower = n_range[0]
        upper = n_range[1]
        fig, ax = plt.subplots()
        ax.scatter(range(1, len(n_list)), n_list[1:])
        ax.axhline(lower, color='r')
        ax.axhline(upper, color='g')
        ax.set_xticks(range(1, len(n_list)))
        ax.set_xticklabels([str(i) for i in range(1, len(n_list))])
        # Put the labels
        for i, val in enumerate(n_list[1:]):
            ax.annotate(str("{:.2f}".format(val)), xy=(
                i + 1, val), xytext=(i+1.05, val + 0.05))
        # Fix y-axis limits : from 1 to 3 here
        # Change y-axis' scale
        ax.set_ylim((min(n_range)-0.5), (max(n_range)+0.5))
        plt.xlabel("Number of layers, substrate to air")
        plt.ylabel("Refractive Index (-)")
        plt.title("Optimum Refractive Index ", y=1.05)
        plt.savefig(directory + "/" + "Optimum_RefractiveIndex_Stack.png",
                    dpi=300, bbox_inches='tight')
        plt.show()
        return n_list


def Volumetric_parts_plot(parameters, Experience_results, directory):
    n_Stack = parameters.get("n_Stack")
    vf_range = parameters.get('vf_range')
    vf = Experience_results.get("vf")
    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        # Volumetric parts graph
        lower = vf_range[0]
        upper = vf_range[1]
        fig, ax = plt.subplots()
        ax.scatter(range(1, len(vf)), vf[1:])
        ax.axhline(lower, color='r')
        ax.axhline(upper, color='g')
        ax.set_xticks(range(1, len(vf)))
        ax.set_xticklabels([str(i) for i in range(1, len(vf))])
        # Put the labels
        for i, val in enumerate(vf[1:]):
            ax.annotate(str("{:.3f}".format(val)), xy=(
                i + 1, val), xytext=(i+1.05, val + 0.05))
        # Fix y-axis limits : from 1 to 3 here
        # ax.set_ylim((min(vf_range)), (max(vf_range))) # Change y-axis' scale
        plt.xlabel("Number of layers, substrate to air")
        plt.ylabel("Volumic Fraction (-)")
        plt.title("Volumic Fraction ", y=1.05)
        plt.savefig(directory + "/" + "Optimum_VolumicFraction.png",
                    dpi=300, bbox_inches='tight')
        plt.show()
        parameters.update({'vf_range': vf_range, })


def Stack_plot(parameters, Experience_results, directory):
    """
The goal of this function is to help th user to visualize the generated stack in a way to understand
the role of each layer easier.

Parameters
----------
parameters : dict 
    Dictionnary that contains the different parameters of the experience.
n_Stack : array 
    The real part of refractive index. 
k_Stack : array
    The complex part of refractive index.
Mat_Stack : List of strings
    List of materials.
d_Stack : List of floats
    List of thicknesses.
Wl : numpy array
    Array with the wavelengths of the experience.
vf : List of floats
    Mixing law of each materials n the stack.

Returns
-------
A colored plot with different colors depending on the refractive index of the materials, his R coefficient
if it's considered as a metal or his vf is it's a Cermet.
    """
    if 'max_index' in Experience_results:
        max_index = Experience_results.get('max_index')
        individual = Experience_results.get('tab_best_solution')[max_index]
    else:
        individual = Experience_results.get('d_Stack')
    vf = Experience_results.get("vf")
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack').copy()
    Mat_Stack_print = parameters.get('Mat_Stack_print')

    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)
    values = np.ndarray.tolist(d_Stack[0])
    d_Stack = values

    Wl = parameters["Wl"]

    fig, ax = plt.subplots()  # initializing the plot zone
    Full_Stack_Th = 0  # initializing the variable of the total Thickness of the stack
    color = []  # store the color of each rectangle of each material
    color.append('')  # to avoid the 1st place of the substrate
    n_Stack_Wl = []
    k_Stack_Wl = []
    # to keep same lengths as the other lists (we forget the substrate)
    n_Stack_Wl.append('')
    k_Stack_Wl.append('')
    Cermets = []
    Metals = []
    Dielectrics = []
    for i in range(1, len(d_Stack)):
        # finding the line corrsponding to 590nm in n_Stack
        line = int((590-Wl[0])/(Wl[1]-Wl[0]))
        # creating lists to have just the interessant values (590nm)
        n_Stack_Wl.append(n_Stack[line][i])
        k_Stack_Wl.append(k_Stack[line][i])
        if Mat_Stack_print != None and Mat_Stack_print[i] == 'X':
            Mat_Stack.append('n = ' + str(round(n_Stack_Wl[i], 3)))
        # finding cermets and puting them in a special list
        if '-' in Mat_Stack[i] and ('air' not in Mat_Stack[i] and 'Air' not in Mat_Stack[i] and 'AIR' not in Mat_Stack[i] and 'vaccum' not in Mat_Stack[i] and 'Vaccum' not in Mat_Stack[i] and 'VACCUM' not in Mat_Stack[i]):
            Cermets.append(Mat_Stack[i])
            Cermets.append(i)
            Cermets.append(vf[i])
    for i in range(1, len(d_Stack)):
        # incrementing the full thickness variable in a way to have the stack length
        Full_Stack_Th = Full_Stack_Th + d_Stack[i]
        # I'm starting to fin the color of each material with the cermets. Here, the color is purple and higher is the vf, the darker it is.
        if '-' in Mat_Stack[i] and ('air' not in Mat_Stack[i] and 'Air' not in Mat_Stack[i] and 'AIR' not in Mat_Stack[i] and 'vaccum' not in Mat_Stack[i] and 'Vaccum' not in Mat_Stack[i] and 'VACCUM' not in Mat_Stack[i]):
            r = 1-0.67*vf[i]
            g = 0.75-0.75*vf[i]
            # defining a type of "range" for the color betwin a low vf and a high vf
            b = 0.76-0.22*vf[i]
            # applying the rgb color code  corresponding to this material in the list
            color.append((r, g, b))
        else:  # the material is not a Cermet
            # calculating the R of the material : if R>0.5 we consider him as a metal, else as a dielectric.
            R = ((n_Stack_Wl[i]-1)**2+k_Stack_Wl[i]**2) / \
                ((n_Stack_Wl[i]+1)**2+k_Stack_Wl[i]**2)
            if (R > 0.5):  # metal
                # as we did for cermets, the metal will have a grey color (as darker as the R is high)
                grey = 0.9-((R-0.5)*0.704/0.5)
                color.append((grey, grey, grey))
                Metals.append(Mat_Stack[i])  # store metals in a special list
            else:  # the material is a dielectric
                Dielectrics.append(Mat_Stack[i])  # stoe them in a special list
                # we follow the same reflexion but with more colors, so we define some key index values and make some "color range" between from blue to red and moving on the chroma circle between.
                if (n_Stack_Wl[i] == 1):
                    color.append((1, 1, 1))
                elif (n_Stack_Wl[i] <= 1.2):  # moving from white to light blue
                    r = 1-((n_Stack_Wl[i]-1)/0.2*38/255)
                    g = 1-((n_Stack_Wl[i]-1)/0.2*28/255)
                    b = 1-((n_Stack_Wl[i]-1)/0.2*12/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 1.5):  # moving from light blue to blue
                    r = 217/255-((n_Stack_Wl[i]-1.2)/0.3*74/255)
                    g = 227/255-((n_Stack_Wl[i]-1.2)/0.3*57/255)
                    b = 243/255-((n_Stack_Wl[i]-1.2)/0.3*23/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 1.7):  # moving from blue to dark blue
                    r = 143/255-((n_Stack_Wl[i]-1.5)/0.2*75/255)
                    g = 170/255-((n_Stack_Wl[i]-1.5)/0.2*56/255)
                    b = 220/255-((n_Stack_Wl[i]-1.5)/0.2*24/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 2):  # moving from dark blue to light green
                    r = 68/255+((n_Stack_Wl[i]-1.7)/0.3*129/255)
                    g = 114/255+((n_Stack_Wl[i]-1.7)/0.3*110/255)
                    b = 196/255-((n_Stack_Wl[i]-1.7)/0.3*16/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 2.5):  # moving from light green to dark green
                    r = 197/255-((n_Stack_Wl[i]-2)/0.5*66/255)
                    g = 224/255-((n_Stack_Wl[i]-2)/0.5*36/255)
                    b = 180/255-((n_Stack_Wl[i]-2)/0.5*88/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 3.25):  # moving from dark green to light orange
                    r = 131/255+((n_Stack_Wl[i]-2.5)/0.75*106/255)
                    g = 188/255-((n_Stack_Wl[i]-2.5)/0.75*59/255)
                    b = 92/255-((n_Stack_Wl[i]-2.5)/0.75*37/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 4):  # moving from light orange to orange
                    r = 237/255-((n_Stack_Wl[i]-3.25)/0.75*40/255)
                    g = 129/255-((n_Stack_Wl[i]-3.25)/0.75*39/255)
                    b = 55/255-((n_Stack_Wl[i]-3.25)/0.75*38/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 4.5):  # moving from orange to ligth red
                    r = 197/255+((n_Stack_Wl[i]-3.5)/0.5*58/255)
                    g = 90/255-((n_Stack_Wl[i]-3.5)/0.5*90/255)
                    b = 17/255-((n_Stack_Wl[i]-3.5)/0.5*17/255)
                    color.append((r, g, b))
                elif (n_Stack_Wl[i] <= 4.8):  # moving from red to red
                    r = 1-((n_Stack_Wl[i]-3.8)/0.3*63/255)
                    g = 0
                    b = 0
                    color.append((r, g, b))
                else:  # moving from red to dark red
                    r = 192
                    g = 0
                    b = 0
                    color.append((r, g, b))
    # creating scale depending on the total thickness
    if (Full_Stack_Th < 2000 and Full_Stack_Th != 0):  # nanometers
        # New_d_Stack is d_Stack converted in the scale unit (nothing changes here)
        New_d_Stack = d_Stack
        ax.set_ylim(0, Full_Stack_Th)
        ax.set_aspect('equal')
        ax.set_xlim(0, Full_Stack_Th/2)
        plt.ylabel("Thickness (nm)")
    elif (Full_Stack_Th < 2000000 and Full_Stack_Th != 0):  # micrometers
        Full_Stack_Th = Full_Stack_Th/1000  # puting it into micrometers
        New_d_Stack = []
        # here, New_D_Stack takes d_Stack values /1000 (in micrometers) in a way to fit the plots with the scale
        for i in range(len(d_Stack)):
            New_d_Stack.append(d_Stack[i]/1000)
        ax.set_ylim(0, Full_Stack_Th)
        ax.set_aspect('equal')
        ax.set_xlim(0, Full_Stack_Th/2)
        plt.ylabel("Thickness (μm)")
    else:  # milimeters
        if (Full_Stack_Th != 0):
            Full_Stack_Th = Full_Stack_Th/1000000  # milimeters
            New_d_Stack = []
            for i in range(len(d_Stack)):
                New_d_Stack.append(d_Stack[i]/1000000)  # milimeters
            ax.set_ylim(0, Full_Stack_Th)
            ax.set_aspect('equal')
            ax.set_xlim(0, Full_Stack_Th/2)
            plt.ylabel("Thickness (mm)")
    Current_Th = 0  # to start ploting in the right order, we start from the bottom by inisializing the cuurent thickness to 0
    for i in range(1, len(New_d_Stack)):  # for each material of the stack
        # defining a rectangle of the thickness of the current material and with the color defined earlier
        rectangle = Rectangle((0, Current_Th), Full_Stack_Th/2,
                              New_d_Stack[i], color=color[i], alpha=1)
        ax.add_patch(rectangle)  # adding the rectangle to the plot zone
        # if the material is a cermet, we want to show it by addind circles into the rectangle of the concerned cermet
        if '-' in Mat_Stack[i]:
            if 'air' in Mat_Stack[i] or 'Air' in Mat_Stack[i] or 'AIR' in Mat_Stack[i] or 'vaccum' in Mat_Stack[i] or 'Vaccum' in Mat_Stack[i] or 'VACCUM' in Mat_Stack[i]:
                circle_color = 'white'  # finding his color
            else:
                circle_color = 'black'
            nb_circle = 0  # the number of circle is the number of lines of circles into the rectangle depending of the thickness of the rectangle
            if New_d_Stack[i] < 5*Full_Stack_Th/137:  # longer is the material's layer,
                nb_circle = 1
                Radius_div = New_d_Stack[i]/4
            # more line of circles are presents
            elif New_d_Stack[i] < 10*Full_Stack_Th/137:
                nb_circle = 2
                Radius_div = New_d_Stack[i]/7
            elif New_d_Stack[i] < 15*Full_Stack_Th/137:
                nb_circle = 3
                Radius_div = New_d_Stack[i]/10
            elif New_d_Stack[i] < 20*Full_Stack_Th/137:
                nb_circle = 4
                Radius_div = New_d_Stack[i]/13
            elif New_d_Stack[i] < 25*Full_Stack_Th/137:
                nb_circle = 5
                Radius_div = New_d_Stack[i]/16
            elif New_d_Stack[i] < 30*Full_Stack_Th/137:
                nb_circle = 6
                Radius_div = New_d_Stack[i]/19
            elif New_d_Stack[i] < 35*Full_Stack_Th/137:
                nb_circle = 7
                Radius_div = New_d_Stack[i]/22
            elif New_d_Stack[i] < 40*Full_Stack_Th/137:
                nb_circle = 8
                Radius_div = New_d_Stack[i]/25
            elif New_d_Stack[i] < 45*Full_Stack_Th/137:
                nb_circle = 9
                Radius_div = New_d_Stack[i]/28
            elif New_d_Stack[i] < 50*Full_Stack_Th/137:
                nb_circle = 10
                Radius_div = New_d_Stack[i]/31
            elif New_d_Stack[i] < 55*Full_Stack_Th/137:
                nb_circle = 11
                Radius_div = New_d_Stack[i]/34
            elif New_d_Stack[i] < 60*Full_Stack_Th/137:
                nb_circle = 12
                Radius_div = New_d_Stack[i]/37
            elif New_d_Stack[i] < 65*Full_Stack_Th/137:
                nb_circle = 13
                Radius_div = New_d_Stack[i]/40
            elif New_d_Stack[i] < 70*Full_Stack_Th/137:
                nb_circle = 14
                Radius_div = New_d_Stack[i]/43
            elif New_d_Stack[i] < 75*Full_Stack_Th/137:
                nb_circle = 15
                Radius_div = New_d_Stack[i]/46
            elif New_d_Stack[i] < 80*Full_Stack_Th/137:
                nb_circle = 16
                Radius_div = New_d_Stack[i]/49
            elif New_d_Stack[i] < 85*Full_Stack_Th/137:
                nb_circle = 17
                Radius_div = New_d_Stack[i]/52
            elif New_d_Stack[i] < 90*Full_Stack_Th/137:
                nb_circle = 18
                Radius_div = New_d_Stack[i]/55
            elif New_d_Stack[i] < 95*Full_Stack_Th/137:
                nb_circle = 19
                Radius_div = New_d_Stack[i]/58
            elif New_d_Stack[i] < 100*Full_Stack_Th/137:
                nb_circle = 20
                Radius_div = New_d_Stack[i]/61
            elif New_d_Stack[i] < 105*Full_Stack_Th/137:
                nb_circle = 21
                Radius_div = New_d_Stack[i]/64
            elif New_d_Stack[i] < 110*Full_Stack_Th/137:
                nb_circle = 22
                Radius_div = New_d_Stack[i]/67
            elif New_d_Stack[i] < 115*Full_Stack_Th/137:
                nb_circle = 23
                Radius_div = New_d_Stack[i]/70
            elif New_d_Stack[i] < 120*Full_Stack_Th/137:
                nb_circle = 24
                Radius_div = New_d_Stack[i]/73
            elif New_d_Stack[i] < 125*Full_Stack_Th/137:
                nb_circle = 25
                Radius_div = New_d_Stack[i]/76
            elif New_d_Stack[i] < 130*Full_Stack_Th/137:
                nb_circle = 26
                Radius_div = New_d_Stack[i]/79
            elif New_d_Stack[i] < 135*Full_Stack_Th/137:
                nb_circle = 27
                Radius_div = New_d_Stack[i]/82
            else:
                nb_circle = 28
                Radius_div = New_d_Stack[i]/85
            first_center = (
                Full_Stack_Th/2-((int((Full_Stack_Th/2)/(3*Radius_div))-1)*3*Radius_div))/2
            for k in range(nb_circle):  # plotting the number of lines
                # plotting on all the rectangle width
                for j in range(0, int((Full_Stack_Th/2)/(3*Radius_div))):
                    if k % 2 == 0:  # adding a small feature that impair line have not same margin that pair ones
                        if nb_circle == 1:
                            # radius already defined
                            circle = Circle(
                                (first_center+j*3*Radius_div, Current_Th+New_d_Stack[i]/2), Radius_div, color=circle_color)
                        else:
                            circle = Circle((first_center+j*3*Radius_div, Current_Th +
                                            New_d_Stack[i] - 2*Radius_div - k*3*Radius_div), Radius_div, color=circle_color)
                    else:
                        if (j != int((Full_Stack_Th/2)/(3*Radius_div))-1):
                            circle = Circle((first_center+1.5*Radius_div+j*3*Radius_div, Current_Th +
                                            New_d_Stack[i] - 2*Radius_div - k*3*Radius_div), Radius_div, color=circle_color)
                    ax.add_patch(circle)  # adding cirlce to the plot
        # the current thickness where we are plotting is incrementing before going to the next material
        Current_Th = Current_Th+New_d_Stack[i]
    legend = []  # creating lists to create the legend
    legend_color = []
    M_multiples = []
    for i in range(len(Metals)):
        for j in range(len(Mat_Stack)):
            # to avoid a legend where there are multiple time the same material in a case that we have a repetition of one or more materials in the stack.
            if (Metals[i] == Mat_Stack[j] and Mat_Stack[j] not in M_multiples and color[j] != ''):
                legend.append(Mat_Stack[j])
                legend_color.append(color[j])
                M_multiples.append(Mat_Stack[j])
                break
    if len(Cermets) > 3 and len(Cermets) != 0:
        for i in range(len(Cermets)):
            present = False
            if isinstance(Cermets[i], str):
                if i != 0 and i != 3:
                    for j in range(i-2):
                        if Cermets[i] == Cermets[j] and Cermets[i+2] == Cermets[j+2]:
                            present = True
                elif i == 3:
                    if Cermets[i] == Cermets[0] and Cermets[i+2] == Cermets[2]:
                        present = True
                if present == False:
                    text = Cermets[i] + ', vf = ' + str(round(Cermets[i+2], 3))
                    legend.append(text)
                    legend_color.append(color[Cermets[i+1]])
    else:
        if len(Cermets) != 0:
            text = Cermets[0] + ', vf = ' + str(round(Cermets[2], 3))
            legend.append(text)
            legend_color.append(color[Cermets[1]])
    Die_multiples = []
    for i in range(len(Dielectrics)):
        for j in range(len(Mat_Stack)):
            present = False
            if (Dielectrics[i] == Mat_Stack[j] and color[j] != ''):
                if '-' in Mat_Stack[j]:
                    for k in range(len(Die_multiples)-1):
                        if Die_multiples[k] == Mat_Stack[j] and vf[j] == Die_multiples[k+1]:
                            present = True
                            break
                    if present == False:
                        legend.append(
                            Mat_Stack[j] + ', vf = ' + str(round(vf[j], 3)))
                        legend_color.append(color[j])
                        Die_multiples.append(Mat_Stack[j])
                        Die_multiples.append(vf[j])
                        break
                elif Mat_Stack[j] not in Die_multiples:
                    legend.append(Mat_Stack[j])
                    legend_color.append(color[j])
                    Die_multiples.append(Mat_Stack[j])
                    if vf is not None:
                        Die_multiples.append(vf[j])
                    else:
                        Die_multiples.append(1)
                    break
    legend_rectangles = []
    for i in range(len(legend)):
        legend_rectangles.append(mpatches.Rectangle(
            (0, 0), 20, 10, facecolor=legend_color[i], edgecolor='black', label=legend[i]))  # plotting rectangles for the legend
    legend = plt.legend(handles=legend_rectangles, loc='lower left', bbox_to_anchor=(
        1, 0), handleheight=2, handlelength=2.2, ncol=math.ceil(len(Mat_Stack)/14))
    plt.title("Thin layers thicknesses", y=1.05)  # adding title
    plt.gca().xaxis.set_ticks([])
    plt.savefig(directory + "/" + "Stack_plot.png", dpi=300,
                bbox_inches='tight')  # saving figure
    plt.show()


def Explain_results(parameters, Experience_results):
    # Go to find the best in all the result
    if parameters["name_selection"] == "selection_max":
        max_value = max(Experience_results["tab_perf"])  # finds the maximum
    if parameters["name_selection"] == "selection_min":
        max_value = min(Experience_results["tab_perf"])  # finds the minimum
    max_index = Experience_results["tab_perf"].index(
        max_value)  # finds the maximum's index (where he is)

    # I've just found my maximum, out of all my runs. It's the best of the best! Congratulations!

    Sol_Spec = parameters.get("Sol_Spec")
    Sol_Spec_int = trapezoid(Sol_Spec, parameters["Wl"])

    # Calculation of Rs, Ts, As du max (solar performances)
    Rs, Ts, As = evaluate_RTA_s(
        Experience_results["tab_best_solution"][max_index], parameters)
    # Calculation le R, T, A (Reflectivity and other, for plot a curve)
    R, T, A = RTA_curve_inco(
        Experience_results["tab_best_solution"][max_index], parameters)
    # I set at least one value other than 0 to avoid errors when calculating the integral.

    if all(value == 0 for value in T):
        T[0] = 10**-301
    if all(value == 0 for value in R):
        R[0] = 10**-301
    if all(value == 0 for value in A):
        A[0] = 10**-301

    # Upstream
    # Opening the solar spectrum
    # Reminder: GT spectrum => Global spectrum, i.e., the spectrum of the sun + reflection from the environment
    # GT Spectrum = Direct Spectrum (DC) + Diffuse Spectrum
    # This is the spectrum seen by the surface
    Wl_Sol_1, Sol_Spec_1, name_Sol_Spec_1 = open_SolSpec(
        'Materials/SolSpec.txt', 'GT')
    Sol_Spec_1 = np.interp(parameters["Wl"], Wl_Sol_1, Sol_Spec_1)
    # Integration of the solar spectrum, raw in W/m2
    Sol_Spec_int_1 = trapezoid(Sol_Spec_1, parameters["Wl"])
    # Writing the solar spectrum modified by the treatment's transmittance
    Sol_Spec_mod_T = T * Sol_Spec_1
    # integration of the T-modified solar spectrum, result in W/m2
    Sol_Spec_mod_T_int = trapezoid(Sol_Spec_mod_T, parameters["Wl"])
    # Integration of the solar spectrum modified by the treatment's reflectance, according to the spectrum
    Sol_Spec_mod_R = R * Sol_Spec_1
    # integration of the R-modified solar spectrum, result in W/m2
    Sol_Spec_mod_R_int = trapezoid(Sol_Spec_mod_R, parameters["Wl"])
    # Integration of the solar spectrum modified by the treatment's absorbance, according to the spectrum
    Sol_Spec_mod_A = A * Sol_Spec_1
    # integration of the A-modified solar spectrum, result in W/m2
    Sol_Spec_mod_A_int = trapezoid(Sol_Spec_mod_A, parameters["Wl"])
    # Calculation of the upstream solar efficiency, for example, the efficiency of the PV solar cell with the modified spectrum
    Ps_amont = SolarProperties(
        parameters["Wl"], parameters["Signal_PV"], Sol_Spec_mod_T)
    # Calculation of the upstream treatment solar efficiency with an unmodified spectrum
    Ps_amont_ref = SolarProperties(
        parameters["Wl"], parameters["Signal_PV"], Sol_Spec_1)
    # Calculation of the integration of the useful upstream solar spectrum
    Sol_Spec_mod_amont = Sol_Spec_1 * parameters["Signal_PV"]
    Sol_Spec_mod_amont_int = trapezoid(Sol_Spec_mod_amont, parameters["Wl"])
    # Calculation of the integration of the useful upstream solar spectrum with T-modified spectrum
    Sol_Spec_mod_T_amont = Sol_Spec_mod_T * parameters["Signal_PV"]
    Sol_Spec_mod_T_amont_int = trapezoid(
        Sol_Spec_mod_T_amont, parameters["Wl"])

    # Downstream
    # Opening the solar spectrum, which may be different from the first one depending on the cases
    # Reminder: DC spectrum => Direct spectrum, i.e., only the spectrum of the sun, concentrable by an optical system
    Wl_Sol_2, Sol_Spec_2, name_Sol_Spec_2 = open_SolSpec(
        'Materials/SolSpec.txt', 'DC')
    Sol_Spec_2 = np.interp(parameters["Wl"], Wl_Sol_2, Sol_Spec_2)
    # Integration of the solar spectrum, raw in W/m2
    Sol_Spec_int_2 = trapezoid(Sol_Spec_2, parameters["Wl"])
    # Writing the solar spectrum modified by the treatment's reflectance
    Sol_Spec_mod_R_2 = R * Sol_Spec_2
    # integration of the R-modified solar spectrum, result in W/m2
    Sol_Spec_mod_R_int_2 = trapezoid(Sol_Spec_mod_R_2, parameters["Wl"])
    # Calculation of the downstream solar efficiency, for example, the efficiency of the thermal absorber
    Ps_aval = SolarProperties(
        parameters["Wl"], parameters["Signal_Th"], Sol_Spec_mod_R_2)
    # Calculation of the downstream treatment solar efficiency with an unmodified spectrum
    Ps_aval_ref = SolarProperties(
        parameters["Wl"], parameters["Signal_Th"], Sol_Spec_2)
    # Calculation of the integration of the useful downstream solar spectrum
    Sol_Spec_mod_aval = Sol_Spec_2 * parameters["Signal_Th"]
    Sol_Spec_mod_aval_int = trapezoid(Sol_Spec_mod_aval, parameters["Wl"])
    # Calculation of the integration of the useful downstream solar spectrum
    Sol_Spec_mod_R_aval = Sol_Spec_mod_R_2 * parameters["Signal_Th"]
    Sol_Spec_mod_R_aval_int = trapezoid(Sol_Spec_mod_R_aval, parameters["Wl"])
    # Update the results
    Experience_results.update({
        "Rs": Rs,
        "Ts": Ts,
        "As": As,
        "R": R,
        "T": T,
        "A": A,
        "Sol_Spec_int": Sol_Spec_int,
        "Sol_Spec_int_1": Sol_Spec_int_1,
        "Sol_Spec_mod_T": Sol_Spec_mod_T,
        "Sol_Spec_mod_T_int": Sol_Spec_mod_T_int,
        "Sol_Spec_mod_R": Sol_Spec_mod_R,
        "Sol_Spec_mod_R_int": Sol_Spec_mod_R_int,
        "Sol_Spec_mod_A": Sol_Spec_mod_A,
        "Sol_Spec_mod_A_int": Sol_Spec_mod_A_int,
        "Ps_amont": Ps_amont,
        "Ps_amont_ref": Ps_amont_ref,
        "Sol_Spec_mod_amont": Sol_Spec_mod_amont,
        "Sol_Spec_mod_amont_int": Sol_Spec_mod_amont_int,
        "Sol_Spec_mod_T_amont": Sol_Spec_mod_T_amont,
        "Sol_Spec_mod_T_amont_int": Sol_Spec_mod_T_amont_int,
        "Wl_Sol_2": Wl_Sol_2,
        "Sol_Spec_2": Sol_Spec_2,
        "name_Sol_Spec_2": name_Sol_Spec_2,
        "Sol_Spec_int_2": Sol_Spec_int_2,
        "Sol_Spec_mod_R_2": Sol_Spec_mod_R_2,
        "Sol_Spec_mod_R_int_2": Sol_Spec_mod_R_int_2,
        "Ps_aval": Ps_aval,
        "Ps_aval_ref": Ps_aval_ref,
        "Sol_Spec_mod_aval": Sol_Spec_mod_aval,
        "Sol_Spec_mod_aval_int": Sol_Spec_mod_aval_int,
        "Sol_Spec_mod_R_aval": Sol_Spec_mod_R_aval,
        "Sol_Spec_mod_R_aval_int": Sol_Spec_mod_R_aval_int,
        "max_index": max_index
    })


def Explain_results_fit(parameters, Experience_results):
    # Go to find the best in all the result
    if parameters["name_selection"] == "selection_max":
        max_value = max(Experience_results["tab_perf"])  # finds the maximum
    if parameters["name_selection"] == "selection_min":
        max_value = min(Experience_results["tab_perf"])  # finds the minimum
    max_index = Experience_results["tab_perf"].index(
        max_value)  # finds the maximum's index (where he is)

    # I've just found my maximum, out of all my runs. It's the best of the best! Congratulations!

    Sol_Spec = parameters.get("Sol_Spec")
    Sol_Spec_int = trapezoid(Sol_Spec, parameters["Wl"])

    # Calculation of Rs, Ts, As du max (solar performances)
    Rs, Ts, As = evaluate_RTA_s(
        Experience_results["tab_best_solution"][max_index], parameters)
    # Calculation le R, T, A (Reflectivity and other, for plot a curve)
    R, T, A = RTA_curve(
        Experience_results["tab_best_solution"][max_index], parameters)
    # I set at least one value other than 0 to avoid errors when calculating the integral.

    if all(value == 0 for value in T):
        T[0] = 10**-301
    if all(value == 0 for value in R):
        R[0] = 10**-301
    if all(value == 0 for value in A):
        A[0] = 10**-301

    # Update the results
    Experience_results.update({
        "Rs": Rs,
        "Ts": Ts,
        "As": As,
        "R": R,
        "T": T,
        "A": A,
        "Sol_Spec_int": Sol_Spec_int,
        "Sol_Spec_mod_R": R * Sol_Spec,
        "Sol_Spec_mod_T": T * Sol_Spec,
        "max_index": max_index
    })


def Convergences_txt(parameters, Experience_results, directory):
    # My goal is to pick up some values (with equidistant_value) of the alogorithm convergence (stored in tab_dev)
    tab_dev = Experience_results["tab_dev"]
    algo = parameters["algo"]
    selection = parameters["selection"]
    tab_perf_dev = []
    for i in range(len(tab_dev)):
        # I check all over tab_dev
        data_dev = []
        data_dev = tab_dev[i]
        # I take some values (initially 5) equidistants
        data_dev = valeurs_equidistantes(data_dev, 5)
        # I invert my list because initially, the 1st value is the one at the begining of the problem, and the last one
        # is the last cost function calculated, so normally my best for this run
        data_dev.reverse()
        # If I've launched DEvol in selection_max mode, the real values are 1 - fcout
        if algo.__name__ == "DEvol":
            if selection.__name__ == "selection_max":
                data_dev = [1 - x for x in data_dev]
        tab_perf_dev.append(data_dev)
    # I pass the list of list in an array
    tab_perf_dev = np.array(tab_perf_dev, dtype=float)
    # Writing of tab_perf_dev in a txt file
    np.savetxt(directory + '/Convergence.txt',
               tab_perf_dev, fmt='%.18e', delimiter='  ')

    tab_perf_dev = []
    for i in range(len(tab_dev)):
        # I check all over tab_dev
        data_dev = []
        data_dev = tab_dev[i]
        # I take some values (initially 25) equidistants
        data_dev = valeurs_equidistantes(data_dev, 25)
        # I invert my list because initially, the 1st value is the one at the begining of the problem, and the last one
        # is the last cost function calculated, so normally my best for this run
        data_dev.reverse()
        # If I've launched DEvol in selection_max mode, the real values are 1 - fcout
        if algo.__name__ == "DEvol":
            if selection.__name__ == "selection_max":
                data_dev = [1 - x for x in data_dev]
        tab_perf_dev.append(data_dev)
    # I pass the list of list in an array
    tab_perf_dev = np.array(tab_perf_dev, dtype=float)
    # Writing of tab_perf_dev in a txt file
    np.savetxt(directory + '/Convergence_25.txt',
               tab_perf_dev, fmt='%.18e', delimiter='  ')


def Generate_txt(parameters, Experience_results, directory):
    language = Experience_results.get('language')
    if language != 'en' and language != 'fr':
        language = 'en'
    tab_perf = Experience_results["tab_perf"]
    tab_seed = Experience_results["tab_seed"]
    tab_temps = Experience_results["tab_temps"]
    tab_best_solution = Experience_results["tab_best_solution"]
    tab_dev = Experience_results["tab_dev"]
    Wl = parameters["Wl"]
    Sol_Spec_mod_R = Experience_results["Sol_Spec_mod_R"]
    Sol_Spec_mod_T = Experience_results["Sol_Spec_mod_T"]
    R = Experience_results["R"]
    T = Experience_results["T"]
    A = Experience_results["A"]

    filename = directory + "/performance.txt"
    with open(filename, "w") as file:
        for value in tab_perf:
            file.write(str(value) + "\n")

    filename = directory + "/seed.txt"
    with open(filename, "w") as file:
        for value in tab_seed:
            file.write(str(value) + "\n")

    filename = directory + "/time.txt"
    with open(filename, "w") as file:
        for value in tab_temps:
            file.write(str(value) + "\n")

    filename = directory + "/Stacks.txt"
    with open(filename, "w") as file:
        for value in tab_best_solution:
            np.savetxt(file, value.reshape(1, -1), fmt='%.18e', delimiter=' ')

    filename = directory + "/Convergence.txt"
    with open(filename, "w") as file:
        for value in tab_dev:
            np.savetxt(file, value.reshape(1, -1), fmt='%.18e', delimiter=' ')

    filename = directory + "/Sol_Spec_mod_R.txt"
    with open(filename, "w") as file:
        for i in range(len(Wl)):
            file.write(str(Wl[i]) + "\t" + str(Sol_Spec_mod_R[i]) + "\n")

    filename = directory + "/Sol_Spec_mod_T.txt"
    with open(filename, "w") as file:
        for i in range(len(Wl)):
            file.write(str(Wl[i]) + "\t" + str(Sol_Spec_mod_T[i]) + "\n")

    if language == "fr":
        print("Les résultats ont été écrits dans le dossier")
    if language == "en":
        print("The results were written in the folders")

    filename = directory + "/RTA.txt"
    with open(filename, "w") as file:
        for i in range(len(A)):
            file.write(str(Wl[i]) + "\t" + str(R[i]) +
                       "\t" + str(T[i]) + "\t" + str(A[i]) + "\n")

    if language == "fr":
        print("Les données RTA du meilleur empillement ont été écrites dans cet ordre")
    if language == "en":
        print("The RTA data for the best stack were written in the folder")


def Generate_perf_rh_txt(parameters, Experience_results, directory):
    """
Write the heliothermal efficiency, the solar absoptance and the thermal emissivity
of spectral selective coating for thermal absorber into a texte files.
"""
    tab_best_solution = Experience_results.get("tab_best_solution")
    Wl = parameters.get("Wl")
    Ang = parameters.get('Ang')
    C = parameters.get('C')
    T_air = parameters.get('T_air')
    T_abs = parameters.get('T_abs')
    Mat_Stack = parameters.get('Mat_Stack')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')

    # Integration of solar spectrum, raw en W/m2
    I = trapezoid(Sol_Spec, Wl)

    # Created list
    perf_rh = []

    for i in range(len(tab_best_solution)):
        # Calculate the R, T, A of each solution
        R, T, A = RTA_curve(
            Experience_results["tab_best_solution"][i], parameters)

        # Calculation of the solar absorption
        A_s = 0
        A_s = SolarProperties(Wl, A, Sol_Spec)
        # Calculation of the black body
        BB_shape = BB(T_abs, Wl)
        # calculation of the emittance of the surface
        E_BB_T_abs = E_BB(Wl, A, BB_shape)
        # Calculation of the solar thermal yield. Argument of the function helio_th(A_s, E_BB, T_stack, T_air, C, I,  r_Opt = 0.7, FFabs=1):
        rH = helio_th(A_s, E_BB_T_abs, T_abs, T_air, C, I,  r_Opt=0.7, FFabs=1)

        # Add value in perf_rh as list
        perf_rh.append([rH, A_s, E_BB_T_abs])

    # Convert perf_rh in a numpy array
    perf_rh_np = np.array(perf_rh)

    filename = directory + "/performance_rh.txt"
    # Open the files with writing write
    with open(filename, "w") as file:
        # Parcourir chaque ligne du tableau numpy
        for row in perf_rh_np:
            # Convertir la ligne en chaîne de caractères avec un format lisible
            file.write("\t".join(map(str, row)) + "\n")

    print("The performances of selective coatings were written")


def Optimization_txt(parameters, Experience_results, directory):
    language = Experience_results.get('language')
    if language != 'en' and language != 'fr':
        language = 'en'
    launch_time = Experience_results.get("launch_time")
    Comment = Experience_results.get("Comment")
    algo = parameters.get("algo")
    evaluate = parameters.get("evaluate")
    selection = parameters.get("selection")
    mutation_DE = parameters.get("mutation_DE")

    name_Sol_Spec = parameters.get("name_Sol_Spec")
    Sol_Spec = parameters.get("Sol_Spec")
    Wl = parameters.get("Wl")

    Mat_Stack_print = parameters.get("Mat_Stack_print")
    nb_total_layer = parameters.get("nb_total_layer")
    Th_Substrate = parameters.get("Th_Substrate")
    Th_range = parameters.get("Th_range")
    n_range = parameters.get("n_range")
    Ang = parameters.get("Ang")
    C = parameters.get("C")
    T_abs = parameters.get("T_abs")
    Lambda_cut_1 = parameters.get("Lambda_cut_1")
    Lambda_cut_2 = parameters.get("Lambda_cut_2")
    T_air = parameters.get("T_air")
    poids_PV = parameters.get('poids_PV')
    pop_size = parameters.get("pop_size")
    crossover_rate = parameters.get("crossover_rate")
    evaluate_rate = parameters.get("evaluate_rate")
    mutation_rate = parameters.get("mutation_rate")
    f1 = parameters.get("f1")
    f2 = parameters.get("f2")
    mutation_delta = parameters.get("mutation_delta")
    precision_AlgoG = parameters.get("precision_AlgoG")
    nb_generation = parameters.get("nb_generation")
    nb_run = Experience_results.get('nb_run')
    cpu_used = parameters.get("cpu_used")
    time_real = parameters.get("time_real")
    tab_temps = Experience_results.get("tab_temps")
    seed = parameters.get("seed")

    filename = directory + "/Optimization.txt"
    script_name = os.path.basename(__file__)
    if language == "fr":
        with open(filename, "w") as file:
            file.write("Le nom du fichier est : " + str(script_name) + "\n")
            file.write("Heure de lancement " + str(launch_time) + "\n")
            file.write(str(Comment) + "\n")
            file.write("_____________________________________________" + "\n")
            file.write("Le nom de la fonction d'optimisation est : " +
                       str(algo.__name__) + "\n")
            file.write("Le nom de la fonction d'évaluation est : " +
                       str(evaluate.__name__) + "\n")
            file.write("Le nom de la fonction de sélection est : " +
                       str(selection.__name__) + "\n")
            if mutation_DE is not None:
                file.write(
                    "Si optimisation par DE, la mutation est : " + mutation_DE + "\n")
            file.write("\n")
            file.write(
                "L'emplacement et le nom du spectre solaire est :" + str(name_Sol_Spec) + "\n")
            file.write("La valeur d'irradiance : " +
                       str("{:.1f}".format(trapezoid(Sol_Spec, Wl))) + " W/m²" + "\n")
            file.write("\n")
            file.write("Nom du dossier :\t" + str(directory) + "\n")
            file.write("Matériaux de l'empillement\t" +
                       str(Mat_Stack_print) + "\n")
            file.write("Le nombre de couche minces est \t" +
                       str(nb_total_layer) + "\n")
            file.write("Domaine des longueurs d'ondes \t" + str(min(Wl)) + " nm à " +
                       str(max(Wl)) + " nm, pas de " + str(Wl[1]-Wl[0]) + " nm" + "\n")
            file.write("Epaisseur du substrat, en nm \t" +
                       str(Th_Substrate) + "\n")
            file.write("Plage des épaisseur des couches minces\t" +
                       str(Th_range[0]) + " à " + str(Th_range[1]) + " nm" + "\n")
            file.write("Plage des indices des couches minces\t" +
                       str(n_range[0]) + " à " + str(n_range[1]) + "\n")
            file.write("Angle d'incidence sur le stack\t" +
                       str(Ang) + "°" + "\n")
            file.write("Le taux de concentration est\t" + str(C) + "\n")
            file.write("La température de l'air est\t" +
                       str(T_air) + " K" + "\n")
            file.write("La température de l'absorbeur' est\t" +
                       str(T_abs) + " K" + "\n")
            if evaluate.__name__ == "evaluate_low_e" or evaluate.__name__ == "evaluate_RTR":
                file.write(
                    "Pour les profils d'optimisaiton low-e et RTR " + "\n")
                file.write("La longueur d'onde de coupure UV est \t" +
                           str(Lambda_cut_1) + " nm" + "\n")
                file.write("La longueur d'onde de coupure IR est \t" +
                           str(Lambda_cut_2) + " nm" + "\n")
            if evaluate.__name__ == "evaluate_netW_PV_CSP":
                file.write(
                    "Pour les profils d'optimisaiton evaluate_netW_PV_CSP" + "\n")
                file.write("Le coût fictif du PV est \t" +
                           str(poids_PV) + "\n")
            file.write("Taille de la population\t" + str(pop_size) + "\n")
            file.write("Taux de crossover\t" + str(crossover_rate) + "\n")
            file.write("Taux d'évaluation\t" + str(evaluate_rate) + "\n")
            file.write("Taux de mutation\t" + str(mutation_rate) + "\n")
            file.write("Valeurs de f1 et f2\t" +
                       str(f1) + " & " + str(f2) + "\n")
            file.write("Etendue de la mutation\t" + str(mutation_delta) + "\n")
            file.write("Precision de l'algo en auto\t" +
                       str(precision_AlgoG) + "\n")
            file.write("Nombre de génération\t" + str(nb_generation) + "\n")
            file.write("Nb de Lancement\t" + str(nb_run) + "\n")
            file.write("Nb de processeur disponible\t" +
                       str(cpu_count()) + "\n")
            file.write("Nb de processeur utilisé\t" + str(cpu_used) + "\n")
            file.write("Le temps réel d'éxécution (en s) total est de :\t" +
                       str("{:.2f}".format(time_real)) + "\n")
            file.write("La somme du temps de calcul (en s) processeur est de :\t" +
                       str("{:.2f}".format(sum(tab_temps)) + "\n"))
            file.write("La valeur du seed est: " + str(seed) + "\n")

        print("Les noms et valeurs des variables de la simulation ont été écrites")
    if language == "en":
        with open(filename, "w") as file:
            file.write("The filename is: " + str(script_name) + "\n")
            file.write("Launch time: " + str(launch_time) + "\n")
            file.write(str(Comment) + "\n")
            file.write("_____________________________________________" + "\n")
            file.write("The name of the optimization function is: " +
                       str(algo.__name__) + "\n")
            file.write("The name of the evaluation function is: " +
                       str(evaluate.__name__) + "\n")
            file.write("The name of the selection function is: " +
                       str(selection.__name__) + "\n")
            if mutation_DE is not None:
                file.write(
                    "If optimizing with DE, the mutation is: " + mutation_DE + "\n")
            file.write("\n")
            file.write(
                "The location and name of the solar spectrum is: " + str(name_Sol_Spec) + "\n")
            file.write("The irradiance value: " +
                       str("{:.1f}".format(trapezoid(Sol_Spec, Wl))) + " W/m²" + "\n")
            file.write("\n")
            file.write("Folder name: " + str(directory) + "\n")
            file.write("Materials in the stack: " +
                       str(Mat_Stack_print) + "\n")
            file.write("The number of thin layers: " +
                       str(nb_total_layer) + "\n")
            file.write("Wavelength range: " + str(min(Wl)) + " nm to " +
                       str(max(Wl)) + " nm, step of " + str(Wl[1]-Wl[0]) + " nm" + "\n")
            file.write("Substrate thickness, in nm: " +
                       str(Th_Substrate) + "\n")
            file.write("Range of thin layer thickness: " +
                       str(Th_range[0]) + " to " + str(Th_range[1]) + " nm" + "\n")
            file.write("Range of thin layer indices: " +
                       str(n_range[0]) + " to " + str(n_range[1]) + "\n")
            file.write("Incident angle on the stack: " + str(Ang) + "°" + "\n")
            file.write("Concentration ratio: " + str(C) + "\n")
            file.write("Air temperature: " + str(T_air) + " K" + "\n")
            file.write("Absorber temperature: " + str(T_abs) + " K" + "\n")
            if evaluate.__name__ == "evaluate_low_e" or evaluate.__name__ == "evaluate_RTR":
                file.write("For low-e and RTR optimization profiles" + "\n")
                file.write("UV cutoff wavelength: " +
                           str(Lambda_cut_1) + " nm" + "\n")
                file.write("IR cutoff wavelength: " +
                           str(Lambda_cut_2) + " nm" + "\n")
            if evaluate.__name__ == "evaluate_netW_PV_CSP":
                file.write(
                    "For evaluate_netW_PV_CSP optimization profile" + "\n")
                file.write("PV fictitious cost: " + str(poids_PV) + "\n")
            file.write("Population size: " + str(pop_size) + "\n")
            file.write("Crossover rate: " + str(crossover_rate) + "\n")
            file.write("Evaluation rate: " + str(evaluate_rate) + "\n")
            file.write("Mutation rate: " + str(mutation_rate) + "\n")
            file.write("Values of f1 and f2: " +
                       str(f1) + " & " + str(f2) + "\n")
            file.write("Mutation range: " + str(mutation_delta) + "\n")
            file.write("Precision of the algorithm in auto: " +
                       str(precision_AlgoG) + "\n")
            file.write("Number of generations: " + str(nb_generation) + "\n")
            file.write("Number of run: " + str(nb_run) + "\n")
            file.write("Number of available CPU: " + str(cpu_count()) + "\n")
            file.write("Number of used CPU: " + str(cpu_used) + "\n")
            file.write("Total execution time (in s): " +
                       str("{:.2f}".format(time_real)) + "\n")
            file.write("Sum of processor computation time (in s): " +
                       str("{:.2f}".format(sum(tab_temps)) + "\n"))
            file.write("Seed value : " + str(seed) + "\n")
        print("The names and values of the simulation variables have been written.")


def Simulation_amont_aval_txt(parameters, Experience_results, directory):
    evaluate = parameters.get("evaluate")
    launch_time = Experience_results.get("launch_time")
    Comment = Experience_results.get("Comment")
    name_PV = Experience_results.get("name_PV")
    name_Th = Experience_results.get("name_Th")
    name_Sol_Spec = Experience_results.get("name_Sol_Spec")
    name_Sol_Spec_2 = Experience_results.get("name_Sol_Spec_2")
    Sol_Spec_int = Experience_results.get("Sol_Spec_int")
    Sol_Spec = parameters.get("Sol_Spec")
    Sol_Spec_2 = Experience_results.get("Sol_Spec_2")
    Lambda_cut_1 = parameters.get("Lambda_cut_1")
    Lambda_cut_2 = parameters.get("Lambda_cut_2")
    Wl = parameters.get("Wl")
    Sol_Spec_mod_T_int = Experience_results.get("Sol_Spec_mod_T_int")
    Sol_Spec_mod_A_int = Experience_results.get("Sol_Spec_mod_A_int")
    Sol_Spec_mod_R_int = Experience_results.get("Sol_Spec_mod_R_int")
    Sol_Spec_mod_R_int_2 = Experience_results.get("Sol_Spec_mod_R_int_2")
    Ps_amont = Experience_results.get("Ps_amont")
    Ps_amont_ref = Experience_results.get("Ps_amont_ref")
    Ps_aval = Experience_results.get("Ps_aval")
    Ps_aval_ref = Experience_results.get("Ps_aval_ref")
    Sol_Spec_mod_amont_int = Experience_results.get("Sol_Spec_mod_amont_int")
    Sol_Spec_mod_T_amont_int = Experience_results.get(
        "Sol_Spec_mod_T_amont_int")
    Sol_Spec_int_2 = Experience_results.get("Sol_Spec_int_2")
    Sol_Spec_mod_aval_int = Experience_results.get("Sol_Spec_mod_aval_int")
    Sol_Spec_mod_R_aval_int = Experience_results.get("Sol_Spec_mod_R_aval_int")
    R = Experience_results.get('R')
    T = Experience_results.get('T')

    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    Lambda_cut_2 = parameters.get('Lambda_cut_2')

    if evaluate.__name__ == "evaluate_netW_PV_CSP" or evaluate.__name__ == "evaluate_RTR" or evaluate.__name__ == "evaluate_low_e":

        filename = directory + "/simulation_amont_aval.txt"
        script_name = os.path.basename(__file__)
        with open(filename, "w") as file:
            file.write("Le nom du fichier est : " + str(script_name) + "\n")
            file.write("Heure de lancement " + str(launch_time) + "\n")
            file.write(str(Comment) + "\n")
            file.write("_____________________________________________" + "\n")
            file.write(
                "Le nom du fichier amont et le n° de la colone est : " + name_PV + "\n")
            file.write(
                "Le nom du fichier avant et le n° de la colone est : " + name_Th + "\n")
            file.write(
                "Le nom du spectre solaire utilisé pour l'optimisation ': " + name_Sol_Spec + "\n")
            file.write("L'intégration de ce spectre solaire (en W/m2) est " +
                       str("{:.2f}".format(Sol_Spec_int)) + "\n")
            file.write("La puissance transmise par le traitement du spectre solaire incident (en W/m2) est " +
                       str("{:.2f}".format(Sol_Spec_mod_T_int)) + "\n")
            file.write("La puissance réfléchie par le traitement du spectre solaire incident (en W/m2) est " +
                       str("{:.2f}".format(Sol_Spec_mod_R_int)) + "\n")
            file.write("La puissance absorbée par le traitement du spectre solaire incident (en W/m2) est " +
                       str("{:.2f}".format(Sol_Spec_mod_A_int)) + "\n")
            if Lambda_cut_1 != 0 and Lambda_cut_2 != 0:
                Wl_1 = np.arange(min(Wl), Lambda_cut_1, (Wl[1]-Wl[0]))
                Wl_2 = np.arange(Lambda_cut_1, Lambda_cut_2, (Wl[1]-Wl[0]))
                Wl_3 = np.arange(Lambda_cut_2, max(
                    Wl)+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
                # P_low_e = np.concatenate([R[0:len(Wl_1)],T[len(Wl_1):(len(Wl_2)+len(Wl_1)-1)], R[(len(Wl_2)+len(Wl_1)-1):]])
                file.write("\n")
                # Partie avec le spectre GT
                file.write("Calcul avec le spectre': " + name_Sol_Spec + "\n")
                # a = trapezoid(Sol_Spec[0:len(Wl_1)]* R[0:len(Wl_1)], Wl_1)
                # file.write("La puissance solaire réfléchie du début du spectre à Lambda_cut_UV (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                a = trapezoid(Sol_Spec[len(Wl_1):(
                    len(Wl_2)+len(Wl_1))] * T[len(Wl_1):(len(Wl_2)+len(Wl_1))], Wl_2)
                file.write("La puissance solaire transmise de Lambda_UV à Lambda_IR (en W/m2) est " +
                           str("{:.2f}".format(a)) + "\n")
                # a = trapezoid(Sol_Spec[(len(Wl_2)+len(Wl_1)):]* R[(len(Wl_2)+len(Wl_1)):], Wl_3)
                # file.write("La puissance solaire réfléchie à partir de Lambda_IR (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                # Partie avec le spectre DC
                file.write("Calcul avec le spectre': " +
                           name_Sol_Spec_2 + "\n")
                a = trapezoid(Sol_Spec_2[0:len(Wl_1)] * R[0:len(Wl_1)], Wl_1)
                file.write("La puissance solaire réfléchie du début du spectre à Lambda_cut_UV (en W/m2) est " +
                           str("{:.2f}".format(a)) + "\n")
                # a = trapezoid(Sol_Spec_2[len(Wl_1):(len(Wl_2)+len(Wl_1))]* T[len(Wl_1):(len(Wl_2)+len(Wl_1))], Wl_2)
                # file.write("La puissance solaire transmise de Lambda_UV à Lambda_IR (en W/m2) est " + str("{:.2f}".format(a)) + "\n")
                a = trapezoid(
                    Sol_Spec_2[(len(Wl_2)+len(Wl_1)):] * R[(len(Wl_2)+len(Wl_1)):], Wl_3)
                file.write("La puissance solaire réfléchie à partir de Lambda_IR (en W/m2) est " +
                           str("{:.2f}".format(a)) + "\n")
                del a, Wl_1, Wl_2, Wl_3

                file.write("\n")
                file.write(
                    "En amont (partie cellule PV sur un système solaire PV/CSP) : " + "\n")
                file.write("Le nom du spectre solaire est': " +
                           name_Sol_Spec + "\n")
                file.write("L'intégration de ce spectre solaire (en W/m2) est " +
                           str("{:.2f}".format(Sol_Spec_int)) + "\n")
                file.write("La puissance transmise par le traitement du spectre solaire GT (en W/m2) est " +
                           str("{:.2f}".format(Sol_Spec_mod_T_int)) + "\n")
                file.write("L'efficacité (%) de la cellule avec le spectre solaire non modifié (sans traitement) est " +
                           str("{:.3f}".format(Ps_amont_ref)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " +
                           str("{:.2f}".format(Sol_Spec_mod_amont_int)) + "\n")
                file.write("L'efficacité (%) de la cellule avec le spectre solaire modifié (avec traitement) est " +
                           str("{:.3f}".format(Ps_amont)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " +
                           str("{:.2f}".format(Sol_Spec_mod_T_amont_int)) + "\n")
                file.write("\n")
                file.write(
                    "En aval (partie absorbeur thermique sur un système solaire PV/CSP) : " + "\n")
                file.write("Le nom du spectre solaire est : " +
                           name_Sol_Spec_2 + "\n")
                file.write("L'intégration de ce spectre solaire (en W/m2) est " +
                           str("{:.2f}".format(Sol_Spec_int_2)) + "\n")
                file.write("La puissance réfléchie par le traitement du spectre solaire DC (en W/m2) est " +
                           str("{:.2f}".format(Sol_Spec_mod_R_int_2)) + "\n")
                file.write("L'efficacité (%) du traitement absorbant avec le spectre solaire non modifié (sans traitement) est " +
                           str("{:.3f}".format(Ps_aval_ref)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " +
                           str("{:.2f}".format(Sol_Spec_mod_aval_int)) + "\n")
                file.write("L'efficacité (%) du traitement absorbant avec le spectre solaire modifié (avec traitement) est " +
                           str("{:.3f}".format(Ps_aval)) + "\n")
                file.write("Soit une puissance utile (en W/m2) de " +
                           str("{:.2f}".format(Sol_Spec_mod_R_aval_int)) + "\n")

            print("Le fichier simulation_amont_aval.txt est écrit")


def Generate_materials_txt(parameters, Experience_results, directory):
    language = Experience_results.get('language')
    Mat_Stack = parameters.get("Mat_Stack")
    n_Stack = parameters.get("n_Stack")
    k_Stack = parameters.get("k_Stack")
    tab_best_solution = Experience_results.get('tab_best_solution')
    max_index = Experience_results.get('max_index')
    individual = tab_best_solution[max_index]
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(
        individual, n_Stack, k_Stack, Mat_Stack, parameters)
    Wl = parameters.get("Wl")
    R = Experience_results.get("R")
    T = Experience_results.get("T")
    A = Experience_results.get("A")
    # I write materials indexes in a text file
    Mats = []
    # I obtain the vf in individual
    vf = individual[len(Mat_Stack):,]
    for i in range(len(Mat_Stack)):
        name = Mat_Stack[i]
        n = n_Stack[:, i]
        k = k_Stack[:, i]
        # plot the graph
        plt.title("N et k de" + name, y=1.05)
        plt.plot(Wl, n, label="n extrapolé")
        plt.plot(Wl, k, label="k extrapolé")
    # =============================================================================
    #     # Open the initial file
    #     Wl_2, n_2, k_2 = open_material(result[0][i])
    #     plt.plot(Wl_2, n_2, 'o', label = "n data")
    #     plt.plot(Wl_2, k_2, 'o', label = "k data")
    # =============================================================================
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        plt.ylabel("n and k values (-)")
        plt.savefig(directory + "/" + "refractive_index" +
                    name + ".png", dpi=300, bbox_inches='tight')
        plt.close()
        # filename = "/" + str(name) + ".txt"
        if '-' in name:
            for j in range(i):
                if name not in Mats:
                    filename = directory + "/" + \
                        str(name) + "_vf=" + str(round(vf[i], 4)) + ".txt"
                    Mats.append(name)
                elif Mat_Stack[j] == name and j != i and vf[i] != vf[j]:
                    filename = directory + "/" + \
                        str(name) + "_vf=" + str(round(vf[i], 4)) + ".txt"
        else:
            if name not in Mats:
                filename = directory + "/" + str(name) + ".txt"
                Mats.append(name)
        with open(filename, "w") as file:
            for i in range(len(n)):
                file.write(str(Wl[i]) + "\t" + str(n[i]) +
                           "\t" + str(k[i]) + "\n")
    if language == "en":
        print("The n,k of each material have been writed in a file named as them")
    elif language == "fr":
        print("Les n et k de chaque matériau ont été écrits dans un fichier du même nom")


def Reflectivity_plot_fit(parameters, Experience_results, directory):

    if 'evaluate' in parameters:
        evaluate = parameters.get("evaluate")
    else:
        raise ValueError("You must use a evaluate fonction")

    if 'fit' not in evaluate.__name__:
        raise ValueError("You must use a fit fonction")

    if 'Signal_fit' in parameters:
        Signal_fit = parameters.get("Signal_fit")
    else:
        raise ValueError(
            "Reflectivity_plot_fit must be used an experimental signal, named 'Signal_fit'")

    Wl = parameters.get("Wl")
    R = Experience_results.get("R")

    # Reflectivity plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectivity (-)', color="black")
    ax1.plot(Wl, R, color="black", label="Measure")
    ax1.plot(Wl, Signal_fit, color="red", label="Fit")
    ax1.legend()
    ax1.set_ylim(0, 1)  # Change y-axis' scale
    fig.tight_layout()
    ax1.legend()
    plt.title("Reflectivity measured vs fitted", y=1.05)
    # Save the plot.
    plt.savefig(directory + "/" + "Optimum_Reflectivity_fit.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def Transmissivity_plot_fit(parameters, Experience_results, directory):

    if 'evaluate' in parameters:
        evaluate = parameters.get("evaluate")
    else:
        raise ValueError("You must use a evaluate fonction")

    if 'fit' not in evaluate.__name__:
        raise ValueError("You must use a fit fonction")

    if 'Signal_fit_2' in parameters:
        Signal_fit = parameters.get("Signal_fit_2")
    else:
        raise ValueError(
            "Transmissivity_plot_fit must be used an experimental signal, named 'Signal_fit'")

    Wl = parameters.get("Wl")
    T = Experience_results.get("T")

    # Reflectivity plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transmissivity (-)', color="black")
    ax1.plot(Wl, T, color="black", label="Measure")
    ax1.plot(Wl, Signal_fit, color="red", label="Fit")
    ax1.legend()
    ax1.set_ylim(0, 1)  # Change y-axis' scale
    fig.tight_layout()
    ax1.legend()
    plt.title("Transmissivity measured vs fitted", y=1.05)
    # Save the plot.
    plt.savefig(directory + "/" + "Optimum_Transmissivity_fit.png",
                dpi=300, bbox_inches='tight')
    plt.show()
