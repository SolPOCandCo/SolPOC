# -*- coding: utf-8 -*-
"""
Created on 27 07 2023
COPS v 0.9.0
@author: A.Grosjean, A.Soum-Glaude, A.Moreau & P.Bennet
contact : antoine.grosjean@epf.fr

List of main functions used and developed for COPS. For use them without any implementation work see the other Python script
"""

import numpy as np
import math
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import random

def RTA3C(Wl, d, n, k, Ang=0):
    """_________________________________________________
    Exemple of RTA
    RTA is the key function : it's allow us to calcul reflectivity (R) transmissivity (T) and absorptivity(A)
    This function is a exemple working only with 3 thin layer on the substrat
    This tutorial functions is not used in the code
    """

    """
    The RTAA function calculates the reflectivity, the transmissivity, and the absorptivity of a group of thin layers applied on a substrate
    as reflectivity + transmissivity + absorptivity = 1
    The number after 'RTA' means the amount of thn layers applied on the substrate (we count the substrate in the amount). The RTA 3C code is logically for 1 substrate and 2 thin layers maximum

    The numpy (np) utilisation is a time saver on the launch
    => the time saved compared to the launch of an RTA function for one wavelength in a for loop is about 100 times faster

    Input arguments :
    l : the wavelength in nm, has a vector type
    d : substrate and thin layers thickness in nm
    n : real parts of materials refraction indexes. n is a 2D table with the thin layers indexes in the columns and the wavelengths in the rows
    k : complex parts of material refraction indexes. k is a 2D table with the thin layers extinction coefficients in the columns and the wavelengths in the rows
    Ang : incidence angle of the radiation in degrees

    Output :
    Refl is a column vector which includes the reflectivity. Column indexes corresponds to wavelengths'
    Trans is a column vector which includes the transmissivity. Column indexes corresponds to wavelengts'
    Abs, a column vector which includes the absorptivity. Column indexes corresponds to wavelengths'

    Test :
    Write these variables :
    l = np.arange(600,750,100). We can notice that two wavelengths are calculated : 600 and 700 nm
    d = np.array([[1000000, 150, 180]]). 1 mm of substrate, 150 nm of n°1 layer, 180 of n°2 and empty space (n=1, k=0)
    n = np.array([1.5, 1.23,1.14],[1.5, 1.2,1.1]])
    k = np.array([[0, 0.1,0.05], [0, 0.1, 0.05]])
    Ang = 0

    # Run functon
    Refl, Trans, Abs = RTA3C(l, d, n, k, Ang)
    
    For the indexes notation n and k, understand that
    @ 600 nm n = 1.5 for the substrate, n = 1.23 for the layer n°1 and n = 1.14 for the layer n°2
    @ 700 nm n = 1.5 for the substrate, n = 1.20 for the layer n°1 and n = 1.1 for the layer n°2
    @ 600 nm k = 0 for the substrate, k = 0.1 for the layer n°1 and k = 0.05 for the layer n°2
    @ 700 nm k = 0 for the substrate, k = 0.1 for the layer n°1 and k = 0.05 for the layer n°2

    We can get : Refl = array([0.00767694, 0.00903544]), Trans = array([0.60022613, 0.64313401]), Abs = array([0.39209693, 0.34783055])
    => The reflectivity is 0.00767694 (number between 0 and 1) at 600 nm and 0.00903544 at 700 nm
    """
    # Add an air layer on top
    n = np.append(n, np.ones((len(Wl), 1)), axis=1) # replacement of 2 by len(l)
    k = np.append(k, np.zeros((len(Wl), 1)), axis=1) # replacement of 2 by len(l)
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
    nS = n[:,0] # I take the 1st column which includes the n of the substrate for wavelengths
    kS = k[:,0]
    Ns = nS + 1j*kS
    PhiS = np.arcsin(N0*np.sin(Phi0)/Ns)
    qSPolaS = Ns*np.cos(PhiS)
    qSPolaP = Ns/np.cos(PhiS) # Ok here 
    
    # Multilayers (layer 1 is the closest one to the substrate)
    nj= np.delete(n,0, axis=1)
    kj= np.delete(k,0, axis=1)
    dj= np.delete(d,0, axis=1)

    numlayers = nj.shape[1] # nj is just a table 
    Nj = np.zeros((numlayers,1,len(Wl)), dtype=complex) # OK

    # was a column in Scilab, row here
    Phij = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    qjPolaS = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    qjPolaP = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    thetaj = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    MpolaS = np.zeros((2, 2*numlayers,len(Wl)), dtype=complex)
    MpolaP = np.zeros((2, 2*numlayers,len(Wl)), dtype=complex)
    Ms = np.zeros((2, 2,len(Wl)), dtype=complex)
    Mp = np.zeros((2, 2,len(Wl)), dtype=complex)

    sous_tableaux = np.split(nj,nj.shape[1],axis=1)
    nj = np.array([el.reshape(1,len(Wl)) for el in sous_tableaux]) # el.reshape(1,2) becomes el.reshape(1,len(l))
    sous_tableaux = np.split(kj,kj.shape[1],axis=1)
    kj = np.array([el.reshape(1,len(Wl)) for el in sous_tableaux])

    dj = np.squeeze(dj) #
    # Note  : invert table with numpy.transpose()
    for LayerJ in range(numlayers): 
        Nj[LayerJ] = nj[LayerJ] + 1j * kj[LayerJ]
        Phij[LayerJ] = np.arcsin(N0 * np.sin(Phi0) / Nj[LayerJ])
        qjPolaS[LayerJ] = Nj[LayerJ] * np.cos(Phij[LayerJ])
        qjPolaP[LayerJ] = Nj[LayerJ] / np.cos(Phij[LayerJ])
        thetaj[LayerJ] = (2 * np.pi / Wl) * dj[LayerJ] * Nj[LayerJ] * np.cos(Phij[LayerJ]) # OK

        # Characteristic matrix of layer j
        """ Calcul of MpolaS"""
        MpolaS[0, 2*LayerJ] = np.cos(thetaj[LayerJ]) # In Scilab MpolaS(1,2*LayerJ-1)
        MpolaS[0, 2*LayerJ+1] = -1j/qjPolaS[LayerJ]*np.sin(thetaj[LayerJ]) # In Scilab MpolaS(1,2*LayerJ)
        MpolaS[1, 2*LayerJ] = -1j*qjPolaS[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaS[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        """ Calculation of MpolaP"""
        MpolaP[0, 2*LayerJ] = np.cos(thetaj[LayerJ])
        MpolaP[0, 2*LayerJ+1] = -1j/qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ] = -1j*qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        #print(MpolaS)
    
    # Global characteristic (transfer) matrix [Furman92, Andersson80]
    if numlayers == 1: # Substrate only
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        Ms = M1s
        Mp = M1p
    elif numlayers == 2: # Substrate + 1 layer
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M2s = np.array([[MpolaS[0,2], MpolaS[0,3]], [MpolaS[1,2], MpolaS[1,3]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        M2p = np.array([[MpolaP[0,2], MpolaP[0,3]], [MpolaP[1,2], MpolaP[1,3]]])
        # Matrix multiplication with conservation of the 3rd axis (z in an orthonormal coordonate system, named 1 here) constant
        Ms = np.einsum('nkl,kml->nml', M2s, M1s)
        Mp = np.einsum('nkl,kml->nml', M2p, M1p)
    elif numlayers == 3: # Substrate + 2 layers
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M2s = np.array([[MpolaS[0,2], MpolaS[0,3]], [MpolaS[1,2], MpolaS[1,3]]])
        M3s = np.array([[MpolaS[0,4], MpolaS[0,5]], [MpolaS[1,4], MpolaS[1,5]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        M2p = np.array([[MpolaP[0,2], MpolaP[0,3]], [MpolaP[1,2], MpolaP[1,3]]])
        M3p = np.array([[MpolaP[0,4], MpolaP[0,5]], [MpolaP[1,4], MpolaP[1,5]]])
        Ms = np.einsum('nkl,kml->nml', M3s, np.einsum('nkl,kml->nml', M2s, M1s))
        Mp = np.einsum('nkl,kml->nml', M3p, np.einsum('nkl,kml->nml', M2p, M1p))
        
    # Matrix element
    m11s = Ms[0,0]
    m12s = Ms[0,1]
    m21s = Ms[1,0]
    m22s = Ms[1,1]
        
    m11p = Mp[0,0]
    m12p = Mp[0,1]
    m21p = Mp[1,0]
    m22p = Mp[1,1]
        
    # Fresnel total reflexion and transmission coefficient
    rs = (q0PolaS*m11s-qSPolaS*m22s+q0PolaS*qSPolaS*m12s-m21s)/(q0PolaS*m11s+qSPolaS*m22s+q0PolaS*qSPolaS*m12s+m21s)
    rp = (q0PolaP*m11p-qSPolaP*m22p+q0PolaP*qSPolaP*m12p-m21p)/(q0PolaP*m11p+qSPolaP*m22p+q0PolaP*qSPolaP*m12p+m21p)
    ts = 2*q0PolaS/(q0PolaS*m11s+qSPolaS*m22s+q0PolaS*qSPolaS*m12s+m21s)
    tp = 2*q0PolaP/(q0PolaP*m11p+qSPolaP*m22p+q0PolaP*qSPolaP*m12p+m21p)
    
    # Power transmittance
    Rs = (np.real(rs)) ** 2 + (np.imag(rs)) ** 2;
    Rp = (np.real(rs)) ** 2 + (np.imag(rp)) ** 2;
    Refl = (Rs + Rp) / 2 # this stands only when the incident light is unpolarized (ambient)
        
    # Power transmittance
    #Transmittance of the multilayer stack only (substrate transmittance is not taken into account !)
    Ts = np.real(qSPolaS) / np.real(q0PolaS) * ((np.real(ts) ** 2) + (np.imag(ts) ** 2))
    Tp = np.real(qSPolaP) / np.real(q0PolaP) * ((np.real(tp) ** 2) + (np.imag(tp) ** 2))
    TransMultilayer = (Ts + Tp) / 2 # This stands only when the incident light is unpolarized (ambient)
        
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
    See the function RTA3C for a example / tutoral and the version of the function write for 3 layer (2 thin layer + the substrat)
    RTA calculates the reflectivity, transmissivty and absorptivity using Abélès matrix formalism
    The Abélès matrix formalism provide the best ratio accurency / speed for stack below 100 thin layers
    The present version of RTA work for a infinit number of thin layer, but we not recommand to go over 100
    Parameters
    ----------
    l : array
        Wavelength, in nanometer
    d : array
        Tickness of stack, including the substrat
    n : array 
        DESCRIPTION.
    k : array
        DESCRIPTION.
    Ang : int, optional
        Incidence angle (in degres) of the light one the optical stack. The default is 0 degres, so light perpendicular at the substrat

    Returns
    -------
    Refl : array
        the stack reflectivity, for each wavelength.
    Trans : TYPE
        the stack transmissivity, for each wavelength.
    Abs : TYPE
        the stack absorptivituy, for each wavelength.

    """

    # Add an air layer on top
    n = np.append(n, np.ones((len(Wl), 1)), axis=1)
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
    nS = n[:,0] # I take the 1st column whihc includes the n of the substrate for wavelengths
    kS = k[:,0]
    Ns = nS + 1j*kS
    PhiS = np.arcsin(N0*np.sin(Phi0)/Ns)
    qSPolaS = Ns*np.cos(PhiS)
    qSPolaP = Ns/np.cos(PhiS) # Ok here 
    
    # Multilayers (layer 1 is the one closest to the substrate)
    nj= np.delete(n,0, axis=1)
    kj= np.delete(k,0, axis=1)
    dj= np.delete(d,0, axis=1)

    numlayers = nj.shape[1] # nj is just a table 
    Nj = np.zeros((numlayers,1,len(Wl)), dtype=complex) # OK
    """3D Matrix here. 
    "z" axis corresponds to different wavelengths"""
    Phij = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    qjPolaS = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    qjPolaP = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    thetaj = np.zeros((numlayers,1,len(Wl)), dtype=complex)
    MpolaS = np.zeros((2, 2*numlayers,len(Wl)), dtype=complex)
    MpolaP = np.zeros((2, 2*numlayers,len(Wl)), dtype=complex)
    Ms = np.zeros((2, 2,len(Wl)), dtype=complex)
    Mp = np.zeros((2, 2,len(Wl)), dtype=complex)
    """Resizing of nj and kj
    """
    sous_tableaux = np.split(nj,nj.shape[1],axis=1)
    nj = np.array([el.reshape(1,len(Wl)) for el in sous_tableaux])
    sous_tableaux = np.split(kj,kj.shape[1],axis=1)
    kj = np.array([el.reshape(1,len(Wl)) for el in sous_tableaux])
    """ Transforms a (1,3) vector into a (3,) vector
    """
    dj = np.squeeze(dj) #    
    for LayerJ in range(numlayers): 
        Nj[LayerJ] = nj[LayerJ] + 1j * kj[LayerJ]
        Phij[LayerJ] = np.arcsin(N0 * np.sin(Phi0) / Nj[LayerJ])
        qjPolaS[LayerJ] = Nj[LayerJ] * np.cos(Phij[LayerJ])
        qjPolaP[LayerJ] = Nj[LayerJ] / np.cos(Phij[LayerJ])
        thetaj[LayerJ] = (2 * np.pi / Wl) * dj[LayerJ] * Nj[LayerJ] * np.cos(Phij[LayerJ]) # OK
        
        # Characteristic matrix of layer j
        """ Calculation of MpolaS"""
        MpolaS[0, 2*LayerJ] = np.cos(thetaj[LayerJ]) # In Scilab MpolaS(1,2*LayerJ-1)
        MpolaS[0, 2*LayerJ+1] = -1j/qjPolaS[LayerJ]*np.sin(thetaj[LayerJ]) # In Scilab MpolaS(1,2*LayerJ)
        MpolaS[1, 2*LayerJ] = -1j*qjPolaS[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaS[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        """ Calculation of MpolaP"""
        MpolaP[0, 2*LayerJ] = np.cos(thetaj[LayerJ])
        MpolaP[0, 2*LayerJ+1] = -1j/qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ] = -1j*qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        #print(MpolaS)
    
    # Global characteristic (transfer) matrix [Furman92, Andersson80]
    if numlayers == 1: # Substrate only
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        Ms = M1s
        Mp = M1p
    else : # The ultime code =D
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        for i in range(numlayers):
            exec(f"M{i + 1}s = np.array([[MpolaS[0,{i * 2}], MpolaS[0,{i * 2 + 1}]], [MpolaS[1,{i * 2}], MpolaS[1,{i * 2 + 1}]]])")
        # Calculation of Mp elements
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        for i in range(numlayers):
            exec(f"M{i + 1}p = np.array([[MpolaP[0,{i * 2}], MpolaP[0,{i * 2 + 1}]], [MpolaP[1,{i * 2}], MpolaP[1,{i * 2 + 1}]]])")
        # Calculation of Ms and Mp 
        Ms = M1s
        Mp = M1p
        for i in range(2, numlayers + 1):
            Mi_s = eval(f"M{i}s")
            Mi_p = eval(f"M{i}p")
            Ms = np.einsum('nkl,kml->nml', Mi_s, Ms)
            Mp = np.einsum('nkl,kml->nml', Mi_p, Mp)
    # Matrix element
    m11s = Ms[0,0]
    m12s = Ms[0,1]
    m21s = Ms[1,0]
    m22s = Ms[1,1]
        
    m11p = Mp[0,0]
    m12p = Mp[0,1]
    m21p = Mp[1,0]
    m22p = Mp[1,1]
        
    # Fresnel total reflexion and transmission coefficient
    rs = (q0PolaS*m11s-qSPolaS*m22s+q0PolaS*qSPolaS*m12s-m21s)/(q0PolaS*m11s+qSPolaS*m22s+q0PolaS*qSPolaS*m12s+m21s)
    rp = (q0PolaP*m11p-qSPolaP*m22p+q0PolaP*qSPolaP*m12p-m21p)/(q0PolaP*m11p+qSPolaP*m22p+q0PolaP*qSPolaP*m12p+m21p)
    ts = 2*q0PolaS/(q0PolaS*m11s+qSPolaS*m22s+q0PolaS*qSPolaS*m12s+m21s)
    tp = 2*q0PolaP/(q0PolaP*m11p+qSPolaP*m22p+q0PolaP*qSPolaP*m12p+m21p)
    
    # Power transmittance
    Rs = (np.real(rs)) ** 2 + (np.imag(rs)) ** 2;
    Rp = (np.real(rs)) ** 2 + (np.imag(rp)) ** 2;
    Refl = (Rs + Rp) / 2 # this stands only when the incident light is unpolarized (ambient)
        
    # Power transmittance
    #Transmittance of the multilayer stack only (substrate transmittance is not taken into account !)
    Ts = np.real(qSPolaS) / np.real(q0PolaS) * ((np.real(ts) ** 2) + (np.imag(ts) ** 2))
    Tp = np.real(qSPolaP) / np.real(q0PolaP) * ((np.real(tp) ** 2) + (np.imag(tp) ** 2))
    TransMultilayer = (Ts + Tp) / 2 # This stands only when the incident light is unpolarized (ambient)
        
    # Transmittance of the substrate
    d = np.squeeze(d)
    TransSub = np.exp((-4*np.pi*kS*d[0])/Wl)
        
    # Transmittance of the substrate + multilayer stack
    Trans = TransMultilayer * TransSub
        
    # Power absorptance
    Abs = 1 - Refl - Trans
    return Refl, Trans, Abs

def nb_compo(Mat_Stack):
    """
    Gives back the amount of composite thin layers (made up of two materials). As a cermet or a porous material,
    a composite thin layer includes the dash - in it string
    Exemple : 'W-Al2O3' => composite layer of Al2O3 matrix with W inclusion (cermet type)
              ' air-SiO2' =>  composite layer of SiO2 matrix with air inclusion, (porous type)
    """
    nb = 0 
    for i in Mat_Stack: 
        if "-" in i:
            nb += 1
    return nb

def interpolate_with_extrapolation(x_new, x, y):
    """
    This fonction provide linear extrapolation for refractive index data
    Extrapolation is necessary, because refractive index may cannot covert the wavelenght domain used
    
    Parameters
    ----------
    x : Numpy array of float
        Here, x represent the wavelegenth domain present in the materials files
    y : Numpy array of float
        Here, y represent the refractive index (n or k) present in the materials files
    x_new : Numpy array of float
        The new wavelenght domain where the refractive index (y) must be extrapoled

    Returns
    -------
    TYPE
        Numpy array of float
        Here, y represent the refractive index (n or k) extrapoled
        y_new have the same dimension than x_new

    Exemple :
        
    # Original data
    Wl_mat = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800])
    n_mat = np.array([1.75, 1.640625, 1.5625, 1.515625, 1.5, 1.515625, 1.5625, 1.640625, 1.75])
    
    # Wavelength domain used
    Wl = np.arange(200, 1001, 50)
    
    # Interpolation with linear extrapolation 
    n_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)
    
    n_map : array([2.1875  , 2.078125, 1.96875 , 1.859375, 1.75    , 1.640625,
           1.5625  , 1.515625, 1.5     , 1.515625, 1.5625  , 1.640625,
           1.75    , 1.859375, 1.96875 , 2.078125, 2.1875  ])
    """
    interp_function = interp1d(x, y, kind='linear', fill_value='extrapolate')
    y_new = interp_function(x_new)
    return y_new

def Made_Stack(Mat_Stack, Wl):
    """
    This key fonction strat with a list a material with describe the stack 
    It returns two table numpy array, on for the real part of the refractive index (n), and the other for the imaginary part (k) 

    Parameters
    ----------
    Mat_Stack : List of string, with containt each material of the stack. See as exemple the function write_stack_period
    Exemple : Mat_Stack = ['BK7', 'TiO2', 'SiO2'] is a stack of TiO2 and SiO2 deposited on BK7 subtrate
    The SiO2 layer is in contact with the air
    
    DESCRIPTION.
    Wl : numpy array
        The list of the wavelenght

    Returns
    -------
    n_Stack : numpy array
        Each line is a different wavelenght
        Each colonne is a different material
    k_Stack : nympy array
        Same than for n_Stack, but for k (the imaginary part).

    Exemple 1
    Mat_Stack = ['BK7', 'TiO2', 'SiO2']
    Wl = [400, 450, 500]
    
    n_Stack, k_Stack = Made_Stack(Mat_Stack, Wl)
    n_Stack : array([[1.5309    , 2.84076063, 1.48408   ],
           [1.5253    , 2.78945014, 1.479844  ],
           [1.5214    , 2.69067759, 1.476849  ]])
    The value must be understand like : 
                    BK7      TiO2    SiO2
       400 nm     1.5309	2.84076	 1.48408
       450 nm     1.5253	2.78945	 1.47984
       500 nm     1.5214	2.69068	 1.47685
    As exemple, le value 1.5214 is the real part of the refractive index of BK7, at 500 nm
    """
    # Creation of the Stack
    # I search if the name of a thin layer material is separated by a dash -
    # If yes, it's a composite material 
    no_dash = True
    for s in Mat_Stack: 
        if "-" in s:
            no_dash = False
            break
        
    if no_dash : # If no_dash is true, I enter the loop
        n_Stack = np.zeros((len(Wl),len(Mat_Stack)))
        k_Stack = np.zeros((len(Wl),len(Mat_Stack)))
        for i in range(len(Mat_Stack)):
            Wl_mat, n_mat, k_mat = open_material(Mat_Stack[i])    
            # Interpolation 
            n_mat = interpolate_with_extrapolation(Wl,Wl_mat, n_mat)
            k_mat = interpolate_with_extrapolation(Wl,Wl_mat, k_mat)
            n_Stack[:,i] = n_mat[:,]
            k_Stack[:,i] = k_mat[:,]
    
        return n_Stack, k_Stack
    
    else : # Else, there must be a dash, so two materials 
        n_Stack = np.zeros((len(Wl),len(Mat_Stack),2))
        k_Stack = np.zeros((len(Wl),len(Mat_Stack),2))
        for i in range(len(Mat_Stack)):
            # I open the first material
            list_mat = []
            list_mat = Mat_Stack[i].split("-")
            if len(list_mat) == 1: 
                # the list includes one material. I charge it as usual
                # Row: wavelenght, column : material indexes 
                Wl_mat, n_mat, k_mat = open_material(Mat_Stack[i])    
                # Interpolation 
                n_mat = interpolate_with_extrapolation(Wl,Wl_mat, n_mat)
                k_mat = interpolate_with_extrapolation(Wl,Wl_mat, k_mat)
                n_Stack[:,i,0] = n_mat[:,]
                k_Stack[:,i,0] = k_mat[:,]
            if len(list_mat) == 2: 
                # the list includes two materials. I place the second on the z=2 axis
                Wl_mat, n_mat, k_mat = open_material(list_mat[0])    
                # Interpolation 
                n_mat = interpolate_with_extrapolation(Wl,Wl_mat, n_mat)
                k_mat = interpolate_with_extrapolation(Wl,Wl_mat, k_mat)
                n_Stack[:,i,0] = n_mat[:,]
                k_Stack[:,i,0] = k_mat[:,]    
                # Opening of the second material 
                Wl_mat, n_mat, k_mat = open_material(list_mat[1])    
                # Interpolation 
                n_mat = interpolate_with_extrapolation(Wl, Wl_mat, n_mat)
                k_mat = interpolate_with_extrapolation(Wl, Wl_mat, k_mat)
                n_Stack[:,i,1] = n_mat[:,]
                k_Stack[:,i,1] = k_mat[:,]      
        return n_Stack, k_Stack

def Made_Stack_vf(n_Stack, k_Stack, vf=[0]):
    """
    n_Stack_vf, or k_stack_vf means an n and k calculated by a Bruggeman function (EMA mixing law).
    These are the values to be injected into RTA
    If vf = 0 for all materials, then n_Stack_vf = n_Stack (idem for k)
    Otherwise, calculate : 
    
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
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if all(elem == 0 for elem in vf): #  (vf ==0).all():
        """ All ths vf(s) = 0. It's not necessary to launch Bruggman. 
        """
        return n_Stack, k_Stack
    else : 
        """ vf.all == [0] is not True. At least vf exists, it means that only one layer of the stack 
        is made of two materials. n_Stack and k_Stack are 3D table.For exemple, for "W-Al2O3",
        W datas are in range [:,:,0] and Al2O3 datas are in range [:,:,1]
        """
        n_Stack_vf = np.empty((n_Stack.shape[0], np.shape(n_Stack)[1]))
        k_Stack_vf = np.empty((k_Stack.shape[0], np.shape(k_Stack)[1]))
        # old version
        #n_Stack_vf = []
        #k_Stack_vf = []
        #n_Stack_vf = np.array(n_Stack_vf)
        #k_Stack_vf = np.array(k_Stack_vf)
        for i in range (np.shape(k_Stack)[1]):
            #  I check every layer and recup the n and the k of the matrix (M) and of the inclusions (I)
            # If the layer is made of one material only, the datas are in nI and kI
            # => nM and km are full of zeros in this case
            nM= np.copy(n_Stack[:, i, 1])
            kM= np.copy(k_Stack[:, i, 1])
            nI= np.copy(n_Stack[:, i, 0])
            kI= np.copy(k_Stack[:, i, 0])
                #kM= k_Stack[:,i,1].copy()
                #nI= n_Stack[:,i,0].copy()
                #kI= k_Stack[:,i,0].copy()
            n_Stack_vf[:,i], k_Stack_vf[:,i] = Bruggeman(nM, kM, nI, kI, vf[i])
                #n_Stack_vf[:,i], k_Stack_vf[:,i] = Bruggeman_np(nM, kM, nI, kI, vf[i])
        return n_Stack_vf, k_Stack_vf

def Bruggeman(nM, kM, nI, kI, VF):
    """
    Bruggemann function. 
    Allow us to calcalted the complexe refractive index of a mixture of two materials, using a EMA (Effective Medium Approximation)
    Parameters
    ----------
    nM : array
        Real part of refractive index of host Matrix (M is for Matrix)
    kM : array
        Complexe part of refractive index of host Matrix (M is for Matrix)
    nI : array
        Real part of refractive index of inclusion (I is for Inclusion)
    kI : TYPE
        Complexe part of refractive index of inclusion (I is for Inclusion)
    VF : int
        Volumic Fraction of inclusions in host matrix. Number between 0 et 1 (0 and 100%)     

    Returns
    -------
    nEffective : array
        Real part of the refractive index of the effective medium : the mixture of the host Matrix and the embedded particules
    kEffective : array
        Complexe part of the refractive index of the effective medium : the mixture of the host Matrix and the embedded particules
    
    Noted than If vf = 0 :
        nEffective = nM and kEffective = kM
    Noted than If vf = 1.0 : 
        nEffective = nI and kEffective = kI
    """
    if VF == 0 :
        nEffective = nI
        kEffective = kI
        return nEffective, kEffective
    
    eM = (nM + 1j*kM)**2
    eI = (nI + 1j*kI)**2
    y = 2
    nEffective = np.zeros(np.shape(nM))
    kEffective = np.zeros(np.shape(nM))
    for l in range(np.shape(nM)[0]):
        a = -y
        b = (VF*y + VF - 1)*eI[l] - (VF*y + VF - y)*eM[l]
        c = eM[l]*eI[l]
        p = np.roots([a, b, c])
        e1 = p[0]
        e2 = p[1]
        if np.imag(e1) > 0:
            Neffective = np.sqrt(e1)
        elif np.imag(e2) > 0:
            Neffective = np.sqrt(e2)
        else:
            if np.real(e1) > 0:
                Neffective = np.sqrt(e1)
            elif np.real(e2) > 0:
                Neffective = np.sqrt(e2)
        nEffective[l] = np.real(Neffective)
        kEffective[l] = np.imag(Neffective)
    
    return nEffective , kEffective

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
        Black Body Luminance, in W.m**-3
        Note the pi factor, for transform W.m**-3.sr**-1 into W.m**-3
    """
    h = 6.62607015e-34 # Planck constant
    c = 299792458 # Speed of light
    k = 1.380649e-23 # Boltzmann constant
    BB_spec = 2 * np.pi * h * c**2 / (Wl * 1e-9)**5 / (np.exp(h * c / (Wl * 1e-9 * k * T)) - 1) * 1e-9
    return BB_spec

def SolarProperties(Wl, R, SolSpec):
    """
    Parameters
    ----------
    R : array
        Stack Optical Properties, for different Wavelength, properly intepoled
        Not than R is not necessary the reflectivity, can be transmissivity or absorptivity
    Wl : array
        Wavelength, in nm
    SolSpec : Vector. SolarSpectrum used, properly intepoled in W.m2nm-1
    R and SolSpec must have the same length
    Returns
    -------
    R_s : float
        Solar Properties, accorting to the Strack Optical Properties
        => not necessary Solar Reflectance. 
    """
    if len(Wl) != len(R) or len(Wl) != len(SolSpec) or len(R) != len(SolSpec):
        raise ValueError("Vectors l, R, and SolSpec must have the same length.")
    try:
        R_Spec = []
        R_Spec = R * SolSpec
        SolSpec_int = trapz(SolSpec, Wl)
        R_int = trapz(R_Spec, Wl)
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
        Black Body Luminance, in W.m**-3, properly intepoled
    A and BB must have the same length ! 
    Returns
    -------
    E_BB : float
        Thermal emissivty, according to the black body irradiance BB
    """
    if len(l) != len(A) or len(l) != len(BB) or len(A) != len(BB):
        raise ValueError("Vectors l, R, and SolSpec must have the same length.")
    try:
        E_BB_Spec = []
        E_BB_Spec = A * BB
        BB_Spec_int = trapz(BB, l)
        E_int = trapz(E_BB_Spec, l)
        E_BB = E_int / BB_Spec_int
    except:
            raise ValueError("Vectors l, R, and SolSpec must be a numpy array.")

    return E_BB

def helio_th(A_s, E_BB, T_stack, T_air, C, I,  r_Opt = 0.7, FFabs=1):
    """
    Calculate the heliothermal efficiency
    Parameters
    ----------
    A_s : Float
        Solar Absoptivity, calculate previsouly
    E_BB : Float
        Thermal emissivity, calculate previsouly according to the temperature T_Stack
    T_stack : Float
        Temperature, in Kelvin, of the stack 
    T_air : Float
        Temperature, in Kevlin, of the environement.
    C : Float
        Solar concentration .
    I : Float
        Solar irradiance on the stack, in W/m2. Normaly calculat with integration of the solar spectrum
    r_Opt : TYPE, optional
        Optical performance of the optical concentrator, used with the solar selectiv stack. The default is 0.7.
    FFabs : Float, optional
        Ratio between the absorting surface and the emissivity surface. The default is 1.

    Returns
    -------
    rHelio : Flaot
        Heliothermal efficiency, the conversion between photo to thermal, according the operatining conditions (T_stack, T_air, C, I etc)
    """
    sigma = 5.67037321e-8
    rHelio = A_s - E_BB * FFabs * sigma * (T_stack**4 - T_air**4)/(r_Opt * C * I)
    return rHelio

def open_material(name): 
    """
    Open a text file which contain refractive index from Materials folder
    Exemple Wl, n, k = open_material("Ag") open a file named Ag.txt, in the Materials/
    ----------
    name : a string
        The name a texte files witch contain Wavelength, and refractive index values
        The texte file must be formated according the following : 
            1 row : wavelenght in nm
            2 row : real part of refractive index
            3 row : imaginary part of the refractive index
            row seperate by a tabulation 

    Returns
    -------
    Wl : numpy array 
        Wavelength, in nanometer (nm) 
    n : numpy array
        Real part of the Refractive Index
    k : numpy array
        Complexe part of the Refractive Index.
    """
    
    assert isinstance(name, str), f"Argument 'name' must be a string but had type {type(name)}"
    # Initialise an empty table
    tableau3D = []
    name="Materials/" + name + ".txt"
    try: 
        # Open the file in read-only mode
        file = open(name, "r")
        # use readlines to read all the lines of the file
        # The "lines" variable is a list with all the lines from the file
        lines = file.readlines()
        # Close the file after reading the lines
        file.close()
        
        # Make an iteration on the lines
        nb_line = len(lines)
        for i in range (nb_line):
            values = lines[i].split("\t")
            values[2] =values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)
            
    except FileNotFoundError:
        print(f"File {name} was not found.")
    # Transform the list into a numpy table
    tableau3D = np.array(tableau3D)
    

    Wl = []  
    Wl = tableau3D[:,0]
    n = []
    n = tableau3D[:,1]
    k = []
    k = tableau3D[:,2]
    
    return Wl, n, k

def open_SolSpec(name = 'Materials/SolSpec.txt', type_spec="DC"):     
    """
    Name : string
    Open txt file with the solar spectra data. Normal name is SolSpec.txt in material files
    SolSpec is a table, with one wavelength per line. 
    
    type_spec : string 
    Optional is for the type for solar spectra
    DC : Direct and Circumsolar
        The sun irradiance with come from directly from the sun and his corona
    GT Global Tilt
        The sun irradiance with comme from the sun and the near environement (reflexion / albedo)
    Extr : Extra-terra solar spectrum. 
    """
    # Initialise an empty table
    tableau3D = []
    try: 
        # Open the file in read-only mode
        file = open(name, "r")
        # use readlines to read all the lines of the file
        # The "lines" variable is a list with all the lines from the file
        lines = file.readlines()
        # Close the file after reading the lines
        file.close()
        
        # Make an iteration on the lines
        nb_line = len(lines)
        for i in range (nb_line):
            values = lines[i].split("\t")
            values[2] =values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)
            
    except FileNotFoundError:
        print("Le fichier n'a pas été trouvé.")
    # Transform the list into a numpy table
    tableau3D = np.array(tableau3D)
    # Extract wished datas   
    Wl = []  
    Wl = tableau3D[:,0]
    spec = []
    if type_spec == "DC":
        spec = tableau3D[:,1]
    if type_spec == "Extr":
        spec = tableau3D[:,2]
    if type_spec == "GT":
        spec = tableau3D[:,3]
    
    # Upadted on 05/05/2023. I'm adding the solar spectrum in the name, to directly have the spectrum type
    name = name + " type_de_spectre:" +type_spec

    return Wl, spec, name

def open_Spec_Signal(name, nb_col):  
    """
    Open a spectral respond into a file
    Parameters
    ----------
    name : The name of a file
    nb_col : Int
        The number of read colone in the file.

    Returns
    -------
    Wl : array
        Wavelenght, must be in nm into the file
    spec : array
        The value present in the file, according the Wavelength
    name_f : string
        The name of the file open, with the number of the colomun used
        As :  name + " ,col n° " + str(nb_col)
    """

    # Initialise an empty table
    tableau3D = []
    try: 
        # Open the file in read-only mode
        file = open(name, "r")
        # use readlines to read all the lines of the file
        # The "lines" variable is a list with all the lines from the file
        lines = file.readlines()
        # Close the file after reading the lines
        file.close()
        
        # Make an iteration on the lines
        nb_line = len(lines)
        for i in range (nb_line):
            values = lines[i].split("\t")
            values[2] =values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)
            
    except FileNotFoundError:
        print("Le fichier n'a pas été trouvé.")
    # Transform the list into a numpy table
    tableau3D = np.array(tableau3D)
    # Extract wished datas   
    Wl = []  
    Wl = tableau3D[:,0]
    spec = []
    spec = tableau3D[:,nb_col]
    name_f = name + " ,col n° " + str(nb_col)
    return Wl, spec, name_f

def eliminate_duplicates(lst):
    """
    Remove duplicates from a list
    Exemple A = [1, 2, 3, 3, 4, 4, 5]
    B, C = eleminate_duplicates(A)
    B = [1, 2, 3, 4, 5]
    C = [3, 5]
    
    Parameters
    ----------
    lst : List of values

    Returns
    -------
    unique_elements : List
        List of values without duplica.
    indices_removed : list of indice removed values in the list
    """
    unique_elements = []
    indices_removed = []
      
    for i, element in enumerate(lst):
        if element not in unique_elements:
            unique_elements.append(element)
        else:
          indices_removed.append(i)
      
    return unique_elements, indices_removed

def write_stack_period (Subtrat, Mat_Periode, nb_periode):
    """
    Builds a stack by repeating a material period multiple times on top of a substrate.
        Exemple 1 : 
        Mat_Stack = write_stack_period(["BK7"], ["TiO2_I", "SiO2_I"], 3)
        Mat_Stack :  ['BK7', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I']
        Exemple 2:
        Mat_Stack = write_stack_period(["BK7", "TiO2", "Al2O3",], ["TiO2_I", "SiO2_I"], 2)
        Mat_Stack  : ['BK7', 'TiO2', 'Al2O3', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I']

    Parameters
    ----------
    Subtrat : List of string
         Each elements of this list is a string as valid material (a material with an associate texte files in Material/).
    Mat_Periode : List of string
        a list of a string. Each elements of this list is a string as valid material
    nb_periode : Int 
        the number of time were the Mat_Periode must be repeted

    Returns
    -------
    Subtrat : List of string
        DESCRIPTION.
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
        Returns a small list of y equidistant values from a large list lst
    Returns
    -------
    result : list
        Returns a small list of y equidistant values from a large list lst.

    """
    # Enables to return a small list of y equidistant values from a long list
    x = 5
    n = len(lst)
    interval = (n // (x -1))-1 # I substract 1 from the interval to avoid the out of range error
    result = [lst[i*interval] for i in range(x)]
    return result

def valeurs_equidistantes(liste, n=5):
    """
    From a large list, returns a small list with equidistant values

    Parameters
    ----------
    liste : list or array
        A large list 
    n : Int number, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    petite_liste : list or array
        Returns a small list of y equidistant values from a large list lst.
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
    Give a vector of Wavelength (in nm), optimized for selectif coating optimisation / calculation of performancess
        280 to 2500 nm (solar domain) with a 5 nm step for the calculation for performances
        2500 nm to 30µm (IR domain) with a 50 nm step for the calculation of thermal emissivity (named E_BB in this code)

    Returns
    -------
    Wl : array
        Wavelenght, in nm

    """
    Wl_1 = np.arange(280 , 2500 , 5)
    Wl_2 = np.arange(2500 , 30050 , 50)
    Wl = np.concatenate((Wl_1,Wl_2))
    return Wl

def exemple_evaluate(individual): 
    """
    Example of an evaluate function. The individual is a list. 
    The sum of squares of each term in the list is sought. 
    Example of evaluate function (= cost function) for a optimisation method
   
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


def Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack, parameters) :
    """
    For understand Individual_to_Stack work, we sudjet to run a lunche the main script with the following : 
    Mat_Stack : ['BK7', 'W-Al2O3', 'SiO2']
    Wl = np.arange(400 , 600, 50)
    
    Know note than the first thin layer is a composite layer, made of W and Al2O3 (BK7 is the stack):
    We need the refractive index of W AND Al2O3 for the layer 1, and we need to optimise the tickness AND volumic fraction in the W-Al2O3 layer 
    See EMA or Brugeman function for definition of volumuc fraction
    Each individual is now a array of lenght 6, as exemple : 
    individual : [1.00000000e+06, 40, 125, 0, 0.3, 0]
        The [1.00000000e+06, 40, 125] part of the list contain the thickness, in nm
        The [0, 0.3, 0] part of the list contain the volumic fraction, between 0 and 1
    k_Stack and n_Stack are array of float, of size (4, 3, 2), noted here (x, y, z) dimension
        x dimension is for wavelenght
        y dimension is for each layer
        z dimension is for stored the value of W AND Al2O3 with the same x and y dimension
        
    As exemple : 
    n_Stack : 
        array([[[1.5309    , 0.        ],
                [3.39      , 1.66518263],
                [1.48408   , 0.        ]],

               [[1.5253    , 0.        ],
                [3.30888889, 1.65954554],
                [1.479844  , 0.        ]],

               [[1.5214    , 0.        ],
                [3.39607843, 1.65544143],
                [1.476849  , 0.        ]],

               [[1.5185    , 0.        ],
                [3.5       , 1.65232045],
                [1.474652  , 0.        ]]])
    
    The purpose of Individual_to_Stack is to transform in such case the individual, n_Stack and k_Stack
    
    Parameters
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index 
    n_Stack : array 
        The real part of refractive index. 
        Can be of size (x, y, 2), with x the len of wavelenght and y the number of layer
    k_Stack : array
        The complexe part of refractive index
        Can be of size (x, y, 2), with x the len of wavelenght and y the number of layer
    Mat_Stack : List of string
        List of materials.
    parameters : Dict
        dictionary witch contain all parameters 

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
        The comlexe part of refractive index
        Must be size of (x, y) with x the len of wavelenght and y the number of layer  
    """
    
    # Add in work with vf(s)
    if 'nb_layer' in parameters:
        if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
            raise ValueError("It is not possible to work with theoretical and composite layers at the same time.")
    
    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        vf = []
        vf = individual[len(Mat_Stack):len(individual)]
        individual_list = individual.tolist()# Conversion is a list
        del individual_list[len(Mat_Stack):len(individual)]
        individual = np.array(individual_list)  # Conversion in a table
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
        vf= np.array(vf)
        n_Stack, k_Stack = Made_Stack_vf(n_Stack, k_Stack, vf)
    
    if 'nb_layer' in parameters:
        nb_layer = parameters.get('nb_layer')
        for i in range(nb_layer):
            # I check the value of the layer's index
            n = individual[nb_layer + len(Mat_Stack)]
            # I add the layer of n index and k = 0 to the Stack
            n_Stack = np.insert(n_Stack, len(Mat_Stack) + i, n, axis = 1)
            k_Stack = np.insert(k_Stack, len(Mat_Stack) + i, 0, axis = 1)
            index_to_remove = np.where(individual == n)[0][0]
            individual = np.delete(individual, index_to_remove)
        # As I did in previous versions, I transform d_Strack into an array
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
    else : 
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
    
    return d_Stack, n_Stack, k_Stack

def evaluate_R(individual, parameters):
    """
    Cost function for the average reflectivity at one or several wavelength
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        individual describe a stack of thin layers, substrat included. Each number are thickness in nm
        Exemple : [1000000, 100, 50, 120, 70] is a stack of 4 thin layers, respectivly of 100 nm, 50 nm, 120 nm and 70 nm
        The 70 nm thick layer is in contact with air
        The 100 nm thick layer is in contact with the substrat, here 1 mm thcik
        1 individual = 1 stack = 1 possible solution 
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    R_mean : Int (float)
        The average reflectance
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')
    # Creation of 
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack, parameters)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    R_mean = np.mean(R)
    return R_mean

def evaluate_T(individual, parameters):
    """
    Cost function for the average transmissivity at one or several wavelength
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    T_mean: Int (float)
        The average transmittance
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    
    # change 
    T_mean = np.mean(T)
    return T_mean

def evaluate_R_s(individual, parameters):
    """
    Calcul the solar reflectance of an individual
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    R_s : Int (float)
        The solar reflectance
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
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    # Calculation
    R_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    R_s = SolarProperties(Wl, R, Sol_Spec)
    return R_s

def evaluate_T_s(individual, parameters):
    """
    Calcul the solar transmittance of an individual
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    T_s : Int (float)
        The solar transmittance
    """
    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')#, 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    T_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_s = SolarProperties(Wl, T, Sol_Spec)
    return T_s

def evaluate_A_s(individual, parameters):
    """
    Calcul the solar absoptance of an individual
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    A_s : Int (float)
        The solar absoptance
    """

    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')#, 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    A_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    A_s = SolarProperties(Wl, A, Sol_Spec)
    return A_s

def evaluate_R_Brg(individual, parameters):
    """
    Cost function for a Bragg mirror
    Maximise the average reflectivity between 500 to 650 nm (defaut value)
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        individual describe a stack of thin layers, substrat included. Each number are thickness in nm
        Exemple : [1000000, 100, 50, 120, 70] is a stack of 4 thin layers, respectivly of 100 nm, 50 nm, 120 nm and 70 nm
        The 70 nm thick layer is in contact with air
        The 100 nm thick layer is in contact with the substrat, here 1 mm thcik
        1 individual = 1 stack = 1 possible solution 
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    R_mean : Int (float)
        The average reflectance
    """
    Wl = parameters.get('Wl')
    Ang = parameters.get('Ang')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')
    #Wl_targ = 550 
    # Creation of 
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack, parameters)

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
    Calculate the solar transmissivity WITH a PV cells signal
    With the following ligne code in the main script
    if evaluate.__name__ == "evaluate_T_PV":
        parameters["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec
    
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    T_PV: Int (float)
        Solar transmissivity WITH a PV cells signal
    """

    Wl = parameters.get('Wl')#,
    Ang = parameters.get('Ang')#
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec_with_PV')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    T_PV = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Sol_Spec is Sol_Spec_with_PV
    T_PV = SolarProperties(Wl, T, Sol_Spec)
    return T_PV

def evaluate_A_pv(individual, parameters):
    """
    Calculate the solar absoptivity WITH a PV cells signal
    With the following ligne code in the main script
    if evaluate.__name__ == "evaluate_T_PV":
        parameters["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec
    
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    T_PV: Int (float)
        Solar transmissivity WITH a PV cells signal
    """

    Wl = parameters.get('Wl')#,
    Ang = parameters.get('Ang')#
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec_with_PV') # Sol_Spec_with_pv = Sol_Spec * Signal_PV
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    T_PV = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Sol_Spec is Sol_Spec_with_PV
    A_PV = SolarProperties(Wl, A, Sol_Spec)
    return A_PV

def evaluate_T_vis(individual, parameters):
    """
    Calculate the optical transmittance with a human eye input
    The solar spectrum (Sol_Spec) has been remplaced by a human eye sensivity to wavelenght during the process
    See the following code lines in the main script
    
    Wl_H_eye , Signal_H_eye , name_H_eye = open_Spec_Signal('Materials/Human_eye.txt', 1)
    Signal_H_eye = np.interp(Wl, Wl_H_eye, Signal_H_eye) # Interpolate the signal
    
    parameters["Sol_Spec_with_Human_eye"] = Signal_H_eye 
    
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    T_PV: Int (float)
        Solar transmissivity WITH a PV cells signal
    """
    
    Wl = parameters.get('Wl')#,
    Ang = parameters.get('Ang')#
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec_Heye = parameters.get('Sol_Spec_with_Human_eye')
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    T_vis = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_vis = SolarProperties(Wl, T, Sol_Spec_Heye)
    return T_vis

def evaluate_low_e(individual, parameters):
    """
    Calculate the low_e performances
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    P_low_e: Int (float)
        Low_e performances 
    """
    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')#, 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # The profile is reflective from 0 to Lambda_cut_1
    # The profil is transparent from Lambda_cut_1 to + inf
    Lambda_cut_1 = parameters.get('Lambda_cut_2')
    d_Stack = np.array(individual)
    # Calculation of the domains 
    Wl_1 = np.arange(min(Wl),Lambda_cut_1,(Wl[1]-Wl[0]))
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)

    # Calculation of the RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calculation 
    # Transmitted solar flux on the Wl-1 part
    P_low_e = np.concatenate([T[0:len(Wl_1)],R[len(Wl_1):]])
    P_low_e = SolarProperties(Wl, P_low_e, Sol_Spec)
    
    return P_low_e

def evaluate_rh(individual, parameters):
    """
    Calculate the heliothermal efficiency 
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    rH: Int (float)
        Heliothermal efficiency 
    """
    
    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')
    C = parameters.get('C')
    T_air = parameters.get('T_air')
    T_abs = parameters.get('T_abs')
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # Integration of solar spectrum, raw en W/m2
    I =  trapz(Sol_Spec, Wl)
    # Creation of the stack
    d_Stack = np.array(individual)
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)

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
    rH = helio_th(A_s, E_BB_T_abs, T_abs, T_air, C, I,  r_Opt = 0.7, FFabs=1)
    
    return rH
    
def evaluate_RTR(individual, parameters):
    """
    Calculate the performance according a RTR shape
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    P_RTR: Int (float)
        performance according a RTR shape
    """  
    Wl = parameters.get('Wl')#
    Ang = parameters.get('Ang')#
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    # The profile is reflective from 0 to Lambda_cut_1
    Lambda_cut_1 = parameters.get('Lambda_cut_1')
    # The profile is transparent from Lambda_cut_1 to Lambda_cut_1
    Lambda_cut_1 = parameters.get('Lambda_cut_2')
    # Treatment of the optimization of the n(s)
    Mat_Stack = parameters.get('Mat_Stack')
    """
    Why Individual_to_Stack ?
    individual come from an optimization process, and must be transforme in d_Stack by the Individual_to_Stack function 
    1 individual ~ 1 list of thickness
    """
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    Wl_1 = np.arange(min(Wl),Lambda_cut_1+(Wl[1]-Wl[0]),(Wl[1]-Wl[0]))
    Wl_2 = np.arange(Lambda_cut_1, Lambda_cut_1+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    # Calculation of the RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    P_low_e = np.concatenate([R[0:len(Wl_1)],T[len(Wl_1):(len(Wl_2)+len(Wl_1)-1)], R[(len(Wl_2)+len(Wl_1)-1):]])
    P_RTR = SolarProperties(Wl, P_low_e, Sol_Spec)
    
    return P_RTR

def evaluate_netW_PV_CSP(individual, parameters):
    """
    Calculate the performance according a RTR shape
    1 individual = 1 output of one optimization function = 1 possible solution
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    P_RTR: Int (float)
        performance according a RTR shape
    """  
    
    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')#, 0)
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
  
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    # I calculate Rs
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Intégration du spectre solaire, brut en W/m2
    Sol_Spec_PV = Sol_Spec * Signal_PV 
    Sol_Spec_PV_int = trapz(Sol_Spec_PV, Wl) * poids_PV
    Sol_Spec_Th = Sol_Spec * Signal_Th
    Sol_Spec_Th_int = trapz(Sol_Spec_Th, Wl) 
    
    # Integration of the absorbed power by the PV
    Sol_Spec_T_PV = Sol_Spec * T * Signal_PV 
    Sol_Spec_T_PV_int = trapz(Sol_Spec_T_PV, Wl) * poids_PV
    
    # Integration of the absorbed power by the PV
    Sol_Spec_R_Th = Sol_Spec * R * Signal_Th
    Sol_Spec_R_Th_int = trapz(Sol_Spec_R_Th, Wl)
    
    net_PV_CSP = (Sol_Spec_T_PV_int + Sol_Spec_R_Th_int) / (Sol_Spec_PV_int + Sol_Spec_Th_int)
    return net_PV_CSP

def evaluate_RTA_s(individual, parameters):
    """
    Calcul the solar reflectance, the solar transmittance and the solar absoptance 
    for a ful spectrum
    Parameters
    ----------
    individual : array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary witch contain all parameters 

    Returns
    -------
    R_s : Float
        solar reflectanc
    T_s : Float
        solar transmittance
    A_s : Float
        solar absorptance
    """
    # Calculates the solar reflectance, solar transmittance and the absorptance
    # Every individual is a list of thickness. 
    # I set the variables Wl, Ang, n_Stack, k_Stack and SolSpec in global
    
    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')#, 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Sol_Spec = parameters.get('Sol_Spec')
    Mat_Stack = parameters.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    R_s, T_s, A_s = 0 , 0 , 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
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

def RTA_curve(individual, parameters):
    """
    Parameters
    ----------
    individual : numpy array
        individual is an output of optimisation method (algo)
        List of thickness in nm, witch can be added with volumic fraction or refractif index
    parameters : Dict
        dictionary with contain all "global" variables
    Returns
    -------
    R : List
        Reflectance of the stack, according the wavelenght list in the parameters
    T : List
        Transmittance of the stack, according the wavelenght list in the parameters
    A : List
        Absoptance of the stack, according the wavelenght list in the parameters
    """
    Wl = parameters.get('Wl')#, np.arange(280,2505,5))
    Ang = parameters.get('Ang')#, 0)
    n_Stack = parameters.get('n_Stack')
    k_Stack = parameters.get('k_Stack')
    Mat_Stack = parameters.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  parameters)
    
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    return R , T , A

def generate_population(chromosome_size, parameters):
    """
    See : function optimize_gn
    This function generates the 1st generation for the genetic optimization process. 
    That is, a series of thin film stacks, each thickness of which is within the range for genetic algo, optimize_gn'.

    Parameters
    ----------
    chromosome_size : Int 
        The lenght of individual, so the number of chromosone 
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    population : numpy array
        DESCRIPTION.
    """
    pop_size= parameters.get('pop_size')
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
            else : 
                individual += [np.random.randint(Th_range[0], Th_range[1])]
        population.append(individual)

    return population

def selection_min(population, evaluate, evaluate_rate, parameters):
    """
    Parameters
    ----------
    population : List of array 
        Population is a list of the different indivudals 
        Each individual is a stack, so a list of ticknesss
    evaluate : fonction
        the name of a evluatuib fonction (the cost function), defined previously. 
    evaluate_rate : Float
        DESCRIPTION.
    parameters : Dict
        "parameters" is a dictionary with contain all "global" variables
        
    Returns
    -------
    parents : TYPE
        DESCRIPTION.
    Uses the evaluate function to calculate indiduals performances according to a function
    In the evaluate program, if the fuction callvalue is evaluate_R_s, the code replaces "evaluate" by "evaluate_R_s"
    => the function name is adaptative ! 
    
    Select according to the minimum
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
    Selectionne selon le max 
    """
    scores = [evaluate(individual, parameters) for individual in population]
    parents = []
    for i in range(int(len(population)*evaluate_rate)):
        parent1 = population[scores.index(max(scores))]
        scores.pop(scores.index(max(scores)))
        parents.append(parent1)
    return parents

# New crossover version, by mask. # We totally mix the genes
def crossover(parents, crossover_rate , pop_size):
    """
    See : optimize_gn
    """
    children = []
    for i in range((pop_size-len(parents))//2): # We make two child for each parents
        parent1 = parents[np.random.randint(0,len(parents)-1)]
        parent2 = parents[np.random.randint(0,len(parents)-1)]
        if np.random.uniform(0, 1) < crossover_rate:
            mask = [np.random.choice([0, 1]) for _ in range(len(parent1))]
            child1 = [parent1[i] if mask[i] == 0 else parent2[i] for i in range(len(parent1))]
            child2 = [parent2[i] if mask[i] == 0 else parent1[i] for i in range(len(parent1))]
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
    See : optimize_gn

    This function enables the mutation of the childs (the new stacks), during their births.
    During his birth, a child has a % of chance (mutatin_rate) to mutate
    Some thicknesses vary about +/- mutation_delta
    Addition of an if loop to avoid a negative thickness
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
                    children[i][j + 1] += np.random.uniform(-mutation_delta, mutation_delta)
                    if children[i][j + 1] <= 0:
                        children[i][j + 1] = 0
    return children

def optimize_ga(evaluate, selection, parameters):
    """
    Parameters
    ----------
    evaluate : String
        Name of the evaluation fonction 
    selection : String
        Name of the selection fonction 

    Returns
    -------
    best_solution : numpy array
        The best stack of thn film (a list a thickness = individual) whitch provide the high cost function 
    dev : numpy array
        the value of the best solution during the optimisation process
    nb_run : Int 
        The number of epoch
    seed : Int
        Value of the seed, used in the random number generator

    """
    Mat_Stack = parameters.get('Mat_Stack')
    mod = parameters.get('Mod_Algo')
    pop_size = parameters.get('pop_size')
    crossover_rate = parameters.get('crossover_rate')
    evaluate_rate = parameters.get('evaluate_rate')
    mutation_rate = parameters.get('mutation_rate')
    mutation_delta = parameters.get('mutation_delta')
    Precision_AlgoG = parameters.get('Precision_AlgoG')
    nb_generation= parameters.get('nb_generation')
    d_Stack_Opt = parameters.get('d_Stack_Opt')

    # Seed 
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**31)
       np.random.seed(seed)
       
    np.random.seed(np.random.randint(1,2**31))
    
    # Settings of the optimization 
    population = np.zeros(0)
    dev = float("inf")
    dev_tab = []
    nb_run = 0
    chromosome_size = len(Mat_Stack) -1 # Number of thin layers
    population = generate_population(chromosome_size, parameters)
    
    if mod == "for":
        """
        The "for" mod launches the genetic algorithm for an accurate number of generations'
        """
        for i in range(nb_generation):
            parents = selection(population, evaluate, evaluate_rate, parameters)
            children = crossover(parents, crossover_rate, pop_size)
            children = mutation(children, mutation_rate, mutation_delta, d_Stack_Opt)
            population = parents + children
            scores = [evaluate(individual, parameters) for individual in population]
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
            scores = [evaluate(individual, parameters) for individual in population]
            dev = np.std(scores)
            dev_tab.append(dev)
            nb_run = nb_run + 1
    # End of the optimization
    scores = [evaluate(individual, parameters) for individual in population]
    #dev = np.std(scores)
    #dev = "{:.2e}".format(dev)
    

    # /!\ Can be a problem because we keep the minimum of the best scores here. 
    # But we can optimize by looking for the maximum.
    # But if the optimization is good, the minimum of the best scores should be equivalent to te maximum
    
    best_solution=population[scores.index(max(scores))]
    return best_solution, dev_tab, nb_run, seed


def optimize_strangle(evaluate, selection, parameters):
    """
    Parameters
    ----------
    evaluate : String
        Name of the evaluation fonction 
    selection : String
        Name of the selection fonction 

    Returns
    -------
    best_solution : numpy array
        The best stack of thn film (a list a thickness = individual) whitch provide the high cost function 
    dev : numpy array
        the value of the best solution during the optimisation process
    nb_run : Int 
        The number of epoch
    seed : Int
        Value of the seed, used in the random number generator

    """
    # I search for the variables in the settings
    Mat_Stack = parameters.get('Mat_Stack')
    mod = parameters.get('Mod_Algo')
    # Settings of the optimization 
    pop_size = parameters.get('pop_size')
    evaluate_rate = parameters.get('evaluate_rate')
    Precision_AlgoG = parameters.get('Precision_AlgoG')
    nb_generation= parameters.get('nb_generation')
    
    # Option 1 
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
    
    # Launch of the problem
    population = np.zeros(0)
    dev = float("inf")
    tab_dev = []
    chromosome_size = len(Mat_Stack) -1 # Number of thin layers
    population = generate_population(chromosome_size, parameters)
    if mod == "for":
        nb_run = 0
        for i in range(nb_generation):
            parents = selection(population, evaluate, evaluate_rate, parameters)
            children = children_strangle(pop_size, parents, chromosome_size)
            population = parents + children 
            scores = [evaluate(individual, parameters) for individual in parents]
            dev = np.std(scores)
            tab_dev.append(dev)
            nb_run = nb_run + 1
            # Final test optimization
    else:
        dev = float("inf")
        while dev > Precision_AlgoG:
            parents = selection(population, evaluate, evaluate_rate, parameters)
            children = children_strangle(pop_size, parents, chromosome_size)
            population = parents + children
            # Final test optimization
            scores = [evaluate(individual, parameters) for individual in parents]
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
    
    best_solution=population[scores.index(max(scores))]
    
    
    return best_solution, tab_dev, nb_run , seed

def children_strangle(pop_size, parents, chromosome_size):
    """
    See : the function optimize_strangle

    Parameters
    ----------
    pop_size : Int
        the number of individuals in the population.
    parents : List of array 
        List of individuals selected for be the parent of the next generation 
    chromosome_size : Int
        The length of each individual

    Returns
    -------
    children : List of array 
        List of individuals born from the parent 

    """
    children = []
    for i in range(pop_size-len(parents)):
        # 0 and 200 are, in nm, the ranges of the substrate
        individual = [1000000] 
        for j in range(chromosome_size): 
            min_values = min([sublist[j+1] for sublist in parents])
            max_values = max([sublist[j+1] for sublist in parents])
            individual = individual + [np.random.randint(min_values,max_values+1)]
        children.append(individual)
    return children

def DEvol(f_cout, f_selection, parameters):
    """
    Main author : A.Moreau, Photon team, University of Clermont Auvergne, France and Antoine Grosjean
    "This DE is a current to best. Hypertuned on the chirped problem 
    Abrupt elimination of individuals not respecting the bounds
    (compare with what happens if you just put back to the edge
     could be a good idea on some problems)"

    Parameters
    ----------
    evaluate : Collable 
        evaluation fonction, give in evaluate
    selection : Collable
        selection fonction, give in selection

    Returns
    -------
    best_solution : numpy array
        The best stack of thn film (a list a thickness = individual) whitch provide the high cost function 
    dev : numpy array
        the value of the best solution during the optimisation process
    nb_run : Int 
        The number of epoch
    seed : Int
        Value of the seed, used in the random number generator
    """
    selection = f_selection.__name__, 

    # DE settings - potential settings of the function
    cr = parameters.get('mutation_rate') #cr=0.5; # Probability to give parents settings to his child.
    f1 = parameters.get('f1') #f1=0.9;
    f2 = parameters.get('f2') #f2=0.8;
    
    #Following seed problem when using the code, the seed can be manually targeting 
    
    # Option 1 
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
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
    else : 
        nb_layer = 0 
    
    chromosome_size = len(Mat_Stack) + nb_layer -1 # Number of thin layers
    
    d_Stack_Opt = parameters.get('d_Stack_Opt')
    if isinstance(d_Stack_Opt, type(None)):
        d_Stack_Opt = ["no"] * chromosome_size
    
    X_min = [Th_Substrate]
    X_max = [Th_Substrate]
    for i in range(chromosome_size):
        if isinstance(d_Stack_Opt[i], (int, float)):
            X_min += [d_Stack_Opt[i]]
            X_max += [d_Stack_Opt[i]]
        else : 
            X_min += [Th_range[0]]
            X_max += [Th_range[1]]
         
    if 'n_plage' in parameters:
        n_range = parameters.get('n_plage')
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

    ############################# End of the code lines of COPS

    n=X_min.size

    # Initialization of the population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Random draw in the range between X_min and X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        # Change, because I usually want to maximize. In the other algorithms, a function
        # selection is used. 
        if selection[0] == "selection_min":
            cost[k]=f_cout(omega[k], parameters)
        elif selection[0] == "selection_max": 
            cost[k]=1-f_cout(omega[k], parameters)

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    # Initializations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])
    
    mutation_DE = parameters.get('mutation_DE')

    # DE loop
    while evaluation<budget-population:
        for k in range(0,population):
            crossover=(np.random.random(n)<cr)
            # *crossover+(1-crossover)*omega[k] : crossover step
            
            
            if mutation_DE == "current_to_best":
                # current to best
                # y(x) = x + F1 (a-b) + F2(best - x)
                X = (omega[k] + f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)])+f2*(best-omega[k]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_to_best": 
                # rand to best
                # y = c + F1 *(a-b) + F2(best - c)
                X = (omega[np.random.randint(population)] + f1*(omega[np.random.randint(population)]-omega[np.random.randint(population)])+f2*(best-omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_1": 
                #best 1 
                X = (best - f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_2": 
                # best 2 
                X = (best + f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)] + omega[np.random.randint(population)] - omega[np.random.randint(population)] ))*crossover+(1-crossover)*omega[k]
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
   
            if np.prod((X>=X_min)*(X<=X_max)):
                if selection[0] == "selection_min":
                    tmp=f_cout(X, parameters)
                elif selection[0] == "selection_max": 
                    tmp=1-f_cout(X, parameters)
                evaluation=evaluation+1
                if (tmp<cost[k]) :
                    cost[k]=tmp
                    omega[k]=X

        generation=generation+1
        #print('generation:',generation,'evaluations:',evaluation)
        #
        who=np.argmin(cost)
        best=omega[who]
        convergence.append(cost[who])

    convergence=convergence[0:generation+1]

    return [best,convergence,budget,seed]

def DEvol_Video(f_cout, f_selection, parameters):
    """ 
    Sub version of DE.
    Used by the main author of COPS for provide video of the optimization process
    The stack tickness is save during the process
    """
    selection = f_selection.__name__, 

# DE settings - pontential settings of the function
    cr = parameters.get('mutation_rate')
    #cr=0.5; # Probability to give parents settings to his child.
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
    #print(nb_generation)
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
    else : 
        nb_layer = 0 
    
    chromosome_size = len(Mat_Stack) + nb_layer -1 # Number of thin layers
    
    X_min = [Th_Substrate]
    X_max = [Th_Substrate]
    for i in range(chromosome_size):
         X_min += [Th_range[0]]
         X_max += [Th_range[1]]
         
    if 'n_plage' in parameters:
        n_range = parameters.get('n_plage')
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

    ############################# End of the code line of COPS

    n=X_min.size

    # Initialization of the population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Random draw in the range between X_min and X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        # Change, because I usually want to maximize. In the other algorithm, a function
        # selection is used. 
        if selection[0] == "selection_min":
            cost[k]=f_cout(omega[k], parameters)
        elif selection[0] == "selection_max": 
            cost[k]=1-f_cout(omega[k], parameters)

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    
    """ 
    Bug fix on the 27/06/2023. We shouldn't use .append but .copy
    Reprodiuce the bug
    
    best_tab= []
    best_tab.append(best)
    """
    #print(best)
    best_tab= np.copy(best)
    #print(best_tab)
    
    # Initializations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])
    
    mutation_DE = parameters.get('mutation_DE')

    # DE loop
    while evaluation<budget-population:
        for k in range(0,population):
            crossover=(np.random.random(n)<cr)

            if mutation_DE == "current_to_best":
                # current to best
                # y(x) = x + F1 (a-b) + F2(best - x)
                X = (omega[k] + f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)])+f2*(best-omega[k]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "rand_to_best": 
                # rand to best
                # y = c + F1 *(a-b) + F2(best - c)
                X = (omega[np.random.randint(population)] + f1*(omega[np.random.randint(population)]-omega[np.random.randint(population)])+f2*(best-omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_1": 
                #best 1 
                X = (best - f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)]))*crossover+(1-crossover)*omega[k]
            elif mutation_DE == "best_2": 
                # best 2 
                X = (best + f1*(omega[np.random.randint(population)] - omega[np.random.randint(population)] + omega[np.random.randint(population)] - omega[np.random.randint(population)] ))*crossover+(1-crossover)*omega[k]
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
            
            if np.prod((X>=X_min)*(X<=X_max)):
                if selection[0] == "selection_min":
                    tmp = f_cout(X, parameters)
                elif selection[0] == "selection_max": 
                    tmp = 1 - f_cout(X, parameters)
                evaluation=evaluation+1
                if (tmp<cost[k]) :
                    cost[k]=tmp
                    omega[k]=X

        generation=generation+1
        #print('generation:',generation,'evaluations:',evaluation)
        #
        who=np.argmin(cost)
        best=omega[who]
        """ 
        Old buggy version : 
        #best_tab.append(best)
        Bug fix.
        """
        best_tab = np.vstack((best_tab,best))
        #np.append(best_tab, best)
        convergence.append(cost[who])
        
    convergence=convergence[0:generation+1]

    return [best, best_tab, convergence, budget, seed]

class Particle():
    """
    Class Particle, for PSO (Particle Swarm Optimization) optimization 
    see function "PSO()"
    """
    def __init__(self, position, velocity, score_ref = 0):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_score = score_ref

def PSO(evaluate, selection, parameters):
    """
    PSO : particle swarm optimization 
    Need to work with class Particle()
    
    The proposed parameters for PSO are defaults values. They are NOT optimized for coatings optimization
        inertia_weight = 0.8
        cognitive_weight = 1.5
        social_weight = 1.5
        
    Parameters
    ----------
    evaluate : Collable 
        evaluation fonction, give in evaluate
    selection : Collable
        selection fonction, give in selection

    Returns
    -------
    best_solution : numpy array
        The best stack of thn film (a list a thickness = individual) whitch provide the high cost function 
    dev : numpy array
        the value of the best solution during the optimisation process
    nb_run : Int 
        The number of epoch
    seed : Int
        Value of the seed, used in the random number generator
    
    Need  generate_neighbor() and  acceptance_probability() functions 
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
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    # Creation of lower_bound and upper bound
    Th_range = parameters.get('Th_range')
    chromosome_size = len(Mat_Stack) -1 # Number of thin layers
    lower_bound = np.array([Th_range[0]] * chromosome_size) # Define lower bounds for each dimension
    lower_bound = np.insert(lower_bound, 0, Th_Substrate) # I add the thickness of the substrate in the bounds 
    upper_bound = np.array([Th_range[1]] * chromosome_size) # Define upper bounds for each dimension
    upper_bound = np.insert(upper_bound, 0, Th_Substrate) # I add the thickness of the substrate in the bounds 
    
    # Start
    num_dimensions = len(lower_bound)
    particles = []
    convergence = [] # List of best values durint the optimization process
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
            particle.position = np.clip(particle.position + particle.velocity, lower_bound, upper_bound)

            # Update of the global best score
            if selection[0] == "selection_min":
                score = evaluate(particle.position, parameters)
                # Update of the personal and the global best score
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position

                if score < global_best_score:
                    global_best_score = score
                    convergence.append(global_best_score) # Adding the newest best values 
                    global_best_position = particle.position
                
            if selection[0] == "selection_max": 
                score = evaluate(particle.position, parameters)
                # Update of the personal and the global best score
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position

                if score > global_best_score:
                    global_best_score = score
                    convergence.append(global_best_score) # Adding the newest best values 
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
    neighbor[index] = random.uniform(Th_range[0], Th_range[1])  # Choose a random value between -1 and 1 for the selected sublist
    return neighbor

def acceptance_probability(current_score, new_score, temperature):
    """
    Calculates the acceptance probability for the simulated annealing algorithm.

    Args:
        current_score: The current score or energy.
        new_score: The score or energy of the new state.
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
    evaluate : Collable 
        evaluation fonction, give in evaluate
    selection : Collable
        selection fonction, give in selection

    Returns
    -------
    best_solution : numpy array
        The best stack of thn film (a list a thickness = individual) whitch provide the high cost function 
    dev : numpy array
        the value of the best solution during the optimisation process
    nb_run : Int 
        The number of epoch
    seed : Int
        Value of the seed, used in the random number generator
    
    Need  generate_neighbor() and  acceptance_probability() functions 
    """
    # Stack : refractive index of the materials. Each colonne is a different layer. Each lign is a different wavelenght. Z axe (if present) is for mixture material
    Mat_Stack = parameters.get('Mat_Stack')
    Th_Substrate = parameters.get('Th_Substrate')
    # number of iteration of the annealing
    nb_generation = parameters.get('nb_generation')
    Th_range = parameters.get('Th_range')
    
    # Get the name of the selection function
    selection = selection.__name__,
    
    # Settings of the simulated annealing
    initial_temperature = 3000.0
    final_temperature = 0.01
    cooling_rate = 0.95
    current_temperature = initial_temperature
    
    # Seed fixation
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    # Creation of lower_bound and upper bound
    
    chromosome_size = len(Mat_Stack) - 1 # Number of thin layers
    # Generation of the initial solution
    current_solution = [random.uniform(Th_range[0], Th_range[1]) for _ in range(chromosome_size)]  # Generate a random solution
    current_solution = np.insert(current_solution, 0, Th_Substrate) # I add the thickness of the substrate between bounds
    
    # Initialization 
    best_solution = current_solution.copy()
    best_score = evaluate(best_solution, parameters)
    
    convergence = [] # List of best values durint the optimization process
    convergence.append(best_score)  # First best values 

    # Start of annealing
    while current_temperature > final_temperature:
        
        for _ in range(nb_generation):
            neighbor_solution = generate_neighbor(current_solution, parameters)
            
            # Evaluate the score of the neighbor according of 
            if selection[0] == "selection_min":
                neighbor_score = evaluate(neighbor_solution, parameters)
            elif selection[0] == "selection_max": 
                neighbor_score = 1- evaluate(neighbor_solution, parameters)
            
            neighbor_score = evaluate(neighbor_solution, parameters)

            if acceptance_probability(exemple_evaluate(current_solution), neighbor_score, current_temperature) > random.uniform(0, 1):
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
                    
        current_temperature *= cooling_rate
        
    #best_score : score (cost function) of the best solution
    return [best_solution, convergence, nb_generation, seed] 

def generate_mutant(solution, step_size):
    """
    Function for One_plus_One optimisation method
    
    Parameters
    ----------
    solution : TYPE
        DESCRIPTION.
    step_size : TYPE
        DESCRIPTION.

    Returns
    -------
    mutant : TYPE
        DESCRIPTION.

    """
    # Modification of the mutant start at 1 and not 0, because the 1st value is the substrat thickness, witch cannot be modified
    mutant = solution.copy()  # Copy of the initial solution
    mutant[1:] +=  np.random.normal(0, step_size, len(solution)-1)
    #return mutant.tolist()
    return mutant

def One_plus_One_ES(evaluate, selection, parameters):
    """
    The algorithm mentioned here is referred to as One_plus_One instead of (1+1) 
    because using (1+1)as a name for a function is not recommended. 
    However, it is important to note that the presented algorithm may not be the (1+1)-ES version.
    Although the algorithm presented here is (1+1)-ES, we cannot confirm with certainty 
    that it is theexact (1+1)-ES implementation based on information at our disposal 
    See P.Bennet thesis and or Nikolaus Hansen et al. Comparing results of 31 algorithms from the black-box optimization
    benchmarking BBOB-2009 | Proceedings of the 12th annual conference companion on Genetic
    and evolutionary computation. 2010.
    
    Main author : A.Moreau, Photon team, University of Clermont Auvergne, France and Antoine Grosjean
    "This DE is a current to best. Hypertuned on the chirped problem 
    Abrupt elimination of individuals not respecting the bounds
    (compare with what happens if you just put back to the edge
     could be a good idea on some problems)"

    Parameters
    ----------
    evaluate : Collable 
        evaluation fonction, give in evaluate
    selection : Collable
        selection fonction, give in selection

    Returns
    -------
    best_solution : numpy array
        The best stack of thn film (a list a thickness = individual) whitch provide the high cost function 
    dev : numpy array
        the value of the best solution during the optimisation process
    nb_run : Int 
        The number of epoch
    seed : Int
        Value of the seed, used in the random number generator
    """

    # Stack : refractive index of the materials. Each colonne is a different layer. Each lign is a different wavelenght. Z axe (if present) is for mixture material
    Mat_Stack = parameters.get('Mat_Stack')
    # Interation 
    pop_size = parameters.get('pop_size')
    nb_generation = parameters.get('nb_generation')
    #print(nb_generation)
    num_iterations = pop_size * nb_generation
    Th_Substrate = parameters.get('Th_Substrate')
    Th_range = parameters.get('Th_range')
    
    # Step size scaling factor
    step_size_factor = parameters.get('mutation_delta')
    
    # Get the selection function name
    selection = selection.__name__,
    
    # Parameter for (1+1)-ES 
    initial_step_size = 10  # Taille de pas initiale
    
    # Fixation of the seed
    if 'seed' in parameters:
        seed = parameters.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    # Creation of the solution
    
    chromosome_size = len(Mat_Stack) - 1 # Number of thin layers
    # Generation of the initial solution
    initial_solution = [random.uniform(Th_range[0], Th_range[1]) for _ in range(chromosome_size)]  # Generate a random solution
    initial_solution = np.insert(initial_solution, 0, Th_Substrate) # I add the thickness of the substrate between bounds      
    
    current_solution = initial_solution
    current_step_size = initial_step_size
    
    current_score = evaluate(current_solution, parameters)
    
    convergence = [] # List of best values durint the optimization process
    convergence.append(current_score)

    for _ in range(num_iterations):
        mutant_solution = generate_mutant(current_solution, current_step_size)
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

    return [current_solution, convergence, num_iterations, seed]
