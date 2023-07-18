# -*- coding: utf-8 -*-
"""
Created on 28 06 2023
@author: A.grosjean, A.soum-glaude, A.moreau & P.Bennet
contact : antoine.grosjean@epf.fr
"""

import numpy as np
import math
from scipy.integrate import trapz
import random
import datetime

def RTA3C(l, d, n, k, Ang=0):
    """_________________________________________________
    Exemple of RTA
    RTA is the key function : it's allow us to calcul reflectivity (R) transmissivity (T) and absorptivity(A)
    This function is a exemple working only with 3 thin layer on the substrat
    This tutorial functions is not used in the code
    """

    """
    La function RTA calcule la refléctivité, la transmissivité et l'absorptivité d'un ensemble de couches minces déposées sur un substrat
    tel que refléctivité + transmissivité + absorptivité  = 1 
    Le chiffre apres 'RTA' 'désigne le nombre de couche mince déposée sur le substrat INCLUT. Le code RTA 3C est donc pour 1 substrat et 2 couches minces max

    l'utilisation de numpy (np) permet un gain de temps sur le lancement
    => le gain de temps par rapport au lancement d'une fonction RTA écrite pour une seule longueur d'onde dans une boucle for est de l'ordre de 100 fois

    Arguments d'entrée : 
    l : la longueur d'onde en nm, de type vecteur
    d :  les épaisseurs du substrat et des couches mince en nm
    n :  les parties réel des indices de réfraction des matériaux. n est un tableau 2D avec le colone les indices de chaque couche et en ligne les longueurs d'onde
    k : les coefficients d'absorption soit les parties complexes des matériaux. k est un tableau 2D avec le colone les coef d'extinction de chaque couche et en ligne les longueurs d'onde
    Ang : l'angle d'incidence d'incidence du rayonnement en degres

    Sortie : 
    Refl, un vecteur colonne qui contient la réflectivité. Les indices en colone correspondent aux longueur d'onde'
    Trans, un vecteur colonne qui contient la transmivité. Les indices en colone correspondent aux longueur d'onde'
    Abs, un vecteur colonne qui contient l'absorptivité. Les indices en colone correspondent aux longueur d'onde'

    Test : 
    Ecire les variables suivante : 
    l = np.arange(600,750,100). Note que ici deux longueurs sont calculer : 600 et 700 nm
    d = np.array([[1000000, 150, 180]]). 1 mm de substrat, 150 nm de couche n°1, 180 de n°2 puit du vide (n=1, k=0)
    n = np.array([[1.5, 1.23,1.14],[1.5, 1.2,1.1]])
    k = np.array([[0, 0.1,0.05], [0, 0.1, 0.05]])
    Ang = 0 

    # Exécuter la fonction
    Refl, Trans, Abs = RTA3C(l, d, n, k , Ang)
    
    Pour la notation des indices n et k, comprendre que 
    @ 600 nm n = 1.5 pour le substrat , n = 1.23 pour la couche n°1 et n = 1.14 pour la couche n°2
    @ 700 nm n = 1.5 pour le substrat , n = 1.20 pour la couche n°1 et n = 1.1 pour la couche n°2
    @ 600 nm k = 0 pour le substrat , k = 0.1 pour la couche n°1 et k = 0.05 pour la couche n°2
    @ 700 nm k = 0 pour le substrat , n = 0.1 pour la couche n°1 et k = 0.05 pour la couche n°2

    Cela permet d'obtenir : Refl = array([0.00767694, 0.00903544]), Trans = array([0.60022613, 0.64313401]), Abs = array([0.39209693, 0.34783055])
    => La réflectivité vaut 0.00767694 (nombre entre 0 et 1) à 600 nm et 0.00903544 à 700 nm
    """
    # Add an air layer on top
    n = np.append(n, np.ones((len(l), 1)), axis=1) # remplacement de 2 par len(l)
    k = np.append(k, np.zeros((len(l), 1)), axis=1) # remplacement de 2 par len(l)
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
    nS = n[:,0] # Je prend la 1er colone, qui contient le n du substrat pour les longueurs d'onde
    kS = k[:,0]
    Ns = nS + 1j*kS
    PhiS = np.arcsin(N0*np.sin(Phi0)/Ns)
    qSPolaS = Ns*np.cos(PhiS)
    qSPolaP = Ns/np.cos(PhiS) # Ok jusque là 
    
    # Multilayers (layer 1 is the one closest to the substrate)
    nj= np.delete(n,0, axis=1)
    kj= np.delete(k,0, axis=1)
    dj= np.delete(d,0, axis=1)

    numlayers = nj.shape[1] # nj est un tableau 
    Nj = np.zeros((numlayers,1,len(l)), dtype=complex) # OK

    # colone dans Scilab, ligne ici
    Phij = np.zeros((numlayers,1,len(l)), dtype=complex)
    qjPolaS = np.zeros((numlayers,1,len(l)), dtype=complex)
    qjPolaP = np.zeros((numlayers,1,len(l)), dtype=complex)
    thetaj = np.zeros((numlayers,1,len(l)), dtype=complex)
    MpolaS = np.zeros((2, 2*numlayers,len(l)), dtype=complex)
    MpolaP = np.zeros((2, 2*numlayers,len(l)), dtype=complex)
    Ms = np.zeros((2, 2,len(l)), dtype=complex)
    Mp = np.zeros((2, 2,len(l)), dtype=complex)

    sous_tableaux = np.split(nj,nj.shape[1],axis=1)
    nj = np.array([el.reshape(1,len(l)) for el in sous_tableaux]) # el.reshape(1,2) devient el.reshape(1,len(l))
    sous_tableaux = np.split(kj,kj.shape[1],axis=1)
    kj = np.array([el.reshape(1,len(l)) for el in sous_tableaux])

    dj = np.squeeze(dj) #
    # Note  : inverser un tableau avec numpy.transpose()
    for LayerJ in range(numlayers): 
        Nj[LayerJ] = nj[LayerJ] + 1j * kj[LayerJ]
        Phij[LayerJ] = np.arcsin(N0 * np.sin(Phi0) / Nj[LayerJ])
        qjPolaS[LayerJ] = Nj[LayerJ] * np.cos(Phij[LayerJ])
        qjPolaP[LayerJ] = Nj[LayerJ] / np.cos(Phij[LayerJ])
        thetaj[LayerJ] = (2 * np.pi / l) * dj[LayerJ] * Nj[LayerJ] * np.cos(Phij[LayerJ]) # OK

        # Characteristic matrix of layer j
        """ Calcul of MpolaS"""
        MpolaS[0, 2*LayerJ] = np.cos(thetaj[LayerJ]) # Dans Scilab MpolaS(1,2*LayerJ-1)
        MpolaS[0, 2*LayerJ+1] = -1j/qjPolaS[LayerJ]*np.sin(thetaj[LayerJ]) # Dans Scilab MpolaS(1,2*LayerJ)
        MpolaS[1, 2*LayerJ] = -1j*qjPolaS[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaS[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        """ Calcul of MpolaP"""
        MpolaP[0, 2*LayerJ] = np.cos(thetaj[LayerJ])
        MpolaP[0, 2*LayerJ+1] = -1j/qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ] = -1j*qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        #print(MpolaS)
    
    # Global characteristic (transfer) matrix [Furman92, Andersson80]
    if numlayers == 1: # Substrat seul
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        Ms = M1s
        Mp = M1p
    elif numlayers == 2: # Substrat + 1 couche
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M2s = np.array([[MpolaS[0,2], MpolaS[0,3]], [MpolaS[1,2], MpolaS[1,3]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        M2p = np.array([[MpolaP[0,2], MpolaP[0,3]], [MpolaP[1,2], MpolaP[1,3]]])
        # Multiplication de matrice en gardant le 3eme axe (z dans un rep ortho, ici nommé l ) constant
        Ms = np.einsum('nkl,kml->nml', M2s, M1s)
        Mp = np.einsum('nkl,kml->nml', M2p, M1p)
    elif numlayers == 3: # Substrat + 2 couches
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
    TransSub = np.exp((-4*np.pi*kS*d[0])/l)
        
    # Transmittance of the substrate + multilayer stack
    Trans = TransMultilayer * TransSub
        
    # Power absorptance
    Abs = 1 - Refl - Trans
    return Refl, Trans, Abs

def RTA(l, d, n, k, Ang=0):
    """
    See the function RTA3C for a example / tutoral and the version of the function write for 3 layer (2 thin layer + the substrat)
    RTA calcul the reflectivity, transmissivty and absorptivity using Abélès matrices
    The Abélès matrices provide the best ratio accurency / speed for stack below 100 thin layers
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
    n = np.append(n, np.ones((len(l), 1)), axis=1)
    k = np.append(k, np.zeros((len(l), 1)), axis=1)
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
    nS = n[:,0] # Je prend la 1er colone, qui contient le n du substrat pour les longueurs d'onde
    kS = k[:,0]
    Ns = nS + 1j*kS
    PhiS = np.arcsin(N0*np.sin(Phi0)/Ns)
    qSPolaS = Ns*np.cos(PhiS)
    qSPolaP = Ns/np.cos(PhiS) # Ok jusque là 
    
    # Multilayers (layer 1 is the one closest to the substrate)
    nj= np.delete(n,0, axis=1)
    kj= np.delete(k,0, axis=1)
    dj= np.delete(d,0, axis=1)

    numlayers = nj.shape[1] # nj est un tableau 
    Nj = np.zeros((numlayers,1,len(l)), dtype=complex) # OK
    """Matrice tableau 3D ici. 
    l'axe "z" correspond aux différentes longueurs d'ondes """
    Phij = np.zeros((numlayers,1,len(l)), dtype=complex)
    qjPolaS = np.zeros((numlayers,1,len(l)), dtype=complex)
    qjPolaP = np.zeros((numlayers,1,len(l)), dtype=complex)
    thetaj = np.zeros((numlayers,1,len(l)), dtype=complex)
    MpolaS = np.zeros((2, 2*numlayers,len(l)), dtype=complex)
    MpolaP = np.zeros((2, 2*numlayers,len(l)), dtype=complex)
    Ms = np.zeros((2, 2,len(l)), dtype=complex)
    Mp = np.zeros((2, 2,len(l)), dtype=complex)
    """Redimensionnement de nj et kj
    """
    sous_tableaux = np.split(nj,nj.shape[1],axis=1)
    nj = np.array([el.reshape(1,len(l)) for el in sous_tableaux])
    sous_tableaux = np.split(kj,kj.shape[1],axis=1)
    kj = np.array([el.reshape(1,len(l)) for el in sous_tableaux])
    """ Transforme un vecteur (1,3) en vecteur (3,)
    """
    dj = np.squeeze(dj) #    
    for LayerJ in range(numlayers): 
        Nj[LayerJ] = nj[LayerJ] + 1j * kj[LayerJ]
        Phij[LayerJ] = np.arcsin(N0 * np.sin(Phi0) / Nj[LayerJ])
        qjPolaS[LayerJ] = Nj[LayerJ] * np.cos(Phij[LayerJ])
        qjPolaP[LayerJ] = Nj[LayerJ] / np.cos(Phij[LayerJ])
        thetaj[LayerJ] = (2 * np.pi / l) * dj[LayerJ] * Nj[LayerJ] * np.cos(Phij[LayerJ]) # OK
        """Changement par rapport à Scilab, du au index de Python. La 1er case est noté 0,0 dans Python et 
        1,1 dans Scilab. Ici LayerJ commence à 0 et non plus à 1 mais l'arret de la boucle for reste le même (dernier interval 
        exclus dans Python.
        Chaque index x de Mpola doit être réduit de 1. L'Index y doit être augmenter de  +1 le LayerJ-1 devient LayerJ 
        t LayerJ deient LayerJ+1 
        """
        # Characteristic matrix of layer j
        """ Calcul de MpolaS"""
        MpolaS[0, 2*LayerJ] = np.cos(thetaj[LayerJ]) # Dans Scilab MpolaS(1,2*LayerJ-1)
        MpolaS[0, 2*LayerJ+1] = -1j/qjPolaS[LayerJ]*np.sin(thetaj[LayerJ]) # Dans Scilab MpolaS(1,2*LayerJ)
        MpolaS[1, 2*LayerJ] = -1j*qjPolaS[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaS[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        """ Calcul de MpolaP"""
        MpolaP[0, 2*LayerJ] = np.cos(thetaj[LayerJ])
        MpolaP[0, 2*LayerJ+1] = -1j/qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ] = -1j*qjPolaP[LayerJ]*np.sin(thetaj[LayerJ])
        MpolaP[1, 2*LayerJ+1] = np.cos(thetaj[LayerJ])
        #print(MpolaS)
    
    # Global characteristic (transfer) matrix [Furman92, Andersson80]
    if numlayers == 1: # Substrat seul
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        Ms = M1s
        Mp = M1p
    else : # The ultime code =D
        M1s = np.array([[MpolaS[0,0], MpolaS[0,1]], [MpolaS[1,0], MpolaS[1,1]]])
        for i in range(numlayers):
            exec(f"M{i + 1}s = np.array([[MpolaS[0,{i * 2}], MpolaS[0,{i * 2 + 1}]], [MpolaS[1,{i * 2}], MpolaS[1,{i * 2 + 1}]]])")
        # Calcul des élèmens de Mp
        M1p = np.array([[MpolaP[0,0], MpolaP[0,1]], [MpolaP[1,0], MpolaP[1,1]]])
        for i in range(numlayers):
            exec(f"M{i + 1}p = np.array([[MpolaP[0,{i * 2}], MpolaP[0,{i * 2 + 1}]], [MpolaP[1,{i * 2}], MpolaP[1,{i * 2 + 1}]]])")
        # Calcul de Ms et Mp 
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
    TransSub = np.exp((-4*np.pi*kS*d[0])/l)
        
    # Transmittance of the substrate + multilayer stack
    Trans = TransMultilayer * TransSub
        
    # Power absorptance
    Abs = 1 - Refl - Trans
    return Refl, Trans, Abs

def nb_compo(Mat_Stack):
    """
    Renvoie le nombre de couche mince composite, c'est à dire composé de deux matériaux, comme un cermet ou un matériaux poraux
    une couche mince composite comprend le tiret du 6 - dans sa chaine de caractère
    Exemple : 'W-Al2O3' => couche composite de W est de Al2O3, ici de type cermet
              ' air-SiO2' =>  couche composite d'auir est de SiO2, ici de type poreux
    """
    nb = 0 
    for i in Mat_Stack: 
        if "-" in i:
            nb += 1
    return nb

def Made_Stack(Mat_Stack, Wl):
    """
    This key fonction strat with a list a material with describe the stack 
    It's return two table numpy array, on for the real part of the refractive index (n), and the other for the imaginary part (k) 

    Parameters
    ----------
    Mat_Stack : List of string, with containt each material of the stack. See as exemple the function Ecrit_Stack_Periode
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
    # Création du Stack
    # Je recherche si le nom d'un matériaux d'une couche mince est séparer par un tiret du 6 -
    # Si oui, c'est un matériaux composite 
    no_dash = True
    for s in Mat_Stack: 
        if "-" in s:
            no_dash = False
            break
        
    if no_dash : # Si no_dask est true, je rentre dans la boucle
        n_Stack = np.zeros((len(Wl),len(Mat_Stack)))
        k_Stack = np.zeros((len(Wl),len(Mat_Stack)))
        for i in range(len(Mat_Stack)):
            Wl_mat, n_mat, k_mat = open_material(Mat_Stack[i])    
            # Interpolation 
            n_mat = np.interp(Wl,Wl_mat, n_mat)
            k_mat = np.interp(Wl,Wl_mat, k_mat)
            n_Stack[:,i] = n_mat[:,]
            k_Stack[:,i] = k_mat[:,]
    
        return n_Stack, k_Stack
    
    else : # sinon, il doit y avoir un -, donc deux matériaux 
        n_Stack = np.zeros((len(Wl),len(Mat_Stack),2))
        k_Stack = np.zeros((len(Wl),len(Mat_Stack),2))
        for i in range(len(Mat_Stack)):
            # J'ouvre le 1er matériaux 
            list_mat = []
            list_mat = Mat_Stack[i].split("-")
            if len(list_mat) == 1: 
                # la liste contient un matériaux. Je charge comme d'hab
                # ligne: longueur d'onde, colone : indice des matériaux 
                Wl_mat, n_mat, k_mat = open_material(Mat_Stack[i])    
                # Interpolation 
                n_mat = np.interp(Wl,Wl_mat, n_mat)
                k_mat = np.interp(Wl,Wl_mat, k_mat)
                n_Stack[:,i,0] = n_mat[:,]
                k_Stack[:,i,0] = k_mat[:,]
            if len(list_mat) == 2: 
                # la liste contient deux matériaux. Je place le second sur l'axe z=2
                Wl_mat, n_mat, k_mat = open_material(list_mat[0])    
                # Interpolation 
                n_mat = np.interp(Wl,Wl_mat, n_mat)
                k_mat = np.interp(Wl,Wl_mat, k_mat)
                n_Stack[:,i,0] = n_mat[:,]
                k_Stack[:,i,0] = k_mat[:,]    
                # Ouverture du second materiaux 
                Wl_mat, n_mat, k_mat = open_material(list_mat[1])    
                # Interpolation 
                n_mat = np.interp(Wl, Wl_mat, n_mat)
                k_mat = np.interp(Wl, Wl_mat, k_mat)
                n_Stack[:,i,1] = n_mat[:,]
                k_Stack[:,i,1] = k_mat[:,]      
        return n_Stack, k_Stack

def Made_Stack_vf(n_Stack, k_Stack, vf=[0]):
    """
    n_Stack_vf, ou k_stack_vf veut dire un n et k calculé par une fonction de Bruggeman (loi de mélange EMA)
    Ce sont les valeurs à injecter dans RTA
    Si vf = 0 pour tout les matériaux, alors n_Stack_vf = n_Stack (idem pour k)
    Sinon il faut le calculer 
    Parameters
    ----------
    n_Stack : array, in 2 or 3 dimensional
        DESCRIPTION.
    k_Stack : array, in 2 or 3 dimensional
        DESCRIPTION.
    vf : TYPE, optional
        DESCRIPTION. The default is [0].

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if all(elem == 0 for elem in vf): #  (vf ==0).all():
        """ Tout les vf = 0. Il n'est pas nécessaire de lancer Bruggman. 
        """
        return n_Stack, k_Stack
    else : 
        """ vf.all == [0] n'est pas True. Au moin un vf existe, c-a-d qu'une couche de l'empillement 
        est composé de deux matériaux. n_Stack et k_Stack sont des tableaux 3D.Par exemple pour "W-Al2O3"
        les données de W sont dans les tranches [:,:,0] et les données de Al2O3 sont dans les tranches [:,:,1]
        """
        n_Stack_vf = np.empty((n_Stack.shape[0], np.shape(n_Stack)[1]))
        k_Stack_vf = np.empty((k_Stack.shape[0], np.shape(k_Stack)[1]))
        # ancienne version 
        #n_Stack_vf = []
        #k_Stack_vf = []
        #n_Stack_vf = np.array(n_Stack_vf)
        #k_Stack_vf = np.array(k_Stack_vf)
        for i in range (np.shape(k_Stack)[1]):
            #  Je parcours chaque couche et récupérer le n et k de la matrice (M) et des inclusions (I)
            # Si la couche est d'un seul matériaux les données sont dans nI et kI
            # => nM et km sont alors plein de zéro
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
    Fonction de Bruggemann. 
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

def BB(T, l):
    """
    Parameters
    ----------
    T : Int
        Black Body Temperature , in Kelvin
    l : 1D vector, array of int
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
    BB_spec = 2 * np.pi * h * c**2 / (l * 1e-9)**5 / (np.exp(h * c / (l * 1e-9 * k * T)) - 1) * 1e-9
    return BB_spec

def SolarProperties(l, R, SolSpec):
    """
    Parameters
    ----------
    R : array
        Stack Optical Properties, for different Wavelength, properly intepoled
        Not than R is not necessary the reflectivity, can be transmissivity or absorptivity
    L : array
        Wavelength, in nm
    SolSpec : Vector. SolarSpectrum used, properly intepoled in W.m2nm-1
    R and SolSpec must have the same length
    Returns
    -------
    R_s : float
        Solar Properties, accorting to the Strack Optical Properties
        => not necessary Solar Reflectance. 
    """
    if len(l) != len(R) or len(l) != len(SolSpec) or len(R) != len(SolSpec):
        raise ValueError("Vectors l, R, and SolSpec must have the same length.")
    try:
        R_Spec = []
        R_Spec = R * SolSpec
        SolSpec_int = trapz(SolSpec, l)
        R_int = trapz(R_Spec, l)
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
    try :
        type(name) == "str"        
        # Initialiser un tableau vide
        tableau3D = []
        name="Materials/" + name + ".txt"
        try: 
            # Ouvrir le fichier en lecture seule
            file = open(name, "r")
            # utiliser readlines pour lire toutes les lignes du fichier
            # La variable "lignes" est une liste contenant toutes les lignes du fichier
            lines = file.readlines()
            # fermez le fichier après avoir lu les lignes
            file.close()
            
            # Itérer sur les lignes
            nb_line = len(lines)
            for i in range (nb_line):
                values = lines[i].split("\t")
                values[2] =values[2].rstrip("\n")
                values = [float(val) for val in values]
                tableau3D.append(values)
                
        except FileNotFoundError:
            print("Le fichier n'a pas été trouvé.")
        # Transformer la liste en tableau numpy
        tableau3D = np.array(tableau3D)
        
    except FileNotFoundError:
        print("Le nom du fichier n'est pas un string.")
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
    # Initialiser un tableau vide
    tableau3D = []
    try: 
        # Ouvrir le fichier en lecture seule
        file = open(name, "r")
        # utiliser readlines pour lire toutes les lignes du fichier
        # La variable "lignes" est une liste contenant toutes les lignes du fichier
        lines = file.readlines()
        # fermez le fichier après avoir lu les lignes
        file.close()
        
        # Itérer sur les lignes
        nb_line = len(lines)
        for i in range (nb_line):
            values = lines[i].split("\t")
            values[2] =values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)
            
    except FileNotFoundError:
        print("Le fichier n'a pas été trouvé.")
    # Transformer la liste en tableau numpy
    tableau3D = np.array(tableau3D)
    # Extraire les données que l'on souhaite   
    Wl = []  
    Wl = tableau3D[:,0]
    spec = []
    if type_spec == "DC":
        spec = tableau3D[:,1]
    if type_spec == "Extr":
        spec = tableau3D[:,2]
    if type_spec == "GT":
        spec = tableau3D[:,3]
    
    # Mise à jour du 05/05/2023. Je rajoute le type de spectre solaire au nom, pour avoir le type de spectre
    name = name + " type_de_spectre:" +type_spec

    return Wl, spec, name

def open_Spec_Signal(name, nb_col):     
    """

    """
    # Initialiser un tableau vide
    tableau3D = []
    try: 
        # Ouvrir le fichier en lecture seule
        file = open(name, "r")
        # utiliser readlines pour lire toutes les lignes du fichier
        # La variable "lignes" est une liste contenant toutes les lignes du fichier
        lines = file.readlines()
        # fermez le fichier après avoir lu les lignes
        file.close()
        
        # Itérer sur les lignes
        nb_line = len(lines)
        for i in range (nb_line):
            values = lines[i].split("\t")
            values[2] =values[2].rstrip("\n")
            values = [float(val) for val in values]
            tableau3D.append(values)
            
    except FileNotFoundError:
        print("Le fichier n'a pas été trouvé.")
    # Transformer la liste en tableau numpy
    tableau3D = np.array(tableau3D)
    # Extraire les données que l'on souhaite   
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

def Write_Stack_Periode (Subtrat, Mat_Periode, nb_periode):
    """
    Subtrat : a list of a string. Each elements of this list is a string as valid material (a material with an associate texte files in Material/)
    see : open_material fonction 
    Mat Periode : a list of a string. Each elements of this list is a string as valid material
    nb_periode : the number of time were the Mat_Periode must be repeted
    
    Write the stack 
    Exemple 1 : 
    Mat_Stack = Ecrit_Stack_Periode(["BK7"], ["TiO2_I", "SiO2_I"], 3)
    Mat_Stack :  ['BK7', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I']
    
    Exemple 2:
    Mat_Stack = Ecrit_Stack_Periode(["BK7", "TiO2", "Al2O3",], ["TiO2_I", "SiO2_I"], 2)
    Mat_Stack  : ['BK7', 'TiO2', 'Al2O3', 'TiO2_I', 'SiO2_I', 'TiO2_I', 'SiO2_I']
    """
    for i in range(nb_periode):
        Subtrat += Mat_Periode 
    return Subtrat

def equidistant_values(lst):
    # Permet de renvoyer une petite liste result de y valeur équidistante à partir d'une grande liste lst
    x = 5
    n = len(lst)
    interval = (n // (x -1))-1 # je retire 1 à l'interval, pour éviter d'être out of range
    result = [lst[i*interval] for i in range(x)]
    return result

def valeurs_equidistantes(liste, n=5):
    """
    A partir d'une grande liste, renvoie une petite liste avec des valeurs équidistante'
    Parameters
    ----------
    liste : TYPE
        Une grande liste.
    n : Int number, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    petite_liste : TYPE : List

    """
    # déterminer la distance entre chaque valeur
    distance = len(liste) / (n - 1)
    # initialiser la petite liste
    petite_liste = [liste[0]]
    # ajouter les valeurs équidistantes à la petite liste
    for i in range(1, n - 1):
        index = int(i * distance)
        petite_liste.append(liste[index])
    petite_liste.append(liste[-1])
    # renvoyer la petite liste
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
    Exemple de fonction evaluate. L'individu est une liste. 
    On cherche la somme des carrés de chaque termes de la liste. 
    Fontion d'évaluation d'exemple pour un algo génétique
    """
    # convertir une liste en array np.array(population[1])
    score = 0
    for sub_list in individual:
        score += sub_list*sub_list
    return score

def Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack, conteneur) :
    
    # Ajout dans le travail avec des vf
    if 'nb_layer' in conteneur:
        if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
            raise ValueError("Il n'est pas possible de travailler avec des couches théorique et composite en même temps")
    
    if len(n_Stack.shape) == 3 and n_Stack.shape[2] == 2:
        vf = []
        vf = individual[len(Mat_Stack):len(individual)]
        individual_list = individual.tolist()  # Conversion en liste
        del individual_list[len(Mat_Stack):len(individual)]
        individual = np.array(individual_list)  # Conversion en tableau
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
        vf= np.array(vf)
        n_Stack, k_Stack = Made_Stack_vf(n_Stack, k_Stack, vf)
    
    if 'nb_layer' in conteneur:
        nb_layer = conteneur.get('nb_layer')
        for i in range(nb_layer):
            # Je vais chercher la valeur de l'indice de la couche
            n = individual[nb_layer + len(Mat_Stack)]
            # J'ajoute au Stack la couche d'indice n et k = 0
            n_Stack = np.insert(n_Stack, len(Mat_Stack) + i, n, axis = 1)
            k_Stack = np.insert(k_Stack, len(Mat_Stack) + i, 0, axis = 1)
            index_to_remove = np.where(individual == n)[0][0]
            individual = np.delete(individual, index_to_remove)
        # Comme dans les versions précédante, je transforme d_Strack en array
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
    else : 
        d_Stack = np.array(individual)
        d_Stack = d_Stack.reshape(1, len(individual))
    
    return d_Stack, n_Stack, k_Stack

def evaluate_R(individual, conteneur):
    """
    Cost function for the average reflectivity at one or several wavelength
    """
    Wl = conteneur.get('Wl')
    Ang = conteneur.get('Ang')
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Mat_Stack = conteneur.get('Mat_Stack')
    # Creation of 
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack, conteneur)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    R_mean = np.mean(R)
    return R_mean

def evaluate_T(individual, conteneur):
    """
    Cost function for the average transmissivity at one or several wavelength
    """
    Wl = conteneur.get('Wl')
    Ang = conteneur.get('Ang')
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)

    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    
    # change 
    T_mean = np.mean(T)
    return T_mean

def evaluate_R_s(individual, conteneur):
    """
    Parameters
    ----------
    individual : TYPE
        DESCRIPTION.
    conteneur : TYPE
        DESCRIPTION.

    Returns
    -------
    R_s : TYPE
        DESCRIPTION.

    """
    Wl = conteneur.get('Wl')
    Ang = conteneur.get('Ang')
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    # Je calcul Rs
    R_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    R_s = SolarProperties(Wl, R, Sol_Spec)
    return R_s

def evaluate_T_s(individual, conteneur):
    
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    T_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_s = SolarProperties(Wl, T, Sol_Spec)
    return T_s

def evaluate_A_s(individual, conteneur):
    """
    Cost function for the solar absorptance
    """

    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    A_s = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    A_s = SolarProperties(Wl, A, Sol_Spec)
    return A_s

def evaluate_T_pv(individual, conteneur):
    """
    Calculate the solar transmissivity WITH a PV cells signal
    With the following ligne code in the main script
    
    if evaluate.__name__ == "evaluate_T_PV":
        conteneur["Sol_Spec_with_PV"] = Signal_PV * Sol_Spec
    """

    Wl = conteneur.get('Wl')#,
    Ang = conteneur.get('Ang')#
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('Sol_Spec_with_PV')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    T_s_PV = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_s_PV = SolarProperties(Wl, T, Sol_Spec)
    return T_s_PV

def evaluate_T_vis(individual, conteneur):
    """
    Calculate the optical transmittance with a human eye input
    The solar spectrum (Sol_Spec) has been remplaced by a human eye sensivity to wavelenght during the process
    See the following code lines in the main script
    
    Wl_H_eye , Signal_H_eye , name_H_eye = open_Spec_Signal('Materials/Human_eye.txt', 1)
    Signal_H_eye = np.interp(Wl, Wl_H_eye, Signal_H_eye) # Interpolate the signal
    
    conteneur["Sol_Spec_with_Human_eye"] = Signal_H_eye 
    """
    Wl = conteneur.get('Wl')#,
    Ang = conteneur.get('Ang')#
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('Sol_Spec_with_Human_eye')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    T_H_eye = 0
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    T_H_eye = SolarProperties(Wl, T, Sol_Spec)
    return T_H_eye

def evaluate_low_e(individual, conteneur):
    
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    # Le profil est réflecteur de 0 à Lambda_cut_min
    # Le profil est transparrant de Lambda_cut_min à + inf
    Lambda_cut = conteneur.get('Lambda_cut')
    d_Stack = np.array(individual)
    # Calcul des domaines 
    Wl_1 = np.arange(min(Wl),Lambda_cut,(Wl[1]-Wl[0]))
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)

    # Calcul du RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calcul 
    # Transmitted solar flux on the Wl-1 part
    P_low_e = np.concatenate([T[0:len(Wl_1)],R[len(Wl_1):]])
    P_low_e = SolarProperties(Wl, P_low_e, Sol_Spec)
    
    return P_low_e

def evaluate_rh(individual, conteneur):

    
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')
    C = conteneur.get('C')
    Tair = conteneur.get('T_air')
    Tabs = conteneur.get('T_abs')
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    # Intégration du spectre solaire, brut en W/m2
    I =  trapz(Sol_Spec, Wl)
    # Creation du stack
    d_Stack = np.array(individual)
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)

    # Calcul du RTA
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calcul de l'absoptance solaire 
    A_s = 0 
    A_s = SolarProperties(Wl, A, Sol_Spec)
    # Calcul du corps noir
    BB_shape = BB(Tabs, Wl)
    # calcul de l'émittance du revetement à uhe corps noir
    E_BB_Tabs = E_BB(Wl, A, BB_shape)
    
    # Calcul du rendement héliothermique. Argument de la fonction helio_th(A_s, E_BB, T_stack, T_air, C, I,  r_Opt = 0.7, FFabs=1):
    rH = helio_th(A_s, E_BB_Tabs, Tabs, Tair, C, I,  r_Opt = 0.7, FFabs=1)
    
    return rH
    
def evaluate_RTR(individual, conteneur):
    # Calcul la reflectance solaire
    #Chaque individu est une liste d'épaisseur. 
    #Je met les variables Wl, Ang, n_Stack, k_Stack et SolSpec sont en global
    
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    # Le profil est réflecteur de 0 à Lambda_cut_min
    Lambda_cut_min = conteneur.get('Lambda_cut_min')
    # Le profil est transparrant de Lambda_cut_min à Lambda_cut
    Lambda_cut = conteneur.get('Lambda_cut')
    # traitemement de l'optimisation des n
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    # Calcul des domaines 
    # Va de min à Lambda_cut_min inclu
    Wl_1 = np.arange(min(Wl),Lambda_cut_min+(Wl[1]-Wl[0]),(Wl[1]-Wl[0]))
    Wl_2 = np.arange(Lambda_cut_min, Lambda_cut+(Wl[1]-Wl[0]), (Wl[1]-Wl[0]))
    # Calcul du RTA
    d_Stack = d_Stack.reshape(1, len(individual))
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Calcul 
    # Transmitted solar flux on the Wl-1 part
    # ancienne version fausse. Correction au 27/04/2023
    #P_low_e = np.concatenate([R[0:len(Wl_1)],T[len(Wl_1):len(Wl_2)], R[len(Wl_2):]])
    P_low_e = np.concatenate([R[0:len(Wl_1)],T[len(Wl_1):(len(Wl_2)+len(Wl_1)-1)], R[(len(Wl_2)+len(Wl_1)-1):]])
    P_low_e = SolarProperties(Wl, P_low_e, Sol_Spec)
    
    return P_low_e

def evaluate_netW_PV_CSP(individual, conteneur):
    """
    Parameters
    ----------
    individual : 1D array, like a list
        individual describe a stack of thin layers, substrat included. Each number are thickness in nm
        Exemple : [1000000, 100, 50, 120, 70] is a stack of 4 thin layers, respectivly of 100 nm, 50 nm, 120 nm and 70 nm
        The 70 nm thick layer is in contact with air
        The 100 nm thick layer is in contact with the substrat, here 1 mm thcik
        1 individual = 1 stack = 1 possible solution 
    conteneur : Dict
        Contain all different data in a dictionaire 
    Returns
    -------
    Net_W
    """
    
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')

    # traitemement de l'optimisation des n
    Mat_Stack = conteneur.get('Mat_Stack')
    
    """Get the "cost of PV". We need to give more importance to the PV part. Without that, the optimization process not provide
    a RTR like coating, but a near perfect mirror
    Without cost of PV the best coating a dielectric mirror, witch reflected all the sun light without transmited solar flux to the PV cells
    """
    # PV part
    poids_PV = conteneur.get('poids_PV')
    Signal_PV = conteneur.get('Signal_PV')
    # Thermal part
    Signal_Th = conteneur.get('Signal_Th')
  
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    # Je calcul Rs
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    # Intégration du spectre solaire, brut en W/m2
    Sol_Spec_PV = Sol_Spec * Signal_PV 
    Sol_Spec_PV_int = trapz(Sol_Spec_PV, Wl) * poids_PV
    Sol_Spec_Th = Sol_Spec * Signal_Th
    Sol_Spec_Th_int = trapz(Sol_Spec_Th, Wl) 
    
    #Intégration de la puissance absorbé par le PV
    Sol_Spec_T_PV = Sol_Spec * T * Signal_PV 
    Sol_Spec_T_PV_int = trapz(Sol_Spec_T_PV, Wl) * poids_PV
    
    # Intégration de la puissance absorbé par le PV
    Sol_Spec_R_Th = Sol_Spec * R * Signal_Th
    Sol_Spec_R_Th_int = trapz(Sol_Spec_R_Th, Wl)
    
    net_PV_CSP = (Sol_Spec_T_PV_int + Sol_Spec_R_Th_int) / (Sol_Spec_PV_int + Sol_Spec_Th_int)
    return net_PV_CSP

def evaluate_RTA_s(individual, conteneur):
    # Calcul la reflectance solaire, transmittance solaire et l'absorptance
    #Chaque individu est une liste d'épaisseur. 
    #Je met les variables Wl, Ang, n_Stack, k_Stack et SolSpec sont en global
    
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Sol_Spec = conteneur.get('SolSpec')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
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

def RTA_curve(individual, conteneur):
    """
    Parameters
    ----------
    individual : numpy array
        individual is a stack, a list a thickness.
    conteneur : Dict
        dictionary with contain all "global" variables
    Returns
    -------
    R : List
        Reflectance of the stack, according the wavelenght list in the conteneur
    T : List
        Transmittance of the stack, according the wavelenght list in the conteneur
    A : List
        Absoptance of the stack, according the wavelenght list in the conteneur
    """
    Wl = conteneur.get('Wl')#, np.arange(280,2505,5))
    Ang = conteneur.get('Ang')#, 0)
    n_Stack = conteneur.get('n_Stack')
    k_Stack = conteneur.get('k_Stack')
    Mat_Stack = conteneur.get('Mat_Stack')
    
    d_Stack, n_Stack, k_Stack = Individual_to_Stack(individual, n_Stack, k_Stack, Mat_Stack,  conteneur)
    
    R, T, A = RTA(Wl, d_Stack, n_Stack, k_Stack, Ang)
    return R , T , A

def generate_population(chromosome_size, conteneur):
    """
    See : function optimize_agn
    This function generates the 1st generation for the genetic optimization process. 
    That is, a series of thin film stacks, each thickness of which is within the range for genetic algo, optimize_agn'.

    Parameters
    ----------
    chromosome_size : Int 
        The lenght of individual, so the number of chromosone 
    conteneur : TYPE
        DESCRIPTION.

    Returns
    -------
    population : numpy array
        DESCRIPTION.
    """
    pop_size= conteneur.get('pop_size')
    plage_ep = conteneur.get('Ep_plage')
    Ep_Substrack = conteneur.get('Ep_Substrack')
    # Je vais chercher d_Stack_Opt
    d_Stack_Opt = conteneur.get('d_Stack_Opt')
    
    # Si d_Stack_Opt n'existe pas dans le conteneur, il est quand même créer, mais il est de type NoneType
    # Cela veut dire que toutes les épaisseurs doivent être optimiser. 
    
    if isinstance(d_Stack_Opt, type(None)):
        d_Stack_Opt = ["no"] * chromosome_size
    
    population = []
    for i in range(pop_size):
        # 0 et 200 sont, en nm, les plages d'épaisseur du subtrat
        individual = [Ep_Substrack]
        for j in range(chromosome_size):
            if isinstance(d_Stack_Opt[j], (int, float)):
                individual += [d_Stack_Opt[j]]
            else : 
                individual += [np.random.randint(plage_ep[0], plage_ep[1])]
        population.append(individual)

    return population

def selection_min(population, evaluate, evaluate_rate, conteneur):
    """
    Parameters
    ----------
    population : List of list 
        Population is a list of the different indivudals 
        Each individual is a stack, so a list of ticknesss
    evaluate : fonction
        the name of a evluatuib fonction (the cost function), defined previously. 
    evaluate_rate : Float
        DESCRIPTION.
    conteneur : Dict
        "conteneur" is a dictionary with contain all "global" variables
        
    Returns
    -------
    parents : TYPE
        DESCRIPTION.
    Utilise la fonction evaluate pour calculer les performances des individus selon une fonction
    Si dans l'appel de la fonction dans le programme evaluate vaut evaluate_R_s le code remplace
    "evaluate" par "evaluate_R_s"
    => le nom de fonction est adaptatif ! 
    
    Selectionne selon le min
    """
    scores = [evaluate(individual, conteneur) for individual in population]
    parents = []
    for i in range(int(len(population)*evaluate_rate)):
        parent1 = population[scores.index(min(scores))]
        scores.pop(scores.index(min(scores)))
        parents.append(parent1)
    return parents

def selection_max(population, evaluate, evaluate_rate, conteneur):
    """
    Selectionne selon le max 
    """
    scores = [evaluate(individual, conteneur) for individual in population]
    parents = []
    for i in range(int(len(population)*evaluate_rate)):
        parent1 = population[scores.index(max(scores))]
        scores.pop(scores.index(max(scores)))
        parents.append(parent1)
    return parents

# Nouvelle version du crossover, par mask. # On mélange complétement les gènes
def crossover(parents, crossover_rate , pop_size):
    """
    See : optimize_agn
    """
    children = []
    for i in range((pop_size-len(parents))//2): # On fait deux enfants par parents
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

# Nouvelle version de la mutation
# Chaque gène de l'enfant à mutatin_rate chance de muter 
def mutation(children, mutation_rate, mutation_delta):
    """
    See : optimize_agn
    
    Cette fonction permet de faire muter les enfants (les nouveaux empillements), lors de leurs naissances.
    Lors de sa naissance un enfant à un % de chance (mutation_rate) de muter
    Certaines épaisseurs varie, de l'ordre de +/- mutation_delta.
    Ajout d'une boucle if pour que l'épaisseur ne soit pas négative
    """
    for i in range(1,len(children)):
        for j in range(np.shape(children)[1]-1):
            if np.random.uniform(0, 1) < mutation_rate:
                children[i][j+1] += np.random.uniform(-mutation_delta, mutation_delta)
                if children[i][j+1] <= 0:
                    children[i][j+1] = 0 #abs(children[i][mutation_point])
    return children

def optimize_agn(evaluate, selection, conteneur):
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
    Mat_Stack = conteneur.get('Mat_Stack')
    mod = conteneur.get('Mod_Algo')
    pop_size = conteneur.get('pop_size')
    crossover_rate = conteneur.get('crossover_rate')
    evaluate_rate = conteneur.get('evaluate_rate')
    mutation_rate = conteneur.get('mutation_rate')
    mutation_delta = conteneur.get('mutation_delta')
    Precision_AlgoG = conteneur.get('Precision_AlgoG')
    nb_generation= conteneur.get('nb_generation')

    # Seed 
    if 'seed' in conteneur:
        seed = conteneur.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    np.random.seed(np.random.randint(1,2**32 - 1))
    
    # Paramètre de l'optimisation 
    population = np.zeros(0)
    dev = float("inf")
    nb_run = 0
    chromosome_size = len(Mat_Stack) -1 # Nombre de couche minces
    population = generate_population(chromosome_size, conteneur)
    
    if mod == "for":
        """
        Le mod "for" lance l'algo génétique pour un nombre de génération précis'
        """
        for i in range(nb_generation):
            parents = selection(population, evaluate, evaluate_rate, conteneur)
            children = crossover(parents, crossover_rate, pop_size)
            children = mutation(children, mutation_rate, mutation_delta)
            population = parents + children
            nb_run = nb_run + 1
            # test de fin d'optimisation
    else:
        """
        Le mod "while" (comprendre si on n'écrit pas for) lance l'algo génétique pour un nombre infini générations, tant que le l'algo n'a pas conversé
        """
        while dev > Precision_AlgoG:
            parents = selection(population, evaluate, evaluate_rate)
            children = crossover(parents, crossover_rate, pop_size)
            children = mutation(children, mutation_rate, mutation_delta)
            population = parents + children
            # test de fin d'optimisation
            scores = [evaluate(individual) for individual in population]
            dev = np.std(scores)
            nb_run = nb_run + 1
    # fin de l'optimisation
    scores = [evaluate(individual, conteneur) for individual in population]
    dev = np.std(scores)
    dev = "{:.2e}".format(dev)
    

    # /!\ Peut être un soucis, car ici on selectionne le min des meilleurs scores. 
    # Hors, on peut optimiser en cherchant le max.
    # Mais si l'optimisation est bien fait le min des meilleurs scores doit correspondres aux max
    
    best_solution=population[scores.index(max(scores))]
    return best_solution, dev, nb_run, seed


def optimize_strangle(evaluate, selection, conteneur):
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
    #Je fais chercher les variables dans le conteneur
    Mat_Stack = conteneur.get('Mat_Stack')
    mod = conteneur.get('Mod_Algo')
    # Paramètre de l'optimisation 
    pop_size = conteneur.get('pop_size')
    evaluate_rate = conteneur.get('evaluate_rate')
    Precision_AlgoG = conteneur.get('Precision_AlgoG')
    nb_generation= conteneur.get('nb_generation')
    
    # Option 1 
    if 'seed' in conteneur:
        seed = conteneur.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
    
    # Lancement du problème
    population = np.zeros(0)
    dev = float("inf")
    chromosome_size = len(Mat_Stack) -1 # Nombre de couche minces
    population = generate_population(chromosome_size, conteneur)
    if mod == "for":
        nb_run = 0
        for i in range(nb_generation):
            parents = selection(population, evaluate, evaluate_rate, conteneur)
            children = children_strangle(pop_size, parents, chromosome_size)
            population = parents + children 
            nb_run = nb_run + 1
            # test de fin d'optimisation
    else:
        while dev > Precision_AlgoG:
            parents = selection(population, evaluate, evaluate_rate)
            children = children_strangle(pop_size, parents, chromosome_size)
            population = parents + children
            # test de fin d'optimisation
            scores = [evaluate(individual) for individual in population]
            dev = np.std(scores)
            nb_run = nb_run + 1
    # fin de l'optimisation
    scores = [evaluate(individual, conteneur) for individual in population]
    dev = np.std(scores)
    dev = "{:.2e}".format(dev)
    
    
    # /!\ Peut être un soucis, car ici on selectionne le min des meilleurs scores. 
    # Hors, on peut optimiser en cherchant le max.
    # Mais si l'optimisation est bien fait le min des meilleurs scores doit correspondres aux max
    
    best_solution=population[scores.index(max(scores))]
    
    
    return best_solution, dev, nb_run , seed

def children_strangle(pop_size, parents, chromosome_size):
    """
    See : optimize_strangle
    
    Cette fonction permet de générer la 1er génération d'enfants par étouffement'
    """
    children = []
    for i in range(pop_size-len(parents)):
        # 0 et 200 sont, en nm, les plages d'épaisseur du subtrat
        individual = [1000000] 
        for j in range(chromosome_size): 
            min_values = min([sublist[j+1] for sublist in parents])
            max_values = max([sublist[j+1] for sublist in parents])
            individual = individual + [np.random.randint(min_values,max_values)]
        children.append(individual)
    return children

def DEvol(f_cout, f_selection, conteneur):
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

    # Paramètres de DE - paramètres potentiels de la fonction
    cr = conteneur.get('mutation_rate') #cr=0.5; # Chances de passer les paramètres du parent à son rejeton.
    f1 = conteneur.get('f1') #f1=0.9;
    f2 = conteneur.get('f2') #f2=0.8;
    
    #Following seed problem when using the code, the seed can be manually targeting 
    
    # Option 1 
    if 'seed' in conteneur:
        seed = conteneur.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)

    # Calcul du budget : 
    pop_size = conteneur.get('pop_size')
    nb_generation = conteneur.get('nb_generation')
    budget = pop_size * nb_generation
    
    # Je donne la valeur de la population 
    population = pop_size
    
    # calcul des X_min et X_max
    Plage_ep = conteneur.get('Ep_plage')
    Plage_vf = conteneur.get('vf_plage')
    Ep_Substrack = conteneur.get('Ep_Substrack')
    Mat_Stack = conteneur.get('Mat_Stack')
    if 'nb_layer' in conteneur:
        nb_layer = conteneur.get('nb_layer')
    else : 
        nb_layer = 0 
    
    chromosome_size = len(Mat_Stack) + nb_layer -1 # Nombre de couche minces
    
    X_min = [Ep_Substrack]
    X_max = [Ep_Substrack]
    for i in range(chromosome_size):
         X_min += [Plage_ep[0]]
         X_max += [Plage_ep[1]]
         
    if 'n_plage' in conteneur:
        Plage_n = conteneur.get('n_plage')
        for i in range(nb_layer):
            X_min += [Plage_n[0]]
            X_max += [Plage_n[1]]
    
    if 'vf_plage' in conteneur:
        Plage_vf = conteneur.get('vf_plage')
        for i in range(len(Mat_Stack)):
            if "-" in Mat_Stack[i]:
                X_min += [Plage_vf[0]]
                X_max += [Plage_vf[1]]
            else:
                X_min += [Plage_vf[0]]
                X_max += [Plage_vf[0]]

    """
    idée ; je parcours la liste Mat_stack. Je créer un X_max_2. quand je croise un - 
    je emet entre vf[0] et vf[1] entre X_max et X min, sinon Xmin_2 = Xmax_2 = 0 
    Si a la fin je n'ai que des 0, je n'en fait rien. Si j'ai valeurs pas que = 0 , je rajoute le vecteur'
    for s in Mat_Stack: 
        if "-" in s:
            no_dash = False
            break
            
    if no_dash: # Si no_dask est true, je rentre dans la boucle
        for i in range(chromosome_size):
            X_min += [Plage_vf[0]]
            X_max += [Plage_vf[1]]*
    """
               
    # je met les list en array
    X_min = np.array(X_min)
    X_max = np.array(X_max)

    ############################# fin des lignes propres à COPS

    n=X_min.size

    # Initialisation de la population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Tirage aléatoire dans le domaine défini par X_min et X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        # Changement, car souvent je veux maximiser. Dans les autres algo, cela se fait via une fonction
        # selection. 
        if selection[0] == "selection_min":
            cost[k]=f_cout(omega[k], conteneur)
        elif selection[0] == "selection_max": 
            cost[k]=1-f_cout(omega[k], conteneur)

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    # Initialisations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])
    
    mutation_DE = conteneur.get('mutation_DE')

    # Boucle DE
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
                    tmp=f_cout(X, conteneur)
                elif selection[0] == "selection_max": 
                    tmp=1-f_cout(X, conteneur)
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

def DEvol_Video(f_cout, f_selection, conteneur):
    """ 
    Sub version of DE.
    Used by the main author of COPS for provide video of the optimization process
    The stack tickness is save during the process
    """
    selection = f_selection.__name__, 

# Paramètres de DE - paramètres potentiels de la fonction
    cr = conteneur.get('mutation_rate')
    #cr=0.5; # Chances de passer les paramètres du parent à son rejeton.
    f1 = conteneur.get('f1')
    f2 = conteneur.get('f2')
    """
    Suite à des problèmes sur le serveur Colossus, je fixe le seed dans la fonction. 
    Choisir l'une des options
    """
    # Option 1 
    seed = conteneur.get('seed')
    np.random.seed(seed)

    # Calcul du budget : 
    pop_size = conteneur.get('pop_size')
    nb_generation = conteneur.get('nb_generation')
    #print(nb_generation)
    budget = pop_size * nb_generation
   # print(budget)
    
    # Je donne la valeur de la population 
    population = pop_size
    
    # calcul des X_min et X_max
    Plage_ep = conteneur.get('Ep_plage')
    Plage_vf = conteneur.get('vf_plage')
    Ep_Substrack = conteneur.get('Ep_Substrack')
    Mat_Stack = conteneur.get('Mat_Stack')
    if 'nb_layer' in conteneur:
        nb_layer = conteneur.get('nb_layer')
    else : 
        nb_layer = 0 
    
    chromosome_size = len(Mat_Stack) + nb_layer -1 # Nombre de couche minces
    
    X_min = [Ep_Substrack]
    X_max = [Ep_Substrack]
    for i in range(chromosome_size):
         X_min += [Plage_ep[0]]
         X_max += [Plage_ep[1]]
         
    if 'n_plage' in conteneur:
        Plage_n = conteneur.get('n_plage')
        for i in range(nb_layer):
            X_min += [Plage_n[0]]
            X_max += [Plage_n[1]]
    
    if 'vf_plage' in conteneur:
        Plage_vf = conteneur.get('vf_plage')
        for i in range(len(Mat_Stack)):
            if "-" in Mat_Stack[i]:
                X_min += [Plage_vf[0]]
                X_max += [Plage_vf[1]]
            else:
                X_min += [Plage_vf[0]]
                X_max += [Plage_vf[0]]

    # je met les list en array
    X_min = np.array(X_min)
    X_max = np.array(X_max)

    ############################# fin des lignes propres à COPS

    n=X_min.size

    # Initialisation de la population
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    # Tirage aléatoire dans le domaine défini par X_min et X_max.
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        # Changement, car souvent je veux maximiser. Dans les autres algo, cela se fait via une fonction
        # selection. 
        if selection[0] == "selection_min":
            cost[k]=f_cout(omega[k], conteneur)
        elif selection[0] == "selection_max": 
            cost[k]=1-f_cout(omega[k], conteneur)

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]
    
    """ 
    Correction du bug au 27/06/2023. Il ne faut pas utiliser .append mais .copy
    Reproduire le bug
    
    best_tab= []
    best_tab.append(best)
    """
    #print(best)
    best_tab= np.copy(best)
    #print(best_tab)
    
    # Initialisations
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])
    
    mutation_DE = conteneur.get('mutation_DE')

    # Boucle DE
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
                    tmp = f_cout(X, conteneur)
                elif selection[0] == "selection_max": 
                    tmp = 1 - f_cout(X, conteneur)
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
        Ancienne version buggé : 
        #best_tab.append(best)
        Correction du bug.
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

def PSO(evaluate, selection, conteneur):
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
    Mat_Stack = conteneur.get('Mat_Stack')
    # number of particules is storage in pop_size
    num_particles = conteneur.get('pop_size')
    # number of particules is storage in nb_generation
    num_iterations = conteneur.get('nb_generation')
    Ep_Substrack = conteneur.get('Ep_Substrack')

    selection = selection.__name__,

    # Parameters just for PSO. They are NOT optimized for coatings optimization or photonics
    inertia_weight = 0.8
    cognitive_weight = 1.5
    social_weight = 1.5
    
    # Fixation du seed
    if 'seed' in conteneur:
        seed = conteneur.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    # Creation of lower_bound and upper bound
    Plage_ep = conteneur.get('Ep_plage')
    chromosome_size = len(Mat_Stack) -1 # Nombre de couche minces
    lower_bound = np.array([Plage_ep[0]] * chromosome_size) # Définir les bornes inférieures pour chaque dimension
    lower_bound = np.insert(lower_bound, 0, Ep_Substrack) # j'ajoute l'épaisseur du substrat dans les bornes 
    upper_bound = np.array([Plage_ep[1]] * chromosome_size) # Définir les bornes supérieures pour chaque dimension
    upper_bound = np.insert(upper_bound, 0, Ep_Substrack) # j'ajoute l'épaisseur du substrat dans les bornes 
    
    # Start
    num_dimensions = len(lower_bound)
    particles = []
    convergence = [] # List of best values durint the optimization process
    # Initialisation 

    if selection[0] == "selection_min":
        global_best_position = np.zeros(num_dimensions)
        global_best_score = float('inf')
        score_ref = float('inf')
        
    elif selection[0] == "selection_max": 
        global_best_position = np.array(0 * num_dimensions)
        global_best_score = 0
        score_ref = 0

    # Initialisation
    for _ in range(num_particles):
        position = np.random.uniform(lower_bound, upper_bound)
        velocity = np.random.uniform(lower_bound * 0.1, upper_bound * 0.1)
        particle = Particle(position, velocity, score_ref)
        particles.append(particle)
        
        # Mise à jour du meilleur score global
        if selection[0] == "selection_min":
            score = evaluate(position, conteneur)
            if score < global_best_score:
                global_best_score = score
                global_best_position = position

            # Mise à jour du meilleur score personnel du particle
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = position
                
        elif selection[0] == "selection_max": 
            score = evaluate(position, conteneur)
            if score > global_best_score:
                global_best_score = score
                global_best_position = position

            # Mise à jour du meilleur score personnel du particle
            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = position
    
    convergence.append(global_best_score)  # First best values 

    # Optimisation
    for _ in range(num_iterations):
        for particle in particles:
            # Mise à jour de la vitesse et de la position
            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_weight * np.random.rand() * (particle.best_position - particle.position) +
                                 social_weight * np.random.rand() * (global_best_position - particle.position))
            particle.position = np.clip(particle.position + particle.velocity, lower_bound, upper_bound)

            # Mise à jour du meilleur score global
            if selection[0] == "selection_min":
                score = evaluate(particle.position, conteneur)
                # Mise à jour du meilleur score personnel et global
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position

                if score < global_best_score:
                    global_best_score = score
                    convergence.append(global_best_score) # Adding the newest best values 
                    global_best_position = particle.position
                
            if selection[0] == "selection_max": 
                score = evaluate(particle.position, conteneur)
                # Mise à jour du meilleur score personnel et global
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position

                if score > global_best_score:
                    global_best_score = score
                    convergence.append(global_best_score) # Adding the newest best values 
                    global_best_position = particle.position
                    
    # global_best_score : score (cost function) of the best position 
    return [global_best_position, convergence, num_iterations, seed] 

def generate_neighbor(solution, conteneur):
    """
    Function for simulated_annealing algorithm
    """
    Plage_ep = conteneur.get('Ep_plage')
    
    neighbor = solution.copy()
    # random.randint start at 1 and not 0, because the 1st value is the substrat thickness, witch cannot be modified
    index = random.randint(1, len(neighbor) - 1)
    neighbor[index] = random.uniform(Plage_ep[0], Plage_ep[1])  # Choisir une valeur aléatoire entre -1 et 1 pour la sous-liste sélectionnée
    return neighbor

def acceptance_probability(current_score, new_score, temperature):
    """
    Function for simulated_annealing algo
    """
    if new_score < current_score:
        return 1.0
    return math.exp((current_score - new_score) / temperature)

def simulated_annealing(evaluate, selection, conteneur):
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
    Mat_Stack = conteneur.get('Mat_Stack')
    Ep_Substrack = conteneur.get('Ep_Substrack')
    # number of iteration of the annealing
    nb_generation = conteneur.get('nb_generation')
    Plage_ep = conteneur.get('Ep_plage')
    
    # Get the name of the selection function
    selection = selection.__name__,
    
    # Paramètres du recuit simulé
    initial_temperature = 3000.0
    final_temperature = 0.01
    cooling_rate = 0.95
    current_temperature = initial_temperature
    
    # Fixation du seed
    if 'seed' in conteneur:
        seed = conteneur.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    # Creation of lower_bound and upper bound
    
    chromosome_size = len(Mat_Stack) - 1 # Nombre de couche minces
    # Génération de la solution initiale
    current_solution = [random.uniform(Plage_ep[0], Plage_ep[1]) for _ in range(chromosome_size)]  # Générer une solution aléatoire
    current_solution = np.insert(current_solution, 0, Ep_Substrack) # j'ajoute l'épaisseur du substrat dans les bornes 
    
    # Initialisation 
    best_solution = current_solution.copy()
    best_score = evaluate(best_solution, conteneur)
    
    convergence = [] # List of best values durint the optimization process
    convergence.append(best_score)  # First best values 

    # Start of annealing
    while current_temperature > final_temperature:
        
        for _ in range(nb_generation):
            neighbor_solution = generate_neighbor(current_solution, conteneur)
            
            # Evaluate the score of the neighbor according of 
            if selection[0] == "selection_min":
                neighbor_score = evaluate(neighbor_solution, conteneur)
            elif selection[0] == "selection_max": 
                neighbor_score = 1- evaluate(neighbor_solution, conteneur)
            
            neighbor_score = evaluate(neighbor_solution, conteneur)

            if acceptance_probability(exemple_evaluate(current_solution), neighbor_score, current_temperature) > random.uniform(0, 1):
                current_solution = neighbor_solution
                # Keeping the current solution, depending of the selection methode (min or max)
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
    Function for One_plus_One optimisation methode
    
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
    mutant = solution.copy()  # Copie de la solution initiale
    mutant[1:] +=  np.random.normal(0, step_size, len(solution)-1)
    #return mutant.tolist()
    return mutant

def One_plus_One_ES(evaluate, selection, conteneur):
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
    Mat_Stack = conteneur.get('Mat_Stack')
    # Interation 
    pop_size = conteneur.get('pop_size')
    nb_generation = conteneur.get('nb_generation')
    #print(nb_generation)
    num_iterations = pop_size * nb_generation
    Ep_Substrack = conteneur.get('Ep_Substrack')
    Plage_ep = conteneur.get('Ep_plage')
    
    # Facteur d'échelle de la taille de pas
    step_size_factor = conteneur.get('mutation_delta')
    
    # Get the selection function name
    selection = selection.__name__,
    
    # Parmeter for (1+1)-ES 
    initial_step_size = 10  # Taille de pas initiale
    
    # Fixation du seed
    if 'seed' in conteneur:
        seed = conteneur.get('seed')
        np.random.seed(seed)
    else : 
       seed = random.randint(1 , 2**32 - 1)
       np.random.seed(seed)
       
    # Creation of the solution
    
    chromosome_size = len(Mat_Stack) - 1 # Nombre de couche minces
    # Génération de la solution initiale
    initial_solution = [random.uniform(Plage_ep[0], Plage_ep[1]) for _ in range(chromosome_size)]  # Générer une solution aléatoire
    initial_solution = np.insert(initial_solution, 0, Ep_Substrack) # j'ajoute l'épaisseur du substrat dans les bornes      
    
    current_solution = initial_solution
    current_step_size = initial_step_size
    
    current_score = evaluate(current_solution, conteneur)
    
    convergence = [] # List of best values durint the optimization process
    convergence.append(current_score)

    for _ in range(num_iterations):
        mutant_solution = generate_mutant(current_solution, current_step_size)
        mutant_score = evaluate(mutant_solution, conteneur)
        
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
