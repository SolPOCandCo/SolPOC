## COPS
COPS (from French "Code d'Optimisation des Performances Solaires", Optimisation Code for Solar Performances in english) is a simple and fast code running under Python 3.9. The code has been developed during the author's Ph.D Thesis at PROMES CNRS (Perpignan, 66, France) defended in 2018. Just from complex refractive indices of real materials, the code COPS can models and optimizes with advenced methode the optical behavior (reflectivity, transmissivity or absorptivity) of a thin layers stack deposited on a solid substrate.

This version is actualy under developpement. COPS are tested under Windows, using Spyder as IDE. Most comment are in French, please forgive us !
This code can be used for scientific research or academic educations. 

## About COPS
- Study the optical behavior of thin-film stacks, calculating reflectivity, transmissivity and absorptivity over the solar spectrum (280 - 2500 nm) and beyond.
- Working with thin layers stack ranging from a substrate (without thin film) to an 'infinite' stack of thin films. 
- Work with refractif index data of real materials, found in peer-reviewed papers.  
- Optimize stack optical performances according to several cost functions, including cost functions for building and solar thermal uses. 
- Propose 6 different optimization methods based on evolutionary algorithm (EA) 
- Work with multiprocessing (using more of 1 CPU), using a pool
- Work with a spectral range from UV to IR (typically 280 nm to 30 µm, can be modified by the users) 
- Automatically write results (.txt files and .png images) to a folder 
- Use Effective Medium Approximation methods (EMA) to model the optical behavior of material mixtures (dielectric mixtures, metal-ceramic composites, porous materials, etc.). 
- Propose a simplified user interface, bringing together useful variables in a few lines of code.

## Installation

Currently, a command such as `pip install cops` is not available. 

Installing COPS means downloading the entire project, including the /Materials folder, which contains the optical indices of the materials as txt files. Then open the file "optimisation_multiprocess.py" from an IDE who can run Python code. The "function_COPS.py" file contains all COPS functions and must be visible to optimisation_multiprocess.py file. All the code in "function_COPS.py" does not need to be modified for run a optimisation process. 

COPS can be used from optimisation_multiprocess.py simply by modifying the code from line 17 to line 71. This replaces a kind of GUI. COPS is designed to be used by users with little knowledge of Python. Lines 17 to 71 describe the problem to be optimized and set the parameters. You don't need to modify a single line of code after line 71. If launched correctly, COPS synthesizes the results in the console. It automatically writes the results to a folder, named with the date and time of execution. 

## User Guide

A User Guide and differents tutorial are present in the tutorial folders. As first users, give us any feedback that will help us make the code easier to understandfor others users. 

## Example of COPS use

COPS can be used for sevaral purposes, but not limited : 
- antireflective coatings for human eye vision, PV cells or solar thermal application
- coatings for radiative cooling
- coatings for optical instruments
- dielectric / Bragg mirrors
- low-e coatings (for solar control glass) for building application 
- reflective coatings, using metalic or dielectric layers
- selective coatings for solar thermal applications (absorb the solar spectrum without radiative losses) 
- PV mirrors for PV/CSP or PV/CST applications

See the tutorial folder for more details. 

## For specialists

The code uses a conventional method known as the Transfer Matrix Method (TMM) for calculation of the stack optical properties (Reflectance, Transmittance and Absorptance), in all wavelenghts with a incidence angle. This method based on Fresnel equations has been detailed in the scientifique literature. The TMM method is used in key function named RTA, using the complex refractive index and thickness of thin layers deposited on a substrate. The complex refractive index of materials are available in folder “Materials”, and mainly come from of the RefractiveInde.info website. The website share refractive index of materials in peer-reviewed papers. 

```
RefractiveIndex.INFO - Refractive index database
```
The complex refractive indix of composite layers, such as cermet (W-Al2O3, mixture of dieliectric and metal) or porous materials (such as mixture of air and dielectric, like air-SiO2) were estimated by applying an Effective Medium Approximation (EMA) method. Such EMA methods consider a macroscopically inhomogeneous medium where quantities such as the dielectric function vary in space, and are often used in materials sciences. Different EMA theories have been reported in the literature, and the Bruggeman method is used here. 

COPS code (in a later version written in Scilab) has already provided scientific publication: 
- A.Grosjean et al, Influence of operating conditions on the optical optimization of solar selective absorber coatings, Solar Energy Materials and Solar Cells, Volume 230, 2021, 111280, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2021.111280.
- A.Grosjean et al, Replacing silver by aluminum in solar mirrors by improving solar reflectance with dielectric top layers, Sustainable Materials and Technologies, Volume 29, 2021, e00307, ISSN 2214-9937, https://doi.org/10.1016/j.susmat.2021.e00307.
- A.Grosjean et al Comprehensive simulation and optimization of porous SiO2 antireflective coating to improve glass solar transmittance for solar energy applications, Solar Energy Materials and Solar Cells, Volume 182, 2018, Pages 166-177, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2018.03.040.
- Danielle Ngoue, Antoine Grosjean, (...), Ceramics for concentrated solar power (CSP): From thermophysical properties to solar absorbers, In Elsevier Series on Advanced Ceramic Materials, Advanced Ceramics for Energy Conversion and Storage, Elsevier, 2020, Pages 89-127, ISBN 9780081027264,

## References
If you use COPS and if this is relevant, please cite the papers associated with. Another paper, just for COPS, is on way

```
@PHDTHESIS{grosj2018,
url = "http://www.theses.fr/2018PERP0002",
title = "Etude, modélisation et optimisation de surfaces fonctionnelles pour les collecteurs solaires thermiques à concentration",
author = "Grosjean, Antoine",
year = "2018",
note = "Thèse de doctorat dirigée par Thomas, Laurent et Soum-Glaude, Audrey Sciences pour l’ingénieur Perpignan 2018",
note = "2018PERP0002",
url = "http://www.theses.fr/2018PERP0002/document",
}
```
Even if COPS is quite simple code, this is a research-grade program. We actually do research with it. Do not hesite to contact us, for help, academic projet or cited to current version of our work

## Contributors
Here is a list of contributors to COPS : 
* Pauline Bennet, UCA, France (@Ellawin)
* Antoine Moreau, UCA, France  (@AnMoreau)
* Andrey Soum-Glaude, PROMES-CNRS, France

Many thank to Thalita Drumond, EPF, France (@thalitadru) and Antoine Gademer, EPF, France
