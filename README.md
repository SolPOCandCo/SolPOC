## COPS
COPS (from French "Code d'Optimisation des Performances Solaires", Optimisation Code for Solar Performances in english) is a simple and fast code running under Python 3.9. The code has been developed during the author's Ph.D Thesis at PROMES CNRS (Perpignan, 66, France) defended in 2018. Just from complex refractive indices of real materials, the code COPS can models and optimizes with advenced methode the optical behavior (reflectivity, transmissivity or absorptivity) of a thin layers stack deposited on a solid substrate.

This version is actualy under developpement. COPS are tested under Windows, using Spyder as IDE. Most comment are in French, please forgive us !
This code can be used for scientific research or academic educations. 

## About COPS
- Study the optical behavior of thin-film stacks, calculating reflectivity, transmissivity and absorptivity over the solar spectrum (280 - 2500 nm) and beyond.
- Working with thin layers stacks ranging from a substrate (without thin film) to an infinite stack of thin films
- Optimize stack optical performances according to several cost functions, including cost functions for building and solar thermal. 
- Propose 6 different optimization methods based on evolutionary algorithm (EA) 
- Work with multiprocessing (using more of 1 CPU), using a pool
- Work with a spectral range from UV to IR (typically 280 nm to 30 µm, can be modified by the users) 
- Automatically write results (.txt files and .png images) to a folder 
- Use effective medium approximatio (EMA) to model the optical behavior of material mixtures (dielectric mixtures, metal-ceramic composites, porous materials, etc.). 
- Propose a simplified user interface, bringing together all useful variables in a few lines of code.

## Installation

Currently, a command such as `pip install cops` is not available. 

Installing COPS means downloading the entire project, including the /Materials folder, which contains the optical indices of the materials as txt files. Then open the file "optimisation_multiprocess.py" from an IDE who can run Python code. The "function_COPS.py" file contains all COPS functions and must be visible to optimisation_multiprocess.py file. All the code in "function_COPS.py" does not need to be modified for run a optimisation process. 

COPS can be used from optimisation_multiprocess.py simply by modifying the code from line 17 to line 71. This replaces a kind of GUI. COPS is designed to be used by users with little knowledge of Python. Lines 17 to 71 describe the problem to be optimized and set the parameters. You don't need to modify a single line of code after line 71. If launched correctly, COPS synthesizes the results in the console. It automatically writes the results to a folder, named with the date and time of execution. 

## User Guide

A User Guide and differents tutorial are present in the tutorial folders. As first users, give us any feedback that will help us make the code easier for others to understand

## Example of COPS use

COPS can be used for sevaral purposes, but not limited : 
- antireflective coatings for human eye vision, PV cells or solar thermal application
- coatings for radiative cooling
- dielectric / Bragg mirror
- low-e coatings (solar control glass) for building application 
- reflective coatings, using metalic or dielectric coating 
- selective coatings for solar thermal applications
- PV mirrors

See the tutorial folder for more details. 

## For specialists

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
* Pauline Bennet, UGA, France (@Ellawin)
* Antoine Moreau, UGA, France  (@AnMoreau)
* Andrey Soum-Glaude, PROMES-CNRS, France

Many thank to Thalita Drumond, EPF, France (@thalitadru) and Antoine Gademer, EPF, France
