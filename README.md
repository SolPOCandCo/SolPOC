## COPS
COPS (from French : "Code d'Optimisation des Performances Solaires") is a simple and fast code running under Python. Just from complex refractive indices of real materials, the code can models and optimizes with advenced methode the optical behavior of a simple or highly complex stack of thin layers deposited on a solid substrate.
## About COPS

## Installation

Currently, a command such as pip install cops is not available. 

Installing COPS means downloading the entire project, including the /Materials folder, which contains the optical indices of the materials as txt files. Then open the file "optimisation_multiprocess.py" from an IDE who can run Python code. The "function_COPS.py" file contains all COPS functions and must be visible to optimisation_multiprocess.py file. All the code in "function_COPS.py" does not need to be modified for run a optimisation process. 

COPS can be used from optimisation_multiprocess.py simply by modifying the code from line 17 to line 71. This replaces a kind of GUI. COPS is designed to be used by users with little knowledge of Python. Lines 17 to 71 describe the problem to be optimized and set the parameters. You don't need to modify a single line of code after line 71. If launched correctly, COPS synthesizes the results in the console. It automatically writes the results to a folder, named with the date and time of execution. 

## User Guide

## What COPS can do

## What COPS cannot do

## For specialists

## References
If you use COPS and if this is relevant, please cite the papers associated with. Another paper, just for COPS, is on way

@PHDTHESIS{grosj2018,
url = "http://www.theses.fr/2018PERP0002",
title = "Etude, modélisation et optimisation de surfaces fonctionnelles pour les collecteurs solaires thermiques à concentration",
author = "Grosjean, Antoine",
year = "2018",
note = "Thèse de doctorat dirigée par Thomas, Laurent et Soum-Glaude, Audrey Sciences pour l’ingénieur Perpignan 2018",
note = "2018PERP0002",
url = "http://www.theses.fr/2018PERP0002/document",
}

Even if COPS is quite simple code, this is a research-grade program. We actually do research with it. Do not hesite to contact us, for help, academic projet or cited to current version of our work

## Contributors
Here is a list of contributors to COPS : 
* Pauline Bennet, UGA, France (@Ellawin)
* Antoine Moreau, UGA, France  (@AnMoreau)
* Andrey Soum-Glaude, PROMES-CNRS, France

Many thank to Thalita Drumond, EPF, France (thalitadru) and Antoine Gademer, EPF, France
