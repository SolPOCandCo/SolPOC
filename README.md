## SolPOC
SolPOC (Solar Performance Optimization Code) is a simple and fast code running under Python 3.9. The code has been developed during the author's Ph.D Thesis at PROMES CNRS (Perpignan, 66, France) defended in 2018. The project was named COPS (Code d'Optimization des Performance Solaires) between 2018 to november 2023. 

SolPOC is a Python code designed to solve Maxwell's equations in a multilayered thin film structure.
The code is specifically designed for research in coatings, thin film deposition, and materials research for solar energy applications (thermal and PV).
The code uses a stable method to quickly calculate reflectivity, transmissivity, and absorptivity from a stack of thin films over a full solar spectrum. SolPOC comes with several optimization methods, a multiprocessing pool, and a comprehensive database of refractive indices for real materials.
In the end, SolPOC is simple to use for no-coder users thanks to main script, which regroup all necessary variables and automatically save important results in text files and PNG images.

The code are tested under Windows, using different IDE. 
This code can be used for scientific research or academic educations. 

## SolPOC highlights
- Study the optical behavior of thin-film stacks, calculating reflectivity, transmissivity and absorptivity over the solar spectrum (280 - 2500 nm) and beyond.
- Working with thin layers stack ranging from a substrate (without thin film) to an 'infinite' stack of thin films. 
- Work with refractive index data of real materials, found in peer-reviewed papers.  
- Optimize stack optical performances according to several cost functions, including cost functions for building and solar thermal uses. 
- Propose 6 different optimization methods based on Evolutionary Algorithm (EA) 
- Work with multiprocessing (using more of 1 CPU), using a pool
- Work with a spectral range from UV to IR (typically 280 nm to 30 µm, can be modified by the users) 
- Automatically write results (`.txt` files and `.png` images) to a folder 
- Use Effective Medium Approximation methods (EMA) to model the optical behavior of material mixtures (dielectric mixtures, metal-ceramic composites, porous materials, etc.). 
- Propose a simplified user interface, bringing together useful variables in a few lines of code.

## Installation
### From PyPI (Python Package Index)
From a Python IDE (e.g. Spyder) type the following command in the IPython Console:

`pip install solpoc`

This will download and install solpoc. 
You may need to restart the IPython kernel so the new package is taken into account.
Once this is done, type the following command (again into the IPython Console): 

`!solpoc-init`

This will create a folder called `ProjectSolPoc` (within your Windows user folder), containing the base `optimisation_multiprocess.py` script inside. For instance, if your username is `toto`, then the file will be at 
`C:\Users\toto\ProjectSolPoc`. Your are free to move and rename this folder as you see fit.

Now you can open and launch the file `optimisation_multiprocess.py` from an IDE which can run Python code. 

### From sources (i.e. by downloading this repo)
Clone or download the repo to your machine. If you've downloaded an archive, you need to unzip it. After that you should have a folder containing the full repo:

```
C:\<path_to_repo_folder>\SolPOC
   | ---  Examples
   | ---  docs
   | ---  solpoc
          | --- scripts
          | --- Materials
          | --- __init__.py
          | --- ...
   | ---  ...
   | ---  pyproject.toml
```

Open this folder with your Python IDE. Then, on the IPython Console, type the following command:

`pip install -e .`

This will install the local version of solpoc (instead of the one in PyPI archives).
You may need to restart the IPython kernel so the new package is taken into account.

You may navigate to `solpoc\scripts` to recover the main script `optimisation_multiprocess.py`, that you can launch with your IDE.

### Uninstalling
In either case, to uninstalll the package, open your Python IDE and type on the IPython kernel:

`pip uninstall solpoc -y`

You may need to restart the IPython kernel so the change is taken into account.


## Usage

SolPOC can be used from `optimisation_multiprocess.py` simply by modifying the code from line 17 to line 67. This replaces a kind of GUI (Graphic Users Interface). SolPOC is designed to be used by users with little knowledge of Python. Lines 17 to 67 describe the problem to be optimized and set the parameters. You don't need to modify a single line of code after line 67. If launched correctly, the code synthesizes the results. It automatically writes the results to a folder, named with the date and time of execution with several pictures and textes files. 

## User Guide

A User Guide and different tutorials are present in the [docs](./docs/Readme.md) folders. As first users, give us any feedback that will help us make the code easier to understand for others users. 

## Example of SolPOC use

The code can be used for several purposes, but not limited : 
- antireflective coatings for human eye vision, PV cells or solar thermal application
- coatings for radiative cooling
- coatings for optical instruments
- dielectric / Bragg mirrors
- low-e coatings (for solar control glass) for building application 
- reflective coatings, using metallic or dielectric layers
- selective coatings for solar thermal applications (absorb the solar spectrum without radiative losses) 

See the [examples folder](./Examples/) for more details. 

## For specialists

The code uses a conventional method known as the Abélès formalism for calculation of the stack optical properties (Reflectance, Transmittance and Absorptance), in all wavelengths with a incidence angle. This method for solve Maxwell's equations has been detailed in the scientific literature. The Abélès formalisme is used in key function named RTA, using the complex refractive index and thickness of thin layers deposited on a substrate. The RTA function is coded with Numpy package for time speed-up. The complex refractive index of materials are available in folder “Materials”, and mainly come from of the RefractiveInde.info website. The website share refractive index of materials in peer-reviewed papers. 

```
RefractiveIndex.INFO - Refractive index database
```
The complex refractive index of composite layers, such as cermet (W-Al2O3, mixture of dielectric and metal) or porous materials (such as mixture of air and dielectric, like air-SiO2) were estimated by applying an Effective Medium Approximation (EMA) method. Such EMA methods consider a macroscopically inhomogeneous medium where quantities such as the dielectric function vary in space, and are often used in materials sciences. Different EMA theories have been reported in the literature, and the Bruggeman method is used here. 

SolPOC and COPS code (the previous name, a code version written in Scilab) has already provided scientific publication: 
- A.Grosjean et al, Influence of operating conditions on the optical optimization of solar selective absorber coatings, Solar Energy Materials and Solar Cells, Volume 230, 2021, 111280, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2021.111280.
- A.Grosjean et al, Replacing silver by aluminum in solar mirrors by improving solar reflectance with dielectric top layers, Sustainable Materials and Technologies, Volume 29, 2021, e00307, ISSN 2214-9937, https://doi.org/10.1016/j.susmat.2021.e00307.
- A.Grosjean et al Comprehensive simulation and optimization of porous SiO2 antireflective coating to improve glass solar transmittance for solar energy applications, Solar Energy Materials and Solar Cells, Volume 182, 2018, Pages 166-177, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2018.03.040.
- Danielle Ngoue, Antoine Grosjean, (...), Ceramics for concentrated solar power (CSP): From thermophysical properties to solar absorbers, In Elsevier Series on Advanced Ceramic Materials, Advanced Ceramics for Energy Conversion and Storage, Elsevier, 2020, Pages 89-127, ISBN 9780081027264,

## References
If you use SolPOC and if this is relevant, please cite the papers associated with. Another paper, just for SolPOC, is on way

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
Even if SolPOC is quite simple code, this is a research-grade program. We actually do research with it. Do not hesitate to contact us, for help, academic project or cited to current version of our work

## Contributors
The main author of the code is Antoine Grosjean, EPF, France (antoine.grosjean@epf.fr) 
Here is a list of major contributors to SolPOC : 
* Pauline Bennet, UCA, France (@Ellawin)
* Antoine Moreau, UCA, France  (@AnMoreau)
* Andrey Soum-Glaude, PROMES-CNRS, France
  
Many thank to Thalita Drumond (@thalitadru), Titouan Février and Antoine Gademer from EPF, France. 
We would like to thank to Denis Langevin Université Clermont Auvergne France and Amine Mahammou, PROMES-CNRS, France
