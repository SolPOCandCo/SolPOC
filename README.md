# SolPOC
## Introduction
SolPOC, Solar Performance Optimization Code is a simple and fast code running under Python 3. 
The code is designed to solve Maxwell's equations in a multilayered thin film structure in 1 dimension. 
The package is specifically designed for research, industrial and academic research in optical coatings, thin film deposition or in solar energy applications (thermal, photovoltaic, etc.). 
The SolPOC code use a stable method (Abélès matrix) to quickly calculate the optical behavior (reflectivity, transmissivity, and absorptivity) from a stack of thin films deposited on a solid substrate over a full solar spectrum just from complex refractive indices of real materials. 
SolPOC comes with several optimization methods, specific cost functions for optic or solar energy applications and a comprehensive database of refractive indices for real materials.

In the end, SolPOC is simple to use for no-coder users thanks to main script, which regroup all necessary variables and automatically saves important results in text files and PNG images.
Thanks to Python and the use of a multiprocessing pool most problems can be solved in a couple of minutes. 
This code can be used for scientific research or academic education. 

## SolPOC highlights
- Study the optical behavior of thin-film stacks, calculating reflectivity, transmissivity and absorptivity over the solar spectrum (280 - 2500 nm) and beyond.
- Working with thin layers stack ranging from a substrate (without thin film) to an 'infinite' stack of thin films. 
- Work with refractive index data of real materials, found in peer-reviewed papers.  
- Optimize stack optical performances according to several cost functions, including cost functions for building and solar thermal uses. 
- Propose different optimization methods based on Evolutionary Algorithm (EA) 
- Work with a spectral range from UV to IR (typically 280 nm to 30 µm, can be modified by the users) 
- Automatically write results (`.txt` files and `.png` images) to a folder 
- Use Effective Medium Approximation methods (EMA) to model the optical behavior of material mixtures (dielectric mixtures, metal-ceramic composites, porous materials, etc.). 
- Propose template scripts which bringing together useful variables in a few lines of code.

## Installation
### From PyPI (Python Package Index)
On a terminal window (e.g. the terminal console in your Python IDE) type the following command:

`pip install solpoc`

This will download and install solpoc. Once this is done, type the following command on your terminal: 

`solpoc-init`

This will create a folder called `ProjectSolPoc` (within your Windows user folder), containing the base `template_optimization.py` script inside. For instance, if your username is `toto`, then the file will be at 
`C:\Users\toto\ProjectSolPoc`. Your are free to move and rename this folder as you see fit.

Now you can open and launch the file `template_optimization.py` from your python IDE (or from your terminal).

> [!NOTE]
> **For Spyder users**: you can use the IPython console within Spyder. Just prefix the commands with a `!` sign:
> - `!pip install solpoc` to install
> - `!solpoc-init` to start a project folder
>
> You may need to restart the IPython kernel so the new package is taken into account.


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

Open this folder with your Python IDE. Then, on the terminal console, type the following command:

`pip install -e .`

This will install the local version of solpoc (instead of the one in PyPI archives).

You may navigate to `solpoc\scripts` to recover the main template script `template_optimization.py`, that you can launch with your IDE.

### Uninstalling
In either case, to uninstalll the package, open your Python IDE and type on the terminal console:

`pip uninstall solpoc -y`

## Quickstart
SolPOC is designed to be used by users with little knowledge of Python, reason why an initial template script is provided after calling `solpoc-init`. 
SolPOC can then be used from `template_optimization.py` simply by modifying the desired parameters in the indicated section:
``` python
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - START                                #
#----------------------------------------------------------------------------#
# %%  Main : You can start modifying parameters from this point on
# Comment to be written in the simulation text file
Comment = "A sentence to be written in the final text file"
# Ordered list of materials used in the stack
Mat_Stack = ["BK7", "Al", "air-SiO2", "TiO2", "SiO2", "TiO2"]
...
#----------------------------------------------------------------------------#
#                   SCRIPT PARAMETERS - END                                  #
#----------------------------------------------------------------------------#
# %% You should stop modifying anything :)
```
This code section defines the problem to be optimized and sets the main parameters. Note that parameters are accompanied by a short comment describing them.

> [!CAUTION] 
> You don't need to modify a single line of code after the `SCRIPT PARAMETERS - END` markup. 
 
If launched correctly, the code optimizes the given problem and outputs numeric and graphical results. It automatically writes the results to a folder, named with the date and time of execution with several pictures and texts files. 

> [!TIP]
> Please refer to the [User Guide](.docs/UserGuide.pdf) (section 3.11 and Appendix 2) for a complete list of parameters available for customization.

## Support
SolPOC has been tested under Windows, using Spyder, Visual Studio or PyCharm as IDEs. 
Please report to the main author [A. Grosjean](antoine.grosjean@epf.fr) any bug or ideas for further implementation. You may do so by opening a [GitHub issue](https://github.com/SolPOCandCo/SolPOC/issues) describing your request.

## Detailed documentation and user guide

A User Guide and different tutorials are present in the [docs](./docs/Readme.md) folder. 
As first users, give us any feedback that will help us make the code easier to understand for others users. 

## Examples of SolPOC use cases

The code can be used for several purposes, but not limited : 
- anti-reflective coatings for human eye vision, PV cells or solar thermal application
- coatings for radiative cooling
- coatings for optical instruments
- dielectric / Bragg mirrors
- low-e coatings (for solar control glass) for building application 
- reflective coatings, using metallic or dielectric layers
- selective coatings for solar thermal applications (absorb the solar spectrum without radiative losses) 

See the [examples folder](./Examples/) for more details. 

## For specialists

The code uses a conventional method known as the Abélès formalism for calculation of the stack optical properties (Reflectance, Transmittance and Absorptance, RTA for short), in all wavelengths with a incidence angle. This method to solve Maxwell's equations has been detailed in the scientific literature. The Abélès formalisme is used in key function named `RTA`, using the complex refractive index and thickness of thin layers deposited on a substrate. The `RTA` function is coded with Numpy package for time speed-up. The complex refractive index of materials are available in folder “Materials”, and mainly come from of the [RefractiveIndex.info](https://RefractiveIndex.info) website. The website shares refractive index of materials from peer-reviewed papers. 

```
RefractiveIndex.INFO - Refractive index database
```
The complex refractive index of composite layers, such as cermet (W-Al2O3, mixture of dielectric and metal) or porous materials (such as mixture of air and dielectric, like air-SiO2) were estimated by applying an Effective Medium Approximation (EMA) method. Such EMA methods consider a macroscopically inhomogeneous medium where quantities such as the dielectric function vary in space, and are often used in materials sciences. Different EMA theories have been reported in the literature, and the Bruggeman method is used here. 

SolPOC and COPS code (the previous name, a code version written in Scilab) has already supported scientific publications: 
- A. Grosjean et al, Influence of operating conditions on the optical optimization of solar selective absorber coatings, Solar Energy Materials and Solar Cells, Volume 230, 2021, 111280, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2021.111280.
- A. Grosjean et al, Replacing silver by aluminum in solar mirrors by improving solar reflectance with dielectric top layers, Sustainable Materials and Technologies, Volume 29, 2021, e00307, ISSN 2214-9937, https://doi.org/10.1016/j.susmat.2021.e00307.
- A. Grosjean et al, Comprehensive simulation and optimization of porous SiO2 antireflective coating to improve glass solar transmittance for solar energy applications, Solar Energy Materials and Solar Cells, Volume 182, 2018, Pages 166-177, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2018.03.040.
- Danielle Ngoue, Antoine Grosjean, (...), Ceramics for concentrated solar power (CSP): From thermophysical properties to solar absorbers, In Elsevier Series on Advanced Ceramic Materials, Advanced Ceramics for Energy Conversion and Storage, Elsevier, 2020, Pages 89-127, ISBN 9780081027264,

## Citation
If you use SolPOC and it is relevant to your work, please cite the papers associated with. Another paper, just for SolPOC, is on the way.

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
Even if SolPOC is quite simple code, this is a research-grade program. We actually do research with it. 
Do not hesitate to contact us, for help, academic projects or cited to current version of our work.

## Contributors
The main author and maintainer of the code is Antoine Grosjean, EPF, France (antoine.grosjean@epf.fr). 

Here is a list of major contributors to SolPOC : 
* Pauline Bennet, UCA, France (@Ellawin)
* Antoine Moreau, UCA, France  (@AnMoreau)
* Andrey Soum-Glaude, PROMES-CNRS, France
  
Many thanks to Thalita Drumond (@thalitadru), Titouan Février and Antoine Gademer from EPF, France. 
We would like to thank to Denis Langevin Université Clermont Auvergne France and Amine Mahammou, PROMES-CNRS, France
