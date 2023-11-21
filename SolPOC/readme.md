Welcome to the main folder of SolPOC. This folder contain the code with the refractive index necessary for the code. 
As commande as `pip instal` is not available, install SolPOC mean downloading the full folder. 
Pip install as no interest, as we recommand to directly use the main scrip `optimization_multiprocess.py`. 

This folder contain :

- A subfolder name Materials, witch contain data for the code. The data as present as texte files for be readable for the code. Materials folder contain refractive index of real materials; mostly founded is peer reviewed litterature of refractiveindex.info web site. Other data, as solar spectra are also available
- A `CurveRTA.py` files. This files is NOT for optimization. This files is only for plot the RTA (meaning reflectivity, transmissivity and absoprtivity) of the thin layer stack. If you know the thin layers thicknesses, use this script. 
- The main script `optimization_multiprocess.py`. This is how SolPOC is supose to be used. Open the `optimization_multiprocess.py`and change the parameters between the two lines.If you do not know the thin layers thicknesses and want optimized then, use this script. 
- A `function_SolPOC.py`. This files contain all functions for `CurveRTA.py`and `optimization_multiprocess.py` work. 
