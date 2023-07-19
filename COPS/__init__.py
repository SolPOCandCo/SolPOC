# encoding: utf-8
#
#Copyright (C) 2023, Antoine Grosjean
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
    COPS - a code designed to solve Maxwell's equations in a multilayered thin film structure.
    COPS is specifically designed for research in coatings, thin film deposition, and materials 
    research for solar energy applications (thermal and PV).
    The code uses a stable method to quickly calculate reflectivity, transmissivity, and absorptivity 
    from a stack of thin films over a full solar spectrum. COPS comes with several optimization methods, 
    a multiprocessing pool, and a comprehensive database of refractive indices for real materials.
    In the end, COPS is simple to use for no-coder users thanks to main script, which regroup all necessary 
    variables and automatically save important results in text files and PNG images.
"""
__name__ = 'COPS'
__version__ = '0.9'
__date__ = "19/07/2023"   # MM/DD/YYY
__author__ = 'Antoine Grosjean'


## make accessible everything from `core` directly from the COPS base package
from COPS.core import *
from COPS.function_COPS import *
