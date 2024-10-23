import os
import shutil as sh
from importlib import resources
pkg_path = resources.files(__package__)

def init():
    """"Function executed when user calls solpoc-init"""
    src = pkg_path.joinpath('scripts').as_posix()
    dst = base_dst = "ProjectSolPOC"
    i = 2
    while os.path.exists(dst):
        dst = base_dst + f"_{i}"
        i += 1
    sh.copytree(src, dst)
    sh.rmtree(os.path.join(dst, '__pycache__'), ignore_errors=True)
    print(f"SolPOC project directory created at {os.path.abspath(dst)}")
    # Create a placeholder materials directory
    materials_path = os.path.join(dst, 'Materials')
    os.makedirs(materials_path, exist_ok=True)
    readme = ("Created by SolPOC\n"
        "*** This is your custom materials folder  ***\n"
        "To include a new material XXX, simple create a file XXX.txt in this folder "
        "and it will be found by SolPOC."
    )
    readme_path = os.path.join(materials_path, 'readme.md')
    with open(readme_path, 'w') as f:
        f.write(readme)

    # Create a placeholder Fit directory
        # Create a placeholder materials directory
    fit_path = os.path.join(dst, 'Materials')
    os.makedirs(fit_path, exist_ok=True)
    readme = ("Created by SolPOC\n"
        "*** Folder reserved to store files for signal fitting  ***\n"
        "The reference file containing the signal to fit "
        "should be named signal_fit.txt "
    )
    readme_path = os.path.join(fit_path, 'readme.md')
    with open(readme_path, 'w') as f:
        f.write(readme)