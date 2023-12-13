import os
import shutil as sh
from importlib import resources
pkg_path = resources.files(__package__)

def init():
    src = pkg_path.joinpath('scripts').as_posix()
    dst = base_dst = "ProjectSolPOC"
    i = 0
    while os.path.exists(dst):
        dst = base_dst + f"_{i}"
    sh.copytree(src, dst)
    sh.rmtree(os.path.join(dst, '__pycache__'))
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