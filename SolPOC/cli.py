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
