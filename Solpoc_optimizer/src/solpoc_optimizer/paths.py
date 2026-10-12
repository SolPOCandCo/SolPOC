from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------
# DOSSIERS INTERNES DU PACKAGE
# ---------------------------------------------------------

# Dossier src/solpoc_optimizer/
PACKAGE_DIR = Path(__file__).resolve().parent

# Interface installée dans le package
INTERFACE_PACKAGE_DIR = PACKAGE_DIR / "interface"

# Modèles originaux fournis avec le package
DEFAULT_MANUAL_INTERFACE_DIR = INTERFACE_PACKAGE_DIR / "manual_interface"

# Fonctions personnalisées originales fournies avec le package
DEFAULT_NEW_FUNCTIONS_DIR = PACKAGE_DIR / "new_functions"


# ---------------------------------------------------------
# WORKSPACE UTILISATEUR
# ---------------------------------------------------------


def get_workspace_dir() -> Path:
    """
    Retourne le dossier de travail de SolPOC Optimizer.

    L'utilisateur peut définir un emplacement personnalisé avec
    la variable d'environnement SOLPOC_OPTIMIZER_HOME.
    """

    custom_workspace = os.getenv("SOLPOC_OPTIMIZER_HOME")

    if custom_workspace:
        return Path(custom_workspace).expanduser().resolve()

    return Path.cwd() / "workspace"


WORKSPACE_DIR = get_workspace_dir()


# ---------------------------------------------------------
# COPIES MODIFIABLES PAR L'UTILISATEUR
# ---------------------------------------------------------

# Copie externe des templates de l'interface manuelle
MANUAL_PLANS_DIR = WORKSPACE_DIR / "manual_interface"

# Copie externe des fonctions personnalisées
USER_NEW_FUNCTIONS_DIR = WORKSPACE_DIR / "new_functions"


# ---------------------------------------------------------
# DOSSIERS D'EXÉCUTION
# ---------------------------------------------------------

PLAN_EXPERIENCE_DIR = WORKSPACE_DIR / "plan_experience"

PLAN_EXECUTED_DIR = WORKSPACE_DIR / "plan_executer"

PLAN_FAILED_DIR = WORKSPACE_DIR / "plan_failed"

RUNS_DIR = WORKSPACE_DIR / "runs"

HASHES_FILE = PLAN_EXECUTED_DIR / "hashes.json"


# ---------------------------------------------------------
# CRÉATION DES DOSSIERS
# ---------------------------------------------------------


def create_project_directories() -> None:
    """
    Crée le workspace et ses sous-dossiers.

    Cette fonction ne copie pas les modèles.
    La copie est réalisée par initialize_workspace().
    """

    directories = [
        WORKSPACE_DIR,
        MANUAL_PLANS_DIR,
        USER_NEW_FUNCTIONS_DIR,
        PLAN_EXPERIENCE_DIR,
        PLAN_EXECUTED_DIR,
        PLAN_FAILED_DIR,
        RUNS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
