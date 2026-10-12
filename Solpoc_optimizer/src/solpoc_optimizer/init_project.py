from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

from solpoc_optimizer.paths import (
    MANUAL_PLANS_DIR,
    USER_NEW_FUNCTIONS_DIR,
    WORKSPACE_DIR,
    create_project_directories,
)


# Packages contenant les fichiers modèles
MANUAL_INTERFACE_PACKAGE = "solpoc_optimizer.interface.manual_interface"

NEW_FUNCTIONS_PACKAGE = "solpoc_optimizer.new_functions"


def _copy_resource_tree(
    source: Any,
    destination: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Copie récursivement les ressources d'un package
    vers un dossier réel du système.

    Args:
        source:
            Ressource obtenue avec importlib.resources.files().

        destination:
            Dossier de destination dans le workspace.

        overwrite:
            Écrase les fichiers existants lorsque True.

    Returns:
        Un tuple :
        - nombre de fichiers copiés ;
        - nombre de fichiers préservés.
    """

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_files = 0
    preserved_files = 0

    for item in source.iterdir():
        # Ignore les caches Python.
        if item.name == "__pycache__":
            continue

        if item.name.endswith((".pyc", ".pyo")):
            continue

        target = destination / item.name

        if item.is_dir():
            copied, preserved = _copy_resource_tree(
                source=item,
                destination=target,
                overwrite=overwrite,
            )

            copied_files += copied
            preserved_files += preserved
            continue

        if target.exists() and not overwrite:
            preserved_files += 1
            continue

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with item.open("rb") as source_file:
            with target.open("wb") as target_file:
                shutil.copyfileobj(
                    source_file,
                    target_file,
                )

        copied_files += 1

    return copied_files, preserved_files


def _create_workspace_readme() -> None:
    """Crée une documentation minimale dans le workspace."""

    readme_path = WORKSPACE_DIR / "README.md"

    if readme_path.exists():
        return

    content = """# SolPOC Optimizer workspace

This directory contains the editable files and generated results
used by SolPOC Optimizer.

## manual_interface

Contains editable copies of the experiment templates.

You may modify the parameters directly in these files without
changing the installed package.

## new_functions

Contains editable copies of the additional Python functions used
by the scheduler.

The scheduler loads these external copies in priority.

## Other directories

- plan_experience: plans waiting to be executed
- plan_executer: successfully executed plans
- plan_failed: plans that could not be executed
- runs: experiment results
"""

    readme_path.write_text(
        content,
        encoding="utf-8",
    )


def initialize_workspace(
    overwrite: bool = False,
) -> Path:
    """
    Initialise le workspace de SolPOC Optimizer.

    Copie :
    - les templates de manual_interface ;
    - les fonctions de new_functions.

    Les fichiers déjà présents sont conservés par défaut.
    """

    create_project_directories()

    manual_source = resources.files(MANUAL_INTERFACE_PACKAGE)

    functions_source = resources.files(NEW_FUNCTIONS_PACKAGE)

    manual_copied, manual_preserved = _copy_resource_tree(
        source=manual_source,
        destination=MANUAL_PLANS_DIR,
        overwrite=overwrite,
    )

    functions_copied, functions_preserved = _copy_resource_tree(
        source=functions_source,
        destination=USER_NEW_FUNCTIONS_DIR,
        overwrite=overwrite,
    )

    _create_workspace_readme()

    print()
    print("SolPOC Optimizer workspace initialized:")
    print(WORKSPACE_DIR.resolve())

    print()
    print("Manual interface templates:")
    print(f"  Copied: {manual_copied}")
    print(f"  Preserved: {manual_preserved}")
    print(f"  Directory: {MANUAL_PLANS_DIR.resolve()}")

    print()
    print("Custom functions:")
    print(f"  Copied: {functions_copied}")
    print(f"  Preserved: {functions_preserved}")
    print(f"  Directory: {USER_NEW_FUNCTIONS_DIR.resolve()}")

    return WORKSPACE_DIR


def main() -> int:
    """
    Point d'entrée de la commande
    solpoc-optimizer-init.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Initialize the SolPOC Optimizer workspace "
            "and copy editable templates and functions."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing templates and custom "
            "functions. User modifications may be lost."
        ),
    )

    arguments = parser.parse_args()

    initialize_workspace(
        overwrite=arguments.force,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
