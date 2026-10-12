# src/solpoc_optimizer/hashing.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from solpoc_optimizer.paths import HASHES_FILE


# Ces clés sont ajoutées ou utilisées par le scheduler,
# mais ne décrivent pas le contenu scientifique de l'expérience.
_IGNORED_KEYS = {
    "filename",
    "priority",
    "cpu_used",
}


def hash_plan(plan_dict: dict) -> str:
    """
    Calcule un hash MD5 déterministe pour un plan d'expérience.

    Les clés techniques sont ignorées. Deux plans scientifiquement
    identiques produisent donc le même hash, même si leur nom,
    leur priorité ou le nombre de CPU utilisés diffèrent.
    """

    cleaned_plan = {
        key: value for key, value in plan_dict.items() if key not in _IGNORED_KEYS
    }

    json_string = json.dumps(
        cleaned_plan,
        sort_keys=True,
        default=str,
    )

    return hashlib.md5(json_string.encode("utf-8")).hexdigest()


def _resolve_hashes_file(
    hashes_file: Path | str | None,
) -> Path:
    """
    Retourne le chemin du fichier de hashes à utiliser.

    Si aucun chemin n'est fourni, le chemin centralisé défini
    dans solpoc_optimizer.paths est utilisé.
    """

    if hashes_file is None:
        return HASHES_FILE

    return Path(hashes_file)


def load_hashes_db(
    hashes_file: Path | str | None = None,
) -> dict:
    """
    Charge la base des hashes déjà exécutés.

    Renvoie un dictionnaire vide si le fichier n'existe pas
    ou s'il est vide.
    """

    target_file = _resolve_hashes_file(hashes_file)

    if not target_file.exists():
        return {}

    with target_file.open("r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        return {}

    return json.loads(content)


def save_hashes_db(
    hashes_db: dict,
    hashes_file: Path | str | None = None,
) -> None:
    """Sauvegarde la base des hashes dans un fichier JSON."""

    target_file = _resolve_hashes_file(hashes_file)

    target_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with target_file.open("w", encoding="utf-8") as file:
        json.dump(
            hashes_db,
            file,
            indent=2,
            ensure_ascii=False,
        )


def is_already_executed(
    plan_hash: str,
    hashes_db: dict,
) -> bool:
    """Vérifie si le hash d'un plan est déjà enregistré."""

    return plan_hash in hashes_db


def register_executed_plan(
    plan_hash: str,
    filename: str,
    hashes_db: dict,
    hashes_file: Path | str | None = None,
) -> None:
    """
    Enregistre un plan exécuté dans la base en mémoire
    et sauvegarde immédiatement cette base sur le disque.
    """

    hashes_db[plan_hash] = filename

    save_hashes_db(
        hashes_db,
        hashes_file,
    )
