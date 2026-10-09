import json
import hashlib
from pathlib import Path


# Clés ignorées car elles ne font pas partie du contenu scientifique du plan
_IGNORED_KEYS = {"filename", "priority", "cpu_used"}


def hash_plan(plan_dict: dict) -> str:
    """
    Calcule un hash MD5 déterministe d'un plan d'expérience.

    Les clés techniques (filename, priority, cpu_used) sont ignorées :
    deux plans identiques scientifiquement mais avec des priorités différentes
    produiront le même hash.

    Args:
        plan_dict: Le plan d'expérience sous forme de dictionnaire.

    Returns:
        Le hash MD5 du plan sous forme de chaîne hexadécimale.
    """
    cleaned = {k: v for k, v in plan_dict.items() if k not in _IGNORED_KEYS}

    json_string = json.dumps(cleaned, sort_keys=True, default=str)

    return hashlib.md5(json_string.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Gestion du cache hashes.json
# ---------------------------------------------------------------------------

# Valeur par défaut (pour compatibilité rétroactive)
_DEFAULT_HASHES_FILE = (
    Path(__file__).resolve().parent.parent
    / "experiences_scheduler"
    / "plan_executer"
    / "hashes.json"
)


def load_hashes_db(hashes_file: Path | str = None) -> dict:
    """
    Charge le cache des hashes depuis le fichier spécifié.

    Args:
        hashes_file: Chemin vers le fichier hashes.json.
                     Si None, utilise le chemin par défaut.

    Returns:
        Dictionnaire { hash: filename } des plans déjà exécutés.
        Renvoie un dict vide si le fichier n'existe pas encore.
    """
    if hashes_file is None:
        hashes_file = _DEFAULT_HASHES_FILE

    hashes_file = Path(hashes_file)

    if not hashes_file.exists():
        return {}

    with open(hashes_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def save_hashes_db(hashes_db: dict, hashes_file: Path | str = None) -> None:
    """
    Sauvegarde le cache des hashes dans le fichier spécifié.

    Args:
        hashes_db: Dictionnaire { hash: filename } à persister.
        hashes_file: Chemin vers le fichier hashes.json.
                     Si None, utilise le chemin par défaut.
    """
    if hashes_file is None:
        hashes_file = _DEFAULT_HASHES_FILE

    hashes_file = Path(hashes_file)
    hashes_file.parent.mkdir(parents=True, exist_ok=True)

    with open(hashes_file, "w", encoding="utf-8") as f:
        json.dump(hashes_db, f, indent=2, ensure_ascii=False)


def is_already_executed(plan_hash: str, hashes_db: dict) -> bool:
    """
    Vérifie si un plan a déjà été exécuté.

    Args:
        plan_hash: Le hash MD5 du plan à vérifier.
        hashes_db: Le cache chargé via load_hashes_db().

    Returns:
        True si le plan a déjà été exécuté, False sinon.
    """
    return plan_hash in hashes_db


def register_executed_plan(
    plan_hash: str, filename: str, hashes_db: dict, hashes_file: Path | str = None
) -> None:
    """
    Enregistre un plan comme exécuté dans le cache (en mémoire + sur disque).

    Args:
        plan_hash: Le hash MD5 du plan.
        filename: Le nom du fichier JSON correspondant.
        hashes_db: Le cache en mémoire (modifié en place).
        hashes_file: Chemin vers le fichier hashes.json.
                     Si None, utilise le chemin par défaut.
    """
    hashes_db[plan_hash] = filename
    save_hashes_db(hashes_db, hashes_file)
