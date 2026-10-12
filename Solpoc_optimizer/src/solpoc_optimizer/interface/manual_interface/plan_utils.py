import json
from datetime import datetime
from pathlib import Path

from solpoc_optimizer.paths import (
    PLAN_EXPERIENCE_DIR,
    create_project_directories,
)


VAR_TO_JSON = {
    "Comment": "Comment",
    "Mat_Stack": "Mat_Stack",
    "Wl": "Wl",
    "Th_Substrate": "Th_Substrate",
    "Th_range": "Th_range",
    "n_range": "n_range",
    "nb_layer": "nb_layer",
    "Ang": "Ang",
    "pop_size": "pop_size",
    "crossover_rate": "crossover_rate",
    "f1": "f1",
    "f2": "f2",
    "mutation_DE": "mutation_DE",
    "budget": "budget",
    "nb_run": "nb_run",
    "cpu_used": "cpu_used",
    "seed": "seed",
    "d_Stack_Opt": "d_Stack_Opt",
    "Lambda_cut_1": "Lambda_cut_1",
    "lambda_cut_1": "lambda_cut_1",
    "lambda_cut_2": "lambda_cut_2",
    "Mat_Option": "Mat_Option",
    "Mode_choose_material": "Mode_choose_material",
    "vf_range": "vf_range",
    "C": "C",
    "T_air": "T_air",
    "T_abs": "T_abs",
    "algo": "algo",
    "cost_function": "cost_function",
    "selection": "selection",
}


BASE_SCHEMA = {
    "template": None,
    "Comment": None,
    "Wl": None,
    "open_SolSpec": None,
    "open_Spec_Signal": None,
    "Ang": 0,
    "Sol_Spec": None,
    "name_Sol_Spec": None,
    "d_Stack": None,
    "Mat_Stack": None,
    "n_Stack": None,
    "k_Stack": None,
    "vf": None,
    "Th_range": None,
    "Th_Substrate": None,
    "vf_range": None,
    "Lambda_cut_1": None,
    "Lambda_cut_2": None,
    "pop_size": None,
    "Th_Substrate": None,
    "vf_range": None,
    "Lambda_cut_1": None,
    "Lambda_cut_2": None,
    "pop_size": None,
    "crossover_rate": None,
    "f1": None,
    "f2": None,
    "mutation_DE": None,
    "budget": None,
    "nb_run": None,
    "cpu_used": None,
    "seed": None,
    "algo": None,
    "cost_function": None,
    "selection": None,
    "nb_layer": None,
    "n_range": None,
    "d_Stack_Opt": None,
    "C": None,
    "T_air": None,
    "T_abs": None,
    "Signal_H_eye": None,
    "poids_PV": None,
    "Signal_PV": None,
    "Signal_Th": None,
    "Signal_fit": None,
    "Signal_fit_2": None,
    "precision_AlgoG": None,
    "mutation_rate": None,
    "mutation_delta": None,
    "evaluate_rate": None,
    "Mat_Option": None,
    "coherency_limit": None,
    "Mode_choose_material": None,
}


def parse_value(value, json_key):
    """
    Convertit une valeur Python brute en une valeur
    compatible avec la sérialisation JSON.
    """

    if value is None:
        return None

    # Wl : ndarray, tuple ou liste → liste de nombres
    if json_key == "Wl":
        try:
            return [int(v) if float(v) == int(v) else float(v) for v in value]
        except (TypeError, ValueError):
            return list(value)

    # Intervalles : tuple → liste
    if json_key in {"Th_range", "n_range", "vf_range"}:
        if isinstance(value, (tuple, list)):
            return list(value)

        return value

    # Entiers
    if json_key in {
        "pop_size",
        "budget",
        "nb_run",
        "cpu_used",
        "nb_layer",
    }:
        return int(value)

    # Nombres flottants
    if json_key in {
        "crossover_rate",
        "f1",
        "f2",
        "Ang",
        "Th_Substrate",
        "Lambda_cut_1",
        "Lambda_cut_2",
        "lambda_cut_1",
        "lambda_cut_2",
        "C",
        "T_air",
        "T_abs",
    }:
        return float(value)

    # Seed : entier ou None
    if json_key == "seed":
        return int(value)

    # Listes de matériaux
    if json_key in {"Mat_Stack", "Mat_Option"}:
        if isinstance(value, list):
            return value

        return list(value)

    # Liste mixte, par exemple ["no", "no", 10]
    if json_key == "d_Stack_Opt":
        if isinstance(value, list):
            return value

        return list(value)

    # Les fonctions SolPOC ne peuvent pas être enregistrées
    # directement dans un fichier JSON.
    # On conserve donc uniquement leur nom.
    if json_key in {
        "algo",
        "cost_function",
        "selection",
    }:
        if callable(value):
            return value.__name__

        return str(value).removeprefix("sol.")

    return value


import numpy as np


def compact_wavelength_range(value) -> list:
    """
    Convertit un domaine de longueurs d'onde en format compact :

        [début, fin, pas]

    Exemple :
        np.arange(400, 800, 5)
        devient
        [400, 800, 5]
    """

    # Le format est déjà compact
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [
            parse_number(value[0]),
            parse_number(value[1]),
            parse_number(value[2]),
        ]

    wavelengths = np.asarray(value)

    if wavelengths.ndim != 1:
        raise ValueError("Wl doit être un tableau NumPy à une dimension.")

    if wavelengths.size < 2:
        raise ValueError(
            "Wl doit contenir au moins deux longueurs d'onde pour déterminer le pas."
        )

    steps = np.diff(wavelengths)
    step = steps[0]

    # Vérification que le pas est constant
    if not np.allclose(steps, step):
        raise ValueError(
            "Le domaine Wl n'est pas régulièrement espacé. "
            "Il ne peut pas être converti au format "
            "[début, fin, pas]."
        )

    start = wavelengths[0]

    # np.arange exclut la borne supérieure.
    # On la retrouve avec dernière valeur + pas.
    stop = wavelengths[-1] + step

    return [
        parse_number(start),
        parse_number(stop),
        parse_number(step),
    ]


def parse_number(value):
    """
    Transforme un nombre NumPy en nombre Python compatible JSON.
    """

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def generate_json(
    local_vars: dict,
    template_name: str,
    priority: int,
) -> Path:
    """
    Construit et sauvegarde le plan JSON à partir
    des variables locales d'un template.

    Le fichier est enregistré dans le dossier
    PLAN_EXPERIENCE_DIR défini dans paths.py.
    """

    # Crée le workspace et tous ses sous-dossiers
    # lorsqu'ils n'existent pas encore.
    create_project_directories()

    experiment = BASE_SCHEMA.copy()
    experiment["template"] = template_name

    for variable_name, json_key in VAR_TO_JSON.items():
        if variable_name not in local_vars:
            continue

        value = local_vars[variable_name]

        if variable_name == "Wl" or json_key == "Wl":
            experiment[json_key] = compact_wavelength_range(value)
        else:
            experiment[json_key] = parse_value(
                value,
                json_key,
            )

    # Sécurité supplémentaire :
    # le dossier plan_experience doit exister.
    PLAN_EXPERIENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_slug = template_name.strip().replace(" ", "_")

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss_%f")

    filename = f"{template_slug}_{timestamp}_{priority}.json"

    filepath = PLAN_EXPERIENCE_DIR / filename

    with filepath.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Plan sauvegardé : {filepath.resolve()}")

    return filepath
