import numpy as np
import matplotlib

matplotlib.use("Agg")  # Backend non-interactif (pas d'affichage)
import matplotlib.pyplot as plt
import time
import os
import re
import solpoc as sol


from datetime import datetime
from multiprocessing import Pool
import json
import ast
import sys

from contextlib import contextmanager
from pathlib import Path
# import importlib


# Ensure package imports work when running this script directly: add parent and project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from solpoc_optimizer.hashing import (
    hash_plan,
    load_hashes_db,
    register_executed_plan,
    is_already_executed,
)


from solpoc_optimizer.paths import (
    WORKSPACE_DIR,
    PLAN_EXPERIENCE_DIR,
    PLAN_EXECUTED_DIR,
    PLAN_FAILED_DIR,
    RUNS_DIR,
    HASHES_FILE,
    create_project_directories,
)

"""


"""


"""
SolPOC v 0.9.7
@authors: De Mongolfier.Guillaume, Gharbi Yassine
contact : yassine.gharbi@epfedu.fr et guillaume.demontgolfier@epfedu.fr

Other contributors:
    - ``Stack_plot`` function by Titouan Fevrier
    - Numpy-style formatted docstrings for automatic sphinx doc generation by Titouan Fevrier

Description:
    This script implements an experience scheduler for the SolPOC project. It reads experiment plans from JSON files, checks for duplicates using hashing, executes the experiments in parallel, and organizes results with detailed logging and visualization.

Usage:
    1. Place JSON experiment plans in the `plan_experience` directory.
    2. Run this script to execute the experiments. Results will be saved in the `runs` directory, organized by timestamp and priority.
    3. The script automatically handles duplicates, errors, and generates comprehensive reports and plots for each experiment.
Launch : 
    to start the scheduler make sure to be in the root of the repository and run:

        python -m Solpoc_optimizer.experiences_scheduler.projet_final
"""


try:
    from solpoc import function_R_s_weighted as test
except ImportError as error:
    print(
        "WARNING: function_R_s_weighted module not found. "
        f"Falling back to sol module. Details: {error}"
    )
    test = sol


def get_from_modules(name):
    try:
        return getattr(test, name)
    except (AttributeError, Exception):
        return getattr(sol, name)


def build_params(row):
    allowed_fields = {
        "Wl",
        "Ang",
        "Sol_Spec",
        "name_Sol_Spec",
        "d_Stack",
        "Mat_Stack",
        "n_Stack",
        "k_Stack",
        "vf",
        "Th_range",
        "Th_Substrate",
        "vf_range",
        "Lambda_cut_1",
        "Lambda_cut_2",
        "pop_size",
        "crossover_rate",
        "f1",
        "f2",
        "mutation_DE",
        "budget",
        "algo",
        "cost_function",
        "selection",
        "n_range",
        "d_Stack_Opt",
        "C",
        "T_air",
        "T_abs",
        "Signal_H_eye",
        "poids_PV",
        "Signal_PV",
        "Signal_Th",
        "Signal_fit",
        "Signal_fit_2",
        "precision_AlgoG",
        "mutation_rate",
        "mutation_delta",
        "evaluate_rate",
        "Mat_Option",
        "coherency_limit",
        "Mode_choose_material",
        "nb_run",
        "seed",
        "nb_layer",
    }

    params = {}
    for key, value in row.items():
        if key not in allowed_fields:
            continue
        # Ignore None
        if value is None:
            continue
        # Ignore NaN
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        params[key] = value

    # Casts spéciaux
    int_fields = ["nb_run", "seed", "nb_layer"]
    for field in int_fields:
        if field in params:
            params[field] = int(params[field])

    return params


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convertit le tableau en liste Python
        if isinstance(obj, (np.integer,)):
            return int(obj)  # Convertit les entiers NumPy
        if isinstance(obj, (np.floating,)):
            return float(obj)  # Convertit les flottants NumPy
        return super().default(obj)


@contextmanager
def working_directory(directory: Path):
    """
    Change temporairement le dossier courant,
    puis revient au dossier initial.
    """

    previous_directory = Path.cwd()

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.chdir(directory)

    try:
        yield
    finally:
        os.chdir(previous_directory)


def main_for_parameters(
    parameters,
    nb_run,
    launch_time,
    Comment,
    name_Sol_Spec,
    cpu_used,
    algo,
    cost_function,
    selection,
):
    directory = parameters["directory"]
    with working_directory(WORKSPACE_DIR):
        sol.run_main(parameters)

    mp_pool = Pool(cpu_used)
    tasks = [(i, parameters, algo, cost_function, selection) for i in range(nb_run)]
    results = mp_pool.map(run_problem_solution, tasks)
    mp_pool.close()
    mp_pool.join()

    tab_best_solution, tab_dev, tab_perf, tab_n_iter, tab_temps, tab_seed = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for best_solution, perf, dev, n_iter, temps, seed in results:
        tab_best_solution.append(best_solution)
        tab_dev.append(dev)
        tab_perf.append(perf)
        tab_n_iter.append(n_iter)
        tab_temps.append(temps)
        tab_seed.append(seed)

    Experience_results = {
        "tab_perf": tab_perf,
        "tab_dev": tab_dev,
        "tab_best_solution": tab_best_solution,
        "tab_n_iter": tab_n_iter,
        "tab_temps": tab_temps,
        "tab_seed": tab_seed,
        "Comment": Comment,
        "language": "en",
        "name_Sol_Spec": name_Sol_Spec,
        "launch_time": launch_time,
        "cpu_used": cpu_used,
        "nb_run": nb_run,
    }

    end_of_time = time.time()
    time_real = end_of_time - parameters["dawn_of_time"]
    parameters.update({"time_real": time_real})

    print("Total real time: {:.2f}s".format(time_real))
    print("Processor calculation time: {:.2f}s".format(sum(tab_temps)))

    sol.Explain_results(parameters, Experience_results)
    sol.Convergences_txt(parameters, Experience_results, directory)
    sol.Generate_txt(parameters, Experience_results, directory)
    sol.Optimization_txt(parameters, Experience_results, directory)
    sol.Generate_materials_txt(parameters, Experience_results, directory)

    sol.Reflectivity_plot(parameters, Experience_results, directory)
    print("Graphique de réflectivité sauvegardé.")

    sol.Transmissivity_plot(parameters, Experience_results, directory)
    print("Graphique de transmissivité sauvegardé.")

    sol.OpticalStackResponse_plot(parameters, Experience_results, directory)
    print("Graphique de réponse de la pile optique sauvegardé.")

    sol.Convergence_plots(parameters, Experience_results, directory)
    print("Graphiques de convergence sauvegardés.")

    sol.Convergence_plots_2(parameters, Experience_results, directory)
    print("Graphiques de convergence 2 sauvegardés.")

    sol.Consistency_curve_plot(parameters, Experience_results, directory)
    print("Graphique de courbe de cohérence sauvegardé.")

    sol.Optimum_thickness_plot(parameters, Experience_results, directory)
    print("Graphique d'épaisseur optimale sauvegardé.")

    sol.Optimum_refractive_index_plot(parameters, Experience_results, directory)
    print("Graphique d'indice de réfraction optimale sauvegardé.")

    sol.Volumetric_parts_plot(parameters, Experience_results, directory)
    print("Graphique des parties volumétriques sauvegardé.")

    sol.Stack_plot(parameters, Experience_results, directory)
    print("Graphique de la pile sauvegardé.")


def build_wl(Wl):
    if isinstance(Wl, str):
        return get_from_modules(Wl)()
    elif isinstance(Wl, list) and len(Wl) == 3:
        start, stop, step = Wl
        return np.arange(start, stop, step)
    return Wl


def run_problem_solution(args):
    i, parameters, algo, cost_function, selection = args
    t1 = time.time()

    this_run_params = {}
    this_run_params.update(parameters)
    this_run_params["seed"] = parameters["seed_list"][i]

    best_solution, dev, n_iter, seed = algo(cost_function, selection, this_run_params)

    t2 = time.time()
    temps = t2 - t1

    if type(best_solution) != list:
        best_solution = best_solution.tolist()
    best_solution = np.array(best_solution)
    dev = np.array(dev)
    perf = cost_function(best_solution, parameters)

    print(
        "I finished case #",
        str(i + 1),
        " in ",
        "{:.1f}".format(temps),
        " seconds.",
        " Best: ",
        "{:.4f}".format(perf),
        flush=True,
    )

    return best_solution, perf, dev, n_iter, temps, seed


def _run_bragg_flow(parameters, directory, nb_run, algo, cost_function, selection):
    """
    Exécution séquentielle inspirée de template_Bragg_mirror.py.
    Lance nb_run optimisations, garde la meilleure, génère tous les plots.
    """
    results = []

    for i, seed in enumerate(parameters["seed_list"]):
        t1 = time.time()

        this_run_params = parameters.copy()
        this_run_params["seed"] = seed

        best_solution, dev, n_iter, seed_used = algo(
            cost_function, selection, this_run_params
        )

        t2 = time.time()
        temps = t2 - t1

        best_solution = np.array(best_solution)
        dev = np.array(dev)
        perf = cost_function(best_solution, parameters)

        print(
            f"I finished case #{i + 1} in {temps:.1f} seconds. Best: {perf:.4f}",
            flush=True,
        )
        results.append((best_solution, perf, dev, n_iter, temps, seed_used))

    # Meilleure solution globale (perf maximale)
    best_result = max(results, key=lambda x: x[1])
    best_solution_overall = best_result[0]

    # Structure attendue par les fonctions sol.*
    Experience_results = {
        "tab_perf": [r[1] for r in results],
        "tab_dev": [r[2] for r in results],
        "tab_best_solution": [r[0] for r in results],
        "tab_n_iter": [r[3] for r in results],
        "tab_temps": [r[4] for r in results],
        "tab_seed": [r[5] for r in results],
        "Comment": parameters.get("Comment", ""),
        "language": "en",
        "name_Sol_Spec": parameters.get("name_Sol_Spec"),
        "launch_time": datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"),
        "cpu_used": 1,
        "nb_run": nb_run,
        # Clé utilisée par Stack_plot
        "d_Stack": best_solution_overall,
    }

    parameters.update({"time_real": sum(r[4] for r in results)})

    # ── Fichiers texte ──────────────────────────────────────────────────────
    try:
        sol.Explain_results(parameters, Experience_results)
        sol.Convergences_txt(parameters, Experience_results, directory)
        sol.Generate_txt(parameters, Experience_results, directory)
        sol.Optimization_txt(parameters, Experience_results, directory)
        sol.Generate_materials_txt(parameters, Experience_results, directory)
    except Exception as e:
        print(f"[WARN] Fichiers texte : {e}")

    # ── Courbes RTA ─────────────────────────────────────────────────────────
    try:
        sol.Reflectivity_plot(parameters, Experience_results, directory)
        sol.Transmissivity_plot(parameters, Experience_results, directory)
        sol.OpticalStackResponse_plot(parameters, Experience_results, directory)
    except Exception as e:
        print(f"[WARN] Plots RTA : {e}")

    # ── Courbes de convergence ───────────────────────────────────────────────
    try:
        sol.Convergence_plots(parameters, Experience_results, directory)
        sol.Convergence_plots_2(parameters, Experience_results, directory)
    except Exception as e:
        print(f"[WARN] Convergence plots : {e}")

    # ── Courbe de cohérence ──────────────────────────────────────────────────
    try:
        sol.Consistency_curve_plot(parameters, Experience_results, directory)
    except Exception as e:
        print(f"[WARN] Consistency curve : {e}")

    # ── Stack ────────────────────────────────────────────────────────────────
    try:
        sol.Stack_plot(parameters, Experience_results, directory)
        print("Graphique de la pile sauvegardé.")
    except Exception as e:
        print(f"[WARN] Stack plot : {e}")


def remove_empty_run_directories() -> None:
    """Supprime les anciens dossiers vides présents dans runs."""

    if not RUNS_DIR.exists():
        return

    for item in sorted(
        RUNS_DIR.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not item.is_dir():
            continue

        try:
            item.rmdir()  # Fonctionne uniquement si le dossier est vide
            print(f"Ancien dossier vide supprimé : {item}")
        except OSError:
            pass


SOLPOC_DIRECTORY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}h\d{2}(?:-\d+)?$")


def remove_empty_solpoc_directories() -> None:
    """
    Supprime les dossiers vides créés automatiquement par SolPOC
    directement dans experiences_scheduler.
    """

    if not WORKSPACE_DIR.exists():
        return

    for item in WORKSPACE_DIR.iterdir():
        if not item.is_dir():
            continue

        # Ne sélectionne que les dossiers du type :
        # 2026-06-15-12h36
        # 2026-06-15-12h36-2
        if not SOLPOC_DIRECTORY_PATTERN.fullmatch(item.name):
            continue

        try:
            item.rmdir()  # Fonctionne uniquement si le dossier est vide
            print(f"Dossier SolPOC vide supprimé : {item}")
        except OSError:
            # Le dossier contient des fichiers, on le conserve.
            pass


# ---------------------------------------------------------------------------


def main() -> int:
    create_project_directories()
    remove_empty_run_directories()
    remove_empty_solpoc_directories()

    running = True
    launch_time_global = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    # ── POINT 1 : charger le cache des hashes au démarrage ──────────────────
    hashes_db = load_hashes_db(HASHES_FILE) or {}
    """
    if hashes_db:
        print(f"Cache chargé : {len(hashes_db)} experience(s) déjà exécutée(s).")
        print("\n=== Voici les expériences déjà exécutées ===")

        for plan_hash, filename in hashes_db.items():
            print(f"• {filename}")
            print(f"  hash : {plan_hash}")
            """
    # ────────────────────────────────────────────────────────────────────────

    plan_files = [f.name for f in PLAN_EXPERIENCE_DIR.glob("*.json")]
    plans = []
    for f in plan_files:
        with open(PLAN_EXPERIENCE_DIR / f, "r", encoding="utf-8") as file:
            plan = json.load(file)
            plan["filename"] = f
            filename_without_ext = f.replace(".json", "")
            try:
                priority_str = filename_without_ext.rsplit("_", 1)[-1]
                plan["priority"] = int(priority_str)
            except (ValueError, IndexError):
                plan["priority"] = 999
            plans.append(plan)

    plans.sort(key=lambda x: x.get("priority", 999))

    while running:
        Wl_Sol = None
        Wl_PV = None
        name_Sol_Spec = None
        name_PV = None

        algo = None
        cost_function = None
        selection = None

        if not plans:
            running = False
            print("Aucune ligne restante à traiter. Fin de la boucle.")
            break

        first_min_row = plans.pop(0)
        template_name = first_min_row.get("template", "unknown")

        # ── POINT 2 : vérifier le hash AVANT toute exécution ────────────────
        plan_hash = hash_plan(first_min_row)

        if is_already_executed(plan_hash, hashes_db):
            already_filename = hashes_db[plan_hash]
            print(
                f"[SKIP] {first_min_row['filename']} — experience identique à "
                f"'{already_filename}' déjà exécutée (hash: {plan_hash[:8]}…)."
            )
            # Déplacer quand même vers plan_executer pour ne pas le relancer
            (PLAN_EXPERIENCE_DIR / first_min_row["filename"]).rename(
                PLAN_EXECUTED_DIR / first_min_row["filename"]
            )
            continue
        # ────────────────────────────────────────────────────────────────────

        original_row = first_min_row.copy()

        # Initialisation of parameter Comment
        Comment = first_min_row.get("Comment", "")

        # changement des variable a modifier
        first_min_row["Wl"] = build_wl(first_min_row["Wl"])

        # utilise la fonction de Mat_Stack si Mat_Stack est un str
        if isinstance(first_min_row["Mat_Stack"], str):
            func_name = first_min_row["Mat_Stack"].split("(")[0]
            args_str = first_min_row["Mat_Stack"].split("(")[1].rstrip(")")
            args = ast.literal_eval(f"[{args_str}]")
            first_min_row["Mat_Stack"] = get_from_modules(func_name)(*args)

        # Pour selection
        if first_min_row.get("selection") is not None:
            selection = get_from_modules(first_min_row["selection"])

        # Pour algo
        if first_min_row.get("algo") is not None:
            algo = get_from_modules(first_min_row["algo"])

        # Pour cost_function
        if first_min_row.get("cost_function") is not None:
            cost_function = get_from_modules(first_min_row["cost_function"])

        # print(f"\nPremière ligne avec priorité minimale apres transformation : \n{first_min_row}")

        # Open the solar spectrum
        if isinstance(first_min_row["open_SolSpec"], str):
            args = [
                a.strip().strip("'\"") for a in first_min_row["open_SolSpec"].split(",")
            ]
            open_solspec = get_from_modules("open_SolSpec")
            Wl_Sol, first_min_row["Sol_Spec"], first_min_row["name_Sol_Spec"] = (
                open_solspec(*args)
            )
            name_Sol_Spec = first_min_row["name_Sol_Spec"]

        if isinstance(first_min_row["open_Spec_Signal"], str):
            args = [
                a.strip().strip("'\"")
                for a in first_min_row["open_Spec_Signal"].split(",")
            ]
            # convertir en int les arguments qui sont des nombres
            args = [int(a) if a.isdigit() else a for a in args]
            open_signal = get_from_modules("open_Spec_Signal")
            Wl_PV, first_min_row["Signal_PV"], name_PV = open_signal(*args)

        if first_min_row["Mat_Stack"] is not None and first_min_row["Wl"] is not None:
            first_min_row["n_Stack"], first_min_row["k_Stack"] = sol.Made_Stack(
                first_min_row["Mat_Stack"], first_min_row["Wl"]
            )

        # ajouter les paramètres manquant
        if first_min_row["Wl"] is not None and Wl_Sol is not None:
            first_min_row["Sol_Spec"] = np.interp(
                first_min_row["Wl"], Wl_Sol, first_min_row["Sol_Spec"]
            )

        # ajouter les paramètres manquant
        if first_min_row["Wl"] is not None and Wl_PV is not None:
            first_min_row["Signal_PV"] = np.interp(
                first_min_row["Wl"], Wl_PV, first_min_row["Signal_PV"]
            )

        # mettre a jour les parametres
        params = build_params(first_min_row)

        # Ajouter les fonctions
        if "algo" in first_min_row and algo is not None:
            params["algo"] = algo
        if "cost_function" in first_min_row and cost_function is not None:
            params["cost_function"] = cost_function
        if "selection" in first_min_row and selection is not None:
            params["selection"] = selection

        # print(f"nb de layer : {params['nb_layer']}")
        # print(f"n_range : {params['n_range']}")
        # print(params)
        try:
            parameters = sol.get_parameters(**params)

        except Exception as error:
            filename = first_min_row["filename"]

            print(f"[ERROR] Impossible de préparer le plan '{filename}' : {error}")

            source = PLAN_EXPERIENCE_DIR / filename
            destination = PLAN_FAILED_DIR / filename

            PLAN_FAILED_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:
                if destination.exists():
                    destination.unlink()

                if source.exists():
                    source.rename(destination)

                    print(f"[FAILED] Plan déplacé dans : {destination}")

            except Exception as move_error:
                print(
                    f"[ERROR] Impossible de déplacer le plan "
                    f"dans plan_failed : {move_error}"
                )

            continue

        print(f"name_Sol_Spec dans parameters : {parameters.get('name_Sol_Spec')}")
        nb_run = params.get("nb_run", 1)

        # Lire cpu_used depuis le JSON, par défaut 4 si la clé est absente ou non convertible
        try:
            cpu_used = int(first_min_row["cpu_used"])
        except (KeyError, TypeError, ValueError):
            cpu_used = 4

        # dossier global pour tous les resultats
        global_run_dir = RUNS_DIR / launch_time_global
        global_run_dir.mkdir(exist_ok=True)

        # Capturer l'heure de résultat de ce plan d'expérience spécifique
        result_time = datetime.now().strftime("%Hh%Mm%Ss")

        # Dossier spécifique à cette expérience
        priority = first_min_row.get("priority", "unknown")
        directory = global_run_dir / f"{template_name}_{result_time}_{priority}"
        directory.mkdir(exist_ok=True)

        parameters["directory"] = str(directory)
        parameters["dawn_of_time"] = time.time()

        # Exécuter le main pour cette ligne
        try:
            # Special handling for certain example templates
            t_lower = (template_name or "").lower()
            print(f"Processing template: {template_name}")
            if "bragg" in t_lower:
                print("#" * 1000)
                _run_bragg_flow(
                    parameters, str(directory), nb_run, algo, cost_function, selection
                )
            else:
                main_for_parameters(
                    parameters,
                    nb_run,
                    launch_time_global,
                    Comment,
                    name_Sol_Spec,
                    cpu_used,
                    algo=algo,
                    cost_function=cost_function,
                    selection=selection,
                )
            # Si réussi, déplacer vers plan_executer
            (PLAN_EXPERIENCE_DIR / first_min_row["filename"]).rename(
                PLAN_EXECUTED_DIR / first_min_row["filename"]
            )

            # ── POINT 3 : enregistrer le hash après succès ───────────────────
            register_executed_plan(
                plan_hash, first_min_row["filename"], hashes_db, HASHES_FILE
            )
            print(
                f"[HASH] Plan '{first_min_row['filename']}' enregistré (hash: {plan_hash[:8]}…)."
            )
            # ────────────────────────────────────────────────────────────────

        except Exception as e:
            print(
                f"An error occurred while processing {first_min_row['filename']}: {e}"
            )
            # Déplacer le plan échoué vers plan_failed avec préfixe
            (PLAN_EXPERIENCE_DIR / first_min_row["filename"]).rename(
                PLAN_FAILED_DIR / first_min_row["filename"]
            )
            continue  # Passer au suivant

    # Fermer toutes les figures matplotlib pour éviter la fuite mémoire
    plt.close("all")

    # Nettoyage après la fin de toutes les expériences
    remove_empty_run_directories()
    remove_empty_solpoc_directories()

    return 0


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
"""
if RUNS_DIR.exists():
    for item in RUNS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("2026-"):
            try:
                item.rmdir()  # Supprime uniquement le dossier s'il est vide
                # print(f"Dossier vide supprimé : {item.name}")
            except OSError:
                pass  # Le dossier n'est pas vide

                
    # Supprimer les dossiers vides à l'intérieur de runs
    if RUNS_DIR.exists():
        for item in sorted(
            RUNS_DIR.rglob("*"), key=lambda x: len(x.parts), reverse=True
        ):
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                    print(f"Dossier vide supprimé dans runs : {item}")
                except OSError:
                    pass  # Pas vide ou impossible à supprimer
"""

# IMPORTANT
# Tout les paramètres ne sont pas dans la fonctions get_parameters comme Wl_Sol, Sol_Spec, name_Sol_Spec
# Donc a ajouter sinon crash
# Penser a sortir les paramètres en trop de la fonctions get_parameters par la suite

# Beginning of the main loop. The code must be in this loop to work in multiprocessing


# modifier les valeur first_min_row[] qui sont des numpy.float en variable direct (int)

# faire en sorte que l'utilisateur puisse modifier son spectre solaire directemment dans le fichier json
# Regarder pour PV pour fair ela meme chose

# hacher les plans d'expérience

# plan d'expérience 2 = PVcells
# plan d'expérience 3 = template_low_e
# plan d'expérience 1 = AR
# plan d'expérience 4 = spectral splitting
# plan d'expérience 5 = selective coating
# potentiellement rajouter le code de optimization et de bragg_mirror pour faire en sorte que ca fonctionne pour ces 2 cas

# rajouter les comment pour chaque plan d'expérience dans le json
