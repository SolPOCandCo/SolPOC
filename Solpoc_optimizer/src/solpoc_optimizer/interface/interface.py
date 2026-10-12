import tkinter as tk
from tkinter import messagebox, ttk
import json
import re
import ast
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import solpoc as sol
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from solpoc_optimizer.init_project import initialize_workspace
from solpoc_optimizer.paths import (
    MANUAL_PLANS_DIR,
    PLAN_EXPERIENCE_DIR,
    USER_NEW_FUNCTIONS_DIR,
    create_project_directories,
)


class SolpocInterface(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("SOLPOC UI")
        self.geometry("1100x600")
        self.configure(bg="grey")

        # Template actuellement sélectionné
        self.selected_template = None

        # Stockage temporaire des plans
        self.temp_plans = []

        # Dictionnaire des champs de saisie des paramètres
        self.parameter_entries = {}

        # ---------------------------------------------------------------
        # Configuration des templates : chaque clé est un nom de template,
        # la valeur est la liste des paramètres associés
        # ---------------------------------------------------------------
        self.templates_config = {
            "AR": [
                "Comment",
                "Mat_Stack",
                "algo",
                "selection",
                "cost_function",
                "Wl (start, stop, step)",
                "Th_Substrate (nm)",
                "Th_range (min, max)",
                "n_range (min, max)",
                "nb_layer",
                "Ang (°)",
                "pop_size",
                "crossover_rate",
                "f1",
                "mutation_DE",
                "budget",
                "nb_run",
                "cpu_used",
                "seed",
            ],
            "Bragg Mirror": [
                "Comment",
                "Mat_Stack",
                "algo",
                "selection",
                "cost_function",
                "Wl (start, stop, step)",
                "Th_Substrate (nm)",
                "Th_range (min, max)",
                "Ang (°)",
                "pop_size",
                "crossover_rate",
                "f1",
                "f2",
                "mutation_DE",
                "budget",
                "nb_run",
                "seed",
            ],
            "Low-e": [
                "Comment",
                "Mat_Stack",
                "algo",
                "selection",
                "cost_function",
                "Wl (start, stop, step)",
                "Th_Substrate (nm)",
                "Th_range (min, max)",
                "Ang (°)",
                "d_Stack_Opt",
                "Lambda_cut_1 (nm)",
                "pop_size",
                "crossover_rate",
                "f1",
                "f2",
                "mutation_DE",
                "budget",
                "nb_run",
                "cpu_used",
                "seed",
            ],
            "Optimization with Materials": [
                "Mat_Stack",
                "Mat_Option",
                "algo",
                "selection",
                "cost_function",
                "Th_range (min, max)",
                "Th_Substrate (nm)",
                "Wl (start, stop, step)",
                "Ang (°)",
                "pop_size",
                "crossover_rate",
                "f1",
                "mutation_DE",
                "budget",
                "Mode_choose_material",
                "seed",
            ],
            "PV Cells": [
                "Comment",
                "Mat_Stack",
                "algo",
                "selection",
                "cost_function",
                "Wl (start, stop, step)",
                "Th_Substrate (nm)",
                "Th_range (min, max)",
                "vf_range (min, max)",
                "Ang (°)",
                "pop_size",
                "crossover_rate",
                "f1",
                "mutation_DE",
                "budget",
                "nb_run",
                "cpu_used",
                "seed",
            ],
            "Selective Coating": [
                "Comment",
                "Mat_Stack",
                "algo",
                "selection",
                "cost_function",
                "Th_Substrate (nm)",
                "Th_range (min, max)",
                "vf_range (min, max)",
                "Ang (°)",
                "C",
                "T_air (K)",
                "T_abs (K)",
                "pop_size",
                "crossover_rate",
                "f1",
                "f2",
                "mutation_DE",
                "budget",
                "nb_run",
                "cpu_used",
                "seed",
            ],
            "Spectral Splitting": [
                "Comment",
                "Mat_Stack",
                "algo",
                "selection",
                "cost_function",
                "Wl (start, stop, step)",
                "Th_Substrate (nm)",
                "Th_range (min, max)",
                "vf_range (min, max)",
                "Ang (°)",
                "lambda_cut_1 (nm)",
                "lambda_cut_2 (nm)",
                "pop_size",
                "crossover_rate",
                "f1",
                "f2",
                "mutation_DE",
                "budget",
                "nb_run",
                "cpu_used",
                "seed",
            ],
            "Curve RTA": [
                "Mat_Stack",
                "d_Stack",
                "vf",
                "Wl (start, stop, step)",
                "Ang (°)",
            ],
        }

        # Correspondance entre le nom du template et son fichier Python
        self.file_map = {
            "AR": "plan_AR.py",
            "Bragg Mirror": "plan_Bragg_mirror.py",
            "Low-e": "plan_Low_e.py",
            "Optimization with Materials": "plan_Optimization_with_materials.py",
            "PV Cells": "plan_PV_cells.py",
            "Selective Coating": "plan_Selective_coating.py",
            "Spectral Splitting": "plan_Spectral_splitting.py",
            "Curve RTA": "template_curve_RTA.py",
        }

        # Correspondance entre le label affiché dans l'UI et la variable dans le fichier template
        self.param_to_var = {
            "Comment": "Comment",
            "Mat_Stack": "Mat_Stack",
            "algo": "algo",
            "selection": "selection",
            "cost_function": "cost_function",
            "Wl (start, stop, step)": "Wl",
            "Th_Substrate (nm)": "Th_Substrate",
            "Th_range (min, max)": "Th_range",
            "n_range (min, max)": "n_range",
            "nb_layer": "nb_layer",
            "Ang (°)": "Ang",
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
            "Lambda_cut_1 (nm)": "Lambda_cut_1",
            "Mat_Option": "Mat_Option",
            "Mode_choose_material": "Mode_choose_material",
            "vf_range (min, max)": "vf_range",
            "C": "C",
            "T_air (K)": "T_air",
            "T_abs (K)": "T_abs",
            "lambda_cut_1 (nm)": "lambda_cut_1",
            "lambda_cut_2 (nm)": "lambda_cut_2",
            "d_Stack": "d_Stack",
            "vf": "vf",
        }

        # Type attendu pour chaque paramètre (utilisé pour la validation des saisies)
        self.param_type = {
            "Comment": "text",
            "Mat_Stack": "list",
            "Mat_Option": "list",
            "algo": "function_ref",
            "selection": "function_ref",
            "cost_function": "function_ref",
            "Wl (start, stop, step)": "wavelength",
            "Th_Substrate (nm)": "number",
            "Th_range (min, max)": "range",
            "n_range (min, max)": "range",
            "vf_range (min, max)": "range",
            "nb_layer": "int",
            "Ang (°)": "number",
            "d_Stack_Opt": "list",
            "Lambda_cut_1 (nm)": "number",
            "lambda_cut_1 (nm)": "number",
            "lambda_cut_2 (nm)": "number",
            "C": "number",
            "T_air (K)": "number",
            "T_abs (K)": "number",
            "pop_size": "int",
            "budget": "int",
            "nb_run": "int",
            "cpu_used": "int",
            "seed": "optional_int",
            "crossover_rate": "rate",
            "f1": "number",
            "f2": "number",
            "mutation_DE": "text",
            "Mode_choose_material": "text",
            "d_Stack": "list",
            "vf": "list",
        }

        # Fonctions pour algo
        self.algo_options = [
            "DEvol",
            "optimize_ga",
            "optimize_strangle",
            "PSO",
            "simulated_annealing",
            "One_plus_One_ES",
        ]

        # Fonctions pour cost_function
        self.cost_function_options = [
            "evaluate_example",
            "evaluate_R",
            "evaluate_T",
            "evaluate_R_s",
            "evaluate_T_s",
            "evaluate_A_s",
            "evaluate_R_Brg",
            "evaluate_T_pv",
            "evaluate_A_pv",
            "evaluate_T_vis",
            "evaluate_low_e",
            "evaluate_rh",
            "evaluate_RTR",
            "evaluate_netW_PV_CSP",
            "evaluate_R_s_AOI",
            "evaluate_TRT",
            "evaluate_RTA_s",
            "evaluate_EBB",
            "evaluate_fit_R",
            "evaluate_fit_T",
            "evaluate_fit_T2face",
            "evaluate_fit_RT",
        ]

        # Fonctions pour selection
        self.selection_options = ["selection_max", "selection_min"]

        # Construction de l'interface
        self.create_header()
        self.create_content_area()
        self.show_template_view()

    # ------------------------------------------------------------------
    # CHARGEMENT DES VALEURS PAR DÉFAUT depuis le fichier template Python
    # ------------------------------------------------------------------
    def load_defaults(self, template_name):
        """Lit le fichier template associé et extrait les valeurs par défaut
        de chaque paramètre pour pré-remplir les champs de saisie."""

        filename = self.file_map.get(template_name)
        if not filename:
            return {}

        # Le template est lu depuis la copie modifiable du workspace utilisateur.
        filepath = MANUAL_PLANS_DIR / filename

        # Si le fichier manque, on tente de recopier les ressources d'origine
        # sans écraser les fichiers déjà modifiés par l'utilisateur.
        if not filepath.exists():
            initialize_workspace(overwrite=False)

        if not filepath.exists():
            messagebox.showwarning(
                "Template not found",
                (
                    f"The template file '{filename}' could not be found.\n\n"
                    f"Expected location:\n{filepath}"
                ),
            )
            return {}

        with filepath.open("r", encoding="utf-8") as f:
            content = f.read()

            # Ignore tout ce qui suit la zone modifiable
            content = content.split("# %% You should stop modifying")[0]

            # Gère les affectations sur plusieurs lignes par exemple :
            # selection = (
            #     sol.selection_max
            # )
            content = re.sub(
                r"(\w+)\s*=\s*\(\s*\n\s*(sol\.\w+)\s*\n\s*\)", r"\1 = \2", content
            )

            # Supprime les lignes commentées pour éviter de lire de fausses valeurs
            content = "\n".join(
                line
                for line in content.splitlines()
                if not line.strip().startswith("#")
            )

        # Limite la recherche à la zone bornée par les marqueurs START / END
        start_marker = "SCRIPT PARAMETERS - START"
        end_marker = "SCRIPT PARAMETERS - END"

        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx]

        defaults = {}

        # --- Affectations multiples : f1, f2 = 0.9, 0.8 ---
        lines = re.findall(r"([\w\s,]+)=\s*([^\n#]+)", content)

        for vars, values in lines:
            vars_list = [v.strip() for v in vars.split(",")]
            values_list = [v.strip() for v in values.split(",")]

            if len(vars_list) == len(values_list):
                for var_name, value in zip(vars_list, values_list):
                    for param, var in self.param_to_var.items():
                        if var == var_name:
                            defaults[param] = self.simplify_default(param, value)

        # --- Affectations simples : var = valeur ---
        for param, var in self.param_to_var.items():
            if param in defaults:
                continue

            match = re.search(rf"{var}\s*=\s*([^#\n]+)", content)

            if match:
                raw = match.group(1).strip()
                defaults[param] = self.simplify_default(param, raw)

        # Valeur par défaut de seed si absente du fichier
        if (
            "seed" in self.templates_config[self.selected_template]
            and "seed" not in defaults
        ):
            defaults["seed"] = "None"

        # cpu_used par défaut à 4
        if "cpu_used" in self.templates_config[self.selected_template]:
            defaults["cpu_used"] = "4"

        return defaults

    # ------------------------------------------------------------------
    # SIMPLIFICATION DE LA VALEUR BRUTE pour l'affichage dans les champs
    # ------------------------------------------------------------------
    def simplify_default(self, param_name, raw_value):
        """Convertit une valeur brute lue dans le fichier template
        en une chaîne simple adaptée à l'affichage dans un champ Entry."""

        param_type = self.param_type.get(param_name, "text")

        # Liste Python → chaîne séparée par des virgules : ["A", "B"] → "A, B"
        if param_type == "list":
            if "write_stack_period" in raw_value or "write_stack" in raw_value:
                materials = re.findall(r'["\']([^"\']+)["\']', raw_value)
                if materials:
                    return ", ".join(materials)
            try:
                values = ast.literal_eval(raw_value)
                if isinstance(values, list):
                    return ", ".join(str(v).strip('"').strip("'") for v in values)
            except (ValueError, SyntaxError):
                pass

        # Tuple Python → "min, max" : (0.1, 0.5) → "0.1, 0.5"
        if param_type == "range":
            try:
                values = ast.literal_eval(raw_value)
                if isinstance(values, tuple) and len(values) == 2:
                    return f"{values[0]}, {values[1]}"
            except (ValueError, SyntaxError):
                pass

        # Texte : supprime les guillemets encadrants
        if param_type == "text":
            return raw_value.strip('"').strip("'")

        # Nombre : évalue l'expression pour simplifier (ex: 1e-3 → "0.001")
        if param_type == "number":
            try:
                return str(eval(raw_value))
            except Exception:
                return raw_value

        # Longueur d'onde : extrait le contenu de np.arange( ) ou sol.Wl_selectif( )
        if param_type == "wavelength":
            value = raw_value.strip()
            if value.startswith("np.arange(") and value.endswith(")"):
                return value[len("np.arange(") : -1]
            if value.startswith("sol.Wl_selectif(") and value.endswith(")"):
                return value[len("sol.Wl_selectif(") : -1]

        # Référence des fonctions SOLPOC : sol.DEvol, sol.selection_max, etc.
        if param_type == "function_ref":
            value = raw_value.strip()
            if value.startswith("sol."):
                value = value[len("sol.") :]
            return value

        # Valeur brute pour tous les autres cas
        return raw_value

    # ------------------------------------------------------------------
    # CONSTRUCTION DU HEADER (titre + navigation)
    # ------------------------------------------------------------------
    def create_header(self):
        """Crée la barre du haut avec le titre SOLPOC UI
        et les deux labels de navigation Template / Parameters."""

        self.header_frame = tk.Frame(self, bg="black", height=150)
        self.header_frame.pack(fill="x", padx=20, pady=20)

        # Titre principal
        self.create_label(self.header_frame, "SOLPOC UI", ("Arial", 20, "bold")).pack(
            pady=(15, 10)
        )

        # Frame contenant les labels de navigation
        nav_frame = tk.Frame(self.header_frame, bg="black")
        nav_frame.pack()

        # Label "Template" — cliquable via binding <Button-1>
        self.nav_label_template = tk.Label(
            nav_frame,
            text="Template",
            font=("Arial", 12, "bold"),
            bg="black",
            fg="white",
            width=20,
            padx=10,
            pady=5,
        )
        self.nav_label_template.grid(row=0, column=0, padx=5)

        # Label "Parameters" — cliquable via binding <Button-1>
        self.nav_label_parameters = tk.Label(
            nav_frame,
            text="Parameters",
            font=("Arial", 12, "bold"),
            bg="black",
            fg="white",
            width=20,
            padx=10,
            pady=5,
        )
        self.nav_label_parameters.grid(row=0, column=1, padx=5)

    def update_nav_highlight(self, active_page):
        """Met à jour la couleur des labels de navigation selon la page active.

        Logique d'affichage :
        - Sur la page Template  → le label Template est blanc (actif),
                                   le label Parameters est grisé (pas encore accessible)
        - Sur la page Parameters → le label Parameters est blanc (actif),
                                    le label Template est grisé (on est en train de saisir)
        """
        if active_page == "template":
            # Page Template active : Parameters grisé
            self.nav_label_template.config(fg="white")
            self.nav_label_parameters.config(fg="grey")
        else:
            # Page Parameters active : Template grisé
            self.nav_label_template.config(fg="grey")
            self.nav_label_parameters.config(fg="white")

    # ------------------------------------------------------------------
    # ZONE DE CONTENU PRINCIPALE
    # ------------------------------------------------------------------
    def create_content_area(self):
        """Crée le frame principal qui accueille le contenu des pages."""
        self.content_frame = tk.Frame(self, bg="black")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def create_label(self, parent, text, font=("Arial", 12), bg="black", fg="white"):
        """Utilitaire : crée et retourne un tk.Label avec le style par défaut."""
        return tk.Label(parent, text=text, font=font, bg=bg, fg=fg)

    def clear_content(self):
        """Vide tous les widgets de la zone de contenu principale."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    # ------------------------------------------------------------------
    # PAGE TEMPLATE
    # ------------------------------------------------------------------
    def show_template_view(self):
        """Affiche la page de sélection des templates.
        - Colonne gauche : un bouton cliquable par template
        - Colonne droite : résumé des plans déjà enregistrés
        """
        self.clear_content()

        # Grise le label "Parameters" car on est sur la page Template
        self.update_nav_highlight("template")

        # --- Colonne gauche : liste des templates ---
        left_frame = tk.Frame(self.content_frame, bg="black", width=300)
        left_frame.pack(side="left", fill="y", padx=(0, 10))

        # --- Colonne droite : résumé ---
        right_frame = tk.Frame(self.content_frame, bg="black")
        right_frame.pack(side="right", fill="both", expand=True)

        self.create_label(left_frame, "Templates", ("Arial", 14, "bold")).pack(pady=10)

        # Frame qui contiendra les boutons de sélection
        buttons_frame = tk.Frame(left_frame, bg="black")
        buttons_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # Crée un bouton par template — un clic sélectionne et ouvre les paramètres
        for template_name in self.templates_config:
            btn = tk.Button(
                buttons_frame,
                text=template_name,
                width=25,
                anchor="w",
                command=lambda t=template_name: self.select_template(t),
            )
            btn.pack(pady=3, fill="both", expand=True)

        # --- Résumé des plans enregistrés ---
        self.create_label(
            right_frame, "Summary of experiances plans", ("Arial", 14, "bold")
        ).pack(pady=10)

        self.summary_text = tk.Text(right_frame, font=("Arial", 11), wrap="word")
        self.summary_text.pack(fill="both", expand=True, padx=20, pady=10)

        # Charge et affiche le contenu du dossier plans_experiences
        self.refresh_summary()

    def select_template(self, template_name):
        """Sélectionne un template et redirige vers la vue appropriée."""

        self.selected_template = template_name

        if template_name == "Curve RTA":
            self.show_rta_view()
        else:
            self.show_parameters_view()

    # ------------------------------------------------------------------
    # PAGE PARAMETERS
    # ------------------------------------------------------------------
    def show_parameters_view(self):
        """Affiche la page de saisie des paramètres pour le template sélectionné.
        Contient :
        - Un titre avec le nom du template
        - Priority
        - Les champs dynamiques propres au template
        - Un bouton Back (retour sans sauvegarder) et un bouton Confirm
        """

        # Vérifie qu'un template a bien été sélectionné
        if not self.selected_template:
            messagebox.showwarning(
                "Wait",
                "First, select a template from the Template tab.",
            )
            return

        self.clear_content()

        # Grise le label "Template" car on est sur la page Parameters
        self.update_nav_highlight("parameters")

        # Charge les valeurs par défaut depuis le fichier template
        defaults = self.load_defaults(self.selected_template)

        # Conteneur principal de la page
        container = tk.Frame(self.content_frame, bg="black")
        container.pack(fill="both", expand=True)

        # Titre avec le nom du template sélectionné
        self.create_label(
            container, f"Parameters - {self.selected_template}", ("Arial", 16, "bold")
        ).pack(pady=20)

        # Frame pour les champs de paramètres du template
        scroll_frame = tk.Frame(container, bg="black")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Réinitialise le dictionnaire des champs
        self.parameter_entries = {}

        # Récupère la liste des paramètres pour ce template
        parameters = self.templates_config[self.selected_template]

        # Création des groupes de paramètres
        groups = [
            ["Comment"],
            ["algo", "pop_size", "mutation_DE", "crossover_rate"],
            ["f1", "f2", "budget"],
            ["nb_run", "cpu_used", "seed"],
            ["Mat_Stack", "Wl (start, stop, step)", "Ang (°)", "nb_layer"],
            ["selection", "cost_function"],
            ["n_range (min, max)", "Th_range (min, max)", "Th_Substrate (nm)"],
        ]

        # Priority
        self.meta_entries = {}

        meta_frame = tk.Frame(container, bg="black")
        meta_frame.pack(fill="x", padx=20, pady=10)
        meta_fields = ["Priority"]

        for i, field in enumerate(meta_fields):
            self.create_label(meta_frame, field).grid(
                row=0, column=i * 2, padx=10, pady=5
            )

            if field == "Priority":
                # Menu déroulant pour la priorité (valeurs 1, 2, 3)
                entry = ttk.Combobox(
                    meta_frame, value=[1, 2, 3], width=18, state="readonly"
                )
                entry.current(1)  # Valeur par défaut : 2
            else:
                entry = tk.Entry(meta_frame, width=20)

            entry.grid(row=0, column=i * 2 + 1, padx=10, pady=5)
            self.meta_entries[field] = entry

        # --- Champs dynamiques propres au template ---
        current_row = 0

        for group in groups:
            visibles_param = [p for p in group if p in parameters]

            if not visibles_param:
                continue

            # Crée un bloc visuel pour chaque groupe
            groupe_frame = tk.Frame(
                scroll_frame,
                bg="#1f1f1f",
                highlightbackground="grey",
                highlightthickness=1,
                padx=6,
                pady=3,
            )

            scroll_frame.grid_columnconfigure(0, weight=1)

            groupe_frame.grid(
                row=current_row,
                column=0,
                columnspan=10,
                sticky="ew",
                padx=20,
                pady=6,
            )

            for col_index, param_name in enumerate(visibles_param):
                col = col_index * 3

                # Label du paramètre (row=0 fixe à l'intérieur du groupe_frame)
                self.create_label(groupe_frame, param_name).grid(
                    row=0, column=col, padx=10, pady=8, sticky="w"
                )

                # Type du paramètre
                param_type = self.param_type.get(param_name, "text")

                # Champ de saisie (spéciaux)
                if param_name == "algo":
                    entry = ttk.Combobox(
                        groupe_frame,
                        values=self.algo_options,
                        width=22,
                        state="readonly",
                    )

                elif param_name == "cost_function":
                    entry = ttk.Combobox(
                        groupe_frame,
                        values=self.cost_function_options,
                        width=22,
                        state="readonly",
                    )

                elif param_name == "selection":
                    entry = ttk.Combobox(
                        groupe_frame,
                        values=self.selection_options,
                        width=22,
                        state="readonly",
                    )

                elif param_type == "wavelength":
                    entry = tk.Frame(groupe_frame, bg="black")

                    entry_start = tk.Entry(entry, width=6)
                    entry_stop = tk.Entry(entry, width=6)
                    entry_step = tk.Entry(entry, width=4)

                    entry_start.pack(side="left", padx=(0, 2))
                    entry_stop.pack(side="left", padx=(0, 2))
                    entry_step.pack(side="left")

                    entry.entries = [entry_start, entry_stop, entry_step]

                elif param_type == "range":
                    entry = tk.Frame(groupe_frame, bg="black")

                    entry_min = tk.Entry(entry, width=6)
                    entry_max = tk.Entry(entry, width=6)
                    entry_min.pack(side="left", padx=(0, 2))
                    entry_max.pack(side="left")

                    entry.entries = [entry_min, entry_max]

                elif param_name == "Comment":
                    entry = tk.Entry(groupe_frame, width=150)

                else:
                    entry = tk.Entry(groupe_frame, width=14)

                entry.grid(
                    row=0,
                    column=col + 1,
                    padx=10,
                    pady=8,
                    sticky="w",
                )

                if col_index < len(visibles_param) - 1:
                    separator = tk.Label(
                        groupe_frame,
                        text="|",
                        fg="#777",
                        bg="black",
                        font=("Arial", 12, "bold"),
                    )
                    separator.grid(
                        row=0,
                        column=col + 2,
                        padx=8,
                        pady=8,
                        sticky="w",
                    )

                # Pré-remplit avec la valeur par défaut si disponible
                if param_name in defaults:
                    default_value = defaults[param_name]

                    # wavelength
                    if param_type == "wavelength":
                        parts = [p.strip() for p in default_value.split(",")]

                        if len(parts) == 3:
                            entry.entries[0].insert(0, parts[0])
                            entry.entries[1].insert(0, parts[1])
                            entry.entries[2].insert(0, parts[2])

                    # range
                    elif param_type == "range":
                        parts = [p.strip() for p in default_value.split(",")]

                        if len(parts) == 2:
                            entry.entries[0].insert(0, parts[0])
                            entry.entries[1].insert(0, parts[1])

                    elif isinstance(entry, ttk.Combobox):
                        entry.set(default_value)

                    else:
                        entry.insert(0, default_value)

                self.parameter_entries[param_name] = entry
            current_row += 1

        # --- Boutons du bas ---
        bottom_frame = tk.Frame(container, bg="black")
        bottom_frame.pack(fill="x", pady=20)

        # Bouton pour tout sauvegarder et quitter
        tk.Button(
            bottom_frame,
            text="Finalize & Save All",
            width=20,
            bg="#4CAF50",
            fg="white",
            command=self.finalize_all_plans,
        ).pack(side="right", padx=20)

        # Bouton pour ajouter le plan actuel à la file
        tk.Button(
            bottom_frame,
            text="Confirm",
            width=20,
            command=self.validate_parameters,
        ).pack(side="right", padx=10)

        # Bouton pour Back
        tk.Button(
            bottom_frame,
            text="← Back",
            width=10,
            command=self.show_template_view,
        ).pack(side="left", padx=20)

    # ------------------------------------------------------------------
    # VALIDATION ET SAUVEGARDE DES PARAMÈTRES
    # ------------------------------------------------------------------
    def validate_parameters(self):
        """Vérifie que tous les champs sont remplis et du bon type,
        puis génère et sauvegarde le fichier JSON du plan d'expérience."""

        # Vérifie chaque champ du template
        for param_name, entry in self.parameter_entries.items():
            if param_name.startswith("__"):
                continue
            if self.param_type.get(param_name) == "wavelength":
                value = (
                    f"{entry.entries[0].get().strip()}, "
                    f"{entry.entries[1].get().strip()}, "
                    f"{entry.entries[2].get().strip()}"
                )

            elif self.param_type.get(param_name) == "range":
                value = f"{entry.entries[0].get().strip()}, {entry.entries[1].get().strip()}"
            else:
                value = entry.get().strip()

            # Champ vide
            if not value:
                messagebox.showwarning("Warning", f"Please fill out : {param_name}")
                return

            # Type incorrect
            if not self.validate_type(param_name, value):
                messagebox.showwarning(
                    "Incorrect type",
                    f"The '{param_name}' field is of the wrong type",
                )
                return

        # Récupère les métadonnées
        priority = int(self.meta_entries["Priority"].get())

        # Crée un dictionnaire pour stocker les valeurs de ce plan
        values_selected = {}
        for label, widget in self.parameter_entries.items():
            if self.param_type.get(label) == "wavelength":
                values_selected[label] = (
                    f"{widget.entries[0].get().strip()}, "
                    f"{widget.entries[1].get().strip()}, "
                    f"{widget.entries[2].get().strip()}"
                )

            elif self.param_type.get(label) == "range":
                values_selected[label] = (
                    f"{widget.entries[0].get().strip()}, {widget.entries[1].get().strip()}"
                )
            else:
                values_selected[label] = widget.get()

        # On prépare le paquet complet de ce plan à mettre en attente
        current_plan = {
            "template": self.selected_template,
            "values": values_selected,
            "priority": priority,
        }

        # On l'ajoute dans la liste temporaire
        self.temp_plans.append(current_plan)

        # Message de confirmation sans quitter la page
        count = len(self.temp_plans)
        messagebox.showinfo(
            "Plan added",
            f"Plan {count} has been queued.\nYou can add another one or click 'Finalize'.",
        )

    def finalize_all_plans(self):
        """Boucle sur la liste d'attente et appelle la fonction de sauvegarde."""
        if len(self.temp_plans) == 0:
            messagebox.showwarning("Warning", "No plans have been confirmed yet.")
            return

        # Parcours chaque plan stocké dans la liste
        for plan in self.temp_plans:
            # Remet le template associé au plan courant
            self.selected_template = plan["template"]

            # Sauvegarde le plan courant
            self.build_and_save_json(plan["values"], plan["priority"])

        # Message de succès finale
        messagebox.showinfo(
            "Finalization Complete",
            f"Success, {len(self.temp_plans)} plans have been saved successfully!",
        )

        # On vide la liste d'attente pour la prochaine fois
        self.temp_plans = []

        # On retourne à l'écran d'accueil
        self.show_template_view()

    # ------------------------------------------------------------------
    # VALIDATION DU TYPE D'UN CHAMP
    # ------------------------------------------------------------------
    def validate_type(self, param_name, value):
        """Vérifie que la valeur saisie correspond au type attendu du paramètre.
        Retourne True si valide, False sinon."""

        param_type = self.param_type.get(param_name, "text")
        value = value.strip()

        # Texte non vide
        if param_type == "text":
            return value != ""

        # Entier strictement positif
        if param_type == "int":
            return value.isdigit() and int(value) > 0

        # Entier positif ou None (ex: seed)
        if param_type == "optional_int":
            return value == "None" or (value.isdigit() and int(value) > 0)

        # Nombre flottant ou entier
        if param_type == "number":
            try:
                float(value)
                return True
            except ValueError:
                return False

        # Taux entre 0 et 1
        if param_type == "rate":
            try:
                number = float(value)
                return 0 <= number <= 1
            except ValueError:
                return False

        # Intervalle (min, max)
        if param_type == "range":
            normalized = self.normalize_range(value)
            if normalized is None:
                return False
            try:
                values = ast.literal_eval(normalized)
                if not isinstance(values, tuple) or len(values) != 2:
                    return False
                min_v, max_v = values
                return (
                    isinstance(min_v, (int, float))
                    and isinstance(max_v, (int, float))
                    and min_v <= max_v
                )
            except (ValueError, SyntaxError):
                return False

        # Liste de chaînes ou de valeurs
        if param_type == "list":
            normalized = self.normalize_list(value)
            if normalized is None:
                return False
            try:
                values = ast.literal_eval(normalized)
                return isinstance(values, list)
            except (ValueError, SyntaxError):
                return False

        # Longueur d'onde : np.arange, sol.Wl_selectif, ou "start, stop, step"
        if param_type == "wavelength":
            if value.startswith("np.arange(") or value.startswith("sol.Wl_selectif("):
                return True
            cleaned = value.strip("[]() ")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) == 3:
                try:
                    [float(p) for p in parts]
                    return True
                except ValueError:
                    return False
            return False

        # Référence de fonction SOLPOC choisie dans une liste déroulante
        if param_type == "function_ref":
            return value != ""

    def value_list(self, value):
        """Vérifie qu'une valeur est une liste de nombres (int ou float)."""
        try:
            values = ast.literal_eval(value)
            if not isinstance(values, list):
                return False
            for item in values:
                if not isinstance(item, (int, float)):
                    return False
            return True
        except (ValueError, SyntaxError):
            return False

    # ------------------------------------------------------------------
    # NORMALISATION DES VALEURS SAISIES
    # ------------------------------------------------------------------
    def normalize_range(self, value):
        """Convertit "min max" ou "min, max" en tuple Python "(min, max)"."""
        value = value.strip()

        # Déjà au bon format
        if value.startswith("(") and value.endswith(")"):
            return value

        parts = [p.strip() for p in value.replace(" ", ",").split(",") if p.strip()]

        if len(parts) == 2:
            return f"({parts[0]}, {parts[1]})"

        return None

    def normalize_list(self, value):
        """Convertit "A, B, C" en liste Python '["A", "B", "C"]'."""
        value = value.strip()

        # Déjà au bon format
        if value.startswith("["):
            return value

        parts = [p.strip().strip('"').strip("'") for p in value.split(",") if p.strip()]

        if parts:
            items = ", ".join(f'"{p}"' for p in parts)
            return f"[{items}]"

        return None

    # ------------------------------------------------------------------
    # CONVERSION DE LA VALEUR BRUTE VERS LE TYPE JSON APPROPRIÉ
    # ------------------------------------------------------------------
    def parse_value(self, raw: str, json_key: str):
        """Convertit une chaîne brute issue du champ de saisie
        vers le type Python correct pour la sérialisation JSON."""

        raw = raw.strip()

        # Valeur nulle
        if raw == "" or raw.lower() in ("none", "null"):
            return None

        # Booléens
        if raw.lower() == "true":
            return True
        if raw.lower() == "false":
            return False

        # Listes numériques : Wl, Th_range, n_range, vf_range -> [start, stop, step]
        list_keys = {"Wl", "Th_range", "n_range", "vf_range"}
        if json_key in list_keys:
            cleaned = raw.strip("[]() ")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            result = []
            for p in parts:
                try:
                    result.append(int(p))
                except ValueError:
                    result.append(float(p))
            return result

        # Entiers
        if json_key in {"pop_size", "budget", "nb_run", "cpu_used", "nb_layer"}:
            try:
                return int(float(raw))
            except ValueError:
                return raw

        # Flottants
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
            try:
                return float(raw)
            except ValueError:
                return raw

        # Seed : None ou entier
        if json_key == "seed":
            if raw.lower() in ("none", "null", ""):
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        # Chaînes de texte sans guillemets
        if json_key in {
            "mutation_DE",
            "Comment",
            "Mode_choose_material",
            "algo",
            "cost_function",
            "selection",
        }:
            return raw.strip("\"'")

        # Listes de matériaux : "BK7, TiO2" -> ["BK7", "TiO2"]
        if json_key in {"Mat_Stack", "Mat_Option"}:
            if raw.startswith("["):
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [p.strip().strip("\"'") for p in raw.split(",") if p.strip()]

        # d_Stack_Opt : liste mixte "no, no, 10" → ["no", "no", 10]
        if json_key == "d_Stack_Opt":
            if raw.startswith("["):
                try:
                    return ast.literal_eval(raw)
                except Exception:
                    pass
            parts = [p.strip().strip("\"'") for p in raw.split(",") if p.strip()]
            result = []
            for p in parts:
                try:
                    result.append(float(p) if "." in p else int(p))
                except ValueError:
                    result.append(p)
            return result

        # Fallback générique : tente d'évaluer la valeur comme expression Python
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, tuple):
                return list(parsed)
            return parsed
        except Exception:
            pass

        return raw.strip("\"'")

    # ------------------------------------------------------------------
    # CONSTRUCTION ET SAUVEGARDE DU FICHIER JSON
    # ------------------------------------------------------------------
    def build_and_save_json(self, parameter_entries: dict, priority: int) -> str:
        """Construit le dictionnaire du plan d'expérience à partir des saisies,
        puis le sauvegarde dans un fichier JSON horodaté dans plans_experiences/."""

        # Schéma de base avec tous les champs initialisés à None
        experiment = {
            "template": self.selected_template,
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

        # Table de correspondance : label UI → clé JSON
        ui_label_to_json_key = {
            "Comment": "Comment",
            "Mat_Stack": "Mat_Stack",
            "Wl (start, stop, step)": "Wl",
            "Th_Substrate (nm)": "Th_Substrate",
            "Th_range (min, max)": "Th_range",
            "n_range (min, max)": "n_range",
            "nb_layer": "nb_layer",
            "Ang (°)": "Ang",
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
            "Lambda_cut_1 (nm)": "Lambda_cut_1",
            "lambda_cut_1 (nm)": "lambda_cut_1",
            "lambda_cut_2 (nm)": "lambda_cut_2",
            "Mat_Option": "Mat_Option",
            "Mode_choose_material": "Mode_choose_material",
            "vf_range (min, max)": "vf_range",
            "C": "C",
            "T_air (K)": "T_air",
            "T_abs (K)": "T_abs",
        }

        # Remplit le dictionnaire avec les valeurs saisies par l'utilisateur
        for ui_label, text_value in parameter_entries.items():
            if ui_label.startswith("__"):
                continue

            # Cherche la clé JSON correspondante
            json_key = ui_label_to_json_key.get(ui_label, ui_label)
            experiment[json_key] = self.parse_value(text_value.strip(), json_key)

        # Crée le dossier de sauvegarde défini dans paths.py si nécessaire.
        PLAN_EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)

        # Nom du fichier : Template_date_heure_priorité.json
        template_slug = self.selected_template.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss_%f")
        filename = f"{template_slug}_{timestamp}_{priority}.json"
        filepath = PLAN_EXPERIENCE_DIR / filename

        # Sauvegarde en JSON indenté (lisible).
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=4, ensure_ascii=False)

        return str(filepath)

    # ------------------------------------------------------------------
    # AFFICHAGE DU RÉSUMÉ DES PLANS ENREGISTRÉS
    # ------------------------------------------------------------------
    def refresh_summary(self):
        """Lit les plans JSON du workspace et les affiche dans le résumé."""

        self.summary_text.delete("1.0", tk.END)

        # Les plans sont maintenant centralisés dans paths.py.
        if not PLAN_EXPERIENCE_DIR.exists():
            self.summary_text.insert(
                tk.END, "No experimental designs have been saved yet."
            )
            return

        # Liste les fichiers JSON triés par nom.
        files = sorted(
            PLAN_EXPERIENCE_DIR.glob("*.json"),
            key=lambda path: path.name,
        )

        if not files:
            self.summary_text.insert(
                tk.END, "No experimental designs have been saved yet."
            )
            return

        # Clés internes à ne pas afficher dans le résumé.
        meta_keys = {
            "template",
            "Comment",
            "open_SolSpec",
            "open_Spec_Signal",
            "Sol_Spec",
            "name_Sol_Spec",
            "algo",
            "cost_function",
            "selection",
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
            "coherency_limit",
            "n_Stack",
            "k_Stack",
            "vf",
            "d_Stack",
        }

        for i, filepath in enumerate(files, start=1):
            filename = filepath.name

            with filepath.open("r", encoding="utf-8") as f:
                exp = json.load(f)

            # La priorité est le dernier segment du nom de fichier.
            priority_from_filename = filepath.stem.split("_")[-1]

            self.summary_text.insert(tk.END, "─" * 60 + "\n")
            self.summary_text.insert(
                tk.END, f"  Plan {i} : {exp.get('template', '?')}\n"
            )
            self.summary_text.insert(tk.END, f"  File: {filename}\n")
            self.summary_text.insert(tk.END, f"  Comment: {exp.get('Comment', '')}\n")
            self.summary_text.insert(
                tk.END, f"  Priority : {priority_from_filename}\n\n"
            )

            for key, value in exp.items():
                if key in meta_keys or value is None:
                    continue
                self.summary_text.insert(tk.END, f"    • {key} : {value}\n")

            self.summary_text.insert(tk.END, "\n")

        self.summary_text.insert(tk.END, "─" * 60 + "\n")

    # ------------------------------------------------------------------
    # PAGE CURVE RTA (vue améliorée avec couches dynamiques)
    # ------------------------------------------------------------------
    def show_rta_view(self):
        """Affiche la vue Curve RTA avec :
        - Un panneau gauche pour la saisie des couches et paramètres
        - Un panneau droit pour la visualisation du stack et du graphe RTA
        """

        self.clear_content()

        self.update_nav_highlight("parameters")

        container = tk.Frame(self.content_frame, bg="black")
        container.pack(fill="both", expand=True)

        self.create_label(container, "Curve RTA", ("Arial", 18, "bold")).pack(pady=10)

        # ========================================================
        # PANNEAU GAUCHE : SAISIE
        # ========================================================

        input_frame = tk.Frame(container, bg="black")
        input_frame.pack(side="left", fill="y", padx=20, pady=20)

        # --------------------------------------------------------
        # NOMBRE DE COUCHES (spinbox)
        # --------------------------------------------------------

        tk.Label(
            input_frame,
            text="Number of layers",
            bg="black",
            fg="white",
            font=("Arial", 11),
        ).grid(row=0, column=0, sticky="w", pady=8)

        self.nb_layers_var = tk.IntVar(value=4)

        nb_layers_spin = tk.Spinbox(
            input_frame,
            from_=1,
            to=20,
            textvariable=self.nb_layers_var,
            width=8,
            command=self.generate_layer_fields,
        )

        nb_layers_spin.grid(row=0, column=1, sticky="w", pady=8)

        # ========================================================
        # FRAME COUCHES DYNAMIQUES
        # ========================================================

        self.layers_frame = tk.Frame(input_frame, bg="black")
        self.layers_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Stockage des champs de saisie par couche
        self.material_entries = []
        self.thickness_entries = []

        # ========================================================
        # AUTRES PARAMÈTRES (Wl, Ang)
        # ========================================================

        other_frame = tk.Frame(input_frame, bg="black")
        other_frame.grid(row=2, column=0, columnspan=2, pady=20)

        self.rta_entries = {}

        fields = [
            ("Wl (start, stop, step)", "280, 2500, 5"),
            ("Ang", "0"),
        ]

        for i, (label, default) in enumerate(fields):
            tk.Label(
                other_frame,
                text=label,
                bg="black",
                fg="white",
                font=("Arial", 11),
            ).grid(row=i, column=0, sticky="w", pady=8)

            entry = tk.Entry(other_frame, width=25)

            entry.insert(0, default)

            entry.grid(row=i, column=1, padx=10, pady=8)

            self.rta_entries[label] = entry

        # ========================================================
        # BOUTONS
        # ========================================================

        # Limite de cohérence (utilisée pour distinguer couches cohérentes/incohérentes)
        self.coherency_limit_var = tk.IntVar(value=2000)

        tk.Button(
            input_frame,
            text="Advanced Options",
            width=20,
            command=self.show_advanced_options,
        ).grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(
            input_frame,
            text="Plot RTA",
            width=20,
            bg="#4CAF50",
            fg="white",
            command=self.compute_rta_curve,
        ).grid(row=4, column=0, columnspan=2, pady=20)

        tk.Button(
            input_frame,
            text="← Back",
            width=15,
            command=self.show_template_view,
        ).grid(row=5, column=0, sticky="w", pady=20)

        # ========================================================
        # PANNEAU DROIT : VISUALISATION
        # ========================================================

        right_frame = tk.Frame(container, bg="black")
        right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Zone du schéma du stack (aperçu en temps réel)
        self.stack_frame = tk.Frame(right_frame, bg="white", height=180)
        self.stack_frame.pack(fill="x", pady=(0, 10))

        # Zone du graphe RTA (calculé au clic sur Plot RTA)
        self.graph_frame = tk.Frame(right_frame, bg="white")
        self.graph_frame.pack(fill="both", expand=True)

        # Génération initiale des champs de couches avec les valeurs par défaut
        self.generate_layer_fields()

    # ------------------------------------------------------------------
    # CALCUL ET AFFICHAGE DE LA COURBE RTA
    # ------------------------------------------------------------------
    def compute_rta_curve(self):
        """Récupère les paramètres saisis, calcule R, T, A via SOLPOC
        et affiche le graphe ainsi que le schéma du stack."""

        try:
            # ====================================================
            # RÉCUPÉRATION DES INPUTS
            # ====================================================

            mat_stack = [entry.get().strip() for entry in self.material_entries]

            d_stack = [float(entry.get().strip()) for entry in self.thickness_entries]

            wl_values = [
                float(x.strip())
                for x in self.rta_entries["Wl (start, stop, step)"].get().split(",")
            ]

            wl_start, wl_stop, wl_step = wl_values

            ang = float(self.rta_entries["Ang"].get())

            # ====================================================
            # VECTEUR DE LONGUEURS D'ONDE
            # ====================================================

            Wl = np.arange(wl_start, wl_stop, wl_step)

            # ====================================================
            # INDICES OPTIQUES DES MATÉRIAUX
            # ====================================================

            n_Stack, k_Stack = sol.Made_Stack(mat_stack, Wl)

            # ====================================================
            # SPECTRE SOLAIRE (interpolé sur Wl)
            # ====================================================

            Wl_Sol, Sol_Spec, name_Sol_Spec = sol.open_SolSpec(
                "Materials/SolSpec.txt", "GT"
            )

            Sol_Spec = np.interp(Wl, Wl_Sol, Sol_Spec)

            # ====================================================
            # STRUCTURE DE PARAMÈTRES SOLPOC
            # ====================================================

            parameters = sol.get_parameters(
                Wl=Wl,
                Ang=ang,
                d_Stack=d_stack,
                vf=None,
                Th_Substrate=d_stack[0],
                Mat_Stack=mat_stack,
                Sol_Spec=Sol_Spec,
                n_Stack=n_Stack,
                k_Stack=k_Stack,
                coherency_limit=self.coherency_limit_var.get(),
            )

            # ====================================================
            # CALCUL RTA
            # ====================================================

            R, T, A = sol.RTA_curve_inco(d_stack, parameters)

            # ====================================================
            # NETTOYAGE DES ANCIENS GRAPHES
            # ====================================================

            for widget in self.graph_frame.winfo_children():
                widget.destroy()

            for widget in self.stack_frame.winfo_children():
                widget.destroy()

            # ====================================================
            # SCHÉMA DU STACK (après calcul, avec cohérence limitée)
            # ====================================================

            stack_fig = Figure(figsize=(7, 2), dpi=100)

            stack_ax = stack_fig.add_subplot(111)

            colors = plt.cm.Set3(np.linspace(0, 1, len(mat_stack)))

            layer_width = 1

            coherency_limit = self.coherency_limit_var.get()

            for i, (mat, thickness) in enumerate(zip(mat_stack, d_stack)):
                # Couche incohérente si son épaisseur dépasse la limite de cohérence
                is_incoherent = thickness > coherency_limit

                rect = Rectangle(
                    (i * layer_width, 0),
                    layer_width,
                    1,
                    facecolor=colors[i],
                    edgecolor="black",
                    linewidth=1.5,
                    hatch="////" if is_incoherent else "",
                )

                stack_ax.add_patch(rect)

                # Nom du matériau
                stack_ax.text(
                    i * layer_width + 0.5,
                    0.65,
                    mat,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

                # Épaisseur en nm
                stack_ax.text(
                    i * layer_width + 0.5,
                    0.30,
                    f"{thickness:.0f} nm",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            # Limites et nettoyage des axes
            stack_ax.set_xlim(0, len(mat_stack))
            stack_ax.set_ylim(0, 1)
            stack_ax.set_xticks([])
            stack_ax.set_yticks([])

            for spine in stack_ax.spines.values():
                spine.set_visible(False)

            stack_ax.set_title(
                f"Materials Stack  (incoherent if > {coherency_limit} nm)",
                fontsize=14,
                fontweight="bold",
            )

            # Canvas tkinter pour le stack
            stack_canvas = FigureCanvasTkAgg(stack_fig, master=self.stack_frame)
            stack_canvas.draw()
            stack_canvas.get_tk_widget().pack(fill="both", expand=True)

            # ====================================================
            # GRAPHE RTA
            # ====================================================

            fig = Figure(figsize=(7, 5), dpi=100)
            ax = fig.add_subplot(111)

            ax.plot(Wl, R, label="Reflectance")
            ax.plot(Wl, T, label="Transmittance")
            ax.plot(Wl, A, label="Absorptance")

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Value")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True)

            # Canvas tkinter pour le graphe RTA
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("RTA Error", str(e))

    # ------------------------------------------------------------------
    # FENÊTRE OPTIONS AVANCÉES
    # ------------------------------------------------------------------
    def show_advanced_options(self):
        """Ouvre une fenêtre modale pour configurer la limite de cohérence."""

        window = tk.Toplevel(self)
        window.title("Advanced Options")
        window.geometry("300x120")
        window.configure(bg="black")

        tk.Label(
            window,
            text="Coherency Limit",
            bg="black",
            fg="white",
            font=("Arial", 11),
        ).pack(pady=(20, 5))

        entry = tk.Entry(
            window,
            textvariable=self.coherency_limit_var,
            width=15,
        )
        entry.pack(pady=5)

        tk.Button(
            window,
            text="Save",
            command=lambda: [self.update_stack_plot(), window.destroy()],
        ).pack(pady=15)

    # ------------------------------------------------------------------
    # GÉNÉRATION DES CHAMPS DE COUCHES DYNAMIQUES
    # ------------------------------------------------------------------
    def generate_layer_fields(self):
        """Crée dynamiquement les champs Matériau / Épaisseur
        selon le nombre de couches sélectionné dans le Spinbox."""

        # Vide les anciens champs
        for widget in self.layers_frame.winfo_children():
            widget.destroy()

        self.material_entries = []
        self.thickness_entries = []

        nb_layers = self.nb_layers_var.get()

        # ========================================================
        # EN-TÊTES DES COLONNES
        # ========================================================

        tk.Label(
            self.layers_frame,
            text="Material",
            bg="black",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=0, padx=10)

        tk.Label(
            self.layers_frame,
            text="Thickness (nm)",
            bg="black",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=1, padx=10)

        # ========================================================
        # CHAMPS DYNAMIQUES (valeurs par défaut pour les 4 premières couches)
        # ========================================================

        default_materials = ["BK7", "Al2O3", "Al", "air"]
        default_thickness = [1000000, 50, 200, 50]

        for i in range(nb_layers):
            # Champ Matériau
            mat_entry = tk.Entry(self.layers_frame, width=18)

            if i < len(default_materials):
                mat_entry.insert(0, default_materials[i])

            mat_entry.grid(row=i + 1, column=0, padx=10, pady=4)

            # Champ Épaisseur
            thick_entry = tk.Entry(self.layers_frame, width=12)

            if i < len(default_thickness):
                thick_entry.insert(0, str(default_thickness[i]))

            thick_entry.grid(row=i + 1, column=1, padx=10, pady=4)

            # Mise à jour automatique du schéma à chaque frappe
            mat_entry.bind("<KeyRelease>", lambda e: self.update_stack_plot())
            thick_entry.bind("<KeyRelease>", lambda e: self.update_stack_plot())

            self.material_entries.append(mat_entry)
            self.thickness_entries.append(thick_entry)

        # Affiche le schéma initial du stack
        self.update_stack_plot()

    # ------------------------------------------------------------------
    # MISE À JOUR EN TEMPS RÉEL DU SCHÉMA DU STACK
    # ------------------------------------------------------------------
    def update_stack_plot(self):
        """Redessine le schéma du stack en temps réel selon les valeurs saisies."""

        for widget in self.stack_frame.winfo_children():
            widget.destroy()

        try:
            mat_stack = [entry.get().strip() for entry in self.material_entries]

            d_stack = [
                float(entry.get().strip()) if entry.get().strip() != "" else 0
                for entry in self.thickness_entries
            ]

            stack_fig = Figure(figsize=(7, 2), dpi=100)
            stack_ax = stack_fig.add_subplot(111)

            colors = plt.cm.Set3(np.linspace(0, 1, len(mat_stack)))

            layer_width = 1
            coherency_limit = self.coherency_limit_var.get()

            for i, (mat, thickness) in enumerate(zip(mat_stack, d_stack)):
                is_incoherent = thickness > coherency_limit

                rect = Rectangle(
                    (i * layer_width, 0),
                    layer_width,
                    1,
                    facecolor=colors[i],
                    edgecolor="black",
                    linewidth=1.5,
                    hatch="////" if is_incoherent else "",
                )
                stack_ax.add_patch(rect)

                # Nom du matériau
                stack_ax.text(
                    i * layer_width + 0.5,
                    0.65,
                    mat,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

                # Épaisseur en nm
                stack_ax.text(
                    i * layer_width + 0.5,
                    0.30,
                    f"{thickness:.0f} nm",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            stack_ax.set_xlim(0, len(mat_stack))
            stack_ax.set_ylim(0, 1)
            stack_ax.set_xticks([])
            stack_ax.set_yticks([])

            for spine in stack_ax.spines.values():
                spine.set_visible(False)

            stack_ax.set_title(
                "Materials Stack",
                fontsize=14,
                fontweight="bold",
            )

            stack_canvas = FigureCanvasTkAgg(stack_fig, master=self.stack_frame)
            stack_canvas.draw()
            stack_canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception:
            # Silencieux : les erreurs de saisie en cours ne bloquent pas l'UI
            pass


def main() -> int:
    """Initialise le workspace puis lance l'interface graphique."""

    # Crée la structure du workspace si elle n'existe pas encore.
    create_project_directories()

    # On vérifie qu'il existe au moins un vrai template et une vraie
    # fonction utilisateur, sans compter les éventuels __init__.py.
    manual_templates_available = any(
        path.is_file() and path.name != "__init__.py"
        for path in MANUAL_PLANS_DIR.iterdir()
    )

    custom_functions_available = any(
        path.is_file() and path.name != "__init__.py"
        for path in USER_NEW_FUNCTIONS_DIR.iterdir()
    )

    # Copie seulement les fichiers manquants. Les modifications utilisateur
    # déjà présentes dans le workspace ne sont pas écrasées.
    if not manual_templates_available or not custom_functions_available:
        initialize_workspace(overwrite=False)

    app = SolpocInterface()
    app.mainloop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
