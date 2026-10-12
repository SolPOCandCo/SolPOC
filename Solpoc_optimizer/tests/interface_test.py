import json
import os
import pytest
from Solpoc_optimizer.Interface.interface import SolpocInterface


TEMPLATES = [
    "AR",
    "Bragg Mirror",
    "Low-e",
    "Optimization with Materials",
    "PV Cells",
    "Selective Coating",
    "Spectral Splitting",
    "Curve RTA",
]


TEST_VALUES = {
    "Comment": "Test",
    "Mat_Stack": "BK7, TiO2",
    "Mat_Option": "SiO2, TiO2",
    "algo": "DEvol",
    "selection": "selection_max",
    "cost_function": "evaluate_R",
    "Wl (start, stop, step)": "280, 2505, 5",
    "Th_Substrate (nm)": "1000000",
    "Th_range (min, max)": "0, 200",
    "n_range (min, max)": "1.4, 2.4",
    "vf_range (min, max)": "0.1, 0.9",
    "nb_layer": "3",
    "Ang (°)": "0",
    "pop_size": "30",
    "crossover_rate": "0.5",
    "f1": "0.9",
    "f2": "0.8",
    "mutation_DE": "current_to_best",
    "budget": "1000",
    "nb_run": "8",
    "cpu_used": "4",
    "seed": "None",
    "d_Stack_Opt": "no, no, 10",
    "Lambda_cut_1 (nm)": "800",
    "lambda_cut_1 (nm)": "800",
    "lambda_cut_2 (nm)": "1200",
    "Mode_choose_material": "sigmoid",
    "C": "100",
    "T_air (K)": "300",
    "T_abs (K)": "350",
    "d_Stack": "[1000000, 50, 200, 50]",
    "vf": "[1, 0.5, 0.5, 1]",
}


UI_LABEL_TO_JSON_KEY = {
    "Wl (start, stop, step)": "Wl",
    "Th_Substrate (nm)": "Th_Substrate",
    "Th_range (min, max)": "Th_range",
    "n_range (min, max)": "n_range",
    "vf_range (min, max)": "vf_range",
    "Ang (°)": "Ang",
    "Lambda_cut_1 (nm)": "Lambda_cut_1",
    "T_air (K)": "T_air",
    "T_abs (K)": "T_abs",
}


def make_interface(template):
    interface = SolpocInterface.__new__(SolpocInterface)
    interface.selected_template = template
    interface.templates_config = {
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
    return interface


class TestJsonValidation:
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_json_generation(self, template, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        interface = make_interface(template)

        parameter_entries = {
            param: TEST_VALUES[param] for param in interface.templates_config[template]
        }

        filepath = interface.build_and_save_json(parameter_entries, 1)

        assert os.path.exists(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        assert data["template"] == template

        for param in interface.templates_config[template]:
            json_key = UI_LABEL_TO_JSON_KEY.get(param, param)
            assert json_key in data

        int_keys = ["pop_size", "budget", "nb_run", "cpu_used", "nb_layer"]
        float_keys = [
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
        ]
        list_keys = [
            "Wl",
            "Th_range",
            "n_range",
            "vf_range",
            "Mat_Stack",
            "Mat_Option",
            "d_Stack_Opt",
            "d_Stack",
            "vf",
        ]
        string_keys = [
            "Comment",
            "mutation_DE",
            "Mode_choose_material",
            "algo",
            "cost_function",
            "selection",
        ]

        for key in int_keys:
            if key in data and data[key] is not None:
                assert isinstance(data[key], int)

        for key in float_keys:
            if key in data and data[key] is not None:
                assert isinstance(data[key], float)

        for key in list_keys:
            if key in data and data[key] is not None:
                assert isinstance(data[key], list)

        for key in string_keys:
            if key in data and data[key] is not None:
                assert isinstance(data[key], str)

        if "seed" in data:
            assert data["seed"] is None or isinstance(data["seed"], int)
