# SolPOC Optimizer

## Overview

**SolPOC Optimizer** is an experiment orchestration tool built on top of the SolPOC optical optimization library.

The original SolPOC workflow is based on running one template with one set of parameters at a time. SolPOC Optimizer extends this workflow by allowing users to prepare several experiment plans as JSON files and execute them automatically as a prioritized campaign.

The project contains two main components:

* an **Interface** used to configure and generate experiment plans;
* a **Scheduler** used to validate, prioritize, execute, archive, and track experiments.

The scheduler is a batch orchestrator: it processes all plans currently present in the pending-plan directory. It is not a time-based or continuously running scheduler.

The main objectives are:

* automating repeated SolPOC experiments;
* improving reproducibility;
* centralizing experiment parameters;
* avoiding duplicate computations;
* organizing generated results;
* supporting multiprocessing execution.

---

## Main Features

* JSON-based experiment plans;
* priority-based execution;
* automatic conversion of JSON values into SolPOC-compatible objects;
* parallel execution of repeated optimization runs;
* duplicate detection using experiment hashes;
* automatic separation of successful and failed plans;
* centralized project paths;
* timestamped result directories;
* automatic generation of SolPOC reports and plots;
* cleanup of empty temporary directories created during execution.

---

## Supported Workflows

The scheduler currently supports the main SolPOC workflows used in the project, including:

* anti-reflective coatings;
* low-emissivity coatings;
* photovoltaic cells;
* spectral splitting;
* selective coatings;
* experimental optical signal fitting.

The following workflows are available but should currently be considered **experimental**:

* Bragg mirror optimization;
* optimization with material selection.

These advanced workflows may require additional JSON parameters and can produce partial output or warnings depending on the selected SolPOC functions.

---

## Project Structure

```text
SolPOC/
├── Solpoc_optimizer/
│   ├── __init__.py
│   ├── paths.py
│   │
│   ├── experiences_scheduler/
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── plan_experience/
│   │   ├── plan_executer/
│   │   │   └── hashes.json
│   │   ├── plan_failed/
│   │   └── runs/
│   │
│   ├── Interface/
│   └── new_functions/
│
└── tests/
```

The working directories are centralized in:

```text
Solpoc_optimizer/paths.py
```

### Runtime Directories

| Directory                   | Purpose                                            |
| --------------------------- | -------------------------------------------------- |
| `plan_experience/`          | Pending JSON experiment plans                      |
| `plan_executer/`            | Successfully executed or skipped duplicate plans   |
| `plan_failed/`              | Plans that failed during preparation or execution  |
| `runs/`                     | Generated reports, plots, and optimization results |
| `plan_executer/hashes.json` | Cache of previously executed experiments           |

The runtime directories are created automatically when the scheduler starts.

---

## Experiment Workflow

1. The Interface, or the user, creates one JSON file per experiment.
2. The JSON file is placed in `plan_experience/`.
3. The scheduler loads all pending plans.
4. The plans are sorted by priority.
5. A hash is generated for each plan.
6. Previously executed duplicate plans are skipped.
7. New experiments are converted into SolPOC parameters.
8. Repeated runs are executed, using multiprocessing when applicable.
9. Reports and plots are generated.
10. Successful plans are moved to `plan_executer/`.
11. Failed plans are moved to `plan_failed/`.
12. Results are stored in `runs/`.

---

## Experiment Plans

Each JSON file represents one experiment.

A plan may contain:

* the experiment template;
* wavelength settings;
* optical stack definitions;
* material definitions;
* layer thicknesses;
* optimization bounds;
* the optimization algorithm;
* the cost function;
* the selection method;
* the number of repeated runs;
* CPU settings;
* solar spectra or target signals;
* an optional comment.

Example of a simplified plan:

```json
{
  "template": "AR",
  "Wl": [400, 800, 5],
  "Mat_Stack": ["BK7", "X", "X", "X"],
  "algo": "Differential_Evolution",
  "cost_function": "R_s",
  "selection": "selection_min",
  "nb_run": 4,
  "cpu_used": 4,
  "Comment": "Anti-reflective coating test"
}
```

The exact parameters depend on the selected SolPOC workflow.

### Priority

The current scheduler reads the priority from the last numeric part of the JSON filename.

Example:

```text
AR_2026-06-15_12h00m00s_1.json
```

In this example, the priority is `1`.

Lower values are processed first. Files without a valid numeric suffix receive a default low-priority value.

---

## Duplicate Detection

Before execution, each experiment plan is converted into a reproducible hash.

If an identical experiment has already been executed:

* the experiment is skipped;
* no duplicate optimization is launched;
* the JSON file is moved to `plan_executer/`;
* the existing hash entry is reused.

The cache is stored in:

```text
plan_executer/hashes.json
```

This mechanism avoids unnecessary calculations and helps keep experiment campaigns consistent.

---

## Installation

SolPOC Optimizer currently runs directly from the repository and depends on the existing `solpoc` package.

Create and activate a virtual environment, then install the required dependencies:

```bash
pip install solpoc numpy matplotlib
```

Install any additional dependencies required by the selected SolPOC workflow.

### Optional RCWA Support

SolPOC may display the following warning:

```text
WARNING: The RCWA solver will not be available because an S4 installation has not been found.
```

This warning does not prevent standard thin-film workflows from running. It only means that RCWA calculations requiring S4 are unavailable.

---

## How to Use

### 1. Create Experiment Plans

Create plans from the graphical Interface or prepare JSON files manually.

Place the generated files in:

```text
Solpoc_optimizer/experiences_scheduler/plan_experience/
```

### 2. Launch the Scheduler

Run the scheduler from the repository root:

```bash
python -m Solpoc_optimizer.experiences_scheduler.scheduler
```

Running the scheduler as a module is especially important on Windows because the project uses multiprocessing.

Do not launch it from an arbitrary working directory with a direct file path unless the package has already been installed.

### 3. Retrieve the Results

All generated outputs are grouped inside `runs/`.

Example:

```text
runs/
└── 2026-06-15_12h36m30s/
    ├── Low-e_12h51m02s_1/
    ├── PV Cells_12h52m10s_1/
    ├── Selective Coating_12h52m30s_2/
    ├── Spectral Splitting_12h52m45s_2/
    └── AR_12h54m20s_3/
```

Depending on the workflow, each experiment directory may contain:

* convergence data;
* optimization summaries;
* material data;
* reflectivity plots;
* transmissivity plots;
* optical stack response plots;
* consistency plots;
* optimal thickness plots;
* refractive-index plots;
* stack visualizations;
* generated text reports.

---

## Scheduler Behavior

The scheduler dynamically resolves JSON fields into SolPOC-compatible objects, including:

* wavelength arrays;
* material stacks;
* optimization algorithms;
* cost functions;
* selection functions;
* spectra and optical signals;
* numerical optimization parameters.

The main execution stages are:

1. load the plan;
2. determine its priority;
3. check its hash;
4. prepare SolPOC parameters;
5. create a result directory;
6. execute repeated optimization runs;
7. aggregate the results;
8. generate reports and plots;
9. archive the plan;
10. register the experiment hash.

If an exception occurs during a plan, the scheduler logs the error, moves the plan to `plan_failed/`, and continues with the next experiment.

---

## Interface

The Interface is used to create structured experiment plans without manually editing every JSON field.

Typical workflow:

1. launch the Interface;
2. select a SolPOC experiment template;
3. add the template;
4. review or modify the parameters;
5. choose the execution settings;
6. generate the plan;
7. verify that the JSON file has been saved in `plan_experience/`.

Advanced workflows should still be reviewed manually before launching a long campaign, especially when they require custom spectra, custom signals, or material-selection parameters.

---

## Custom Functions

Additional project-specific functions can be stored in:

```text
Solpoc_optimizer/new_functions/
```

The scheduler is designed to resolve project-specific functions from this module and fall back to the original `solpoc` module when no custom implementation is available.

The package import must be configured correctly for this fallback mechanism to work as intended.

Custom modules should be imported through the package path:

```python
from Solpoc_optimizer.new_functions import function_R_s_weighted
```

---

## Testing

Tests are stored in the `tests/` directory.

The current test suite mainly focuses on:

* experiment hashing;
* duplicate detection;
* comparison of generated results;
* selected parameter-conversion behaviors.

Run the tests with:

```bash
python -m pytest
```

End-to-end scheduler tests and complete GUI tests are still areas for future improvement.

Application code should not depend on modules stored in `tests/`. Shared runtime functions, such as hashing utilities, should be placed inside the `Solpoc_optimizer` package and imported by the tests.

---

## Current Limitations

* The scheduler processes a batch of existing JSON files; it does not continuously watch the directory.
* Bragg mirror output generation is still partially specialized.
* Material optimization requires a complete seed configuration and additional workflow-specific handling.
* Some SolPOC plotting functions may emit non-interactive backend warnings when the scheduler uses Matplotlib's `Agg` backend.
* RCWA workflows require an external S4 installation.
* The project is currently being prepared for standalone packaging.

---

## Recommended Execution Practices

* Run the scheduler from the repository root.
* Use `python -m ...` instead of executing `scheduler.py` directly.
* Keep generated results out of version control.
* Keep `.gitignore` files inside runtime directories when the empty directory structure must remain visible on GitHub.
* Validate advanced JSON plans before launching long computations.
* Keep all path definitions centralized in `Solpoc_optimizer/paths.py`.
* Do not write generated data inside an installed package directory once the project is distributed as a standalone package.

---

## Roadmap

Planned improvements include:

* standalone packaging as `solpoc-optimizer`;
* a dedicated command-line entry point;
* JSON schema validation;
* stronger end-to-end tests;
* improved error reports;
* complete Bragg mirror support;
* complete two-stage material optimization;
* user-configurable workspace directories;
* improved integration between the Interface and Scheduler.
