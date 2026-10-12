# SolPOC Optimizer

SolPOC Optimizer est une extension du projet scientifique **SolPOC** permettant de simplifier la création, la planification et l’exécution d’expériences d’optimisation de couches minces.

Le projet ajoute une couche logicielle au-dessus de SolPOC sans modifier son fonctionnement principal. Il fournit :

* une interface graphique ;
* des plans d’expérience manuels ;
* une génération standardisée de fichiers JSON ;
* un planificateur d’expériences, ou scheduler ;
* une gestion automatique des priorités ;
* une détection des expériences déjà exécutées ;
* une organisation automatique des plans et des résultats ;
* un package Python installable localement ;
* trois commandes utilisables directement depuis un terminal.

---

## Sommaire

1. [Fonctionnalités principales](#fonctionnalités-principales)
2. [Architecture générale](#architecture-générale)
3. [Organisation du package](#organisation-du-package)
4. [Organisation du workspace](#organisation-du-workspace)
5. [Prérequis](#prérequis)
6. [Installation locale](#installation-locale)
7. [Initialisation du workspace](#initialisation-du-workspace)
8. [Utilisation de l’interface graphique](#utilisation-de-linterface-graphique)
9. [Utilisation des plans manuels](#utilisation-des-plans-manuels)
10. [Format des plans JSON](#format-des-plans-json)
11. [Lancement du scheduler](#lancement-du-scheduler)
12. [Fonctionnement du scheduler](#fonctionnement-du-scheduler)
13. [Gestion des priorités](#gestion-des-priorités)
14. [Détection des doublons](#détection-des-doublons)
15. [Gestion des erreurs](#gestion-des-erreurs)
16. [Commandes disponibles](#commandes-disponibles)
17. [Configuration du workspace](#configuration-du-workspace)
18. [Développement local](#développement-local)
19. [Résolution des problèmes fréquents](#résolution-des-problèmes-fréquents)
20. [Contributeurs](#contributeurs)

---

# Fonctionnalités principales

## Interface graphique

L’interface graphique permet à l’utilisateur de :

* sélectionner un type d’expérience ;
* remplir les paramètres dans un formulaire ;
* vérifier les types de données saisis ;
* visualiser un résumé de l’expérience ;
* générer automatiquement un plan JSON compatible avec le scheduler.

L’interface repose sur **Tkinter**, la bibliothèque graphique standard de Python.

---

## Plans d’expérience manuels

Les utilisateurs qui préfèrent travailler directement avec Python peuvent utiliser les modèles présents dans :

```text
Workspace/manual_interface/
```

Chaque plan manuel contient les paramètres scientifiques de l’expérience, puis appelle :

```python
generate_json(
    locals(),
    template_name="Nom du template",
    priority=priority,
)
```

La fonction `generate_json()` transforme les paramètres du script en fichier JSON et l’enregistre automatiquement dans :

```text
Workspace/plan_experience/
```

---

## Scheduler

Le scheduler permet de :

* lire tous les plans JSON disponibles ;
* les trier par priorité ;
* vérifier si une expérience identique a déjà été exécutée ;
* reconstruire les paramètres nécessaires à SolPOC ;
* sélectionner dynamiquement les fonctions d’optimisation ;
* exécuter les calculs ;
* utiliser plusieurs cœurs du processeur ;
* sauvegarder les graphiques et les rapports ;
* déplacer les plans réussis ou échoués ;
* continuer avec les plans suivants lorsqu’une expérience échoue.

---

## Packaging

SolPOC Optimizer est organisé comme un package Python installable.

Après installation, trois commandes sont disponibles :

```powershell
solpoc-optimizer-init
solpoc-interface
solpoc-optimize
```

L’utilisateur n’a donc pas besoin de lancer directement les fichiers internes du projet.

---

# Architecture générale

Le workflow général de SolPOC Optimizer est le suivant :

```text
Interface graphique ─────┐
                         │
Plans manuels Python ─────┼──> Plan JSON
                         │
                         └──> plan_experience
                                  │
                                  ▼
                              Scheduler
                                  │
                   ┌──────────────┼──────────────┐
                   │              │              │
                   ▼              ▼              ▼
               Priorité         Hash         Paramètres
                   │              │              │
                   └──────────────┼──────────────┘
                                  │
                                  ▼
                                SolPOC
                                  │
                                  ▼
                              Résultats
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              plan_executer                plan_failed
```

SolPOC reste le moteur scientifique principal. SolPOC Optimizer ajoute une couche permettant de préparer et d’automatiser les expériences.

---

# Organisation du package

L’architecture principale du projet est la suivante :

```text
Solpoc_optimizer/
├── pyproject.toml
├── README.md
└── src/
    └── solpoc_optimizer/
        ├── __init__.py
        ├── __main__.py
        ├── cli.py
        ├── init_project.py
        ├── paths.py
        ├── hashing.py
        │
        ├── interface/
        │   ├── __init__.py
        │   ├── interface.py
        │   └── manual_interface/
        │       ├── plan_utils.py
        │       └── templates Python
        │
        ├── experiences_scheduler/
        │   ├── __init__.py
        │   └── scheduler.py
        │
        └── new_functions/
            ├── __init__.py
            └── fonctions supplémentaires
```

## `pyproject.toml`

Le fichier `pyproject.toml` contient la configuration du package :

* nom du projet ;
* version ;
* dépendances ;
* configuration du dossier `src` ;
* commandes disponibles dans le terminal ;
* fichiers inclus dans le package.

Il associe notamment les commandes :

```text
solpoc-optimizer-init
solpoc-interface
solpoc-optimize
```

aux fonctions Python correspondantes.

---

## `__init__.py`

Les fichiers `__init__.py` indiquent à Python que les dossiers sont des packages importables.

Ils permettent notamment d’utiliser :

```python
import solpoc_optimizer
```

Le fichier principal peut également exposer la version du package :

```python
__version__ = "0.1.0"
```

---

## `__main__.py`

Le fichier `__main__.py` permet de lancer le package comme un module Python :

```powershell
python -m solpoc_optimizer
```

Il délègue ensuite l’exécution aux fonctions définies dans `cli.py`.

---

## `cli.py`

Le fichier `cli.py` centralise les points d’entrée en ligne de commande.

Il permet de lancer :

* l’initialisation du workspace ;
* l’interface graphique ;
* le scheduler.

Il sert d’intermédiaire entre les commandes déclarées dans `pyproject.toml` et les modules internes du package.

---

## `init_project.py`

Le fichier `init_project.py` initialise le workspace.

Il est responsable de :

* la création des répertoires ;
* la copie des plans manuels ;
* la copie des fonctions personnalisables ;
* la conservation des fichiers déjà présents ;
* la restauration forcée des modèles avec l’option `--force`.

---

## `paths.py`

Le fichier `paths.py` centralise tous les chemins utilisés par le projet.

Il définit notamment :

```python
WORKSPACE_DIR
PLAN_EXPERIENCE_DIR
PLAN_EXECUTED_DIR
PLAN_FAILED_DIR
RUNS_DIR
HASHES_FILE
```

Par défaut, le workspace est créé dans le dossier courant :

```python
Path.cwd() / "Workspace"
```

Il est également possible de définir un emplacement personnalisé avec la variable d’environnement :

```text
SOLPOC_OPTIMIZER_HOME
```

---

## `hashing.py`

Le fichier `hashing.py` permet d’identifier les expériences déjà exécutées.

Le hash d’un plan est calculé à partir de ses paramètres scientifiques.

Certains champs ne sont pas pris en compte dans le calcul du hash :

```text
filename
priority
cpu_used
```

Ainsi, deux plans ayant les mêmes paramètres scientifiques restent considérés comme identiques, même si leur nom, leur priorité ou le nombre de processeurs utilisés diffèrent.

---

## `scheduler.py`

Le fichier `scheduler.py` contient le planificateur d’expériences.

Il est responsable de :

* la lecture des plans JSON ;
* leur tri par priorité ;
* la vérification des hashes ;
* la reconstruction des longueurs d’onde ;
* la création des données optiques ;
* la sélection des fonctions SolPOC ;
* l’exécution des optimisations ;
* la génération des résultats ;
* le déplacement des plans ;
* la gestion des erreurs.

---

# Organisation du workspace

Le workspace est séparé du code installé.

Après initialisation, il contient :

```text
Workspace/
├── manual_interface/
├── new_functions/
├── plan_experience/
├── plan_executer/
├── plan_failed/
├── runs/
└── hashes.json
```

## `manual_interface`

Contient les plans d’expérience Python modifiables par l’utilisateur.

---

## `new_functions`

Contient les fonctions supplémentaires ou personnalisées utilisées par le scheduler.

Ces fichiers peuvent être modifiés sans toucher au package installé.

---

## `plan_experience`

Contient les plans JSON en attente d’exécution.

Le scheduler lit les fichiers présents dans ce dossier.

---

## `plan_executer`

Contient les plans ayant été exécutés avec succès ou ignorés car une expérience identique avait déjà été réalisée.

Le nom du dossier est conservé tel qu’il est utilisé dans le code du projet.

---

## `plan_failed`

Contient les plans ayant provoqué une erreur.

Une erreur sur un plan ne bloque pas l’exécution des autres plans.

---

## `runs`

Contient les résultats produits par SolPOC :

* rapports texte ;
* paramètres utilisés ;
* solutions optimales ;
* courbes de convergence ;
* réflectivité ;
* transmissivité ;
* absorption ;
* représentation de la pile optique ;
* données liées aux matériaux.

---

## `hashes.json`

Contient les hashes des expériences exécutées avec succès.

Ce fichier permet au scheduler d’éviter de relancer une expérience identique.

---

# Prérequis

Avant d’installer SolPOC Optimizer, il faut disposer de :

* Python ;
* `pip` ;
* un environnement virtuel Python ;
* une installation locale fonctionnelle de SolPOC ;
* Git pour récupérer et versionner le projet.

Le projet a été développé avec SolPOC version :

```text
0.9.7
```

Certaines fonctionnalités RCWA nécessitent une installation séparée de **S4**.

Sans S4, SolPOC peut afficher l’avertissement suivant :

```text
WARNING: The RCWA solver will not be available because an S4 installation has not been found.
```

Cet avertissement n’empêche pas l’utilisation des fonctionnalités ne dépendant pas du solveur RCWA.

---

# Installation locale

## 1. Créer ou activer un environnement virtuel

Sous PowerShell :

```powershell
python -m venv .venv
```

Puis :

```powershell
.\.venv\Scripts\Activate.ps1
```

Le nom de l’environnement doit apparaître au début de la ligne du terminal.

Exemple :

```text
(.venv) PS C:\MonProjet>
```

---

## 2. Installer SolPOC localement

Se placer à la racine du projet SolPOC :

```powershell
cd "chemin\vers\SolPOC"
```

Puis lancer :

```powershell
python -m pip install -e .
```

L’option `-e` installe le projet en mode éditable.

Les modifications apportées au code source local sont ainsi directement prises en compte.

---

## 3. Installer SolPOC Optimizer

Se placer dans le dossier :

```text
SolPOC/Solpoc_optimizer
```

Puis lancer :

```powershell
python -m pip install --no-deps -e .
```

L’option :

```text
-e
```

installe SolPOC Optimizer en mode éditable.

L’option :

```text
--no-deps
```

évite que `pip` tente de remplacer ou de télécharger une autre version de SolPOC alors qu’il est déjà installé localement.

Dans un environnement où toutes les dépendances sont correctement déclarées et accessibles, l’installation peut aussi être effectuée avec :

```powershell
python -m pip install -e .
```

---

## 4. Vérifier l’installation

Vérifier que le package peut être importé :

```powershell
python -c "import solpoc_optimizer; print(solpoc_optimizer.__file__)"
```

Vérifier les commandes disponibles :

```powershell
Get-Command solpoc-optimizer-init
Get-Command solpoc-interface
Get-Command solpoc-optimize
```

Les commandes doivent provenir du dossier `Scripts` de l’environnement virtuel actif.

---

# Initialisation du workspace

Se placer dans le dossier dans lequel le workspace doit être créé.

Par exemple :

```powershell
cd "chemin\vers\Solpoc_optimizer"
```

Puis lancer :

```powershell
solpoc-optimizer-init
```

Le dossier suivant est créé dans le répertoire courant :

```text
Workspace/
```

Exemple :

```text
C:\Projet\Solpoc_optimizer\Workspace
```

La commande crée également les sous-dossiers nécessaires et copie les modèles de plans manuels et les fonctions personnalisées.

---

## Restaurer les modèles

Pour recopier les modèles d’origine :

```powershell
solpoc-optimizer-init --force
```

Attention : cette commande peut remplacer les fichiers personnalisés présents dans :

```text
Workspace/manual_interface/
Workspace/new_functions/
```

Il est recommandé de sauvegarder les modifications avant d’utiliser l’option `--force`.

---

# Utilisation de l’interface graphique

Pour lancer l’interface :

```powershell
solpoc-interface
```

L’utilisateur peut ensuite :

1. sélectionner un template ;
2. remplir les paramètres ;
3. vérifier le résumé de l’expérience ;
4. générer le plan JSON.

Le JSON est automatiquement sauvegardé dans :

```text
Workspace/plan_experience/
```

---

# Utilisation des plans manuels

Les plans manuels sont situés dans :

```text
Workspace/manual_interface/
```

Se placer dans ce dossier :

```powershell
cd Workspace\manual_interface
```

Puis lancer le plan souhaité :

```powershell
python nom_du_plan.py
```

Exemple de structure d’un plan manuel :

```python
import numpy as np
import solpoc as sol

from plan_utils import generate_json


priority = 1

Comment = "Exemple de miroir de Bragg"

Mat_Stack = [
    "BK7",
    "SiO2",
    "TiO2",
    "SiO2",
    "TiO2",
]

Wl = np.arange(400, 800, 5)

Th_Substrate = 1e6
Th_range = (0, 200)
Ang = 0

algo = sol.DEvol
selection = sol.selection_max
cost_function = sol.evaluate_R_Brg

pop_size = 30
crossover_rate = 0.5
f1 = 0.9
f2 = 0.8
mutation_DE = "current_to_best"

budget = 2000
nb_run = 8
seed = 2905804230


if __name__ == "__main__":
    generate_json(
        locals(),
        template_name="Bragg Mirror",
        priority=priority,
    )
```

Le fichier JSON est automatiquement généré dans :

```text
Workspace/plan_experience/
```

---

# Format des plans JSON

Les plans utilisent un format JSON structuré.

Exemple simplifié :

```json
{
    "template": "Bragg Mirror",
    "Comment": "Exemple de miroir de Bragg",
    "Wl": [
        400,
        800,
        5
    ],
    "Mat_Stack": [
        "BK7",
        "SiO2",
        "TiO2",
        "SiO2",
        "TiO2"
    ],
    "Th_Substrate": 1000000.0,
    "Th_range": [
        0,
        200
    ],
    "Ang": 0.0,
    "pop_size": 30,
    "budget": 2000,
    "nb_run": 8,
    "algo": "DEvol",
    "cost_function": "evaluate_R_Brg",
    "selection": "selection_max",
    "seed": 2905804230
}
```

---

## Format des longueurs d’onde

Les longueurs d’onde sont enregistrées sous la forme :

```json
"Wl": [
    400,
    800,
    5
]
```

Les trois valeurs correspondent à :

```text
[début, fin, pas]
```

Le scheduler reconstruit ensuite le tableau NumPy :

```python
np.arange(400, 800, 5)
```

La borne de fin suit donc le fonctionnement de `numpy.arange()` et n’est pas incluse dans le tableau généré.

---

## Fonctions SolPOC

Les fonctions Python ne peuvent pas être enregistrées directement dans un fichier JSON.

Elles sont donc enregistrées avec leur nom :

```json
"algo": "DEvol",
"cost_function": "evaluate_R_Brg",
"selection": "selection_max"
```

Le scheduler retrouve ensuite dynamiquement les fonctions correspondantes dans SolPOC ou dans les fonctions personnalisées.

---

# Lancement du scheduler

Pour exécuter les plans présents dans `plan_experience` :

```powershell
solpoc-optimize
```

Il est recommandé de lancer cette commande depuis le même dossier que celui utilisé pour :

```powershell
solpoc-optimizer-init
```

Cela garantit que le même dossier `Workspace` est utilisé.

---

# Fonctionnement du scheduler

Le scheduler suit les étapes suivantes :

```text
1. Lecture des plans JSON
2. Extraction de la priorité
3. Tri des plans
4. Calcul du hash
5. Vérification des doublons
6. Conversion des paramètres
7. Reconstruction de Wl
8. Création de n_Stack et k_Stack
9. Sélection des fonctions SolPOC
10. Création du dossier de résultats
11. Exécution de l’optimisation
12. Génération des rapports et graphiques
13. Enregistrement du hash
14. Déplacement du plan
15. Passage au plan suivant
```

---

## Reconstruction de `Wl`

Le scheduler utilise la fonction :

```python
build_wl()
```

Lorsqu’il reçoit :

```json
"Wl": [
    400,
    800,
    5
]
```

il reconstruit :

```python
np.arange(400, 800, 5)
```

---

## Construction de la pile optique

Lorsque `Mat_Stack` et `Wl` sont disponibles, le scheduler construit automatiquement :

```python
n_Stack, k_Stack = sol.Made_Stack(
    Mat_Stack,
    Wl,
)
```

Ces données sont ensuite transmises aux fonctions scientifiques de SolPOC.

---

## Sélection dynamique des fonctions

Les noms enregistrés dans le JSON sont transformés en fonctions Python au moment de l’exécution.

Exemple :

```json
"algo": "DEvol"
```

devient :

```python
sol.DEvol
```

Le scheduler recherche d’abord la fonction dans les fonctions personnalisées, puis dans le module principal de SolPOC.

---

## Multiprocessing

Les différentes répétitions d’une expérience peuvent être exécutées en parallèle avec :

```python
multiprocessing.Pool
```

Le paramètre :

```json
"cpu_used": 4
```

permet de définir le nombre de processus utilisés.

Si `cpu_used` est absent ou invalide, une valeur par défaut est utilisée.

---

# Gestion des priorités

La priorité est intégrée au nom du fichier JSON.

Exemple :

```text
Bragg_Mirror_2026-06-18_10h30m00s_1.json
```

Le dernier nombre correspond à la priorité :

```text
1
```

Les plans sont triés par priorité croissante.

Ainsi :

```text
priorité 1
```

est exécutée avant :

```text
priorité 2
```

Une priorité non lisible reçoit une valeur par défaut élevée afin d’être traitée après les autres plans.

---

# Détection des doublons

Avant d’exécuter un plan, le scheduler calcule son hash.

Le hash représente les paramètres scientifiques du plan.

Les champs suivants sont ignorés :

```text
filename
priority
cpu_used
```

Deux plans ayant les mêmes paramètres scientifiques produisent donc le même hash.

Lorsqu’un hash est déjà présent dans :

```text
Workspace/hashes.json
```

le scheduler affiche un message de type :

```text
[SKIP] Expérience identique déjà exécutée.
```

Le plan n’est pas exécuté une deuxième fois et est déplacé dans :

```text
Workspace/plan_executer/
```

Le hash est enregistré uniquement après une exécution réussie.

---

# Gestion des erreurs

Le traitement de chaque expérience est protégé par des blocs `try/except`.

Lorsqu’un plan provoque une erreur :

1. l’erreur est affichée dans le terminal ;
2. le plan est déplacé dans `plan_failed` ;
3. le scheduler passe au plan suivant ;
4. le programme ne s’arrête pas complètement.

Exemple :

```text
[ERROR] Impossible de préparer le plan :
n_Stack is missing
```

Le plan concerné est déplacé dans :

```text
Workspace/plan_failed/
```

Cette gestion permet à une expérience incorrecte de ne pas bloquer toute la file d’attente.

---

# Commandes disponibles

## Initialiser le workspace

```powershell
solpoc-optimizer-init
```

---

## Restaurer les modèles

```powershell
solpoc-optimizer-init --force
```

---

## Lancer l’interface graphique

```powershell
solpoc-interface
```

---

## Lancer le scheduler

```powershell
solpoc-optimize
```

---

## Vérifier le chemin du workspace

```powershell
python -c "from solpoc_optimizer.paths import WORKSPACE_DIR; print(WORKSPACE_DIR)"
```

---

## Vérifier le fichier `paths.py` utilisé

```powershell
python -c "import solpoc_optimizer.paths as p; print(p.__file__)"
```

---

# Configuration du workspace

## Emplacement par défaut

Lorsque la variable `SOLPOC_OPTIMIZER_HOME` n’est pas définie, le workspace est situé dans :

```python
Path.cwd() / "Workspace"
```

Cela signifie que son emplacement dépend du dossier courant depuis lequel la commande est lancée.

Exemple :

```powershell
cd C:\Projet\Solpoc_optimizer
solpoc-optimizer-init
```

Le workspace sera créé dans :

```text
C:\Projet\Solpoc_optimizer\Workspace
```

Pour retrouver le même workspace, il faut lancer les commandes suivantes depuis ce même dossier :

```powershell
solpoc-interface
solpoc-optimize
```

---

## Emplacement personnalisé

Il est possible de définir un emplacement personnalisé avec :

```text
SOLPOC_OPTIMIZER_HOME
```

Dans le terminal PowerShell actuel :

```powershell
$env:SOLPOC_OPTIMIZER_HOME = "C:\MonWorkspace"
```

Pour enregistrer la variable pour l’utilisateur Windows :

```powershell
[Environment]::SetEnvironmentVariable(
    "SOLPOC_OPTIMIZER_HOME",
    "C:\MonWorkspace",
    "User"
)
```

Lorsque cette variable existe, elle est prioritaire sur le dossier courant.

---

## Supprimer la variable d’environnement

Dans le terminal actuel :

```powershell
Remove-Item Env:SOLPOC_OPTIMIZER_HOME -ErrorAction SilentlyContinue
```

Pour la supprimer des variables utilisateur Windows :

```powershell
[Environment]::SetEnvironmentVariable(
    "SOLPOC_OPTIMIZER_HOME",
    $null,
    "User"
)
```

Fermer puis rouvrir PowerShell après la suppression.

---

# Développement local

Le package est installé en mode éditable :

```powershell
python -m pip install --no-deps -e .
```

Les modifications réalisées dans :

```text
src/solpoc_optimizer/
```

sont donc directement prises en compte.

Une réinstallation est principalement nécessaire après une modification de :

```text
pyproject.toml
```

notamment lorsqu’une nouvelle commande console est ajoutée.

Dans ce cas :

```powershell
python -m pip install --no-deps -e . --force-reinstall
```

---

## Vérifier la version réellement importée

```powershell
python -c "import solpoc_optimizer; print(solpoc_optimizer.__file__)"
```

Pour vérifier une fonction précise :

```powershell
python -c "import inspect; import solpoc_optimizer.paths as p; print(inspect.getsource(p.get_workspace_dir))"
```

---

## Plans manuels copiés dans le workspace

Les plans présents dans :

```text
src/solpoc_optimizer/interface/manual_interface/
```

servent de modèles d’origine.

Les plans réellement utilisés par l’utilisateur sont copiés dans :

```text
Workspace/manual_interface/
```

Modifier uniquement le modèle du package ne modifie pas automatiquement la copie déjà présente dans le workspace.

Pour mettre à jour la copie :

```powershell
solpoc-optimizer-init --force
```

Cette commande pouvant écraser des personnalisations, il est recommandé de sauvegarder les plans modifiés.

---

# Résolution des problèmes fréquents

## La commande n’est pas reconnue

Exemple :

```text
solpoc-optimize : commande introuvable
```

Vérifier que l’environnement virtuel est actif :

```powershell
.\.venv\Scripts\Activate.ps1
```

Puis réinstaller le package :

```powershell
python -m pip install --no-deps -e .
```

---

## Le workspace est créé au mauvais emplacement

Vérifier le chemin utilisé :

```powershell
python -c "from solpoc_optimizer.paths import WORKSPACE_DIR; print(WORKSPACE_DIR)"
```

Vérifier la variable d’environnement :

```powershell
$env:SOLPOC_OPTIMIZER_HOME
```

Si aucune variable n’est définie, le workspace dépend du dossier courant.

---

## Les modifications de `paths.py` ne sont pas prises en compte

Vérifier le fichier importé :

```powershell
python -c "import solpoc_optimizer.paths as p; print(p.__file__)"
```

Vérifier le contenu réellement exécuté :

```powershell
python -c "import inspect; import solpoc_optimizer.paths as p; print(inspect.getsource(p.get_workspace_dir))"
```

---

## Le plan manuel utilise un ancien `plan_utils.py`

Ajouter temporairement dans le plan :

```python
import plan_utils

print(plan_utils.__file__)
```

Cela permet de connaître le fichier réellement chargé.

Le plan manuel utilise généralement :

```text
Workspace/manual_interface/plan_utils.py
```

et non directement le modèle présent dans le package.

---

## `Wl` contient toutes les longueurs d’onde dans le JSON

Le JSON doit enregistrer `Wl` sous la forme compacte :

```json
"Wl": [
    400,
    800,
    5
]
```

La fonction de conversion doit appeler :

```python
compact_wavelength_range(value)
```

pour la clé :

```text
Wl
```

Vérifier également que le bon fichier `plan_utils.py` est utilisé.

---

## Erreur `n_Stack is missing`

Le scheduler doit construire :

```python
n_Stack, k_Stack = sol.Made_Stack(
    Mat_Stack,
    Wl,
)
```

avant l’appel à :

```python
sol.get_parameters(**params)
```

Le plan doit également contenir un `Mat_Stack` et un domaine `Wl` valides.

---

## Avertissement `FigureCanvasAgg is non-interactive`

Exemple :

```text
FigureCanvasAgg is non-interactive, and thus cannot be shown
```

Cet avertissement est lié au backend non interactif de Matplotlib.

Les graphiques ne sont pas affichés dans une fenêtre, mais ils restent sauvegardés dans les dossiers de résultats.

Ce message n’est pas une erreur bloquante.

---

## Avertissement S4 / RCWA

Exemple :

```text
The RCWA solver will not be available because an S4 installation has not been found.
```

S4 est requis uniquement pour certaines fonctionnalités utilisant le solveur RCWA.

Les autres expériences peuvent continuer à fonctionner sans S4.

---

## Un plan incorrect arrête tout le scheduler

Vérifier que l’appel suivant est placé dans un bloc `try/except` :

```python
parameters = sol.get_parameters(**params)
```

Toute la préparation et l’exécution d’un plan doivent être protégées afin de déplacer le plan en erreur dans `plan_failed` puis de passer au suivant.

---

# Exemples de templates

Selon les modèles présents dans le workspace, SolPOC Optimizer peut notamment être utilisé pour :

* les revêtements antireflets ;
* les cellules photovoltaïques ;
* les revêtements Low-e ;
* les miroirs de Bragg ;
* la séparation spectrale ;
* les revêtements sélectifs ;
* l’optimisation de matériaux ;
* l’ajustement de signaux optiques.

---

# Limites actuelles

Les limites et perspectives du projet comprennent notamment :

* l’ajout de tests unitaires automatisés ;
* la mise en place d’une intégration continue ;
* l’amélioration de la validation des paramètres ;
* la prise en charge complète de tous les templates SolPOC ;
* l’installation simplifiée des dépendances optionnelles ;
* l’amélioration des messages d’erreur ;
* l’accélération de certains calculs ;
* l’intégration éventuelle de calculs GPU ;
* l’amélioration de l’interface utilisateur.

---

# Contributeurs

Projet de semestre réalisé à l’EPF Engineering School en collaboration avec le CNRS.

Contributeurs :

* Maxence Baïssas ;
* Guillaume De Montgolfier ;
* Victor Eymard ;
* Yassine Gharbi ;
* Émile Zanna.

---

# Résumé rapide

Installation locale :

```powershell
python -m pip install -e .
cd Solpoc_optimizer
python -m pip install --no-deps -e .
```

Initialisation :

```powershell
solpoc-optimizer-init
```

Interface graphique :

```powershell
solpoc-interface
```

Scheduler :

```powershell
solpoc-optimize
```

Workflow :

```text
Interface ou plan manuel
        ↓
Plan JSON
        ↓
plan_experience
        ↓
Scheduler
        ↓
SolPOC
        ↓
runs
        ↓
plan_executer ou plan_failed
```
