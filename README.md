L3_internship_Francois_Illan
=============================

The following file gathers the functions I made during my L3 Internship at the Lumin/Jean Perrin labs. I analyzed the kinetics of some signals produced by organoids shot by a laser beam. I also studied the propagation of these signals across the organoid and the theoretical temperature profile in the organoid.


WORKFLOW
--------

Kinetics of excited points:
    If you want to get the kinetics of the excited points, first use eclairement_moyen, and then acquire_pts_excites. Then you will be able to use affichage_rise_decay_all_runs and affichage_rise_decay_same_parameters (you can also use the function tracer_allure_moyenne to get the mean signal).

Propagation of the signal:
    In order to study the propagation of the signal, first execute acquire_zone_cartographie, then you can execute propagation_activation_complete and visualisation_grille_temporelle_interactive (I think that these algorithms need improvement).

Theoretical temperature profile:
    Finally, if you want to study theoretically the temperature in the organoid, you can use profil_temperature.


ENVIRONMENT & SETUP
--------------------

These functions are developed and tested using the following environment:
    - OS: Linux (Ubuntu)
    - Python: 3.11.15 (via Miniconda)
    - Environment Name: napari-env

Installation:
    To install all the required libraries, you may run this command in your terminal:

        pip install -r requirements.txt

File organisation:
    To ensure the functions run properly, your data directory should be organized as follows:

    /home/lumin/Documents/Data
    ├── Algos_Python
    ├── Images_organoides
    │   ├── 2025-08-26
    │   │   ├── Run01
    │   │   │   └── Data_pts_excites
    │   │   ├── Run02
    │   │   │   └── Data_pts_excites
    │   │   ├── Run03
    │   │   │   └── Data_pts_excites
    │   │   ├── Run04
    │   │   │   └── Data_pts_excites
    │   │   ├── Run05
    │   │   │   └── Data_pts_excites
    │   │   ├── Run06
    │   │   │   └── Data_pts_excites
    │   │   ├── Run07
    │   │   │   └── Data_pts_excites
    │   │   ├── Run08
    │   │   │   └── Data_pts_excites
    │   │   ├── Run09
    │   │   │   ├── Analysis
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run10
    │   │   │   ├── Analysis
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run11
    │   │   │   ├── Analysis
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run12
    │   │   │   └── Data_pts_excites
    │   │   ├── Run13
    │   │   │   ├── Analysis
    │   │   │   ├── Data_pts_excites
    │   │   │   ├── Data_ts_les_pts
    │   │   │   └── Images
    │   │   ├── Run14
    │   │   │   ├── Data_pts_excites
    │   │   │   ├── Data_ts_les_pts
    │   │   │   │   └── zone_carto_4_1_400
    │   │   │   └── Images
    │   │   └── Run15
    │   │       ├── Data_pts_excites
    │   │       ├── Images
    │   │       └── Images_rognee(602-1799)
    │   ├── 2025-08-27
    │   │   ├── Run011
    │   │   │   └── Data_pts_excites
    │   │   ├── Run02
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run03
    │   │   │   └── Data_pts_excites
    │   │   ├── Run04
    │   │   │   └── Data_pts_excites
    │   │   ├── Run05
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run06
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run07
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run08
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run09
    │   │   │   └── Data_pts_excites
    │   │   ├── Run10
    │   │   │   └── Data_pts_excites
    │   │   ├── Run11
    │   │   │   └── Data_pts_excites
    │   │   ├── Run12
    │   │   │   └── Data_pts_excites
    │   │   ├── Run13
    │   │   │   └── Data_pts_excites
    │   │   ├── Run14
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run15
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run16
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run17
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run18
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   ├── Run19
    │   │   │   ├── Data_pts_excites
    │   │   │   └── Images
    │   │   └── Run20
    │   │       └── Data_pts_excites
    │   └── 2025-08-28
    │       ├── Run01
    │       │   └── Data_pts_excites
    │       ├── Run02
    │       │   └── Data_pts_excites
    │       ├── Run03
    │       │   └── Data_pts_excites
    │       ├── Run04
    │       │   └── Data_pts_excites
    │       ├── Run05
    │       │   └── Data_pts_excites
    │       ├── Run06
    │       │   └── Data_pts_excites
    │       ├── Run07
    │       │   └── Data_pts_excites
    │       ├── Run08
    │       │   └── Data_pts_excites
    │       ├── Run09
    │       │   └── Data_pts_excites
    │       ├── Run10
    │       │   └── Data_pts_excites
    │       ├── Run11
    │       │   └── Data_pts_excites
    │       ├── Run12
    │       │   └── Data_pts_excites
    │       ├── Run13
    │       │   └── Data_pts_excites
    │       ├── Run14
    │       │   └── Data_pts_excites
    │       └── Run15
    │           └── Data_pts_excites

    Only the Run** that contains a case Images are interesting, we don't analyse the others. For all these runs, there is also a file called Run**.csv, parameters.txt adn evolution_temporelle_filtre.csv
    In the case Images, there are every frame in a .tiff
    You should put the file utils_analysis3.py next to the file utils_analysis2.py in the case Algos_Python.
