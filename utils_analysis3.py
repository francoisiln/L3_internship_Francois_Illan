import pandas as pd
import numpy as np
import tifffile as tff
import glob
import os
import matplotlib.pyplot as plt
import sys
import sys
sys.path.append('/home/lumin/Documents/Data/Algos_Python/')
import utils_analysis2 as utana
from scipy.signal import savgol_filter

utana.matplotlib_style()

def eclairement_moyen(Day, Run_number):
    """Calculate the average global illumination over time for a given run, 
    filter out artifact frames, and save the result as a CSV file.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run

    Returns:
        None: saves the filtered global mean dataframe to a CSV file and displays a plot
    """
    import sys
    import os
    import glob
    import re
    import shutil
    import pandas as pd
    import numpy as np
    import tifffile as tff
    import matplotlib.pyplot as plt

    base_day_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/'
    run_dir = os.path.join(base_day_dir, f'Run{Run_number}')
    special_run_dirs = glob.glob(os.path.join(base_day_dir, f'Run{Run_number}*'))
    
    for d in special_run_dirs:
        if os.path.basename(d) != f'Run{Run_number}':
            print(f"[INFO] Dossier spécial détecté ({os.path.basename(d)}). Pas d'excitation pour ce Run.")
            
            if os.path.exists(run_dir):
                try:
                    shutil.rmtree(run_dir)
                    print(f"[NETTOYAGE] Dossier fantôme supprimé avec succès : {run_dir}")
                except Exception:
                    pass
            return

    csv_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Run{Run_number}.csv'
    img_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Images/'
    output_final = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/evolution_temporelle_filtre.csv'
    param_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Parameters.txt'
    
    if os.path.exists(output_final):
        print(f"[SKIP] Le fichier {output_final} existe déjà. Calcul ignoré pour Run{Run_number}.")
        return

    filenames = sorted(glob.glob(os.path.join(img_dir, '*.tif*')))
    if not filenames:
        print(f"[ERREUR] Aucune image .tif trouvée dans {img_dir}")
        return
        
    global_means = []
    for img_path in filenames:
        img = tff.imread(img_path)
        global_means.append(np.mean(img))

    results_global_df = pd.DataFrame(global_means, columns=['Global_Mean'])

    tstart_frames = []
    cycle_time_s = 0.05  
    
    if os.path.exists(param_path):
        print(f"Extraction des Tstart pour l'éclairement global depuis : {param_path}")
        tstart_secondes = []
        
        with open(param_path, 'r', encoding='cp1252') as f:
            lines = f.readlines()
            
        for line in lines:
            if 'Cycle time' in line:
                match_cycle = re.findall(r'\d+\.?\d*', line)
                if match_cycle:
                    cycle_time_s = float(match_cycle[0]) * 1e-3  
                    print(f"  -> Cycle time détecté : {cycle_time_s} s")

        en_section_signals = False
        for line in lines:
            if 'Digital Signals' in line:
                en_section_signals = True
                continue
            if en_section_signals and 'DS1' in line:
                parts = line.split()
                if len(parts) >= 2:
                    tstart_secondes.append(float(parts[1]))
                    
        tstart_frames = [int(round(t / cycle_time_s)) for t in tstart_secondes]
        print(f"  -> Tstart secondes extraits : {tstart_secondes}")
        print(f"  -> Indices de frames calculés : {tstart_frames}")
    else:
        print(f"[ATTENTION] Fichier Parameters.txt introuvable à l'adresse : {param_path}. Le nettoyage global se fera par seuil de dérivée.")

    for col in results_global_df.columns:
        if tstart_frames:
            for idx in tstart_frames:
                indices_a_supprimer = range(idx, min(idx + 2, len(results_global_df)))
                results_global_df.loc[indices_a_supprimer, col] = np.nan
        else:
            derivee = results_global_df[col].diff()
            pics_indices = results_global_df.index[derivee > 400].tolist()
            for idx in pics_indices:
                indices_a_supprimer = range(idx, min(idx + 2, len(results_global_df)))
                results_global_df.loc[indices_a_supprimer, col] = np.nan
        
        results_global_df[col] = results_global_df[col].interpolate(method='linear', limit_direction='both')

    results_global_df.to_csv(output_final, index_label='Frame')
    print(f"Fichier unique sauvegardé avec succès ici : {output_final}")

    plt.figure(figsize=(10, 5))
    plt.plot(results_global_df.index * 50e-3, results_global_df['Global_Mean'], color='black', lw=1.5)
    plt.title("Intensité moyenne de l'image complète au cours du temps (Filtrée via Parameters.txt)")
    plt.xlabel('Temps (Secondes)')
    plt.ylabel('Moyenne globale (A.U.)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def acquire_pts_excites(Day, Run_number):
    """Extract, filter, and clean calcium traces from experimental artifacts 
    defined in a coordinate CSV file.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run

    Returns:
        None: saves the individual point traces as .npy files and the compiled dataframe to a CSV file
    """
    import sys
    import os
    import glob
    import re
    import shutil
    import pandas as pd
    import numpy as np
    import tifffile as tff
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    base_day_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/'
    run_dir = os.path.join(base_day_dir, f'Run{Run_number}')
    special_run_dirs = glob.glob(os.path.join(base_day_dir, f'Run{Run_number}*'))
    
    for d in special_run_dirs:
        if os.path.basename(d) != f'Run{Run_number}':
            print(f"[INFO] Dossier spécial détecté ({os.path.basename(d)}). Pas d'excitation pour ce Run. Fin de la fonction.")
            if os.path.exists(run_dir):
                try:
                    shutil.rmtree(run_dir)
                    print(f"[NETTOYAGE] Dossier fantôme supprimé avec succès : {run_dir}")
                except Exception:
                    pass
            return

    csv_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Run{Run_number}.csv'
    img_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Images/'
    data_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Data_pts_excites/'
    param_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Parameters.txt'
    
    chemin_global_filtre = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/evolution_temporelle_filtre.csv'
    output_nouveau_path = os.path.join(data_dir, 'Pts_excites_sans_pics.csv')

    # --- LOGIQUE D'ÉCRASEMENT ---
    # Si le dossier existe déjà, on le supprime entièrement et on repart de zéro
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"[SKIP] Le dossier {data_dir} existe déjà et n'est pas vide. Calcul ignoré pour Run{Run_number}.")
        return

    if not os.path.exists(csv_path):
        print(f"[ANNULATION] Le fichier de coordonnées CSV est introuvable : {csv_path}")
        return

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(chemin_global_filtre):
        print(f"Fichier global introuvable ({chemin_global_filtre}). Lancement de eclairement_moyen...")
        eclairement_moyen(Day, Run_number)

    tstart_frames = []
    cycle_time_s = 0.05  
    
    if os.path.exists(param_path):
        print(f"Extraction des Tstart depuis : {param_path}")
        tstart_secondes = []
        
        with open(param_path, 'r', encoding='cp1252') as f:
            lines = f.readlines()
            
        for line in lines:
            if 'Cycle time' in line:
                match_cycle = re.findall(r'\d+\.?\d*', line)
                if match_cycle:
                    cycle_time_s = float(match_cycle[0]) * 1e-3  
                    print(f"  -> Cycle time détecté : {cycle_time_s} s")

        en_section_signals = False
        for line in lines:
            if 'Digital Signals' in line:
                en_section_signals = True
                continue
            if en_section_signals and 'DS1' in line:
                parts = line.split()
                if len(parts) >= 2:
                    tstart_secondes.append(float(parts[1]))
        
        tstart_frames = [int(round(t / cycle_time_s)) for t in tstart_secondes]
        print(f"  -> Tstart secondes extraits : {tstart_secondes}")
        print(f"  -> Indices de frames calculés : {tstart_frames}")
    else:
        print(f"[ATTENTION] Fichier Parameters.txt introuvable à l'adresse : {param_path}. Le nettoyage se fera par seuil de dérivée.")

    df = pd.read_csv(csv_path)
    x_list = (1024 - df['x_um'] * 22.5 / 13).round().astype(int).tolist()
    y_list = (1024 - df['y_um'] * 22.5 / 13).round().astype(int).tolist()

    filenames = sorted(glob.glob(os.path.join(img_dir, '*.tif*')))
    size = 5
    all_traces = []

    for img_path in filenames:
        img = tff.imread(img_path)
        frame_values = []
        for x, y in zip(x_list, y_list):
            roi = img[max(0, y-size):min(1024, y+size), max(0, x-size):min(1024, x+size)]
            frame_values.append(np.mean(roi))
        all_traces.append(frame_values)

    results_filtre_df = pd.DataFrame(all_traces, columns=[f'Point_{i+1}' for i in range(len(x_list))])

    for col in results_filtre_df.columns:
        if tstart_frames:
            for idx in tstart_frames:
                indices_a_supprimer = range(idx, min(idx + 2, len(results_filtre_df)))
                results_filtre_df.loc[indices_a_supprimer, col] = np.nan
        else:
            derivee = results_filtre_df[col].diff()
            pics_indices = results_filtre_df.index[derivee > 200].tolist()
            for idx in pics_indices:
                indices_a_supprimer = range(idx, min(idx + 2, len(results_filtre_df)))
                results_filtre_df.loc[indices_a_supprimer, col] = np.nan
                
        results_filtre_df[col] = results_filtre_df[col].interpolate(method='linear', limit_direction='both')

    print(f"Chargement du bruit de fond global depuis : {chemin_global_filtre}")
    df_global_charge = pd.read_csv(chemin_global_filtre)
    global_means = df_global_charge['Global_Mean'].values

    grossiere_baseline = pd.Series(global_means).rolling(window=301, center=True, min_periods=1).median()
    signal_sans_derivee = global_means - grossiere_baseline
    seuil_intensite_pic = 15
    pics_par_intensite = signal_sans_derivee > seuil_intensite_pic

    derivee_globale = pd.Series(global_means).diff()
    pics_par_derivee = derivee_globale > 40

    est_un_pic = pics_par_intensite | pics_par_derivee

    masque_élargi = pd.Series(est_un_pic).rolling(window=60, center=False, min_periods=1).max()
    masque_élargi = masque_élargi.shift(-5, fill_value=0).astype(bool).values

    df_baseline = pd.Series(global_means.copy())
    df_baseline[masque_élargi] = np.nan
    baseline_interp_lineaire = df_baseline.interpolate(method='linear', limit_direction='both').values

    baseline_savgol_complete = savgol_filter(baseline_interp_lineaire, window_length=401, polyorder=2)

    global_baseline_interp = baseline_interp_lineaire.copy()
    global_baseline_interp[masque_élargi] = baseline_savgol_complete[masque_élargi]

    global_pics_brut = global_means - global_baseline_interp

    derivee_globale = pd.Series(global_pics_brut).diff()
    est_un_pic = (global_pics_brut > 15) | (derivee_globale > 40)
    indices_tous_pics = np.where(est_un_pic)[0]

    if len(indices_tous_pics) == 0:
        print(f"[INFO] Aucun pic détecté mathématiquement pour Run{Run_number}. Le signal filtré sera pris tel quel.")
        global_pics_brut = np.zeros(len(global_means))
        idx_debut, idx_milieu, idx_fin = 0, 0, 0
    else:
        ruptures = np.where(np.diff(indices_tous_pics) > 10)[0]
        fin_index_dans_liste = ruptures[0] if len(ruptures) > 0 else len(indices_tous_pics) - 1
        indices_premier_pic = indices_tous_pics[:fin_index_dans_liste + 1]

        idx_debut = int(indices_premier_pic[0])
        idx_milieu = int(indices_premier_pic[np.argmax(global_pics_brut[indices_premier_pic])])
        idx_fin = int(indices_premier_pic[-1])

    plt.figure(figsize=(10, 3))
    plt.plot(results_filtre_df.index*50e-3, global_means, color='black', lw=1.5, label='Signal Global Original')
    plt.plot(results_filtre_df.index*50e-3, global_baseline_interp, color='red', lw=2, linestyle='--', label='Baseline sans pics')
    plt.title('Intensité moyenne globale et sa Baseline interpolée')
    plt.xlabel('Temps (Secondes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(results_filtre_df.index*50e-3, global_pics_brut, color='green', lw=1.5, label='global_pics_brut')
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.title('Signal Global Intermédiaire : Dérive supprimée, mais petits pics conservés')
    plt.xlabel('Temps (Secondes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    denominateur = (global_pics_brut[idx_milieu] - (global_pics_brut[idx_debut] + global_pics_brut[idx_fin]) / 2)
    coeffs = []

    for column in results_filtre_df.columns:
        if denominateur == 0:
            coeff = 0
        else:
            coeff = (results_filtre_df[column][idx_milieu] - (results_filtre_df[column][idx_debut] + results_filtre_df[column][idx_fin]) / 2) / denominateur
        coeffs.append(coeff)

        signal_final = results_filtre_df[column] - global_pics_brut * coeff
        
        plt.figure(figsize=(10, 3))
        plt.plot(results_filtre_df.index * 0.05, signal_final, lw=1, alpha=1, label="après")
        plt.plot(results_filtre_df.index * 0.05, results_filtre_df[column], lw=1, alpha=1, label="avant")
        plt.title("Superposition AVANT / APRÈS")
        plt.xlabel("Temps (Secondes)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 3))
    for coeff, column in zip(coeffs, results_filtre_df.columns):
        signal_final = results_filtre_df[column] - global_pics_brut * coeff
        plt.plot(results_filtre_df.index * 0.05, signal_final, lw=1, alpha=1)
    plt.title("Superposition AVANT / APRÈS")
    plt.xlabel("Temps (Secondes)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    for i, (coeff, column) in enumerate(zip(coeffs, results_filtre_df.columns)):
        signal_soustrait = results_filtre_df[column].values - global_pics_brut * coeff
        nom_fichier = f"({x_list[i]},{y_list[i]}).npy"
        np.save(os.path.join(data_dir, nom_fichier), signal_soustrait)

    fichiers_npy = glob.glob(os.path.join(data_dir, "(*,*).npy"))
    dict_all_points = {}
    for filepath in sorted(fichiers_npy):
        filename = os.path.basename(filepath)
        dict_all_points[filename.replace('.npy', '')] = np.load(filepath)
        
    data_pt_par_pt_df = pd.DataFrame(dict_all_points)
    data_pt_par_pt_df.insert(0, 'Frame', range(1, len(data_pt_par_pt_df) + 1))

    data_pt_par_pt_df.to_csv(output_nouveau_path, index=False)
    print(f"[SUCCÈS] Traitement terminé. Fichier créé : {output_nouveau_path}")

def acquire_rise_decay(Day, Run_number, overwrite=True):
    """Calculate and plot the dynamic rise and decay times (tau) for each maxima that have been detected for every excited point.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        overwrite (bool): if True, deletes any existing rise/decay CSV file before computing (default: True)

    Returns:
        None: writes a comprehensive statistical summary to a CSV file and generates diagnostic plots
    """
    import sys
    import os
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # ─── INITIALISATION ET SÉCURITÉ ───────────────────────────────────────────
    base_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/'
    dossiers_existants = glob.glob(os.path.join(base_dir, f"Run{Run_number}*"))
    
    for d in dossiers_existants:
        if d.endswith('*'):
            print(f"[SÉCURITÉ] Le dossier '{os.path.basename(d)}' contient une étoile. Traitement abandonné.")
            return

    sys.path.append('/home/lumin/Documents/Data/Algos_Python/')
    import utils_analysis2 as utana2
    import utils_analysis3 as utana3

    utana2.matplotlib_style()

    csv_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Run{Run_number}.csv'
    data_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Data_pts_excites/'
    output_csv_path = os.path.join(data_dir, 'data_rise_decay.csv')

    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(output_csv_path) and overwrite:
        try:
            os.remove(output_csv_path)
        except Exception:
            pass

    print("Début de acquire_pts_excites")
    utana3.acquire_pts_excites(Day, Run_number)
    
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    x_list = (1024 - df['x_um'] * 22.5 / 13).round().astype(int).tolist()
    y_list = (1024 - df['y_um'] * 22.5 / 13).round().astype(int).tolist()
    coordonnees_points = np.column_stack((x_list, y_list))
    
    liste_tau_decay = [0 for _ in range(len(coordonnees_points))]
    liste_tau_rise = [0 for _ in range(len(coordonnees_points))]
    indices_pics_globaux = []

    # ─── TRAITEMENT UNITAIRE ET TRACÉS INDIVIDUELS ────────────────────────────
    print(f"--- Début de l'analyse cinétique pour {len(coordonnees_points)} points ---")

    for i in range(len(coordonnees_points)):
        x_c, y_c = coordonnees_points[i]
        tau_decay, tau_rise, pics_point, signal = rise_decay_dff(Day, Run_number, x_c, y_c)
        liste_tau_decay[i] = tau_decay
        liste_tau_rise[i] = tau_rise
        indices_pics_globaux.extend(pics_point)

        vecteur_temps = np.arange(len(signal)) * 0.05
        
        plt.figure(figsize=(10, 5))
        plt.plot(vecteur_temps, signal, color='blue', lw=1.5, label='Fluorescence')
        
        for idx_p, t_p in enumerate(np.array(pics_point) * 0.05):
            plt.axvline(x=t_p, color='orange', linestyle='--', alpha=0.8, lw=1.2, label="Pics détectés" if idx_p == 0 else "")
        
        plt.title(f'Fluorescence filtrée au point (x={x_c}, y={y_c})')
        plt.xlabel('Temps (Secondes)')
        plt.ylabel('Intensité (A.U.)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    indices_pics_globaux = np.unique(indices_pics_globaux)
    indices_pics_globaux.sort()

    # ─── FILTRAGE DYNAMIQUE DES OUTLIERS ──────────────────────────────────────
    tous_les_decays = np.array([el[0] for sl in liste_tau_decay for el in sl if not np.isnan(el[0]) and el[0] < 1000])
    tous_les_rises = np.array([el[0] for sl in liste_tau_rise for el in sl if not np.isnan(el[0]) and el[0] < 1000])

    seuil = 5

    if len(tous_les_decays) > 0:
        while True:
            seuil_dynamique_decay = seuil * np.median(tous_les_decays)
            mask = tous_les_decays > seuil_dynamique_decay
            if not np.any(mask): 
                break 
            tous_les_decays = tous_les_decays[~mask]
    else:
        seuil_dynamique_decay = 1.0

    if len(tous_les_rises) > 0:
        while True:
            seuil_dynamique_rise = seuil * np.median(tous_les_rises)
            mask = tous_les_rises > seuil_dynamique_rise
            if not np.any(mask): 
                break 
            tous_les_rises = tous_les_rises[~mask]
    else:
        seuil_dynamique_rise = 1.0

    # ─── TRACÉ DES ÉVOLUTIONS CHRONOLOGIQUES ──────────────────────────────────
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1, title="Évolution du Tau Decay par point d'acquisition", xlabel="Instant du pic (s)", ylabel="Tau Decay (s)")
    ax2 = plt.subplot(1, 2, 2, title="Évolution du Tau Rise par point d'acquisition", xlabel="Instant du début de la montée (s)", ylabel="Tau Rise (s)")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    palette = plt.colormaps.get_cmap('viridis').resampled(len(coordonnees_points))

    for i in range(len(coordonnees_points)):
        couleur = palette(i)
        
        # Chronologie Decay
        points_d = np.array([[el[0], el[1] * 0.05] for el in liste_tau_decay[i] if not np.isnan(el[0])])
        if points_d.size > 0:
            points_d = points_d[np.argsort(points_d[:, 1])]
            valid = points_d[:, 0] < seuil_dynamique_decay
            if np.any(valid):
                ax1.plot(points_d[valid, 1], points_d[valid, 0], color=couleur, alpha=0.5, lw=1.5)
                ax1.scatter(points_d[valid, 1], points_d[valid, 0], color=couleur, alpha=0.9, edgecolors='black', s=50)

        # Chronologie Rise
        points_r = np.array([[el[0], el[1] * 0.05] for el in liste_tau_rise[i] if not np.isnan(el[0])])
        if points_r.size > 0:
            points_r = points_r[np.argsort(points_r[:, 1])]
            valid = points_r[:, 0] < seuil_dynamique_rise
            if np.any(valid):
                ax2.plot(points_r[valid, 1], points_r[valid, 0], color=couleur, alpha=0.5, lw=1.5)
                ax2.scatter(points_r[valid, 1], points_r[valid, 0], color=couleur, alpha=0.9, edgecolors='black', s=50)

    plt.tight_layout()
    plt.show()

    # ─── BINNING PAR IMPULSION ET EXPORT CSV ──────────────────────────────────
    run_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/'
    fichier_csv_excitations = os.path.join(run_dir, 'indices_excitations.csv')
    temps_excitations_secondes = pd.read_csv(fichier_csv_excitations)['Frame_Index'].values * 0.05 if os.path.exists(fichier_csv_excitations) else []

    fourchette_pics, fourchette = 100, 150  
    liste_pics_regroupes = []
    
    if indices_pics_globaux.size > 0:
        groupe_courant = [indices_pics_globaux[0]]
        for p in indices_pics_globaux[1:]:
            if p - groupe_courant[-1] <= fourchette_pics:
                groupe_courant.append(p)
            else:
                liste_pics_regroupes.append(int(np.mean(groupe_courant)))
                groupe_courant = [p]
        liste_pics_regroupes.append(int(np.mean(groupe_courant)))

    donnees_decay_par_impulsion = {p: [] for p in liste_pics_regroupes}
    donnees_rise_par_impulsion = {p: [] for p in liste_pics_regroupes}
    rows_individual_data = []

    for idx_pt, (x_c, y_c) in enumerate(coordonnees_points):
        for frame_impulsion in liste_pics_regroupes:
            tau_d_trouve, frame_d_trouve = np.nan, np.nan
            for tau, idx in liste_tau_decay[idx_pt]:
                if not np.isnan(tau) and tau < seuil_dynamique_decay and np.abs(idx - frame_impulsion) <= fourchette:
                    tau_d_trouve, frame_d_trouve = tau, idx
                    donnees_decay_par_impulsion[frame_impulsion].append(tau)
                    break

            tau_r_trouve, frame_r_trouve = np.nan, np.nan
            for tau, idx in liste_tau_rise[idx_pt]:
                if not np.isnan(tau) and tau < seuil_dynamique_rise and np.abs(idx - frame_impulsion) <= fourchette:
                    tau_r_trouve, frame_r_trouve = tau, idx
                    donnees_rise_par_impulsion[frame_impulsion].append(tau)
                    break

            if not np.isnan(tau_d_trouve) or not np.isnan(tau_r_trouve):
                rows_individual_data.append({
                    'Type_Ligne': 'Donnee_Individuelle', 'Impulsion_Frame_Moyenne': frame_impulsion, 'Impulsion_Temps_s': frame_impulsion * 0.05,
                    'Point_X': x_c, 'Point_Y': y_c, 'Tau_Rise_s': tau_r_trouve, 'Frame_Start_Rise': frame_r_trouve,
                    'Tau_Decay_s': tau_d_trouve, 'Frame_Start_Decay': frame_d_trouve,
                    'Stat_Metrique': np.nan, 'Stat_Rise_Valeur': np.nan, 'Stat_Decay_Valeur': np.nan
                })

    rows_stats = []
    metrics = [('Min', np.min), ('Q1', lambda x: np.percentile(x, 25)), ('Mediane', np.median),
               ('Q3', lambda x: np.percentile(x, 75)), ('Max', np.max), ('Moyenne', np.mean), ('Std', np.std)]
    
    for f_imp in liste_pics_regroupes:
        d_vals, r_vals = np.array(donnees_decay_par_impulsion[f_imp]), np.array(donnees_rise_par_impulsion[f_imp])
        for name, func in metrics:
            rows_stats.append({
                'Type_Ligne': 'Stat_Global_Impulsion', 'Impulsion_Frame_Moyenne': f_imp, 'Impulsion_Temps_s': f_imp * 0.05,
                'Point_X': np.nan, 'Point_Y': np.nan, 'Tau_Rise_s': np.nan, 'Frame_Start_Rise': np.nan,
                'Tau_Decay_s': np.nan, 'Frame_Start_Decay': np.nan,
                'Stat_Metrique': name,
                'Stat_Rise_Valeur': func(r_vals) if r_vals.size > 0 else np.nan,
                'Stat_Decay_Valeur': func(d_vals) if d_vals.size > 0 else np.nan
            })

    pd.DataFrame(rows_individual_data + rows_stats).to_csv(output_csv_path, index=False)
    print("[SUCCÈS] Tableau complet sauvegardé.")

    # ─── TRACÉ DES DISTRIBUTIONS PAR BOXPLOT ET VIOLIN ────────────────────────
    temps_moyens_d = [f * 0.05 for f in liste_pics_regroupes if donnees_decay_par_impulsion[f]]
    donnees_brutes_decay = [donnees_decay_par_impulsion[f] for f in liste_pics_regroupes if donnees_decay_par_impulsion[f]]
    temps_moyens_r = [f * 0.05 for f in liste_pics_regroupes if donnees_rise_par_impulsion[f]]
    donnees_brutes_rise = [donnees_rise_par_impulsion[f] for f in liste_pics_regroupes if donnees_rise_par_impulsion[f]]

    # Tracé 1 : Boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, data, pos in zip([ax1, ax2], ["Distribution du Tau Decay par impulsion", "Distribution du Tau Rise par impulsion"], [donnees_brutes_decay, donnees_brutes_rise], [temps_moyens_d, temps_moyens_r]):
        ax.set_title(title)
        ax.set_xlabel("Instant du pic (s)")
        ax.set_ylabel("Temps caractéristique (s)")
        ax.grid(True, alpha=0.3)
        if data:
            ax.boxplot(data, positions=pos, widths=2, zorder=5, patch_artist=True,
                       boxprops=dict(facecolor='#EAEAEA'), medianprops=dict(color='red', linewidth=2), manage_ticks=False)
        for t_v in temps_excitations_secondes:
            ax.axvline(x=t_v, color='red', linestyle=':', alpha=0.6, lw=1.2)
    plt.tight_layout()
    plt.show()

    # Tracé 2 : Violins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, data, pos in zip([ax1, ax2], ["Distribution du Tau Decay par impulsion", "Distribution du Tau Rise par impulsion"], [donnees_brutes_decay, donnees_brutes_rise], [temps_moyens_d, temps_moyens_r]):
        ax.set_title(title)
        ax.set_xlabel("Instant du pic (s)")
        ax.set_ylabel("Temps caractéristique (s)")
        ax.grid(True, alpha=0.3)
        if data:
            v = ax.violinplot(data, positions=pos, widths=7, showmedians=True)
            for pc in v['bodies']:
                pc.set_facecolor('#EAEAEA')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            v['cmedians'].set_color('red')
            v['cmedians'].set_linewidth(3.5)
        for t_v in temps_excitations_secondes:
            ax.axvline(x=t_v, color='red', linestyle=':', alpha=0.6, lw=1.2)
    plt.tight_layout()
    plt.show()

def rise_decay_all_runs():
    """Batch process and analyze calcium transient kinetics (rise and decay times) 
    across all experimental runs for multiple specific dates.

    Args:
        None

    Returns:
        None: orchestrates the sequential execution of acquire_rise_decay for each run and date
    """
    import sys
    import importlib

    sys.path.append('/home/lumin/Documents/Data/Algos_Python/')
    import utils_analysis2 as utana2
    import utils_analysis3 as utana3

    importlib.reload(utana3)
    utana2.matplotlib_style()

    # ─── BATCH PROCESSING MULTI-DATES ─────────────────────────────────────────
    
    # Session du 2025-08-26 (Runs 01 à 15)
    Day = "2025-08-26"
    for i in range(1, 16):
        Run_number = f"{i:02d}"
        print(f"{Day},{Run_number} commencé")
        utana3.acquire_rise_decay(Day, Run_number)

    # Session du 2025-08-27 (Runs 01 à 20)
    Day = "2025-08-27"
    for i in range(1, 21):
        Run_number = f"{i:02d}"
        print(f"{Day},{Run_number} commencé")
        utana3.acquire_rise_decay(Day, Run_number)

    # Session du 2025-08-28 (Runs 01 à 15)
    Day = "2025-08-28"
    for i in range(1, 16):
        Run_number = f"{i:02d}"
        print(f"{Day},{Run_number} commencé")
        utana3.acquire_rise_decay(Day, Run_number)

def affichage_rise_decay_same_param():
    """Group experimental runs by their digital excitation signal protocol, filter out 
    aberrant kinetic values, and generate comparative grid plots for rise and decay times.

    Args:
        None

    Returns:
        None: processes and aggregates data directly from disk to display multiple matplotlib figures
    """
    import os
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    base_dir = '/home/lumin/Documents/Data/Images_organoides/'
    jours_disponibles = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    groupes_protocoles = {}

    # ─── 1. PARSING DE L'ARBORESCENCE ET PARSING DES PARAMÈTRES ──────────────────
    for jour in jours_disponibles:
        chemin_jour = os.path.join(base_dir, jour)
        for run_dir in sorted(glob.glob(os.path.join(chemin_jour, "Run*"))):
            if '*' in run_dir or not os.path.isdir(run_dir):
                continue
                
            run_name = os.path.basename(run_dir)
            csv_data_path = os.path.join(run_dir, 'Data_pts_excites', 'data_rise_decay.csv')
            
            param_path = os.path.join(run_dir, 'Parameters.txt')
            if not os.path.exists(param_path):
                param_path = os.path.join(run_dir, 'Parameters.csv')
                
            if os.path.exists(csv_data_path) and os.path.exists(param_path):
                try:
                    liste_tstart = []
                    with open(param_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for ligne in f:
                            elements = ligne.strip().split()
                            if len(elements) >= 3 and elements[0].startswith('DS') and elements[0][2:].isdigit():
                                try:
                                    liste_tstart.append(float(elements[1]))
                                except ValueError:
                                    continue
                    
                    if not liste_tstart:
                        continue
                        
                    cle_protocole = tuple(np.sort(np.unique(liste_tstart)).round(1))
                    df_brut = pd.read_csv(csv_data_path)
                    df_individuel = df_brut[df_brut['Type_Ligne'] == 'Donnee_Individuelle'].copy()
                    
                    if df_individuel.empty:
                        continue
                        
                    df_individuel['Jour_Source'] = jour[-2:]
                    df_individuel['Run_Source'] = run_name
                    
                    groupes_protocoles.setdefault(cle_protocole, []).append(df_individuel)
                    
                except Exception:
                    pass

    # ─── 2. AGREGATION ET FILTRAGE STATISTIQUE PAR PROTOCOLE ───────────────────
    for index_prot, (protocole, liste_dfs) in enumerate(groupes_protocoles.items(), 1):
        if not liste_dfs:
            continue
            
        impulsions_theoriques_s = sorted([float(t) for t in protocole])
        df_global_protocole = pd.concat(liste_dfs, ignore_index=True)

        # Filtrage vectorisé des valeurs aberrantes (Seuil dynamique)
        seuil_facteur = 5
        for col in ['Tau_Decay_s', 'Tau_Rise_s']:
            vals = df_global_protocole[col].dropna().values
            vals = vals[vals > 0]
            if len(vals) > 0:
                mediane = np.median(vals)
                seuil_dynamique = seuil_facteur * mediane
                # Filtrage itératif convergent pour exclure les valeurs hors limites
                masque_valides = vals <= seuil_dynamique
                while np.any(~masque_valides) and len(vals[masque_valides]) > 0:
                    vals = vals[masque_valides]
                    mediane = np.median(vals)
                    seuil_dynamique = seuil_facteur * mediane
                    masque_valides = vals <= seuil_dynamique
                
                df_global_protocole = df_global_protocole[
                    (df_global_protocole[col].isna()) | (df_global_protocole[col] <= seuil_dynamique)
                ]
        
        # Formatage du sous-titre listant les sources incluses
        df_sources = df_global_protocole[['Jour_Source', 'Run_Source']].drop_duplicates().sort_values(by=['Jour_Source', 'Run_Source'])
        liste_sources_str = [f"{row['Jour_Source']}({row['Run_Source']})" for _, row in df_sources.iterrows()]
        sous_titre_runs = "Runs inclus : " + ", ".join(liste_sources_str)
        
        instants_impulsions = np.sort(df_global_protocole['Impulsion_Temps_s'].unique())
        
        structure_decay, structure_rise, positions_temps = [], [], []
        scatter_decay_x, scatter_decay_y = [], []
        scatter_rise_x, scatter_rise_y = [], []
        jitter_amplitude = 0.5 
        
        for t_imp in instants_impulsions:
            sous_df = df_global_protocole[df_global_protocole['Impulsion_Temps_s'] == t_imp]
            decays_vals = sous_df['Tau_Decay_s'].dropna().values
            rises_vals = sous_df['Tau_Rise_s'].dropna().values
            
            if len(decays_vals) > 0 or len(rises_vals) > 0:
                structure_decay.append(list(decays_vals))
                structure_rise.append(list(rises_vals))
                positions_temps.append(t_imp)
                
                scatter_decay_x.extend(np.random.normal(t_imp, jitter_amplitude, size=len(decays_vals)))
                scatter_decay_y.extend(decays_vals)
                scatter_rise_x.extend(np.random.normal(t_imp, jitter_amplitude, size=len(rises_vals)))
                scatter_rise_y.extend(rises_vals)

        if not scatter_decay_x and not scatter_rise_x:
            continue

        # Regroupement des indices de pics proches
        fourchette_pics, fourchette = 100, 150  
        indices_pics_globaux = np.unique(np.array([int(round(t / 0.05)) for t in instants_impulsions]))
        indices_pics_globaux.sort()

        liste_pics_regroupes = []
        if len(indices_pics_globaux) > 0:
            groupe_courant = [indices_pics_globaux[0]]
            for p in indices_pics_globaux[1:]:
                if p - groupe_courant[-1] <= fourchette_pics:
                    groupe_courant.append(p)
                else:
                    liste_pics_regroupes.append(int(np.mean(groupe_courant)))
                    groupe_courant = [p]
            liste_pics_regroupes.append(int(np.mean(groupe_courant)))

        donnees_decay_par_impulsion = {p: [] for p in liste_pics_regroupes}
        donnees_rise_par_impulsion = {p: [] for p in liste_pics_regroupes}

        # Indexation par fenêtrage temporel (Frames d'acquisition à 0.05s)
        for _, row in df_global_protocole.iterrows():
            t_reel = row['Impulsion_Temps_s']
            tau_d, tau_r = row['Tau_Decay_s'], row['Tau_Rise_s']
            if pd.isna(t_reel):
                continue
                
            frame_reelle = int(round(t_reel / 0.05))
            for frame_impulsion in liste_pics_regroupes:
                if np.abs(frame_reelle - frame_impulsion) <= fourchette:
                    if not pd.isna(tau_d):
                        donnees_decay_par_impulsion[frame_impulsion].append(tau_d)
                    if not pd.isna(tau_r):
                        donnees_rise_par_impulsion[frame_impulsion].append(tau_r)
                    break

        temps_moyens_d, donnees_brutes_decay = [], []
        temps_moyens_r, donnees_brutes_rise = [], []
        for frame_impulsion in liste_pics_regroupes:
            vals_d = donnees_decay_par_impulsion[frame_impulsion]
            vals_r = donnees_rise_par_impulsion[frame_impulsion]
            if len(vals_d) > 0:
                temps_moyens_d.append(frame_impulsion * 0.05)
                donnees_brutes_decay.append(vals_d)
            if len(vals_r) > 0:
                temps_moyens_r.append(frame_impulsion * 0.05)
                donnees_brutes_rise.append(vals_r)

        # ─── 3. GENERATION DES GRAPHIQUES PLOT MATPLOTLIB ────────────────────────
        fig, axs = plt.subplots(3, 2, figsize=(16, 15), sharex=True)
        plt.suptitle(f"Protocole d'excitation n°{index_prot} : {impulsions_theoriques_s} secondes\n{sous_titre_runs}", fontsize=14, fontweight='bold')
        
        # Rangée 0: Scatters individuels
        axs[0, 0].scatter(scatter_decay_x, scatter_decay_y, color='#2E685A', alpha=0.4, edgecolors='none', s=15, zorder=4)
        axs[0, 0].set_title("Données Individuelles : Tau Decay (Points)", fontsize=11, fontweight='semibold')
        axs[0, 0].set_ylabel("Tau Decay (s)"); axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].scatter(scatter_rise_x, scatter_rise_y, color='#A39E1A', alpha=0.5, edgecolors='none', s=15, zorder=4)
        axs[0, 1].set_title("Données Individuelles : Tau Rise (Points)", fontsize=11, fontweight='semibold')
        axs[0, 1].set_ylabel("Tau Rise (s)"); axs[0, 1].grid(True, alpha=0.3)

        # Rangée 1: Boxplots Historiques Superposés
        axs[1, 0].set_title("Distribution Historique : Tau Decay (Superposés)", fontsize=11, fontweight='semibold')
        axs[1, 0].set_ylabel("Tau Decay (s)"); axs[1, 0].grid(True, alpha=0.3)
        if structure_decay:
            axs[1, 0].boxplot(structure_decay, positions=positions_temps, widths=2, zorder=5, patch_artist=True,
                            boxprops=dict(facecolor='#D1E8E2', color='black', linewidth=1.2, alpha=0.5), manage_ticks=False)

        axs[1, 1].set_title("Distribution Historique : Tau Rise (Superposés)", fontsize=11, fontweight='semibold')
        axs[1, 1].set_ylabel("Tau Rise (s)"); axs[1, 1].grid(True, alpha=0.3)
        if structure_rise:
            axs[1, 1].boxplot(structure_rise, positions=positions_temps, widths=2, zorder=5, patch_artist=True,
                            boxprops=dict(facecolor='#EFEA5A', color='black', linewidth=1.2, alpha=0.5), manage_ticks=False)

        # Rangée 2: Boxplots Regroupés par Impulsions
        axs[2, 0].set_title("Distribution du Tau Decay par impulsion (Regroupés)")
        axs[2, 0].set_xlabel("Instant du pic (Secondes)"); axs[2, 0].set_ylabel("Tau Decay (s)"); axs[2, 0].grid(True, alpha=0.3)

        axs[2, 1].set_title("Distribution du Tau Rise par impulsion (Regroupés)")
        axs[2, 1].set_xlabel("Instant du début de la montée (Secondes)"); axs[2, 1].set_ylabel("Tau Rise (s)"); axs[2, 1].grid(True, alpha=0.3)

        if len(donnees_brutes_decay) > 0:
            axs[2, 0].boxplot(donnees_brutes_decay, positions=temps_moyens_d, widths=2, zorder=5, patch_artist=True,
                            boxprops=dict(facecolor='#EAEAEA', color='black', linewidth=1.5),
                            whiskerprops=dict(color='black', linewidth=1.5, linestyle='--'),
                            capprops=dict(color='black', linewidth=1.5),
                            medianprops=dict(color='red', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='black', markersize=4, alpha=0.5),
                            manage_ticks=False)
            for t_pos, vals in zip(temps_moyens_d, donnees_brutes_decay):
                axs[2, 0].text(t_pos, axs[2, 0].get_ylim()[0], f'N={len(vals)}', ha='center', va='bottom', fontsize=7, color='black')

        if len(donnees_brutes_rise) > 0:
            axs[2, 1].boxplot(donnees_brutes_rise, positions=temps_moyens_r, widths=2, zorder=5, patch_artist=True,
                            boxprops=dict(facecolor='#EAEAEA', color='black', linewidth=1.5),
                            whiskerprops=dict(color='black', linewidth=1.5, linestyle='--'),
                            capprops=dict(color='black', linewidth=1.5),
                            medianprops=dict(color='red', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='black', markersize=4, alpha=0.5),
                            manage_ticks=False)
            for t_pos, vals in zip(temps_moyens_r, donnees_brutes_rise):
                axs[2, 1].text(t_pos, axs[2, 1].get_ylim()[0], f'N={len(vals)}', ha='center', va='bottom', fontsize=7, color='black')

        # Traçage des lignes de repères de stimulation rouge en pointillés
        for t_th in impulsions_theoriques_s:
            for row in range(3):
                axs[row, 0].axvline(x=t_th, color='red', linestyle=':', alpha=0.6, lw=1.2, zorder=0)
                axs[row, 1].axvline(x=t_th, color='red', linestyle=':', alpha=0.6, lw=1.2, zorder=0)

        axs[2, 0].legend([plt.Line2D([0], [0], color='red', linestyle=':')], ['Impulsion paramètre'], loc='upper right')

        all_x_points = scatter_decay_x + scatter_rise_x
        if all_x_points:
            plt.xlim(min(all_x_points) - 5, max(all_x_points) + 5)

        plt.tight_layout()
        plt.show()

def affichage_rise_decay_all_runs():
    """Aggregate kinetic data across all available experimental days and runs, filter out 
    artifact and aberrant values, and generate global distribution plots for rise and decay times.

    Args:
        None

    Returns:
        tuple: contains the global average rise time and decay time (tau_rise_moyen, tau_decay_moyen) as floats
    """
    import os
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    base_dir = '/home/lumin/Documents/Data/Images_organoides/'
    all_data_list = []

    # ─── 1. COLLECTE GLOBALE DES DONNÉES ──────────────────────────────────────
    jours_disponibles = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for jour in jours_disponibles:
        chemin_jour = os.path.join(base_dir, jour)
        for run_dir in sorted(glob.glob(os.path.join(chemin_jour, "Run*"))):
            if '*' in run_dir or not os.path.isdir(run_dir):
                continue
                
            csv_data_path = os.path.join(run_dir, 'Data_pts_excites', 'data_rise_decay.csv')
            
            if os.path.exists(csv_data_path):
                try:
                    df_brut = pd.read_csv(csv_data_path)
                    df_individuel = df_brut[df_brut['Type_Ligne'] == 'Donnee_Individuelle'].copy()
                    
                    if not df_individuel.empty:
                        df_individuel = df_individuel[(df_individuel['Tau_Decay_s'] > 0) & (df_individuel['Tau_Rise_s'] > 0)]
                        
                        if not df_individuel.empty:
                            df_individuel['Jour_Source'] = jour[-2:]
                            df_individuel['Run_Source'] = os.path.basename(run_dir)
                            all_data_list.append(df_individuel)
                except Exception:
                    pass

    if not all_data_list:
        print("Aucune donnée trouvée.")
        return np.nan, np.nan

    # ─── 2. CONCATÉNATION ET FILTRAGE DES OUTLIERS ────────────────────────────
    df_global = pd.concat(all_data_list, ignore_index=True)

    seuil = 5
    for col in ['Tau_Decay_s', 'Tau_Rise_s']:
        vals = df_global[col].dropna().values
        vals = vals[vals > 0]
        if len(vals) > 0:
            while True:
                seuil_dynamique = seuil * np.median(vals)
                mask = vals > seuil_dynamique
                if not np.any(mask):
                    break
                vals = vals[~mask]
            df_global = df_global[(df_global[col].isna()) | (df_global[col] <= seuil_dynamique)]
    
    # ─── 3. BINNING SPATIOTEMPOREL ET JITTER ──────────────────────────────────
    instants_impulsions = np.sort(df_global['Impulsion_Temps_s'].unique())
    
    jitter_amplitude = 0.5 
    scatter_decay_x = np.random.normal(df_global['Impulsion_Temps_s'], jitter_amplitude)
    scatter_decay_y = df_global['Tau_Decay_s']
    scatter_rise_x = np.random.normal(df_global['Impulsion_Temps_s'], jitter_amplitude)
    scatter_rise_y = df_global['Tau_Rise_s']

    fourchette_pics, fourchette = 100, 150  
    indices_pics_globaux = np.unique(np.array([int(round(t / 0.05)) for t in instants_impulsions]))
    indices_pics_globaux.sort()

    liste_pics_regroupes = []
    if len(indices_pics_globaux) > 0:
        groupe_courant = [indices_pics_globaux[0]]
        for p in indices_pics_globaux[1:]:
            if p - groupe_courant[-1] <= fourchette_pics:
                groupe_courant.append(p)
            else:
                liste_pics_regroupes.append(int(np.mean(groupe_courant)))
                groupe_courant = [p]
        liste_pics_regroupes.append(int(np.mean(groupe_courant)))

    donnees_decay_par_impulsion = {p: [] for p in liste_pics_regroupes}
    donnees_rise_par_impulsion = {p: [] for p in liste_pics_regroupes}

    for _, row in df_global.iterrows():
        frame_reelle = int(round(row['Impulsion_Temps_s'] / 0.05))
        for frame_impulsion in liste_pics_regroupes:
            if np.abs(frame_reelle - frame_impulsion) <= fourchette:
                donnees_decay_par_impulsion[frame_impulsion].append(row['Tau_Decay_s'])
                donnees_rise_par_impulsion[frame_impulsion].append(row['Tau_Rise_s'])
                break

    temps_moyens_d = [f * 0.05 for f in liste_pics_regroupes if donnees_decay_par_impulsion[f]]
    donnees_brutes_decay = [donnees_decay_par_impulsion[f] for f in liste_pics_regroupes if donnees_decay_par_impulsion[f]]
    
    temps_moyens_r = [f * 0.05 for f in liste_pics_regroupes if donnees_rise_par_impulsion[f]]
    donnees_brutes_rise = [donnees_rise_par_impulsion[f] for f in liste_pics_regroupes if donnees_rise_par_impulsion[f]]

    # ─── 4. PLOTTING GLOBAL ───────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(16, 15))
    plt.suptitle("Analyse Globale", fontsize=16, fontweight='bold')

    # Tracés individuels (Scatter)
    axs[0, 1].scatter(scatter_decay_x, scatter_decay_y, color='#2E685A', alpha=0.3, s=10)
    axs[0, 1].set_title("Tau Decay : Individuel")

    axs[0, 0].scatter(scatter_rise_x, scatter_rise_y, color='#A39E1A', alpha=0.3, s=10)
    axs[0, 0].set_title("Tau Rise : Individuel")

    # Distributions (Boxplots)
    if donnees_brutes_decay:
        axs[1, 1].boxplot(donnees_brutes_decay, positions=temps_moyens_d, widths=8, patch_artist=True, manage_ticks=False)
        for t_pos, vals in zip(temps_moyens_d, donnees_brutes_decay):
            axs[1, 1].text(t_pos, axs[1, 1].get_ylim()[0], f'N={len(vals)}', ha='center', va='bottom', fontsize=7)
    axs[1, 1].set_title("Distribution Decay (Par impulsion)")

    if donnees_brutes_rise:
        axs[1, 0].boxplot(donnees_brutes_rise, positions=temps_moyens_r, widths=8, patch_artist=True, manage_ticks=False)
        for t_pos, vals in zip(temps_moyens_r, donnees_brutes_rise):
            axs[1, 0].text(t_pos, axs[1, 0].get_ylim()[0], f'N={len(vals)}', ha='center', va='bottom', fontsize=7)
    axs[1, 0].set_title("Distribution Rise (Par impulsion)")

    for ax in axs.flatten():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Instant des impulsions (s)")
        ax.set_ylabel("Temps caractéristique (s)")
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    tau_rise_moyen = df_global['Tau_Rise_s'].mean()
    tau_decay_moyen = df_global['Tau_Decay_s'].mean()

    return tau_rise_moyen, tau_decay_moyen

'''def rise_decay(Day, Run_number, x_cible, y_cible):
        """Detect local calcium peaks at a specific coordinate and fit exponential models 
        to extract their individual characteristic rise and decay times (tau).

        Args:
            Day (str): date or identifier of the experimental day (folder name)
            Run_number (int or str): number of the specific experimental run
            x_cible (int): target x-coordinate of the processed point
            y_cible (int): target y-coordinate of the processed point

        Returns:
            tuple: contains lists of fitted tau_decay tuples, tau_rise tuples, detected peak indices, and the flattened signal array
        """
        import sys
        import os
        import numpy as np
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks
        from scipy.ndimage import uniform_filter1d 

        # ─── INITIALISATION ET SÉCURITÉ ───────────────────────────────────────────
        sys.path.append('/home/lumin/Documents/Data/Algos_Python/')
        import utils_analysis2 as utana2

        utana2.matplotlib_style()

        data_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Data_pts_excites/'
        chemin_fichier = os.path.join(data_dir, f"({x_cible},{y_cible}).npy")
        
        if not os.path.exists(chemin_fichier):
            return [], [], []
            
        signal = np.load(chemin_fichier)

        # ─── TRAITEMENT DU SIGNAL ET DÉTECTION DES PICS ───────────────────────────
        baseline = uniform_filter1d(signal, size=500)
        signal_flat = signal - baseline
        signal_smooth = uniform_filter1d(signal_flat, size=10)

        indices_pics, _ = find_peaks(
            signal_smooth,
            distance=800,        
            prominence=100,      
            width=10,            
            rel_height=0.6,
        )

        indices_valides = []
        for pic in indices_pics:
            fenetre_locale = signal[pic:min(pic + 20, len(signal))]
            if len(fenetre_locale) > 1:
                if np.min(np.diff(fenetre_locale)) < -400:
                    continue
            indices_valides.append(pic)

        indices_pics = np.array(indices_valides)
        indices_pics = indices_pics[(indices_pics > 100) & (indices_pics < len(signal) - 5)]

        tau_decay = [0 for _ in range(len(indices_pics))]
        tau_rise = [0 for _ in range(len(indices_pics))]

        # ─── COUPE ET FIT EXPO : DECAY ────────────────────────────────────────────
        for i in range(1, 1 + len(indices_pics)):
            idx_start = indices_pics[i-1]
            
            if i < len(indices_pics):
                prochain_pic = indices_pics[i]
                if (prochain_pic - idx_start) < 300:  
                    idx_start = prochain_pic
                    idx_fin = min(len(signal), (i+1 < len(indices_pics) and indices_pics[i+1] - 150) or len(signal))
                else:
                    idx_fin = min(len(signal), prochain_pic - 150)
            else:
                idx_fin = len(signal)

            derivee_potentielle = np.diff(signal[idx_start:idx_fin])
            indices_remontee = np.where(derivee_potentielle > 40)[0]
            
            if len(indices_remontee) > 0 and indices_remontee[0] > 0.9 * (idx_fin - idx_start):
                idx_fin = idx_start + indices_remontee[0]

            if idx_fin - idx_start < 30:
                tau_decay[i-1] = (np.nan, idx_start)
                continue

            signal_decroissant = signal[idx_start:idx_fin].tolist()
            vecteur_temps_decroissant = np.arange(len(signal_decroissant)) * 0.05

            def modele_expo_decay(t, K, tau, C):
                return K * np.exp(-t / tau) + C

            p0 = [signal_decroissant[0], 2.0, signal_decroissant[-1]]
            
            try:
                popt, _ = curve_fit(modele_expo_decay, vecteur_temps_decroissant, signal_decroissant, p0=p0, maxfev=10000)
                tau_decay[i-1] = (popt[1], idx_start)
            except RuntimeError:
                tau_decay[i-1] = (np.nan, idx_start)

        # ─── COUPE ET FIT EXPO : RISE ─────────────────────────────────────────────
        for i in range(1, 1 + len(indices_pics)):
            idx_fin = indices_pics[i-1]
            
            if i > 1 and (idx_fin - indices_pics[i-2]) < 300:  
                tau_rise[i-1] = (np.nan, idx_fin)
                continue

            borne_theorique = max(0, idx_fin - 120)
            indices_remontee = np.where(np.diff(signal[borne_theorique:idx_fin]) > 100)[0]
            idx_start = borne_theorique + indices_remontee[-1] if len(indices_remontee) > 0 else borne_theorique

            while idx_start < len(signal) - 5:
                if np.mean(signal[idx_start:idx_start+3]) < np.mean(signal[idx_start+3:idx_start+6]):
                    break
                idx_start += 1

            if idx_fin - idx_start < 4:
                tau_rise[i-1] = (np.nan, idx_start)
                continue

            signal_croissant = signal[idx_start:idx_fin].tolist()
            vecteur_temps_croissant = np.arange(len(signal_croissant)) * 0.05

            def modele_expo_rise(t, K, tau, C):
                return K * (1 - np.exp(-t / tau)) + C

            p0 = [signal_croissant[-1] - signal_croissant[0], 2.0, signal_croissant[0]]
            
            try:
                popt, _ = curve_fit(modele_expo_rise, vecteur_temps_croissant, signal_croissant, p0=p0, maxfev=10000)
                tau_rise[i-1] = (popt[1], idx_start)
            except RuntimeError:
                tau_rise[i-1] = (np.nan, idx_start)

        return tau_decay, tau_rise, indices_pics, signal_flat'''

def rise_decay_dff(Day, Run_number, x_cible, y_cible):
    """Detect local calcium peaks on a rolling Delta F/F (DFF) normalized signal 
    at a specific coordinate and extract their individual rise and decay kinetics.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        x_cible (int): target x-coordinate of the processed point
        y_cible (int): target y-coordinate of the processed point

    Returns:
        tuple: contains lists of fitted tau_decay tuples, tau_rise tuples, detected peak indices, and the normalized signal array (dff)
    """
    import sys
    import os
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d, percentile_filter

    # ─── INITIALISATION ET CHARGEMENT ─────────────────────────────────────────
    sys.path.append('/home/lumin/Documents/Data/Algos_Python/')
    import utils_analysis2 as utana2

    utana2.matplotlib_style()

    data_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Data_pts_excites/'
    chemin_fichier = os.path.join(data_dir, f"({x_cible},{y_cible}).npy")
    
    if not os.path.exists(chemin_fichier):
        return [], [], []
        
    signal = np.load(chemin_fichier)

    # ─── CALCUL DU NOUVEAU ΔF/F ───────────────────────────────────────────────
    window_size = int(30 / 0.05)  # 30 sec -> 600 frames
    f0 = percentile_filter(signal, percentile=8, size=window_size)
    f0_safe = np.where(np.abs(f0) < 1e-6, 1e-6, f0)

    signal_flat = signal - f0
    signal_dff = signal_flat / f0_safe  
    signal_smooth = uniform_filter1d(signal_dff, size=10)

    # ─── DÉTECTION DES PICS SUR LE ΔF/F ───────────────────────────────────────
    indices_pics, _ = find_peaks(
        signal_smooth,
        distance=800,
        prominence=0.05,   
        width=10,
        rel_height=0.6,
    )

    indices_valides = []
    for pic in indices_pics:
        fenetre_locale = signal[pic:min(pic + 20, len(signal))]
        if len(fenetre_locale) > 1:
            if np.min(np.diff(fenetre_locale)) < -400:
                continue
        indices_valides.append(pic)

    indices_pics = np.array(indices_valides)
    indices_pics = indices_pics[(indices_pics > 100) & (indices_pics < len(signal) - 5)]

    tau_decay = [0 for _ in range(len(indices_pics))]
    tau_rise  = [0 for _ in range(len(indices_pics))]

    # ─── COUPE ET FIT EXPO : DECAY ────────────────────────────────────────────
    for i in range(1, 1 + len(indices_pics)):
        idx_start = indices_pics[i-1]

        if i < len(indices_pics):
            prochain_pic = indices_pics[i]
            if (prochain_pic - idx_start) < 300:
                idx_start = prochain_pic
                idx_fin = min(len(signal), (i+1 < len(indices_pics) and indices_pics[i+1] - 150) or len(signal))
            else:
                idx_fin = min(len(signal), prochain_pic - 150)
        else:
            idx_fin = len(signal)

        derivee_potentielle = np.diff(signal[idx_start:idx_fin])
        indices_remontee = np.where(derivee_potentielle > 40)[0]

        if len(indices_remontee) > 0 and indices_remontee[0] > 0.9 * (idx_fin - idx_start):
            idx_fin = idx_start + indices_remontee[0]

        if idx_fin - idx_start < 30:
            tau_decay[i-1] = (np.nan, idx_start)
            continue

        signal_decroissant = signal[idx_start:idx_fin].tolist()
        vecteur_temps_decroissant = np.arange(len(signal_decroissant)) * 0.05

        def modele_expo_decay(t, K, tau, C):
            return K * np.exp(-t / tau) + C

        p0 = [signal_decroissant[0], 2.0, signal_decroissant[-1]]

        try:
            popt, _ = curve_fit(modele_expo_decay, vecteur_temps_decroissant, signal_decroissant, p0=p0, maxfev=10000)
            tau_decay[i-1] = (popt[1], idx_start)
        except RuntimeError:
            tau_decay[i-1] = (np.nan, idx_start)

    # ─── COUPE ET FIT EXPO : RISE ─────────────────────────────────────────────
    for i in range(1, 1 + len(indices_pics)):
        idx_fin = indices_pics[i-1]

        if i > 1 and (idx_fin - indices_pics[i-2]) < 300:
            tau_rise[i-1] = (np.nan, idx_fin)
            continue

        borne_theorique = max(0, idx_fin - 120)
        indices_remontee = np.where(np.diff(signal[borne_theorique:idx_fin]) > 100)[0]
        idx_start = borne_theorique + indices_remontee[-1] if len(indices_remontee) > 0 else borne_theorique

        while idx_start < len(signal) - 5:
            if np.mean(signal[idx_start:idx_start+3]) < np.mean(signal[idx_start+3:idx_start+6]):
                break
            idx_start += 1

        if idx_fin - idx_start < 4:
            tau_rise[i-1] = (np.nan, idx_start)
            continue

        signal_croissant = signal[idx_start:idx_fin].tolist()
        vecteur_temps_croissant = np.arange(len(signal_croissant)) * 0.05

        def modele_expo_rise(t, K, tau, C):
            return K * (1 - np.exp(-t / tau)) + C

        p0 = [signal_croissant[-1] - signal_croissant[0], 2.0, signal_croissant[0]]

        try:
            popt, _ = curve_fit(modele_expo_rise, vecteur_temps_croissant, signal_croissant, p0=p0, maxfev=10000)
            tau_rise[i-1] = (popt[1], idx_start)
        except RuntimeError:
            tau_rise[i-1] = (np.nan, idx_start)

    return tau_decay, tau_rise, indices_pics, signal_dff

def tracer_allure_moyenne(): 
    """Retrieve the global average kinetic parameters across all runs, generate a 
    standardized calcium spike regressor, and plot its average temporal profile.

    Args:
        None

    Returns:
        None: extracts parameters from the global analysis to display the fitted regressor curve
    """
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    sys.path.append('/home/lumin/Documents/Data/Algos_Python/')
    import utils_analysis2 as utana2
    import utils_analysis3 as utana3

    utana2.matplotlib_style()

    # ─── EXTRACTION ET GÉNÉRATION DU REGRESSEUR ──────────────────────────────
    tau_rise_moyen, tau_decay_moyen = utana3.affichage_rise_decay_all_runs()

    exposure      = 0.05
    window_reg    = tau_rise_moyen + tau_decay_moyen * 5
    window_center = tau_rise_moyen

    regressor = utana2.create_regressor(
        exposure=exposure,
        rise_time=tau_rise_moyen,
        decay_time=tau_decay_moyen,
        window_reg=window_reg,
        window_center=window_center
    )

    temps_reg = np.linspace(0, window_reg, len(regressor))

    # ─── TRACÉ GRAPHIQUE ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(temps_reg, regressor, color='#2E685A', lw=2,
            label=f'τ_rise={tau_rise_moyen:.2f}s  τ_decay={tau_decay_moyen:.2f}s')
    plt.axvline(x=window_center + tau_rise_moyen, color='red', linestyle='--', alpha=0.7, lw=1.5, label='Pic')
    plt.title("Regresseur calcique moyen (tous runs)", fontsize=13, fontweight='bold')
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude normalisée")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def acquire_zone_cartographie(Day, Run_number, pas, num_ref, taille_carre):
    """Generate a spatial grid of points around a target coordinate, extract their 
    calcium traces, and clean them by removing global background illumination artifacts.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        pas (int): grid step size in pixels used to downsample the target square area
        num_ref (int): reference index among the excited points to center the target coordinate square
        taille_carre (int): width/height of the square grid area in pixels

    Returns:
        None: saves individual grid point traces as .npy files and compiles the dataset into a CSV file
    """
    import sys
    import os
    import glob
    import re
    import shutil
    import pandas as pd
    import numpy as np
    import tifffile as tff
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    # ─── INITIALISATION ET SÉCURITÉ RUN ───────────────────────────────────────
    base_day_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/'
    run_dir = os.path.join(base_day_dir, f'Run{Run_number}')
    special_run_dirs = glob.glob(os.path.join(base_day_dir, f'Run{Run_number}*'))
    
    for d in special_run_dirs:
        if os.path.basename(d) != f'Run{Run_number}':
            print(f"[INFO] Dossier spécial détecté ({os.path.basename(d)}). Pas d'excitation pour ce Run. Fin de la fonction.")
            if os.path.exists(run_dir):
                try:
                    shutil.rmtree(run_dir)
                    print(f"[NETTOYAGE] Dossier fantôme supprimé avec succès : {run_dir}")
                except Exception:
                    pass
            return

    csv_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Run{Run_number}.csv'
    img_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Images/'
    
    # Configuration du sous-dossier dynamique à l'intérieur de Data_ts_les_pts
    base_data_pts_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Data_ts_les_pts/'
    nom_sous_dossier = f'zone_carto_{pas}_{num_ref}_{taille_carre}'
    data_dir = os.path.join(base_data_pts_dir, nom_sous_dossier)
    
    param_path = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Parameters.txt'
    chemin_global_filtre = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/evolution_temporelle_filtre.csv'
    output_nouveau_path = os.path.join(data_dir, 'data_pt_par_pt.csv')

    if os.path.exists(data_dir):
        print(f"[NETTOYAGE] Suppression des anciennes données dans : {data_dir}")
        shutil.rmtree(data_dir)

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ANNULATION] Le fichier de coordonnées CSV est introuvable : {csv_path}")
        return
        
    if not os.path.exists(chemin_global_filtre):
        print(f"Fichier global introuvable ({chemin_global_filtre}). Lancement de eclairement_moyen...")
        eclairement_moyen(Day, Run_number)

    tstart_frames = [] 
    cycle_time_s = 0.05  
    
    # ─── LECTURE DES PARAMÈTRES MATÉRIELS ─────────────────────────────────────
    if os.path.exists(param_path):
        print(f"Extraction des Tstart pour l'éclairement global depuis : {param_path}")
        tstart_secondes = []
        
        with open(param_path, 'r', encoding='cp1252') as f:
            lines = f.readlines()
            
        for line in lines:
            if 'Cycle time' in line:
                match_cycle = re.findall(r'\d+\.?\d*', line)
                if match_cycle:
                    cycle_time_s = float(match_cycle[0]) * 1e-3  
                    print(f"  -> Cycle time détecté : {cycle_time_s} s")

        en_section_signals = False
        for line in lines:
            if 'Digital Signals' in line:
                en_section_signals = True
                continue
            if en_section_signals and 'DS1' in line:
                parts = line.split()
                if len(parts) >= 2:
                    tstart_secondes.append(float(parts[1]))
                    
        tstart_frames = [int(round(t / cycle_time_s)) for t in tstart_secondes]
        print(f"  -> Tstart secondes extraits : {tstart_secondes}")
        print(f"  -> Indices de frames calculés : {tstart_frames}")
    else:
        print(f"[ATTENTION] Fichier Parameters.txt introuvable. Le nettoyage global se fera par seuil de dérivée.")

    # ─── MAILLAGE DE LA GRILLE SPATIALE SOLIDE ────────────────────────────────
    df = pd.read_csv(csv_path)

    first_x_pixel = int(round(1024 - df['x_um'].iloc[num_ref] * 22.5 / 13))
    first_y_pixel = int(round(1024 - df['y_um'].iloc[num_ref] * 22.5 / 13))

    x_min = max(0, first_x_pixel - taille_carre // 2)
    x_max = min(1024, first_x_pixel + taille_carre // 2)
    y_min = max(0, first_y_pixel - taille_carre // 2)
    y_max = min(1024, first_y_pixel + taille_carre // 2)

    x_coords = np.arange(x_min, x_max, pas)
    y_coords = np.arange(y_min, y_max, pas)
    xx, yy = np.meshgrid(x_coords, y_coords)

    x_list = xx.flatten().tolist()
    y_list = yy.flatten().tolist()

    print(f"Nombre de points générés dans le carré : {len(x_list)}")

    # ─── EXTRACTION DES TRACES DEPUIS LES STACKS TIF ──────────────────────────
    filenames = sorted(glob.glob(os.path.join(img_dir, '*.tif*')))
    size = 5
    all_traces = []

    for img_path in filenames:
        img = tff.imread(img_path)
        frame_values = []
        for x, y in zip(x_list, y_list):
            roi = img[max(0, y-size):min(1024, y+size), max(0, x-size):min(1024, x+size)]
            frame_values.append(np.mean(roi))
        all_traces.append(frame_values)

    results_filtre_df = pd.DataFrame(all_traces, columns=[f'Point_{i+1}' for i in range(len(x_list))])

    # ─── NETTOYAGE INDIVIDUEL PAR ARTEFACT TEMPOREL ───────────────────────────
    for col in results_filtre_df.columns:
        if tstart_frames:
            for idx in tstart_frames:
                indices_a_supprimer = range(idx, min(idx + 20, len(results_filtre_df)))
                results_filtre_df.loc[indices_a_supprimer, col] = np.nan
        else:
            derivee = results_filtre_df[col].diff()
            pics_indices = results_filtre_df.index[derivee > 200].tolist()
            for idx in pics_indices:
                indices_a_supprimer = range(idx, min(idx + 20, len(results_filtre_df)))
                results_filtre_df.loc[indices_a_supprimer, col] = np.nan
                
        results_filtre_df[col] = results_filtre_df[col].interpolate(method='linear', limit_direction='both')

    # ─── EXTRACTION DE LA BASELINE DU BRUIT DE FOND GLOBAL ────────────────────
    print(f"Chargement du bruit de fond global depuis : {chemin_global_filtre}")
    df_global_charge = pd.read_csv(chemin_global_filtre)
    global_means = df_global_charge['Global_Mean'].values

    grossiere_baseline = pd.Series(global_means).rolling(window=301, center=True, min_periods=1).median()
    signal_sans_derivee = global_means - grossiere_baseline
    pics_par_intensite = signal_sans_derivee > 15

    derivee_globale = pd.Series(global_means).diff()
    pics_par_derivee = derivee_globale > 40
    est_un_pic = pics_par_intensite | pics_par_derivee

    masque_élargi = pd.Series(est_un_pic).rolling(window=60, center=False, min_periods=1).max()
    masque_élargi = masque_élargi.shift(-5, fill_value=0).astype(bool).values

    df_baseline = pd.Series(global_means.copy())
    df_baseline[masque_élargi] = np.nan
    baseline_interp_lineaire = df_baseline.interpolate(method='linear', limit_direction='both').values

    baseline_savgol_complete = savgol_filter(baseline_interp_lineaire, window_length=401, polyorder=2)

    global_baseline_interp = baseline_interp_lineaire.copy()
    global_baseline_interp[masque_élargi] = baseline_savgol_complete[masque_élargi]

    global_pics_brut = global_means - global_baseline_interp

    derivee_globale = pd.Series(global_pics_brut).diff()
    est_un_pic = (global_pics_brut > 15) | (derivee_globale > 40)
    indices_tous_pics = np.where(est_un_pic)[0]

    if len(indices_tous_pics) == 0:
        print(f"[INFO] Aucun pic détecté mathématiquement pour Run{Run_number}. Le signal filtré sera pris tel quel.")
        global_pics_brut = np.zeros(len(global_means))
        idx_debut, idx_milieu, idx_fin = 0, 0, 0
    else:
        ruptures = np.where(np.diff(indices_tous_pics) > 10)[0]
        fin_index_dans_liste = ruptures[0] if len(ruptures) > 0 else len(indices_tous_pics) - 1
        indices_premier_pic = indices_tous_pics[:fin_index_dans_liste + 1]

        idx_debut = int(indices_premier_pic[0])
        idx_milieu = int(indices_premier_pic[np.argmax(global_pics_brut[indices_premier_pic])])
        idx_fin = int(indices_premier_pic[-1])

    # Tracé 1 : Baseline globale
    plt.figure(figsize=(10, 3))
    plt.plot(results_filtre_df.index*50e-3, global_means, color='black', lw=1.5, label='Signal Global Original')
    plt.plot(results_filtre_df.index*50e-3, global_baseline_interp, color='red', lw=2, linestyle='--', label='Baseline sans pics')
    plt.title('Intensité moyenne globale et sa Baseline interpolée')
    plt.xlabel('Temps (Secondes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Tracé 2 : Signal global intermédiaire
    plt.figure(figsize=(10, 3))
    plt.plot(results_filtre_df.index*50e-3, global_pics_brut, color='green', lw=1.5, label='global_pics_brut')
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.title('Signal Global Intermédiaire : Dérive supprimée, mais petits pics conservés')
    plt.xlabel('Temps (Secondes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ─── SOUSTRACTION DU BRUIT DU FOND PAR POINT ET COMPILATION ───────────────
    denominateur = (global_pics_brut[idx_milieu] - (global_pics_brut[idx_debut] + global_pics_brut[idx_fin]) / 2)
    coeffs = []

    plt.figure(figsize=(10, 3))
    for column in results_filtre_df.columns:
        if denominateur == 0:
            coeff = 0
        else:
            coeff = (results_filtre_df[column][idx_milieu] - (results_filtre_df[column][idx_debut] + results_filtre_df[column][idx_fin]) / 2) / denominateur
        coeffs.append(coeff)

        signal_final = results_filtre_df[column] - global_pics_brut * coeff
        plt.plot(results_filtre_df.index * 0.05, signal_final, lw=1, alpha=1)

    plt.title("Compilation de tous les points")
    plt.xlabel("Temps (Secondes)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ─── SAUVEGARDE UNITAIRE NPY ET PACKAGING CSV ─────────────────────────────
    for i, (coeff, column) in enumerate(zip(coeffs, results_filtre_df.columns)):
        signal_soustrait = results_filtre_df[column].values - global_pics_brut * coeff
        nom_fichier = f"({x_list[i]},{y_list[i]}).npy"
        np.save(os.path.join(data_dir, nom_fichier), signal_soustrait)

    fichiers_npy = glob.glob(os.path.join(data_dir, "(*,*).npy"))
    dict_all_points = {}
    for filepath in sorted(fichiers_npy):
        filename = os.path.basename(filepath)
        dict_all_points[filename.replace('.npy', '')] = np.load(filepath)
        
    data_pt_par_pt_df = pd.DataFrame(dict_all_points)
    data_pt_par_pt_df.insert(0, 'Frame', range(1, len(data_pt_par_pt_df) + 1))

    data_pt_par_pt_df.to_csv(output_nouveau_path, index=False)
    print(f"[SUCCÈS] Traitement terminé. Fichier créé : {output_nouveau_path}")

def profil_temperature(Day, Run_number, profondeur=50e-6):
    """Simulate and plot the 3D spatiotemporal temperature distribution induced by 
    femtosecond laser pulse trains using an interactive matplotlib interface.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        profondeur (float): focal depth inside the tissue in meters (default: 50e-6)

    Returns:
        None: opens an interactive UI with sliders for time, depth, and wavelength selection
    """
    import sys
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons
    from matplotlib.ticker import FormatStrFormatter

    # ─── 1. CHARGEMENT ET PARAMÈTRES PHYSIQUES ────────────────────────────────
    csv_path = f"/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Run{Run_number}.csv"
    raw = np.genfromtxt(csv_path, delimiter=',', names=True)
    points_data = np.column_stack((raw['x_um'], raw['y_um']))

    wx, wy, wz = 2e-6, 2e-6, 8e-6
    D = 1.43e-7
    tau_macro, f_rep, dt_femi = 1e-3, 1e6, 1e-6
    E_pulse = 50e-3 / f_rep
    P_moyen = 50e-3
    rho, Cp, T_amb = 1000.0, 4180.0, 37.0

    absorption_dict = {
        '940 nm (0.305 cm^-1)': 0.305 * 100,
        '1035 nm (0.20 cm^-1)': 0.20 * 100,
        '1100 nm (0.19 cm^-1)': 0.19 * 100
    }

    # ─── 2. CONVERSION DES COORDONNÉES ET MAILLAGE SPATIAL ───────────────────
    scale_factor = 13 / 22.5
    x_px = 1024 - points_data[:, 0] * (22.5 / 13)
    y_px = 1024 - points_data[:, 1] * (22.5 / 13)
    x_centers_m = x_px * scale_factor * 1e-6
    y_centers_m = y_px * scale_factor * 1e-6

    L, N = 200e-6, 120
    x_mean_m, y_mean_m = np.mean(x_centers_m), np.mean(y_centers_m)
    x = np.linspace(x_mean_m - L, x_mean_m + L, N)
    y = np.linspace(y_mean_m - L, y_mean_m + L, N)
    X, Y = np.meshgrid(x, y)

    t_pulses = np.array([30.0, 90.0, 150.0, 210.0])

    # ─── 3. LOGIQUE DU MODÈLE THERMIQUE (FONCTION DE GREEN) ───────────────────
    def G_base(X, Y, u, z, dT_amp):
        sx2 = wx**2 / 2 + 4 * D * u
        sy2 = wy**2 / 2 + 4 * D * u
        sz2 = wz**2 / 2 + 4 * D * u

        norm = (
            np.sqrt(1 + 8 * D * u / wx**2)
            * np.sqrt(1 + 8 * D * u / wy**2)
            * np.sqrt(1 + 8 * D * u / wz**2)
        )

        sum_exp = np.zeros((u.shape[0], X.shape[0], X.shape[1]))
        for xc, yc in zip(x_centers_m, y_centers_m):
            expo = -(X - xc)**2 / sx2 - (Y - yc)**2 / sy2 - z**2 / sz2
            sum_exp += np.exp(expo)

        return dT_amp * sum_exp / norm

    def T_pulse_multi_fs(X, Y, t, z=0.0, mu_a=1.0):
        dT_total = np.zeros_like(X)
        V_focus = (np.pi**1.5) * (wx * wy * wz) / np.sqrt(2)
        Factor_abs = (np.exp(-mu_a * profondeur) - np.exp(-mu_a * (profondeur + wz))) / (rho * Cp * V_focus)
        dT_amp_pulse = E_pulse * Factor_abs
        dT_amp_macro = P_moyen * Factor_abs

        for t_start in t_pulses:
            if t < t_start:
                continue

            t_elapsed = t - t_start

            if t_elapsed > (tau_macro + 2e-3):
                u = np.array([t_elapsed - tau_macro / 2]).reshape(1, 1, 1)
                dT_total += G_base(X, Y, u, z, dT_amp_macro * tau_macro)[0]
                continue

            max_duration = min(t_elapsed, tau_macro)
            n_pulses_total = int(np.floor(max_duration / dt_femi)) + 1
            if n_pulses_total <= 0:
                continue

            t_cutoff = max(0.0, max_duration - 2e-3)
            n_pulses_old = int(np.floor(t_cutoff / dt_femi))

            if n_pulses_old > 0:
                u_old = np.linspace(t_elapsed - t_cutoff, t_elapsed, 3).reshape(-1, 1, 1)
                Gu_old = G_base(X, Y, u_old, z, dT_amp_macro)
                dT_total += np.trapz(Gu_old, u_old.ravel(), axis=0)

            n_pulses_recent = n_pulses_total - n_pulses_old
            if n_pulses_recent > 0:
                indices_recent = np.arange(n_pulses_old, n_pulses_total)
                t_fs_pulses = t_start + indices_recent * dt_femi
                u_recent = t - t_fs_pulses
                u_recent = u_recent[u_recent > 1e-11].reshape(-1, 1, 1)
                if len(u_recent) > 0:
                    dT_total += np.sum(G_base(X, Y, u_recent, z, dT_amp_pulse), axis=0)

        return T_amb + dT_total

    # Print de sécurité initial des pics de température
    X_pts, Y_pts = x_centers_m.reshape(-1, 1), y_centers_m.reshape(-1, 1)
    for label, mu_a in absorption_dict.items():
        print(f"--- {label} ---")
        for i, t_start in enumerate(t_pulses):
            t_peak = t_start + tau_macro
            data = T_pulse_multi_fs(X_pts, Y_pts, t_peak, z=0.0, mu_a=mu_a)
            print(f"  Pulse #{i+1} (t = {t_peak:.5f} s) : T_max = {data.max():.3f} °C")

    current_mu_a = absorption_dict['1035 nm (0.20 cm^-1)']

    # ─── 4. TRACÉ INITIAL DE LA SUBPLOTS MATPLOTLIB ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.35, right=0.75)

    t0_grossier, dt0_fin, z0_m = 30.0, 1.0 * 1e-3, 0.0
    t0 = t0_grossier + dt0_fin

    data0 = T_pulse_multi_fs(X, Y, t0, z=z0_m, mu_a=current_mu_a)

    im = ax.imshow(
        data0, extent=[-L * 1e6, L * 1e6, -L * 1e6, L * 1e6],
        origin='lower', cmap='hot', vmin=37.0, vmax=38.5
    )
    cbar = fig.colorbar(im, ax=ax, label='Température (°C)')
    cbar.formatter = FormatStrFormatter('%.2f')
    cbar.update_ticks()

    ax.set_xlabel('x (µm relative)')
    ax.set_ylabel('y (µm relative)')
    ax.set_aspect('equal')

    title = ax.set_title(f'Foyer à {profondeur*1e6:.0f} µm | t = {t0:.4f} s\nProfondeur absolue observée = {(profondeur + z0_m)*1e6:.1f} µm')

    # ─── 5. CONSTRUCTION ET ACCROCHAGE DES INTERFACES WIDGETS ─────────────────
    s_t_grossier = Slider(plt.axes([0.2, 0.18, 0.5, 0.03]), 't Global (s)', 0.0, 250.0, valinit=t0_grossier, valstep=1.0)
    s_dt_fin = Slider(plt.axes([0.2, 0.11, 0.5, 0.03]), 'dt Fin (ms)', -1.0, 50.0, valinit=1.0)
    z_max_um = 40.0
    s_z = Slider(plt.axes([0.2, 0.04, 0.5, 0.03]), 'z / foyer (µm)', -z_max_um, z_max_um, valinit=0.0)
    radio_lambda = RadioButtons(plt.axes([0.77, 0.45, 0.21, 0.15]), list(absorption_dict.keys()), active=1)

    fig.active_slider = s_dt_fin
    fig.s_t_grossier, fig.s_dt_fin, fig.s_z, fig.radio_lambda = s_t_grossier, s_dt_fin, s_z, radio_lambda

    def update_logic():
        nonlocal current_mu_a
        t = s_t_grossier.val + s_dt_fin.val * 1e-3
        z_m = s_z.val * 1e-6

        data = T_pulse_multi_fs(X, Y, t, z=z_m, mu_a=current_mu_a)
        im.set_data(data)
        im.set_clim(37.0, max(37.1, data.max()))

        status = "Relaxation passive"
        for i, t_start in enumerate(t_pulses):
            if t_start <= t <= t_start + tau_macro:
                status = f"Train fs #{i+1} ACTIF"
                break
            elif t_start - 1e-3 <= t < t_start:
                status = "Avant l'impulsion"
                break

        title.set_text(
            f'Foyer à {profondeur*1e6:.0f} µm | t = {t:.4f} s ({status})\n'
            f'Max T = {data.max():.2f} °C | Profondeur absolue = {(profondeur + z_m)*1e6:.1f} µm'
        )
        fig.canvas.draw_idle()

    s_t_grossier.on_changed(lambda val: setattr(fig, 'active_slider', s_t_grossier) or update_logic())
    s_dt_fin.on_changed(lambda val: setattr(fig, 'active_slider', s_dt_fin) or update_logic())
    s_z.on_changed(lambda val: setattr(fig, 'active_slider', s_z) or update_logic())
    
    def on_lambda_clicked(label):
        nonlocal current_mu_a
        current_mu_a = absorption_dict[label]
        update_logic()
    radio_lambda.on_clicked(on_lambda_clicked)

    # ─── 6. CAPTURE ET NAVIGATION CLAVIER INTERACTIF ─────────────────────────
    last_key_time = 0.0

    def on_key(event):
        nonlocal last_key_time
        if event.key not in ['left', 'right'] or fig.active_slider is None:
            return

        now = time.time()
        time_diff = now - last_key_time
        last_key_time = now

        if fig.active_slider == s_t_grossier:
            base_step, val_min, val_max = 1.0, 0.0, 250.0
        elif fig.active_slider == s_dt_fin:
            base_step, val_min, val_max = 0.2, -1.0, 50.0
        elif fig.active_slider == s_z:
            base_step, val_min, val_max = 0.5, -z_max_um, z_max_um

        step = base_step * 4.0 if time_diff < 0.15 else base_step
        new_val = fig.active_slider.val + (step if event.key == 'right' else -step)
        fig.active_slider.set_val(np.clip(new_val, val_min, val_max))

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def cartographie_correlation_automatique(Day, Run_number, max_lag, event_threshold, pas_sous_echantillonnage, num_ref, taille, ref_x=None, ref_y=None):
    """Compute and display event-based cross-correlation maps and hierarchically 
    clustered coincidence matrices relative to a selected reference coordinate.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        max_lag (int): maximum number of frames allowed for time-lagged cross-correlation
        event_threshold (float): standard deviation threshold used to filter significant derivative events
        pas_sous_echantillonnage (int): downsampling factor used during spatial grid generation
        num_ref (int): reference index among the excited points to center the target coordinate square
        taille (int): width/height of the square grid area in pixels
        ref_x (int, optional): custom x-coordinate for the reference point (default: None, uses first CSV point)
        ref_y (int, optional): custom y-coordinate for the reference point (default: None, uses first CSV point)

    Returns:
        None: opens an interactive figure with a 2D spatial correlation map and a hover-responsive clustered matrix
    """
    # Attention ! Ajouter un %matplotlib widget avant d'appeler la fonction
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.cluster import hierarchy
    
    # 1. Chargement des données (Adapté au sous-dossier dynamique)
    base_dir = f"/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/"
    df_meta = pd.read_csv(os.path.join(base_dir, f"Run{Run_number}.csv"))
    
    base_data_pts_dir = os.path.join(base_dir, "Data_ts_les_pts/")
    nom_sous_dossier = f"zone_carto_{pas_sous_echantillonnage}_{num_ref}_{taille}"
    csv_path = os.path.join(base_data_pts_dir, nom_sous_dossier, "data_pt_par_pt.csv")
    
    df_data = pd.read_csv(csv_path)

    if "Frame" in df_data.columns:
        df_data = df_data.drop(columns=["Frame"])

    # 2. Soustraction du bruit de fond
    df_data = df_data.sub(df_data.iloc[0], axis=1).iloc[1:]

    # 3. Calcul de la dérivée et filtrage
    dX = np.diff(df_data.values, axis=0)
    dX_scaled = dX / (np.std(dX, axis=0) + 1e-6)
    dX_event = np.where(np.abs(dX_scaled) > event_threshold, dX_scaled, 0.0)

    # 4. Extraction des coordonnées
    coords = [tuple(map(int, col.strip("()").split(","))) for col in df_data.columns]
    df_coords = pd.DataFrame(coords, columns=["x", "y"])

    if ref_x is not None and ref_y is not None:
        cx, cy = int(ref_x), int(ref_y)
    else:
        cx = int(round(1024 - df_meta["x_um"].iloc[0] * 22.5 / 13))
        cy = int(round(1024 - df_meta["y_um"].iloc[0] * 22.5 / 13))

    idx_ref = np.argmin((df_coords["x"] - cx) ** 2 + (df_coords["y"] - cy) ** 2)
    cx, cy = df_coords.loc[idx_ref, ["x", "y"]]

    # 5. Calcul de la corrélation événementielle
    n_pts = dX_event.shape[1]
    v_ref = dX_event[:, idx_ref]
    scores_ref = np.zeros(n_pts)

    # Échantillonnage de la matrice inter-points
    max_cluster = 500
    idx_sub = np.arange(0, n_pts, int(np.ceil(n_pts / max_cluster))) if n_pts > max_cluster else np.arange(n_pts)
    X_sub = dX_event[:, idx_sub]
    n_sub = len(idx_sub)
    C_sub = np.zeros((n_sub, n_sub))

    for lag in range(-max_lag, max_lag + 1):
        s1, s2 = (slice(-lag, None), slice(None, lag)) if lag < 0 else ((slice(None, -lag), slice(lag, None)) if lag > 0 else (slice(None), slice(None)))
        
        # Corrélation Point Cible
        v_lag, X_lag = v_ref[s1], dX_event[s2, :]
        corr_ref = np.dot(v_lag, X_lag) / (np.linalg.norm(v_lag) * np.linalg.norm(X_lag, axis=0) + 1e-6)
        scores_ref = np.where(np.abs(corr_ref) > np.abs(scores_ref), corr_ref, scores_ref)

        # Corrélation Matrice Inter-points
        X1, X2 = X_sub[s1, :], X_sub[s2, :]
        corr_mat = np.dot(X1.T, X2) / (np.outer(np.linalg.norm(X1, axis=0), np.linalg.norm(X2, axis=0)) + 1e-6)
        C_sub = np.where(np.abs(corr_mat) > np.abs(C_sub), corr_mat, C_sub)

    # 6. Clustering hiérarchique
    Z = hierarchy.linkage(1 - np.clip(C_sub, -1, 1), method="ward")
    order = hierarchy.leaves_list(Z)
    C_ordered = C_sub[order, :][:, order]
    coords_ordered = df_coords.iloc[idx_sub[order]][["x", "y"]].values

    # 7. Reconstruction de la matrice 2D spatiale
    df_coords["correlation"] = scores_ref
    matrice_2d = df_coords.pivot(index="y", columns="x", values="correlation").values

    # 8. Affichage et widget interactif
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    im1 = ax1.imshow(np.flipud(matrice_2d), cmap="bwr", vmin=-1, vmax=1, extent=[df_coords["x"].min(), df_coords["x"].max(), df_coords["y"].min(), df_coords["y"].max()])
    ax1.plot(cx, cy, "g+", markersize=12, markeredgewidth=2)
    ax1.set_title(f"Carte d'Activation ($\sigma$ = {event_threshold})")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(C_ordered, cmap="bwr", vmin=-1, vmax=1)
    ax2.set_title("Coïncidences Triées")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    annot = ax2.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="black", alpha=0.8), color="white")
    annot.set_visible(False)

    def on_hover(event):
        if event.inaxes == ax2 and event.xdata is not None and event.ydata is not None:
            j = int(np.clip(round(event.xdata), 0, len(order) - 1))
            i = int(np.clip(round(event.ydata), 0, len(order) - 1))

            xA, yA = coords_ordered[i]
            xB, yB = coords_ordered[j]
            
            annot.set_text(f"A: ({xA}, {yA})\nB: ({xB}, {yB})\nCorr: {C_ordered[i, j]:.3f}")
            annot.xy = (event.xdata, event.ydata)
            annot.set_ha("right" if j > len(order) * 0.6 else "left")
            annot.set_position((-15, 15) if j > len(order) * 0.6 else (15, 15))
            annot.set_visible(True)
        else:
            if annot.get_visible():
                annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.tight_layout()
    plt.show()

def propagation_activation_complete(Day, Run_number, pas_sous_echantillonnage, num_ref,taille):
    """Track and analyze multicellular calcium waves across the organoid using chronologically 
    constrained Dijkstra propagation and interactive 2D spatial visualization.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        pas_sous_echantillonnage (int): downsampling factor used during spatial grid generation

    Returns:
        None: launches an interactive ipywidgets dashboard with wave propagation tracking, 
              directional filters, and local/global velocity calculation graphs
    """
    # Attention ! Ajouter un %matplotlib widget avant d'appeler la fonction
    rayon_initiateur = 10
    rayon_propagation = 10
    fenetre_temporelle = 20
    seuil_derivee = 0.02
    taille_min_vague = 30
    etendue_spatiale_min = 30
    tolerance_fusion = 0
    distance_contact_fusion = 0
    seuil_points_vitesse = 100
    time_per_frame = 0.05
    tolerance_centrifuge = 0.3
    seuil_r2_min = 0.15

    import os
    import heapq
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    import pandas as pd
    from scipy.spatial import cKDTree
    import ipywidgets as widgets
    from IPython.display import display

    # ── Chargement des données ────────────────────────────────────────────────────
    base_data_pts_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/Data_ts_les_pts/'
    nom_sous_dossier = f'zone_carto_{pas_sous_echantillonnage}_{num_ref}_{taille}'
    data_dir = os.path.join(base_data_pts_dir, nom_sous_dossier)
    csv_path = os.path.join(data_dir, 'data_pt_par_pt.csv')

    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)

    um_per_pixel_original = 22.5 / 13
    um_per_pixel = um_per_pixel_original * pas_sous_echantillonnage

    coord_cols = [col for col in df.columns if '(' in col and ')' in col]
    coords     = np.array([list(map(int, col.strip('()').split(','))) for col in coord_cols], dtype=float)
    signaux    = np.array([df[col].values for col in coord_cols])
    n_pts, n_frames = signaux.shape

    # Normalisation dF/F0
    F0      = np.percentile(signaux, 10, axis=1, keepdims=True)
    F0      = np.where(F0 == 0, 1, F0)
    signaux = (signaux - F0) / F0

    # ── 1. DÉTECTION DES ACTIVATIONS ─────────────────────────────────────────────
    derivee = np.diff(signaux, axis=1)
    pt_indices_list, frame_indices_list = [], []
    periode_refractaire = fenetre_temporelle

    for i in range(n_pts):
        idx_au_dessus = np.where(derivee[i] > seuil_derivee)[0]
        if len(idx_au_dessus) == 0:
            continue
        dernier_f = -periode_refractaire - 1
        for f in idx_au_dessus:
            if f >= dernier_f + periode_refractaire:
                pt_indices_list.append(i)
                frame_indices_list.append(f)
                dernier_f = f

    pt_indices    = np.array(pt_indices_list)
    frame_indices = np.array(frame_indices_list)
    n_events      = len(pt_indices)

    if n_events == 0:
        print("[INFO] Aucune activation détectée.")
        return

    event_coords = coords[pt_indices]
    event_frames = frame_indices
    tree         = cKDTree(event_coords)

    # ── 2. IDENTIFICATION DES INITIATEURS ────────────────────────────────────────
    initiateurs = set()
    for e_idx in range(n_events):
        t       = event_frames[e_idx]
        pt      = pt_indices[e_idx]
        voisins = tree.query_ball_point(event_coords[e_idx], rayon_initiateur)
        est_init = True
        for ve in voisins:
            if ve == e_idx or pt_indices[ve] == pt:
                continue
            if (t - fenetre_temporelle) <= event_frames[ve] < t:
                est_init = False
                break
        if est_init:
            initiateurs.add(e_idx)

    # ── 3. PROPAGATION PAR DIJKSTRA CHRONOLOGIQUE ────────────────────────────────
    vagues   = []
    assignes = set()

    for init in sorted(initiateurs, key=lambda e: event_frames[e]):
        if init in assignes:
            continue

        groupe       = {init}
        parent_map   = {}
        coord_centre = event_coords[init]

        num_secteurs = 36
        max_distance_par_secteur = np.zeros(num_secteurs)
        pts_physiques_enfants = set()

        heap = [(event_frames[init], init)]

        while heap:
            t_cur, cur = heapq.heappop(heap)
            if event_frames[cur] != t_cur:
                continue

            pt_cur  = pt_indices[cur]
            voisins = tree.query_ball_point(event_coords[cur], rayon_propagation)

            for ve in voisins:
                if ve in groupe or ve in assignes or pt_indices[ve] == pt_cur or pt_indices[ve] in pts_physiques_enfants:
                    continue

                t_v = event_frames[ve]
                dt  = t_v - t_cur

                if 0 < dt <= fenetre_temporelle:
                    dist_ve_centre = np.linalg.norm(event_coords[ve] - coord_centre)
                    dx = event_coords[ve, 0] - coord_centre[0]
                    dy = event_coords[ve, 1] - coord_centre[1]
                    angle = np.arctan2(dy, dx)
                    if angle < 0:
                        angle += 2 * np.pi
                    secteur_enfant = int(angle // (2 * np.pi / num_secteurs)) % num_secteurs
                    if dist_ve_centre < max_distance_par_secteur[secteur_enfant]:
                        continue

                    meilleur_parent  = None
                    max_frame_parent = -1

                    for candidat_parent in groupe:
                        dist_liaison = np.linalg.norm(event_coords[ve] - event_coords[candidat_parent])
                        if dist_liaison > rayon_propagation:
                            continue
                        dist_parent_centre = np.linalg.norm(event_coords[candidat_parent] - coord_centre)
                        if dist_parent_centre >= dist_ve_centre:
                            continue
                        t_candidat = event_frames[candidat_parent]
                        if t_candidat < t_v and t_candidat > max_frame_parent:
                            max_frame_parent = t_candidat
                            meilleur_parent  = candidat_parent

                    if meilleur_parent is not None:
                        groupe.add(ve)
                        pts_physiques_enfants.add(pt_indices[ve])
                        max_distance_par_secteur[secteur_enfant] = dist_ve_centre
                        parent_map[ve] = meilleur_parent
                        heapq.heappush(heap, (t_v, ve))

        assignes.update(groupe)

        if len(groupe) >= taille_min_vague:
            g_coords  = event_coords[list(groupe)]
            dx        = g_coords[:, 0].max() - g_coords[:, 0].min()
            dy        = g_coords[:, 1].max() - g_coords[:, 1].min()
            etendue_g = np.sqrt(dx**2 + dy**2)
            if etendue_g >= etendue_spatiale_min:
                vagues.append({'points': groupe, 'initiateur': init, 'parent_map': parent_map})

    # ── 3b. FUSION DES VAGUES PROCHES ────────────────────────────────────────────
    if len(vagues) > 1:
        fusion_active = True
        while fusion_active:
            fusion_active = False
            n_vagues = len(vagues)
            i = 0
            while i < n_vagues and not fusion_active:
                j = i + 1
                while j < n_vagues and not fusion_active:
                    v1, v2    = vagues[i], vagues[j]
                    idx_init1 = v1['initiateur']
                    idx_init2 = v2['initiateur']
                    f1        = event_frames[idx_init1]
                    f2        = event_frames[idx_init2]
                    if abs(f1 - f2) <= tolerance_fusion:
                        dist_init = np.linalg.norm(event_coords[idx_init1] - event_coords[idx_init2])
                        if dist_init <= distance_contact_fusion:
                            coords_v1    = event_coords[list(v1['points'])]
                            coords_v2    = event_coords[list(v2['points'])]
                            tree_v2      = cKDTree(coords_v2)
                            dists_min, _ = tree_v2.query(coords_v1, k=1)
                            if dists_min.min() <= distance_contact_fusion:
                                points_fus     = v1['points'].union(v2['points'])
                                parent_map_fus = {**v1['parent_map'], **v2['parent_map']}
                                init_fus       = idx_init1 if f1 <= f2 else idx_init2
                                vagues[i] = {'points': points_fus, 'initiateur': init_fus, 'parent_map': parent_map_fus}
                                vagues.pop(j)
                                fusion_active = True
                                break
                    j += 1
                if fusion_active:
                    break
                i += 1

    # ── 3c. CALCUL DES VITESSES ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"[ANALYSE] VITESSES DE PROPAGATION :")
    print(f"-> Facteur de conversion : {um_per_pixel:.4f} µm/pixel")
    print("="*80)

    for idx, vague in enumerate(vagues):
        total_points = len(vague['points'])
        init_idx     = vague['initiateur']
        coord_depart = event_coords[init_idx]
        frame_depart = event_frames[init_idx]

        g_coords = event_coords[list(vague['points'])]
        dx = g_coords[:, 0].max() - g_coords[:, 0].min()
        dy = g_coords[:, 1].max() - g_coords[:, 1].min()
        vague['etendue'] = np.sqrt(dx**2 + dy**2) * um_per_pixel

        parent_map        = vague['parent_map']
        vitesses_liaisons = []
        liens_vitesse     = []

        for enfant_idx, parent_idx in parent_map.items():
            pos_parent = event_coords[parent_idx]
            pos_enfant = event_coords[enfant_idx]
            distance_um = np.linalg.norm(pos_enfant - pos_parent) * um_per_pixel
            dt_frames   = event_frames[enfant_idx] - event_frames[parent_idx]
            dt_secondes = dt_frames * time_per_frame
            dist_parent_au_depart = np.linalg.norm(pos_parent - coord_depart)
            dist_enfant_au_depart = np.linalg.norm(pos_enfant - coord_depart)
            if (dt_secondes > 0
                    and dist_parent_au_depart <= dist_enfant_au_depart
                    and not np.array_equal(pos_parent, pos_enfant)):
                vitesses_liaisons.append(distance_um / dt_secondes)
                liens_vitesse.append((parent_idx, enfant_idx))

        vitesse_locale = np.mean(vitesses_liaisons) if vitesses_liaisons else 0.0
        vitesse_p25    = np.percentile(vitesses_liaisons, 25) if vitesses_liaisons else 0.0
        vitesse_p75    = np.percentile(vitesses_liaisons, 75) if vitesses_liaisons else 0.0

        e_ref     = min(vague['points'], key=lambda e: event_frames[e])
        coord_ref = event_coords[e_ref]
        frame_ref = event_frames[e_ref]

        dists_px = np.array([np.linalg.norm(event_coords[e] - coord_ref) for e in vague['points']])
        temps_s  = np.array([(event_frames[e] - frame_ref) * time_per_frame for e in vague['points']])

        masque = temps_s > 0
        if masque.sum() >= 5:
            t_m = temps_s[masque]
            d_m = dists_px[masque] * um_per_pixel
            vitesse_regression = max(np.dot(t_m, d_m) / np.dot(t_m, t_m), 0.0)
            d_pred = vitesse_regression * t_m
            ss_res = np.sum((d_m - d_pred)**2)
            ss_tot = np.sum((d_m - d_m.mean())**2)
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            vitesse_regression = 0.0
            r2                 = 0.0

        vague['vitesse_locale']     = vitesse_locale
        vague['liens_vitesse']      = liens_vitesse
        vague['vitesse_regression'] = vitesse_regression
        vague['r2_vitesse']         = r2
        vague['vitesse_moyenne']    = vitesse_locale
        vague['coord_depart']       = coord_depart
        vague['frame_depart']       = frame_depart

        frames_groupe      = np.array([event_frames[e] for e in vague['points']])
        duree_s            = (frames_groupe.max() - frames_groupe.min()) * time_per_frame
        densite_temporelle = total_points / max(duree_s, 0.1)

        flag = ""
        if densite_temporelle > 100:
            flag += "  ⚠ Densité élevée — probable sur-fusion"
        if r2 < seuil_r2_min and vitesse_regression > 0:
            flag += "  ⚠ R² faible — propagation non radiale"
        if vitesse_regression == 0.0 and masque.sum() >= 5:
            flag += "  ⚠ Activation synchrone — pas de front détectable"

        print(f"Groupe #{idx + 1} :{flag}")
        print(f"  - Total de points       : {total_points}")
        print(f"  - Étendue spatiale      : {vague['etendue']:.1f} µm")
        print(f"  - Durée                 : {duree_s:.2f} s")
        print(f"  - Départ (t₀)           : X={coord_depart[0]:.1f}, Y={coord_depart[1]:.1f} | Frame {int(frame_depart)}")
        print(f"  - Vitesse locale (mean) : {vitesse_locale:.2f} µm/s  [P25={vitesse_p25:.2f}  P75={vitesse_p75:.2f}]")
        print(f"  - Vitesse globale (reg) : {vitesse_regression:.2f} µm/s  (R²={r2:.3f})")
        print("-" * 60)

    if len(vagues) == 0:
        print("[INFO] Aucune vague retenue après filtrage.")
        return

    # ── UTILITAIRE ANGULAIRE ──────────────────────────────────────────────────────
    def angle_event_deg(e, coord_centre):
        dx = event_coords[e, 0] - coord_centre[0]
        dy = event_coords[e, 1] - coord_centre[1]
        a  = np.degrees(np.arctan2(dy, dx))
        return a % 360.0

    def dans_secteur(e, coord_centre, a_min, a_max):
        if np.linalg.norm(event_coords[e] - coord_centre) < 1e-6:
            return True
        a = angle_event_deg(e, coord_centre)
        if a_min <= a_max:
            return a_min <= a <= a_max
        else:
            return a >= a_min or a <= a_max

    # ── PRÉPARATION DES CONTENEURS ────────────────────────────────────────────────
    out_fig1 = widgets.Output()
    out_fig2 = widgets.Output()
    out_fig3 = widgets.Output()

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pts_actifs_unique = np.unique(pt_indices)

    # ── 4a. GRAPHIQUE GLOBAL ──────────────────────────────────────────────────────
    with out_fig1:
        fig_all, ax_all = plt.subplots(figsize=(7, 6))
        ax_all.set_facecolor('#0a0a0a')
        fig_all.patch.set_facecolor('#0a0a0a')
        ax_all.scatter(coords[pts_actifs_unique, 0], coords[pts_actifs_unique, 1], c='#1c1c1c', s=12, zorder=1)

        all_waves_x, all_waves_y, all_waves_num = [], [], []
        first_small_x, first_small_y = [], []
        first_large_x, first_large_y = [], []

        for idx_g, vague in enumerate(vagues):
            f_min_groupe = min(event_frames[e] for e in vague['points'])
            is_large     = len(vague['points']) > seuil_points_vitesse
            for e in vague['points']:
                all_waves_x.append(event_coords[e, 0])
                all_waves_y.append(event_coords[e, 1])
                all_waves_num.append(idx_g + 1)
                if event_frames[e] == f_min_groupe:
                    if is_large:
                        first_large_x.append(event_coords[e, 0])
                        first_large_y.append(event_coords[e, 1])
                    else:
                        first_small_x.append(event_coords[e, 0])
                        first_small_y.append(event_coords[e, 1])

        sc_all = ax_all.scatter(all_waves_x, all_waves_y, c=all_waves_num, cmap='viridis', s=25, zorder=2, alpha=0.8)
        if first_small_x:
            ax_all.scatter(first_small_x, first_small_y, c='red', s=45, zorder=5, edgecolors='white', linewidths=0.5)
        if first_large_x:
            ax_all.scatter(first_large_x, first_large_y, c='white', s=65, zorder=6, edgecolors='black', linewidths=0.7)

        cbar_all = fig_all.colorbar(sc_all, ax=ax_all, fraction=0.046, pad=0.04)
        cbar_all.set_label("Numéro de la vague", color='white')
        cbar_all.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar_all.ax.yaxis.get_ticklabels(), color='white')
        ax_all.set_xlim(x_min - 10, x_max + 10)
        ax_all.set_ylim(y_min - 10, y_max + 10)
        ax_all.invert_yaxis()
        plt.tight_layout()
        plt.show()

    # ── 4b. WIDGET INTERACTIF (FIGURE 2) ─────────────────────────────────────────
    etat_partage = {'vague_num': 1}

    with out_fig2:
        fig_wid, ax_wid = plt.subplots(figsize=(7, 6))
        fig_wid.patch.set_facecolor('#0a0a0a')
        ax_wid.set_facecolor('#0a0a0a')
        ax_wid.scatter(coords[pts_actifs_unique, 0], coords[pts_actifs_unique, 1], c='#1c1c1c', s=12, zorder=1)

        vague_init  = vagues[0]
        ev_init     = sorted(vague_init['points'], key=lambda e: event_frames[e])
        frames_init = np.array([event_frames[e] for e in ev_init])

        sc_wid = ax_wid.scatter(
            [event_coords[e, 0] for e in ev_init],
            [event_coords[e, 1] for e in ev_init],
            c=frames_init * time_per_frame, cmap='viridis', s=40, zorder=3
        )
        cbar_wid = fig_wid.colorbar(sc_wid, ax=ax_wid, fraction=0.046, pad=0.04)
        cbar_wid.set_label("Temps (s)", color='white')
        cbar_wid.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar_wid.ax.yaxis.get_ticklabels(), color='white')

        sc_first_wid = ax_wid.scatter(
            [vague_init['coord_depart'][0]], [vague_init['coord_depart'][1]],
            c='white', s=65, zorder=5, edgecolors='black'
        )
        text_box = ax_wid.text(
            0.02, 0.98, "", transform=ax_wid.transAxes,
            color='white', fontsize=9, va='top', zorder=6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#111111', alpha=0.85, edgecolor='#444444')
        )
        ax_wid.set_xlim(x_min - 10, x_max + 10)
        ax_wid.set_ylim(y_min - 10, y_max + 10)
        ax_wid.invert_yaxis()
        quiver_wid = None
        plt.tight_layout()
        plt.show()

    def rafraichir_vague_widget(vague_num):
        nonlocal quiver_wid
        etat_partage['vague_num'] = vague_num
        if quiver_wid is not None:
            quiver_wid.remove()
            quiver_wid = None

        vague    = vagues[vague_num - 1]
        ev_list  = sorted(vague['points'], key=lambda e: event_frames[e])
        frames_g = np.array([event_frames[e] for e in ev_list])

        sc_wid.set_offsets(np.column_stack((
            [event_coords[e, 0] for e in ev_list],
            [event_coords[e, 1] for e in ev_list]
        )))
        sc_wid.set_array(frames_g * time_per_frame)
        sc_wid.set_clim(frames_g.min() * time_per_frame, frames_g.max() * time_per_frame)
        sc_first_wid.set_offsets([vague['coord_depart']])
        cbar_wid.update_normal(sc_wid)

        text_box.set_text(
            f"Vague #{vague_num}\n"
            f"Initiateur : Frame {int(vague['frame_depart'])}\n"
            f"Points : {len(ev_list)}\n"
            f"Étendue : {vague['etendue']:.1f} µm\n"
            f"Vitesse locale (mean) : {vague['vitesse_locale']:.2f} µm/s\n"
        )
        ax_wid.set_title(f"Vague {vague_num}/{len(vagues)}", color='white')
        fig_wid.canvas.draw_idle()

    slider_dynamique = widgets.IntSlider(min=1, max=len(vagues), step=1, value=1, description='N° Vague :')
    out_vague_logic  = widgets.interactive_output(rafraichir_vague_widget, {'vague_num': slider_dynamique})

    # ── 5. WIDGET D'INSPECTION PAS-À-PAS (FIGURE 3) ──────────────────────────────
    cmap_enfants = cm.get_cmap('tab20')
    cache_historiques = {}

    def construire_historique(vague_cible):
        points_vague   = vague_cible['points']
        parent_map_v   = vague_cible['parent_map']
        coord_centre_v = vague_cible['coord_depart']

        f_all = [event_frames[e] for e in points_vague]
        frame_debut       = min(f_all)
        frame_fin         = max(f_all)
        toutes_les_frames = np.arange(frame_debut, frame_fin + 1)

        frames_uniques    = sorted(set(f_all))
        n_frames_vague    = len(frames_uniques)
        couleur_par_frame = {
            f: cmap_enfants(i / max(n_frames_vague - 1, 1))
            for i, f in enumerate(frames_uniques)
        }

        voisin_recent_map = {}
        for enfant in points_vague:
            t_enfant     = event_frames[enfant]
            coord_enfant = event_coords[enfant]
            meilleur_voisin  = None
            max_frame_voisin = -1
            min_dist_voisin  = float('inf')
            for p in points_vague:
                if p == enfant:
                    continue
                t_p = event_frames[p]
                if t_p < t_enfant:
                    dist = np.linalg.norm(coord_enfant - event_coords[p])
                    if dist <= rayon_propagation:
                        if t_p > max_frame_voisin:
                            max_frame_voisin = t_p
                            min_dist_voisin  = dist
                            meilleur_voisin  = p
                        elif t_p == max_frame_voisin and dist < min_dist_voisin:
                            min_dist_voisin = dist
                            meilleur_voisin = p
            if meilleur_voisin is not None:
                voisin_recent_map[enfant] = (meilleur_voisin, min_dist_voisin, max_frame_voisin)

        historique = []
        for f_actuelle in toutes_les_frames:
            nouveaux_voisins = [
                enfant for enfant in points_vague
                if event_frames[enfant] == f_actuelle and enfant in parent_map_v
            ]
            if f_actuelle == frame_debut:
                nouveaux_voisins.append(vague_cible['initiateur'])

            front_actif = [
                e for e in points_vague
                if (f_actuelle - fenetre_temporelle) <= event_frames[e] <= f_actuelle
                and event_frames[e] < f_actuelle
            ]
            if f_actuelle == frame_debut:
                front_actif = [vague_cible['initiateur']]

            deja_acceptes = [
                e for e in points_vague
                if event_frames[e] < (f_actuelle - fenetre_temporelle)
            ]
            historique.append({
                'frame'           : f_actuelle,
                'deja_acceptes'   : deja_acceptes,
                'front_actuel'    : front_actif,
                'nouveaux_voisins': nouveaux_voisins,
            })

        return historique, couleur_par_frame, coord_centre_v, voisin_recent_map

    # ── FIGURE 3 : GRILLE 2×2 ────────────────────────────────────────────────────
    with out_fig3:
        fig_ins, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig_ins.patch.set_facecolor('#0a0a0a')

        ax_ins     = axs[0, 0]
        ax_cum     = axs[0, 1]
        ax_vit_cum = axs[1, 0]
        ax_vit_moy = axs[1, 1]

        for ax in [ax_ins, ax_cum, ax_vit_cum, ax_vit_moy]:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            ax.spines[:].set_color('#444444')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        ax_ins.scatter(coords[pts_actifs_unique, 0], coords[pts_actifs_unique, 1], c='#111111', s=10, alpha=0.4)
        sc_ins_deja     = ax_ins.scatter([], [], c='#2ecc71', s=35, alpha=0.6, label='Éteints / Validés')
        sc_ins_front    = ax_ins.scatter([], [], c='#00e5ff', s=75, edgecolors='white', linewidths=1.2, label='Front actif')
        sc_ins_nouveaux = ax_ins.scatter([], [], c='#ff1744', s=75, edgecolors='yellow', linewidths=1.0, label='Activations')
        sc_ins_horszone = ax_ins.scatter([], [], c='#555555', s=20, alpha=0.35, label='Hors secteur', zorder=2)
        sc_ins_init     = ax_ins.scatter([], [], c='white', s=140, edgecolors='black', marker='*', zorder=10, label='Initiateur')
        
        ligne_secteur_min, = ax_ins.plot([], [], color='#FFD700', lw=1.5, ls='--', zorder=8, alpha=0.85)
        ligne_secteur_max, = ax_ins.plot([], [], color='#FF8C00', lw=1.5, ls='--', zorder=8, alpha=0.85)
        secteur_patch_holder = [None]

        ax_ins.set_xlim(x_min - 10, x_max + 10)
        ax_ins.set_ylim(y_min - 10, y_max + 10)
        ax_ins.invert_yaxis()
        ax_ins.legend(loc='upper right', facecolor='#111111', labelcolor='white', fontsize=7, ncol=2)

        ax_cum.set_xlabel("Temps depuis t₀ (s)", color='white')
        ax_cum.set_ylabel("Distance depuis l'initiateur (µm)", color='white')
        ax_cum.set_title("Distance–temps (secteur angulaire)", color='white', fontsize=9)

        ax_vit_cum.set_xlabel("Temps depuis t₀ (s)", color='white')
        ax_vit_cum.set_ylabel("Vitesse locale (µm/s)", color='white')
        ax_vit_cum.set_title("Nuage de vitesses (secteur angulaire)", color='white', fontsize=9)

        ax_vit_moy.set_xlabel("Temps depuis t₀ (s)", color='white')
        ax_vit_moy.set_ylabel("Vitesse moyenne (µm/s)", color='white')
        ax_vit_moy.set_title("Moyenne des vitesses par frame (secteur)", color='white', fontsize=9)

        ligne_vit_tendance, = ax_vit_moy.plot([], [], color='#ff9800', lw=2.5, ls='-', label='Tendance non-linéaire', zorder=5)
        ax_vit_moy.legend(loc='upper right', facecolor='#111111', labelcolor='white', fontsize=8)

        artistes_cum     = {}
        artistes_vit_cum = {}
        artistes_vit_moy = {}

        plt.tight_layout()
        plt.show()

    # ── CURSEURS ANGULAIRES ───────────────────────────────────────────────────────
    style_ang = {'description_width': '90px'}
    layout_ang = widgets.Layout(width='55%')

    slider_angle_min = widgets.IntSlider(
        min=0, max=359, step=1, value=0,
        description='Angle min (°) :',
        style=style_ang, layout=layout_ang,
        continuous_update=True
    )
    slider_angle_max = widgets.IntSlider(
        min=0, max=360, step=1, value=360,
        description='Angle max (°) :',
        style=style_ang, layout=layout_ang,
        continuous_update=True
    )
    label_secteur = widgets.Label(
        value="⟳ Secteur : 0° → 360° (tous les points)",
        layout=widgets.Layout(margin='0 0 0 10px')
    )
    
    label_vitesse_globale = widgets.HTML(
        value="<span style='color:#00e5ff; font-weight:bold;'>📊 Vitesse moyenne du secteur : En attente...</span>"
    )

    btn_reset_secteur = widgets.Button(
        description='↺ Reset secteur (360°)',
        button_style='warning',
        layout=widgets.Layout(width='180px', height='28px')
    )
    def _reset_secteur(_):
        slider_angle_min.value = 0
        slider_angle_max.value = 360
    btn_reset_secteur.on_click(_reset_secteur)

    def _update_label_secteur(change=None):
        a0 = slider_angle_min.value
        a1 = slider_angle_max.value
        if a0 == 0 and a1 == 360:
            label_secteur.value = "⟳ Secteur : 0° → 360° (tous les points)"
        elif a0 > a1:
            label_secteur.value = f"⟳ Secteur : {a0}° → 360° puis 0° → {a1}° (wrap-around)"
        else:
            label_secteur.value = f"⟳ Secteur : {a0}° → {a1}°"

    slider_angle_min.observe(_update_label_secteur, names='value')
    slider_angle_max.observe(_update_label_secteur, names='value')

    # ── UTILITAIRES GRAPHIQUES DU SECTEUR ────────────────────────────────────────
    def _dessiner_secteur(coord_centre, a_min, a_max):
        r = max(x_max - x_min, y_max - y_min) * 0.55

        def extremite(angle_deg):
            a_rad = np.radians(angle_deg)
            return (coord_centre[0] + r * np.cos(a_rad), coord_centre[1] + r * np.sin(a_rad))

        p_min = extremite(a_min)
        p_max = extremite(a_max % 360)

        ligne_secteur_min.set_data([coord_centre[0], p_min[0]], [coord_centre[1], p_min[1]])
        ligne_secteur_max.set_data([coord_centre[0], p_max[0]], [coord_centre[1], p_max[1]])

        if secteur_patch_holder[0] is not None:
            secteur_patch_holder[0].remove()
            secteur_patch_holder[0] = None

        theta1 = a_min
        theta2 = a_max if a_max > a_min else a_max + 360
        wedge = mpatches.Wedge(
            center=(coord_centre[0], coord_centre[1]),
            r=r,
            theta1=theta1, theta2=theta2,
            color='#FFD700', alpha=0.07, zorder=3
        )
        ax_ins.add_patch(wedge)
        secteur_patch_holder[0] = wedge

    # ── FONCTIONS DE NETTOYAGE ────────────────────────────────────────────────────
    def _vider_graphes_cumulatifs():
        for sc in artistes_cum.values(): sc.remove()
        artistes_cum.clear()
        for sc in artistes_vit_cum.values(): sc.remove()
        artistes_vit_cum.clear()
        for sc in artistes_vit_moy.values(): sc.remove()
        artistes_vit_moy.clear()
        ligne_vit_tendance.set_data([], [])
        for ax in [ax_cum, ax_vit_cum, ax_vit_moy]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    # ── FONCTION PRINCIPALE DE RAFRAÎCHISSEMENT (PAS-A-PAS) ──────────────────────
    def rafraichir_inspection(etape_idx, angle_min, angle_max):
        vague_num_cible = etat_partage['vague_num']
        if vague_num_cible not in cache_historiques:
            cache_historiques[vague_num_cible] = construire_historique(vagues[vague_num_cible - 1])
        historique, couleur_par_frame, coord_centre_v, voisin_recent_map = cache_historiques[vague_num_cible]
        
        vague_cible = vagues[vague_num_cible - 1]
        frame_depart = event_frames[vague_cible['initiateur']]
        secteur_complet = (angle_min == 0 and angle_max >= 360)
        
        etat_partage['points_vague'] = list(vague_cible['points'])
        n_etapes = len(historique)
        etape_idx_safe = min(etape_idx, n_etapes - 1)
        etape = historique[etape_idx_safe]
        fr = etape['frame']

        def filtrer(lst):
            if secteur_complet: return lst
            return [e for e in lst if dans_secteur(e, coord_centre_v, angle_min, angle_max % 360 if angle_max < 360 else 360)]

        deja_f = filtrer(etape['deja_acceptes'])
        front_f = filtrer(etape['front_actuel'])
        nouv_f = filtrer(etape['nouveaux_voisins'])

        if secteur_complet:
            hors_zone = []
        else:
            tous_presents = set(etape['deja_acceptes'] + etape['front_actuel'] + etape['nouveaux_voisins'])
            hors_zone = [e for e in tous_presents if not dans_secteur(e, coord_centre_v, angle_min, angle_max % 360 if angle_max < 360 else 360)]

        sc_ins_deja.set_offsets(event_coords[deja_f] if deja_f else np.empty((0, 2)))
        sc_ins_front.set_offsets(event_coords[front_f] if front_f else np.empty((0, 2)))
        sc_ins_nouveaux.set_offsets(event_coords[nouv_f] if nouv_f else np.empty((0, 2)))
        sc_ins_horszone.set_offsets(event_coords[hors_zone] if hors_zone else np.empty((0, 2)))
        sc_ins_init.set_offsets([coord_centre_v])

        _dessiner_secteur(coord_centre_v, angle_min, angle_max % 360 if angle_max < 360 else 0)

        secteur_label = "" if secteur_complet else f" [{angle_min}°→{angle_max}°]"
        ax_ins.set_title(
            f"Propagation Causalité — Vague #{vague_num_cible}{secteur_label} |\n"
            f"Frame : {fr} ({fr * time_per_frame:.2f}s) | Vert : {len(deja_f)} | Cyan : {len(front_f)} | Rouge : {len(nouv_f)}" + 
            (f" | Gris : {len(hors_zone)}" if hors_zone else ""), color='white', fontsize=9
        )

        clef_etat = (vague_num_cible, angle_min, angle_max)
        if getattr(rafraichir_inspection, '_derniere_clef', None) != clef_etat:
            _vider_graphes_cumulatifs()
            rafraichir_inspection._derniere_clef = clef_etat

        for f_futur in list(artistes_cum.keys()):
            if f_futur > fr:
                artistes_cum[f_futur].remove(); del artistes_cum[f_futur]
                if f_futur in artistes_vit_cum: artistes_vit_cum[f_futur].remove(); del artistes_vit_cum[f_futur]
                if f_futur in artistes_vit_moy: artistes_vit_moy[f_futur].remove(); del artistes_vit_moy[f_futur]

        frames_deja = set(artistes_cum.keys())

        for h in historique[:etape_idx_safe + 1]:
            f_ajout = h['frame']
            if f_ajout in frames_deja:
                continue

            pts_bruts = h['nouveaux_voisins']
            pts_sect = pts_bruts if secteur_complet else [e for e in pts_bruts if dans_secteur(e, coord_centre_v, angle_min, angle_max % 360 if angle_max < 360 else 360)]

            if not pts_sect:
                artistes_cum[f_ajout] = ax_cum.scatter([], [], s=0)
                artistes_vit_cum[f_ajout] = ax_vit_cum.scatter([], [], s=0)
                artistes_vit_moy[f_ajout] = ax_vit_moy.scatter([], [], s=0)
                continue

            t_relatif = (f_ajout - frame_depart) * time_per_frame
            couleur_rgba = '#00e5ff'

            dists = np.array([np.linalg.norm(event_coords[e] - coord_centre_v) * um_per_pixel for e in pts_sect])
            sc_d = ax_cum.scatter(np.full(len(dists), t_relatif), dists, color=[couleur_rgba] * len(dists), s=30, alpha=0.85, zorder=3, linewidths=0)
            artistes_cum[f_ajout] = sc_d

            vitesses_frame = []
            for enfant in pts_sect:
                if enfant in voisin_recent_map:
                    _, dist_px, max_f_voisin = voisin_recent_map[enfant]
                    dist_um = dist_px * um_per_pixel
                    dt_s = (event_frames[enfant] - max_f_voisin) * time_per_frame
                    if dt_s > 0 and dist_um > 0:
                        vitesses_frame.append(dist_um / dt_s)

            if vitesses_frame:
                v_arr = np.array(vitesses_frame)
                sc_v = ax_vit_cum.scatter(np.full(len(v_arr), t_relatif), v_arr, color=[couleur_rgba] * len(v_arr), s=30, alpha=0.85, zorder=3, linewidths=0)
                artistes_vit_cum[f_ajout] = sc_v

                v_moy = np.mean(v_arr)
                sc_vm = ax_vit_moy.scatter([t_relatif], [v_moy], color=[couleur_rgba], s=55, edgecolors='white', linewidths=0.6, zorder=4)
                artistes_vit_moy[f_ajout] = sc_vm
            else:
                artistes_vit_cum[f_ajout] = ax_vit_cum.scatter([], [], s=0)
                artistes_vit_moy[f_ajout] = ax_vit_moy.scatter([], [], s=0)

        # Recalcul des limites et tendances polynomiales
        t_historiques, d_historiques, v_historiques = [], [], []
        for h in historique[:etape_idx_safe + 1]:
            f_ch = h['frame']
            if f_ch in artistes_cum:
                offsets_d = artistes_cum[f_ch].get_offsets()
                if offsets_d.size > 0:
                    t_historiques.extend(offsets_d[:, 0])
                    d_historiques.extend(offsets_d[:, 1])
            if f_ch in artistes_vit_moy:
                offsets_v = artistes_vit_moy[f_ch].get_offsets()
                if offsets_v.size > 0:
                    v_historiques.append((offsets_v[0, 0], offsets_v[0, 1]))

        if t_historiques:
            t_arr, d_arr = np.array(t_historiques), np.array(d_historiques)
            ax_cum.set_xlim(-0.1, max(t_arr.max() * 1.1, 1.0))
            ax_cum.set_ylim(-10, max(d_arr.max() * 1.1, 50.0))
        else:
            ax_cum.set_xlim(-0.1, 1.0); ax_cum.set_ylim(-10, 50.0)

        if v_historiques:
            v_hist_arr = np.array(v_historiques)
            t_v, v_m = v_hist_arr[:, 0], v_hist_arr[:, 1]
            ax_vit_cum.set_xlim(-0.1, max(t_v.max() * 1.1, 1.0))
            ax_vit_moy.set_xlim(-0.1, max(t_v.max() * 1.1, 1.0))
            
            # Recoller toutes les valeurs individuelles pour borner l'axe des ordonnées du nuage
            t_toutes_vit, v_toutes_vit = [], []
            for h in historique[:etape_idx_safe + 1]:
                f_ch = h['frame']
                if f_ch in artistes_vit_cum:
                    off_v = artistes_vit_cum[f_ch].get_offsets()
                    if off_v.size > 0:
                        t_toutes_vit.extend(off_v[:, 0])
                        v_toutes_vit.extend(off_v[:, 1])
            
            if v_toutes_vit:
                v_toutes_arr = np.array(v_toutes_vit)
                v_max_lim = max(v_toutes_arr.max() * 1.1, 20.0)
                ax_vit_cum.set_ylim(-2, v_max_lim)
                ax_vit_moy.set_ylim(-2, max(v_m.max() * 1.1, 20.0))
            else:
                ax_vit_cum.set_ylim(-2, 20.0); ax_vit_moy.set_ylim(-2, 20.0)

            # Fit polynomial de tendance (Degré 2 ou 1 selon les points disponibles)
            if len(t_v) >= 3:
                deg = 2 if len(t_v) > 5 else 1
                poly_coefs = np.polyfit(t_v, v_m, deg)
                t_reg = np.linspace(t_v.min(), t_v.max(), 100)
                v_reg = np.polyval(poly_coefs, t_reg)
                ligne_vit_tendance.set_data(t_reg, v_reg)
            else:
                ligne_vit_tendance.set_data([], [])

            v_sec_globale = np.mean(v_m)
            label_vitesse_globale.value = f"<span style='color:#00e5ff;font-weight:bold;'>📊 Vitesse moyenne du secteur : {v_sec_globale:.2f} µm/s</span>"
        else:
            ax_vit_cum.set_xlim(-0.1, 1.0); ax_vit_cum.set_ylim(-2, 20.0)
            ax_vit_moy.set_xlim(-0.1, 1.0); ax_vit_moy.set_ylim(-2, 20.0)
            ligne_vit_tendance.set_data([], [])
            label_vitesse_globale.value = "<span style='color:#ff1744;font-weight:bold;'>📊 Vitesse moyenne du secteur : Aucun point dans ce secteur</span>"

        fig_ins.canvas.draw_idle()

    # Définition dynamique des étapes maximales pour le curseur pas-à-pas
    max_etapes_possible = max([len(construire_historique(v)[0]) for v in vagues])
    slider_ins = widgets.IntSlider(
        min=0, max=max_etapes_possible - 1, step=1, value=0,
        description='Étape :', style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')
    )
    label_vague_ins = widgets.Label(value="↑ Inspection pas-à-pas de la vague sélectionnée dans le widget N° Vague")

    out_ins_logic = widgets.interactive_output(
        rafraichir_inspection, {'etape_idx': slider_ins, 'angle_min': slider_angle_min, 'angle_max': slider_angle_max}
    )

    # ── ASSEMBLAGE DU DASHBOARD FINAL ────────────────────────────────────────────
    bloc_angulaire = widgets.VBox([
        widgets.HTML("<span style='color:#FFD700;font-weight:bold;'>🔶 Filtre angulaire (relatif à l'initiateur)</span>"),
        widgets.HBox([slider_angle_min, slider_angle_max]),
        widgets.HBox([label_secteur, btn_reset_secteur]),
        widgets.Box([label_vitesse_globale], layout=widgets.Layout(margin='8px 0px 4px 0px'))
    ], layout=widgets.Layout(border='1px solid #FFD700', padding='8px', margin='4px 0px', border_radius='4px'))

    top_row    = widgets.HBox([out_fig1, out_fig2])
    bottom_row = widgets.VBox([
        widgets.HTML("<hr style='border-color:#444444;'/><h3>🔍 Mode Pas-à-Pas & Profils Sectoriels de Cusalité</h3>"),
        widgets.HBox([slider_ins, label_vague_ins]),
        bloc_angulaire,
        out_fig3
    ])

    dashboard = widgets.VBox([slider_dynamique, out_vague_logic, top_row, bottom_row, out_ins_logic])
    display(dashboard)

def visualisation_grille_temporelle_interactive(Day, Run_number, pas_sous_echantillonnage, num_ref, taille):
    """Load and process spatial grid .npy files from the dynamic subfolder to 
    launch an ultra-fluid interactive 2D animation dashboard of the calcium activity over time.

    Args:
        Day (str): date or identifier of the experimental day (folder name)
        Run_number (int or str): number of the specific experimental run
        pas_sous_echantillonnage (int): downsampling factor used during spatial grid generation
        num_ref (int): reference index among the excited points to center the target coordinate square
        taille (int): width/height of the square grid area in pixels

    Returns:
        None: launches an interactive ipywidgets player directly within the Jupyter Notebook
    """
    # Attention ! Ajouter un %matplotlib widget avant d'appeler la fonction
    import os
    import glob
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import ipywidgets as widgets

    # ─── 1. CONFIGURATION DES CHEMINS ET SOUS-DOSSIER DYNAMIQUE ──────────────
    base_dir = f'/home/lumin/Documents/Data/Images_organoides/{Day}/Run{Run_number}/'
    csv_path = os.path.join(base_dir, f'Run{Run_number}.csv')
    
    base_data_pts_dir = os.path.join(base_dir, 'Data_ts_les_pts/')
    nom_sous_dossier = f'zone_carto_{pas_sous_echantillonnage}_{num_ref}_{taille}'
    data_dir = os.path.join(base_data_pts_dir, nom_sous_dossier)

    # ─── 2. CHARGEMENT ET CALCUL DU POINT CIBLE RÉFÉRENCE ────────────────────
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier CSV de métadonnées est introuvable : {csv_path}")

    df_meta = pd.read_csv(csv_path)

    # Calcul du pixel cible basé sur le num_ref fourni en argument
    first_x_pixel = int(round(1024 - df_meta['x_um'].iloc[num_ref] * 22.5 / 13))
    first_y_pixel = int(round(1024 - df_meta['y_um'].iloc[num_ref] * 22.5 / 13))

    print(f"[INFO] Point cible ref_idx {num_ref} calculé : ({first_x_pixel}, {first_y_pixel})")

    # ─── 3. CHARGEMENT DES DONNÉES ET RECONSTRUCTION DE LA MATRICE 3D ────────
    fichiers_npy = glob.glob(os.path.join(data_dir, "(*,*).npy"))
    if not fichiers_npy:
        raise FileNotFoundError(f"Aucun fichier .npy trouvé dans le dossier : {data_dir}")

    coordonnees = []
    signaux = []
    for filepath in fichiers_npy:
        nom_fichier = os.path.basename(filepath)
        match = re.match(r"\((\d+),(\d+)\)\.npy", nom_fichier)
        if match:
            coordonnees.append((int(match.group(1)), int(match.group(2))))
            signaux.append(np.load(filepath))

    coordonnees = np.array(coordonnees)
    num_frames = len(signaux[0])

    unique_x = np.sort(np.unique(coordonnees[:, 0]))
    unique_y = np.sort(np.unique(coordonnees[:, 1]))

    x_to_idx = {x_val: idx for idx, x_val in enumerate(unique_x)}
    y_to_idx = {y_val: idx for idx, y_val in enumerate(unique_y)}

    grille_temporelle = np.full((len(unique_y), len(unique_x), num_frames), np.nan)
    for (x, y), sig in zip(coordonnees, signaux):
        grille_temporelle[y_to_idx[y], x_to_idx[x], :] = sig

    vmin_global = np.nanmin(grille_temporelle)
    vmax_global = np.nanmax(grille_temporelle)

    # ─── 4. INITIALISATION DE LA FIGURE COMPACTE MATPLOTLIB ──────────────────
    plt.close('all')  # Nettoyage initial des anciennes figures

    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [unique_x.min(), unique_x.max(), unique_y.max(), unique_y.min()]

    # Affichage de la première frame par défaut
    im = ax.imshow(grille_temporelle[:, :, 0], cmap='viridis', 
                   vmin=vmin_global, vmax=vmax_global, extent=extent, aspect='equal')

    # Marqueur fixe de la cible de centrage
    ax.plot(first_x_pixel, first_y_pixel, marker='+', color='red', markersize=10, 
            markeredgewidth=1.5, label=f"Cible ref_{num_ref} ({first_x_pixel}, {first_y_pixel})")

    fig.colorbar(im, ax=ax, label="Intensité")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(False)
    ax.set_title(f"Run {Run_number} - Frame 0 (0.00 s)")

    # ─── 5. LOGIQUE DE RAFRAÎCHISSEMENT ULTRA-RAPIDE ─────────────────────────
    def rafraichir_image(frame_idx):
        # Mise à jour pure de la matrice de données (Zéro reconstruction graphique)
        im.set_data(grille_temporelle[:, :, frame_idx])
        
        # Mise à jour du titre dynamique
        ax.set_title(f"Run {Run_number} - Frame {frame_idx} ({frame_idx * 0.05:.2f} s)")
        
        # Commande de rendu asynchrone optimisé pour l'interactivité
        fig.canvas.draw_idle()

    # ─── 6. CURSEUR INTERACTIF ET LIAISON WIDGET ─────────────────────────────
    slider_dynamique = widgets.IntSlider(
        min=0, 
        max=num_frames - 1, 
        step=1, 
        value=0, 
        description='Frame :', 
        continuous_update=True  # Réactivité instantanée lors de la glisse
    )

    widgets.interact(rafraichir_image, frame_idx=slider_dynamique)