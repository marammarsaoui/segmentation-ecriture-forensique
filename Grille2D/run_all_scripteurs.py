"""
run_all_scripteurs.py
─────────────────────
Lance grille_2d.py sur tous les fichiers JSON du dossier json_bm/,
en séquentiel. Les résultats de chaque scripteur sont sauvegardés
dans outputs/scripteur_{N}/ comme d'habitude.

Un CSV de synthèse global est produit à la fin :
  outputs/synthese_tous_scripteurs.csv

À la toute fin, l'ensemble du dossier outputs/ est archivé dans un
fichier zip horodaté (outputs_archive_AAAAMMJJ_HHMMSS.zip), pour garder
une trace complète du run même si outputs/ est réutilisé ou modifié
par la suite.

UTILISATION
  python3 run_all_scripteurs.py <minS_y> <minS_x> [dossier_json]
  Ex : python3 run_all_scripteurs.py 5 3
  Ex : python3 run_all_scripteurs.py 5 3 ../json_bm
"""

import sys
import os
import json
import glob
import random
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import warnings
import time

warnings.filterwarnings("ignore")


########

# Étape 0 : Arguments

if len(sys.argv) < 3:
    print("Usage : python3 run_all_scripteurs.py <minS_y> <minS_x> [dossier_json]")
    sys.exit(1)

minS_y       = int(sys.argv[1])
minS_x       = int(sys.argv[2])
dossier_json = sys.argv[3] if len(sys.argv) > 3 else "../json_bm"
ARI_SEUIL    = 0.80

fichiers = sorted(glob.glob(os.path.join(dossier_json, "*_with_BM.json")))
if not fichiers:
    print(f"Aucun fichier *_with_BM.json trouvé dans {dossier_json}")
    sys.exit(1)

os.makedirs("outputs", exist_ok=True)
print(f"\n{'='*60}")
print(f"{len(fichiers)} scripteurs trouvés dans {dossier_json}")
print(f"minS_y={minS_y}  minS_x={minS_x}  seuil switch={ARI_SEUIL}")
print(f"{'='*60}\n")


########

# Fonctions communes (identiques à grille_2d.py)

def copie(data_src):
    return [{"BM": s["BM"], "Points": [dict(p) for p in s["Points"]]}
            for s in data_src]

def cascade(coords, coords_Y, N_liste, data_seg, eps_y, eps_x, mSy, mSx):
    labels_ligne   = DBSCAN(eps=eps_y, min_samples=mSy).fit_predict(coords_Y)
    lignes_uniques = sorted(set(labels_ligne) - {-1})
    labels_mot     = np.full(len(coords), -1, dtype=int)
    mot_global     = 0
    for ligne_id in lignes_uniques:
        idx            = np.where(labels_ligne == ligne_id)[0]
        coords_X_ligne = coords[idx, 0].reshape(-1, 1)
        labels_local   = DBSCAN(eps=eps_x, min_samples=mSx).fit_predict(coords_X_ligne)
        mots_locaux    = sorted(set(labels_local) - {-1})
        for i, idx_pt in enumerate(idx):
            if labels_local[i] != -1:
                labels_mot[idx_pt] = mot_global + labels_local[i]
        if mots_locaux:
            mot_global += max(mots_locaux) + 1
    N_to_ligne = {N_liste[i]: int(labels_ligne[i]) for i in range(len(N_liste))}
    N_to_mot   = {N_liste[i]: int(labels_mot[i])   for i in range(len(N_liste))}
    for seg in data_seg:
        for p in seg["Points"]:
            p["LigneID"] = N_to_ligne.get(p["N"], -1)
            p["MotID"]   = N_to_mot.get(p["N"], -1)
    df = pd.DataFrame(data_seg)
    df["LigneID"] = df["Points"].apply(
        lambda pts: Counter(p["LigneID"] for p in pts).most_common(1)[0][0])
    df["MotID"] = df["Points"].apply(
        lambda pts: Counter(p["MotID"] for p in pts).most_common(1)[0][0])
    df_valid = df[df["MotID"] != -1]
    ari = adjusted_rand_score(
        df_valid["BM"].tolist(), df_valid["MotID"].tolist()
    ) if len(df_valid) > 1 else 0.0
    n_mots   = df["MotID"].nunique() - (1 if -1 in df["MotID"].values else 0)
    n_lignes = len(lignes_uniques)
    return n_lignes, n_mots, ari

def cascade_XN(coords, coords_Y, T_norm, N_liste, data_seg, eps_y, eps_x, mSy, mSx):
    labels_ligne   = DBSCAN(eps=eps_y, min_samples=mSy).fit_predict(coords_Y)
    lignes_uniques = sorted(set(labels_ligne) - {-1})
    labels_mot     = np.full(len(coords), -1, dtype=int)
    mot_global     = 0
    for ligne_id in lignes_uniques:
        idx       = np.where(labels_ligne == ligne_id)[0]
        coords_XT = np.column_stack([coords[idx, 0], T_norm[idx]])
        labels_local = DBSCAN(eps=eps_x, min_samples=mSx).fit_predict(coords_XT)
        mots_locaux  = sorted(set(labels_local) - {-1})
        for i, idx_pt in enumerate(idx):
            if labels_local[i] != -1:
                labels_mot[idx_pt] = mot_global + labels_local[i]
        if mots_locaux:
            mot_global += max(mots_locaux) + 1
    N_to_ligne = {N_liste[i]: int(labels_ligne[i]) for i in range(len(N_liste))}
    N_to_mot   = {N_liste[i]: int(labels_mot[i])   for i in range(len(N_liste))}
    for seg in data_seg:
        for p in seg["Points"]:
            p["LigneID"] = N_to_ligne.get(p["N"], -1)
            p["MotID"]   = N_to_mot.get(p["N"], -1)
    df = pd.DataFrame(data_seg)
    df["LigneID"] = df["Points"].apply(
        lambda pts: Counter(p["LigneID"] for p in pts).most_common(1)[0][0])
    df["MotID"] = df["Points"].apply(
        lambda pts: Counter(p["MotID"] for p in pts).most_common(1)[0][0])
    df_valid = df[df["MotID"] != -1]
    ari = adjusted_rand_score(
        df_valid["BM"].tolist(), df_valid["MotID"].tolist()
    ) if len(df_valid) > 1 else 0.0
    n_mots   = df["MotID"].nunique() - (1 if -1 in df["MotID"].values else 0)
    n_lignes = len(lignes_uniques)
    return n_lignes, n_mots, ari

def gen_couleurs(vals, seed):
    random.seed(seed)
    return {v: "#aaaaaa" if v == -1 else
            "#{:06x}".format(random.randint(0, 0xFFFFFF))
            for v in vals}

def plot_texte(df, col, colors, titre, nom):
    fig, ax = plt.subplots(figsize=(13, 7))
    for _, row in df.iterrows():
        pts = row["Points"]
        ax.plot([p["X"] for p in pts], [p["Y"] for p in pts],
                color=colors[row[col]], linewidth=2)
        ax.scatter([p["X"] for p in pts], [p["Y"] for p in pts],
                   color=colors[row[col]], s=6, alpha=0.5)
    ax.set_title(titre)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(nom, dpi=150)
    plt.close(fig)


########

# Grilles communes à tous les scripteurs

grille_y_1 = np.round(np.arange(0.05, 1.00, 0.05), 3)
grille_x_1 = np.round(np.arange(0.10, 3.00, 0.10), 3)


########

# Boucle principale sur tous les scripteurs

lignes_synthese = []

for num, fichier in enumerate(fichiers, 1):

    scripteur = os.path.basename(fichier).replace("_with_BM.json", "")
    out       = f"outputs/scripteur_{scripteur}"
    os.makedirs(out, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"[{num}/{len(fichiers)}] Scripteur {scripteur}")
    print(f"{'─'*60}")
    t_debut = time.time()

    # ── Chargement ──────────────────────────────────────────────
    with open(fichier) as f:
        data = json.load(f)
    data_filtre = [seg for seg in data if seg["BM"] != "P"]
    n_bm_reels  = len(set(seg["BM"] for seg in data_filtre))
    n_strokes   = len(data_filtre)

    points_bruts = []
    for seg in data_filtre:
        for p in seg["Points"]:
            points_bruts.append([p["X"], p["Y"], p["N"], p["Time_MS"]])

    coords   = np.array([[p[0], p[1]] for p in points_bruts])
    coords_Y = coords[:, 1].reshape(-1, 1)
    N_liste  = [p[2] for p in points_bruts]

    print(f"  Strokes={n_strokes}  Points={len(coords)}  BM={n_bm_reels}")

    # ── Passe 1 grossière ───────────────────────────────────────
    rows_1 = []
    done   = 0
    total  = len(grille_y_1) * len(grille_x_1)
    for ey in grille_y_1:
        for ex in grille_x_1:
            nl, nm, ari = cascade(
                coords, coords_Y, N_liste, copie(data_filtre),
                ey, ex, minS_y, minS_x)
            rows_1.append((ey, ex, nl, nm, ari))
            done += 1
            if done % 100 == 0:
                print(f"  passe1 : {done}/{total} ...", flush=True)

    df1   = pd.DataFrame(rows_1, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])
    best1 = df1.loc[df1["ARI"].idxmax()]

    # ── Passe 2 zoom ────────────────────────────────────────────
    ey_c = best1.eps_y
    ex_c = best1.eps_x
    grille_y_2 = np.round(np.arange(max(0.01, ey_c - 0.08), ey_c + 0.09, 0.01), 3)
    grille_x_2 = np.round(np.arange(max(0.05, ex_c - 0.20), ex_c + 0.21, 0.05), 3)

    rows_2 = []
    for ey in grille_y_2:
        for ex in grille_x_2:
            nl, nm, ari = cascade(
                coords, coords_Y, N_liste, copie(data_filtre),
                ey, ex, minS_y, minS_x)
            rows_2.append((ey, ex, nl, nm, ari))

    df2   = pd.DataFrame(rows_2, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])
    best2 = df2.loc[df2["ARI"].idxmax()]

    eps_y_opt    = float(best2.eps_y)
    eps_x_opt    = float(best2.eps_x)
    ari_opt      = float(best2.ARI)
    n_mots_opt   = int(best2.n_mots)
    n_lignes_opt = int(best2.n_lignes)
    methode_opt  = "cascade_X"
    T_norm_opt   = None

    print(f"  Spatial : eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}  ARI={ari_opt:.3f}")

    # ── Switch temporel si ARI < seuil ──────────────────────────
    if ari_opt < ARI_SEUIL:
        print(f"  ARI={ari_opt:.3f} < {ARI_SEUIL} → switch (X, Time_MS norm)")

        T_brut  = np.array([p[3] for p in points_bruts], dtype=float)
        X_range = float(coords[:, 0].max() - coords[:, 0].min())
        T_norm  = ((T_brut - T_brut.min()) /
                   (T_brut.max() - T_brut.min()) * X_range)

        rows_t1 = []
        done = 0
        for ey in grille_y_1:
            for ex in grille_x_1:
                try:
                    nl, nm, ari = cascade_XN(
                        coords, coords_Y, T_norm, N_liste,
                        copie(data_filtre), ey, ex, minS_y, minS_x)
                except Exception:
                    ari, nl, nm = 0.0, 0, 0
                rows_t1.append((ey, ex, nl, nm, ari))
                done += 1
                if done % 100 == 0:
                    print(f"  switch passe1 : {done}/{total} ...", flush=True)

        df_t1   = pd.DataFrame(rows_t1, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])
        best_t1 = df_t1.loc[df_t1["ARI"].idxmax()]

        grille_y_t2 = np.round(np.arange(
            max(0.01, best_t1.eps_y - 0.08), best_t1.eps_y + 0.09, 0.01), 3)
        grille_x_t2 = np.round(np.arange(
            max(0.05, best_t1.eps_x - 0.20), best_t1.eps_x + 0.21, 0.05), 3)
        rows_t2 = []
        for ey in grille_y_t2:
            for ex in grille_x_t2:
                try:
                    nl, nm, ari = cascade_XN(
                        coords, coords_Y, T_norm, N_liste,
                        copie(data_filtre), ey, ex, minS_y, minS_x)
                except Exception:
                    ari, nl, nm = 0.0, 0, 0
                rows_t2.append((ey, ex, nl, nm, ari))

        df_t2   = pd.DataFrame(rows_t2, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])
        best_t2 = df_t2.loc[df_t2["ARI"].idxmax()]

        if best_t2.ARI > ari_opt:
            eps_y_opt    = float(best_t2.eps_y)
            eps_x_opt    = float(best_t2.eps_x)
            ari_opt      = float(best_t2.ARI)
            n_mots_opt   = int(best_t2.n_mots)
            n_lignes_opt = int(best_t2.n_lignes)
            methode_opt  = "cascade_XN"
            T_norm_opt   = T_norm
            print(f"  Switch OK : eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}  ARI={ari_opt:.3f}")
        else:
            print(f"  Switch non retenu : X seul reste meilleur ({ari_opt:.3f})")

    # ── Surface 3D ──────────────────────────────────────────────
    df_all = pd.concat([df1, df2], ignore_index=True)\
               .sort_values("ARI", ascending=False)\
               .drop_duplicates(subset=["eps_y", "eps_x"])\
               .sort_values(["eps_y", "eps_x"])
    try:
        pivot = df_all.pivot_table(index="eps_y", columns="eps_x", values="ARI")
        pivot = pivot.interpolate(axis=1).interpolate(axis=0)
        EX_mesh, EY_mesh = np.meshgrid(pivot.columns.values, pivot.index.values)
        fig3d = plt.figure(figsize=(12, 7))
        ax3d  = fig3d.add_subplot(111, projection="3d")
        surf  = ax3d.plot_surface(EX_mesh, EY_mesh, pivot.values,
                                   cmap="RdYlGn", vmin=0, vmax=1,
                                   alpha=0.85, edgecolor="none")
        ax3d.scatter([eps_x_opt], [eps_y_opt], [ari_opt],
                     color="red", s=80, zorder=5,
                     label=f"opt ({eps_y_opt:.3f},{eps_x_opt:.3f}) ARI={ari_opt:.3f}")
        plt.colorbar(surf, ax=ax3d, shrink=0.5, label="ARI")
        ax3d.set_xlabel("eps_x"); ax3d.set_ylabel("eps_y"); ax3d.set_zlabel("ARI")
        ax3d.set_title(f"ARI = f(eps_y, eps_x) — Scripteur {scripteur}\n"
                       f"Methode : {methode_opt}")
        ax3d.legend(fontsize=9)
        fig3d.tight_layout()
        fig3d.savefig(f"{out}/ari_surface3d_{scripteur}.png", dpi=150)
        plt.close(fig3d)
    except Exception as e:
        print(f"  Surface 3D indisponible : {e}")

    # ── Visualisation texte ──────────────────────────────────────
    labels_ligne_vis   = DBSCAN(eps=eps_y_opt, min_samples=minS_y).fit_predict(coords_Y)
    lignes_uniques_vis = sorted(set(labels_ligne_vis) - {-1})
    labels_mot_vis     = np.full(len(coords), -1, dtype=int)
    mot_global_vis     = 0

    for ligne_id in lignes_uniques_vis:
        idx = np.where(labels_ligne_vis == ligne_id)[0]
        if methode_opt == "cascade_XN" and T_norm_opt is not None:
            coords_seg = np.column_stack([coords[idx, 0], T_norm_opt[idx]])
        else:
            coords_seg = coords[idx, 0].reshape(-1, 1)
        labels_local = DBSCAN(eps=eps_x_opt, min_samples=minS_x).fit_predict(coords_seg)
        mots_locaux  = sorted(set(labels_local) - {-1})
        for i, idx_pt in enumerate(idx):
            if labels_local[i] != -1:
                labels_mot_vis[idx_pt] = mot_global_vis + labels_local[i]
        if mots_locaux:
            mot_global_vis += max(mots_locaux) + 1

    N_to_ligne_vis = {N_liste[i]: int(labels_ligne_vis[i]) for i in range(len(N_liste))}
    N_to_mot_vis   = {N_liste[i]: int(labels_mot_vis[i])   for i in range(len(N_liste))}
    data_vis = copie(data_filtre)
    for seg in data_vis:
        for p in seg["Points"]:
            p["LigneID"] = N_to_ligne_vis.get(p["N"], -1)
            p["MotID"]   = N_to_mot_vis.get(p["N"], -1)

    df_vis = pd.DataFrame(data_vis)
    df_vis["LigneID"] = df_vis["Points"].apply(
        lambda pts: Counter(p["LigneID"] for p in pts).most_common(1)[0][0])
    df_vis["MotID"] = df_vis["Points"].apply(
        lambda pts: Counter(p["MotID"] for p in pts).most_common(1)[0][0])

    df_vis_valid = df_vis[df_vis["MotID"] != -1]
    ari_vis = adjusted_rand_score(
        df_vis_valid["BM"].tolist(), df_vis_valid["MotID"].tolist()
    ) if len(df_vis_valid) > 1 else 0.0
    n_mots_vis   = df_vis["MotID"].nunique() - (1 if -1 in df_vis["MotID"].values else 0)
    n_lignes_vis = len(lignes_uniques_vis)

    bm_colors = gen_couleurs(sorted(df_vis["BM"].unique(), key=str), 42)
    plot_texte(df_vis, "BM", bm_colors,
               f"Vérité terrain — BM réels (scripteur {scripteur})",
               f"{out}/vis_BM_{scripteur}.png")

    mot_colors = gen_couleurs(sorted(df_vis["MotID"].unique(), key=str), 99)
    plot_texte(df_vis, "MotID", mot_colors,
               f"Cascade DBSCAN — MotID détectés ({n_mots_vis} mots)\n"
               f"eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}  "
               f"ARI={ari_vis:.3f}  Methode={methode_opt}",
               f"{out}/vis_motID_{scripteur}.png")

    ligne_colors = gen_couleurs(sorted(df_vis["LigneID"].unique(), key=str), 7)
    plot_texte(df_vis, "LigneID", ligne_colors,
               f"Lignes détectées ({n_lignes_vis} lignes)  eps_y={eps_y_opt:.3f}",
               f"{out}/vis_lignes_{scripteur}.png")

    t_fin = time.time()
    print(f"  → ARI={ari_vis:.3f}  Mots={n_mots_vis}/{n_bm_reels}"
          f"  Methode={methode_opt}  Temps={t_fin-t_debut:.0f}s")

    # ── Ligne de synthèse ────────────────────────────────────────
    lignes_synthese.append({
        "scripteur"   : scripteur,
        "n_strokes"   : n_strokes,
        "n_bm_reels"  : n_bm_reels,
        "eps_y_opt"   : eps_y_opt,
        "eps_x_opt"   : eps_x_opt,
        "n_lignes_opt": n_lignes_opt,
        "n_mots_opt"  : n_mots_opt,
        "ARI"         : round(ari_vis, 3),
        "methode"     : methode_opt,
        "temps_s"     : round(t_fin - t_debut, 1),
    })


########

# Synthèse globale

df_synthese = pd.DataFrame(lignes_synthese)
chemin_csv  = "outputs/synthese_tous_scripteurs.csv"
df_synthese.to_csv(chemin_csv, index=False)

print(f"\n{'='*60}")
print(f"SYNTHÈSE GLOBALE — {len(lignes_synthese)} scripteurs")
print(f"{'='*60}")
print(df_synthese[["scripteur", "ARI", "methode", "n_mots_opt",
                    "n_bm_reels", "temps_s"]].to_string(index=False))
print(f"\nARI moyen  : {df_synthese['ARI'].mean():.3f}")
print(f"ARI médian : {df_synthese['ARI'].median():.3f}")
print(f"Switch utilisé : {(df_synthese['methode'] == 'cascade_XN').sum()}"
      f"/{len(df_synthese)} scripteurs")
print(f"\nCSV synthèse : {chemin_csv}")


########

# Archivage horodaté du dossier outputs/ complet
#
# À la différence du CSV de synthèse (un seul fichier, souvent écrasé
# ou complété d'un run à l'autre), cette archive fige un instantané
# complet de tout le dossier outputs/ (figures, CSV par scripteur,
# synthèse) au moment précis où ce run s'est terminé. Utile pour
# comparer plusieurs runs entre eux (par ex. avec des minS_y/minS_x
# différents) sans que l'un n'écrase les résultats de l'autre.

horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
nom_archive_sans_extension = f"outputs_archive_{horodatage}"

try:
    chemin_zip = shutil.make_archive(nom_archive_sans_extension, "zip", "outputs")
    taille_mo = os.path.getsize(chemin_zip) / (1024 * 1024)
    print(f"\nArchive complète du run : {chemin_zip} ({taille_mo:.1f} Mo)")
except Exception as e:
    print(f"\nArchivage impossible : {e}")