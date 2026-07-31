"""
-----------------------------------------------------------------
 grille_2d.py
 OBJECTIF : 
 Recherche automatique des hyperparamètres optimaux (eps_y, eps_x)
 de la cascade DBSCAN pour segmenter un texte manuscrit en mots.
 La recherche se fait par grille 2D en deux passes successives :
   - Passe 1 : grille grossière sur tout l'espace (eps_y, eps_x)
   - Passe 2 : zoom fin autour du meilleur point trouvé en passe 1

 PRINCIPE DE LA DBSCAN
 ────────────────────────────────────────────────────────────────
   1. DBSCAN sur Y seul  → sépare les lignes d'écriture
   2. Pour chaque ligne : DBSCAN sur X → sépare les mots
   3. Les labels de mots sont décalés globalement (mot_global)
      pour garantir leur unicité sur tout le document
   4. Qualité évaluée par l'ARI (Adjusted Rand Index) entre
      les MotID prédits et les labels BM (vérité terrain)

 UTILISATION
   python3 grille_2d.py <fichier_json> <minS_y> <minS_x>
   Ex : python3 grille_2d.py ../json_bm/203_with_BM.json 2 3

 ENTRÉES
    - fichier_json : chemin vers le fichier JSON du scripteur
    - minS_y : min_samples du DBSCAN sur Y (détection lignes)
    - minS_x : min_samples du DBSCAN sur X (détection mots)

 SORTIES
   outputs/scripteur_{N}/vis_BM_{N}.png - vérité terrain
   outputs/scripteur_{N}/vis_motID_{N}.png -mots détectés
   outputs/scripteur_{N}/vis_lignes_{N}.png - lignes détectées
   outputs/scripteur_{N}/ari_surface_{N}.png- ARI = f(eps_y, eps_x)
═══════════════════════════════════════════════════════════════════
"""

import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")

# ################################################################
# ÉTAPE 0 : Lecture des arguments en ligne de commande
# ################################################################
if len(sys.argv) != 4:
    print("Usage : python3 grille_2d.py <fichier_json> <minS_y> <minS_x>")
    sys.exit(1)

fichier   = sys.argv[1]
minS_y    = int(sys.argv[2])
minS_x    = int(sys.argv[3])
scripteur = os.path.basename(fichier).replace("_with_BM.json", "")
out       = f"outputs/scripteur_{scripteur}"
os.makedirs(out, exist_ok=True)

print(f"\n{'='*60}")
print(f"Scripteur : {scripteur}  |  minS_y={minS_y}  minS_x={minS_x}")
print(f"{'='*60}")

# ################################################################
# ÉTAPE 1 : Chargement et préparation des données
# ################################################################
with open(fichier) as f:
    data = json.load(f)
data_filtre = [seg for seg in data if seg["BM"] != "P"]

n_bm_reels = len(set(seg["BM"] for seg in data_filtre))
n_strokes  = len(data_filtre)

points_bruts = []
for seg in data_filtre:
    for p in seg["Points"]:
        points_bruts.append([p["X"], p["Y"], p["N"]])

coords   = np.array([[p[0], p[1]] for p in points_bruts])
coords_Y = coords[:, 1].reshape(-1, 1)
N_liste  = [p[2] for p in points_bruts]

print(f"Strokes : {n_strokes}  |  Points : {len(coords)}  |  BM réels : {n_bm_reels}")

# ################################################################
# ÉTAPE 2 : Copie profonde des données
# ################################################################
def copie(data_filtre):
    return [{"BM": s["BM"], "Points": [dict(p) for p in s["Points"]]}
            for s in data_filtre]

# ################################################################
# ÉTAPE 3 : Cascade DBSCAN
# ################################################################
def cascade(coords, coords_Y, N_liste, data_seg, eps_y, eps_x, mSy, mSx):

    labels_ligne   = DBSCAN(eps=eps_y, min_samples=mSy).fit_predict(coords_Y)
    lignes_uniques = sorted(set(labels_ligne) - {-1})

    labels_mot = np.full(len(coords), -1, dtype=int)
    mot_global = 0

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

# ################################################################
# ÉTAPE 4 : Passe 1 — Grille grossière
# ################################################################
print("\n── Passe 1 : grille grossière ──")

grille_y_1 = np.round(np.arange(0.05, 1.00, 0.05), 3)
grille_x_1 = np.round(np.arange(0.10, 3.00, 0.10), 3)

print(f"Grille : {len(grille_y_1)} × {len(grille_x_1)} = "
      f"{len(grille_y_1)*len(grille_x_1)} combinaisons")

rows_1 = []
total  = len(grille_y_1) * len(grille_x_1)
done   = 0
for ey in grille_y_1:
    for ex in grille_x_1:
        nl, nm, ari = cascade(
            coords, coords_Y, N_liste, copie(data_filtre),
            ey, ex, minS_y, minS_x)
        rows_1.append((ey, ex, nl, nm, ari))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{total} ...")

df1       = pd.DataFrame(rows_1, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])
idx_best1 = df1["ARI"].idxmax()
best1     = df1.loc[idx_best1]
print(f"\nMeilleur Passe 1 :")
print(f"  eps_y={best1.eps_y:.3f}  eps_x={best1.eps_x:.3f}  "
      f"ARI={best1.ARI:.3f}  Mots={int(best1.n_mots)}/{n_bm_reels}")

# ################################################################
# ÉTAPE 5 : Passe 2 — Zoom autour du meilleur
# ################################################################
print("\n── Passe 2 : zoom autour du meilleur ──")

ey_c = best1.eps_y
ex_c = best1.eps_x

grille_y_2 = np.round(np.arange(max(0.01, ey_c - 0.08), ey_c + 0.09, 0.01), 3)
grille_x_2 = np.round(np.arange(max(0.05, ex_c - 0.20), ex_c + 0.21, 0.05), 3)

print(f"Grille : {len(grille_y_2)} × {len(grille_x_2)} = "
      f"{len(grille_y_2)*len(grille_x_2)} combinaisons")

rows_2 = []
for ey in grille_y_2:
    for ex in grille_x_2:
        nl, nm, ari = cascade(
            coords, coords_Y, N_liste, copie(data_filtre),
            ey, ex, minS_y, minS_x)
        rows_2.append((ey, ex, nl, nm, ari))

df2       = pd.DataFrame(rows_2, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])
idx_best2 = df2["ARI"].idxmax()
best2     = df2.loc[idx_best2]
print(f"\nMeilleur Passe 2 :")
print(f"  eps_y={best2.eps_y:.3f}  eps_x={best2.eps_x:.3f}  "
      f"ARI={best2.ARI:.3f}  Mots={int(best2.n_mots)}/{n_bm_reels}")

eps_y_opt    = float(best2.eps_y)
eps_x_opt    = float(best2.eps_x)
ari_opt      = float(best2.ARI)
n_mots_opt   = int(best2.n_mots)
n_lignes_opt = int(best2.n_lignes)
methode_opt  = "cascade_X"
N_norm_opt   = None

# ################################################################
# ÉTAPE 6 : Résumé terminal
# ################################################################
print(f"\n{'='*60}")
print(f"RÉSUMÉ — Scripteur {scripteur}")
print(f"{'='*60}")
print(f"eps_y optimal : {eps_y_opt:.3f}")
print(f"eps_x optimal : {eps_x_opt:.3f}")
print(f"n_lignes      : {n_lignes_opt}")
print(f"n_mots        : {n_mots_opt} / {n_bm_reels}")
print(f"ARI           : {ari_opt:.3f}")

# ################################################################
# ÉTAPE 7 : ARI = f(eps_y, eps_x) — Option A (surface 3D)
#            et Option B (régression polynomiale)
# ################################################################
# On dispose déjà de toutes les valeurs ARI calculées en grille 1
# (df1) et en zoom passe 2 (df2). On les combine pour avoir
# la meilleure couverture de l'espace (eps_y, eps_x).
# Combinaison passe 1 + passe 2 pour la surface ARI
df_all = pd.concat([df1, df2], ignore_index=True)
df_all = df_all.sort_values("ARI", ascending=False)\
               .drop_duplicates(subset=["eps_y", "eps_x"])\
               .sort_values(["eps_y", "eps_x"])

print(f"\n── ARI = f(eps_y, eps_x) : {len(df_all)} points disponibles ──")

# ── Option A : Surface 3D ─────────────────────────────────────────
# On affiche ARI comme une surface dans l'espace (eps_y, eps_x, ARI).
# pivot_table reconstruit la grille reguliere pour plot_surface().
# Les trous (NaN) sont remplis par interpolation lineaire simple.
print("Option A : surface 3D ...")

try:
    pivot = df_all.pivot_table(index="eps_y", columns="eps_x", values="ARI")
    # Interpolation des NaN sur les colonnes (axis=1)
    pivot = pivot.interpolate(axis=1).interpolate(axis=0)

    EY = pivot.index.values         # valeurs eps_y (axe Y du pivot)
    EX = pivot.columns.values       # valeurs eps_x (axe X du pivot)
    EX_mesh, EY_mesh = np.meshgrid(EX, EY)
    ARI_mesh = pivot.values

    fig3d = plt.figure(figsize=(12, 7))
    ax3d  = fig3d.add_subplot(111, projection="3d")

    # plot_surface : surface coloree selon l'ARI (cmap RdYlGn)
    surf = ax3d.plot_surface(
        EX_mesh, EY_mesh, ARI_mesh,
        cmap="RdYlGn", vmin=0, vmax=1,
        alpha=0.85, edgecolor="none")

    # Marquer le point optimal (eps_y_opt, eps_x_opt, ari_opt)
    ax3d.scatter(
        [eps_x_opt], [eps_y_opt], [ari_opt],
        color="red", s=80, zorder=5,
        label=f"opt ({eps_y_opt:.3f}, {eps_x_opt:.3f}) → ARI={ari_opt:.3f}")

    plt.colorbar(surf, ax=ax3d, shrink=0.5, label="ARI")
    ax3d.set_xlabel("eps_x")
    ax3d.set_ylabel("eps_y")
    ax3d.set_zlabel("ARI")
    ax3d.set_title(
        f"ARI = f(eps_y, eps_x) — Scripteur {scripteur}\n"
        f"Surface 3D (passe 1 + zoom passe 2)")
    ax3d.legend(fontsize=9)
    fig3d.tight_layout()
    nom_3d = f"{out}/ari_surface3d_{scripteur}.png"
    fig3d.savefig(nom_3d, dpi=150)
    print(f"  Figure 3D sauvegardee : {nom_3d}")
    print(f"  (fermer la fenetre pour continuer...)")
    plt.show()  # fenetre interactive — rotation a la souris

except Exception as e:
    print(f"  Surface 3D indisponible : {e}")



# ################################################################
# ÉTAPE 8 : Visualisation — relancer avec les paramètres optimaux
# ################################################################
print(f"\n── Visualisation texte (eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}) ──")

labels_ligne_vis   = DBSCAN(eps=eps_y_opt, min_samples=minS_y).fit_predict(coords_Y)
lignes_uniques_vis = sorted(set(labels_ligne_vis) - {-1})

labels_mot_vis = np.full(len(coords), -1, dtype=int)
mot_global_vis = 0

for ligne_id in lignes_uniques_vis:
    idx        = np.where(labels_ligne_vis == ligne_id)[0]
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
print(f"Lignes : {n_lignes_vis}  |  Mots : {n_mots_vis}/{n_bm_reels}  |  ARI : {ari_vis:.3f}")

# ################################################################
# ÉTAPE 9 : Génération et sauvegarde des figures texte
# ################################################################
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
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(nom, dpi=150); plt.close(fig)
    print(f"Figure sauvegardée : {nom}")

bm_colors = gen_couleurs(sorted(df_vis["BM"].unique(), key=str), 42)
plot_texte(df_vis, "BM", bm_colors,
           f"Vérité terrain — BM réels (scripteur {scripteur})",
           f"{out}/vis_BM_{scripteur}.png")

mot_colors = gen_couleurs(sorted(df_vis["MotID"].unique(), key=str), 99)
plot_texte(df_vis, "MotID", mot_colors,
           f"Cascade DBSCAN — MotID détectés ({n_mots_vis} mots)\n"
           f"eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}  ARI={ari_vis:.3f}",
           f"{out}/vis_motID_{scripteur}.png")

ligne_colors = gen_couleurs(sorted(df_vis["LigneID"].unique(), key=str), 7)
plot_texte(df_vis, "LigneID", ligne_colors,
           f"Étape 1 — Lignes détectées ({n_lignes_vis} lignes)\n"
           f"eps_y={eps_y_opt:.3f}  minS_y={minS_y}",
           f"{out}/vis_lignes_{scripteur}.png")