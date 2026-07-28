import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import warnings
import os
"""Objectif : 
trouver automatiquement les meilleurs paramètres DBSCAN (eps_y, eps_x) pour segmenter un texte manuscrit en lignes puis en mots
et évaluer la qualité avec l'ARI (Adjusted Rand Index).
Usage: python3 grille.py <fichier_json> <minS_y> <minS_x>
Exemple: python3 grille.py ../json_bm/203_with_BM.json 2 3"""
#############################################################
warnings.filterwarnings("ignore")
#############################################################
#Etape 0 : Récupération des paramètres et le nom du fichier à partir du terminal ######
if len(sys.argv) != 4:
    print("Usage : python3 grille_1d.py <fichier_json> <minS_y> <minS_x>")
    sys.exit(1)

fichier   = sys.argv[1]
minS_y    = int(sys.argv[2])
minS_x    = int(sys.argv[3])
scripteur = os.path.basename(fichier).replace("_with_BM.json", "")
out       = f"outputs/scripteur_{scripteur}"
#utile pour avoir des dossiers de résultats pour chaque scripteur ( grâce à la définition de out à la ligne d'avant)
os.makedirs(out, exist_ok=True)

print(f"\n{'-'*60}")
print(f"Scripteur : {scripteur}  |  minS_y={minS_y}  minS_x={minS_x}")
print(f"{'-'*60}")

########################################################################################""
# Chargement
"""
ÉTAPE 1:Chargement du fichier JSON 
Objectif :
- Charger les strokes
- Filtrer les BM = 'P' (ponctuation)
- Extraire les points bruts (X, Y, N)
"""
###############################

with open(fichier) as f:
    data = json.load(f)
#j'applique un filtre pour ne pas considérer le cas BM'P' des ponctuations
data_filtre = [seg for seg in data if seg["BM"] != "P"]


n_bm_reels = len(set(seg["BM"] for seg in data_filtre))
n_strokes  = len(data_filtre)

points_bruts = []
for seg in data_filtre:
    for p in seg["Points"]:
        points_bruts.append([p["X"], p["Y"], p["N"]])
#Extraire les coordonnées X et Y 
coords   = np.array([[p[0], p[1]] for p in points_bruts])
coords_Y = coords[:, 1].reshape(-1, 1)
#identifier N 
N_liste  = [p[2] for p in points_bruts]

print(f"Strokes : {n_strokes}  |  Points : {len(coords)}  |  BM reels : {n_bm_reels}")

##########################################################################################
""" Etape2: Copie profonde
Éviter toute modification involontaire lors des tests de paramètres
"""
def copie(data_filtre):
    return [{"BM": s["BM"], "Points": [dict(p) for p in s["Points"]]}
            for s in data_filtre]

##########################################################################################

# Cascade DBSCAN
#la fonction clé du code 
"""Etape3: DBSCAN
Objectif :
- Appliquer DBSCAN sur Y pour détecter les lignes
- Appliquer DBSCAN sur X pour détecter les mots dans chaque ligne
- Assigner LigneID et MotID à chaque point
- Calculer ARI, nombre de lignes, nombre de mots
"""

def cascade(coords, coords_Y, N_liste, data_seg, eps_y, eps_x, mSy, mSx):
    #on labelise les lignes pour appliquer en premier lieu DBSCAN sur Y 
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


########################################################################
"""
ÉTAPE 5:grille eps_y (DÉTECTION DES LIGNES)
Objectif :
- Tester plusieurs valeurs de eps_y
- Compter le nombre de lignes détectées
- Identifier les plateaux stables
- Sélectionner eps_y optimal
"""
print("\n -- Grille eps_y --")
grille_y = np.round(np.arange(0.05, 1.0, 0.01), 3)

resultats_y = []
for ey in grille_y:
    labels = DBSCAN(eps=ey, min_samples=minS_y).fit_predict(coords_Y)
    nl     = len(set(labels) - {-1})
    resultats_y.append((ey, nl))
df_y = pd.DataFrame(resultats_y, columns=["eps_y", "n_lignes"])

n_lignes_vals   = df_y["n_lignes"].values
eps_y_vals      = df_y["eps_y"].values
MIN_PLATEAU     = 3
plateau_courant = 1
plateaux        = []

for i in range(1, len(n_lignes_vals)):
    if n_lignes_vals[i] == n_lignes_vals[i - 1]:
        plateau_courant += 1
        if plateau_courant == MIN_PLATEAU:
            idx_debut = i - MIN_PLATEAU + 1
            plateaux.append({
                "n_lignes" : int(n_lignes_vals[i]),
                "eps_debut": float(eps_y_vals[idx_debut]),
                "eps_fin"  : float(eps_y_vals[i]),
            })
    else:
        plateau_courant = 1

for p in plateaux:
    mask        = df_y["n_lignes"] == p["n_lignes"]
    p["eps_fin"] = float(df_y[mask]["eps_y"].max())

n_lignes_min = max(2, n_bm_reels // 12)
n_lignes_max = n_bm_reels // 4
print(f"  Plage coherente : [{n_lignes_min} - {n_lignes_max}]")
print(f"  Plateaux trouves : {[p['n_lignes'] for p in plateaux]}")

plateaux_coherents = [p for p in plateaux
                      if n_lignes_min <= p["n_lignes"] <= n_lignes_max]

if plateaux_coherents:
    n_lignes_cible = int(round(np.sqrt(n_bm_reels)))
    choisi     = min(plateaux_coherents,
                     key=lambda p: abs(p["n_lignes"] - n_lignes_cible))
    plateau_val = choisi["n_lignes"]
    df_y_stable = df_y[df_y["n_lignes"] == plateau_val]
    print(f"  Cible (sqrt BM) : {n_lignes_cible}")
    print(f"  Plateau retenu  : {plateau_val} lignes")
else:
    plateau_val = int(df_y["n_lignes"].mode()[0])
    df_y_stable = df_y[df_y["n_lignes"] == plateau_val]
    print(f"  (fallback mode={plateau_val})")

eps_y_min = float(df_y_stable["eps_y"].min())
eps_y_max = float(df_y_stable["eps_y"].max())
eps_y_opt = float(df_y_stable["eps_y"].median())
print(f"eps_y : [{eps_y_min:.3f} - {eps_y_max:.3f}]  opt={eps_y_opt:.3f}")


###############################
"""
ÉTAPE 6 :grille eps_x (détéction des mots)
Objectif :
- Tester plusieurs eps_x
- Calculer ARI pour chaque valeur
- Trouver eps_x optimal
- Déterminer zones : sur-regroupement, sous-regroupement, zone optimale
"""

print("\n -- Grille eps_x --")
grille_x = np.round(np.arange(0.10, 3.0, 0.05), 3)

resultats_x = []
for ex in grille_x:
    nl, nm, ari = cascade(
        coords, coords_Y, N_liste, copie(data_filtre),
        eps_y_opt, ex, minS_y, minS_x)
    resultats_x.append((ex, nl, nm, ari, abs(nm - n_bm_reels)))

df_x = pd.DataFrame(resultats_x,
    columns=["eps_x", "n_lignes", "n_mots", "ARI", "ecart_BM"])

idx_best  = df_x["ARI"].idxmax()
eps_x_opt = float(df_x.loc[idx_best, "eps_x"])
ari_opt   = float(df_x.loc[idx_best, "ARI"])
n_mots_opt = int(df_x.loc[idx_best, "n_mots"])

seuil_bas  = n_bm_reels * 0.80
seuil_haut = n_bm_reels * 1.20
zone_sur   = df_x[df_x["n_mots"] < seuil_bas]
zone_sous  = df_x[df_x["n_mots"] > seuil_haut]
zone_ok    = df_x[(df_x["n_mots"] >= seuil_bas) & (df_x["n_mots"] <= seuil_haut)]

eps_x_sur_max  = float(zone_sur ["eps_x"].max()) if len(zone_sur)  > 0 else None
eps_x_sous_min = float(zone_sous["eps_x"].min()) if len(zone_sous) > 0 else None
eps_x_ok_min   = float(zone_ok  ["eps_x"].min()) if len(zone_ok)   > 0 else None
eps_x_ok_max   = float(zone_ok  ["eps_x"].max()) if len(zone_ok)   > 0 else None

print(f"eps_x opt : {eps_x_opt:.3f}  ARI={ari_opt:.3f}  Mots={n_mots_opt}/{n_bm_reels}")
print(f"Sur-regroupement  (mots < 80%)  : eps_x > {eps_x_sur_max}")
print(f"Sous-regroupement (mots > 120%) : eps_x < {eps_x_sous_min}")
print(f"Zone optimale     (80-120%)     : eps_x in [{eps_x_ok_min} - {eps_x_ok_max}]")

###############################
"""
ÉTAPE 7 — IMPACT DE minSamples
Objectif :
- Tester plusieurs valeurs de minSamples
- Étudier leur influence sur ARI
- Générer un tableau comparatif
"""
print("\n--- Impact minSamples ---")
minS_vals   = [2, 3, 5]
impact_rows = []
for ms in minS_vals:
    for ex in grille_x:
        nl, nm, ari = cascade(
            coords, coords_Y, N_liste, copie(data_filtre),
            eps_y_opt, ex, ms, ms)
        impact_rows.append((ms, ex, nl, nm, ari))
df_impact = pd.DataFrame(impact_rows,
    columns=["minS", "eps_x", "n_lignes", "n_mots", "ARI"])

"""
ÉTAPE 8: Sauvegarde CSV
"""

pd.DataFrame(resultats_y, columns=["eps_y","n_lignes"]).to_csv(
    f"{out}/grille_epsy_{scripteur}.csv", index=False)
df_x.to_csv(f"{out}/grille_epsx_{scripteur}.csv", index=False)
df_impact.to_csv(f"{out}/impact_minS_{scripteur}.csv", index=False)

# Tableau de synthese — valeurs cles par scripteur
synthese = {
    "scripteur"           : [scripteur],
    "n_strokes"           : [n_strokes],
    "n_bm_reels"          : [n_bm_reels],
    "minS_y"              : [minS_y],
    "minS_x"              : [minS_x],
    "methode"             : ["grille_1D"],
    # eps_y
    "eps_y_opt"           : [eps_y_opt],
    "eps_y_min"           : [eps_y_min],
    "eps_y_max"           : [eps_y_max],
    "n_lignes_plateau"    : [plateau_val],
    # eps_x
    "eps_x_opt"           : [eps_x_opt],
    "eps_x_ok_min"        : [eps_x_ok_min],
    "eps_x_ok_max"        : [eps_x_ok_max],
    "eps_x_sur_max"       : [eps_x_sur_max],
    "eps_x_sous_min"      : [eps_x_sous_min],
    # resultats
    "n_mots_opt"          : [n_mots_opt],
    "ARI"                 : [round(ari_opt, 3)],
    # zones en % de la grille
    "pct_sur_regroupement": [round(len(zone_sur)/len(df_x)*100, 1)],
    "pct_zone_optimale"   : [round(len(zone_ok) /len(df_x)*100, 1)],
    "pct_sous_regroupement":[round(len(zone_sous)/len(df_x)*100, 1)],
}
pd.DataFrame(synthese).to_csv(f"{out}/synthese_grille1d_{scripteur}.csv", index=False)
print(f"\nCSV sauvegardes dans {out}/")


# ETAPE 9 : FIGURES ANALYSE ( 3 graphiques)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(df_y["eps_y"], df_y["n_lignes"], color="#2980b9", linewidth=2)
ax.axvline(eps_y_opt, color="#c0392b", linestyle="--",
           label=f"opt={eps_y_opt:.3f}")
ax.axhspan(plateau_val - 0.4, plateau_val + 0.4,
           alpha=0.15, color="green", label=f"plateau={plateau_val} lignes")
ax.set_title("Grille eps_y -> n_lignes\n(premier plateau stable)")
ax.set_xlabel("eps_y"); ax.set_ylabel("nb lignes")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax2      = axes[1]
ax2_twin = ax2.twinx()
ax2.plot(df_x["eps_x"], df_x["ARI"], color="#8e44ad", linewidth=2, label="ARI")
ax2_twin.plot(df_x["eps_x"], df_x["n_mots"], color="#e67e22",
              linewidth=1.5, linestyle="--", label="nb mots")
ax2_twin.axhline(n_bm_reels, color="black", linestyle=":", alpha=0.5,
                 label=f"BM reels={n_bm_reels}")
ax2.axvline(eps_x_opt, color="#8e44ad", linestyle="--", alpha=0.7,
            label=f"opt={eps_x_opt:.3f}")
if eps_x_ok_min and eps_x_ok_max:
    ax2.axvspan(eps_x_ok_min, eps_x_ok_max, alpha=0.1,
                color="green", label="zone optimale")
ax2.set_title("Grille eps_x -> ARI + nb mots")
ax2.set_xlabel("eps_x"); ax2.set_ylabel("ARI", color="#8e44ad")
ax2_twin.set_ylabel("nb mots", color="#e67e22")
ax2.legend(fontsize=8, loc="lower left")
ax2_twin.legend(fontsize=8, loc="upper right")
ax2.grid(alpha=0.3)

ax3 = axes[2]
colors_ms = {2: "#2980b9", 3: "#27ae60", 5: "#c0392b"}
for ms in minS_vals:
    sub = df_impact[df_impact["minS"] == ms]
    ax3.plot(sub["eps_x"], sub["ARI"], label=f"minS={ms}",
             color=colors_ms[ms], linewidth=1.8)
ax3.set_title("Impact minSamples sur ARI")
ax3.set_xlabel("eps_x"); ax3.set_ylabel("ARI")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

plt.suptitle(
    f"Scripteur {scripteur} — Grille 1D\n"
    f"eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}  "
    f"ARI={ari_opt:.3f}  Mots={n_mots_opt}/{n_bm_reels}",
    fontsize=12)
plt.tight_layout()
plt.savefig(f"{out}/analyse_grille1d_{scripteur}.png", dpi=150)
plt.close()
print(f"Figure analyse sauvegardee")

# FIGURES TEXTE : BM reel, MotID, LigneID

# Relancer cascade avec parametres optimaux pour les figures texte
labels_ligne_vis   = DBSCAN(eps=eps_y_opt, min_samples=minS_y).fit_predict(coords_Y)
lignes_uniques_vis = sorted(set(labels_ligne_vis) - {-1})
labels_mot_vis     = np.full(len(coords), -1, dtype=int)
mot_global_vis     = 0
for ligne_id in lignes_uniques_vis:
    idx          = np.where(labels_ligne_vis == ligne_id)[0]
    coords_X     = coords[idx, 0].reshape(-1, 1)
    labels_local = DBSCAN(eps=eps_x_opt, min_samples=minS_x).fit_predict(coords_X)
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
    ax.set_xlabel("X"); ax.set_ylabel("Y"); 
    fig.tight_layout()
    fig.savefig(nom, dpi=150); plt.close(fig)

# Figure BM reel
bm_colors = gen_couleurs(sorted(df_vis["BM"].unique(), key=str), 42)
plot_texte(df_vis, "BM", bm_colors,
           f"Verite terrain — BM reels (scripteur {scripteur})",
           f"{out}/BM_{scripteur}.png")

# Figure MotID
mot_colors = gen_couleurs(sorted(df_vis["MotID"].unique(), key=str), 99)
plot_texte(df_vis, "MotID", mot_colors,
           f"Grille 1D — MotID ({n_mots_vis} mots)\n"
           f"eps_y={eps_y_opt:.3f}  eps_x={eps_x_opt:.3f}  ARI={ari_vis:.3f}",
           f"{out}/motID_grille1d_{scripteur}.png")

# Figure LigneID
ligne_colors = gen_couleurs(sorted(df_vis["LigneID"].unique(), key=str), 7)
plot_texte(df_vis, "LigneID", ligne_colors,
           f"Lignes detectees ({n_lignes_vis} lignes)\n"
           f"eps_y={eps_y_opt:.3f}  minS_y={minS_y}",
           f"{out}/lignes_grille1d_{scripteur}.png")

print(f"Figures texte sauvegardees dans {out}/")

# RESUME TERMINAL

print(f"\n{'='*60}")
print(f"RESUME — Scripteur {scripteur}  (Grille 1D)")
print(f"{'='*60}")
print(f"eps_y opt    : {eps_y_opt:.3f}  [{eps_y_min:.3f} - {eps_y_max:.3f}]")
print(f"eps_x opt    : {eps_x_opt:.3f}")
print(f"n_lignes     : {n_lignes_vis} (plateau={plateau_val})")
print(f"n_mots       : {n_mots_vis} / {n_bm_reels}")
print(f"ARI          : {ari_vis:.3f}")
print(f"Zone optimale       : eps_x in [{eps_x_ok_min} - {eps_x_ok_max}]")
print(f"Sur-regroupement    : eps_x > {eps_x_sur_max}  ({synthese['pct_sur_regroupement'][0]}% grille)")
print(f"Sous-regroupement   : eps_x < {eps_x_sous_min}  ({synthese['pct_sous_regroupement'][0]}% grille)")
print(f"Zone optimale       : {synthese['pct_zone_optimale'][0]}% de la grille")

# Sauvegarde résumé terminal dans un fichier texte

resume_txt = f"""
{'='*60}
RESUME — Scripteur {scripteur}  (Grille 1D)
{'='*60}
eps_y opt    : {eps_y_opt:.3f}  [{eps_y_min:.3f} - {eps_y_max:.3f}]
eps_x opt    : {eps_x_opt:.3f}
n_lignes     : {n_lignes_vis} (plateau={plateau_val})
n_mots       : {n_mots_vis} / {n_bm_reels}
ARI          : {ari_vis:.3f}
Zone optimale       : eps_x in [{eps_x_ok_min} - {eps_x_ok_max}]
Sur-regroupement    : eps_x > {eps_x_sur_max}  ({synthese['pct_sur_regroupement'][0]}% grille)
Sous-regroupement   : eps_x < {eps_x_sous_min}  ({synthese['pct_sous_regroupement'][0]}% grille)
Zone optimale       : {synthese['pct_zone_optimale'][0]}% de la grille
"""

with open(f"{out}/resume_grille1d_{scripteur}.txt", "w") as f:
    f.write(resume_txt)
print(f"Resume sauvegarde : {out}/resume_grille1d_{scripteur}.txt")
