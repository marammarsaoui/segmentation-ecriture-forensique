"""
 cascade_metriques.py
 OBJECTIF: 
 Comparer 4 métriques de distance (euclidean, manhattan, chebyshev,
 minkowski p=3) pour la passe 2 de la cascade DBSCAN.

 PRINCIPE : 
 Passe 1 : DBSCAN sur Y seul (euclidean, 1D) → détecte les lignes
 Passe 2 : DBSCAN sur (X, Y) en 2D → détecte les mots  avec 4 métriques comparées

 Pourquoi (X, Y) en passe 2 ?
   En 1D (X seul), toutes les métriques sont identiques : la distance
   entre deux scalaires est toujours |x2-x1|. En 2D (X, Y), les
   formules divergent et produisent des clusters différents.
   Cela permet une vraie comparaison des métriques.

 UTILISATION: 
   python3 cascade_metriques.py <fichier_json>
   Ex:python3 cascade_metriques.py ../json_bm/243_with_BM.json 5 3
 SORTIES
   outputs/metriques_{scripteur}.csv- tableau comparatif des métriques
   outputs/metriques_{scripteur}.png-courbes ARI
   outputs/vis_BM_{scripteur}_{metrique}.png
   outputs/vis_motID_{scripteur}_{metrique}.png
   outputs/vis_lignes_{scripteur}_{metrique}.png
"""

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

warnings.filterwarnings("ignore")

########
# ÉTAPE 0 : Lecture des arguments en ligne de commande
#
# Le script attend exactement 3 arguments après son nom :
#   argv[1] = chemin du fichier JSON du scripteur
# Si un argument manque, on affiche l'usage et on arrête proprement.
########

if len(sys.argv) != 2:
    print("Usage : python3 cascade_metriques.py <fichier_json>")
    sys.exit(1)

fichier   = sys.argv[1]

# Extraction de l'identifiant du scripteur depuis le nom de fichier.
# Ex : "../json_bm/243_with_BM.json" → "243"
scripteur = os.path.basename(fichier).replace("_with_BM.json", "")

print(f"\n{'='*65}")
print(f"Scripteur : {scripteur} ")
print(f"Passe 2 : DBSCAN sur (X, Y) — 4 métriques comparées")
print(f"{'='*65}")


########
# ÉTAPE 1 : Chargement et préparation des données
#
# On charge le fichier JSON, on filtre les ponctuations (BM == "P"),
# puis on construit les tableaux numpy nécessaires à DBSCAN :
#   coords   : (n_points, 2) — X et Y de chaque point
#   coords_Y : (n_points, 1) — Y seul pour la passe 1 (lignes)
#   N_liste  : identifiants uniques des points (clé de jointure)
########

with open(fichier) as f:
    data = json.load(f)

# Filtre ponctuation : BM == "P" = segments de ponctuation,
# exclus car ils ne correspondent pas à des mots.
data_filtre = [seg for seg in data if seg["BM"] != "P"]

# n_bm_reels = nombre de mots réels dans le texte (vérité terrain).
# set() élimine les doublons → on compte les labels BM distincts.
n_bm_reels = len(set(seg["BM"] for seg in data_filtre))
n_strokes  = len(data_filtre)

# Aplatissement de la structure hiérarchique (segments → points)
# en une liste plate de triplets [X, Y, N]
points_bruts = []
for seg in data_filtre:
    for p in seg["Points"]:
        points_bruts.append([p["X"], p["Y"], p["N"]])

# coords   : tableau (n_points, 2) contenant X et Y
# coords_Y : colonne Y seule, shape (n_points, 1).
#            Le .reshape(-1, 1) est obligatoire : DBSCAN attend un
#            tableau 2D même pour une seule dimension. -1 signifie
#            "calcule automatiquement cette taille".
# N_liste  : liste des identifiants N, gardés pour la jointure
#            entre les tableaux numpy et les segments d'origine.
coords   = np.array([[p[0], p[1]] for p in points_bruts])
coords_Y = coords[:, 1].reshape(-1, 1)
N_liste  = [p[2] for p in points_bruts]

print(f"Strokes : {n_strokes}  |  Points : {len(coords)}  |  BM réels : {n_bm_reels}")

########
# ÉTAPE 2 : Copie profonde des données
# La cascade enrichit chaque point avec LigneID et MotID.
# Sans copie, ces annotations s'accumuleraient entre les itérations
# de grille et corrupraient les résultats suivants.
# On repart d'une copie fraîche à chaque appel de cascade_XY().
########

def copie(data_filtre):
    return [{"BM": s["BM"], "Points": [dict(p) for p in s["Points"]]}
            for s in data_filtre]


########
# ÉTAPE 3 : Cascade DBSCAN sur (X, Y) avec métrique variable
# Passe 1 : DBSCAN sur Y seul (euclidean, toujours) → lignes
# Passe 2 : DBSCAN sur (X, Y) en 2D avec la métrique choisie → mots
# Le paramètre `metric` détermine la formule de distance utilisée
# dans la passe 2. Le paramètre `p` n'est utilisé que pour minkowski.
# Retourne : (n_lignes, n_mots, ARI)
########

def cascade_XY(coords, coords_Y, N_liste, data_seg,
               eps_y, eps_x, mSy, mSx, metric, p=2):

    # --- Passe 1 : DBSCAN sur Y → détection des lignes ---
    # La métrique est toujours euclidean en 1D (toutes équivalentes).
    # labels_ligne[i] = identifiant de ligne du point i (-1 = bruit).
    labels_ligne   = DBSCAN(eps=eps_y, min_samples=mSy).fit_predict(coords_Y)
    lignes_uniques = sorted(set(labels_ligne) - {-1})

    # Initialisation : tous les points sont bruit (-1) pour les mots.
    # mot_global décale les labels locaux à chaque ligne pour garantir
    # leur unicité sur tout le document (ex: mot 0 ligne 1 ≠ mot 0 ligne 2).
    labels_mot = np.full(len(coords), -1, dtype=int)
    mot_global = 0

    # --- Passe 2 : pour chaque ligne, DBSCAN sur (X, Y) → mots ---
    for ligne_id in lignes_uniques:
        # Indices des points de cette ligne dans le tableau global
        idx = np.where(labels_ligne == ligne_id)[0]

        # Espace 2D (X, Y) pour les points de cette ligne.
        # C'est ici que les métriques produisent des résultats différents :
        # chaque formule mesure différemment la "proximité" entre deux points.
        # NOTE extension future : remplacer coords[idx, 1] par N_norm[idx]
        #      pour passer en (X, Time_MS normalisé).
        coords_XY = np.column_stack([
            coords[idx, 0],   # X : position horizontale
            coords[idx, 1]    # Y : position verticale
        ])

        # Construction du DBSCAN avec la métrique choisie.
        # Minkowski nécessite le paramètre p supplémentaire ;
        # les autres métriques ne l'acceptent pas → if/else obligatoire.
        if metric == "minkowski":
            db = DBSCAN(eps=eps_x, min_samples=mSx,
                        metric="minkowski", p=p)
        else:
            db = DBSCAN(eps=eps_x, min_samples=mSx, metric=metric)

        labels_local = db.fit_predict(coords_XY)
        mots_locaux  = sorted(set(labels_local) - {-1})

        # Décalage des labels locaux par mot_global.
        # Ex : ligne 0 produit mots 0,1,2 → mot_global = 3
        #      ligne 1 produit mots 3,4   → mot_global = 5
        for i, idx_pt in enumerate(idx):
            if labels_local[i] != -1:
                labels_mot[idx_pt] = mot_global + labels_local[i]
        if mots_locaux:
            mot_global += max(mots_locaux) + 1

    # --- Jointure N → LigneID / MotID ---
    # On construit deux dicts pour remonter les labels numpy vers
    # les segments d'origine via l'identifiant N.
    N_to_ligne = {N_liste[i]: int(labels_ligne[i]) for i in range(len(N_liste))}
    N_to_mot   = {N_liste[i]: int(labels_mot[i])   for i in range(len(N_liste))}

    for seg in data_seg:
        for p_pt in seg["Points"]:
            p_pt["LigneID"] = N_to_ligne.get(p_pt["N"], -1)
            p_pt["MotID"]   = N_to_mot.get(p_pt["N"], -1)

    # --- Agrégation au niveau stroke par vote majoritaire ---
    # Un stroke peut chevaucher deux clusters à sa frontière.
    # Counter(...).most_common(1)[0][0] retourne le label le plus
    # fréquent parmi tous les points du stroke.
    df = pd.DataFrame(data_seg)
    df["LigneID"] = df["Points"].apply(
        lambda pts: Counter(p["LigneID"] for p in pts).most_common(1)[0][0])
    df["MotID"] = df["Points"].apply(
        lambda pts: Counter(p["MotID"] for p in pts).most_common(1)[0][0])

    # --- Calcul de l'ARI ---
    # On exclut les strokes bruit (MotID == -1).
    # ARI = 1 → parfait, ARI ≈ 0 → aléatoire.
    df_valid = df[df["MotID"] != -1]
    ari = adjusted_rand_score(
        df_valid["BM"].tolist(), df_valid["MotID"].tolist()
    ) if len(df_valid) > 1 else 0.0

    n_mots   = df["MotID"].nunique() - (1 if -1 in df["MotID"].values else 0)
    n_lignes = len(lignes_uniques)
    return n_lignes, n_mots, ari


########
# ÉTAPE 4 : Définition des métriques à comparer
#
# Chaque tuple : (nom_affichage, nom_sklearn, valeur_p)
# La valeur_p n'est utilisée que pour minkowski (ignorée sinon).
#
# En 2D (X, Y), les 4 métriques produisent des clusters différents :
#   euclidean  : favorise les clusters ronds (distance à vol d'oiseau)
#   manhattan  : favorise les clusters en losange (distance en taxi)
#   chebyshev  : favorise les clusters carrés (max des deux écarts)
#   minkowski  : intermédiaire entre euclidean et chebyshev (p=3)
########

METRIQUES = [
    ("euclidean","euclidean", 2),
    ("manhattan", "manhattan", 1),
    ("chebyshev (L-inf)", "chebyshev", None),
    ("minkowski p=3", "minkowski", 3),
]
# Grilles de recherche- mêmes  que grille_2d.py pour comparabilité.
# np.round(..., 3)
grille_y = np.round(np.arange(0.05, 1.0, 0.05), 3)   # 19 valeurs, pas 0.05
grille_x = np.round(np.arange(0.10, 3.0, 0.10), 3)   # 29 valeurs, pas 0.10

print(f"\nGrille passe 1 : {len(grille_y)} x {len(grille_x)} = "
      f"{len(grille_y)*len(grille_x)} combinaisons par métrique")
print(f"Métriques : {[m[0] for m in METRIQUES]}\n")


########
# ÉTAPE 5 : Boucle principale-grille grossière + zoom pour chaque métrique
# Pour chaque métrique :
#1. Passe 1: balayage grossier de toute la grille (eps_y × eps_x)
#2. Passe 2: zoom fin autour du meilleur point de la passe 1
#On retient le meilleur ARI obtenu sur les deux passes.
########

resultats = []

for nom_m, metric_str, p_val in METRIQUES:
    print(f"── {nom_m} ──")

    best_ari  = -1
    best_ey   = None
    best_ex   = None
    best_mots = None
    rows      = []
    done      = 0
    total     = len(grille_y) * len(grille_x)

    # --- Passe 1 : grille grossière ---
    # try/except : certaines combinaisons extrêmes peuvent provoquer
    # des erreurs numériques dans DBSCAN (eps trop petit, données
    # dégénérées). On leur attribue ARI=0 pour ne pas interrompre la boucle.
    for ey in grille_y:
        for ex in grille_x:
            try:
                nl, nm, ari = cascade_XY(
                    coords, coords_Y, N_liste,
                    copie(data_filtre),
                    ey, ex, 5,5,
                    metric_str, p=p_val if p_val else 2
                )
            except Exception:
                ari, nl, nm = 0.0, 0, 0
            rows.append((ey, ex, nl, nm, ari))
            # Mise à jour du meilleur résultat en temps réel
            if ari > best_ari:
                best_ari  = ari
                best_ey   = ey
                best_ex   = ex
                best_mots = nm
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} ...")

    df_m = pd.DataFrame(rows, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])

    # --- Passe 2 : zoom autour du meilleur point ---
    # Fenêtre : ±0.08 sur eps_y (pas 0.01), ±0.20 sur eps_x (pas 0.05).
    # max(0.01, ...) empêche des valeurs négatives ou trop proches de zéro.
    grille_y2 = np.round(np.arange(
        max(0.01, best_ey - 0.08), best_ey + 0.09, 0.01), 3)
    grille_x2 = np.round(np.arange(
        max(0.05, best_ex - 0.20), best_ex + 0.21, 0.05), 3)

    rows2 = []
    for ey in grille_y2:
        for ex in grille_x2:
            try:
                nl, nm, ari = cascade_XY(
                    coords, coords_Y, N_liste,
                    copie(data_filtre),
                    ey, ex, 5,5,
                    metric_str, p=p_val if p_val else 2
                )
            except Exception:
                ari, nl, nm = 0.0, 0, 0
            rows2.append((ey, ex, nl, nm, ari))
            if ari > best_ari:
                best_ari  = ari
                best_ey   = ey
                best_ex   = ex
                best_mots = nm

    df_m2 = pd.DataFrame(rows2, columns=["eps_y", "eps_x", "n_lignes", "n_mots", "ARI"])

    print(f"  Meilleur : eps_y={best_ey:.3f}  eps_x={best_ex:.3f}"
          f"  ARI={best_ari:.3f}  mots={best_mots}/{n_bm_reels}\n")

    # On stocke tous les résultats de cette métrique pour les figures
    resultats.append({
        "metrique"   : nom_m,
        "metric_str" : metric_str,
        "p_val"      : p_val,
        "eps_y_opt"  : best_ey,
        "eps_x_opt"  : best_ex,
        "n_mots"     : best_mots,
        "ARI"        : round(best_ari, 3),
        "df_grille1" : df_m,
        "df_grille2" : df_m2,
    })


########
# ÉTAPE 6 : Résumé comparatif terminal
# Tableau récapitulatif de toutes les métriques avec leurs paramètres
# optimaux et ARI obtenus. La meilleure métrique est mise en évidence.
########

print(f"\n{'='*65}")
print(f"RÉSUMÉ — Scripteur {scripteur}  (passe 2 sur X, Y)")
print(f"{'='*65}")
print(f"{'Métrique':<22} {'eps_y':>8} {'eps_x':>8} "
      f"{'n_mots':>10} {'ARI':>8}")
print("-" * 65)
for r in resultats:
    print(f"{r['metrique']:<22} {r['eps_y_opt']:>8.3f} "
          f"{r['eps_x_opt']:>8.3f} "
          f"{str(r['n_mots'])+'/'+str(n_bm_reels):>10} "
          f"{r['ARI']:>8.3f}")

# Meilleure métrique = celle avec l'ARI le plus élevé
best_r = max(resultats, key=lambda r: r["ARI"])
print(f"\nMeilleure métrique : {best_r['metrique']}"
      f"  ARI={best_r['ARI']:.3f}")


########
# ÉTAPE 7 : Sauvegarde CSV
# Un seul CSV de synthèse : une ligne par métrique avec ses paramètres
# optimaux et son ARI. Permet de comparer facilement les métriques
# entre scripteurs sur tableur.
########

os.makedirs("outputs", exist_ok=True)
df_resume = pd.DataFrame([{
    "scripteur"  : scripteur,
    "metrique"   : r["metrique"],
    "eps_y_opt"  : r["eps_y_opt"],
    "eps_x_opt"  : r["eps_x_opt"],
    "n_mots"     : r["n_mots"],
    "n_bm_reels" : n_bm_reels,
    "ARI"        : r["ARI"],
} for r in resultats])
csv_path = f"outputs/metriques_{scripteur}.csv"
df_resume.to_csv(csv_path, index=False)
print(f"\nCSV sauvegardé : {csv_path}")

########
# ÉTAPE 8 : Visualisation texte- meilleure métrique, paramètres optimaux
# On relance la cascade une dernière fois avec les paramètres optimaux
# de la meilleure métrique pour produire les 3 figures texte :
#Figure A:(labels BM d'origine)
#Figure B: mots détectés (MotID)
#Figure C: lignes détectées (LigneID)
########

print(f"\n── Visualisation texte (meilleure métrique : {best_r['metrique']}) ──")

eps_y_vis= best_r["eps_y_opt"]
eps_x_vis = best_r["eps_x_opt"]
metric_vis= best_r["metrique"]
metric_sklearn= best_r["metric_str"]
p_vis= best_r["p_val"]

# Relance de la cascade avec les paramètres optimaux
labels_ligne_vis =DBSCAN(eps=eps_y_vis, min_samples=5).fit_predict(coords_Y)
lignes_uniques_vis= sorted(set(labels_ligne_vis) - {-1})

labels_mot_vis= np.full(len(coords), -1, dtype=int)
mot_global_vis= 0

for ligne_id in lignes_uniques_vis:
    idx = np.where(labels_ligne_vis == ligne_id)[0]
    # Espace 2D (X, Y) — même logique que dans cascade_XY()
    coords_XY = np.column_stack([coords[idx, 0], coords[idx, 1]])
    if metric_sklearn == "minkowski":
        db = DBSCAN(eps=eps_x_vis, min_samples=5,
                    metric="minkowski", p=p_vis)
    else:
        db = DBSCAN(eps=eps_x_vis, min_samples=5, metric=metric_sklearn)
    labels_local = db.fit_predict(coords_XY)
    mots_locaux  = sorted(set(labels_local) - {-1})
    for i, idx_pt in enumerate(idx):
        if labels_local[i] != -1:
            labels_mot_vis[idx_pt] = mot_global_vis + labels_local[i]
    if mots_locaux:
        mot_global_vis += max(mots_locaux) + 1

# Jointure et annotation des segments
N_to_ligne_vis = {N_liste[i]: int(labels_ligne_vis[i]) for i in range(len(N_liste))}
N_to_mot_vis   = {N_liste[i]: int(labels_mot_vis[i])   for i in range(len(N_liste))}

data_vis = copie(data_filtre)
for seg in data_vis:
    for p in seg["Points"]:
        p["LigneID"] = N_to_ligne_vis.get(p["N"], -1)
        p["MotID"]   = N_to_mot_vis.get(p["N"], -1)

# Agrégation au niveau stroke
df_vis = pd.DataFrame(data_vis)
df_vis["LigneID"] = df_vis["Points"].apply(
    lambda pts: Counter(p["LigneID"] for p in pts).most_common(1)[0][0])
df_vis["MotID"] = df_vis["Points"].apply(
    lambda pts: Counter(p["MotID"] for p in pts).most_common(1)[0][0])

# ARI de vérification sur le résultat final
df_vis_valid = df_vis[df_vis["MotID"] != -1]
ari_vis = adjusted_rand_score(
    df_vis_valid["BM"].tolist(), df_vis_valid["MotID"].tolist()
) if len(df_vis_valid) > 1 else 0.0
n_mots_vis   = df_vis["MotID"].nunique() - (1 if -1 in df_vis["MotID"].values else 0)
n_lignes_vis = len(lignes_uniques_vis)
print(f"Lignes : {n_lignes_vis}  |  Mots : {n_mots_vis}/{n_bm_reels}  |  ARI : {ari_vis:.3f}")


# Fonctions utilitaires pour la visualisation
# gen_couleurs : associe une couleur hex à chaque label.
#   seed fixe → couleurs stables entre exécutions.
#   label -1 (bruit) → toujours gris (#aaaaaa).
def gen_couleurs(vals, seed):
    random.seed(seed)
    return {v: "#aaaaaa" if v == -1 else
            "#{:06x}".format(random.randint(0, 0xFFFFFF))
            for v in vals}

# plot_texte : trace et sauvegarde une figure avec les strokes
# colorés selon un label (BM, MotID ou LigneID).
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

# Figure A — Vérité terrain (labels BM d'origine)
bm_colors = gen_couleurs(sorted(df_vis["BM"].unique(), key=str), 42)
plot_texte(df_vis, "BM", bm_colors,
           f"Vérité terrain — BM réels (scripteur {scripteur})",
           f"outputs/vis_BM_{scripteur}_{metric_vis}.png")

# Figure B — Mots détectés par la cascade (MotID)
mot_colors = gen_couleurs(sorted(df_vis["MotID"].unique(), key=str), 99)
plot_texte(df_vis, "MotID", mot_colors,
           f"Cascade (X, Y) — MotID détectés ({n_mots_vis} mots)\n"
           f"eps_y={eps_y_vis:.3f}  eps_x={eps_x_vis:.3f}  "
           f"métrique={metric_vis}  ARI={ari_vis:.3f}",
           f"outputs/vis_motID_{scripteur}_{metric_vis}.png")

# Figure C — Lignes détectées par la passe 1 (LigneID)
ligne_colors = gen_couleurs(sorted(df_vis["LigneID"].unique(), key=str), 7)
plot_texte(df_vis, "LigneID", ligne_colors,
           f"Étape 1 — Lignes détectées ({n_lignes_vis} lignes)\n"
           f"eps_y={eps_y_vis:.3f}",
           f"outputs/vis_lignes_{scripteur}_{metric_vis}.png")


########
# ÉTAPE 9 : Figure d'analyse — affichée en dernier
#
# Figure gauche : ARI vs eps_x pour chaque métrique (eps_y fixé à optimal).
#   Permet de comparer visuellement l'effet de eps_x selon la métrique.
# Figure droite : heatmap ARI (eps_y × eps_x) pour la meilleure métrique.
#   Montre la robustesse de la solution optimale.
# plt.show() est appelé ici en dernier : le script se met en pause
# jusqu'à ce que l'utilisateur ferme la fenêtre.
########

colors_m = {
    "euclidean"         : "#2980b9",
    "manhattan"         : "#27ae60",
    "chebyshev (L-inf)" : "#8e44ad",
    "minkowski p=3"     : "#e67e22",
}

fig, ax = plt.subplots(figsize=(10, 5))

# ARI vs eps_x pour chaque métrique, eps_y fixé à sa valeur optimale.
# Permet de comparer visuellement l'effet de eps_x selon la métrique.
for r in resultats:
    sub = r["df_grille1"][r["df_grille1"]["eps_y"] == r["eps_y_opt"]]
    ax.plot(sub["eps_x"], sub["ARI"],
            label=r["metrique"],
            color=colors_m.get(r["metrique"], "gray"),
            linewidth=2)
ax.set_title("ARI vs eps_x par métrique\n(passe 2 sur X, Y)")
ax.set_xlabel("eps_x")
ax.set_ylabel("ARI")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(
    f"Scripteur {scripteur} — Métriques sur (X, Y)\n"
    f"Meilleure : {best_r['metrique']}  "
    f"ARI={best_r['ARI']:.3f}  "
    f"Mots={best_r['n_mots']}/{n_bm_reels}",
    fontsize=12)
plt.tight_layout()
nom_fig = f"outputs/metriques_{scripteur}.png"
plt.savefig(nom_fig, dpi=150)   # sauvegarde d'abord
print(f"Figure analyse sauvegardée : {nom_fig}")
plt.show()                       # affiche en dernier — pause jusqu'à fermeture
plt.close()                      # libère la mémoire