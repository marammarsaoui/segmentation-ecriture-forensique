"""
=========================================================================
 CARTOUCHE:
 # 
#   Code écrit et développé par Maram Marsaoui (marammarsaoui28@gmail.com)
#   Dans le cadre du stage de M1 avec Monsieur Vincent Brault et Madame Fanny Guillet 
# 
 Fichier: lire_le_dossier_JSON_automatiquement.py
 Projet: Segmentation automatique de texte manuscrit en ligne
 Rôle: Compare les scripteurs présents dans les trois dossiers de
       données (json_bm, modified, rangement/zest), puis trace un tracé
       simple (une ligne noire) pour chaque fichier de chaque dossier.

 ENTRÉE:
   ../json_bm/*_with_BM.json
   ../modified/*_with_BM.json
   ../rangement/zest/*_with_bm_zest.json

 SORTIE:
   Terminal : ensembles de scripteurs communs/exclusifs entre dossiers.
   output/{nom_dossier}/trace_{index}.png : un tracé par fichier.

 POINT CRITIQUE HÉRITÉ:
 Le bloc try/except gère deux structures possibles (data[0]["Points"]
 pour un format imbriqué, data directement pour un format plat), mais
 hérite du même bug que JSON_show_first_stroke_of_txt.py : pour les
 dossiers structurés en segments (json_bm, modified), seul le PREMIER
 segment du fichier est tracé, pas le texte entier. Corrigé dans
 lire_tous_les_textes.py, qui boucle sur tous les segments et attribue
 une couleur distincte à chacun.
=========================================================================
"""

import json
import matplotlib.pyplot as plt
import os

dossiers_input = [
    "../json_bm",
    "../modified",
    "../rangement/zest"
]

# ── Suffixes par dossier ──
suffixes = {
    "../json_bm"       : "_with_BM.json",
    "../modified"      : "_with_BM.json",
    "../rangement/zest": "_with_bm_zest.json"
}

# ── Comparaison des dossiers ──
def get_scripteurs(dossier, suffixe):
    return set(
        f.replace(suffixe, "")
        for f in os.listdir(dossier)
        if f.endswith(".json")
    )

bm   = get_scripteurs(dossiers_input[0], suffixes[dossiers_input[0]])
mod  = get_scripteurs(dossiers_input[1], suffixes[dossiers_input[1]])
zest = get_scripteurs(dossiers_input[2], suffixes[dossiers_input[2]])

communs       = sorted(bm & mod,   key=int)
only_modified = sorted(mod - bm,   key=int)
only_bm       = sorted(bm - mod,   key=int)
dans_zest     = sorted(zest,       key=int)
communs_zest  = sorted(bm & zest,  key=int)

print("=" * 45)
print(f"  Scripteurs communs bm/modified  ({len(communs)})      : {communs}")
print(f"  Seulement dans modified         ({len(only_modified)}) : {only_modified}")
print(f"  Seulement dans json_bm          ({len(only_bm)})       : {only_bm}")
print(f"  Scripteurs dans zest            ({len(dans_zest)})     : {dans_zest}")
print(f"  Communs json_bm + zest          ({len(communs_zest)})  : {communs_zest}")
print("=" * 45 + "\n")

# ── Génération des tracés ──
for dossier in dossiers_input:
    suffixe     = suffixes[dossier]
    nom_dossier = os.path.basename(dossier)
    dossier_sortie = os.path.join("output", nom_dossier)
    os.makedirs(dossier_sortie, exist_ok=True)

    fichiers_vus = set()
    for fichier in os.listdir(dossier):
        if not fichier.endswith(".json") or fichier in fichiers_vus:
            continue
        fichiers_vus.add(fichier)

        index = fichier.replace(suffixe, "")

        with open(os.path.join(dossier, fichier), "r") as f:
            data = json.load(f)

        # adapter selon la structure réelle du dossier
        try:
            points = data[0]["Points"]
        except (KeyError, TypeError):
            points = data  # structure plate — à confirmer pour rangement/

        X = [p["X"] for p in points]
        Y = [p["Y"] for p in points]

        plt.figure()
       
        plt.plot(X, Y, "-k")
        plt.title(f"Scripteur {index} — {nom_dossier}")
        plt.savefig(os.path.join(dossier_sortie, f"trace_{index}.png"))
        plt.close()
        print(f"  OK : trace_{index}.png  →  {nom_dossier}/")
