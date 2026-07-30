"""
=========================================================================
 CARTOUCHE:
 Fichier: lire_tous_les_textes.py
 Projet: Segmentation automatique de texte manuscrit en ligne
 Rôle: Version corrigée et enrichie de lire_le_dossier_JSON_automatiquement.py.
       Compare les scripteurs des trois dossiers (mêmes calculs), puis
       trace chaque fichier en attribuant une couleur aléatoire distincte
       à CHAQUE segment (mot) plutôt qu'une seule ligne noire pour tout
       le fichier. C'est ce script qui répond à l'objectif de départ :
       visualiser chaque mot annoté dans une couleur différente.

 ENTRÉE:
   ../json_bm/*_with_BM.json
   ../modified/*_with_BM.json
   ../rangement/zest/*_with_bm_zest.json

 SORTIE:
   Terminal: mêmes comparaisons de scripteurs que le script précédent.
   output_complet_avec_couleur_par_stroke/{nom_dossier}/trace_{index}.png

 CORRECTION APPORTÉE PAR CE SCRIPT:
 La boucle "for segment in data" parcourt désormais TOUS les segments du
 fichier, corrigeant le bug hérité des deux scripts précédents (qui ne
 traçaient que le premier segment pour les dossiers structurés).

 BUG D'AFFICHAGE CONNU DANS CE SCRIPT:
 Deux lignes consécutives du bloc de comparaison affichent un décompte
 puis une liste qui ne correspondent pas l'un à l'autre :
   print(f"Seulement dans modified sont ({len(mod - bm)}) ")
   print(f"Seulement dans modified     ({len(mod-bm)})  : {sorted(bm - mod, key=int)}")
 La seconde ligne annonce un compte calculé sur mod - bm, mais liste
 ensuite les éléments de bm - mod, l'ensemble inverse. Le nombre annoncé
 et la liste affichée ne décrivent donc pas le même ensemble. N'affecte
 pas le tracé final, mais peut induire en erreur si on se fie au
 terminal pour vérifier rapidement quels scripteurs sont communs.
=========================================================================
"""

import json
import matplotlib.pyplot as plt
import os
import random 
"""Ce scripte python (une fonction) récupère une liste "dossiers"et une liste "suffixes" , elles représentent les dossiers dont on veut 
les lires et les tracés sous forme d'image textuelle 
Alors, la sortie est bien un dossier de la liste des entrées où dedans on trouve tous les fichiers png  """
#Dossiers et leurs suffixes
dossiers = ["../json_bm", "../modified", "../rangement/zest"]
#les suffixes vont servire à extraire extraire le numéro du scripteur depuis le nom du fichier 
suffixes  = ["_with_BM.json", "_with_BM.json", "_with_bm_zest.json"]

#Comparaison des scripteurs entre dossiers
#pour une meilleur portabilité , j'ai préféré écrire une fonction ici 
def lire_scripteurs(dossier,suffixe):
    numeros=[]
    fichiers=os.listdir(dossier)
    for fichier in fichiers : 
        if fichier.endswith(".json"):
            #comme je classifie chaque texte d'un scripteur selon le dossier qui represente ça
            #on peut supprimer le suffixe 
            numero=fichier.replace(suffixe,"")
            numeros.append(numero)
    return set(numeros)
bm   = lire_scripteurs(dossiers[0], suffixes[0])
mod  = lire_scripteurs(dossiers[1], suffixes[1])
zest = lire_scripteurs(dossiers[2], suffixes[2])

print(f"Les scripteurs commun entre BM et modified sont  ({len(bm & mod)}) ")
print(f"Seulement dans modified  sont  ({len(mod - bm)})  ")
print(f"Seulement dans modified     ({len(mod-bm)})  : {sorted(bm - mod, key=int)}")
print(f"Scripteurs dans zest      ({len(zest)})  : {sorted(zest, key=int)}")
print(f"Communs entre json_bm et zest    ({len(bm & zest)})  : {sorted(bm & zest, key=int)}")

#Génération des tracés : 
for i in range(len(dossiers)):
    dossier=dossiers[i]
    suffixe=suffixes[i]
    # créer le dossier de sortie
    #on recupère le chemin 
    nom_dossier = os.path.basename(dossier)
    dossier_sortie="output_complet_avec_couleur_par_stroke/" + nom_dossier
    os.makedirs(dossier_sortie, exist_ok=True)

    # lire chaque fichier JSON
    #listdir le permet plus facilemnent que manuellement (j'étais bloqué avec l'implémentation manuelle)
    for fichier in os.listdir(dossier):
        if not fichier.endswith(".json"):
            continue
        #
        index   = fichier.replace(suffixe, "")
        chemin  = dossier + "/" + fichier

        with open(chemin, "r") as f:
            data = json.load(f)

        # tracer TOUS les strokes
        plt.figure()
        for segment in data:
            points = segment["Points"]
            X = [p["X"] for p in points]
            Y = [p["Y"] for p in points]
            #06x pour 6 élémenets hexadécimaux 
            #Deplus , 0xFFFFFF est simplement le blanc alors on aura une couleur visible toujours 
            couleur = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            plt.plot(X, Y,color=couleur )

        plt.title("Scripteur " + index + " — " + nom_dossier)
        #plt.gca()
        plt.savefig(dossier_sortie + "/trace_" + index + ".png")
        plt.close()