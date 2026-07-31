"""
=========================================================================
 CARTOUCHE:
 Fichier: counter.py
 # 
#   Code écrit et développé par Maram Marsaoui (marammarsaoui28@gmail.com)
#   Dans le cadre du stage de M1 avec Monsieur Vincent Brault et Madame Fanny Guillet 
# 
 Projet: Segmentation automatique de texte manuscrit en ligne
 Rôle: Script de diagnostic pour un seul scripteur (203). Compte le
       nombre d'occurrences de chaque label BM dans json_bm/, ce qui
       donne le nombre de traits par mot et le nombre de mots distincts.
       Vérifie ensuite la structure du JSON (clés du premier élément) et
       répète cette vérification sur le fichier correspondant dans
       rangement/zest/, pour comparer les deux structures côte à côte.

 ENTRÉE:
   ../json_bm/203_with_BM.json          (liste de segments {BM, Points})
   ../rangement/zest/203_with_bm_zest.json

 SORTIE:
   Affichage terminal uniquement : nombre de mots distincts, nombre de
   strokes par label BM, nombre d'éléments et clés de chaque fichier.

 CE QUE CE SCRIPT A PERMIS DE DÉCOUVRIR:
 C'est ce script qui a confirmé que json_bm/ et rangement/ n'ont pas la
 même structure interne (d'où le "print("Clés :", ...)" en double, une
 fois par dossier) : information nécessaire pour écrire correctement les
 scripts qui doivent lire les deux organisations sans supposer qu'elles
 sont identiques.
=========================================================================
"""

import json ,collections
#la librairie collections permet d'utiliser un counter pour compter automatiquement les occurrences de chaque élément dans une liste 
with open("../json_bm/203_with_BM.json") as file : 
    data=json.load(file)
#bm_count = collections.Counter(seg["BM"] for seg in data)
#or compter les BM manuellmenet (ce que j'ai préféré)
counter={}
for segment in data: 
    bm =segment["BM"]
    if bm in counter : 
        counter[bm]+=1
    else : 
        counter[bm]=1
#affichage 
print("Pour le scripteur 203 : ")
print("les mots disctints \n",{len(counter)})
print("Strokes par mot : ")
for bm,nb_strokes in counter.items():
    print(f"Pour le label {bm} , on a {nb_strokes} stroke(s)")
#pour revérifier la structure de données ( c'est bien un dictionnaire)
#print(data[0].keys())
print("Nb éléments :", len(data))
print("Clés :", data[0].keys())
# Fichier 2 — rangement
with open("../rangement/zest/203_with_bm_zest.json") as f:
    data2 = json.load(f)

print("Dans le dossier zest de rangement on a : \n")
print("Nb éléments :", len(data2))
print("Clés :", data2[0].keys())
