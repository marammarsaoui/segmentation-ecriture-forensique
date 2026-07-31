"""
=========================================================================
 CARTOUCHE:
 # 
#   Code écrit et développé par Maram Marsaoui (marammarsaoui28@gmail.com)
#   Dans le cadre du stage de M1 avec Monsieur Vincent Brault et Madame Fanny Guillet 
# 
 Fichier: tracer_txt.py
 Projet: Segmentation automatique de texte manuscrit en ligne
 Rôle: Trace tous les segments d'un seul fichier déjà segmenté (issu
       d'un clustering, voir le nom du fichier attendu ci-dessous), avec
       une couleur aléatoire par segment. Sert de vérification visuelle
       rapide sur un résultat de segmentation, sans comparaison à une
       vérité terrain.
 
 ENTRÉE:
   ../203_with_clusters.json (liste de segments {BM ou label, Points})
   Remarque : le nom de fichier suggère un résultat déjà clusterisé
   (probablement une sortie de l'interface ou d'un des scripts grille),
   pas un fichier brut ni un fichier _with_BM.json d'origine.
 
 SORTIE:
   trace_200_cluster_complet.png
 
 REMARQUE SUR LE CODE:
 Le titre de la figure est désactivé (plt.title(...) mis en commentaire)
 et une seconde ligne plt.savefig vers trace_203_complet.png est aussi
 en commentaire : ce script semble avoir servi à comparer deux variantes
 de sortie (avec et sans clustering, scripteur 203 vs le fichier de
 clusters nommé 200 dans le nom de sortie) en activant l'une ou l'autre
 ligne selon le besoin du moment.
=========================================================================
"""
import json
import matplotlib.pyplot as plt
import random 
#"../json_bm/203_with_BM.json"
with open("../203_with_clusters.json", "r") as file:
    data = json.load(file)

print(f"Nombre de strokes : {len(data)}")

#Tracer TOUS les strokes:
X_total = []
Y_total = []

for segment in data:
    points = segment["Points"]  # tous les strokes, pas seulement data[0]

    X = [p["X"] for p in points]
    Y = [p["Y"] for p in points]

    X_total.extend(X)
    Y_total.extend(Y)
    couleur = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    plt.plot(X, Y, color=couleur)  # un trait par stroke

print(f"Nombre total de points : {len(X_total)}")

#plt.title("Scripteur 203 - texte complet")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca()  # inverser l'axe Y : la tablette compte de haut en bas

plt.savefig("trace_200_cluster_complet.png", dpi=150)
#plt.savefig("trace_203_complet.png", dpi=150)
plt.close()
#print("Sauvegardé : trace_203_complet.png")
