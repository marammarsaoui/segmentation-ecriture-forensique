"""
=========================================================================
 CARTOUCHE:
 Fichier: JSON_show_first_stroke_of_txt.py
 Projet: Segmentation automatique de texte manuscrit en ligne
 Rôle: Script le plus élémentaire du dépôt. Lit un seul scripteur (203),
       trace une ligne noire continue à partir de ses points, et
       enregistre l'image obtenue.

 ENTRÉE:
   203_with_BM.json (doit se trouver dans le même dossier que ce script)

 SORTIE:
   trace_203.png

 BUG FONDATEUR DE CE SCRIPT (conservé volontairement, pour la trace):
 La ligne "points = data[0]["Points"]" ne lit que le PREMIER segment
 annoté du fichier, pas l'ensemble du texte : data[0] désigne le premier
 élément de la liste data, pas le document entier. Le commentaire laissé
 dans le code ("la faute venait de ça") marque le moment où cette
 confusion a été identifiée : le nombre de points affiché était trop
 petit parce que seul le premier trait du scripteur était lu, le reste
 du document ignoré silencieusement. Ce bug a été corrigé dans les
 scripts suivants (voir lire_tous_les_textes.py, qui boucle sur tous les
 segments avec "for segment in data").
=========================================================================
"""

import json

import matplotlib.pyplot as plt 
#lire à partir d'un ichier JSON 
#à revoir si j'ai besoin de l'option UTF-8
with open("203_with_BM.json","r") as file : 
    data=json.load(file)
#comme on a la data on peut savoir le nbr de point dans ce fichier 
points = data[0]["Points"]# la faute venait de ça 
print(len(points)) 
#pour extraire les informations
#selon les fichiers JSON et le rapport , on a 11 paramètres 
#pour réaliser le tracé , on a besoin de quoi 
#X,Y et N 
X=[]
for p in points :
    X.append(p["X"])
Y=[]
for p in points :
    Y.append(p["Y"])
#je prends la figure par défaut sinon on peut changer la taille 
plt.figure()
#option pour ligne continue et noire 
plt.plot(X,Y,"-k")
plt.savefig("trace_203.png")
plt.close()