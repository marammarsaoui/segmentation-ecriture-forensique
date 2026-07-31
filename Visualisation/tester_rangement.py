import os, json
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Code écrit et développé par Maram Marsaoui (marammarsaoui28@gmail.com)
#   Dans le cadre du stage de M1 avec Monsieur Vincent Brault et Madame Fanny Guillet 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Input: un dossier 
1-on liste tous ses éléments 
2-on les tri par ordre alphabétique
3-on parcourt tous les fichiers de chaque mot ( c'est un dossier aussi ; liste)

output : affichage sur terminal : pour chaque mot du dossier rangement :
on cherche le nombre max et min de strokes pour écrire ce mot """
# Lister tous les sous-dossiers de rangement
racine = "../rangement"
mots = []
for element in os.listdir(racine):
    chemin_complet = os.path.join(racine, element)
    # garder seulement les dossiers
    if os.path.isdir(chemin_complet):
        mots.append(element)

# trier par ordre alphabétique
mots = sorted(mots)

print(f"{'Mot'} {'Min'} {'Max'}")

for mot in mots:
    chemin = os.path.join(racine, mot)
    suffixe = f"_with_bm_{mot}.json"
    strokes = []

    for fichier in os.listdir(chemin):
        if not fichier.endswith(".json"):
            continue
        with open(os.path.join(chemin, fichier)) as f:
            data = json.load(f)
        strokes.append(len(data))

    
    print(f"{mot:<20} {min(strokes):<6} {max(strokes)}")
