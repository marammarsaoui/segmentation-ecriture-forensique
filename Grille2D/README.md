
# Grille 2D : 

Deux scripts qui cherchent conjointement les hyperparamètres `eps_y` et `eps_x` de la cascade DBSCAN, contrairement à `grille1d/` qui les cherche séparément. `grille_2d.py` traite un scripteur à la fois, en exploration ; `run_all_scripteurs.py` applique la même logique à tous les scripteurs d'un dossier, en traitement par lot.

## Le principe commun : recherche en deux passes

Une grille grossière balaie tout l'espace `(eps_y, eps_x)` (19 × 29 = 551 combinaisons), puis une seconde passe affine autour du meilleur point trouvé, avec un pas plus fin. Ce n'est qu'ensuite que la meilleure combinaison, toutes passes confondues, est retenue.

## `grille_2d.py` : exploration d'un seul scripteur

```bash
python3 grille_2d.py <fichier_json> <minS_y> <minS_x>
```

En plus des trois figures classiques (vérité terrain, lignes, mots), ce script produit une **surface 3D** de l'ARI en fonction de `(eps_y, eps_x)` — utile pour voir d'un coup d'œil si le réglage optimal est stable (une surface plate autour du maximum) ou fragile (un pic étroit). La figure s'ouvre en fenêtre interactive, qu'on peut faire pivoter à la souris.

## `run_all_scripteurs.py` : traitement de tout un dossier

```bash
python3 run_all_scripteurs.py <minS_y> <minS_x> [dossier_json]
```

Applique la même recherche à chaque fichier `*_with_BM.json` d'un dossier, produit une synthèse globale (ARI moyen, ARI médian), et archive l'ensemble du dossier `outputs/` en zip horodaté à la fin du run.

## Une différence qui n'est pas un oubli

`grille_2d.py` ne teste que l'espace spatial `(X, Y)` : la recherche s'arrête là où elle en est, sans recours si l'ARI reste faible. `run_all_scripteurs.py`, lui, inclut un **switch temporel** : si l'ARI spatial reste sous 0.80 pour un scripteur donné, la recherche est relancée dans un espace `(X, Time_MS normalisé)`, et le meilleur des deux résultats est conservé.

Ce n'est pas un oubli dans `grille_2d.py` — c'est cohérent avec son usage : un script d'exploration ponctuelle, où l'utilisateur peut inspecter visuellement le résultat et décider lui-même s'il faut aller plus loin. Le switch automatique a plus de sens dans `run_all_scripteurs.py`, pensé pour tourner sans supervision sur des dizaines de fichiers, où il faut une décision automatique plutôt qu'une inspection manuelle à chaque cas.

## Dépendances

`numpy`, `pandas`, `matplotlib`, `scikit-learn`. `run_all_scripteurs.py` ajoute `glob`, `shutil` et `time` pour le traitement par lot et l'archivage.

## Utilisation

```bash
# Un seul scripteur, avec surface 3D
python3 grille_2d.py ../json_bm/203_with_BM.json 2 3

# Tous les scripteurs d'un dossier, avec switch temporel automatique
python3 run_all_scripteurs.py 5 3 ../json_bm
```
