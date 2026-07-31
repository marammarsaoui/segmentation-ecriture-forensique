
# Grille 1D

Premier script écrit pour prendre en main DBSCAN sur ce type de données. Cherche les hyperparamètres `eps_y` (lignes) et `eps_x` (mots) **séparément**, l'un après l'autre, plutôt que conjointement ; voir `grille2d/` pour la version qui les cherche ensemble.

## Le principe

1. **Grille sur `eps_y` seul** : détection d'un plateau stable (une plage de valeurs donnant le même nombre de lignes pendant plusieurs itérations d'affilée), plutôt qu'un simple maximum, pour éviter de retenir une valeur isolée due au bruit.
2. **Grille sur `eps_x`** (à `eps_y` fixé) : maximisation de l'ARI, avec un classement en trois zones (sur-regroupement, optimale, sous-regroupement) selon le nombre de mots obtenu.

## Limite structurelle, documentée

`eps_x` est cherché à `eps_y` fixé : si le plateau retenu pour `eps_y` est mauvais, la recherche de `eps_x` en hérite sans retour en arrière possible. C'est précisément cette limite qui a motivé l'écriture de `grille_2d.py`.

## Utilisation

```bash
python3 grille_1d.py <fichier_json> <minS_y> <minS_x>
```

Détail complet (pseudo-code, points critiques) dans le guide développeur.
