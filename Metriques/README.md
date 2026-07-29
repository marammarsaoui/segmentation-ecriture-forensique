# metriques/

Comparaison de 4 métriques de distance (euclidienne, manhattan, chebyshev, minkowski p=3) pour la passe 2 de la cascade DBSCAN,celle qui détecte les mots à partir de l'espace (X, Y).

## Fichiers
- `metriques.py`: lance la comparaison des 4 métriques pour un scripteur donné

## Pourquoi comparer des métriques en (X, Y) et pas en X seul ?

En 1D (X seul), toutes les métriques de distance sont mathématiquement équivalentes,la distance entre deux points reste `|x2 - x1|`, quelle que soit la formule utilisée. C'est seulement en 2D (X, Y) que les différentes métriques divergent réellement et produisent des clusters différents :

| Métrique | Forme de cluster favorisée |
|------|
| `euclidean` |
| `manhattan` |
| `chebyshev` (L-∞) |
| `minkowski p=3` |

## Guide utilisateur

### Lancer
```bash
python3 cascade_metriques.py <fichier_json>
```

**Exemple :**
```bash
python3 cascade_metriques.py ../json_bm/243_with_BM.json
```

### Paramètre
| Paramètre | Rôle |
|---|---|
| `fichier_json` | Chemin du fichier `*_with_BM.json` du scripteur à analyser |

`min_samples` est fixé volontairement à `5` pour les deux passes (choix assumé, pas un paramètre exposé en ligne de commande).

### Ce que produit le script

Pour la meilleure métrique retenue (celle avec l'ARI le plus élevé) :
- `outputs/vis_BM_{scripteur}_{metrique}.png`: vérité terrain (BM réels)
- `outputs/vis_motID_{scripteur}_{metrique}.png`: mots détectés
- `outputs/vis_lignes_{scripteur}_{metrique}.png`: lignes détectées

Pour l'ensemble des 4 métriques :
- `outputs/metriques_{scripteur}.csv`: tableau comparatif (eps optimaux, nombre de mots, ARI par métrique)
- `outputs/metriques_{scripteur}.png`: courbe ARI vs `eps_x` pour chaque métrique, à `eps_y` optimal fixé

### Logique interne (résumé)

1. **Passe 1** : DBSCAN sur Y seul (toujours euclidienne, car en 1D) → détection des lignes
2. **Passe 2** : pour chaque métrique, DBSCAN sur (X, Y) → détection des mots, avec une recherche en grille grossière puis un zoom autour du meilleur point
3. Comparaison des 4 ARI obtenus, la meilleure métrique est retenue pour les figures de visualisation finales

### État connu du script

Pas de guide séparé pour ce script , tout le mode d'emploi est ici. Ce fichier n'a pas encore de guide PDF dans `docs/`, contrairement à `grille2d/`.
