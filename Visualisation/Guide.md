# Guide — Lecture et tracé du JSON benchmarké

Ce guide détaille les sept scripts du dossier, dans l'ordre logique de leur rôle : du plus élémentaire (lecture d'un seul scripteur) au plus abouti (comparaison colorée par mot sur trois dossiers entiers).

## Les trois dossiers de données

| Dossier | Suffixe des fichiers | Organisation |
|---|---|---|
| `json_bm` | `_with_BM.json` | un fichier par scripteur, texte complet segmenté en mots |
| `modified` | `_with_BM.json` | un fichier par scripteur, texte complet segmenté en mots |
| `rangement/{mot}` | `_with_bm_{mot}.json` | un sous-dossier par mot benchmarké ; à l'intérieur, un fichier par scripteur, contenant uniquement son écriture de ce mot précis (pas un texte complet) |

La structure interne diffère en conséquence : `json_bm`/`modified` contiennent une liste de segments `{BM, Points}` (plusieurs mots par fichier), alors que `rangement/{mot}` contient directement une liste de traits pour un seul mot (`len(data)` = nombre de traits utilisés pour écrire ce mot précis par ce scripteur).

## Les scripts, dans l'ordre de leur rôle

### 1. `JSON_show_first_stroke_of_txt.py`: le plus élémentaire

**Portée** : un seul fichier, un seul scripteur (203).

Charge `203_with_BM.json`, prend `data[0]["Points"]`, trace une ligne noire continue, enregistre `trace_203.png`.

**Le bug fondateur.** `data[0]` ne désigne que le premier élément de la liste `data`, c'est-à-dire le premier segment annoté, pas l'ensemble du texte. Le commentaire laissé dans ce script (« la faute venait de ça ») marque le moment où cette confusion a été identifiée : le nombre de points affiché était trop petit parce que seul le premier trait du scripteur était lu, le reste du document ignoré silencieusement.

### 2. `counter.py`: comptage et vérification de structure

**Portée** : un seul scripteur (203), mais deux dossiers comparés.

Compte, pour le scripteur 203, le nombre d'occurrences de chaque label `BM` dans `json_bm`, ce qui donne le nombre de traits par mot et le nombre de mots distincts. Vérifie ensuite la structure du JSON en affichant les clés du premier élément (`data[0].keys()`), puis répète la même vérification sur le fichier correspondant dans `rangement/zest`, pour comparer les deux structures côte à côte.

C'est ce script qui a permis de confirmer que `rangement` n'organise pas ses données de la même façon que `json_bm`.

### 3. `lire_le_dossier_JSON_automatiquement.py`: comparaison et tracé simple

**Portée** : les trois dossiers entiers, tous les scripteurs qu'ils contiennent.

Liste les scripteurs présents dans chaque dossier à partir du nom de fichier, puis calcule les ensembles utiles par comparaison (communs, exclusifs à un dossier). Trace ensuite, pour chaque fichier de chaque dossier, une ligne noire unique. Gère deux structures possibles avec un `try/except`, mais hérite du bug du script précédent : pour les dossiers structurés en segments, seul le premier segment est tracé.

### 4. `lire_tous_les_textes.py`: comparaison et tracé coloré par mot

**Portée** : les trois dossiers entiers, tous les scripteurs.

Reprend l'objectif du script précédent, avec deux différences. La comparaison des dossiers est réécrite avec des listes plutôt qu'un dictionnaire. Surtout, le tracé boucle sur chaque segment du fichier et attribue une couleur hexadécimale aléatoire à chaque segment , ce qui corrige le bug des deux scripts précédents et répond à l'objectif recherché : visualiser chaque mot annoté dans une couleur distincte.

**Un bug d'affichage à corriger.** Dans le bloc de comparaison, deux lignes consécutives annoncent un décompte puis une liste qui ne correspondent pas l'un à l'autre :

```python
print(f"Seulement dans modified sont ({len(mod - bm)}) ")
print(f"Seulement dans modified     ({len(mod-bm)})  : {sorted(bm - mod, key=int)}")
```

La seconde ligne affiche un compte calculé sur `mod - bm`, mais liste ensuite les éléments de `bm - mod`, l'ensemble inverse.

### 5. `tester_rangement.py`: statistique du nombre de traits par mot

**Portée** : le dossier `rangement` dans son ensemble, tous mots et tous scripteurs confondus.

Contrairement aux scripts précédents, l'organisation de `rangement` n'est pas un dossier par scripteur mais un sous-dossier par **mot benchmarké** : chaque fichier à l'intérieur contient l'écriture d'un seul scripteur pour ce mot précis (pas un texte entier). Ce script liste ces sous-dossiers, et pour chacun, ouvre tous les fichiers de scripteurs qu'il contient pour compter le nombre de traits (`len(data)`) utilisés pour écrire ce mot. Affiche ensuite, pour chaque mot, le minimum et le maximum de traits observés à travers tous les scripteurs.

Sert à mesurer la variabilité du nombre de traits utilisés pour écrire un même mot d'un scripteur à l'autre, utile pour calibrer les hyperparamètres des scripts de segmentation automatique (`grille_1d.py`, `grille_2d.py`).

### 6. `tracer_parasites.py`: inspection des traits marqués ponctuation

**Portée** : un seul scripteur (203).

Isole et trace uniquement les segments marqués `BM == "P"` (la convention du dépôt pour la ponctuation), pour inspecter visuellement à quoi ressemblent ces traits avant de les filtrer du reste du pipeline.

**Ce que la figure révèle.** Les traits `"P"` ne sont pas concentrés à un seul endroit du texte : ils apparaissent dispersés à des positions variées, avec des tailles très différentes (certains ne sont qu'un point isolé, d'autres de courts traits courbes). Cette dispersion suggère que `BM == "P"` capture un mélange de vraie ponctuation et de bruit d'acquisition (micro-gestes du stylet, hésitations), pas seulement les signes de ponctuation attendus , à garder en tête avant de filtrer ces segments sans les avoir inspectés, puisque le filtre `BM != "P"` est utilisé systématiquement dans tous les scripts `grille_*.py`.

Détail technique : ce script est le seul du groupe à utiliser `plt.gca().invert_yaxis()` (nécessaire car l'axe Y de la tablette compte de haut en bas). Les autres scripts de tracé ne l'utilisent pas — à harmoniser si les figures doivent être comparées entre elles.

### 7. `tracer_txt.py` : tracé d'un résultat déjà segmenté

**Portée** : un seul fichier, déjà issu d'un clustering.

Trace tous les segments d'un fichier `203_with_clusters.json` (le nom suggère une sortie déjà segmentée, pas un fichier `_with_BM.json` d'origine), une couleur aléatoire par segment. Sert de vérification visuelle rapide sur un résultat de segmentation.

Le titre de la figure est désactivé dans le code, et une seconde ligne de sauvegarde vers un autre nom de fichier est aussi en commentaire — ce script semble avoir servi à comparer deux variantes de sortie en activant l'une ou l'autre ligne selon le besoin du moment.

## Récapitulatif

| Script | Portée | Statut |
|---|---|---|
| Lecture d'un scripteur | 1 fichier | bug initial découvert ici |
| Comptage et structure | 1 scripteur, 2 dossiers | diagnostic, toujours utile |
| Comparaison et tracé simple | 3 dossiers entiers | hérite du bug, tracé incomplet |
| Comparaison et tracé coloré | 3 dossiers entiers | version corrigée, retenue |
| Min/max par mot | dossier `rangement` entier | statistique complémentaire |
| Traits parasites | 1 scripteur | diagnostic sur le filtre ponctuation |
| Tracé d'un résultat clusterisé | 1 fichier | vérification visuelle ponctuelle |

## Points critiques et sensibles

- Le bug `data[0]["Points"]` affecte deux des sept scripts. Le tracé simple, s'il est encore utilisé, ne montre jamais que le premier segment d'un fichier structuré.
- Les fichiers de `rangement/{mot}` ne représentent pas un texte complet comme `json_bm`/`modified`, mais l'écriture d'un seul mot par un seul scripteur : toute réutilisation de ces fichiers doit tenir compte de cette différence de granularité (un mot, pas une phrase).
- Le bug d'affichage du script de comparaison colorée ne modifie pas le tracé final, mais peut induire en erreur si on se fie au terminal pour vérifier rapidement quels scripteurs sont communs ou non entre dossiers.
- Le filtre `BM != "P"`, utilisé partout ailleurs dans le dépôt pour exclure la ponctuation, mérite d'être revérifié à la lumière de ce que montre `tracer_parasites.py` : la dispersion des traits « P » suggère un mélange avec du bruit d'acquisition.

## Améliorations possibles

- Corriger le bug d'affichage identifié dans `lire_tous_les_textes.py`.
- Retirer ou corriger le tracé simple, incomplet pour les dossiers structurés, au profit systématique de la version colorée.
- Harmoniser l'inversion d'axe Y entre tous les scripts de tracé.
- Étudier plus précisément la composition des traits marqués `"P"` (vraie ponctuation vs bruit) avant de continuer à les filtrer systématiquement.
