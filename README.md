# Segmentation d'écriture manuscrite en ligne

Interface de classification non supervisée (DBSCAN) pour la segmentation d'écriture manuscrite en ligne, exploration du paramètre eps. Stage de M1 (M1 Mathématiques Appliquées) avec monsieur Vincent Brault au sein du **LJK (Laboratoire Jean Kuntzmann)** et madame Fanny Guillet à la **police scientifique de Lyon**.

**Site vitrine :** 

---

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `interface/` | Interface Streamlit (switch BM, cascade, historique 3D) |
| `grille1d/` | Recherche en grille 1D pour la calibration des paramètres DBSCAN |
| `grille2d/` | Recherche en grille 2D |
| `src/` | Scripts cœur : `create_bm`, `functions_comparison`, `plot_json_bm`, `plot_mot`, `test_functions` |
| `docs/` | Guides utilisateur (Windows/macOS/Linux), notes sur la dimension temporelle |
| `data/` | `rangement/`, `json_bm/`, `modified/`-fichiers JSON de tracés stylet par scripteur |

> Les dossiers `data/rangement/`, `data/BM200/` et `data/modified/` contiennent les tracés complets de scripteurs identifiés. 
## Données

Les dossiers de données bruts (`json_bm/`, `modified/`, `rangement/`) ne sont **pas inclus** dans ce dépôt : ils contiennent des données d'acquisition qui seront rendues disponibles dans un futur article publié par l'équipe de recherche à l'origine de ce projet.

Pour tester l'interface sans ces dossiers, un fichier d'exemple est mis à disposition directement dans `interface/` : il suffit de le charger via le mode « Upload JSON » de l'interface, sans avoir besoin d'accéder aux dossiers de données du dépôt.
---

## Installation

```bash
git clone https://github.com/marammarsaoui/segmentation-manuscrite.git
cd segmentation-manuscrite
pip install -r requirements.txt
```

## Lancer l'interface

```bash
streamlit run interface/interface_eps.py
```

## Documentation

Le guide utilisateur complet est dans [`docs/guide_utilisateur.md`](docs/guide_utilisateur.md), avec des variantes par système d'exploitation dans le même dossier.

##Méthode

Pipeline DBSCAN à deux passes :
1. **Passe 1 (Y)**: détection des lignes d'écriture
2. **Passe 2 (X)**: détection des mots à l'intérieur de chaque ligne
3. **Switch temporel** (`cascade_XN`): activé si l'ARI spatial descend sous 0,80

*Remarque :* Des guides complémentaires pourront être ajoutés dans certains sous-répertoires lorsque leur fonctionnement nécessite des explications plus détaillées. Dans les autres cas, les consignes générales de des READMEs s’appliquent.

---

