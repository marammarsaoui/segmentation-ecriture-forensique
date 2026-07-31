# Interface interactive de segmentation

Une interface Streamlit qui applique en temps réel la cascade DBSCAN sur un texte manuscrit, avec un retour visuel immédiat à chaque réglage.

<p align="center">
  <img src="local.jpeg" width="80%" alt="Résultat de la cascade DBSCAN avec vérité terrain">
</p>

Un seul fichier, `interface_eps.py`, autonome. Elle détecte automatiquement si le fichier chargé contient une vérité terrain (champ `BM`) et adapte tout son comportement en conséquence : cascade + ARI + historique 3D d'un côté, DBSCAN simple sans score de l'autre.

## En bref

- Switch automatique avec / sans vérité terrain
- Historique d'exploration en 3D, coloré par ARI ou par ordre chronologique
- Validation et réparation automatique des fichiers JSON tronqués
- Export sécurisé : détection avant écrasement, journal d'audit

## Utilisation

```bash
streamlit run interface_eps.py
```
Pour tester rapidement sans dossier de données personnel, utilisez le fichier d'exemple fourni : TXT_Jules_Verne.json, à charger via le mode « Upload JSON » de l'interface.
Le détail completarchitecture, logique du switch,, robustesse, pseudo-code annoté est dans [`Guide_interface.md`](./Guide_interface.md).
