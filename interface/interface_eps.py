"""
═══════════════════════════════════════════════════════════════════════
 CARTOUCHE:
 Fichier: interface_eps.py
 
#   Code écrit et développé par Maram
#   Merci de ne pas réutiliser sans autorisation

 Projet: Segmentation automatique de texte manuscrit en ligne
 Contexte: Stage M1 Mathématiques Appliquées
 Rôle: Interface interactive (Streamlit) de la cascade DBSCAN,
                pensée pour le réglage manuel exploratoire, à la différence
                de grille_1d.py / grille_2d.py / run_all_scripteurs.py qui
                font une recherche automatique en traitement par lot.

 ENTRÉES ATTENDUES:
 Un fichier JSON de tracé manuscrit, dans l'un des deux formats suivants :
   Format A (with_BM): liste de segments {"BM": ..., "Points": [...]},
                         chaque point ayant au moins X, Y, N, Tip.
   Format B (brut): liste plate de points {"X", "Y", "Segment", ...},
                         sans annotation de vérité terrain.
 La différence entre les deux est détectée automatiquement (voir _preparer,
 étape 1) et pilote tout le comportement de l'interface ensuite (le switch
 avec BM / sans BM, étapes 4 et 5).

 SORTIES PRODUITES:
   outputs/scripteur_{N}/resultats_{N}.csv : résultats chiffrés
   outputs/scripteur_{N}/export_ey..._ex....png : figures (mode avec BM)
   outputs/scripteur_{N}/vis_dbscan_{N}.png: figure (mode sans BM)
   outputs/scripteur_{N}/{N}_with_BM.json :nouvelle vérité terrain
                                                    reconstruite à partir
                                                    des MotID trouvés
   outputs/scripteur_{N}/trace3D_{N}.csv : historique d'exploration
   outputs/journal.csv :journal d'audit global, jamais écrasé, trace tous les exports de tous les scripteurs

 DÉPENDANCES: 
 streamlit, numpy, pandas, matplotlib, scikit-learn (DBSCAN,
 adjusted_rand_score). Aucune dépendance à un autre fichier du dépôt :
 ce script est autonome.

 UTILISATION : 
   streamlit run interface_eps.py

 STRUCTURE DU FICHIER (dans l'ordre d'exécution) : 
   Étape 0a : robustesse - validation JSON, réparation, journal d'audit
   Étape 0b : source de données - fichier local ou upload
   Étape 0c : validation et réparation, appliquée à la source choisie
   Étape 1  : _preparer() - détecte la présence de BM (LE switch)
   Étape 2  : fonctions communes d'affichage (couleurs, résumé, figures)
   Étape 3  : préparation du dossier de sortie, commun aux deux modes
   Étape 4  : MODE AVEC BM - cascade eps_y/eps_x, ARI, historique 3D
   Étape 5  : MODE SANS BM - DBSCAN à plat, un seul eps, pas d'ARI

 POINTS D'ATTENTION POUR UN REPRENEUR :
 - La variable a_bm (calculée une seule fois, étape 1) est LA variable
   qui décide de tout le reste. Si elle est mal calculée, tout le
   comportement de l'interface en aval est faussé silencieusement.
 - Les labels DBSCAN internes commencent à 0 (convention scikit-learn),
   mais tout ce qui est affiché à l'utilisateur (numéros de cluster,
   nouveau JSON BM exporté) est décalé à +1 pour rester lisible côté
   humain. Ne pas mélanger les deux conventions si le code est étendu.
 - Le mode "avec BM" n'a pas de switch temporel (contrairement à
   grille_2d.py / run_all_scripteurs.py qui basculent sur (X, Time_MS
   normalisé) si l'ARI reste sous 0.80). Le réglage des curseurs eps_y et
   eps_x reste entièrement manuel dans cette interface : aucune recherche
   automatique d'hyperparamètres n'y est intégrée, contrairement aux
   scripts de traitement par lot du dépôt.
 - Le bouton "Exporter nouveau JSON BM" n'existe que dans le mode sans
   BM : en mode avec BM, une vérité terrain existe déjà, ce bouton n'a
   donc pas d'utilité dans ce contexte.
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import json
import re
import glob
import hashlib
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Interface DBSCAN")
st.title("Interface DBSCAN")

SEUIL_TAILLE_MO = 5
JOURNAL_CSV = os.path.join("outputs", "journal.csv")


########

# Étape 0a : Robustesse - validation, réparation, avertissements, journal

def reparer_json_tronque(texte):
    """
    Tente de récupérer un JSON tronqué en coupant au dernier élément complet
    et en refermant les structures ouvertes. Essaie plusieurs combinaisons
    de fermeture, en partant de la fin du texte vers le début.
    Renvoie (data, position_coupure) ou (None, None).
    """
    positions = [m.end() for m in re.finditer(r"\}", texte)]
    candidats_suffixes = ["}]}]", "]}]", "}]}", "]}", "}]", "]"]
    for pos in reversed(positions):
        prefixe = texte[:pos]
        for suffixe in candidats_suffixes:
            try:
                data = json.loads(prefixe + suffixe)
                return data, pos
            except Exception:
                continue
    return None, None


def valider_et_reparer(contenu_bytes):
    """
    Parse le JSON, tente une réparation par troncature si nécessaire.

    Trois issues possibles :
      1. Parsing réussi du premier coup -> (data, None, None).
      2. Échec proche de la fin du fichier -> reparer_json_tronque() est
         appelée ; si elle récupère quelque chose, on renvoie les données
         récupérées avec un avertissement non bloquant (avertissement
         renseigné, erreur = None), pour que l'appelant puisse continuer
         à travailler avec le fichier tout en étant prévenu de la perte.
      3. Échec non réparable (erreur ailleurs qu'en fin de fichier, ou
         réparation infructueuse) -> (None, message_erreur, None), avec un
         extrait de texte autour de la position d'erreur pour diagnostic.
    """
    try:
        data = json.loads(contenu_bytes)
        return data, None, None
    except json.JSONDecodeError as e:
        texte = contenu_bytes.decode("utf-8", errors="replace") if isinstance(contenu_bytes, bytes) else contenu_bytes
        taille_totale = len(texte)

        if e.pos >= taille_totale - 2:
            data_reparee, position_coupure = reparer_json_tronque(texte)
            if data_reparee is not None:
                octets_perdus = taille_totale - position_coupure
                avertissement = (
                    f"Le fichier était tronqué en fin d'écriture. Une réparation "
                    f"automatique a été appliquée : les {octets_perdus} derniers "
                    f"caractères ont été écartés pour ne garder que les éléments "
                    f"complets. Les données situées après ce point sont perdues."
                )
                return data_reparee, None, avertissement

        debut = max(0, e.pos - 60)
        fin = min(taille_totale, e.pos + 60)
        extrait = texte[debut:fin].replace("\n", " ")
        indication = ""
        if e.pos >= taille_totale - 2:
            indication = (" La position correspond à la toute fin du fichier : "
                         "le JSON semble incomplet, probablement une coupure "
                         "lors de l'export ou du transfert du fichier.")
        return None, (f"Le fichier n'est pas un JSON valide (erreur à la position "
                      f"{e.pos} sur {taille_totale}).{indication}\n"
                      f"Contexte autour de l'erreur : ...{extrait}..."), None
    except Exception as e:
        return None, f"Erreur inattendue lors de la lecture du fichier : {e}", None


def avertir_si_volumineux(taille_octets):
    """
    Affiche un simple avertissement (non bloquant) si la taille dépasse
    SEUIL_TAILLE_MO. N'empêche jamais le traitement de continuer : c'est
    une information pour l'utilisateur, pas une validation.
    """
    taille_mo = taille_octets / (1024 * 1024)
    if taille_mo > SEUIL_TAILLE_MO:
        st.sidebar.warning(f"Fichier volumineux ({taille_mo:.1f} Mo) : le traitement "
                           f"peut prendre plusieurs secondes.")


def hash_contenu(contenu_bytes):
    """
    Empreinte courte (10 caractères hexadécimaux) du contenu binaire d'un
    fichier. Sert à distinguer deux fichiers de même nom mais de contenu
    différent (par exemple deux uploads successifs du même scripteur avec
    des données mises à jour) : cette empreinte est stockée dans le
    journal d'audit pour lever ce doute a posteriori.
    """
    return hashlib.md5(contenu_bytes).hexdigest()[:10]


def lister_existants(dossier):
    """
    Vérifie si un dossier de sortie contient déjà des fichiers, et si oui,
    renvoie la date de modification du plus récent d'entre eux.
    Renvoie (existe: bool, date_texte: str ou None). Utilisée par
    exporter_avec_confirmation() pour décider s'il faut demander
    confirmation avant d'écrire.
    """
    if not os.path.isdir(dossier):
        return False, None
    fichiers = glob.glob(os.path.join(dossier, "*"))
    if not fichiers:
        return False, None
    plus_recent = max(fichiers, key=os.path.getmtime)
    date_texte = datetime.fromtimestamp(os.path.getmtime(plus_recent)).strftime(
        "%d/%m/%Y à %Hh%M")
    return True, date_texte


def horodatage_court():
    """Horodatage compact (AAAAMMJJ_HHMMSS), utilisé comme suffixe de
    nom de fichier quand l'utilisateur choisit de créer une nouvelle
    version plutôt que de remplacer un export existant."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def consigner_journal(fichier_source, scripteur, mode, params, resultats, hash_source):
    """
    Ajoute une ligne au journal d'audit global (outputs/journal.csv),
    toujours en ajout (mode="a"), jamais en écrasement : ce fichier
    accumule l'historique de tous les exports CSV réalisés, pour tous les
    scripteurs, même si les exports individuels par scripteur sont
    remplacés au fil du temps. params et resultats sont des dictionnaires
    dépliés (**) directement dans la ligne, ce qui permet à ce journal
    d'accueillir des colonnes différentes selon le mode (avec_BM / brut)
    sans schéma fixe à maintenir.
    """
    os.makedirs("outputs", exist_ok=True)
    ligne = pd.DataFrame([{
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fichier_source": fichier_source,
        "hash_contenu": hash_source,
        "scripteur": scripteur,
        "mode": mode,
        **params,
        **resultats,
    }])
    if os.path.exists(JOURNAL_CSV):
        ligne.to_csv(JOURNAL_CSV, mode="a", header=False, index=False)
    else:
        ligne.to_csv(JOURNAL_CSV, mode="w", header=True, index=False)


def exporter_avec_confirmation(cle_state, dossier, faire_export, label_bouton):
    """
    Enveloppe générique autour de n'importe quelle fonction d'export, pour
    lui ajouter la détection d'écrasement et un dialogue de confirmation.

    Pourquoi passer par st.session_state plutôt qu'un simple if/else :
    Streamlit ré-exécute tout le script à chaque interaction (chaque clic,
    chaque glissement de curseur repart de la ligne 1). Un bouton cliqué
    ne "reste" donc pas cliqué au tour suivant. Pour afficher le choix
    remplacer/nouvelle version PUIS attendre un second clic de
    confirmation, il faut mémoriser "on est en attente de confirmation
    pour cet export précis" quelque part qui survit d'un tour à l'autre :
    c'est le rôle de st.session_state, une mémoire persistante entre les
    ré-exécutions successives du script.
    """
    if st.sidebar.button(label_bouton, key=f"bouton_{cle_state}"):
        st.session_state[f"pending_{cle_state}"] = True

    if st.session_state.get(f"pending_{cle_state}"):
        existe, date_texte = lister_existants(dossier)
        if not existe:
            chemins = faire_export("")
            st.sidebar.success("Export réalisé :\n" + "\n".join(chemins))
            st.session_state[f"pending_{cle_state}"] = False
        else:
            st.sidebar.warning(f"Des résultats existent déjà pour ce scripteur, "
                               f"exportés le {date_texte}.")
            choix = st.sidebar.radio(
                "Que faire ?",
                ["Remplacer les fichiers existants", "Créer une nouvelle version horodatée"],
                key=f"choix_{cle_state}")
            if st.sidebar.button("Confirmer", key=f"confirm_{cle_state}"):
                suffixe = "" if choix.startswith("Remplacer") else f"_{horodatage_court()}"
                chemins = faire_export(suffixe)
                st.sidebar.success("Export réalisé :\n" + "\n".join(chemins))
                st.session_state[f"pending_{cle_state}"] = False


########

# Étape 0b : Source de données
#
# Les deux chemins (upload / local) convergent volontairement vers les
# trois mêmes variables (contenu_brut, scripteur, nom_source), pour que
# tout le code qui suit (étape 0c et au-delà) n'ait jamais besoin de
# savoir d'où vient réellement le fichier. C'est ce qui permet au reste
# du script d'être écrit une seule fois, sans dupliquer la logique de
# validation / préparation / affichage pour chaque source séparément.

st.sidebar.markdown("## Source de données")
source = st.sidebar.radio("Mode", ["Fichier local (json_bm/)", "Upload JSON"])

if source == "Upload JSON":
    fichier_upload = st.sidebar.file_uploader(
        "Charger un fichier JSON (_with_BM.json)", type=["json"])
    if fichier_upload is None:
        st.info("Chargez un fichier JSON dans la barre latérale pour commencer.")
        st.stop()
    scripteur = fichier_upload.name.replace("_with_BM.json", "").replace(".json", "")
    avertir_si_volumineux(fichier_upload.size)
    contenu_brut = fichier_upload.read()
    nom_source = fichier_upload.name
else:
    # Cherche d'abord dans ../json_bm/ (organisation standard du dépôt),
    # puis dans le dossier courant en repli, pour rester utilisable même
    # si l'interface est lancée depuis un autre emplacement.
    json_files = sorted(glob.glob("../json_bm/*_with_BM.json"))
    if not json_files:
        json_files = sorted(glob.glob("*_with_BM.json"))
    if not json_files:
        st.error("Aucun fichier *_with_BM.json trouvé dans ../json_bm/")
        st.stop()
    fichier_local = st.sidebar.selectbox(
        "Scripteur", json_files,
        format_func=lambda x: os.path.basename(x).replace("_with_BM.json", ""))
    scripteur = os.path.basename(fichier_local).replace("_with_BM.json", "")
    with open(fichier_local, "rb") as f:
        contenu_brut = f.read()
    avertir_si_volumineux(len(contenu_brut))
    nom_source = os.path.basename(fichier_local)

########

# Étape 0c : validation et réparation, communes aux deux sources
#
# À partir d'ici, on ne raisonne plus qu'en termes de contenu_brut (des
# octets), quelle que soit la source. La validation/réparation et le
# calcul de l'empreinte se font donc une seule fois, pour les deux
# chemins possibles de l'étape 0b.

data_brute, erreur_json, avertissement_json = valider_et_reparer(contenu_brut)
if erreur_json:
    st.error(erreur_json)
    st.stop()
if avertissement_json:
    st.warning(avertissement_json)

# Calculée après la validation/réparation, sur le contenu tel qu'il a été
# effectivement lu (pas sur une éventuelle version réparée) : deux
# uploads du même fichier produisent la même empreinte, ce qui permet de
# les reconnaître comme identiques dans le journal d'audit.
empreinte_source = hash_contenu(contenu_brut if isinstance(contenu_brut, bytes) else contenu_brut.encode())


########

# Étape 1 : Préparation - détecte la présence de BM (le switch)

def _preparer(data):
    """
    Prépare les données depuis un JSON chargé.
    Gère deux structures :
      Format A (with_BM) : liste de segments {BM, Points:[{X,Y,N,...}]}
      Format B (brut)    : liste plate de points {Segment, N, X, Y, ...}
    a_bm indique si une vérité terrain BM est réellement disponible :
    c'est cette variable qui pilote le switch entre les deux modes
    de l'interface, indépendamment de la structure du fichier.
    """
    premier = data[0] if data else {}
    format_plat = "X" in premier and "Points" not in premier

    if format_plat:
        from collections import defaultdict
        groupes = defaultdict(list)
        for p in data:
            seg_id = p.get("Segment", 0)
            groupes[seg_id].append(p)

        data_f = []
        for seg_id in sorted(groupes.keys()):
            pts_seg = groupes[seg_id]
            pts_tip = [p for p in pts_seg if p.get("Tip", 1) == 1]
            if not pts_tip:
                pts_tip = pts_seg
            data_f.append({"BM": None, "Points": pts_tip})
        a_bm = False
        n_bm = 0
    else:
        a_bm  = any("BM" in seg and seg["BM"] != "P" for seg in data)
        data_f = [seg for seg in data if seg.get("BM", "P") != "P"] if a_bm else list(data)
        n_bm  = len(set(seg["BM"] for seg in data_f)) if a_bm else 0

    pts = []
    for seg in data_f:
        for p in seg.get("Points", []):
            x = p.get("X", p.get("x"))
            y = p.get("Y", p.get("y"))
            n = p.get("N", p.get("n", 0))
            if x is not None and y is not None:
                pts.append([float(x), float(y), int(n)])

    if not pts:
        raise ValueError(
            f"Aucun point X/Y extrait. "
            f"Vérifiez la structure du JSON ({len(data_f)} segments).")

    coords   = np.array([[p[0], p[1]] for p in pts])
    coords_Y = coords[:, 1].reshape(-1, 1)
    N_liste  = [p[2] for p in pts]

    return {
        "data_filtre": data_f, "n_bm_reels": n_bm, "coords": coords,
        "coords_Y": coords_Y, "N_liste": N_liste, "a_bm": a_bm,
    }

D = _preparer(data_brute)
data_filtre = D["data_filtre"]
n_bm_reels  = D["n_bm_reels"]
coords      = D["coords"]
coords_Y    = D["coords_Y"]
N_liste     = D["N_liste"]
a_bm        = D["a_bm"]

# ─────────────────────────────────────────────────────────────────────
# POINT DE BIFURCATION PRINCIPAL DU FICHIER.
# Tout ce qui suit (étapes 4 et 5, la quasi-totalité du reste du script)
# est conditionné par cette seule variable a_bm, calculée une fois pour
# toutes dans _preparer(). Elle n'est jamais recalculée ni modifiée
# ensuite : c'est la source de vérité unique pour savoir dans quel mode
# on se trouve. Si un bug de switch apparaît (mauvais mode affiché),
# c'est dans _preparer() qu'il faut chercher en premier, pas ici.
# ─────────────────────────────────────────────────────────────────────
if a_bm:
    if source != "Upload JSON":
        st.sidebar.markdown(f"**{len(data_filtre)} strokes · {len(coords)} points · "
                            f"{n_bm_reels} BM réels**")
    st.info("Champ BM détecté → mode cascade avec vérité terrain et ARI.")
else:
    if source != "Upload JSON":
        st.sidebar.markdown(f"**{len(data_filtre)} strokes · {len(coords)} points · pas de BM**")
    st.info("Pas de champ BM détecté → mode simplifié, DBSCAN 2D sans vérité terrain.")


########

# Étape 2 : Fonctions communes d'affichage

def gen_couleurs(vals, seed):
    """
    Associe une couleur hexadécimale aléatoire à chaque valeur distincte
    de vals (un cluster, un MotID, un LigneID...). Le bruit (-1) reçoit
    toujours le même gris neutre, jamais une couleur aléatoire, pour
    rester visuellement distinct des vrais groupes sur toutes les
    figures. La graine (seed) fixe garantit que les mêmes valeurs
    donnent toujours les mêmes couleurs d'un rendu à l'autre (sans
    seed fixe, les couleurs changeraient à chaque rechargement de page,
    rendant deux figures incomparables entre elles). sorted(..., key=str)
    assure un ordre de tirage déterministe même si vals mélange des
    types (int, str) selon le mode.
    """
    random.seed(seed)
    return {v: "#aaaaaa" if v == -1 else
            "#{:06x}".format(random.randint(0, 0xFFFFFF))
            for v in sorted(vals, key=str)}


def resumer_clusters(labels, seuil_petit=5):
    compte = Counter(l for l in labels if l != -1)
    n_bruit = int(np.sum(labels == -1))
    total = len(labels)
    petits = sorted(c for c, n in compte.items() if n < seuil_petit)
    texte = (f"{len(compte)} groupe(s) détecté(s). "
             f"{n_bruit} point(s) classé(s) bruit ({n_bruit / total:.0%}). ")
    if petits:
        # affichage humain : les numéros commencent à 1, pas à 0
        petits_affiches = [c + 1 for c in petits]
        texte += (f"{len(petits)} groupe(s) anormalement petit(s) "
                  f"(< {seuil_petit} points, numéros {petits_affiches}).")
    else:
        texte += "Aucun groupe anormalement petit détecté."
    # colonne "cluster" décalée de +1 pour l'affichage ; les calculs
    # internes (labels DBSCAN bruts) restent inchangés
    df_tailles = (pd.DataFrame({"cluster": [c + 1 for c in compte.keys()],
                                "n_points": list(compte.values())})
                  .sort_values("cluster").reset_index(drop=True))
    return texte, df_tailles


def calculer_taille_figure(coords, largeur_cible=12, min_hauteur=3, max_hauteur=18):
    """
    Calcule une taille de figure dont le rapport largeur/hauteur reflète
    l'étendue réelle des coordonnées X et Y du fichier, plutôt qu'une
    taille fixe arbitraire. Une largeur cible est fixée, la hauteur en
    découle proportionnellement, avec des bornes pour éviter une figure
    dégénérée si le texte est très large et peu haut, ou l'inverse.
    """
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    if x_range <= 0:
        x_range = 1.0
    hauteur = largeur_cible * (y_range / x_range)
    hauteur = max(min_hauteur, min(max_hauteur, hauteur))
    return (largeur_cible, hauteur)


def plot_labels_simple(coords, labels, titre):
    """Affichage pour le mode sans BM : numéro au-dessus du cluster (affiché
    à partir de 1, pas 0), bruit en croix rouge."""
    fig, ax = plt.subplots(figsize=calculer_taille_figure(coords))
    for lab in sorted(set(labels)):
        masque = labels == lab
        xs, ys = coords[masque, 0], coords[masque, 1]
        if lab == -1:
            ax.scatter(xs, ys, s=25, marker="x", color="#E24B4A", linewidths=1.2)
            continue
        ax.scatter(xs, ys, s=4)
        # Numéro placé au-dessus du cluster (au-dessus du point le plus haut),
        # centré horizontalement. Affiché à partir de 1 (pas 0) pour la
        # lecture humaine ; le label DBSCAN sous-jacent reste inchangé.
        marge = (ys.max() - ys.min()) * 0.15 + 2
        ax.annotate(str(lab + 1), (xs.mean(), ys.max() + marge), fontsize=8,
                   ha="center", va="bottom",
                   bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                            ec="gray", alpha=0.85))
    ax.axis("equal"); ax.set_title(titre); ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.tight_layout()
    return fig


########

# Étape 3 : Export (commun, mais les fonctions d'export diffèrent selon le mode)
#
# Le dossier de sortie est créé ici, avant même de savoir dans quel mode
# (avec BM / sans BM) on va basculer juste après : les deux modes
# écrivent dans ce même dossier, donc autant le préparer une seule fois
# plutôt que de dupliquer cette ligne dans chaque branche.

out = f"outputs/scripteur_{scripteur}"
os.makedirs(out, exist_ok=True)


########

# Étape 4 : MODE AVEC BM - cascade, eps_y/eps_x, ARI, historique 3D

if a_bm:

    def copie(data_f):
        """
        Copie profonde manuelle de la liste de segments. Nécessaire car
        cascade() enrichit chaque point avec LigneID/MotID à chaque appel ;
        sans repartir d'une copie propre, ces annotations s'accumuleraient
        entre deux appels successifs (par exemple deux essais de curseurs
        différents dans la même session) et corromprait les résultats.
        """
        return [{"BM": s.get("BM", "?"), "Points": [dict(p) for p in s["Points"]]}
                for s in data_f]

    def cascade(eps_y, eps_x, mSy, mSx, coords, coords_Y, N_liste, data_filtre,
                a_bm, n_bm_reels):
        """
        Cascade DBSCAN en deux passes :
          Passe 1 (sur coords_Y) : sépare les lignes d'écriture.
          Passe 2 (sur X, à l'intérieur de chaque ligne) : sépare les mots.
        Les labels de mots obtenus ligne par ligne sont décalés par
        mot_global pour rester uniques sur tout le document (sans ce
        décalage, le mot 0 de la ligne 1 et le mot 0 de la ligne 2
        porteraient le même identifiant et seraient confondus).

        Le résultat est ensuite agrégé au niveau du stroke (pas du point)
        par vote majoritaire : un stroke peut chevaucher deux clusters à
        sa frontière, on lui attribue le label le plus fréquent parmi ses
        propres points.

        L'ARI est calculé en excluant les strokes classés bruit (MotID
        == -1), pour ne pas fausser la comparaison avec des points qui
        n'appartiennent à aucun cluster prédit.

        Renvoie (df, n_lignes, n_mots, ari).
        """
        labels_ligne   = DBSCAN(eps=eps_y, min_samples=mSy).fit_predict(coords_Y)
        lignes_uniques = sorted(set(labels_ligne) - {-1})
        labels_mot     = np.full(len(coords), -1, dtype=int)
        mot_global     = 0
        for ligne_id in lignes_uniques:
            idx          = np.where(labels_ligne == ligne_id)[0]
            coords_X     = coords[idx, 0].reshape(-1, 1)
            labels_local = DBSCAN(eps=eps_x, min_samples=mSx).fit_predict(coords_X)
            mots_locaux  = sorted(set(labels_local) - {-1})
            for i, idx_pt in enumerate(idx):
                if labels_local[i] != -1:
                    labels_mot[idx_pt] = mot_global + labels_local[i]
            if mots_locaux:
                mot_global += max(mots_locaux) + 1

        # Jointure via l'identifiant N : les tableaux numpy (labels_ligne,
        # labels_mot) sont indexés par position, mais les points d'origine
        # sont identifiés par leur N ; ces deux dictionnaires font le pont.
        N_to_ligne = {N_liste[i]: int(labels_ligne[i]) for i in range(len(N_liste))}
        N_to_mot   = {N_liste[i]: int(labels_mot[i])   for i in range(len(N_liste))}
        data_c = copie(data_filtre)
        for seg in data_c:
            for p in seg["Points"]:
                p["LigneID"] = N_to_ligne.get(p["N"], -1)
                p["MotID"]   = N_to_mot.get(p["N"], -1)

        df = pd.DataFrame(data_c)
        df["LigneID"] = df["Points"].apply(
            lambda pts: Counter(p["LigneID"] for p in pts).most_common(1)[0][0])
        df["MotID"] = df["Points"].apply(
            lambda pts: Counter(p["MotID"] for p in pts).most_common(1)[0][0])

        df_valid = df[df["MotID"] != -1]
        ari = adjusted_rand_score(
            df_valid["BM"].tolist(), df_valid["MotID"].tolist()
        ) if len(df_valid) > 1 else 0.0
        n_mots   = df["MotID"].nunique() - (1 if -1 in df["MotID"].values else 0)
        n_lignes = len(lignes_uniques)
        return df, n_lignes, n_mots, ari

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Paramètres DBSCAN")
    eps_y  = st.sidebar.slider("eps_y",  0.05, 1.00, 0.35, step=0.005)
    eps_x  = st.sidebar.slider("eps_x",  0.10, 3.00, 1.30, step=0.05)
    minS_y = st.sidebar.slider("minS_y", 1, 10, 3, step=1)
    minS_x = st.sidebar.slider("minS_x", 1, 10, 3, step=1)
    st.sidebar.markdown("---")
    # Le sélecteur MotID/LigneID a été retiré : la colonne du milieu (col2)
    # affiche déjà les lignes en permanence, donc choisir "LigneID" ici
    # aurait affiché deux fois la même information. La troisième colonne
    # se consacre désormais toujours à MotID.
    mode_affichage = "MotID"

    df, n_lignes, n_mots, ari = cascade(
        eps_y, eps_x, minS_y, minS_x, coords, coords_Y, N_liste, data_filtre,
        a_bm, n_bm_reels)

    if n_mots < n_bm_reels * 0.8:
        zone = "SUR-REGROUPEMENT"
    elif n_mots > n_bm_reels * 1.2:
        zone = "SOUS-REGROUPEMENT"
    else:
        zone = "ZONE OPTIMALE"

    if "historique" not in st.session_state:
        st.session_state.historique = []
    point = {"eps_y": eps_y, "eps_x": eps_x, "ARI": round(ari, 4),
             "n_mots": n_mots, "n_lignes": n_lignes, "zone": zone}
    if not st.session_state.historique or st.session_state.historique[-1] != point:
        st.session_state.historique.append(point)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ARI", f"{ari:.3f}")
    c2.metric("Mots détectés", f"{n_mots} / {n_bm_reels}")
    c3.metric("Lignes", n_lignes)
    c4.metric("Zone", zone)
    c5.metric("Points explorés", len(st.session_state.historique))

    def make_fig(df_plot, col_name, seed, titre, afficher_numeros=False):
        """
        Trace un tracé manuscrit coloré par la colonne col_name (BM,
        MotID ou LigneID selon l'appelant), un stroke à la fois, chaque
        stroke gardant l'ordre de ses points d'origine (donc le tracé
        suit fidèlement le geste d'écriture, contrairement à un simple
        nuage de points). Utilisée pour les trois figures du mode avec BM
        (vérité terrain, lignes, mots).

        Si afficher_numeros=True, un numéro de groupe (à partir de 1) est
        affiché au-dessus de chaque groupe. Ajouté suite à un retour
        utilisateur : le nombre annoncé dans les indicateurs (par exemple
        "Lignes : 16") pouvait sembler ne pas correspondre à ce qui était
        perçu visuellement (par exemple 6 rangées visibles), parce que
        deux couleurs tirées au hasard peuvent se ressembler assez pour
        que l'œil confonde deux groupes distincts en une seule rangée
        (souvent une même ligne coupée en plusieurs morceaux par les
        jambages des lettres). La numérotation permet de vérifier
        directement sur la figure que le compte affiché est correct.
        """
        colors = gen_couleurs(set(df_plot[col_name].unique()), seed)
        fig, ax = plt.subplots(figsize=(7, 5))
        points_par_groupe = {}
        for _, row in df_plot.iterrows():
            pts = row["Points"]
            Xp = [p["X"] for p in pts]; Yp = [p["Y"] for p in pts]
            col = colors.get(row[col_name], "#333333")
            ax.plot(Xp, Yp, color=col, linewidth=1.5)
            ax.scatter(Xp, Yp, color=col, s=3, alpha=0.5)
            if afficher_numeros:
                lab = row[col_name]
                points_par_groupe.setdefault(lab, {"x": [], "y": []})
                points_par_groupe[lab]["x"].extend(Xp)
                points_par_groupe[lab]["y"].extend(Yp)

        if afficher_numeros:
            for lab, pts_groupe in points_par_groupe.items():
                if lab == -1:
                    continue
                xs = np.array(pts_groupe["x"]); ys = np.array(pts_groupe["y"])
                marge = (ys.max() - ys.min()) * 0.15 + 1
                ax.annotate(str(int(lab) + 1), (xs.mean(), ys.max() + marge),
                          fontsize=7, ha="center", va="bottom",
                          bbox=dict(boxstyle="circle,pad=0.12", fc="white",
                                   ec="gray", alpha=0.85))

        ax.set_title(titre, fontsize=10); ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.tight_layout()
        return fig

    col_name = "MotID" if mode_affichage == "MotID" else "LigneID"
    seed     = 99 if mode_affichage == "MotID" else 7
    df_bm    = pd.DataFrame(data_filtre)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_bm = make_fig(df_bm, "BM", 42, f"Vérité terrain - BM réels ({scripteur})")
        st.pyplot(fig_bm); plt.close(fig_bm)
    with col2:
        fig_lig = make_fig(df, "LigneID", 7,
                           f"Lignes détectées - {n_lignes} lignes\neps_y={eps_y:.3f}",
                           afficher_numeros=True)
        st.pyplot(fig_lig); plt.close(fig_lig)
    with col3:
        titre_res = f"{mode_affichage} - {n_mots}/{n_bm_reels} | ARI={ari:.3f} [{zone}]"
        fig_res = make_fig(df, col_name, seed, titre_res, afficher_numeros=True)
        st.pyplot(fig_res); plt.close(fig_res)

    if len(st.session_state.historique) >= 2:
        st.markdown("---")
        st.markdown("### Trace 3D - Exploration (eps_y, eps_x, ARI)")
        df_hist = pd.DataFrame(st.session_state.historique)
        tab1, tab2 = st.tabs(["Nuage cumulé", "Surface session"])

        with tab1:
            # Comme la figure est une image statique (pas de rotation à la
            # souris), on expose l'angle de vue comme deux curseurs.
            c_elev, c_azim, c_temps = st.columns([1, 1, 2])
            with c_elev:
                elev = st.slider("Inclinaison (élévation)", 0, 90, 25, key="elev_nuage")
            with c_azim:
                azim = st.slider("Rotation (azimut)", 0, 360, -60, key="azim_nuage")
            with c_temps:
                colorer_temps = st.checkbox(
                    "Colorer par ordre chronologique plutôt que par ARI",
                    key="colorer_temps",
                    help="Ajoute la dimension temporelle de l'exploration : "
                         "le paramètre de couleur devient l'ordre dans lequel "
                         "les réglages ont été essayés pendant la session, "
                         "plutôt que la valeur de l'ARI.")

            fig3d = plt.figure(figsize=(10, 6))
            ax3d  = fig3d.add_subplot(111, projection="3d")

            if colorer_temps:
                # Le paramètre temps ici est l'ordre chronologique des essais
                # (index dans l'historique de session), pas un temps absolu.
                couleurs = np.arange(len(df_hist))
                sc = ax3d.scatter(df_hist["eps_x"], df_hist["eps_y"], df_hist["ARI"],
                                  c=couleurs, cmap="viridis", s=40, alpha=0.8)
                plt.colorbar(sc, ax=ax3d, shrink=0.5,
                           label="Ordre d'exploration (violet = ancien, jaune = récent)")
            else:
                sc = ax3d.scatter(df_hist["eps_x"], df_hist["eps_y"], df_hist["ARI"],
                                  c=df_hist["ARI"], cmap="RdYlGn", vmin=0, vmax=1, s=40, alpha=0.8)
                plt.colorbar(sc, ax=ax3d, shrink=0.5, label="ARI")

            ax3d.scatter([eps_x], [eps_y], [ari], color="red", s=120, marker="*",
                        zorder=10, label=f"Courant ARI={ari:.3f}")
            ax3d.view_init(elev=elev, azim=azim)
            ax3d.set_xlabel("eps_x"); ax3d.set_ylabel("eps_y"); ax3d.set_zlabel("ARI")
            ax3d.set_zlim(0, 1.05)
            ax3d.set_title(f"Nuage cumulé - {len(df_hist)} pts - {scripteur}", fontsize=10)
            ax3d.legend(fontsize=8)
            fig3d.tight_layout(); st.pyplot(fig3d); plt.close(fig3d)

        with tab2:
            if len(df_hist) >= 6:
                try:
                    pivot = df_hist.pivot_table(index="eps_y", columns="eps_x",
                                                values="ARI", aggfunc="max")
                    pivot = pivot.interpolate(axis=1).interpolate(axis=0)
                    EX_m, EY_m = np.meshgrid(pivot.columns.values, pivot.index.values)
                    fig3ds = plt.figure(figsize=(10, 6))
                    ax3ds  = fig3ds.add_subplot(111, projection="3d")
                    surf = ax3ds.plot_surface(EX_m, EY_m, pivot.values, cmap="RdYlGn",
                                             vmin=0, vmax=1, alpha=0.80, edgecolor="none")
                    best = df_hist.loc[df_hist["ARI"].idxmax()]
                    ax3ds.scatter([best.eps_x], [best.eps_y], [best.ARI], color="red",
                                 s=100, marker="*", zorder=10, label=f"Meilleur ARI={best.ARI:.3f}")
                    plt.colorbar(surf, ax=ax3ds, shrink=0.5, label="ARI")
                    ax3ds.set_xlabel("eps_x"); ax3ds.set_ylabel("eps_y")
                    ax3ds.set_zlabel("ARI"); ax3ds.set_zlim(0, 1.05)
                    ax3ds.set_title(f"Surface session - {scripteur}", fontsize=10)
                    ax3ds.legend(fontsize=8)
                    fig3ds.tight_layout(); st.pyplot(fig3ds); plt.close(fig3ds)
                except Exception as e:
                    st.info(f"Surface indisponible : {e}")
            else:
                st.info(f"Explorez au moins 6 combinaisons ({len(df_hist)} actuellement).")

        if st.button("Réinitialiser la trace 3D"):
            st.session_state.historique = []
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Export")

    def export_figures_png(suffixe):
        """
        Exporte les trois figures affichées à l'écran (BM réels, lignes,
        mode d'affichage courant) en un seul fichier PNG à trois panneaux.
        Le nom de fichier encode eps_y/eps_x/mode, pour distinguer
        plusieurs exports d'un même scripteur à des réglages différents
        sans se fier uniquement à la date de modification.
        """
        fig_exp, axes = plt.subplots(1, 3, figsize=(21, 6))
        bm_c  = gen_couleurs(set(seg.get("BM", "?") for seg in data_filtre), 42)
        lig_c = gen_couleurs(set(df["LigneID"].unique()), 7)
        res_c = gen_couleurs(set(df[col_name].unique()), seed)
        for seg in data_filtre:
            pts = seg["Points"]
            axes[0].plot([p["X"] for p in pts], [p["Y"] for p in pts],
                        color=bm_c.get(seg.get("BM", "?"), "#333"), linewidth=1.5)
        for _, row in df.iterrows():
            pts = row["Points"]
            axes[1].plot([p["X"] for p in pts], [p["Y"] for p in pts],
                        color=lig_c.get(row["LigneID"], "#333"), linewidth=1.5)
            axes[2].plot([p["X"] for p in pts], [p["Y"] for p in pts],
                        color=res_c.get(row[col_name], "#333"), linewidth=1.5)
        axes[0].set_title(f"BM réels - {scripteur}")
        axes[1].set_title(f"Lignes - {n_lignes}  eps_y={eps_y:.3f}")
        axes[2].set_title(f"{mode_affichage} | ARI={ari:.3f} [{zone}]")
        for ax in axes: ax.set_xlabel("X"); ax.set_ylabel("Y")
        nom_png = f"{out}/export_ey{eps_y:.3f}_ex{eps_x:.3f}_{mode_affichage}{suffixe}.png"
        fig_exp.tight_layout(); fig_exp.savefig(nom_png, dpi=150); plt.close(fig_exp)
        return [nom_png]

    def export_csv_resultats(suffixe):
        """
        Exporte une ligne de résultats chiffrés (réglages + ARI) dans un
        CSV par scripteur. Si suffixe == "" (mode "remplacer") et que le
        fichier existe déjà, la nouvelle ligne est ajoutée à la suite des
        anciennes plutôt que de les écraser : ce CSV accumule ainsi
        l'historique des essais pour CE scripteur, contrairement au choix
        "nouvelle version horodatée" qui crée un fichier séparé. Consigne
        aussi une trace dans le journal d'audit global.
        """
        nom_csv = f"{out}/resultats_{scripteur}{suffixe}.csv"
        ligne = {"scripteur": scripteur, "eps_y": eps_y, "eps_x": eps_x,
                "minS_y": minS_y, "minS_x": minS_x, "n_lignes": n_lignes,
                "n_mots": n_mots, "n_bm_reels": n_bm_reels,
                "ARI": round(ari, 4), "zone": zone}
        df_csv = pd.DataFrame([ligne])
        if suffixe == "" and os.path.exists(nom_csv):
            df_csv = pd.concat([pd.read_csv(nom_csv), df_csv], ignore_index=True)
        df_csv.to_csv(nom_csv, index=False)
        consigner_journal(nom_source, scripteur, "avec_BM",
                         {"eps_y": eps_y, "eps_x": eps_x, "minS_y": minS_y, "minS_x": minS_x},
                         {"n_lignes": n_lignes, "n_mots": n_mots, "ARI": round(ari, 4)},
                         empreinte_source)
        return [nom_csv]

    def export_trace_3d(suffixe):
        """
        Exporte l'intégralité de l'historique d'exploration de la session
        courante (tous les couples eps_y/eps_x testés, avec leur ARI) en
        CSV. Contrairement aux autres exports, n'a pas de sens hors du
        mode avec BM : sans ARI, il n'y a pas d'historique à tracer.
        """
        nom_hist = f"{out}/trace3D_{scripteur}{suffixe}.csv"
        pd.DataFrame(st.session_state.historique).to_csv(nom_hist, index=False)
        return [nom_hist]

    def export_sauvegarde_figures(suffixe):
        """
        Variante à deux panneaux (BM réels + résultat courant) de
        export_figures_png, sans le panneau des lignes. Correspond au
        bouton "Sauvegarder figures", pensé comme une capture rapide de
        l'essentiel plutôt que le rapport complet à trois figures.
        """
        bm_c  = gen_couleurs(set(seg.get("BM", "?") for seg in data_filtre), 42)
        res_c = gen_couleurs(set(df[col_name].unique()), seed)
        fig_s, axes = plt.subplots(1, 2, figsize=(14, 6))
        for seg in data_filtre:
            pts = seg["Points"]
            axes[0].plot([p["X"] for p in pts], [p["Y"] for p in pts],
                        color=bm_c.get(seg.get("BM", "?"), "#333"), linewidth=1.5)
        for _, row in df.iterrows():
            pts = row["Points"]
            axes[1].plot([p["X"] for p in pts], [p["Y"] for p in pts],
                        color=res_c.get(row[col_name], "#333"), linewidth=1.5)
        axes[0].set_title(f"BM réels - {scripteur}")
        axes[1].set_title(f"{mode_affichage} | ARI={ari:.3f} [{zone}]")
        nom = f"{out}/interface_ey{eps_y:.3f}_ex{eps_x:.3f}_{mode_affichage}{suffixe}.png"
        fig_s.tight_layout(); fig_s.savefig(nom, dpi=150); plt.close(fig_s)
        return [nom]

    exporter_avec_confirmation("figures_png", out, export_figures_png, "Exporter figures PNG")
    exporter_avec_confirmation("csv_resultats", out, export_csv_resultats, "Exporter CSV résultats")

    st.sidebar.markdown("---")
    if st.sidebar.button("Exporter trace 3D (CSV)"):
        if st.session_state.historique:
            chemins = export_trace_3d("")
            st.sidebar.success(f"Trace 3D : {chemins[0]}")
        else:
            st.sidebar.warning("Aucune trace à exporter.")

    exporter_avec_confirmation("sauvegarde_figures", out, export_sauvegarde_figures,
                               "Sauvegarder figures")


########

# Étape 5 : MODE SANS BM - DBSCAN 2D à plat, un seul eps, pas d'ARI
#
# Ce qui est délibérément absent de ce mode, à ne pas ajouter par erreur
# en pensant réparer un oubli :
#   - pas de cascade Y puis X (un seul DBSCAN sur (X, Y) ensemble) ;
#   - pas d'ARI ni de zone (sur/sous-regroupement) : nécessitent une
#     vérité terrain à comparer, qui n'existe pas ici ;
#   - pas d'historique 3D : s'appuie sur l'ARI comme critère à tracer,
#     absent ici. En revanche, l'export d'un nouveau JSON BM (voir
#     export_json_bm_brut plus bas) reste disponible, spécifiquement
#     pour ce mode.

else:

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Paramètres DBSCAN")
    eps         = st.sidebar.slider("eps", 0.05, 3.0, 1.5, 0.05)
    min_samples = st.sidebar.slider("min_samples", 1, 20, 5, 1)

    labels_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    taux_bruit = float(np.mean(labels_dbscan == -1))

    c1, c2 = st.columns(2)
    c1.metric("Clusters DBSCAN", n_clusters)
    c2.metric("Taux de bruit", f"{taux_bruit:.1%}")

    if taux_bruit > 0.30:
        st.warning(f"Taux de bruit élevé ({taux_bruit:.0%}).")

    texte_resume, df_tailles = resumer_clusters(labels_dbscan)
    st.info(texte_resume)

    fig_dbscan = plot_labels_simple(coords, labels_dbscan,
                                    f"DBSCAN (eps={eps}) - {n_clusters} groupes - {scripteur}")
    st.pyplot(fig_dbscan); plt.close(fig_dbscan)

    with st.expander("Tailles des clusters (nombre de points)"):
        st.dataframe(df_tailles, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Export")

    def export_figure_brut(suffixe):
        """
        Exporte la figure unique du mode sans BM (le résultat DBSCAN brut,
        sans comparaison possible puisqu'il n'y a pas de vérité terrain).
        Recalcule la figure plutôt que de réutiliser fig_dbscan affichée
        plus haut, pour rester indépendante de l'ordre d'exécution du
        script si cette fonction venait à être appelée séparément.
        """
        fig = plot_labels_simple(coords, labels_dbscan,
                                 f"DBSCAN (eps={eps}) - {n_clusters} groupes - {scripteur}")
        nom = f"{out}/vis_dbscan_{scripteur}{suffixe}.png"
        fig.savefig(nom, dpi=150); plt.close(fig)
        return [nom]

    def export_csv_brut(suffixe):
        """
        Équivalent de export_csv_resultats() pour le mode sans BM : mêmes
        principes (ajout à la suite si "remplacer" et fichier existant,
        journal d'audit global), mais avec des colonnes différentes
        puisqu'il n'y a ni ARI, ni n_mots_verite, ni zone dans ce mode.
        """
        nom_csv = f"{out}/resultats_{scripteur}{suffixe}.csv"
        ligne = {"scripteur": scripteur, "eps": eps, "min_samples": min_samples,
                "n_clusters": n_clusters, "taux_bruit": round(taux_bruit, 4)}
        df_csv = pd.DataFrame([ligne])
        if suffixe == "" and os.path.exists(nom_csv):
            df_csv = pd.concat([pd.read_csv(nom_csv), df_csv], ignore_index=True)
        df_csv.to_csv(nom_csv, index=False)
        consigner_journal(nom_source, scripteur, "brut",
                         {"eps": eps, "min_samples": min_samples},
                         {"n_clusters": n_clusters, "taux_bruit": round(taux_bruit, 4)},
                         empreinte_source)
        return [nom_csv]

    def export_json_bm_brut(suffixe):
        """
        Équivalent de export_json_bm() (mode avec BM) pour le mode sans
        BM : construit un nouveau fichier _with_BM.json à partir des
        clusters DBSCAN trouvés, pour permettre de créer une première
        vérité terrain même en partant d'un fichier brut sans annotation.

        Différence avec la version "avec BM" : ici labels_dbscan est
        indexé par point (pas déjà agrégé par stroke dans un DataFrame),
        donc on refait la même jointure par N et le même vote majoritaire
        que cascade() effectue en interne, avant de reconstruire les
        segments.
        """
        N_to_label = {N_liste[i]: int(labels_dbscan[i]) for i in range(len(N_liste))}

        nouveaux_segments = []
        for seg in data_filtre:
            labels_seg = [N_to_label.get(p.get("N", p.get("n")), -1)
                          for p in seg["Points"]]
            label_seg = Counter(labels_seg).most_common(1)[0][0]
            nouveau_bm = str(label_seg + 1) if label_seg != -1 else "P"
            pts_propres = [dict(p) for p in seg["Points"]]
            nouveaux_segments.append({"BM": nouveau_bm, "Points": pts_propres})

        nom_json = f"{out}/{scripteur}_with_BM{suffixe}.json"
        with open(nom_json, "w", encoding="utf-8") as f:
            json.dump(nouveaux_segments, f, ensure_ascii=False, indent=2)
        return [nom_json]

    exporter_avec_confirmation("figure_brut", out, export_figure_brut, "Exporter figure PNG")
    exporter_avec_confirmation("csv_brut", out, export_csv_brut, "Exporter CSV résultats")
    exporter_avec_confirmation("json_bm_brut", out, export_json_bm_brut,
                               "Exporter nouveau JSON BM")
