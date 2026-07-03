"""
data_generator.py
==================
Génère le jeu de données personnalisé du groupe pour le TP INF232.

THEME A : Service de sante universitaire

Principe de génération (voir Annexe du sujet)
----------------------------------------------
1. Le nom complet du chef de groupe est transformé en une chaîne normalisée
   (majuscules, sans espaces ni accents).
2. Cette chaîne est transformée en un entier reproductible (la "graine du
   groupe") via une fonction de hachage polynomial déterministe :

       seed = sum( ord(char_i) * P^i )  mod M

   où P est un nombre premier (31) et M un grand modulo (2^31 - 1, premier
   de Mersenne). Ce choix garantit :
     - le déterminisme total (même chaîne -> même graine, toujours) ;
     - une très faible probabilité de collision entre deux noms différents,
       car de petites variations dans la chaîne d'entrée changent radicalement
       la somme pondérée (effet avalanche partiel) ;
     - un entier de taille raisonnable, utilisable directement comme seed
       pour numpy.random.

3. Cette graine initialise un générateur pseudo-aléatoire standard
   (numpy.random.default_rng), qui produit ensuite l'ensemble des données.

Variables générées
-------------------
- IMC   : Indice de Masse Corporelle (kg/m^2), variable physiologique 1.
- TAS   : Tension Artérielle Systolique (mmHg), variable physiologique 2,
          corrélée à l'IMC (relation physiologique plausible + bruit).
- L     : variable latente ("prédisposition individuelle au risque", facteurs
          non mesurés : hérédité, mode de vie...). N'est JAMAIS exportée dans
          le CSV final : un service de santé réel n'a pas accès à ce type
          d'information. Elle sert uniquement, en interne, à construire
          l'étiquette de risque de façon réaliste (imparfaitement prévisible
          à partir des deux seules mesures disponibles).
- Risque : étiquette binaire ("a_risque" / "non_a_risque"), générée via un
           modèle logistique latent :

               z = b0 + b1*IMC_std + b2*TAS_std + b3*L
               p = sigmoide(z)
               Risque ~ Bernoulli(p)

           Propriété clé (subtilité pédagogique) : quand p est proche de 0
           ou 1 (profils cliniquement "extrêmes"), l'étiquette est quasi
           déterministe. Quand p est proche de 0.5 (profils "limites"), le
           tirage est quasi aléatoire : l'incertitude de l'étiquette croît
           naturellement à mesure qu'on s'approche de la frontière de
           décision. C'est cette propriété qui permet de répondre en Q2/Q4
           à "à partir de quand une estimation/prédiction devient dangereuse".

Taille d'échantillon
---------------------
N = 320 étudiants. Justification : pour qu'une régression linéaire (Q2), un
clustering K-means avec recherche du k optimal par silhouette (Q3), et une
classification supervisée avec un split train/test 80/20 (Q4) donnent des
résultats stables et interprétables, quelques centaines d'individus sont
nécessaires (règle empirique : au moins ~30 par classe/cluster attendu, ici
on vise 4 profils et 2 classes, donc un minimum de 120-150 serait déjà
correct ; 320 offre une marge confortable sans être artificiellement énorme).
"""

import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. Nom du chef de groupe -> graine reproductible
# ------------------------------------------------------------------

def normaliser_nom(nom_complet: str) -> str:
    """Majuscules, sans accents, sans espaces (ex: 'Franck Emery' -> 'FRANCKEMERY')."""
    sans_accents = unicodedata.normalize("NFKD", nom_complet)
    sans_accents = "".join(c for c in sans_accents if not unicodedata.combining(c))
    return "".join(sans_accents.upper().split())


def nom_vers_graine(nom_complet: str) -> int:
    """Hachage polynomial déterministe d'une chaîne en entier 32 bits."""
    chaine = normaliser_nom(nom_complet)
    P = 31
    M = (2 ** 31) - 1  # nombre premier de Mersenne
    graine = 0
    for i, char in enumerate(chaine):
        graine = (graine + ord(char) * pow(P, i, M)) % M
    return graine


# ------------------------------------------------------------------
# 2. Génération du jeu de données à partir de la graine
# ------------------------------------------------------------------

def generer_donnees(graine: int, n: int = 320) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne (df_public, latente) :
      - df_public : DataFrame avec colonnes ['id', 'IMC', 'TAS', 'Risque']
                    (ce que reçoit réellement le service de santé)
      - latente   : Series de la variable latente L, gardée séparée, utilisée
                    uniquement pour la discussion des limites dans le rapport.
    """
    rng = np.random.default_rng(graine)

    # --- IMC : distribution réaliste population étudiante (16 a 35 kg/m^2) ---
    imc = rng.normal(loc=22.5, scale=3.4, size=n)
    imc = np.clip(imc, 15.5, 36.0)

    # --- TAS : liée physiologiquement à l'IMC + bruit propre ---
    tas = 100 + 1.15 * (imc - 22.5) + rng.normal(loc=0, scale=8.5, size=n)
    tas = np.clip(tas, 92.0, 158.0)

    # --- Variable latente (facteurs non mesurés, jamais exportée) ---
    latente = rng.normal(loc=0.0, scale=1.0, size=n)

    # --- Standardisation interne pour construire le score de risque ---
    imc_std = (imc - imc.mean()) / imc.std()
    tas_std = (tas - tas.mean()) / tas.std()

    b0, b1, b2, b3 = -0.4, 0.9, 0.75, 0.9
    z = b0 + b1 * imc_std + b2 * tas_std + b3 * latente
    p = 1 / (1 + np.exp(-z))
    risque_bin = rng.binomial(1, p)

    df = pd.DataFrame({
        "id_etudiant": [f"ETU{i+1:04d}" for i in range(n)],
        "IMC": np.round(imc, 2),
        "TAS": np.round(tas, 1),
        "Risque": np.where(risque_bin == 1, "a_risque", "non_a_risque"),
    })

    return df, pd.Series(latente, name="latente_L_non_exportee")


# ------------------------------------------------------------------
# 3. Exécution directe : preuve de reproductibilité
# ------------------------------------------------------------------

if __name__ == "__main__":
    NOM_CHEF_GROUPE = "PUYUENPOUOPAWOUO MOUNCHILI ADRIEN"
    N_ETUDIANTS = 320

    graine = nom_vers_graine(NOM_CHEF_GROUPE)
    df, latente = generer_donnees(graine, n=N_ETUDIANTS)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "donnees_groupe36.csv"
    df.to_csv(csv_path, index=False)

    print("=" * 60)
    print("GENERATION DES DONNEES - PREUVE DE REPRODUCTIBILITE")
    print("=" * 60)
    print(f"Chef de groupe        : {NOM_CHEF_GROUPE}")
    print(f"Chaine normalisee     : {normaliser_nom(NOM_CHEF_GROUPE)}")
    print(f"Graine obtenue        : {graine}")
    print(f"Nombre d'individus    : {N_ETUDIANTS}")
    print(f"Fichier CSV genere    : {csv_path}")
    print()
    print("Extrait des 5 premieres lignes :")
    print(df.head(5).to_string(index=False))
    print()
    print("Repartition de l'etiquette Risque :")
    print(df["Risque"].value_counts())
    print()
    print("(Relancer ce script produit exactement les memes valeurs ci-dessus :")
    print(" la graine ne depend que de la chaine 'FRANCKEMERY'.)")
