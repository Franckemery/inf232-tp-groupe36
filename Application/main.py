"""
main.py
=======
Point d'entrée unique du TP INF232 - THEME A (Service de sante universitaire).

Exécute, dans l'ordre : génération des données -> Q1 -> Q2 -> Q3 -> Q4.
Toutes les figures sont sauvegardées dans figures/, le jeu de données dans data/.

Usage :
    python main.py
"""

import pandas as pd
from data_generator import nom_vers_graine, normaliser_nom, generer_donnees
import analyse_q1
import analyse_q2
import analyse_q3
import analyse_q4
from utils import titre

NOM_CHEF_GROUPE = "PUYUENPOUOPAWOUO MOUNCHILI ADRIEN"
N_ETUDIANTS = 320


def main():
    titre("TP INF232 - THEME A : SERVICE DE SANTE UNIVERSITAIRE - GROUPE 36")

    # 1. Génération des données
    graine = nom_vers_graine(NOM_CHEF_GROUPE)
    df, _latente = generer_donnees(graine, n=N_ETUDIANTS)
    df.to_csv("data/donnees_groupe36.csv", index=False)
    print(f"Chef de groupe : {NOM_CHEF_GROUPE} (chaîne : {normaliser_nom(NOM_CHEF_GROUPE)})")
    print(f"Graine utilisée : {graine}")
    print(f"Données générées : {len(df)} étudiants -> data/donnees_groupe36.csv")

    # 2. Les quatre analyses
    resultats = {}
    resultats["q1"] = analyse_q1.executer(df)
    resultats["q2"] = analyse_q2.executer(df)
    resultats["q3"] = analyse_q3.executer(df)
    resultats["q4"] = analyse_q4.executer(df)

    titre("TERMINE - Toutes les figures sont dans le dossier figures/")
    return resultats


if __name__ == "__main__":
    main()
