"""
analyse_q1.py
=============
QUESTION 1 (directeur du service de sante) :
« Comment se répartit le premier indicateur physiologique (IMC) ? Y a-t-il
des étudiants dont la mesure sort du lot ? Comment présenter cela simplement
à une équipe médicale non statisticienne ? »

Méthode mobilisée : statistique descriptive univariée
  - indicateurs de tendance centrale et de dispersion (moyenne, médiane,
    écart-type, quartiles)
  - détection d'individus atypiques par la méthode de l'écart interquartile
    (IQR), qui a l'avantage d'être robuste et facile à expliquer sans jargon
    ("un étudiant est signalé si son IMC sort largement de l'intervalle où
    se situe la grande majorité des autres")
  - représentations graphiques : histogramme + boîte à moustaches
"""

import pandas as pd
import matplotlib.pyplot as plt
from utils import titre, sous_titre, sauver_figure, bornes_iqr, detecter_outliers_iqr


def executer(df: pd.DataFrame) -> dict:
    titre("QUESTION 1 - Distribution de l'IMC")

    imc = df["IMC"]

    moyenne = imc.mean()
    mediane = imc.median()
    ecart_type = imc.std()
    minimum, maximum = imc.min(), imc.max()
    basse, haute, q1, q3, iqr = bornes_iqr(imc)
    outliers = detecter_outliers_iqr(imc)

    sous_titre("Résumé statistique")
    print(f"Effectif                 : {len(imc)} étudiants")
    print(f"Moyenne                  : {moyenne:.2f} kg/m²")
    print(f"Médiane                  : {mediane:.2f} kg/m²")
    print(f"Écart-type               : {ecart_type:.2f} kg/m²")
    print(f"Minimum / Maximum        : {minimum:.2f} / {maximum:.2f} kg/m²")
    print(f"Q1 (25%) / Q3 (75%)      : {q1:.2f} / {q3:.2f} kg/m²")
    print(f"Écart interquartile (IQR): {iqr:.2f}")
    print(f"Intervalle 'normal' IQR  : [{basse:.2f} ; {haute:.2f}]")
    print(f"Nombre d'étudiants atypiques (outliers) : {len(outliers)}")
    if len(outliers) > 0:
        print("Détail des cas atypiques :")
        print(df.loc[outliers.index, ["id_etudiant", "IMC"]].to_string(index=False))

    # --- Figure : histogramme + boîte à moustaches côte à côte ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(imc, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].axvline(moyenne, color="red", linestyle="--", label=f"Moyenne = {moyenne:.1f}")
    axes[0].axvline(mediane, color="green", linestyle="--", label=f"Médiane = {mediane:.1f}")
    axes[0].set_title("Distribution de l'IMC")
    axes[0].set_xlabel("IMC (kg/m²)")
    axes[0].set_ylabel("Nombre d'étudiants")
    axes[0].legend(fontsize=8)

    axes[1].boxplot(imc, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#DD8452", alpha=0.7))
    axes[1].set_title("Boîte à moustaches - IMC")
    axes[1].set_ylabel("IMC (kg/m²)")

    fig.suptitle("Question 1 - Répartition de l'IMC dans l'échantillon", fontweight="bold")
    fig.tight_layout()
    sauver_figure(fig, "q1_distribution_imc.png")

    sous_titre("Réponse en langage simple pour l'équipe médicale")
    print(
        f"La grande majorité des étudiants ont un IMC compris entre {basse:.1f} et "
        f"{haute:.1f} kg/m² (valeur typique autour de {mediane:.1f}). "
        f"{len(outliers)} étudiant(s) sortent nettement de cette zone et mériteraient "
        "une attention particulière lors du dépistage."
    )

    return {
        "moyenne": moyenne, "mediane": mediane, "ecart_type": ecart_type,
        "outliers_idx": list(outliers.index), "borne_basse": basse, "borne_haute": haute,
    }


if __name__ == "__main__":
    data = pd.read_csv("data/donnees_groupe36.csv")
    executer(data)
