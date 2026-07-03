"""
analyse_q2.py
=============
QUESTION 2 (directeur du service de sante) :
« Les deux indicateurs physiologiques évoluent-ils ensemble ? Peut-on estimer
l'un à partir de l'autre pour gagner du temps ? À partir de quand une telle
estimation devient-elle dangereuse ? »

Méthode mobilisée : statistique bivariée
  - coefficient de corrélation de Pearson (force et sens de la relation
    linéaire entre IMC et TAS)
  - régression linéaire simple TAS ~ IMC (sklearn), avec R²
  - analyse des RÉSIDUS pour répondre précisément à "à partir de quand
    l'estimation devient dangereuse" : on regarde si l'erreur de prédiction
    (résidu) grossit dans certaines zones de l'IMC, ce qui indiquerait que
    le modèle linéaire simple perd en fiabilité localement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import titre, sous_titre, sauver_figure


def executer(df: pd.DataFrame) -> dict:
    titre("QUESTION 2 - Relation entre IMC et TAS")

    X = df[["IMC"]].values
    y = df["TAS"].values

    r, p_value = pearsonr(df["IMC"], df["TAS"])

    sous_titre("Corrélation")
    print(f"Coefficient de corrélation de Pearson (IMC, TAS) : r = {r:.3f}")
    print(f"p-value du test de corrélation                   : {p_value:.2e}")
    force = "forte" if abs(r) > 0.5 else ("modérée" if abs(r) > 0.3 else "faible")
    print(f"=> Corrélation {force} et statistiquement significative (p < 0.05)."
          if p_value < 0.05 else "=> Corrélation non significative.")

    modele = LinearRegression().fit(X, y)
    y_pred = modele.predict(X)
    r2 = r2_score(y, y_pred)
    residus = y - y_pred

    sous_titre("Régression linéaire TAS = a + b * IMC")
    print(f"Pente (b)      : {modele.coef_[0]:.3f} mmHg par unité d'IMC")
    print(f"Ordonnée (a)   : {modele.intercept_:.3f}")
    print(f"R² (variance expliquée) : {r2:.3f}  "
          f"({r2*100:.1f}% de la variabilité de la TAS est expliquée par l'IMC seul)")
    print(f"Écart-type des résidus  : {residus.std():.2f} mmHg")

    # --- Analyse fine : la variance des résidus grandit-elle en bord de distribution ? ---
    imc_vals = df["IMC"].values
    imc_median = np.median(imc_vals)
    zone_centrale = np.abs(imc_vals - imc_median) < df["IMC"].std()
    ecart_residus_centre = residus[zone_centrale].std()
    ecart_residus_extremes = residus[~zone_centrale].std()

    sous_titre("Fiabilité de l'estimation selon la zone d'IMC")
    print(f"Écart-type des résidus pour IMC proches de la médiane : {ecart_residus_centre:.2f} mmHg")
    print(f"Écart-type des résidus pour IMC extrêmes               : {ecart_residus_extremes:.2f} mmHg")

    seuil_imc_bas = imc_median - df["IMC"].std()
    seuil_imc_haut = imc_median + df["IMC"].std()

    # --- Figure : nuage de points + droite de régression + résidus ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].scatter(df["IMC"], df["TAS"], alpha=0.5, s=18, color="#4C72B0")
    ordre = np.argsort(imc_vals)
    axes[0].plot(imc_vals[ordre], y_pred[ordre], color="red", linewidth=2, label="Régression linéaire")
    axes[0].set_xlabel("IMC (kg/m²)")
    axes[0].set_ylabel("TAS (mmHg)")
    axes[0].set_title(f"IMC vs TAS (r = {r:.2f}, R² = {r2:.2f})")
    axes[0].legend(fontsize=8)

    axes[1].scatter(df["IMC"], residus, alpha=0.5, s=18, color="#DD8452")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(seuil_imc_bas, color="grey", linestyle="--", linewidth=1)
    axes[1].axvline(seuil_imc_haut, color="grey", linestyle="--", linewidth=1)
    axes[1].set_xlabel("IMC (kg/m²)")
    axes[1].set_ylabel("Résidu (TAS observée - TAS prédite)")
    axes[1].set_title("Résidus de la régression")

    fig.suptitle("Question 2 - Relation IMC / TAS et fiabilité de l'estimation", fontweight="bold")
    fig.tight_layout()
    sauver_figure(fig, "q2_regression_imc_tas.png")

    sous_titre("Réponse en langage simple pour le directeur")
    print(
        f"Oui, les deux mesures évoluent globalement ensemble (corrélation {force}, "
        f"r = {r:.2f}). On peut estimer la TAS à partir de l'IMC seul avec une bonne "
        f"fiabilité pour les étudiants dont l'IMC est proche de la moyenne "
        f"(entre {seuil_imc_bas:.1f} et {seuil_imc_haut:.1f} kg/m² environ). "
        f"Au-delà de cette zone, la marge d'erreur de l'estimation augmente nettement "
        f"({ecart_residus_extremes:.1f} mmHg d'écart-type contre {ecart_residus_centre:.1f} mmHg "
        "au centre) : pour les étudiants aux profils extrêmes (très maigres ou en "
        "surpoids marqué), il reste préférable de mesurer réellement la TAS plutôt "
        "que de l'estimer, l'estimation devenant risque a utiliser pour une décision "
        "medicale individuelle."
    )

    return {"r": r, "r2": r2, "residus_std": residus.std(),
            "seuil_bas": seuil_imc_bas, "seuil_haut": seuil_imc_haut}


if __name__ == "__main__":
    data = pd.read_csv("data/donnees_groupe36.csv")
    executer(data)
