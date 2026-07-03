"""
analyse_q4.py
=============
QUESTION 4 (directeur du service de sante) :
« Peut-on prédire automatiquement, dès le dépistage, si un étudiant sera
'à risque', uniquement à partir de l'IMC et de la TAS ? Quelle confiance
accorder à cette prédiction ? Que se passerait-il en cas d'erreur ? »

Méthode mobilisée : classification supervisée
  - régression logistique (modèle simple, interprétable, adapté à un
    contexte médical où la sortie doit rester explicable à des non-experts)
  - séparation train/test (80/20, stratifiée) pour évaluer la généralisation
  - métriques orientées "risque médical" plutôt que la seule exactitude :
    matrice de confusion, rappel (recall) sur la classe 'a_risque', car un
    FAUX NÉGATIF (étudiant réellement à risque mais classé non à risque) a
    un coût humain plus grave qu'un faux positif (fausse alerte).
  - discussion explicite de l'abaissement du seuil de décision pour
    privilégier le rappel, au prix de plus de fausses alertes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              recall_score, precision_score, f1_score,
                              roc_curve, auc)
from utils import titre, sous_titre, sauver_figure


def executer(df: pd.DataFrame) -> dict:
    titre("QUESTION 4 - Prédiction automatique du risque (classification supervisée)")

    X = df[["IMC", "TAS"]].values
    y = (df["Risque"] == "a_risque").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    modele = LogisticRegression().fit(X_train_s, y_train)
    y_proba = modele.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    rappel = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    sous_titre("Performance avec seuil de décision standard (0.5)")
    print(f"Taille échantillon test : {len(y_test)} étudiants")
    print(f"Matrice de confusion :")
    print(f"                 Prédit non-risque | Prédit à-risque")
    print(f"Réel non-risque       {tn:^15} | {fp:^15}")
    print(f"Réel à-risque         {fn:^15} | {tp:^15}")
    print(f"Précision (classe à risque) : {precision:.2f}")
    print(f"Rappel (classe à risque)    : {rappel:.2f}  <- proportion d'étudiants "
          "réellement à risque correctement détectés")
    print(f"F1-score                    : {f1:.2f}")
    print(f"Faux négatifs (danger le plus grave) : {fn} étudiant(s) réellement à "
          "risque, non détectés par le modèle")

    # --- Abaissement du seuil pour privilégier le rappel ---
    seuil_prudent = 0.35
    y_pred_prudent = (y_proba >= seuil_prudent).astype(int)
    rappel_prudent = recall_score(y_test, y_pred_prudent)
    precision_prudente = precision_score(y_test, y_pred_prudent)
    cm_prudent = confusion_matrix(y_test, y_pred_prudent)
    fn_prudent = cm_prudent[1, 0]

    sous_titre(f"Effet d'un seuil plus prudent (seuil = {seuil_prudent})")
    print(f"Rappel avec seuil abaissé    : {rappel_prudent:.2f} (contre {rappel:.2f} avant)")
    print(f"Précision avec seuil abaissé : {precision_prudente:.2f} (contre {precision:.2f} avant)")
    print(f"Faux négatifs restants       : {fn_prudent} (contre {fn} avant)")
    print("=> Abaisser le seuil detecte plus d'étudiants réellement à risque, au prix "
          "de plus de fausses alertes (étudiants convoqués à tort pour un contrôle "
          "complémentaire) - un compromis raisonnable en contexte médical.")

    # --- ROC / AUC ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    coef = modele.coef_[0]
    sous_titre("Poids appris par le modèle (sur variables standardisées)")
    print(f"Poids IMC : {coef[0]:.3f}")
    print(f"Poids TAS : {coef[1]:.3f}")

    # --- Figures ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["non_a_risque", "a_risque"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["non_a_risque", "a_risque"])
    axes[0].set_xlabel("Prédiction")
    axes[0].set_ylabel("Réalité")
    axes[0].set_title("Matrice de confusion (seuil = 0.5)")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    axes[1].plot(fpr, tpr, color="#4C72B0", label=f"ROC (AUC = {roc_auc:.2f})")
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Modèle aléatoire")
    axes[1].set_xlabel("Taux de faux positifs")
    axes[1].set_ylabel("Taux de vrais positifs (rappel)")
    axes[1].set_title("Courbe ROC")
    axes[1].legend(fontsize=8)

    fig.suptitle("Question 4 - Prédiction automatique du risque", fontweight="bold")
    fig.tight_layout()
    sauver_figure(fig, "q4_classification_risque.png")

    sous_titre("Réponse en langage simple pour le directeur")
    print(
        f"Oui, un modèle simple (régression logistique) permet de prédire le risque "
        f"à partir de l'IMC et de la TAS avec une fiabilité correcte mais imparfaite "
        f"(aire sous la courbe ROC = {roc_auc:.2f} sur 1.00). Avec un seuil de décision "
        f"standard, {fn} étudiant(s) réellement à risque passeraient inaperçus dans "
        "cet echantillon test - c'est le danger principal : le modèle ne mesure que "
        "l'IMC et la TAS, alors que le risque réel dépend aussi d'autres facteurs non "
        "mesurés ici (mode de vie, antécédents...). En pratique, il est recommandé "
        f"d'utiliser un seuil de décision plus prudent (ex: {seuil_prudent} au lieu de "
        "0.5) pour minimiser les cas manqués, quitte à convoquer un peu plus "
        "d'étudiants pour un contrôle complémentaire par excès de précaution. Ce "
        "modèle doit rester une aide au tri initial, jamais un diagnostic final."
    )

    return {"rappel": rappel, "precision": precision, "auc": roc_auc,
            "fn_standard": int(fn), "fn_prudent": int(fn_prudent)}


if __name__ == "__main__":
    data = pd.read_csv("data/donnees_groupe36.csv")
    executer(data)
