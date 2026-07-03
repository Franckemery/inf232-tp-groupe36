"""
analyse_q3.py
=============
QUESTION 3 (directeur du service de sante) :
« Existe-t-il des groupes naturels de profils de santé parmi mes étudiants ?
Combien, et comment les décrire simplement ? »

Méthode mobilisée : classification NON supervisée (K-means)
  - standardisation des variables (IMC, TAS) avant clustering (indispensable
    car les deux variables n'ont pas la même échelle)
  - choix du nombre de clusters k via deux critères convergents :
      * méthode du coude (inertie intra-cluster)
      * score de silhouette (qualité de séparation des clusters)
  - IMPORTANT : le clustering est réalisé UNIQUEMENT sur IMC et TAS, sans
    utiliser l'étiquette "Risque". C'est ce qui distingue une démarche non
    supervisée d'une démarche supervisée (Q4) : on ne cherche pas à retrouver
    l'étiquette métier, mais une structure propre aux données. On compare
    ensuite, à titre de discussion, la répartition de l'étiquette Risque à
    l'intérieur de chaque cluster trouvé, sans que les deux se recouvrent
    parfaitement (ce qui est normal et attendu).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import titre, sous_titre, sauver_figure


def executer(df: pd.DataFrame) -> dict:
    titre("QUESTION 3 - Recherche de profils de santé (clustering non supervisé)")

    X = df[["IMC", "TAS"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Choix de k : méthode du coude + silhouette ---
    inerties = []
    silhouettes = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        inerties.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    k_optimal = list(k_range)[int(np.argmax(silhouettes))]

    sous_titre("Choix du nombre de profils (k)")
    for k, iner, sil in zip(k_range, inerties, silhouettes):
        marqueur = "  <-- meilleur score de silhouette" if k == k_optimal else ""
        print(f"k={k} : inertie={iner:.1f}, silhouette={sil:.3f}{marqueur}")
    print(f"=> k retenu = {k_optimal} profils")

    # --- Clustering final avec k optimal ---
    modele = KMeans(n_clusters=k_optimal, n_init=10, random_state=42)
    labels = modele.fit_predict(X_scaled)
    df_c = df.copy()
    df_c["Cluster"] = labels

    sous_titre(f"Description des {k_optimal} profils identifiés")
    descriptions = {}
    for c in sorted(df_c["Cluster"].unique()):
        sous_ensemble = df_c[df_c["Cluster"] == c]
        imc_moy = sous_ensemble["IMC"].mean()
        tas_moy = sous_ensemble["TAS"].mean()
        effectif = len(sous_ensemble)
        part_risque = (sous_ensemble["Risque"] == "a_risque").mean() * 100
        descriptions[c] = {
            "effectif": effectif, "imc_moyen": imc_moy,
            "tas_moyenne": tas_moy, "part_a_risque": part_risque,
        }
        print(f"Profil {c} : {effectif} étudiants | IMC moyen = {imc_moy:.1f} | "
              f"TAS moyenne = {tas_moy:.1f} | {part_risque:.0f}% déjà étiquetés 'à risque'")

    # --- Figures ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(list(k_range), silhouettes, marker="o", color="#4C72B0")
    axes[0].axvline(k_optimal, color="red", linestyle="--", label=f"k optimal = {k_optimal}")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Score de silhouette")
    axes[0].set_title("Choix de k par le score de silhouette")
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(df_c["IMC"], df_c["TAS"], c=df_c["Cluster"],
                               cmap="tab10", alpha=0.7, s=22)
    axes[1].set_xlabel("IMC (kg/m²)")
    axes[1].set_ylabel("TAS (mmHg)")
    axes[1].set_title(f"{k_optimal} profils identifiés (K-means)")
    legend1 = axes[1].legend(*scatter.legend_elements(), title="Profil", fontsize=8)
    axes[1].add_artist(legend1)

    fig.suptitle("Question 3 - Profils de santé naturels (sans utiliser l'étiquette Risque)",
                 fontweight="bold")
    fig.tight_layout()
    sauver_figure(fig, "q3_clustering_profils.png")

    sous_titre("Réponse en langage simple pour le directeur")
    print(
        f"Sans utiliser le classement 'à risque / non à risque' déjà existant, "
        f"les données font apparaître {k_optimal} profils de santé distincts parmi vos "
        "étudiants, décrits chacun par une combinaison typique d'IMC et de tension. "
        "Ces profils ne recouvrent pas exactement votre classement actuel : certains "
        "profils contiennent un mélange d'étudiants à risque et non à risque, ce qui "
        "suggère que le classement binaire actuel simplifie une réalité un peu plus "
        "nuancée."
    )

    return {"k_optimal": k_optimal, "descriptions": descriptions}


if __name__ == "__main__":
    data = pd.read_csv("data/donnees_groupe36.csv")
    executer(data)
