"""
utils.py
========
Fonctions partagées entre les scripts d'analyse (Q1 à Q4) : style graphique
cohérent, sauvegarde de figures, détection d'outliers par IQR, affichage
console structuré.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 11


def titre(texte: str) -> None:
    """Affiche un en-tête de section dans la console."""
    print()
    print("=" * 70)
    print(texte)
    print("=" * 70)


def sous_titre(texte: str) -> None:
    print(f"\n--- {texte} ---")


def sauver_figure(fig, nom_fichier: str) -> Path:
    """Sauvegarde une figure matplotlib dans figures/ et retourne le chemin."""
    chemin = FIG_DIR / nom_fichier
    fig.savefig(chemin, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure enregistree] {chemin}")
    return chemin


def bornes_iqr(serie, k: float = 1.5):
    """Retourne (borne_basse, borne_haute, Q1, Q3, IQR) pour la méthode IQR."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr, q1, q3, iqr


def detecter_outliers_iqr(serie, k: float = 1.5):
    """Retourne la sous-série des valeurs considérées comme atypiques (méthode IQR)."""
    basse, haute, *_ = bornes_iqr(serie, k)
    return serie[(serie < basse) | (serie > haute)]
