# TP INF232 — Groupe 36 — Thème A : Service de santé universitaire

## Mode d'emploi

```bash
pip install -r requirements.txt
python main.py
```

Cela régénère les données (`data/donnees_groupe36.csv`) et toutes les
figures des 4 questions (`figures/*.png`), avec le détail des calculs
affiché dans la console.

## Contenu

- `data_generator.py` — génération déterministe des données (graine issue du
  nom du chef de groupe, voir docstring du fichier)
- `analyse_q1.py` à `analyse_q4.py` — une analyse par question du sujet
- `utils.py` — fonctions partagées (figures, détection d'outliers)
- `main.py` — point d'entrée unique
- `data/donnees_groupe36.csv` — jeu de données généré (320 étudiants)
- `figures/` — les 4 figures produites par l'analyse
