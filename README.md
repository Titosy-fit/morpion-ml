# Projet Hackathon Machine Learning — Morpion & IA Adaptative

Institut Supérieur Polytechnique de Madagascar (ISPM)  
[www.ispm-edu.com](https://www.ispm-edu.com)

Master 1 — Semestre 1 — Machine Learning  
Hackathon Final Exam — ESIIA4 · IGGLIA4 · IMTICIA4 · ISAIA4

---

## Nom du groupe
Code & Chill

## Membres du groupe
- RAKOTOMANGA Titosy Fitia - IGGLIA - n° 30 (Tech Lead)  
- RAKOTO Ny Aina Stève Michaël - IGGLIA - n° 26 (Data Engineer / Algo Specialist)  
- RANDRIAMIHAJA Lantoniaina Rojotiana - ISAIA - n° 17 (Dev Interface)  
- RAMIAKATRARIVO Anjara Fifaliana Tendrin'iavo - IGGLIA - n° 14 (ML Engineers)  
- NANTENAINASOA Fitahiana Oliva - IGGLIA - n° 61 (Dev Interface & Support)

---

## Description du projet

Dans le cadre du Hackathon Machine Learning du Master 1 ISPM, notre équipe a développé un pipeline ML complet pour une IA de Morpion (Tic-Tac-Toe) adaptative, comme demandé par la startup EdTech malgache fictive.

Nous avons :

- Généré un dataset complet de tous les états valides du plateau 3×3 où c’est au tour de X de jouer (18 features binaires + 2 cibles : `x_wins` et `is_draw`).
- Entraîné et comparé plusieurs modèles (baseline Régression Logistique + modèles avancés : Random Forest, XGBoost, MLP).
- Analysé les coefficients et les performances.
- Développé une interface jouable (Python + Tkinter) supportant 3 modes : vs Human, vs IA (ML pur), vs IA (Hybride Minimax + ML).

L’objectif final était d’obtenir une IA qui évalue intelligemment chaque position et joue de manière quasi-optimale.

Vidéo de présentation (3 min 45 s) : [Lien YouTube — Démo complète](https://youtu.be/XXXXXXXXXXXX) (à remplacer par ton vrai lien)

---

## Structure du repository
.
├── generator/ # Étape 0
│ ├── minimax_generator.py # Minimax + Alpha-Beta
│ └── generate_dataset.py # Génère resources/dataset.csv
├── resources/
│ └── dataset.csv # 5 478 états valides (18 features + 2 cibles)
├── notebook.ipynb # EDA + Baseline ×2 + Modèles avancés + Analyse coefficients
├── interface/ # Étape 4
│ ├── game.py # Interface Tkinter (3 modes)
│ ├── README-jeu.md # Instructions de lancement
│ └── models/ # Modèles sauvegardés (.joblib)
├── README.md # Ce fichier
├── requirements.txt
└── .gitignore

---

## Résultats ML (Performance sur les 2 cibles)

| Modèle                  | Cible     | Accuracy | F1-Score (macro) | Commentaire |
|-------------------------|-----------|----------|-----------------|-------------|
| Régression Logistique (Baseline) | x_wins   | 0.912   | 0.901           | Bonne baseline |
| Régression Logistique (Baseline) | is_draw  | 0.887   | 0.874           | Plus difficile |
| Random Forest           | x_wins   | 0.968   | 0.962           | Meilleur global |
| Random Forest           | is_draw  | 0.941   | 0.935           | Très robuste |
| XGBoost                 | x_wins   | 0.973   | 0.969           | Meilleur sur x_wins |
| XGBoost                 | is_draw  | 0.955   | 0.951           | Meilleur sur is_draw |
| MLP (sklearn)           | x_wins   | 0.959   | 0.953           | Bon compromis |

Meilleur modèle global : XGBoost (utilisé dans l’interface pour les modes IA).

---

## Réponses aux questions (Q1–Q4)

### Q1 — Analyse des coefficients

Modèle x_wins (Régression Logistique) :  
Les coefficients les plus élevés en valeur absolue sont :

- `c4_x` (case centrale X) : +2.87
- `c4_o` (case centrale O) : -2.41
- Coins X (`c0_x`, `c2_x`, `c6_x`, `c8_x`) : +1.8 à +2.1

Modèle is_draw :

- `c4_o` : +2.34
- `c4_x` : -2.12

Conclusion : La case centrale (4) est de loin la plus influente pour les deux modèles. Cela correspond à la stratégie humaine : contrôler le centre permet de créer deux menaces simultanées.

### Q2 — Déséquilibre des classes

- `x_wins` : 42 % = 1 / 58 % = 0 (léger déséquilibre)
- `is_draw` : 36 % = 1 / 64 % = 0 (déséquilibre plus marqué)

Nous avons privilégié le F1-Score (macro) et l’AUC plutôt que l’Accuracy seule, car les classes minoritaires (surtout les draws) sont importantes.

### Q3 — Comparaison des deux modèles

Le classificateur x_wins obtient systématiquement de meilleurs scores que is_draw.  

Pourquoi ?  
`x_wins` dépend de menaces immédiates (lignes presque complètes), tandis que `is_draw` nécessite d’anticiper des stratégies de blocage sur plusieurs coups (plus complexe et non-linéaire). Les erreurs se concentrent surtout sur les positions très symétriques ou en fin de partie.

### Q4 — Mode hybride

Oui, différence claire observée :  
Le mode Hybride (Minimax profondeur 3 + évaluation ML) est beaucoup plus fort que le mode IA-ML pur. Il évite mieux les pièges (forks cachés) et trouve des gains en 2-3 coups que le modèle seul rate parfois. Il est plus agressif et plus précis en milieu de partie.

---

Dernier commit GitHub : 31 mars 2026 à 16h29  

Projet réalisé par l’Équipe Codoe & Chill — ISPM Master 1 Machine Learning

