# Analyse en Composantes Principales (ACP) - Projet sur le jeu de données Iris

## Description du projet
Ce projet implémente une Analyse en Composantes Principales (ACP) sur le jeu de données Iris, un ensemble de données très utilisé en apprentissage automatique pour la classification des fleurs d'Iris. L'ACP est une technique statistique qui permet de réduire la dimensionnalité des données tout en conservant le plus d'information possible.

Le but de ce projet est d'extraire les principales composantes des données et de visualiser la projection des données dans un espace de dimension réduite (2D). Ce projet utilise Python avec les bibliothèques `numpy`, `pandas`, `matplotlib`, et `tkinter` pour l'interface graphique.

## Prérequis
Avant d'exécuter le projet, assurez-vous que vous avez installé les bibliothèques nécessaires. Vous pouvez installer les bibliothèques requises avec pip :

```bash
pip install numpy pandas matplotlib
```
## Structure du projet
Le projet est structuré comme suit :

```
ACP_Project/
│
├── data/              # Dossier contenant les fichiers de données (ex. iris.arff.csv)
├── utils/           
│   ├── acp_steps.py   # Fonctions pour normaliser les données, calculer la matrice de covariance, etc.
└── main.py            # Script principal pour lancer l'application et afficher les résultats   
├── README.md          # Fichier README (ce fichier)

```
## Description des fichiers
acp_steps.py : Ce fichier contient les fonctions nécessaires à la normalisation des données, au calcul de la matrice de covariance, aux valeurs et vecteurs propres, ainsi qu'à la projection des données sur les nouvelles composantes principales.

main.py : C'est le script principal où les étapes de l'ACP sont exécutées et où l'interface graphique est lancée pour afficher les résultats de l'analyse.

iris.arff.csv : Le fichier de données utilisé pour l'ACP, contenant les mesures des fleurs d'Iris (longueur et largeur des sépales et pétales).

## Étapes de l'ACP:
L'ACP est réalisée en suivant les étapes suivantes :

Chargement des données : Le fichier iris.arff.csv est chargé dans un DataFrame avec pandas.

Normalisation des données : Les données sont normalisées pour avoir une moyenne nulle et un écart-type de 1, ce qui est essentiel pour l'ACP.

Calcul de la matrice de covariance : La matrice de covariance est calculée pour comprendre la relation entre les variables.

Calcul des valeurs propres et vecteurs propres : Les valeurs et vecteurs propres de la matrice de covariance sont calculés et triés par ordre décroissant.

Projection des données : Les données sont projetées sur les nouvelles composantes principales (PC1, PC2, etc.).

Variance expliquée : La proportion de la variance expliquée par chaque composante principale est calculée.

Visualisation des résultats : Une interface graphique avec tkinter permet d'afficher les résultats de l'ACP, y compris la projection des données dans un espace réduit et la variance expliquée.
## Explication des resultats
Valeurs propres et vecteurs propres : Les valeurs propres indiquent l'importance de chaque composante principale. Les vecteurs propres sont les directions dans l'espace des données correspondant à chaque composante principale.
Variance expliquée : Indique la proportion de la variance totale des données capturée par chaque composante principale. Une forte variance expliquée par les premières composantes signifie que ces composantes représentent bien l'information des données.
Projection des données : Affiche la projection des données dans un espace réduit (2D), en utilisant les deux premières composantes principales.
