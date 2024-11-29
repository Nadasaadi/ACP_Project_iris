import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.acp_steps import normalize_data, compute_correlation_matrix, compute_eigenvalues_eigenvectors, project_data

# Charger les données
data = pd.read_csv('data/iris.arff.csv')
numeric_data = data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']].values

# Normalisation des données
normalized_data = normalize_data(numeric_data)

# Calcul de la matrice de corrélation 
correlation_matrix = compute_correlation_matrix(normalized_data)

# Calcul des valeurs propres et vecteurs propres
eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(correlation_matrix)

# Projection des données
projected_data = project_data(normalized_data, eigenvectors)

# Calcul de la variance expliquée
explained_variance = eigenvalues / np.sum(eigenvalues)
explained_variance_ratio = explained_variance / np.sum(explained_variance) * 100

# Cumul de la variance expliquée
cumulative_variance = np.cumsum(explained_variance_ratio)

# Création de la fenêtre principale
root = tk.Tk()
root.title("Analyse en Composantes Principales (ACP)")

# Créer un cadre pour les boutons et les placer à gauche
button_frame = tk.Frame(root)
button_frame.pack(side="left", padx=10, pady=10)

# Titre principal dans l'interface
title_label = tk.Label(root, text="Résultats de l'Analyse en Composantes Principales (ACP)", font=("Helvetica", 16))
title_label.pack(pady=10)

# Zone de texte défilante pour afficher les résultats
result_text = scrolledtext.ScrolledText(root, width=80, height=20, wrap=tk.WORD)
result_text.pack(padx=20, pady=20)

# Label pour afficher le message en vert concernant la variance expliquée
message_variance_label = tk.Label(root, text="", font=("Helvetica", 12))
message_variance_label.pack(pady=10)

# Fonction pour afficher les résultats dans la zone de texte
def afficher_resultats():
    # Effacer le contenu précédent
    result_text.delete(1.0, tk.END)
    
    message = f"Valeurs propres :\n{eigenvalues}\n\n"
    message += f"Vecteurs propres :\n{eigenvectors}\n\n"
    message += f"Variance expliquée par chaque composante :\n{explained_variance_ratio}\n"
    result_text.insert(tk.END, message)
    
    # Vérification de la variance cumulée et affichage du message en fonction du seuil
    for i, cum_var in enumerate(cumulative_variance):
        if cum_var > 90:  # Si la variance cumulée dépasse 90%
            message_variance = f"\nLes {i+1} premières composantes principales capturent {cum_var:.2f}% de l'information.\n"
            message_variance_label.config(text=message_variance, fg="green")
            break
    else:
        # Si aucune composante ne dépasse 90%, afficher un message par défaut
        message_variance = f"\nLa variance cumulée n'atteint pas 90%.\n"
        message_variance_label.config(text=message_variance, fg="red")

# Fonction pour afficher la projection des données
def afficher_projection():
    # Visualisation des deux premières composantes principales
    plt.scatter(projected_data[:, 0], projected_data[:, 1], c=data['class'].astype('category').cat.codes)
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.title('Projection des données (ACP)')
    plt.show()

# Fonction pour afficher les données standardisées
def afficher_donnees_standardisees():
    # Effacer le contenu précédent
    result_text.delete(1.0, tk.END)
    
    # Afficher les données normalisées dans la zone de texte
    message = f"Données standardisées :\n{normalized_data}\n"
    result_text.insert(tk.END, message)

# Fonction pour afficher la matrice de corrélation
def afficher_matrice_correlation():
    # Calculer la matrice de corrélation des données normalisées
    correlation_matrix = np.corrcoef(normalized_data, rowvar=False)
    
    # Effacer le contenu précédent
    result_text.delete(1.0, tk.END)
    
    # Afficher la matrice de corrélation dans la zone de texte
    message = f"Matrice de corrélation :\n{correlation_matrix}\n"
    result_text.insert(tk.END, message)

# Créer les boutons dans l'ordre souhaité

# Bouton pour afficher les données standardisées
button_donnees_standardisees = tk.Button(button_frame, text="Afficher les données standardisées", command=afficher_donnees_standardisees, font=("Helvetica", 12))
button_donnees_standardisees.pack(pady=10)

# Bouton pour afficher la matrice de corrélation
button_matrice_correlation = tk.Button(button_frame, text="Afficher la matrice de corrélation", command=afficher_matrice_correlation, font=("Helvetica", 12))
button_matrice_correlation.pack(pady=10)

# Bouton pour afficher les résultats ACP
button_afficher = tk.Button(button_frame, text="Afficher les résultats ACP", command=afficher_resultats, font=("Helvetica", 12))
button_afficher.pack(pady=10)

# Bouton pour afficher la projection
button_projection = tk.Button(button_frame, text="Afficher la projection", command=afficher_projection, font=("Helvetica", 12))
button_projection.pack(pady=10)

# Fonction de fermeture de l'application
def fermer_app():
    root.quit()

# Créer un bouton pour fermer l'application
button_quit = tk.Button(button_frame, text="Fermer", command=fermer_app, font=("Helvetica", 12), fg="red")
button_quit.pack(pady=10)

# Lancer l'interface graphique
root.mainloop()
