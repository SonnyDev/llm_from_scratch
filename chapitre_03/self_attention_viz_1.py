import streamlit as st
import numpy as np
import pandas as pd

# Fonction pour calculer les scores d'attention
def calculate_attention_scores(keys, query):
    """Calcule les scores d'attention entre chaque clé et la requête."""
    scores = np.dot(keys, query)  # Produit scalaire
    return scores

# Fonction pour normaliser les scores
def softmax(x):
    """Applique la fonction softmax pour normaliser les scores d'attention."""
    e_x = np.exp(x - np.max(x))  # Stabilisation
    return e_x / e_x.sum(axis=0)

# Fonction pour calculer le vecteur de contexte
def calculate_context_vector(attention_weights, values):
    """Calcule le vecteur de contexte à partir des poids d'attention et des valeurs."""
    return np.dot(attention_weights, values)

# Interface Streamlit
st.title("Visualisation du Mécanisme d'Auto-Attention")

# Saisie de texte
input_text = st.text_area("Texte d'entrée", "journey starts step", height=100)

# Paramètres
tokens = input_text.split()
query_index = st.slider("Sélectionnez le mot de requête", 0, len(tokens) - 1, 0)

# Représentation vectorielle (exemple simple)
# Chaque mot est représenté par un vecteur aléatoire
np.random.seed(0)  # Pour la reproductibilité
vectors = {token: np.random.rand(3) for token in tokens}  # Vecteurs de dimension 3

# Affichage des vecteurs
st.subheader("Représentations vectorielles des mots")
vector_df = pd.DataFrame(vectors).T
vector_df.columns = ["Dimension 1", "Dimension 2", "Dimension 3"]
st.dataframe(vector_df)

# Calcul des scores d'attention
query_vector = vectors[tokens[query_index]]
key_vectors = np.array(list(vectors.values()))
attention_scores = np.array([calculate_attention_scores(key_vectors, query_vector) for _ in key_vectors])

# Normalisation des scores
attention_weights = softmax(attention_scores)

# Affichage des scores d'attention sous forme de matrice
st.subheader("Scores d'attention")
attention_scores_df = pd.DataFrame(attention_scores, index=tokens, columns=tokens)
st.dataframe(attention_scores_df)

# Normalisation des scores pour obtenir les poids d'attention
attention_weights_df = pd.DataFrame(attention_weights, index=tokens, columns=tokens)
st.subheader("Poids d'attention normalisés (softmax)")
st.dataframe(attention_weights_df)

# Calcul du vecteur de contexte
context_vector = calculate_context_vector(attention_weights, key_vectors)

# Affichage du vecteur de contexte
st.subheader("Vecteur de contexte")
st.write(context_vector)

# Visualisation des poids d'attention
st.subheader("Visualisation des poids d'attention")
st.bar_chart(attention_weights)

# Instructions
with st.expander("Comment utiliser"):
    st.write("""
    1. Entrez une séquence de mots dans la zone de texte.
    2. Sélectionnez le mot de requête à partir du curseur.
    3. Observez les scores d'attention et le vecteur de contexte calculés.
    """) 