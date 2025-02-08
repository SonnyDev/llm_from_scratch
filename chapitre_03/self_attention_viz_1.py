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

def get_embeddings(tokens):
    """Retourne les embeddings pour les tokens donnés."""
    # Embeddings prédéfinis
    predefined_embeddings = {
        "Your": [0.43, 0.15, 0.89],
        "journey": [0.55, 0.87, 0.66],
        "starts": [0.57, 0.85, 0.64],
        "with": [0.22, 0.58, 0.33],
        "one": [0.77, 0.25, 0.10],
        "step": [0.05, 0.80, 0.55]
    }
    
    # Initialiser le dictionnaire des embeddings
    embeddings = {}
    np.random.seed(42)  # Pour la reproductibilité des embeddings aléatoires
    
    # Pour chaque token, utiliser l'embedding prédéfini s'il existe, sinon en générer un aléatoire
    for token in tokens:
        if token in predefined_embeddings:
            embeddings[token] = predefined_embeddings[token]
        else:
            embeddings[token] = np.random.rand(3)
    
    return embeddings

# Interface Streamlit
st.title("Visualisation du Mécanisme d'Auto-Attention Simple")

# Saisie de texte avec valeur par défaut correspondant aux embeddings
input_text = st.text_area("Texte d'entrée", "Your journey starts with one step", height=100)

# Paramètres
tokens = input_text.split()
query_index = st.slider("Sélectionnez le mot de requête", 0, len(tokens) - 1, 0)

# Utiliser la fonction pour obtenir les embeddings
vectors = get_embeddings(tokens)

# Affichage des vecteurs
st.subheader("Représentations vectorielles des mots")
vector_df = pd.DataFrame(vectors).T
vector_df.columns = ["Dimension 1", "Dimension 2", "Dimension 3"]
st.dataframe(vector_df)

# Après la sélection de la requête
st.subheader(f"Processus d'attention pour la requête '{tokens[query_index]}'")

# 1. Afficher la requête sélectionnée et son vecteur
query_vector = vectors[tokens[query_index]]
st.write("1. Vecteur de requête (Q):")
st.dataframe(pd.DataFrame([query_vector], columns=["Dimension 1", "Dimension 2", "Dimension 3"], index=[tokens[query_index]]))

# 2. Montrer le calcul des scores d'attention
st.write("2. Calcul des scores d'attention (Q·K):")
key_vectors = np.array(list(vectors.values()))
scores = calculate_attention_scores(key_vectors, query_vector)
scores_calculation = pd.DataFrame({
    'Token': tokens,
    'Score': scores,
    'Calcul': [f"({', '.join(f'{v:.3f}' for v in vectors[token])}) · ({', '.join(f'{v:.3f}' for v in query_vector)}) = {score:.3f}"
               for token, score in zip(tokens, scores)]
})
st.dataframe(scores_calculation)

# 3. Montrer la normalisation softmax
st.write("3. Normalisation des scores (softmax):")
weights = softmax(scores)
softmax_calculation = pd.DataFrame({
    'Token': tokens,
    'Score': scores,
    'e^score': np.exp(scores - np.max(scores)),
    'Poids normalisé': weights
})
st.dataframe(softmax_calculation)

# 4. Montrer le calcul du vecteur de contexte
st.write("4. Calcul du vecteur de contexte (somme pondérée des valeurs):")
context_calculation = pd.DataFrame({
    'Token': tokens,
    'Poids': weights,
    'Valeur': [f"({', '.join(f'{v:.3f}' for v in vectors[token])})" for token in tokens],
    'Contribution': [f"({', '.join(f'{v*w:.3f}' for v, w in zip(vectors[token], [weights[i]]))})" 
                    for i, token in enumerate(tokens)]
})
st.dataframe(context_calculation)

context_vector = np.dot(weights, key_vectors)
st.write("Vecteur de contexte final (somme des contributions):")
st.dataframe(pd.DataFrame([context_vector], 
                         columns=["Dimension 1", "Dimension 2", "Dimension 3"],
                         index=[f"Contexte pour '{tokens[query_index]}'"]))

# Calcul et affichage des matrices complètes pour tous les tokens
st.subheader("Matrices d'attention complètes")

# Calcul des scores d'attention pour tous les tokens
attention_scores = np.zeros((len(tokens), len(tokens)))
attention_weights = np.zeros((len(tokens), len(tokens)))
context_vectors = np.zeros((len(tokens), 3))

for i, token in enumerate(tokens):
    query_vector = vectors[token]
    scores = calculate_attention_scores(key_vectors, query_vector)
    attention_scores[i] = scores
    attention_weights[i] = softmax(scores)
    context_vectors[i] = np.dot(attention_weights[i], key_vectors)

# Affichage des scores d'attention
st.write("Matrice des scores d'attention:")
scores_df = pd.DataFrame(
    attention_scores,
    index=tokens,
    columns=tokens
)
# Mise en forme avec surbrillance de la ligne de la requête
scores_styled = scores_df.style.format("{:.4f}").apply(
    lambda df: pd.Series(['background-color: #e6f3ff' if df.name == tokens[query_index] else '' 
                         for _ in range(len(df.index))], index=df.index), axis=1
)
st.dataframe(scores_styled)

# Affichage des poids d'attention
st.write("Matrice des poids d'attention:")
weights_df = pd.DataFrame(
    attention_weights,
    index=tokens,
    columns=tokens
)
# Mise en forme avec surbrillance de la ligne de la requête
weights_styled = weights_df.style.format("{:.4f}").apply(
    lambda df: pd.Series(['background-color: #e6f3ff' if df.name == tokens[query_index] else '' 
                         for _ in range(len(df.index))], index=df.index), axis=1
)
st.dataframe(weights_styled)

# Affichage des vecteurs de contexte (sans surbrillance)
st.write("Vecteurs de contexte pour tous les tokens:")
context_df = pd.DataFrame(
    context_vectors,
    index=tokens,
    columns=["Dimension 1", "Dimension 2", "Dimension 3"]
)
st.dataframe(context_df.style.format("{:.4f}"))

# Visualisation des poids d'attention
st.subheader("Visualisation des poids d'attention")
# Créer un DataFrame avec l'ordre original des tokens
weights_viz_df = pd.DataFrame(
    attention_weights,
    index=tokens,
    columns=tokens
)[tokens]  # Réorganiser les colonnes dans l'ordre original
st.bar_chart(weights_viz_df)

# Instructions
with st.expander("Comment utiliser"):
    st.write("""
    1. Entrez une séquence de mots dans la zone de texte.
    2. Sélectionnez le mot de requête à partir du curseur.
    3. Observez les scores d'attention et le vecteur de contexte calculés.
    """) 