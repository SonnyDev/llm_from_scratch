import streamlit as st
import torch
import numpy as np
import pandas as pd

def initialize_weights(d_in, d_out):
    """Initialise les matrices de poids pour Q, K, V."""
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    return W_query, W_key, W_value

def get_embeddings(tokens):
    """Retourne les embeddings pour les tokens donnés."""
    predefined_embeddings = {
        "Your": [0.43, 0.15, 0.89],
        "journey": [0.55, 0.87, 0.66],
        "starts": [0.57, 0.85, 0.64],
        "with": [0.22, 0.58, 0.33],
        "one": [0.77, 0.25, 0.10],
        "step": [0.05, 0.80, 0.55]
    }
    
    embeddings = {}
    torch.manual_seed(42)
    for token in tokens:
        if token in predefined_embeddings:
            embeddings[token] = torch.tensor(predefined_embeddings[token], dtype=torch.float32)
        else:
            embeddings[token] = torch.rand(3)
    return embeddings

def calculate_attention(query, keys, values):
    """Calcule l'attention et retourne les scores, poids et vecteur de contexte."""
    scores = torch.matmul(query, keys.t())
    weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.matmul(weights, values)
    return scores, weights, context

st.title("Visualisation du Mécanisme d'Auto-Attention avec Poids Entraînables")

# Saisie de texte
input_text = st.text_area("Texte d'entrée", "Your journey starts with one step", height=100)
tokens = input_text.split()
query_index = st.slider("Sélectionnez le mot de requête", 0, len(tokens) - 1, 0)

# Obtenir les embeddings
embeddings = get_embeddings(tokens)
inputs = torch.stack([embeddings[token] for token in tokens])

# Dimensions
d_in = inputs.shape[1]  # 3
d_out = 2

# Initialiser les poids
W_query, W_key, W_value = initialize_weights(d_in, d_out)

# Après avoir obtenu les embeddings et avant d'afficher les matrices de poids
# Afficher les embeddings d'entrée
st.subheader("Embeddings d'entrée")
embeddings_df = pd.DataFrame(
    inputs.numpy(),
    index=tokens,
    columns=["Dimension 1", "Dimension 2", "Dimension 3"]
)
st.dataframe(embeddings_df.style.format("{:.4f}"))

# Afficher les matrices de poids
st.subheader("Matrices de Poids")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("W_query:")
    st.dataframe(pd.DataFrame(W_query.numpy(), 
                            columns=[f"d_out_{i+1}" for i in range(d_out)],
                            index=[f"d_in_{i+1}" for i in range(d_in)]))

with col2:
    st.write("W_key:")
    st.dataframe(pd.DataFrame(W_key.numpy(),
                            columns=[f"d_out_{i+1}" for i in range(d_out)],
                            index=[f"d_in_{i+1}" for i in range(d_in)]))

with col3:
    st.write("W_value:")
    st.dataframe(pd.DataFrame(W_value.numpy(),
                            columns=[f"d_out_{i+1}" for i in range(d_out)],
                            index=[f"d_in_{i+1}" for i in range(d_in)]))

# Projeter les vecteurs d'entrée
Q = torch.matmul(inputs, W_query)  # [seq_len, d_out]
K = torch.matmul(inputs, W_key)    # [seq_len, d_out]
V = torch.matmul(inputs, W_value)  # [seq_len, d_out]

# Afficher les projections
st.subheader("Projections Q, K, V")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Projections Q:")
    q_df = pd.DataFrame(Q.numpy(), columns=[f"Q_{i+1}" for i in range(d_out)], index=tokens)
    st.dataframe(q_df.style.format("{:.4f}"))

with col2:
    st.write("Projections K:")
    k_df = pd.DataFrame(K.numpy(), columns=[f"K_{i+1}" for i in range(d_out)], index=tokens)
    st.dataframe(k_df.style.format("{:.4f}"))

with col3:
    st.write("Projections V:")
    v_df = pd.DataFrame(V.numpy(), columns=[f"V_{i+1}" for i in range(d_out)], index=tokens)
    st.dataframe(v_df.style.format("{:.4f}"))

# Calcul de l'attention pour le token sélectionné
query_vector = Q[query_index]
scores, weights, context = calculate_attention(query_vector, K, V)

# Afficher les scores et poids d'attention
st.subheader(f"Attention pour '{tokens[query_index]}'")
attention_df = pd.DataFrame({
    'Token': tokens,
    'Score': scores.numpy(),
    'Poids': weights.numpy(),
})
# Appliquer le formatage uniquement aux colonnes numériques
st.dataframe(attention_df.style.format({
    'Score': '{:.4f}',
    'Poids': '{:.4f}'
}))

# Calcul et affichage des matrices complètes
st.subheader("Matrices d'attention complètes")

# Calculer les scores et poids pour tous les tokens
attention_scores = torch.zeros((len(tokens), len(tokens)))
attention_weights = torch.zeros((len(tokens), len(tokens)))
context_vectors = torch.zeros((len(tokens), d_out))

for i in range(len(tokens)):
    scores, weights, context = calculate_attention(Q[i], K, V)
    attention_scores[i] = scores
    attention_weights[i] = weights
    context_vectors[i] = context

# Afficher les matrices
st.write("Matrice des scores d'attention:")
scores_df = pd.DataFrame(attention_scores.numpy(), index=tokens, columns=tokens)
scores_styled = scores_df.style.format("{:.4f}").apply(
    lambda df: pd.Series(['background-color: #e6f3ff' if df.name == tokens[query_index] else '' 
                         for _ in range(len(df.index))], index=df.index), axis=1
)
st.dataframe(scores_styled)

st.write("Matrice des poids d'attention:")
weights_df = pd.DataFrame(attention_weights.numpy(), index=tokens, columns=tokens)
weights_styled = weights_df.style.format("{:.4f}").apply(
    lambda df: pd.Series(['background-color: #e6f3ff' if df.name == tokens[query_index] else '' 
                         for _ in range(len(df.index))], index=df.index), axis=1
)
st.dataframe(weights_styled)

# Visualisation des poids d'attention
st.subheader("Visualisation des poids d'attention")
weights_viz_df = pd.DataFrame(
    attention_weights.numpy(),
    index=tokens,
    columns=tokens
)[tokens]
st.bar_chart(weights_viz_df)

with st.expander("Comment utiliser"):
    st.write("""
    1. Entrez une séquence de mots dans la zone de texte
    2. Sélectionnez un mot de requête avec le curseur
    3. Observez :
       - Les matrices de poids W_q, W_k, W_v
       - Les projections Q, K, V
       - Les scores et poids d'attention
       - Les matrices d'attention complètes
    """) 