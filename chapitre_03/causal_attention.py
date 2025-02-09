import streamlit as st
import torch
import numpy as np
import pandas as pd

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

def create_causal_mask(size):
    """Crée un masque d'attention causale."""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def calculate_attention(query, keys, values, mask=None, dropout_prob=None):
    """Calcule l'attention avec masque causal optionnel et dropout."""
    scores = torch.matmul(query, keys.t())
    
    if mask is not None:
        scores = scores + mask
    
    weights = torch.nn.functional.softmax(scores, dim=-1)
    
    if dropout_prob is not None:
        weights = torch.nn.functional.dropout(weights, p=dropout_prob)
    
    context = torch.matmul(weights, values)
    return scores, weights, context

st.title("Visualisation de l'Attention Causale")

# Saisie de texte
input_text = st.text_area("Texte d'entrée", "Your journey starts with one step", height=100)
tokens = input_text.split()

# Options
col1, col2 = st.columns(2)
with col1:
    use_dropout = st.checkbox("Utiliser le dropout", value=False)
with col2:
    dropout_prob = st.slider("Probabilité de dropout", 0.0, 0.5, 0.1, disabled=not use_dropout)

# Obtenir les embeddings
embeddings = get_embeddings(tokens)
inputs = torch.stack([embeddings[token] for token in tokens])

# Créer le masque causal
causal_mask = create_causal_mask(len(tokens))

# Calcul des matrices d'attention
attention_scores = torch.zeros((len(tokens), len(tokens)))
attention_weights = torch.zeros((len(tokens), len(tokens)))

for i in range(len(tokens)):
    scores, weights, _ = calculate_attention(
        inputs[i], 
        inputs, 
        inputs,
        mask=causal_mask[i],
        dropout_prob=dropout_prob if use_dropout else None
    )
    attention_scores[i] = scores
    attention_weights[i] = weights

# Affichage des matrices
col1, col2 = st.columns(2)

with col1:
    st.subheader("Matrice des scores d'attention")
    scores_df = pd.DataFrame(
        attention_scores.numpy(),
        index=tokens,
        columns=tokens
    )
    st.dataframe(scores_df.style.format("{:.2f}").apply(
        lambda x: ['background-color: #f5f5f5' if v == float('-inf') else '' 
                  for v in x], axis=1
    ))

with col2:
    st.subheader("Matrice des poids d'attention")
    weights_df = pd.DataFrame(
        attention_weights.numpy(),
        index=tokens,
        columns=tokens
    )
    st.dataframe(weights_df.style.format("{:.2f}"))

# Visualisation des poids d'attention
st.subheader("Visualisation des poids d'attention")
weights_viz_df = pd.DataFrame(
    attention_weights.numpy(),
    index=tokens,
    columns=tokens
)[tokens]
st.bar_chart(weights_viz_df)

# Explication
with st.expander("Comment fonctionne l'attention causale"):
    st.write("""
    1. Le masque causal empêche chaque token d'accéder aux tokens futurs
    2. Les éléments au-dessus de la diagonale sont masqués avec -∞
    3. Après softmax, ces positions ont un poids de 0
    4. Le dropout (optionnel) désactive aléatoirement certaines connexions
    """) 