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

def initialize_head_weights(d_in, d_head, num_heads):
    """Initialise les matrices de poids pour chaque tête."""
    torch.manual_seed(123)
    heads = []
    for _ in range(num_heads):
        W_q = torch.nn.Parameter(torch.rand(d_in, d_head), requires_grad=False)
        W_k = torch.nn.Parameter(torch.rand(d_in, d_head), requires_grad=False)
        W_v = torch.nn.Parameter(torch.rand(d_in, d_head), requires_grad=False)
        heads.append((W_q, W_k, W_v))
    return heads

def calculate_attention(query, keys, values, mask=None):
    """Calcule l'attention pour une tête."""
    scores = torch.matmul(query, keys.t())
    if mask is not None:
        scores = scores + mask
    weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.matmul(weights, values)
    return scores, weights, context

st.title("Visualisation de l'Attention Multi-Têtes")

# Paramètres
input_text = st.text_area("Texte d'entrée", "Your journey starts with one step", height=100)
tokens = input_text.split()
num_heads = st.slider("Nombre de têtes d'attention", 2, 4, 2)
d_head = 2  # dimension de sortie pour chaque tête

# Obtenir et afficher les embeddings
embeddings = get_embeddings(tokens)
inputs = torch.stack([embeddings[token] for token in tokens])
d_in = inputs.shape[1]

st.subheader("Embeddings d'entrée")
embeddings_df = pd.DataFrame(
    inputs.numpy(),
    index=tokens,
    columns=["Dimension 1", "Dimension 2", "Dimension 3"]
)
st.dataframe(embeddings_df.style.format("{:.4f}"))

# Créer le masque causal
causal_mask = torch.triu(torch.ones(len(tokens), len(tokens)), diagonal=1)
causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

st.header("Approche 1: Têtes d'attention indépendantes")
st.write("Chaque tête a ses propres matrices de poids W_q, W_k, W_v")

# Initialiser les poids pour chaque tête
heads = initialize_head_weights(d_in, d_head, num_heads)

# Calcul pour chaque tête
all_context_vectors = []

for head_idx, (W_q, W_k, W_v) in enumerate(heads):
    st.subheader(f"Tête d'attention {head_idx + 1}")
    
    # Projections Q, K, V pour cette tête
    Q = torch.matmul(inputs, W_q)
    K = torch.matmul(inputs, W_k)
    V = torch.matmul(inputs, W_v)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Projections Q:")
        st.dataframe(pd.DataFrame(Q.numpy(), columns=[f"Q_{i+1}" for i in range(d_head)], index=tokens).style.format("{:.4f}"))
    with col2:
        st.write("Projections K:")
        st.dataframe(pd.DataFrame(K.numpy(), columns=[f"K_{i+1}" for i in range(d_head)], index=tokens).style.format("{:.4f}"))
    with col3:
        st.write("Projections V:")
        st.dataframe(pd.DataFrame(V.numpy(), columns=[f"V_{i+1}" for i in range(d_head)], index=tokens).style.format("{:.4f}"))
    
    # Calcul des poids d'attention causale
    attention_weights = torch.zeros((len(tokens), len(tokens)))
    context_vectors = torch.zeros((len(tokens), d_head))
    
    for i in range(len(tokens)):
        _, weights, context = calculate_attention(Q[i], K, V, causal_mask[i])
        attention_weights[i] = weights
        context_vectors[i] = context
    
    st.write("Poids d'attention causale:")
    st.dataframe(pd.DataFrame(attention_weights.numpy(), index=tokens, columns=tokens).style.format("{:.4f}"))
    
    st.write("Vecteurs de contexte:")
    context_df = pd.DataFrame(
        context_vectors.numpy(),
        index=tokens,
        columns=[f"dim_{i+1}" for i in range(d_head)]
    )
    st.dataframe(context_df.style.format("{:.4f}"))
    
    all_context_vectors.append(context_vectors)

# Concaténation des vecteurs de contexte
st.subheader("Vecteur de contexte final (concaténation)")
concatenated_context = torch.cat(all_context_vectors, dim=1)
concat_df = pd.DataFrame(
    concatenated_context.numpy(),
    index=tokens,
    columns=[f"head{h+1}_dim{d+1}" for h in range(num_heads) for d in range(d_head)]
)
st.dataframe(concat_df.style.format("{:.4f}"))

st.header("Approche 2: Attention Multi-Têtes Parallèle")
st.write("Une seule matrice de poids partagée entre les têtes")

# Initialiser les poids partagés
d_model = d_head * num_heads
W_qkv = torch.rand(3, d_in, d_model)  # Poids unique pour Q, K, V

# Projections uniques
st.subheader("Projections Q, K, V uniques")
Q = torch.matmul(inputs, W_qkv[0])  # [seq_len, d_model]
K = torch.matmul(inputs, W_qkv[1])  # [seq_len, d_model]
V = torch.matmul(inputs, W_qkv[2])  # [seq_len, d_model]

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Q unique:")
    st.dataframe(pd.DataFrame(
        Q.numpy(),
        index=tokens,
        columns=[f"Q_{i+1}" for i in range(d_model)]
    ).style.format("{:.4f}"))
with col2:
    st.write("K unique:")
    st.dataframe(pd.DataFrame(
        K.numpy(),
        index=tokens,
        columns=[f"K_{i+1}" for i in range(d_model)]
    ).style.format("{:.4f}"))
with col3:
    st.write("V unique:")
    st.dataframe(pd.DataFrame(
        V.numpy(),
        index=tokens,
        columns=[f"V_{i+1}" for i in range(d_model)]
    ).style.format("{:.4f}"))

# Partage entre les têtes
st.subheader("Partage des projections entre les têtes")
for h in range(num_heads):
    st.write(f"Tête {h+1}:")
    start_idx = h * d_head
    end_idx = (h + 1) * d_head
    
    head_projections = {
        'Q': Q[:, start_idx:end_idx],
        'K': K[:, start_idx:end_idx],
        'V': V[:, start_idx:end_idx]
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Q:")
        st.dataframe(pd.DataFrame(
            head_projections['Q'].numpy(),
            index=tokens,
            columns=[f"Q_{i+1}" for i in range(d_head)]
        ).style.format("{:.4f}"))
    with col2:
        st.write("K:")
        st.dataframe(pd.DataFrame(
            head_projections['K'].numpy(),
            index=tokens,
            columns=[f"K_{i+1}" for i in range(d_head)]
        ).style.format("{:.4f}"))
    with col3:
        st.write("V:")
        st.dataframe(pd.DataFrame(
            head_projections['V'].numpy(),
            index=tokens,
            columns=[f"V_{i+1}" for i in range(d_head)]
        ).style.format("{:.4f}"))

# Calcul de l'attention pour chaque tête
all_head_outputs = []
for h in range(num_heads):
    start_idx = h * d_head
    end_idx = (h + 1) * d_head
    
    Q_h = Q[:, start_idx:end_idx]
    K_h = K[:, start_idx:end_idx]
    V_h = V[:, start_idx:end_idx]
    
    head_context = torch.zeros((len(tokens), d_head))
    for i in range(len(tokens)):
        _, _, context = calculate_attention(Q_h[i], K_h, V_h, causal_mask[i])
        head_context[i] = context
    
    all_head_outputs.append(head_context)

# Vecteur de contexte final (concaténation des sorties des têtes)
final_context = torch.cat(all_head_outputs, dim=1)
st.subheader("Vecteur de contexte final")
st.dataframe(pd.DataFrame(
    final_context.numpy(),
    index=tokens,
    columns=[f"dim_{i+1}" for i in range(d_model)]
).style.format("{:.4f}")) 