import pandas as pd
import re
from collections import Counter, defaultdict
import html
import streamlit as st

def get_stats(vocab):
    """Calcule les fréquences des paires de symboles."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """Fusionne une paire donnée dans le vocabulaire."""
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in v_in.items():
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = freq
    return v_out

def get_tokens(text):
    """Convertit le texte en tokens initiaux (caractères)."""
    return ' '.join(list(text))

def get_token_color(token, token_colors):
    """Retourne une couleur cohérente pour un token donné."""
    if token not in token_colors:
        # Utilisation de couleurs prédéfinies au lieu de couleurs aléatoires
        hue = hash(token) % 10 / 10  # 10 teintes différentes
        token_colors[token] = f"hsl({hue * 360}, 70%, 85%)"
    return token_colors[token]

def visualize_tokens(text, vocab, current_step, merge_history):
    """Visualise le texte avec des couleurs pour chaque token."""
    words = text.split()
    html_parts = []
    token_colors = {}
    
    for word in words:
        tokens = list(word)
        
        # Appliquer les fusions dans l'ordre
        for pair in merge_history[:current_step]:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    tokens[i:i + 2] = [''.join(pair)]
                else:
                    i += 1
        
        # Création du HTML pour le mot
        colored_tokens = [
            f'<span style="background-color: {get_token_color(token, token_colors)}; padding: 0 2px; border-radius: 3px; margin: 0 1px;">{html.escape(token)}</span>'
            for token in tokens
        ]
        html_parts.append("".join(colored_tokens))
    
    return " ".join(html_parts)

# Interface Streamlit
st.title("Visualisation BPE")

# Configuration de la mise en page
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Texte d'entraînement",
        value="low lower lowest newer wider newer",
        height=100
    )

with col2:
    num_merges = st.number_input("Nombre total d'itérations", 1, 100, 5)
    current_step = st.number_input("Itération courante", 0, num_merges, 0)
    start_button = st.button("Initialiser/Réinitialiser")

# Initialisation/Réinitialisation
if 'vocab' not in st.session_state or start_button or 'text' not in st.session_state or st.session_state.text != text_input:
    # Créer un vocabulaire initial avec des mots séparés en caractères
    vocab_init = {}
    for word in text_input.split():
        # Séparer le mot en caractères avec des espaces
        chars = ' '.join(list(word))
        if chars in vocab_init:
            vocab_init[chars] += 1
        else:
            vocab_init[chars] = 1
    
    st.session_state.vocab = vocab_init
    st.session_state.merge_history = []
    st.session_state.text = text_input

# Affichage principal
col_stats, col_fusions = st.columns(2)

with col_stats:
    st.subheader(f"Itération {current_step}")
    pairs = get_stats(st.session_state.vocab)
    if pairs:
        pairs_df = pd.DataFrame(
            [(f"({p[0]}, {p[1]})", freq) for p, freq in pairs.items()],
            columns=["Paire", "Fréquence"]
        ).sort_values("Fréquence", ascending=False)
        st.dataframe(pairs_df, height=200)

with col_fusions:
    if st.session_state.merge_history:
        st.subheader("Fusions effectuées")
        # Créer une liste de fusions formatées
        fusions = [
            {
                "N°": i+1,
                "Fusion": f"{merge[0]} + {merge[1]} → {''.join(merge)}"
            }
            for i, merge in enumerate(st.session_state.merge_history[:current_step])
        ]
        # Créer et afficher le DataFrame avec une colonne plus large
        fusions_df = pd.DataFrame(fusions)
        st.dataframe(
            fusions_df.set_index("N°"), 
            height=200,
            column_config={
                "Fusion": st.column_config.Column(
                    width=300  # Ajuster cette valeur selon vos besoins
                )
            }
        )

# Mise à jour des fusions (garder cette partie hors des colonnes)
if current_step > len(st.session_state.merge_history):
    if pairs:
        best_pair = max(pairs.items(), key=lambda x: x[1])[0]
        st.session_state.vocab = merge_vocab(best_pair, st.session_state.vocab)
        st.session_state.merge_history.append(best_pair)

# Visualisation du texte tokenisé
st.markdown("### Texte tokenisé")
colored_text = visualize_tokens(
    st.session_state.text,
    st.session_state.vocab,
    current_step,
    st.session_state.merge_history
)
st.markdown(f'<div style="padding: 10px; background-color: white; border-radius: 5px;">{colored_text}</div>', unsafe_allow_html=True)

# Instructions dans un expander en bas
with st.expander("Comment utiliser"):
    st.write("""
    1. Entrez votre texte d'entraînement
    2. Définissez le nombre total d'itérations
    3. Utilisez le compteur d'itération pour avancer pas à pas
    4. Cliquez sur 'Initialiser' pour recommencer
    """)

# Afficher le vocabulaire final à la dernière itération
if current_step == num_merges:
    st.subheader("Vocabulaire final")
    final_vocab = defaultdict(int)
    
    # Tokeniser le texte complet avec toutes les fusions appliquées
    words = text_input.split()
    for word in words:
        tokens = list(word)
        # Appliquer toutes les fusions
        for pair in st.session_state.merge_history[:current_step]:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    tokens[i:i + 2] = [''.join(pair)]
                else:
                    i += 1
        # Compter la fréquence de chaque token dans le résultat final
        for token in tokens:
            final_vocab[token] += 1
    
    # Afficher le vocabulaire final dans un DataFrame
    final_vocab_df = pd.DataFrame(
        final_vocab.items(),
        columns=["Token", "Fréquence"]
    ).sort_values("Fréquence", ascending=False)
    st.dataframe(final_vocab_df)
