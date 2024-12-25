import streamlit as st
import re
from collections import Counter, defaultdict
import pandas as pd
import random
import html
from colorsys import hsv_to_rgb

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

def get_color():
    """Génère une couleur pastel aléatoire."""
    hue = random.random()
    saturation = 0.3 + random.random() * 0.2
    value = 0.9 + random.random() * 0.1
    rgb = tuple(round(i * 255) for i in hsv_to_rgb(hue, saturation, value))
    return f"rgb{rgb}"

def visualize_tokens(text, vocab):
    """Visualise le texte avec des couleurs pour chaque token."""
    words = st.session_state.text.split()
    html_parts = []
    
    # Dictionnaire des couleurs pour les tokens
    token_colors = {}
    
    for word in words:
        # Commencer avec les caractères individuels
        tokens = list(word)
        
        # Appliquer les fusions dans l'ordre
        for pair in st.session_state.merge_history[:current_step]:
            i = 0
            while i < len(tokens) - 1:
                # Vérifier si la paire actuelle correspond à la fusion
                current_pair = (tokens[i], tokens[i + 1])
                if current_pair == pair:
                    # Fusionner les tokens
                    tokens[i:i + 2] = [''.join(pair)]
                else:
                    i += 1
        
        # Attribution des couleurs aux tokens
        for token in tokens:
            if token not in token_colors:
                token_colors[token] = get_color()
        
        # Création du HTML pour le mot
        colored_tokens = [
            f'<span style="background-color: {token_colors[token]}; padding: 0 2px; border-radius: 3px; margin: 0 1px;">{html.escape(token)}</span>'
            for token in tokens
        ]
        html_parts.append("".join(colored_tokens))
    
    return " ".join(html_parts)

st.title("Visualisation BPE")

# Configuration de la mise en page
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de texte pour l'entrée
    text_input = st.text_area(
        "Texte d'entraînement",
        value="low lower lowest newer wider newer",
        height=100,
        key="input_text"
    )

with col2:
    # Contrôles
    num_merges = st.number_input("Nombre total d'itérations", 1, 100, 5)
    current_step = st.number_input("Itération courante", 0, num_merges, 0)
    start_button = st.button("Initialiser/Réinitialiser")

# État de l'application
if ('vocab' not in st.session_state or 
    start_button or 
    'last_text' not in st.session_state or 
    st.session_state.last_text != text_input):  # Vérifie si le texte a changé
    
    words = text_input.split()
    # Initialisation avec les caractères individuels
    vocab_init = {}
    for word in words:
        word_tokens = ' '.join(list(word))  # Sépare le mot en caractères
        vocab_init[word_tokens] = Counter(words)[word]
    
    st.session_state.vocab = vocab_init
    st.session_state.merge_history = []
    st.session_state.initialized = True
    st.session_state.text = text_input  # Sauvegarde du texte initial
    st.session_state.last_text = text_input  # Sauvegarde pour comparaison

if 'initialized' in st.session_state:
    # Affichage de l'état actuel
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(f"Itération {current_step}")
        
        # Calcul et affichage des statistiques
        pairs = get_stats(st.session_state.vocab)
        if pairs:
            pairs_df = pd.DataFrame(
                [(f"({p[0]}, {p[1]})", freq) for p, freq in pairs.items()],
                columns=["Paire", "Fréquence"]
            ).sort_values("Fréquence", ascending=False)
            st.dataframe(pairs_df, height=200)

    with col4:
        st.subheader("Vocabulaire actuel")
        vocab_df = pd.DataFrame(
            st.session_state.vocab.items(),
            columns=["Token", "Fréquence"]
        )
        st.dataframe(vocab_df, height=200)

    # Mise à jour pour l'itération suivante
    if current_step > len(st.session_state.merge_history):
        if pairs:
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            st.session_state.vocab = merge_vocab(best_pair, st.session_state.vocab)
            st.session_state.merge_history.append(best_pair)
            
    # Affichage des fusions effectuées
    if st.session_state.merge_history:
        st.subheader("Fusions effectuées")
        for i, merge in enumerate(st.session_state.merge_history[:current_step]):
            st.write(f"{i+1}. {merge[0]} + {merge[1]} → {''.join(merge)}")

    # Affichage du texte tokenisé avec des couleurs
    st.markdown("### Texte tokenisé")
    colored_text = visualize_tokens(st.session_state.text, st.session_state.vocab)
    st.markdown(f'<div style="padding: 10px; background-color: white; border-radius: 5px;">{colored_text}</div>', unsafe_allow_html=True)

# Instructions dans un expander en bas
with st.expander("Comment utiliser"):
    st.write("""
    1. Entrez votre texte d'entraînement
    2. Définissez le nombre total d'itérations
    3. Utilisez le compteur d'itération pour avancer pas à pas
    4. Cliquez sur 'Initialiser' pour recommencer
    """)
