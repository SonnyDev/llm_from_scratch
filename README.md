# llm_from_scratch

## Contenu du projet

### Chapitre 2 : Tokenisation et échantillonnage

- `chapitre_02/bpe_visualization.py` : Visualisation du processus de tokenisation BPE (Byte-Pair Encoding)
- `chapitre_02/data_sampling_visualization.py` : Visualisation de l'échantillonnage des données pour l'entraînement
- `chapitre_02/ch02.ipynb` : Notebook Jupyter avec des exemples détaillés

### Chapitre 3 : Mécanismes d'attention

- `chapitre_03/self_attention_viz_1.py` : Visualisation du mécanisme d'auto-attention simple
- `chapitre_03/self_attention_viz_2.py` : Visualisation de l'auto-attention avec poids entraînables
- `chapitre_03/causal_attention.py` : Visualisation de l'attention causale avec et sans dropout
- `chapitre_03/multihead_attention.py` : Visualisation de l'attention multi-têtes avec deux approches différentes

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer une visualisation, utilisez Streamlit :
```bash
streamlit run chapitre_02/bpe_visualization.py
streamlit run chapitre_02/data_sampling_visualization.py
streamlit run chapitre_03/self_attention_viz_1.py
streamlit run chapitre_03/self_attention_viz_2.py
streamlit run chapitre_03/causal_attention.py
streamlit run chapitre_03/multihead_attention.py
```

Chaque visualisation est interactive et permet de :
- Entrer du texte personnalisé
- Ajuster différents paramètres
- Observer les résultats en temps réel
- Comprendre les calculs étape par étape