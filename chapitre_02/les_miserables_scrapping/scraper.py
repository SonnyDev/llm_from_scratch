import requests
from bs4 import BeautifulSoup
import re
import time
import os

def nettoyer_texte(texte):
    # Supprime les balises HTML et les espaces superflus
    texte = re.sub(r'\n+', '\n', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

def scraper_chapitre(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Trouve le contenu principal du chapitre
        contenu = soup.find('div', {'class': 'prp-pages-output'})
        if contenu:
            # Supprime les éléments de navigation et autres éléments non désirés
            for element in contenu.find_all(['table', 'div', 'span']):
                element.decompose()
                
            texte = nettoyer_texte(contenu.get_text())
            return texte
    except Exception as e:
        print(f"Erreur lors du scraping de {url}: {e}")
    return None

def creer_structure_tomes():
    structure = {}
    for tome in range(5, 6):
        structure[tome] = {}
        for livre in range(1, 10):  # Nombre approximatif de livres par tome
            structure[tome][livre] = range(1, 25)  # Nombre approximatif de chapitres par livre
    return structure

def main():
    base_url = "https://fr.wikisource.org/wiki/Les_Mis%C3%A9rables"
    output_dir = "les_miserables"
    os.makedirs(output_dir, exist_ok=True)

    structure = creer_structure_tomes()
    
    for tome, livres in structure.items():
        tome_dir = os.path.join(output_dir, f"Tome_{tome}")
        os.makedirs(tome_dir, exist_ok=True)
        
        for livre, chapitres in livres.items():
            # Pour le tome 5, on utilise le format sans zéro
            livre_str = str(livre)
            
            for chapitre in chapitres:
                url = f"{base_url}/Tome_{tome}/Livre_{livre_str}/{chapitre:02d}"
                
                # Vérifie si le chapitre existe
                texte = scraper_chapitre(url)
                if texte:
                    fichier = os.path.join(tome_dir, f"livre_{livre}_chapitre_{chapitre:02d}.txt")
                    with open(fichier, 'w', encoding='utf-8') as f:
                        f.write(texte)
                    print(f"Chapitre sauvegardé: {fichier}")
                    
                # Pause pour éviter de surcharger le serveur
                time.sleep(2)

if __name__ == "__main__":
    main() 