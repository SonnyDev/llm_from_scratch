import os

def get_file_numbers(filename):
    if not filename.startswith('livre_'):
        return (float('inf'), float('inf'))
    parts = filename.split('_')
    return (int(parts[1]), int(parts[3].split('.')[0]))

def combine_files():
    output_file = "les_miserables_complet.txt"
    base_dir = "les_miserables"
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Parcourir les tomes dans l'ordre
        for tome in range(1, 6):
            tome_dir = os.path.join(base_dir, f"Tome_{tome}")
            if not os.path.exists(tome_dir):
                continue
                
            outfile.write(f"\n{'='*80}\n")
            outfile.write(f"TOME {tome}\n")
            outfile.write(f"{'='*80}\n\n")
            
            # Obtenir tous les fichiers du tome et les trier
            files = sorted(os.listdir(tome_dir), key=get_file_numbers)
            
            current_livre = None
            
            for filename in files:
                if not filename.startswith('livre_'):
                    continue
                    
                # Extraire le numéro du livre
                livre_num = int(filename.split('_')[1])
                
                # Si on change de livre, écrire l'en-tête du nouveau livre
                if livre_num != current_livre:
                    outfile.write(f"\n{'-'*60}\n")
                    outfile.write(f"LIVRE {livre_num}\n")
                    outfile.write(f"{'-'*60}\n\n")
                    current_livre = livre_num
                
                # Écrire le contenu du chapitre
                filepath = os.path.join(tome_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as infile:
                    chapitre_num = int(filename.split('_')[3].split('.')[0])
                    outfile.write(f"CHAPITRE {chapitre_num}\n\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")
            
            # Si c'est le tome 5, ajouter les notes à la fin
            if tome == 5:
                notes_file = os.path.join(tome_dir, "notes.txt")
                if os.path.exists(notes_file):
                    outfile.write(f"\n{'-'*60}\n")
                    outfile.write("NOTES\n")
                    outfile.write(f"{'-'*60}\n\n")
                    with open(notes_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")

if __name__ == "__main__":
    combine_files() 