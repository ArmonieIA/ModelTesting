import os
import shutil

def convert_and_delete_files_recursive(directory):
    # Parcourir le dossier racine et tous les sous-dossiers
    for root, dirs, files in os.walk(directory):
        # Parcourir chaque fichier dans les dossiers
        for file in files:
            # Vérifier si le fichier a l'extension .mbr ou .rpg
            if file.endswith('.mbr') or file.endswith('.rpg') or file.endswith('.MBR') or file.endswith('.RPG'):
                # Construire le chemin complet vers le fichier
                full_file_path = os.path.join(root, file)
                # Créer le nom de fichier de sortie avec l'extension .txt
                new_file_name = os.path.splitext(full_file_path)[0] + '.txt'
                # Copier le fichier en changeant l'extension
                shutil.copy(full_file_path, new_file_name)
                print(f'Converted {full_file_path} to {new_file_name}')
                
                # Si le fichier est un .rpg, le supprimer après la conversion
                if file.endswith('.rpg') or file.endswith('.mbr') or file.endswith('.MBR') or file.endswith('.RPG'):
                    os.remove(full_file_path)
                    print(f'Deleted {full_file_path}')

# Spécifiez le chemin vers le dossier racine contenant les sous-dossiers avec les fichiers RPG3
directory_path = r'path'
convert_and_delete_files_recursive(directory_path)
