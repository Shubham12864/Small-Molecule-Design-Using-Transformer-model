import zipfile
import os

def create_kaggle_zip(output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk('.'):
            # Skip folders that shouldn't be in the zip
            if any(p in root for p in ['venv', '.git', '__pycache__', '.ipynb_checkpoints']):
                continue
            
            for file in files:
                # Skip existing zip files and the script itself
                if file.endswith('.zip') or file == 'make_zip.py':
                    continue
                
                full_path = os.path.join(root, file)
                # This is the key: Replace backslashes with forward slashes for Linux/Kaggle
                relative_path = os.path.relpath(full_path, '.').replace('\\', '/')
                z.write(full_path, relative_path)
    print(f"Successfully created {output_filename} with Linux-style paths!")

if __name__ == "__main__":
    create_kaggle_zip('molecule_transformer_kaggle.zip')
