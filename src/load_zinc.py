import argparse
import zipfile
import csv
import os
import io
def parse_args():
    parser = argparse.ArgumentParser(description="Load ZINC dataset from zip file")
    parser.add_argument("--zip_path", type=str, required=True, help="Path to ZINC zip file")
    parser.add_argument("--max_molecules", type=int, default=None, help="Max molecules to extract (None = all)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: data/smiles.txt)")
    return parser.parse_args()
def extract_smiles_from_txt(content):
    smiles = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            smi = line.split()[0]
            if any(c in smi for c in "CcNnOoSsFBrClI"):
                smiles.append(smi)
    return smiles
def extract_smiles_from_csv(content):
    smiles = []
    reader = csv.reader(io.StringIO(content))
    header = next(reader, None)
    if header is None:
        return smiles
    smiles_col = None
    for idx, col_name in enumerate(header):
        if col_name.strip().lower() in ("smiles", "smi", "canonical_smiles", "smile"):
            smiles_col = idx
            break
    if smiles_col is None:
        print("  [Info] No 'smiles' column found in header, using first column")
        smiles_col = 0
        first_val = header[0].strip()
        if any(c in first_val for c in "CcNnOoSsFBrClI"):
            smiles.append(first_val)
    for row in reader:
        if len(row) > smiles_col:
            smi = row[smiles_col].strip()
            if smi and any(c in smi for c in "CcNnOoSsFBrClI"):
                smiles.append(smi)
    return smiles
def load_zinc_zip(zip_path, max_molecules=None):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    all_smiles = []
    print(f"Opening zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()
        print(f"Files in zip: {len(file_list)}")
        data_files = [f for f in file_list if f.endswith((".txt", ".csv", ".smi", ".tsv", ".sdf"))]
        if not data_files:
            data_files = [f for f in file_list if not f.endswith("/") and "__MACOSX" not in f]
        print(f"Data files found: {len(data_files)}")
        for filename in data_files:
            print(f"  Processing: {filename}")
            try:
                with zf.open(filename) as f:
                    raw = f.read()
                    try:
                        content = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        content = raw.decode("latin-1")
                if filename.endswith(".csv"):
                    smiles = extract_smiles_from_csv(content)
                else:
                    smiles = extract_smiles_from_txt(content)
                print(f"    -> Extracted {len(smiles)} SMILES")
                all_smiles.extend(smiles)
            except Exception as e:
                print(f"    -> Error: {e}")
            if max_molecules and len(all_smiles) >= max_molecules:
                all_smiles = all_smiles[:max_molecules]
                break
    seen = set()
    unique_smiles = []
    for smi in all_smiles:
        if smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)
    if max_molecules:
        unique_smiles = unique_smiles[:max_molecules]
    print(f"\nTotal unique SMILES extracted: {len(unique_smiles)}")
    return unique_smiles
def save_smiles(smiles_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")
    print(f"Saved {len(smiles_list)} SMILES to: {output_path}")
def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = args.output or os.path.join(project_root, "data", "smiles.txt")
    smiles = load_zinc_zip(args.zip_path, max_molecules=args.max_molecules)
    if not smiles:
        print("ERROR: No SMILES found in the zip file!")
        print("Make sure the zip contains .txt, .csv, or .smi files with SMILES strings.")
        return
    save_smiles(smiles, output_path)
    print(f"\nDone! You can now train the model:")
    print(f"  python src/train.py --epochs 10")
if __name__ == "__main__":
    main()
