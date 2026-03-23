import argparse
import zipfile
import csv
import os
import io

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[Warning] RDKit not available. Skipping chemical validation.")


def parse_args():
    parser = argparse.ArgumentParser(description="Load ZINC dataset from zip file")
    parser.add_argument("--zip_path",       type=str, required=True)
    parser.add_argument("--max_molecules",  type=int, default=None)
    parser.add_argument("--output",         type=str, default=None)
    parser.add_argument("--min_len",        type=int, default=5,
                        help="Minimum SMILES length to keep")
    parser.add_argument("--max_len",        type=int, default=120,
                        help="Maximum SMILES length to keep")
    parser.add_argument("--validate",       action="store_true", default=True,
                        help="Validate SMILES with RDKit")
    return parser.parse_args()


def is_valid_smiles(smi):
    if not RDKIT_AVAILABLE:
        return bool(smi and any(c in smi for c in "CcNnOoSsFBrClI"))
    mol = Chem.MolFromSmiles(smi)
    return mol is not None


def extract_smiles_from_txt(content, min_len=5, max_len=120, validate=True):
    smiles   = []
    rejected = 0
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smi = line.split()[0]
        if len(smi) < min_len or len(smi) > max_len:
            rejected += 1
            continue
        if validate and not is_valid_smiles(smi):
            rejected += 1
            continue
        smiles.append(smi)
    return smiles, rejected


def extract_smiles_from_csv(content, min_len=5, max_len=120, validate=True):
    smiles   = []
    rejected = 0
    reader   = csv.reader(io.StringIO(content))
    header   = next(reader, None)
    if header is None:
        return smiles, rejected

    smiles_col = None
    for idx, col_name in enumerate(header):
        if col_name.strip().lower() in ("smiles", "smi", "canonical_smiles", "smile"):
            smiles_col = idx
            break

    if smiles_col is None:
        smiles_col = 0
        first_val  = header[0].strip()
        if len(first_val) >= min_len and is_valid_smiles(first_val):
            smiles.append(first_val)

    for row in reader:
        if len(row) <= smiles_col:
            continue
        smi = row[smiles_col].strip()
        if not smi:
            continue
        if len(smi) < min_len or len(smi) > max_len:
            rejected += 1
            continue
        if validate and not is_valid_smiles(smi):
            rejected += 1
            continue
        smiles.append(smi)

    return smiles, rejected


def load_zinc_zip(zip_path, max_molecules=None, min_len=5, max_len=120, validate=True):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    seen          = set()
    unique_smiles = []
    total_rejected= 0

    print(f"[LoadZinc] Opening: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list  = zf.namelist()
        data_files = [
            f for f in file_list
            if f.endswith((".txt", ".csv", ".smi", ".tsv"))
            and "__MACOSX" not in f
            and not f.endswith("/")
        ]

        if not data_files:
            data_files = [
                f for f in file_list
                if not f.endswith("/") and "__MACOSX" not in f
            ]

        print(f"[LoadZinc] Files found : {len(data_files)}")

        for file_idx, filename in enumerate(data_files, 1):
            print(f"[LoadZinc] [{file_idx}/{len(data_files)}] Processing: {filename}")

            try:
                with zf.open(filename) as f:
                    raw = f.read()
                    try:
                        content = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        content = raw.decode("latin-1")

                if filename.endswith(".csv") or filename.endswith(".tsv"):
                    smiles, rejected = extract_smiles_from_csv(
                        content, min_len, max_len, validate
                    )
                else:
                    smiles, rejected = extract_smiles_from_txt(
                        content, min_len, max_len, validate
                    )

                total_rejected += rejected
                added = 0
                for smi in smiles:
                    if smi not in seen:
                        seen.add(smi)
                        unique_smiles.append(smi)
                        added += 1

                print(f"           Added: {added:,} | "
                      f"Rejected: {rejected:,} | "
                      f"Total so far: {len(unique_smiles):,}")

            except Exception as e:
                print(f"           Error: {e}")

            if max_molecules and len(unique_smiles) >= max_molecules:
                unique_smiles = unique_smiles[:max_molecules]
                break

    lengths  = [len(s) for s in unique_smiles]
    avg_len  = round(sum(lengths) / len(lengths), 1) if lengths else 0
    max_seen = max(lengths) if lengths else 0
    min_seen = min(lengths) if lengths else 0

    print(f"\n[LoadZinc] Extraction complete")
    print(f"[LoadZinc] Unique SMILES  : {len(unique_smiles):,}")
    print(f"[LoadZinc] Total rejected : {total_rejected:,}")
    print(f"[LoadZinc] Avg length     : {avg_len}")
    print(f"[LoadZinc] Min length     : {min_seen}")
    print(f"[LoadZinc] Max length     : {max_seen}")

    return unique_smiles


def save_smiles(smiles_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")
    print(f"[LoadZinc] Saved {len(smiles_list):,} SMILES to: {output_path}")


def main():
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path  = args.output or os.path.join(project_root, "data", "smiles.txt")

    smiles = load_zinc_zip(
        zip_path      = args.zip_path,
        max_molecules = args.max_molecules,
        min_len       = args.min_len,
        max_len       = args.max_len,
        validate      = args.validate,
    )

    if not smiles:
        print("[LoadZinc] ERROR: No SMILES extracted.")
        print("[LoadZinc] Check that zip contains .txt, .csv, or .smi files.")
        return

    save_smiles(smiles, output_path)

    print(f"\n[OK] Done! Now train with:")
    print(f"     python src/train.py --epochs 20")


if __name__ == "__main__":
    main()