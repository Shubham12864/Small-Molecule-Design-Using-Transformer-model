import argparse
import sys
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[Warning] RDKit not installed. Install with: pip install rdkit-pypi")
def compute_properties(smiles):
    result = {
        "smiles": smiles,
        "valid": False,
        "molecular_weight": None,
        "logp": None,
    }
    if not RDKIT_AVAILABLE:
        print("[Error] RDKit is not available.")
        return result
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result
    result["valid"] = True
    result["molecular_weight"] = round(Descriptors.MolWt(mol), 2)
    result["logp"] = round(Descriptors.MolLogP(mol), 2)
    return result
def predict_properties(smiles_list):
    results = []
    for smi in smiles_list:
        props = compute_properties(smi)
        results.append(props)
    return results
def print_results(results):
    print(f"\n{'SMILES':<40} {'Valid':<8} {'Mol. Weight':<14} {'LogP'}")
    print("-" * 75)
    for r in results:
        valid_str = "YES" if r["valid"] else "NO"
        mw_str = f"{r['molecular_weight']:.2f}" if r["molecular_weight"] is not None else "N/A"
        logp_str = f"{r['logp']:.2f}" if r["logp"] is not None else "N/A"
        print(f"{r['smiles']:<40} {valid_str:<8} {mw_str:<14} {logp_str}")
def parse_args():
    parser = argparse.ArgumentParser(description="Predict molecular properties")
    parser.add_argument(
        "--smiles",
        nargs="+",
        default=None,
        help="SMILES strings to analyze (space-separated)",
    )
    return parser.parse_args()
def main():
    args = parse_args()
    if args.smiles:
        smiles_list = args.smiles
    else:
        smiles_list = [
            "CCO",                      
            "c1ccccc1",                  
            "CC(=O)O",                  
            "CC(=O)Oc1ccccc1C(=O)O",   
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  
            "INVALID_SMILES",            
            "CC(=O)NC1=CC=C(O)C=C1",    
        ]
        print("Using default example molecules.")
        print("Tip: Pass your own with --smiles 'CCO' 'c1ccccc1'\n")
    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required for property prediction.")
        print("Install with: pip install rdkit-pypi")
        return
    results = predict_properties(smiles_list)
    print_results(results)
    valid_count = sum(1 for r in results if r["valid"])
    print(f"\nTotal: {len(results)} molecules | Valid: {valid_count} | Invalid: {len(results) - valid_count}")
if __name__ == "__main__":
    main()
