import argparse
import sys

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, RDConfig
    from rdkit import RDLogger
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
    SASCORER_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE      = False
    SASCORER_AVAILABLE   = False


def compute_properties(smiles):
    result = {
        "smiles":    smiles,
        "valid":     False,
        "mw":        None,
        "logp":      None,
        "qed":       None,
        "sa_score":  None,
        "tpsa":      None,
        "hbd":       None,
        "hba":       None,
        "rot_bonds": None,
        "rings":     None,
        "lipinski":  None,
    }

    if not RDKIT_AVAILABLE:
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    mw   = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)

    result["valid"]     = True
    result["mw"]        = mw
    result["logp"]      = logp
    result["qed"]       = round(QED.qed(mol), 3)
    result["tpsa"]      = round(Descriptors.TPSA(mol), 2)
    result["hbd"]       = hbd
    result["hba"]       = hba
    result["rot_bonds"] = Descriptors.NumRotatableBonds(mol)
    result["rings"]     = Descriptors.RingCount(mol)
    result["lipinski"]  = (
        mw   <= 500 and
        logp <= 5   and
        hbd  <= 5   and
        hba  <= 10
    )

    if SASCORER_AVAILABLE:
        try:
            result["sa_score"] = round(sascorer.calculateScore(mol), 2)
        except Exception:
            pass

    return result


def predict_properties(smiles_list):
    return [compute_properties(smi) for smi in smiles_list]


def get_summary(results):
    valid = [r for r in results if r["valid"]]
    if not valid:
        return None

    def avg(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    lip_pass = sum(1 for r in valid if r["lipinski"])

    return {
        "total":        len(results),
        "valid":        len(valid),
        "invalid":      len(results) - len(valid),
        "lipinski_pass": lip_pass,
        "lipinski_rate": round(100 * lip_pass / len(valid), 1),
        "avg_mw":       avg("mw"),
        "avg_logp":     avg("logp"),
        "avg_qed":      avg("qed"),
        "avg_sa":       avg("sa_score"),
        "avg_tpsa":     avg("tpsa"),
    }


def print_results(results):
    header = (f"\n{'SMILES':<45} {'Valid':<6} {'MW':<8} "
              f"{'LogP':<7} {'QED':<6} {'SA':<6} {'TPSA':<7} "
              f"{'HBD':<5} {'HBA':<5} {'Lipinski'}")
    print(header)
    print("-" * 110)

    for r in results:
        def fmt(val, decimals=2):
            return f"{val:.{decimals}f}" if val is not None else "N/A"

        lip = ("PASS" if r["lipinski"] else "FAIL") if r["lipinski"] is not None else "N/A"
        smiles_display = r["smiles"][:43] + ".." if len(r["smiles"]) > 45 else r["smiles"]

        print(f"{smiles_display:<45} "
              f"{'YES' if r['valid'] else 'NO':<6} "
              f"{fmt(r['mw']):<8} "
              f"{fmt(r['logp']):<7} "
              f"{fmt(r['qed'], 3):<6} "
              f"{fmt(r['sa_score']):<6} "
              f"{fmt(r['tpsa']):<7} "
              f"{str(r['hbd']):<5} "
              f"{str(r['hba']):<5} "
              f"{lip}")

    summary = get_summary(results)
    if summary:
        print(f"\n{'='*50}")
        print(f"  Total          : {summary['total']}")
        print(f"  Valid          : {summary['valid']}")
        print(f"  Invalid        : {summary['invalid']}")
        print(f"  Lipinski Pass  : {summary['lipinski_pass']} ({summary['lipinski_rate']}%)")
        print(f"  Avg QED        : {summary['avg_qed']}")
        print(f"  Avg MW         : {summary['avg_mw']}")
        print(f"  Avg LogP       : {summary['avg_logp']}")
        print(f"  Avg SA Score   : {summary['avg_sa']}")
        print(f"  Avg TPSA       : {summary['avg_tpsa']}")
        print(f"{'='*50}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict molecular properties")
    parser.add_argument("--smiles", nargs="+", default=None)
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
            "ClC1=CC=CC=C1",
            "BrC1=CC=CC=C1",
        ]
        print("Using default example molecules.")
        print("Tip: Pass your own with --smiles 'CCO' 'c1ccccc1'\n")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required. Install with: pip install rdkit-pypi")
        return []

    results = predict_properties(smiles_list)
    print_results(results)
    return results


if __name__ == "__main__":
    main()