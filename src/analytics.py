import os
import csv
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors, QED
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def fingerprints_to_numpy(fps):
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_diversity_score(smiles_list):
    fps = [smiles_to_fingerprint(s) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return 0.0
    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            distances.append(1.0 - tanimoto_similarity(fps[i], fps[j]))
    return float(np.mean(distances))


def compute_batch_stats(results, total_attempts):
    smiles_list = [r["smiles"]  for r in results]
    mw_list     = [r["mw"]     for r in results]
    logp_list   = [r["logp"]   for r in results]
    qed_list    = [r["qed"]    for r in results if r.get("qed") is not None]
    lip_pass    = sum(1 for r in results if r.get("lipinski"))

    return {
        "validity_rate":   round(len(results) / max(total_attempts, 1) * 100, 1),
        "diversity":       round(compute_diversity_score(smiles_list), 3),
        "avg_mw":          round(float(np.mean(mw_list)),   2) if mw_list   else 0,
        "avg_logp":        round(float(np.mean(logp_list)), 2) if logp_list else 0,
        "avg_qed":         round(float(np.mean(qed_list)),  3) if qed_list  else 0,
        "lipinski_pass":   lip_pass,
        "lipinski_rate":   round(100 * lip_pass / max(len(results), 1), 1),
        "mw_list":         mw_list,
        "logp_list":       logp_list,
        "qed_list":        qed_list,
        "num_valid":       len(results),
        "num_attempts":    total_attempts,
    }


def compute_tsne_embedding(generated_smiles, generated_props,
                           reference_smiles=None, reference_names=None,
                           perplexity=5):
    if not SKLEARN_AVAILABLE:
        return []

    all_smiles = list(generated_smiles)
    labels     = [f"Generated #{i+1}" for i in range(len(generated_smiles))]
    groups     = ["Generated"] * len(generated_smiles)
    mw_vals    = [p.get("mw",   0) for p in generated_props]
    logp_vals  = [p.get("logp", 0) for p in generated_props]

    if reference_smiles:
        all_smiles.extend(reference_smiles)
        labels.extend(reference_names or
                      [f"Drug #{i+1}" for i in range(len(reference_smiles))])
        groups.extend(["Known Drug"] * len(reference_smiles))
        for smi in reference_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw_vals.append(round(Descriptors.MolWt(mol),    2))
                logp_vals.append(round(Descriptors.MolLogP(mol), 2))
            else:
                mw_vals.append(0)
                logp_vals.append(0)

    fps           = []
    valid_indices = []
    for i, smi in enumerate(all_smiles):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)

    if len(fps) < 3:
        return []

    X              = fingerprints_to_numpy(fps)
    effective_perp = max(2, min(perplexity, len(fps) - 1))
    tsne           = TSNE(n_components=2, perplexity=effective_perp,
                          random_state=42, max_iter=1000, init="random")
    coords         = tsne.fit_transform(X)

    points = []
    for idx_in_fps, orig_idx in enumerate(valid_indices):
        points.append({
            "x":      float(coords[idx_in_fps, 0]),
            "y":      float(coords[idx_in_fps, 1]),
            "smiles": all_smiles[orig_idx],
            "label":  labels[orig_idx],
            "group":  groups[orig_idx],
            "mw":     mw_vals[orig_idx],
            "logp":   logp_vals[orig_idx],
        })
    return points


def load_known_drugs(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "known_drugs.csv"
        )
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_closest_drugs(generated_smiles_list, top_k=3, csv_path=None):
    drugs = load_known_drugs(csv_path)
    if not drugs:
        return []

    drug_fps = []
    for d in drugs:
        fp = smiles_to_fingerprint(d["SMILES"])
        if fp is not None:
            drug_fps.append((d, fp))

    results = []
    for gen_smi in generated_smiles_list:
        gen_fp = smiles_to_fingerprint(gen_smi)
        if gen_fp is None:
            results.append({"generated": gen_smi, "matches": []})
            continue

        similarities = sorted([
            {
                "name":       drug["Name"],
                "smiles":     drug["SMILES"],
                "category":   drug.get("Category", "Unknown"),
                "similarity": round(tanimoto_similarity(gen_fp, drug_fp) * 100, 1),
            }
            for drug, drug_fp in drug_fps
        ], key=lambda x: x["similarity"], reverse=True)

        results.append({
            "generated": gen_smi,
            "matches":   similarities[:top_k],
        })
    return results