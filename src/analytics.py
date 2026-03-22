"""
Analytics module for molecular generation batch analysis.
Provides fingerprint computation, similarity scoring, t-SNE embedding,
batch statistics, and known drug comparison.
"""

import os
import csv
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def fingerprints_to_numpy(fps):
    """Convert a list of RDKit fingerprints to a numpy array."""
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ---------------------------------------------------------------------------
# Batch Analytics  (Feature 5)
# ---------------------------------------------------------------------------

def compute_diversity_score(smiles_list):
    """
    Compute the average pairwise Tanimoto distance for a list of SMILES.
    Higher values → more diverse molecules.
    Returns a float between 0 and 1.
    """
    fps = [smiles_to_fingerprint(s) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return 0.0

    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = tanimoto_similarity(fps[i], fps[j])
            distances.append(1.0 - sim)  # distance = 1 - similarity
    return float(np.mean(distances))


def compute_batch_stats(results, total_attempts):
    """
    Compute summary statistics for a generation batch.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys: SMILES, MW, LogP
    total_attempts : int
        Total number of generation attempts (valid + invalid)

    Returns
    -------
    dict with keys: validity_rate, diversity, avg_mw, avg_logp, mw_list, logp_list
    """
    smiles_list = [r["SMILES"] for r in results]
    mw_list = [r["MW"] for r in results]
    logp_list = [r["LogP"] for r in results]

    return {
        "validity_rate": round(len(results) / max(total_attempts, 1) * 100, 1),
        "diversity": round(compute_diversity_score(smiles_list), 3),
        "avg_mw": round(float(np.mean(mw_list)), 2) if mw_list else 0,
        "avg_logp": round(float(np.mean(logp_list)), 2) if logp_list else 0,
        "mw_list": mw_list,
        "logp_list": logp_list,
        "num_valid": len(results),
        "num_attempts": total_attempts,
    }


# ---------------------------------------------------------------------------
# Chemical Space t-SNE  (Feature 4)
# ---------------------------------------------------------------------------

def compute_tsne_embedding(generated_smiles, generated_props, reference_smiles=None,
                           reference_names=None, perplexity=5):
    """
    Compute a 2D t-SNE embedding for generated molecules and optional reference drugs.

    Returns
    -------
    list[dict] with keys: x, y, smiles, label, group, mw, logp
    """
    from sklearn.manifold import TSNE

    all_smiles = list(generated_smiles)
    labels = [f"Generated #{i+1}" for i in range(len(generated_smiles))]
    groups = ["Generated"] * len(generated_smiles)
    mw_vals = [p.get("MW", 0) for p in generated_props]
    logp_vals = [p.get("LogP", 0) for p in generated_props]

    if reference_smiles:
        all_smiles.extend(reference_smiles)
        if reference_names:
            labels.extend(reference_names)
        else:
            labels.extend([f"Drug #{i+1}" for i in range(len(reference_smiles))])
        groups.extend(["Known Drug"] * len(reference_smiles))
        # Compute MW/LogP for reference drugs
        for smi in reference_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                from rdkit.Chem import Descriptors
                mw_vals.append(round(Descriptors.MolWt(mol), 2))
                logp_vals.append(round(Descriptors.MolLogP(mol), 2))
            else:
                mw_vals.append(0)
                logp_vals.append(0)

    # Compute fingerprints
    fps = []
    valid_indices = []
    for i, smi in enumerate(all_smiles):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)

    if len(fps) < 3:
        return []

    X = fingerprints_to_numpy(fps)

    # Adjust perplexity to be less than the number of samples
    effective_perp = min(perplexity, len(fps) - 1)
    effective_perp = max(effective_perp, 2)

    tsne = TSNE(n_components=2, perplexity=effective_perp, random_state=42,
                max_iter=1000, init="random")
    coords = tsne.fit_transform(X)

    points = []
    for idx_in_fps, orig_idx in enumerate(valid_indices):
        points.append({
            "x": float(coords[idx_in_fps, 0]),
            "y": float(coords[idx_in_fps, 1]),
            "smiles": all_smiles[orig_idx],
            "label": labels[orig_idx],
            "group": groups[orig_idx],
            "mw": mw_vals[orig_idx],
            "logp": logp_vals[orig_idx],
        })
    return points


# ---------------------------------------------------------------------------
# Drug Comparison  (Feature 7)
# ---------------------------------------------------------------------------

def load_known_drugs(csv_path=None):
    """Load the known drugs CSV and return list of dicts."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "known_drugs.csv")
    drugs = []
    if not os.path.exists(csv_path):
        return drugs
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugs.append(row)
    return drugs


def find_closest_drugs(generated_smiles_list, top_k=3, csv_path=None):
    """
    For each generated SMILES, find the top-k most similar known drugs.

    Returns
    -------
    list[dict] with keys: generated, matches (list of {name, smiles, category, similarity})
    """
    drugs = load_known_drugs(csv_path)
    if not drugs:
        return []

    # Pre-compute drug fingerprints
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

        similarities = []
        for drug, drug_fp in drug_fps:
            sim = tanimoto_similarity(gen_fp, drug_fp)
            similarities.append({
                "name": drug["Name"],
                "smiles": drug["SMILES"],
                "category": drug.get("Category", "Unknown"),
                "similarity": round(sim * 100, 1),
            })

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        results.append({
            "generated": gen_smi,
            "matches": similarities[:top_k],
        })

    return results
