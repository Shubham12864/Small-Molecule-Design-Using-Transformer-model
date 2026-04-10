import csv
import os

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, DataStructs, Descriptors

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    AllChem = None
    DataStructs = None
    Descriptors = None
    RDKIT_AVAILABLE = False

try:
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    TSNE = None
    SKLEARN_AVAILABLE = False


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def fingerprints_to_numpy(fingerprints):
    if not fingerprints:
        return np.zeros((0, 0), dtype=np.float32)

    array = np.zeros((len(fingerprints), fingerprints[0].GetNumBits()), dtype=np.float32)
    for index, fingerprint in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fingerprint, array[index])
    return array


def tanimoto_similarity(fp1, fp2):
    if not RDKIT_AVAILABLE:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_diversity_score(smiles_list):
    if not RDKIT_AVAILABLE:
        return 0.0

    fingerprints = [smiles_to_fingerprint(smiles) for smiles in smiles_list]
    fingerprints = [fingerprint for fingerprint in fingerprints if fingerprint is not None]
    if len(fingerprints) < 2:
        return 0.0

    distances = []
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            distances.append(1.0 - tanimoto_similarity(fingerprints[left], fingerprints[right]))
    return float(np.mean(distances))


def compute_batch_stats(results, total_attempts):
    smiles_list = [row["smiles"] for row in results]
    mw_list = [row["mw"] for row in results if row.get("mw") is not None]
    logp_list = [row["logp"] for row in results if row.get("logp") is not None]
    qed_list = [row["qed"] for row in results if row.get("qed") is not None]
    lip_pass = sum(1 for row in results if row.get("lipinski"))

    canonical_smiles = [row.get("canonical_smiles") or row["smiles"] for row in results]
    unique_valid_smiles = list(dict.fromkeys(canonical_smiles))

    stats = {
        "validity_rate": round(len(results) / max(total_attempts, 1) * 100, 1),
        "uniqueness_rate": round(100 * len(unique_valid_smiles) / max(len(results), 1), 1),
        "num_unique": len(unique_valid_smiles),
        "duplicate_valid_count": len(results) - len(unique_valid_smiles),
        "diversity": round(compute_diversity_score(unique_valid_smiles), 3),
        "avg_mw": round(float(np.mean(mw_list)), 2) if mw_list else 0,
        "avg_logp": round(float(np.mean(logp_list)), 2) if logp_list else 0,
        "avg_qed": round(float(np.mean(qed_list)), 3) if qed_list else 0,
        "lipinski_pass": lip_pass,
        "lipinski_rate": round(100 * lip_pass / max(len(results), 1), 1),
        "mw_list": mw_list,
        "logp_list": logp_list,
        "qed_list": qed_list,
        "num_valid": len(results),
        "num_attempts": total_attempts,
    }
    return stats


def compute_tsne_embedding(
    generated_smiles,
    generated_props,
    reference_smiles=None,
    reference_names=None,
    perplexity=5,
):
    if not SKLEARN_AVAILABLE or not RDKIT_AVAILABLE:
        return []

    all_smiles = list(generated_smiles)
    labels = [f"Generated #{index + 1}" for index in range(len(generated_smiles))]
    groups = ["Generated"] * len(generated_smiles)
    mw_vals = [props.get("mw", 0) for props in generated_props]
    logp_vals = [props.get("logp", 0) for props in generated_props]

    if reference_smiles:
        all_smiles.extend(reference_smiles)
        labels.extend(reference_names or [f"Drug #{index + 1}" for index in range(len(reference_smiles))])
        groups.extend(["Known Drug"] * len(reference_smiles))
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw_vals.append(round(Descriptors.MolWt(mol), 2))
                logp_vals.append(round(Descriptors.MolLogP(mol), 2))
            else:
                mw_vals.append(0)
                logp_vals.append(0)

    fingerprints = []
    valid_indices = []
    for index, smiles in enumerate(all_smiles):
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
            valid_indices.append(index)

    if len(fingerprints) < 3:
        return []

    feature_matrix = fingerprints_to_numpy(fingerprints)
    effective_perplexity = max(2, min(perplexity, len(fingerprints) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        max_iter=1000,
        init="random",
    )
    coordinates = tsne.fit_transform(feature_matrix)

    points = []
    for fp_index, source_index in enumerate(valid_indices):
        points.append(
            {
                "x": float(coordinates[fp_index, 0]),
                "y": float(coordinates[fp_index, 1]),
                "smiles": all_smiles[source_index],
                "label": labels[source_index],
                "group": groups[source_index],
                "mw": mw_vals[source_index],
                "logp": logp_vals[source_index],
            }
        )
    return points


def load_known_drugs(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "known_drugs.csv",
        )
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_closest_drugs(generated_smiles_list, top_k=3, csv_path=None):
    if not RDKIT_AVAILABLE:
        return []

    drugs = load_known_drugs(csv_path)
    if not drugs:
        return []

    drug_fingerprints = []
    for drug in drugs:
        fingerprint = smiles_to_fingerprint(drug["SMILES"])
        if fingerprint is not None:
            drug_fingerprints.append((drug, fingerprint))

    results = []
    for generated_smiles in generated_smiles_list:
        generated_fp = smiles_to_fingerprint(generated_smiles)
        if generated_fp is None:
            results.append({"generated": generated_smiles, "matches": []})
            continue

        similarities = sorted(
            [
                {
                    "name": drug["Name"],
                    "smiles": drug["SMILES"],
                    "category": drug.get("Category", "Unknown"),
                    "similarity": round(tanimoto_similarity(generated_fp, drug_fp) * 100, 1),
                }
                for drug, drug_fp in drug_fingerprints
            ],
            key=lambda row: row["similarity"],
            reverse=True,
        )

        results.append({"generated": generated_smiles, "matches": similarities[:top_k]})
    return results
