# Small-Molecule Design with a Transformer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![RDKit](https://img.shields.io/badge/RDKit-Chemoinformatics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-ff4b4b)
![Split](https://img.shields.io/badge/Split-Scaffold%20Aware-0f766e)
![Sampling](https://img.shields.io/badge/Sampling-Top--k%20%2B%20Top--p-7c3aed)
![Data](https://img.shields.io/badge/Data-249k%20SMILES-f59e0b)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://small-molecule-design-using-transformer-model-fzvu6ve66dz8h8zf.streamlit.app)

Generate novel small molecules as SMILES strings with a Transformer-based PyTorch pipeline designed for molecular sequence modeling, RDKit validation, and interactive exploration. The project includes training, CLI generation, property estimation, and a Streamlit UI for fast experimentation.

> The model is specialized for autoregressive SMILES continuation: give it a fragment such as `C`, `N`, or `c1ccccc1`, sample the next tokens, then inspect chemical validity, QED, MW, LogP, and related metrics.

## What This Model Is Specialized For

- SMILES language modeling with causal masking so each token only attends to previous chemical context.
- Fragment-conditioned generation from short seeds, scaffold-like prompts, or partial SMILES prefixes.
- Safer sampling via `top_k`, `top_p`, repetition penalties, minimum new-token constraints, and invalid-molecule filtering.
- Chemistry-first postprocessing with RDKit validity checks, molecular descriptors, and dashboard visualization.
- Rapid ideation rather than full 3D or protein-conditioned design; 3D conformers are generated after sequence sampling.

## Technical Snapshot

| Item | Value |
| --- | --- |
| Model family | Causal Transformer encoder for autoregressive SMILES generation |
| Core implementation | `nn.TransformerEncoder` with causal mask, GELU activation, `norm_first=True`, and final `LayerNorm` |
| Tokenizer | Regex-based SMILES tokenizer with `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` |
| Dataset in repo | `249,455` SMILES strings in `data/smiles.txt` |
| Current training defaults | `d_model=256`, `nhead=8`, `num_layers=4`, `dim_feedforward=1024`, `max_len=60`, `dropout=0.2` |
| Attention head width | `32` dims per head |
| Approx. parameters (current architecture at `vocab_size=65`) | `3,192,897` |
| Inference controls | `temperature`, `top_k`, `top_p`, `repetition_penalty`, `min_new_tokens`, `max_repeat_run` |
| Validation stack | RDKit descriptors + Lipinski filtering logic in `src/property.py` |
| Offline evaluation | Validity, uniqueness, novelty, held-out overlap, diversity, and descriptor summaries via `src/evaluate.py` |

## Parameter Breakdown

Current architecture parameter count, computed at `vocab_size=65` to match the packaged checkpoint metadata:

| Component | Parameters |
| --- | ---: |
| Token embedding | `16,640` |
| Transformer encoder stack | `3,159,552` |
| Output projection head | `16,705` |
| Total | `3,192,897` |

## Generation Pipeline

```mermaid
flowchart LR
    A["Seed fragment"] --> B["SMILES tokenizer"]
    B --> C["Embedding + positional encoding"]
    C --> D["4-layer causal Transformer encoder"]
    D --> E["Vocabulary projection"]
    E --> F["Sampling: temperature + top-k/top-p + repetition controls"]
    F --> G["Generated SMILES"]
    G --> H["RDKit validation + properties + 3D conformer"]
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Launch the dashboard

```bash
python -m streamlit run streamlit_app.py
```

### Generate molecules

```bash
python src/generate.py --seed_token "N" --num_molecules 5 --temperature 0.8 --top_k 50 --top_p 0.95
```

### Generate with stronger anti-repetition controls

```bash
python src/generate.py --seed_token "C" --num_molecules 10 --temperature 0.85 --top_k 40 --top_p 0.92 --repetition_penalty 1.2 --min_new_tokens 8 --max_repeat_run 4
```

### Score a molecule

```bash
python src/property.py --smiles "c1ccccc1"
```

## Training

1. Place one SMILES string per line in `data/smiles.txt`.
2. Or preprocess a ZINC-style archive with:

```bash
python src/load_zinc.py --zip_path /path/to/zinc_archive.zip
```

3. Train the model:

```bash
python src/train.py --epochs 10 --batch_size 64 --max_len 60 --split_method scaffold --val_split 0.1 --test_split 0.1 --dedup
```

4. Reuse an existing split for repeatable experiments:

```bash
python src/train.py --reuse_split --split_dir data/splits/scaffold_seed42_val10_test10 --epochs 10
```

5. Optional runtime tuning for CPU-heavy environments:

```bash
python src/train.py --epochs 5 --batch_size 64 --num_workers 0 --no_pin_memory
```

Training now saves deterministic split artifacts under `data/splits/<tag>/` by default. Each split directory contains:

- `train.txt`
- `val.txt`
- `test.txt`
- `metadata.json`

## Evaluation

Run the offline evaluator for report-ready metrics:

```bash
python src/evaluate.py --num_molecules 100 --seed_token C
```

Evaluate against a saved split directory to measure novelty against the training split and overlap with the held-out test split:

```bash
python src/evaluate.py --num_molecules 100 --seed_token C --split_dir data/splits/scaffold_seed42_val10_test10
```

## Quick Checks

Run the lightweight built-in smoke tests:

```bash
python -m unittest discover -s tests
```

<details>
<summary>Training defaults from <code>src/train.py</code></summary>

| Argument | Default |
| --- | --- |
| `epochs` | `20` |
| `lr` | `3e-4` |
| `batch_size` | `64` |
| `max_len` | `60` |
| `d_model` | `256` |
| `nhead` | `8` |
| `num_layers` | `4` |
| `dropout` | `0.2` |
| `val_split` | `0.1` |
| `test_split` | `0.1` |
| `split_method` | `scaffold` |
| `dedup` | `True` |
| `num_workers` | `2` |
| `pin_memory` | `True` on CUDA |
| `patience` | `8` |
| `min_delta` | `1e-4` |
| `weight_decay` | `1e-2` |
| `label_smoothing` | `0.05` |
| `checkpoint_name` | `best_model.pt` |
| `output_dir` | `checkpoints/` |
| `split_dir` | auto-generated under `data/splits/` |
| `reuse_split` | `False` |

Training also uses `AdamW`, `OneCycleLR`, gradient clipping at `1.0`, and AMP automatically on CUDA.
</details>

<details>
<summary>Checkpoint and experiment note for technical users</summary>

The packaged `checkpoints/best_model.pt` is intended to load with the current encoder-style `src/model.py`. New training runs write under `checkpoints/` by default, but `src/train.py` now avoids overwriting an existing checkpoint by creating a numbered variant such as `best_model_1.pt`. Paired loss curves are saved as `<checkpoint_stem>_loss_history.csv`, and you can route runs to a separate folder with `--output_dir`.
</details>

<details>
<summary>Notes for technical readers</summary>

- The model operates on linearized SMILES tokens, not molecular graphs.
- The tokenizer recognizes bracketed atoms, aromatic lower-case atoms, ring indices, bond symbols, and halogens such as `Cl` and `Br`.
- `src/train.py` supports random or scaffold-aware train/validation/test splitting, plus saved split artifacts for reproducible experiments.
- `src/evaluate.py` is the preferred path for report-style metrics such as novelty and held-out overlap; the Streamlit app is primarily exploratory.
- `streamlit_app.py` can browse any saved checkpoint under `checkpoints/`, rather than assuming a single fixed artifact.
- 3D structures shown in the app are computed with RDKit after generation, not predicted directly by the network.
</details>
