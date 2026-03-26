# Molecular Transformer for Small Molecule Generation

This project trains a causal Transformer on SMILES strings and uses the trained model to generate new small molecules, score them with RDKit, and explore them in a Streamlit app.

The current repository includes:
- a corrected GPT-style causal model in `src/model.py`
- a training pipeline with scaffold split, deduplication, early stopping, and checkpoint saving
- a generation script with safer sampling controls
- an evaluation script that reports validity, uniqueness, novelty, diversity, and property summaries
- a Streamlit frontend for interactive molecule generation and analysis
- a trained checkpoint in `checkpoints/best_model.pt`

## Project Goal

The goal is to generate drug-like SMILES strings that are:
- chemically valid
- diverse
- reasonably drug-like by simple property filters
- easy to inspect through a lightweight app and CLI workflow

## Repository Layout

- `src/model.py`: causal Transformer model
- `src/train.py`: training loop
- `src/generate.py`: autoregressive generation
- `src/evaluate.py`: batch evaluation and saved metrics
- `src/property.py`: RDKit property calculation
- `src/analytics.py`: diversity and chemical-space helpers
- `streamlit_app.py`: interactive frontend
- `data/smiles.txt`: training data file
- `checkpoints/`: trained checkpoints and loss history

## Install

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python -m streamlit run streamlit_app.py
```

## Generate Molecules

Use the included checkpoint:

```bash
python src/generate.py --seed_token C --num_molecules 10 --checkpoint checkpoints/best_model.pt
```

Useful seed examples:
- `C`
- `N`
- `O`
- `CC`
- `CO`
- `CN`
- `c1ccccc1`
- `NC(=O)`

## Evaluate a Checkpoint

This is the recommended way to judge a trained model.

```bash
python src/evaluate.py --num_molecules 100 --checkpoint checkpoints/best_model.pt
```

Outputs:
- `checkpoints/best_model_evaluation.csv`
- `checkpoints/best_model_evaluation.json`

Reported metrics include:
- validity
- uniqueness
- novelty against `data/smiles.txt`
- diversity
- average QED
- average MW
- average LogP
- average SA score
- Lipinski pass rate

## Train From Scratch

If you already have `data/smiles.txt`:

```bash
python src/train.py --epochs 20 --batch_size 64 --max_len 60 --dropout 0.2 --lr 3e-4 --split_method scaffold --dedup --num_workers 2 --save_last
```

Training outputs:
- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `checkpoints/loss_history.csv`

## Prepare Data From a ZINC Zip

```bash
python src/load_zinc.py --zip_path path/to/zinc.zip --max_molecules 250000
```

This writes the extracted SMILES file to `data/smiles.txt` unless you pass `--output`.

## Current Model Setup

The included checkpoint was trained with:
- `d_model=256`
- `nhead=8`
- `num_layers=4`
- `dim_feedforward=1024`
- `max_len=60`
- `dropout=0.2`

The training script uses:
- canonicalization before splitting
- optional deduplication
- scaffold split by default
- AdamW
- label smoothing
- gradient clipping
- early stopping

## Why Evaluation Should Be Part of the Workflow

Loss alone is not enough for this project.

A model can show a lower validation loss and still generate weak molecules. Running `src/evaluate.py` after training gives you a checkpoint report you can compare across runs. That helps you:
- choose the best checkpoint by generation quality, not just loss
- measure validity and diversity honestly
- see whether molecules are novel relative to the training data
- save a JSON or CSV record for README, app, or future experiments

Recommended workflow:
1. train a checkpoint
2. run `src/evaluate.py`
3. compare metrics between runs
4. use the best checkpoint in the app

## Streamlit Notes

The app is meant for interactive exploration, not for benchmarking. For real model comparison, use `src/evaluate.py`.

The training-loss chart is available in a collapsed diagnostics section so the main UI stays focused on generation results.

## Known Limitations

- evaluation is still based on simple property and diversity metrics, not target-specific activity
- the app is exploratory and not a production cheminformatics tool
- automated tests are still limited
- the included chemical-space view is qualitative, not a rigorous benchmark

## Suggested Next Improvements

- add unit tests for tokenizer, generation, and evaluation
- surface evaluation JSON directly inside the app
- clean remaining UI text and packaging details
- add checkpoint comparison in the frontend
