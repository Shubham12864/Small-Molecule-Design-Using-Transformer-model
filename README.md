# 🧬 Molecular Transformer: AI Drug Discovery Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![RDKit](https://img.shields.io/badge/RDKit-Chemistry%20Engine-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)

A state-of-the-art Generative Deep Learning model designed to hallucinate entirely new drug-like molecules from scratch using **SMILES** string representations and **Transformer Architecture**. The system was trained on **250,000 real-world compounds** from the ZINC database and features a fully interactive **3D Molecular Web Dashboard**.

## ✨ Key Features
- **Transformer Encoder Architecture:** Custom PyTorch implementation leveraging Multi-Head Self-Attention to learn the deep "grammar" and sequential logic of chemical structures.
- **Massive Knowledge Base:** Pre-trained on a quarter-million molecules (`ZINC-250k`), ensuring the generated molecules have highly realistic and chemically viable substructures.
- **Physics & Chemistry Validation:** Automatically verified using the `RDKit` cheminformatics engine. Invalid physical structures are rejected, while valid ones have their Molecular Weight and LogP (Lipophilicity) calculated live.
- **Interactive 3D Web Dashboard:** A sleek, dark-mode Streamlit UI that instantly converts AI-generated 1D text strings into interactive, rotatable **3D ball-and-stick models**.
- **Combinatorial Synthesis ("Seed" Tokens):** Users can force the AI to build molecules starting from specific chemical fragments (e.g., Benzene rings, Carbon chains, Oxygen links) and control the "Neural Creativity" (Temperature) during sampling.

---

## 📸 The 3D Dashboard (`app.py`)

The project includes a stunning Streamlit Web Application that makes interacting with the neural network frictionless.

1. **Install Dependencies:**
   ```bash
   pip install torch rdkit numpy matplotlib jupyter streamlit pandas py3Dmol stmol
   ```
2. **Launch the Engine:**
   ```bash
   python -m streamlit run app.py
   ```
3. **Usage:** Simply select your base elements, choose how many molecules to generate, adjust the creativity temperature, and click **Generate**. The model will output interactive 3D structures in real-time.

---

## 🧠 System Architecture

The core of the system is entirely custom-built in PyTorch:

- **src/tokenizer.py**: A character-level `SmilesTokenizer` that breaks complex chemical syntax (like `CC(=O)Oc1ccccc1C(=O)O`) into discrete machine-readable tokens, mapping them to a dynamic vocabulary size.
- **src/model.py**: The `MoleculeTransformer` utilizes `nn.Embedding`, custom `PositionalEncoding` (using sine/cosine waves for atom positioning), and a stack of `nn.TransformerEncoderLayer` blocks (64-dim, 4 heads).
- **src/property.py & src/generate.py**: Handles the autoregressive generation loop (predicting the next atom token by token) and pipes the final string through `RDKit` to generate spatial coordinates and physical properties.

---

## 🧪 Terminal Usage

If you prefer to interact with the model via terminal, you can use the core scripts:

**Generate a batch of 5 random molecules starting with Nitrogen (N):**
```bash
python src/generate.py --seed_token "N" --num_molecules 5 --temperature 0.8
```

**Analyze the properties of a specific SMILES string:**
```bash
python src/property.py --smiles "c1ccccc1"
```

---

## ⚙️ Train it Yourself

Want to train the AI from scratch on your own dataset?
1. Place your data in `data/smiles.txt` (one SMILES string per line).
2. Run the preprocessing script:
   ```bash
   python src/load_zinc.py
   ```
3. Start the Neural Engine training loop:
   ```bash
   python src/train.py --epochs 10 --batch_size 128 --max_len 60
   ```

*(Note: Training on 250k molecules is incredibly computationally expensive and is recommended to be done on a GPU like an NVIDIA T4 or A100).*


