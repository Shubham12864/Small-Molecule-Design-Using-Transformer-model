# Design Document — Molecule Transformer

## 1. Problem Description

**Goal:** Build a deep learning model that learns the syntax and patterns of molecular SMILES strings and can generate new, chemically plausible molecules.

**Why SMILES?** SMILES (Simplified Molecular Input Line Entry System) is a text-based representation of molecular structures. By treating molecules as sequences of characters, we can apply natural language processing (NLP) techniques — specifically, Transformer models — to learn molecular patterns.

**Key Tasks:**

1. Train a Transformer on SMILES strings via next-token prediction
2. Generate novel SMILES sequences autoregressively
3. Validate and compute properties of generated molecules using RDKit

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
│                                                                     │
│   smiles.txt                                                        │
│       │                                                             │
│       ▼                                                             │
│   ┌──────────┐     ┌──────────────┐     ┌────────────────────────┐ │
│   │Tokenizer │────▸│   Dataset    │────▸│   DataLoader           │ │
│   │(char→ID) │     │(input,target)│     │(batched, shuffled)     │ │
│   └──────────┘     └──────────────┘     └──────────┬─────────────┘ │
│                                                     │               │
│                                                     ▼               │
│                              ┌───────────────────────────────────┐  │
│                              │     MOLECULE TRANSFORMER          │  │
│                              │                                   │  │
│                              │  Token Embedding (vocab → 64-d)   │  │
│                              │         │                         │  │
│                              │         ▼                         │  │
│                              │  Positional Encoding (sinusoidal) │  │
│                              │         │                         │  │
│                              │         ▼                         │  │
│                              │  Transformer Encoder Layer × 2    │  │
│                              │  ┌─────────────────────────────┐  │  │
│                              │  │ Multi-Head Attention (4 hds)│  │  │
│                              │  │ Add & Norm                  │  │  │
│                              │  │ Feed-Forward (64 → 128 → 64)│  │  │
│                              │  │ Add & Norm                  │  │  │
│                              │  └─────────────────────────────┘  │  │
│                              │         │                         │  │
│                              │         ▼                         │  │
│                              │  Linear Head (64 → vocab_size)    │  │
│                              │         │                         │  │
│                              │         ▼                         │  │
│                              │     Logits (per token)            │  │
│                              └───────────────────────────────────┘  │
│                                                     │               │
│                                        CrossEntropyLoss + Adam      │
│                                                     │               │
│                                              Save model.pt          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       GENERATION PIPELINE                           │
│                                                                     │
│   Seed: "C"                                                         │
│       │                                                             │
│       ▼                                                             │
│   ┌──────────────┐                                                  │
│   │ SOS + "C"    │──┐                                               │
│   └──────────────┘  │                                               │
│                     ▼                                               │
│             ┌──────────────┐     ┌──────┐                           │
│             │  Transformer │────▸│argmax│──▸ next token             │
│             └──────────────┘     └──────┘       │                   │
│                     ▲                           │                   │
│                     └───── append ◂─────────────┘                   │
│                        (repeat until EOS)                           │
│                                                                     │
│   Output: "CCO" ──▸ RDKit ──▸ Valid? ✓  MW: 46.07  LogP: -0.00     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

```
                      ┌─────────────┐
                      │ smiles.txt  │
                      │ (raw text)  │
                      └──────┬──────┘
                             │
                    ┌────────▼────────┐
                    │   Tokenizer     │
                    │ char → int IDs  │
                    │ + SOS, EOS, PAD │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Dataset      │
                    │ input = [:-1]   │
                    │ target = [1:]   │
                    │ pad to max_len  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   DataLoader    │
                    │ batch, shuffle  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Transformer   │
                    │  → logits       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Training       Generation     Properties
         (loss →        (argmax →      (RDKit →
          backprop)      SMILES)       MW, LogP)
```

---

## 4. Model Explanation

### Why a Transformer?

Transformers excel at sequence modelling because of **self-attention**, which lets the model look at the entire input sequence when predicting each token. This is important for SMILES because:

- Ring closures (e.g., `c1ccccc1`) require long-range dependencies
- Branching (parentheses) creates nested structures
- Atom types and bonds interact across the sequence

### Model Components

| Component               | Purpose                                           | Size        |
|------------------------|----------------------------------------------------|-------------|
| Token Embedding         | Converts token IDs to dense vectors               | vocab × 64  |
| Positional Encoding     | Adds position info (sinusoidal, not learned)      | max_len × 64|
| Transformer Encoder     | Self-attention + feed-forward (2 layers)          | ~40K params |
| Linear Head             | Projects hidden states to vocabulary logits        | 64 × vocab  |

### Causal Masking

We use an upper-triangular mask so each token can only attend to **previous** tokens (including itself). This enforces the autoregressive property needed for generation.

### Training Objective

- **Loss:** CrossEntropyLoss (ignoring PAD tokens)
- **Optimizer:** Adam (lr = 0.001)
- **Task:** Given tokens `[SOS, C, C, O]`, predict `[C, C, O, EOS]`

---

## 5. Limitations

1. **Small model capacity:** With ~50K parameters and 2 layers, the model has limited ability to capture complex molecular patterns.

2. **Greedy decoding:** Using argmax means the model may generate repetitive sequences. Sampling or beam search would produce more diverse outputs.

3. **No chemical constraints:** The model learns syntax statistically but doesn't enforce chemical rules (valence, aromaticity). Invalid molecules can still be generated.

4. **Small training data:** The included dataset of 100 molecules is for demonstration. Real-world performance requires datasets of 10K–1M+ molecules.

5. **Character-level tokenization:** Treats multi-character atoms (like `Br`, `Cl`) as separate characters, which can lead to fragmentation. A SMILES-aware tokenizer would be more accurate.

6. **No property-guided generation:** Generation is unconditional — we cannot steer the model to generate molecules with specific properties.

7. **Encoder-only architecture:** While a decoder-only or encoder-decoder architecture is more traditional for sequence generation, we use a causal-masked encoder which works but is less conventional.

---

## 6. Future Improvements

1. **Larger dataset:** Train on the full ZINC database (250K+ molecules) for better generalization.

2. **Temperature sampling / top-k / nucleus sampling:** Replace greedy decoding for more diverse and creative generation.

3. **SMILES-aware tokenizer:** Use regex-based tokenization to properly handle multi-character tokens like `Br`, `Cl`, `[NH]`, `@@`.

4. **Conditional generation:** Add property conditioning (e.g., target LogP or MW) to guide molecule generation toward desired properties.

5. **Decoder-only Transformer (GPT-style):** Use a proper autoregressive decoder for more standard language modelling.

6. **Variational autoencoder (VAE):** Combine with a VAE for smooth latent space interpolation between molecules.

7. **Reinforcement learning:** Fine-tune with RL to optimize specific molecular properties (drug-likeness, binding affinity).

8. **3D structure prediction:** Extend to predict 3D molecular conformations from SMILES.

9. **Attention visualization:** Add attention heatmaps to explain which parts of SMILES the model focuses on.
