import streamlit as st
import torch
import sys
import os
import warnings
import py3Dmol
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

from model import MoleculeTransformer
from tokenizer import SmilesTokenizer
from generate import generate_smiles, check_validity
from property import compute_properties
from utils import load_model, get_checkpoint_path

# --- Helper: Generate 3D Model String ---
def generate_3d_mol_block(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            # Fallback: try with random coordinates
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None

# --- Display Setup ---
st.set_page_config(page_title="AI Molecular Engine", page_icon="🧬", layout="wide")

st.title("🧬 AI Molecular Generator")
st.markdown("Transform SMILES seeds into 3D Molecular Structures using Deep Learning.")
st.divider()

# Caching the model
@st.cache_resource
def get_model():
    checkpoint_path = get_checkpoint_path()
    if not os.path.exists(checkpoint_path): return None, None
    try:
        model, tokenizer_data = load_model(MoleculeTransformer, checkpoint_path)
    except Exception:
        return None, None

    tokenizer = SmilesTokenizer()
    if 'vocab' in tokenizer_data:
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.chars = tokenizer_data['chars']
    else:
        tokenizer.stoi = tokenizer_data.get('stoi', {})
        tokenizer.itos = tokenizer_data.get('itos', {})
        tokenizer.chars = list(tokenizer.stoi.keys())
        tokenizer.vocab_size = tokenizer_data.get('vocab_size', len(tokenizer.stoi))
    return model, tokenizer

model, tokenizer = get_model()

if model is None:
    st.error("Error: Could not load the model checkpoint.")
    st.stop()

# --- Layout ---
col_settings, col_results = st.columns([1, 2.5], gap="large")

with col_settings:
    st.header("Generation Settings")
    st.markdown("Customize your molecular generation targets.")
    seed = st.text_input("Seed Sequence (e.g., C, N, O, c1ccccc1)", value="C")
    num_molecules = st.number_input("Number of Molecules to Generate", min_value=1, max_value=20, value=6)
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
    generate_btn = st.button("Generate Molecules", type="primary", use_container_width=True)

with col_results:
    if generate_btn:
        if not seed:
            st.warning("Please enter a valid seed sequence.")
        else:
            st.header("Discovery Results")
            with st.spinner("AI is generating valid molecular structures with 3D topology..."):
                results = []
                attempts = 0
                max_attempts = num_molecules * 10  # Try up to 10x to find valid molecules

                while len(results) < num_molecules and attempts < max_attempts:
                    attempts += 1
                    smiles = generate_smiles(model, tokenizer, seed=seed, max_len=60, temperature=temperature)
                    is_valid = check_validity(smiles)

                    if is_valid:
                        props = compute_properties(smiles)
                        results.append({
                            "SMILES": smiles,
                            "MolBlock": generate_3d_mol_block(smiles),
                            "MW": round(props['molecular_weight'], 2),
                            "LogP": round(props['logp'], 2)
                        })

                if len(results) == 0:
                    st.warning("Could not generate any valid molecules. Try a different seed or increase the temperature.")
                else:
                    if len(results) < num_molecules:
                        st.info(f"Found {len(results)} valid molecules out of {attempts} attempts.")

                    # Render in a 3-column grid
                    grid_cols = st.columns(3)
                    for idx, res in enumerate(results):
                        with grid_cols[idx % 3]:
                            with st.container(border=True):
                                st.markdown("**✅ Valid Molecule**")
                                st.code(res["SMILES"])
                                st.caption(f"**Weight:** {res['MW']} g/mol  |  **LogP:** {res['LogP']}")

                                # 3D Render
                                if res['MolBlock']:
                                    view = py3Dmol.view(width=280, height=220)
                                    view.addModel(res['MolBlock'], 'mol')
                                    view.setStyle({
                                        'stick': {'radius': 0.15},
                                        'sphere': {'scale': 0.25}
                                    })
                                    view.zoomTo()
                                    view.setBackgroundColor('#0e1117')
                                    components.html(view._make_html(), height=230, width=290)
                                else:
                                    st.info("Could not calculate 3D coordinates for this molecule.")
    else:
        st.info("👈 Enter your settings and click **Generate Molecules** to begin.")
