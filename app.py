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
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None

# --- Display Setup ---
st.set_page_config(page_title="AI Molecular Engine", page_icon="🧬", layout="wide")

# Clean, professional CSS
st.markdown("""
    <style>
    .element-card {
        border-radius: 8px;
        padding: 20px;
        background-color: #1e1e1e;
        border: 1px solid #333;
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .valid-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 12px;
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .invalid-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 12px;
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Clean 2-column layout (Settings on left, Results on right)
col_settings, col_results = st.columns([1, 2.5], gap="large")

with col_settings:
    st.header("Generation Settings")
    st.markdown("Customize your molecular generation targets.")
    
    seed = st.text_input("Seed Sequence (e.g., C, N, O, c1ccccc1)", value="C")
    num_molecules = st.number_input("Number of Molecules to Generate", min_value=1, max_value=20, value=6)
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Generate Molecules", type="primary", use_container_width=True)

with col_results:
    if generate_btn:
        if not seed:
            st.warning("Please enter a valid seed sequence.")
        else:
            st.header("Discovery Results")
            with st.spinner("AI Generating structures and calculating 3D topology..."):
                results = []
                for _ in range(num_molecules):
                    smiles = generate_smiles(model, tokenizer, seed=seed, max_len=60, temperature=temperature)
                    is_valid = check_validity(smiles)
                    
                    if is_valid:
                        props = compute_properties(smiles)
                        results.append({
                            "SMILES": smiles, 
                            "Valid": True, 
                            "MolBlock": generate_3d_mol_block(smiles),
                            "MW": round(props['molecular_weight'], 2),
                            "LogP": round(props['logp'], 2)
                        })
                    else:
                        results.append({"SMILES": smiles, "Valid": False, "MolBlock": None})
                
                # Render in a clean 3-column grid
                grid_cols = st.columns(3)
                for idx, res in enumerate(results):
                    with grid_cols[idx % 3]:
                        st.markdown("<div class='element-card'>", unsafe_allow_html=True)
                        
                        if res["Valid"]:
                            st.markdown("<span class='valid-badge'>✓ Valid Molecule</span>", unsafe_allow_html=True)
                            st.code(res["SMILES"])
                            st.caption(f"**Weight:** {res['MW']} g/mol | **LogP:** {res['LogP']}")
                            
                            # Interactive 3D render inside loop
                            if res['MolBlock']:
                                view = py3Dmol.view(width=280, height=220)
                                view.addModel(res['MolBlock'], 'mol')
                                view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                                view.zoomTo()
                                view.setBackgroundColor('#1e1e1e')
                                components.html(view._make_html(), height=220, width=280)
                            else:
                                st.info("Could not calculate 3D coordinates.")
                        else:
                            st.markdown("<span class='invalid-badge'>✗ Invalid Grammar</span>", unsafe_allow_html=True)
                            st.code(res["SMILES"])
                            st.markdown("<div style='height: 250px; display:flex; align-items:center; justify-content:center; color:#666;'>No 3D Model Available</div>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("👈 Enter your settings and click 'Generate Molecules' to begin.")
