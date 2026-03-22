import streamlit as st
import torch
import sys
import os
import warnings
import py3Dmol
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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
from analytics import (
    compute_batch_stats,
    compute_tsne_embedding,
    find_closest_drugs,
    load_known_drugs,
)

# ── Helper: Generate 3D Model String ──
def generate_3d_mol_block(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None

# ── Page Config ──
st.set_page_config(page_title="AI Molecular Engine", page_icon="🧬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        text-align: center;
        padding: 18px 10px;
        border-radius: 10px;
        border: 1px solid #333;
        background: linear-gradient(145deg, #1a1a2e, #16213e);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 13px;
        color: #888;
        margin-top: 4px;
    }
    .drug-match-card {
        padding: 14px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        background-color: rgba(102, 126, 234, 0.05);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 AI Molecular Generator")
st.markdown("Transform SMILES seeds into 3D Molecular Structures using Deep Learning.")
st.divider()

# ── Load Model ──
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

# ── Sidebar-style Settings Column ──
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
                max_attempts = num_molecules * 10

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
                            "LogP": round(props['logp'], 2),
                        })

            if len(results) == 0:
                st.warning("Could not generate any valid molecules. Try a different seed or increase the temperature.")
            else:
                if len(results) < num_molecules:
                    st.info(f"Found {len(results)} valid molecules out of {attempts} attempts.")

                # ── 3D Molecule Grid ──
                grid_cols = st.columns(3)
                for idx, res in enumerate(results):
                    with grid_cols[idx % 3]:
                        with st.container(border=True):
                            st.markdown("**✅ Valid Molecule**")
                            st.code(res["SMILES"])
                            st.caption(f"**Weight:** {res['MW']} g/mol  |  **LogP:** {res['LogP']}")
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
                                st.info("Could not calculate 3D coordinates.")

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # AI / ML Analysis Tabs
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                st.divider()
                st.header("🔬 AI / ML Analysis")

                tab_analytics, tab_space, tab_drugs = st.tabs([
                    "📊 Batch Analytics",
                    "🗺️ Chemical Space",
                    "💊 Drug Comparison",
                ])

                # ── Tab 1: Batch Analytics ──
                with tab_analytics:
                    stats = compute_batch_stats(results, attempts)

                    # Metric cards row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""<div class='metric-card'>
                            <div class='metric-value'>{stats['validity_rate']}%</div>
                            <div class='metric-label'>Validity Rate</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class='metric-card'>
                            <div class='metric-value'>{stats['diversity']}</div>
                            <div class='metric-label'>Diversity Score</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class='metric-card'>
                            <div class='metric-value'>{stats['avg_mw']}</div>
                            <div class='metric-label'>Avg Mol. Weight</div>
                        </div>""", unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""<div class='metric-card'>
                            <div class='metric-value'>{stats['avg_logp']}</div>
                            <div class='metric-label'>Avg LogP</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Property distribution charts
                    chart1, chart2 = st.columns(2)
                    with chart1:
                        fig_mw = px.histogram(
                            x=stats["mw_list"],
                            nbins=max(5, len(stats["mw_list"]) // 2),
                            labels={"x": "Molecular Weight (g/mol)", "y": "Count"},
                            title="Molecular Weight Distribution",
                            color_discrete_sequence=["#667eea"],
                        )
                        fig_mw.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            height=350,
                        )
                        st.plotly_chart(fig_mw, use_container_width=True)

                    with chart2:
                        fig_logp = px.histogram(
                            x=stats["logp_list"],
                            nbins=max(5, len(stats["logp_list"]) // 2),
                            labels={"x": "LogP (Lipophilicity)", "y": "Count"},
                            title="LogP Distribution",
                            color_discrete_sequence=["#764ba2"],
                        )
                        fig_logp.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            height=350,
                        )
                        st.plotly_chart(fig_logp, use_container_width=True)

                    # Summary text
                    st.caption(
                        f"Generated **{stats['num_valid']}** valid molecules "
                        f"from **{stats['num_attempts']}** total attempts. "
                        f"Diversity score measures average pairwise Tanimoto distance "
                        f"(higher = more structurally diverse)."
                    )

                # ── Tab 2: Chemical Space (t-SNE) ──
                with tab_space:
                    with st.spinner("Computing t-SNE embedding of chemical space..."):
                        known_drugs = load_known_drugs()
                        ref_smiles = [d["SMILES"] for d in known_drugs]
                        ref_names = [d["Name"] for d in known_drugs]

                        gen_props = [{"MW": r["MW"], "LogP": r["LogP"]} for r in results]
                        gen_smiles = [r["SMILES"] for r in results]

                        points = compute_tsne_embedding(
                            gen_smiles, gen_props,
                            reference_smiles=ref_smiles,
                            reference_names=ref_names,
                            perplexity=5,
                        )

                    if not points:
                        st.warning("Not enough valid molecules to compute t-SNE embedding.")
                    else:
                        df_pts = pd.DataFrame(points)

                        fig_tsne = px.scatter(
                            df_pts, x="x", y="y",
                            color="group",
                            size=[14 if g == "Generated" else 7 for g in df_pts["group"]],
                            color_discrete_map={
                                "Generated": "#ff6b6b",
                                "Known Drug": "#555577",
                            },
                            title="Chemical Space: Generated Molecules vs Known Drugs",
                            custom_data=["label", "smiles", "mw", "logp"],
                        )
                        fig_tsne.update_traces(
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "SMILES: %{customdata[1]}<br>"
                                "MW: %{customdata[2]}<br>"
                                "LogP: %{customdata[3]}"
                                "<extra></extra>"
                            ),
                            marker=dict(line=dict(width=0.5, color="white")),
                        )
                        fig_tsne.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=550,
                            xaxis_title="t-SNE Dimension 1",
                            yaxis_title="t-SNE Dimension 2",
                            legend_title="Molecule Type",
                            hoverlabel=dict(
                                bgcolor="#1e1e2e",
                                font_size=13,
                                font_family="monospace",
                                font_color="#e0e0e0",
                                bordercolor="#555577",
                            ),
                        )
                        st.plotly_chart(fig_tsne, use_container_width=True)

                        st.caption(
                            "Each point represents a molecule projected into 2D using "
                            "Morgan fingerprints + t-SNE. **Red dots** are your AI-generated molecules. "
                            "**Gray dots** are 50 known FDA-approved drugs. Hover over any point to "
                            "see its SMILES and properties."
                        )

                # ── Tab 3: Drug Comparison ──
                with tab_drugs:
                    with st.spinner("Comparing generated molecules against known drug database..."):
                        comparisons = find_closest_drugs(
                            [r["SMILES"] for r in results], top_k=3
                        )

                    if not comparisons:
                        st.warning("Drug comparison database not available.")
                    else:
                        for idx, comp in enumerate(comparisons):
                            with st.container(border=True):
                                st.markdown(f"**Molecule #{idx + 1}**  `{comp['generated']}`")

                                if comp["matches"]:
                                    match_data = []
                                    for m in comp["matches"]:
                                        match_data.append({
                                            "🏷️ Drug Name": m["name"],
                                            "📊 Similarity": f"{m['similarity']}%",
                                            "💊 Category": m["category"],
                                            "🧪 Drug SMILES": m["smiles"][:45] + ("..." if len(m["smiles"]) > 45 else ""),
                                        })
                                    st.dataframe(
                                        pd.DataFrame(match_data),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                else:
                                    st.info("No matches found.")

                        st.caption(
                            "Similarity is computed using **Tanimoto coefficient** on "
                            "**Morgan fingerprints** (radius=2, 2048 bits). Values above 40% "
                            "indicate meaningful structural overlap with a known pharmaceutical compound."
                        )
    else:
        st.info("👈 Enter your settings and click **Generate Molecules** to begin.")
