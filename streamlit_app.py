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
from generate import generate_smiles
from property import compute_properties
from utils import load_model, get_checkpoint_path
from analytics import (
    compute_batch_stats,
    compute_tsne_embedding,
    find_closest_drugs,
    load_known_drugs,
)


def generate_3d_mol_block(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
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


st.set_page_config(page_title="AI Molecular Engine", page_icon="🧬", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.title("🧬 AI Molecular Generator")
st.markdown("Transform SMILES seeds into 3D Molecular Structures using Deep Learning.")
st.divider()


@st.cache_resource
def get_model():
    checkpoint_path = get_checkpoint_path("best_model.pt")
    if not os.path.exists(checkpoint_path):
        return None, None
    try:
        model, tokenizer_data = load_model(MoleculeTransformer, checkpoint_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None

    tokenizer            = SmilesTokenizer()
    tokenizer.stoi       = tokenizer_data.get("stoi", {})
    tokenizer.itos       = {int(k): v for k, v in tokenizer_data.get("itos", {}).items()}
    tokenizer.vocab_size = tokenizer_data.get("vocab_size", len(tokenizer.stoi))
    return model, tokenizer


model, tokenizer = get_model()

if model is None:
    st.error("Could not load model checkpoint. Please train the model first.")
    st.code("python src/train.py --epochs 20")
    st.stop()


col_settings, col_results = st.columns([1, 2.5], gap="large")

with col_settings:
    st.header("⚙️ Settings")
    seed          = st.text_input("Seed Token", value="C",
                                  help="Starting SMILES fragment e.g. C, N, c1ccccc1")
    num_molecules = st.number_input("Molecules to Generate", min_value=1, max_value=20, value=6)
    temperature   = st.slider("Temperature", min_value=0.1, max_value=1.5,
                              value=0.8, step=0.1,
                              help="Higher = more creative, Lower = more conservative")
    generate_btn  = st.button("🚀 Generate Molecules", type="primary",
                              use_container_width=True)

    st.divider()
    st.subheader("📈 Training History")
    loss_csv = get_checkpoint_path("loss_history.csv")
    if os.path.exists(loss_csv):
        df_loss = pd.read_csv(loss_csv)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df_loss["epoch"], y=df_loss["train_loss"],
            name="Train Loss", line=dict(color="#667eea", width=2)
        ))
        fig_loss.add_trace(go.Scatter(
            x=df_loss["epoch"], y=df_loss["val_loss"],
            name="Val Loss", line=dict(color="#ff6b6b", width=2, dash="dash")
        ))
        fig_loss.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Train vs Val Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.caption("Train the model to see loss curves here.")


with col_results:
    if generate_btn:
        if not seed:
            st.warning("Please enter a valid seed token.")
        else:
            st.header("🔬 Discovery Results")
            with st.spinner("Generating molecules..."):
                results      = []
                attempts     = 0
                max_attempts = num_molecules * 10

                while len(results) < num_molecules and attempts < max_attempts:
                    attempts += 1
                    smiles = generate_smiles(
                        model, tokenizer,
                        seed        = seed,
                        max_len     = 60,
                        temperature = temperature,
                        device      = "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    props = compute_properties(smiles)
                    if not props["valid"]:
                        continue

                    results.append({
                        "smiles":    smiles,
                        "molblock":  generate_3d_mol_block(smiles),
                        "mw":        props["mw"],
                        "logp":      props["logp"],
                        "qed":       props["qed"],
                        "sa_score":  props["sa_score"],
                        "tpsa":      props["tpsa"],
                        "hbd":       props["hbd"],
                        "hba":       props["hba"],
                        "lipinski":  props["lipinski"],
                    })

            if not results:
                st.warning("No valid molecules generated. Try a different seed or higher temperature.")
            else:
                if len(results) < num_molecules:
                    st.info(f"Found {len(results)} valid molecules from {attempts} attempts.")

                grid_cols = st.columns(3)
                for idx, res in enumerate(results):
                    with grid_cols[idx % 3]:
                        with st.container(border=True):
                            lip = "✅ Lipinski PASS" if res["lipinski"] else "⚠️ Lipinski FAIL"
                            st.markdown(f"**Molecule #{idx+1}** — {lip}")
                            st.code(res["smiles"])
                            st.caption(
                                f"**MW:** {res['mw']}  |  "
                                f"**LogP:** {res['logp']}  |  "
                                f"**QED:** {res['qed']}  |  "
                                f"**SA:** {res['sa_score']}"
                            )
                            if res["molblock"]:
                                view = py3Dmol.view(width=280, height=220)
                                view.addModel(res["molblock"], "mol")
                                view.setStyle({
                                    "stick":  {"radius": 0.15},
                                    "sphere": {"scale": 0.25}
                                })
                                view.zoomTo()
                                view.setBackgroundColor("#0e1117")
                                components.html(view._make_html(), height=230, width=290)
                            else:
                                st.info("3D coordinates unavailable.")

                st.divider()
                st.header("📊 AI / ML Analysis")

                tab_analytics, tab_space, tab_drugs = st.tabs([
                    "📊 Batch Analytics",
                    "🗺️ Chemical Space",
                    "💊 Drug Comparison",
                ])

                with tab_analytics:
                    stats = compute_batch_stats(results, attempts)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    for col, label, value in zip(
                        [m1, m2, m3, m4, m5],
                        ["Validity", "Diversity", "Avg QED", "Avg MW", "Lipinski Pass"],
                        [f"{stats['validity_rate']}%",
                         stats['diversity'],
                         stats['avg_qed'],
                         stats['avg_mw'],
                         f"{stats['lipinski_pass']}/{stats['num_valid']}"]
                    ):
                        with col:
                            st.markdown(f"""<div class='metric-card'>
                                <div class='metric-value'>{value}</div>
                                <div class='metric-label'>{label}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    chart1, chart2, chart3 = st.columns(3)
                    with chart1:
                        fig = px.histogram(x=stats["mw_list"],
                            title="MW Distribution",
                            labels={"x": "Molecular Weight"},
                            color_discrete_sequence=["#667eea"])
                        fig.update_layout(template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    with chart2:
                        fig = px.histogram(x=stats["logp_list"],
                            title="LogP Distribution",
                            labels={"x": "LogP"},
                            color_discrete_sequence=["#764ba2"])
                        fig.update_layout(template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    with chart3:
                        fig = px.histogram(x=stats["qed_list"],
                            title="QED Distribution",
                            labels={"x": "QED Score"},
                            color_discrete_sequence=["#ff6b6b"])
                        fig.update_layout(template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                with tab_space:
                    with st.spinner("Computing t-SNE..."):
                        known_drugs = load_known_drugs()
                        points      = compute_tsne_embedding(
                            [r["smiles"] for r in results],
                            [{"mw": r["mw"], "logp": r["logp"]} for r in results],
                            reference_smiles = [d["SMILES"] for d in known_drugs],
                            reference_names  = [d["Name"]   for d in known_drugs],
                            perplexity       = 5,
                        )

                    if not points:
                        st.warning("Not enough molecules for t-SNE.")
                    else:
                        df_pts   = pd.DataFrame(points)
                        fig_tsne = px.scatter(
                            df_pts, x="x", y="y", color="group",
                            color_discrete_map={
                                "Generated": "#ff6b6b",
                                "Known Drug": "#b8b0a1"
                            },
                            title="Chemical Space: Generated vs Known Drugs",
                            custom_data=["label", "smiles", "mw", "logp"],
                        )
                        fig_tsne.update_traces(
                            marker=dict(size=10, line=dict(width=0)),
                            hoverlabel=dict(
                                bgcolor="#111827",
                                bordercolor="#334155",
                                font=dict(color="#f8fafc", size=14),
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "SMILES: %{customdata[1]}<br>"
                                "MW: %{customdata[2]}<br>"
                                "LogP: %{customdata[3]}<extra></extra>"
                            )
                        )
                        fig_tsne.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=500,
                            legend=dict(
                                bgcolor="rgba(15, 23, 42, 0.85)",
                                bordercolor="#334155",
                                borderwidth=1,
                            ),
                        )
                        st.plotly_chart(fig_tsne, use_container_width=True)

                with tab_drugs:
                    with st.spinner("Comparing against drug database..."):
                        comparisons = find_closest_drugs(
                            [r["smiles"] for r in results], top_k=3
                        )
                    if not comparisons:
                        st.warning("Drug database not found at data/known_drugs.csv")
                    else:
                        for idx, comp in enumerate(comparisons):
                            with st.container(border=True):
                                st.markdown(f"**Molecule #{idx+1}** `{comp['generated']}`")
                                if comp["matches"]:
                                    st.dataframe(
                                        pd.DataFrame([{
                                            "Drug":       m["name"],
                                            "Similarity": f"{m['similarity']}%",
                                            "Category":   m["category"],
                                            "SMILES":     m["smiles"][:45] + "..." if len(m["smiles"]) > 45 else m["smiles"],
                                        } for m in comp["matches"]]),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                else:
                                    st.info("No matches found.")
    else:
        st.info("👈 Configure settings and click **Generate Molecules** to begin.")
