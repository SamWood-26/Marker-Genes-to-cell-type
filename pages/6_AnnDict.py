import streamlit as st
import anndict

from anndict_backend import (
    AnnDictLLMConfig,
    parse_genes_flexible,
    predict_cell_types_from_markers,
)

st.set_page_config(page_title="AnnDictionary: Marker Genes → Cell Types", layout="centered")
st.title("AnnDictionary: Marker Genes → Probable Cell Types")

llm_provider = "openai"
llm_model = "gpt-4o-mini" # gpt-4o-mini, gpt-5.2
llm_api_key = ""

if not (llm_provider and llm_model and llm_api_key):
    st.warning("Please open config.py and set your llm_provider / llm_model / llm_api_key.")
    st.stop()
llm_api_key

anndict.configure_llm_backend(llm_provider, llm_model, api_key=llm_api_key)
# Optional: show what anndict thinks is configured
st.write("LLM config:", anndict.get_llm_config())

with st.expander("LLM config", expanded=False):
    st.caption("Best practice: set these in Streamlit secrets for deployment.")
    llm_provider = st.text_input("Provider", value=llm_provider, placeholder="openai / anthropic / google / ...")
    llm_model = st.text_input("Model", value=llm_model, placeholder="provider-specific model name")
    llm_api_key = st.text_input("API key", value=llm_api_key, type="password")

# ---- Inputs ----
species = st.selectbox("Species", ["Homo sapiens", "Mus musculus"], index=0)
tissue = st.text_input("Optional tissue / context (helps disambiguate)", value="")

markers_text = st.text_area(
    "Marker genes (comma / space / newline separated)",
    height=140,
    placeholder="MS4A1, CD79A, CD74\nHLA-DRA\nCD37",
)

top_k = st.slider("How many predictions to show", min_value=1, max_value=10, value=5)

if st.button("Predict cell types", type="primary"):
    genes = parse_genes_flexible(markers_text)
    if not genes:
        st.error("Please enter at least 1 marker gene.")
        st.stop()

    if not (llm_provider and llm_model and llm_api_key):
        st.error("Missing LLM config. Fill provider/model/api_key (or set Streamlit secrets).")
        st.stop()

    cfg = AnnDictLLMConfig(provider=llm_provider, model=llm_model, api_key=llm_api_key)

    with st.spinner("Querying AnnDictionary / LLM..."):
        preds = predict_cell_types_from_markers(
            species=species,
            marker_genes=genes,
            tissue=(tissue.strip() or None),
            top_k=top_k,
            llm_cfg=cfg,
        )

    st.subheader("Predicted cell types")
    if not preds:
        st.warning("No prediction returned (or an error occurred). Try fewer genes / add tissue context.")
    else:
        for i, p in enumerate(preds, start=1):
            st.write(f"**{i}.** {p}")

    st.divider()
    st.caption(f"Input genes ({len(genes)}): " + ", ".join(genes))
