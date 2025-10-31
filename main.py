# main.py
import streamlit as st
import os
import pandas as pd
from noLLM_analysis import *
import numpy as np
import io, pickle, requests
import json, re
from collections import OrderedDict
# from GSEA_function import run_enrichr_enrichment

# Flexible parser for gene input
_num_pat = re.compile(r"""^[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?$""")

def _is_number(s: str) -> bool:
    return bool(_num_pat.match(s.strip()))

def parse_genes_flexible(text: str):
    """
    Parse marker genes from a flexible free-text field.

    Accepts:
      - Comma-, semicolon-, space-, tab-, newline-separated lists: e.g. "LYZ, S100A9  AIF1"
      - Pairs with weights: "LYZ:0.91", "LYZ 0.91", "LYZ\t0.91"
      - JSON list: ["LYZ","S100A9","AIF1"]
      - JSON dict: {"LYZ": 0.91, "S100A9": 0.83}
    Returns:
      - genes: list[str] (unique, order-preserving)
      - weights: dict[str, float] (only for entries with a score)
    """
    text = (text or "").strip()
    genes_order = OrderedDict()
    weights = {}

    if not text:
        return [], {}

    # 1) Try JSON first
    if text[0] in "[{":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        g = item.strip().strip("'\"")
                        if g:
                            genes_order[g] = True
                    elif isinstance(item, dict):
                        gene = item.get("gene") or item.get("name")
                        score = item.get("score") or item.get("weight")
                        if isinstance(gene, str):
                            g = gene.strip().strip("'\"")
                            if g:
                                genes_order[g] = True
                                if isinstance(score, (int, float)) or (isinstance(score, str) and _is_number(score)):
                                    weights[g] = float(score)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    g = str(k).strip().strip("'\"")
                    if g:
                        genes_order[g] = True
                        if isinstance(v, (int, float)) or (isinstance(v, str) and _is_number(v)):
                            weights[g] = float(v)
            return list(genes_order.keys()), weights
        except Exception:
            pass  # fall through to plain-text parsing

    # 2) Plain text parsing
    chunks = re.split(r"[,\n;]+", text)
    for raw in chunks:
        s = raw.strip()
        if not s:
            continue

        # gene:score
        if ":" in s:
            left, right = s.split(":", 1)
            g = left.strip().strip("'\"")
            val = right.strip()
            if g:
                genes_order[g] = True
                if _is_number(val):
                    weights[g] = float(val)
            continue

        # whitespace-delimited
        toks = re.split(r"\s+", s)
        if len(toks) == 2 and _is_number(toks[1]):
            g = toks[0].strip().strip("'\"")
            if g:
                genes_order[g] = True
                weights[g] = float(toks[1])
            continue

        for t in toks:
            g = t.strip().strip("'\"")
            if g:
                genes_order[g] = True

    return list(genes_order.keys()), weights


# --- Session State Initialization ---
if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = None
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""
if "selected_api" not in st.session_state:
    st.session_state.selected_api = "OpenAI"
if "background_context" not in st.session_state:
    st.session_state.background_context = ""
if "Gene_denominator" not in st.session_state:
    st.session_state.Gene_denominator = ""
if "custom_data" not in st.session_state:
    st.session_state.custom_data = ""
if "simple_marker_genes" not in st.session_state:
    st.session_state.simple_marker_genes = ""
if "show_sidebar_pages" not in st.session_state:
    # Controls whether the app's sidebar widgets/content are shown.
    # Default False until user clicks "Save Selection" in the In-Depth Interface.
    st.session_state.show_sidebar_pages = False

st.set_page_config(page_title="Cell Type App Landing Page")

# --- Landing Page Content ---
# st.title("Cell Type Prediction Platform")
# st.markdown("""
# Welcome to the Cell Type Prediction Platform!

# **About:**  
# This app predicts cell types based on user-provided marker genes using curated single-cell datasets, established algorithms, a new algorithm built with CellTypist, and multi-platform AI integration.

# **How to Use:**  
# - Choose your preferred interface below.
# - Follow the instructions in each tab to input your data and get predictions.

# ---
# """)
# --- New landing Page content ---
st.title("Cell Type Prediction Platform")

# Top-row: brief description + Tutorial/Help
colL, colR = st.columns([3, 1])
with colL:
    st.markdown(
        "This app predicts cell types based on user-provided marker genes using curated "
        "single-cell datasets, established algorithms, a new algorithm built with "
        "CellTypist, and multi-platform AI integration. "
        "Select your options, Simple or In-Depth interface, then enter your marker genes"
        "and hit **Run**."
    )
with colR:
    # If you add a 'pages/Help.py', this will deep-link to it; otherwise we show an expander.
    try:
        st.page_link("pages/Help.py", label="📘 Tutorial / Help", icon=":material/help:")
    except Exception:
        pass

with st.expander("Quick Tutorial"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Simple Interface")
        st.markdown(
            """
            - Paste genes separated by commas or newlines  
              e.g. `LYZ, S100A9, AIF1`
            - Click **Run** to analyze using the best modules available  
              based on your gene list and species  
              (species is auto-detected(human or mouse))
            """
        )

    with col2:
        st.markdown("###  In-Depth Interface")
        st.markdown(
            """
            - Supports advanced configurations and filters  
              (species, tissue type, and gene subset)
            - Choose background gene list to run on:  
              **Cell Taxonomy**, **Mouse Liver**, **Human Breast Cancer**, **Upload TSV File**, or **Custom Input**
            - Then select species(human or mouse)
            - Select **tissue type(s)** defualts to "All"
            - Next paste your **marker genes** separated by commas into the text area
            - Then if you want to use AI models, select your preferred AI provider and enter your API key
            - Click **Save Selection** to start analysis and  
              view reslts on the 5 pages available
            """
        )
# --- Tabbed Interface (Simple tab first) ---
tab_simple, tab_classical = st.tabs(["Simple interface", "In-Depth Interface"])

with tab_classical:
    st.header("In-Depth Interface")
    st.write("This is the full-featured interface for advanced users. All options and settings are available here.")
    # Show sidebar content only after Save Selection is pressed.
    if st.session_state.show_sidebar_pages:
        st.sidebar.success("Select a Page Above")
    else:
        # Minimal sidebar hint while sidebar content is hidden
        st.sidebar.info("Configure options here and click 'Save Selection' to enable navigation.")

    #loading data
    @st.cache_data
    def get_data():
       df = load_data()
       df_human = df[df['Species'] == 'Homo sapiens']
       df_mouse = df[df['Species'] == 'Mus musculus']

       total_cells = df['Cell_standard'].nunique()
       human_cells = df_human['Cell_standard'].nunique()
       mouse_cells = df_mouse['Cell_standard'].nunique()
       return df, df_human, df_mouse, total_cells, human_cells, mouse_cells

    #data frames loaded in
    df, df_human, df_mouse, total_cells, human_cells, mouse_cells = get_data()

    st.session_state.background_context = st.selectbox(
       "Select Dataset",
       ["Base","Mouse Liver", "Human Breast Cancer", "Upload TSV File", "Custom Input"]
    )

    # File uploader appears only if 'Upload TSV File' is selected
    if st.session_state.background_context == "Upload TSV File":
        uploaded_file = st.file_uploader("Upload your TSV file", type=["tsv"])

        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file, sep="\t")  # Read TSV file
            if not df_uploaded.empty:
                first_column_name = df_uploaded.columns[0]  # Get first column name
                marker_genes_from_file = df_uploaded[first_column_name].dropna().astype(str).tolist()  # Extract and clean data
                st.session_state.Gene_denominator = np.array(marker_genes_from_file)  # Store in session state

                st.success(f"Extracted {len(marker_genes_from_file)} marker genes from file.")
                st.write("Extracted Marker Genes:", marker_genes_from_file)
            else:
                st.warning("The uploaded file is empty. Please check your file.")

    # Text area appears only if 'Custom Input' is selected
    elif st.session_state.background_context == "Custom Input":
       st.session_state.custom_data = st.text_area("Enter custom gene dataset:")

    #selecting Species as global variable
    species = st.radio(
          "Select Species",
          ("Homo sapiens", "Mus musculus")
       )
    if species == 'Homo sapiens':
       st.session_state.species = "Homo sapiens"
       df_selected = df_human
       total_species_cells = human_cells
    else:
       st.session_state.species = "Mus musculus"
       df_selected = df_mouse
       total_species_cells = mouse_cells

    #selecting Tissue as global variable
    tissue_options = get_all_tissues(df_selected, species)
    selected_tissues = st.multiselect(
       "Select Tissue Type(s)",
       options=tissue_options,
       default=["All"]
    )
    st.session_state.tissue = selected_tissues

    #selcting Marker Genes as Global Variable
    marker_genes_input = st.text_area(
        "Enter Marker Genes",
        placeholder="Enter marker genes, separated by commas. Ex: Gpx2, Rps12, Rpl12, Eef1a1, Rps19, Rpsa, Rps3, "
        "Rps26, Rps24, Rps28, Reg4, Cldn2, Cd24a, Zfas1, Stmn1, Kcnq1, Rpl36a-ps1, Hopx, Cdca7, Smoc2"
    )
    marker_genes = string_to_gene_array(marker_genes_input)
    st.session_state.marker_genes = marker_genes

    # API Selection Section
    st.write("## AI Model Configuration")
    st.write("**Select your preferred AI provider and enter the corresponding API key**")

    # API Provider Selection
    st.session_state.selected_api = st.radio(
        "Choose AI Provider:",
        ["OpenAI", "Google AI"],
        help="Select which AI service you want to use for pages 2, 3, and 4"
    )

    # API Key Input based on selection
    if st.session_state.selected_api == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            "Enter your OpenAI API key", 
            type="password", 
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        if st.session_state.openai_api_key:
            st.success("OpenAI API key configured")
            
    elif st.session_state.selected_api == "Google AI":
        st.session_state.google_api_key = st.text_input(
            "Enter your Google AI API key", 
            type="password", 
            placeholder="AIza...",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        if st.session_state.google_api_key:
            st.success("Google AI API key configured")

    # Show which pages will be affected
    if st.session_state.selected_api == "OpenAI" and st.session_state.openai_api_key:
        st.info("Pages 2, 3, and 4 will use OpenAI GPT models")
    elif st.session_state.selected_api == "Google AI" and st.session_state.google_api_key:
        st.info("Pages 2, 3, and 4 will use Google Gemini models")
    else:
        st.warning("Please configure an API key to use AI-powered features on pages 2, 3, and 4")

    if st.button("Save Selection"):
       api_status = "Configured" if (
           (st.session_state.selected_api == "OpenAI" and st.session_state.openai_api_key) or
           (st.session_state.selected_api == "Google AI" and st.session_state.google_api_key)
       ) else "Not configured"
       
       # Reveal the app's sidebar widgets/content after saving selection
       st.session_state.show_sidebar_pages = True
       
       st.success(f"Selected option: {st.session_state.background_context}, {st.session_state.species}, {st.session_state.tissue}, {st.session_state.marker_genes}")
       st.success(f"AI Provider: {st.session_state.selected_api} {api_status}")
       
       if (st.session_state.background_context == "Upload TSV File") and (st.session_state.Gene_denominator.size > 0):
          st.success("File uploaded successfully!")

with tab_simple:
    st.header("Simple interface")
    st.write("Paste your marker gene list below. This interface is for quick predictions with minimal options.")

    @st.cache_data
    def get_cell_taxonomy_df():
        return load_data()
    cell_taxonomy_df = get_cell_taxonomy_df()

    # Use a form so the user enters genes and clicks a single "Run" button
    with st.form("simple_gene_form"):
        simple_input = st.text_area(
            "Enter marker genes (comma or newline separated):",
            placeholder="e.g. Gpx2, Rps12, Rpl12, ...",
            key="simple_input"
        )
        run_clicked = st.form_submit_button("Run")

    # Store last mode in session state for switching
    if "simple_mode" not in st.session_state:
        st.session_state.simple_mode = None

    if run_clicked and simple_input:
        genes, weights = parse_genes_flexible(simple_input)
        st.session_state.simple_marker_genes = genes
        weights = weights  # optional: store any provided weights

        st.success(f"Saved {len(genes)} marker genes for simple interface.")
        if weights:
            st.info(f"{len(weights)} genes include weights/scores.")
        st.write("Genes:", genes[:10], "..." if len(genes) > 10 else "")
        st.write(f"Total genes entered: {len(genes)}")

        # Define CellTypist sources
        celltypist_sources_human = {
            "Adult_COVID19_PBMC": "https://celltypist.cog.sanger.ac.uk/models/COVID19_PBMC_Wilk/v1/Adult_COVID19_PBMC.pkl",
            "Adult_Human_PrefrontalCortex": "https://celltypist.cog.sanger.ac.uk/models/Human_PFC_Ma/v1/Adult_Human_PrefrontalCortex.pkl",
            "Cells_Adult_Breast": "https://celltypist.cog.sanger.ac.uk/models/Adult_Breast_Kumar/v1/Cells_Adult_Breast.pkl",
            "Healthy_Adult_Heart":"https://celltypist.cog.sanger.ac.uk/models/Human_Heart_Kanemaru/v1/Healthy_Adult_Heart.pkl"
        }
        celltypist_sources_mouse = {
            "Developing_Mouse_Brain":"https://celltypist.cog.sanger.ac.uk/models/Mouse_Devbrain_Manno/v1/Developing_Mouse_Brain.pkl",
            "Mouse_Whole_Brain":"https://celltypist.cog.sanger.ac.uk/models/Adult_MouseBrain_Yao/v1/Mouse_Whole_Brain.pkl"
        }

        species_guess = classify_species_from_genes(genes)
        st.info(f"**Species:** {species_guess}")

        # --- Model Recommendation ---
        model_type, best_source, best_count = None, None, 0
        if species_guess in ["Homo sapiens", "Mus musculus"]:
            with st.spinner("Comparing gene coverage in CellTypist and Cell Taxonomy..."):
                model_type, best_source, best_count = recommend_model_for_genes(
                    species_guess,
                    genes,
                    celltypist_sources_human=celltypist_sources_human,
                    celltypist_sources_mouse=celltypist_sources_mouse,
                    cell_taxonomy_df=cell_taxonomy_df,
                    celltypist_threshold=0.7
                )
            st.session_state.simple_mode = model_type  # Save mode for switching
            if model_type == "celltypist":
                st.success(f"**Recommended: CellTypist** (best match: {best_source}, {best_count} genes found)")
            elif model_type == "celltaxonomy":
                st.success(f"**Recommended: Cell Taxonomy** ({best_count} genes found in database)")
            else:
                st.warning("Could not determine the best model for your gene list.")
        else:
            st.info("Species could not be determined. Model recommendation unavailable.")

        # --- Run prediction and display results ---
        if model_type == "celltypist" and best_source:
            st.info(f"Running CellTypist model: {best_source} (Exact Match)")
            try:
                import pickle
                import requests
                import io
                url = celltypist_sources_human[best_source] if species_guess == "Homo sapiens" else celltypist_sources_mouse[best_source]
                response = requests.get(url)
                model = pickle.load(io.BytesIO(response.content))
                model_genes = set(model["feature_names"])
                input_genes = set(genes) & model_genes
                cell_type_markers = model["cell_types"]
                # Exact match: overlap count for each cell type (no weighting)
                scores = {}
                for cell_type, marker_set in cell_type_markers.items():
                    overlap = len(input_genes & set(marker_set))
                    scores[cell_type] = overlap
                top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
                st.subheader("Top 5 Predicted Cell Types (CellTypist, Exact Match)")
                st.dataframe(pd.DataFrame(top5, columns=["Cell Type", "Score"]), height=220, use_container_width=True)
            except Exception as e:
                st.error(f"Error running CellTypist model: {e}")

        elif model_type == "celltaxonomy":
            st.info("Running Cell Taxonomy prediction")
            try:
                tissue_type = ["All"]
                df_filtered = cell_taxonomy_df[cell_taxonomy_df['Species'] == species_guess]
                top5_cells = infer_top_cell_standards(
                    df_filtered,
                    tissue_type, genes, top_n=5
                )
                # Robust marker extraction and overlap calculation
                # --- FIX: Lowercase all marker genes for comparison ---
                input_genes_set = set(g.lower() for g in genes)
                cell_type_markers = (
                    df_filtered.groupby("Cell_standard")["Cell_Marker"]
                    .apply(lambda x: set(gene.strip().lower() for gene in ",".join(x).split(",") if gene.strip()))
                    .to_dict()
                )
                overlap_data = []
                query_gene_count = len(input_genes_set)
                for cell in top5_cells:
                    markers = cell_type_markers.get(cell, set())
                    total_markers = len(markers)
                    if total_markers == 0:
                        st.warning(f"Cell type '{cell}' has no marker genes in the database.")
                        overlap_count = 0
                        overlap_pct = 0
                    else:
                        overlap_count = len(input_genes_set & markers)
                        overlap_pct = (overlap_count / total_markers * 100)
                    overlap_data.append({
                        "Cell Type": cell,
                        "Query Genes": query_gene_count,
                        "CellType Markers": total_markers,
                        "Overlap Count": overlap_count,
                        "Overlap %": f"{overlap_pct:.1f}%"
                    })
                st.subheader("Top 5 Predicted Cell Types (Cell Taxonomy)")
                st.dataframe(pd.DataFrame(overlap_data), height=220, use_container_width=True)
            except Exception as e:
                st.error(f"Error running Cell Taxonomy prediction: {e}")
        else:
            st.warning("No model selected or insufficient data for prediction.")

        # --- Run GSEA (Enrichr) for the same gene list (always) ---
        with st.spinner("Running Enrichr enrichment..."):
            try:
                gsea_results, gsea_errors = run_enrichr_enrichment(genes)
            except Exception as e:
                gsea_results, gsea_errors = {}, {None: str(e)}

        # show submission / fetch errors
        if gsea_errors:
            for k, msg in gsea_errors.items():
                if k is None:
                    st.error(f"Enrichr submission error: {msg}")
                else:
                    st.warning(f"Enrichr error for {k}: {msg}")

        # display tables two-per-row and compact
        if gsea_results:
            st.subheader("Enrichr: Cell-Type Enrichment Results")
            items = list(gsea_results.items())
            for i in range(0, len(items), 2):
                cols = st.columns(2)
                for j in (0, 1):
                    idx = i + j
                    if idx >= len(items):
                        break
                    lib, df_lib = items[idx]
                    with cols[j]:
                        st.write(f"### {lib}")
                        if df_lib is None or df_lib.empty:
                            st.info("No enrichment results.")
                        else:
                            st.dataframe(df_lib, height=220, use_container_width=True)
                            csv_bytes = df_lib.to_csv(index=False).encode("utf-8")
                            fname = f"enrichr_{lib.replace(' ', '_')}.csv"
                            st.download_button(f"Download {lib} CSV", data=csv_bytes, file_name=fname, mime="text/csv")
        else:
            st.info("No Enrichr results to display.")

    # --- Switch Mode Button ---
    # Only show if a prediction has been made
    if st.session_state.simple_mode in ["celltypist", "celltaxonomy"] and st.session_state.simple_marker_genes:
        switch_label = "Switch to Cell Taxonomy and Run" if st.session_state.simple_mode == "celltypist" else "Switch to CellTypist and Run"
        if st.button(switch_label):
            genes = st.session_state.simple_marker_genes
            species_guess = classify_species_from_genes(genes)
            cell_taxonomy_df = get_cell_taxonomy_df()
            if st.session_state.simple_mode == "celltypist":
                st.session_state.simple_mode = "celltaxonomy"
                st.info("Running Cell Taxonomy prediction (switched mode)")
                try:
                    tissue_type = ["All"]
                    df_filtered = cell_taxonomy_df[cell_taxonomy_df['Species'] == species_guess]
                    top5_cells = infer_top_cell_standards(
                        df_filtered,
                        tissue_type, genes, top_n=5
                    )
                    input_genes_set = set(g.lower() for g in genes)
                    cell_type_markers = (
                        df_filtered.groupby("Cell_standard")["Cell_Marker"]
                        .apply(lambda x: set(gene.strip().lower() for gene in ",".join(x).split(",") if gene.strip()))
                        .to_dict()
                    )
                    overlap_data = []
                    query_gene_count = len(input_genes_set)
                    for cell in top5_cells:
                        markers = cell_type_markers.get(cell, set())
                        total_markers = len(markers)
                        if total_markers == 0:
                            st.warning(f"Cell type '{cell}' has no marker genes in the database.")
                            overlap_count = 0
                            overlap_pct = 0
                        else:
                            overlap_count = len(input_genes_set & markers)
                            overlap_pct = (overlap_count / total_markers * 100)
                        overlap_data.append({
                            "Cell Type": cell,
                            "Query Genes": query_gene_count,
                            "CellType Markers": total_markers,
                            "Overlap Count": overlap_count,
                            "Overlap %": f"{overlap_pct:.1f}%"
                        })
                    st.subheader("Top 5 Predicted Cell Types (Cell Taxonomy)")
                    st.dataframe(pd.DataFrame(overlap_data), height=220, use_container_width=True)
                except Exception as e:
                    st.error(f"Error running Cell Taxonomy prediction: {e}")
            else:
                # Run CellTypist
                st.session_state.simple_mode = "celltypist"
                # Find best CellTypist source again
                celltypist_sources_human = {
                    "Adult_COVID19_PBMC": "https://celltypist.cog.sanger.ac.uk/models/COVID19_PBMC_Wilk/v1/Adult_COVID19_PBMC.pkl",
                    "Adult_Human_PrefrontalCortex": "https://celltypist.cog.sanger.ac.uk/models/Human_PFC_Ma/v1/Adult_Human_PrefrontalCortex.pkl",
                    "Cells_Adult_Breast": "https://celltypist.cog.sanger.ac.uk/models/Adult_Breast_Kumar/v1/Cells_Adult_Breast.pkl",
                    "Healthy_Adult_Heart":"https://celltypist.cog.sanger.ac.uk/models/Human_Heart_Kanemaru/v1/Healthy_Adult_Heart.pkl"
                }
                celltypist_sources_mouse = {
                    "Developing_Mouse_Brain":"https://celltypist.cog.sanger.ac.uk/models/Mouse_Devbrain_Manno/v1/Developing_Mouse_Brain.pkl",
                    "Mouse_Whole_Brain":"https://celltypist.cog.sanger.ac.uk/models/Adult_MouseBrain_Yao/v1/Mouse_Whole_Brain.pkl"
                }
                model_type, best_source, best_count = recommend_model_for_genes(
                    species_guess,
                    genes,
                    celltypist_sources_human=celltypist_sources_human,
                    celltypist_sources_mouse=celltypist_sources_mouse,
                    cell_taxonomy_df=cell_taxonomy_df,
                    celltypist_threshold=0.0 # force CellTypist
                )
                if best_source:
                    st.info(f"Running CellTypist model: {best_source} (switched mode)")
                    try:
                        import pickle
                        import requests
                        import io
                        url = celltypist_sources_human[best_source] if species_guess == "Homo sapiens" else celltypist_sources_mouse[best_source]
                        response = requests.get(url)
                        model = pickle.load(io.BytesIO(response.content))
                        model_genes = set(model["feature_names"])
                        input_genes = set(genes) & model_genes
                        cell_type_markers = model["cell_types"]
                        scores = {}
                        for cell_type, marker_set in cell_type_markers.items():
                            overlap = len(input_genes & set(marker_set))
                            scores[cell_type] = overlap
                        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
                        st.subheader("Top 5 Predicted Cell Types (CellTypist)")
                        st.dataframe(pd.DataFrame(top5, columns=["Cell Type", "Score"]), height=220, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error running CellTypist model: {e}")
                else:
                    st.warning("No CellTypist model available for these genes/species.")

    else:
        st.session_state.simple_marker_genes = []
        st.info("Enter marker genes to begin.")



