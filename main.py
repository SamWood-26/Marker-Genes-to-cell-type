# main.py
import streamlit as st
import os
import pandas as pd
from noLLM_analysis import *

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

st.set_page_config(page_title="Cell Type App Landing Page")

# --- Landing Page Content ---
st.title("Cell Type Prediction Platform")
st.markdown("""
Welcome to the Cell Type Prediction Platform!

**About:**  
This app predicts cell types based on user-provided marker genes using curated single-cell datasets and AI-powered algorithms.

**How to Use:**  
- Choose your preferred interface below.
- Follow the instructions in each tab to input your data and get predictions.

---
""")

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Classical interface", "Simple interface"])

with tab1:
    st.header("Classical interface")
    st.write("This is the full-featured interface for advanced users. All options and settings are available here.")
    st.sidebar.success("Select a Page Above")

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
       
       st.success(f"Selected option: {st.session_state.background_context}, {st.session_state.species}, {st.session_state.tissue}, {st.session_state.marker_genes}")
       st.success(f"AI Provider: {st.session_state.selected_api} {api_status}")
       
       if (st.session_state.background_context == "Upload TSV File") and (st.session_state.Gene_denominator.size > 0):
          st.success("File uploaded successfully!")



with tab2:
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
        genes = [g.strip() for g in simple_input.replace('\n', ',').split(',') if g.strip()]
        st.session_state.simple_marker_genes = genes
        st.success(f"Saved {len(genes)} marker genes for simple interface.")
        st.write("Genes:", genes[:10], "..." if len(genes) > 10 else "")
        st.write(f"Total genes entered: {len(genes)}")

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
            st.info(f"Running CellTypist model: {best_source}")
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
                st.table(pd.DataFrame(top5, columns=["Cell Type", "Score"]))
            except Exception as e:
                st.error(f"Error running CellTypist model: {e}")

        elif model_type == "celltaxonomy":
            st.info("Running Cell Taxonomy prediction")
            try:
                tissue_type = ["All"]
                top5 = infer_top_cell_standards_weighted(
                    cell_taxonomy_df[cell_taxonomy_df['Species'] == species_guess],
                    tissue_type, genes, top_n=5
                )
                st.subheader("Top 5 Predicted Cell Types (Cell Taxonomy)")
                st.table(pd.DataFrame(top5, columns=["Cell Type"]))
            except Exception as e:
                st.error(f"Error running Cell Taxonomy prediction: {e}")
        else:
            st.warning("No model selected or insufficient data for prediction.")

    # --- Switch Mode Button ---
    # Only show if a prediction has been made
    if st.session_state.simple_mode in ["celltypist", "celltaxonomy"] and st.session_state.simple_marker_genes:
        switch_label = "Switch to Cell Taxonomy and Run" if st.session_state.simple_mode == "celltypist" else "Switch to CellTypist and Run"
        if st.button(switch_label):
            genes = st.session_state.simple_marker_genes
            species_guess = classify_species_from_genes(genes)
            cell_taxonomy_df = get_cell_taxonomy_df()
            # Switch mode
            if st.session_state.simple_mode == "celltypist":
                # Run Cell Taxonomy
                st.session_state.simple_mode = "celltaxonomy"
                st.info("Running Cell Taxonomy prediction (switched mode)")
                try:
                    tissue_type = ["All"]
                    top5 = infer_top_cell_standards_weighted(
                        cell_taxonomy_df[cell_taxonomy_df['Species'] == species_guess],
                        tissue_type, genes, top_n=5
                    )
                    st.subheader("Top 5 Predicted Cell Types (Cell Taxonomy)")
                    st.table(pd.DataFrame(top5, columns=["Cell Type"]))
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
                        st.table(pd.DataFrame(top5, columns=["Cell Type", "Score"]))
                    except Exception as e:
                        st.error(f"Error running CellTypist model: {e}")
                else:
                    st.warning("No CellTypist model available for these genes/species.")

    else:
        st.session_state.simple_marker_genes = []
        st.info("Enter marker genes to begin.")
