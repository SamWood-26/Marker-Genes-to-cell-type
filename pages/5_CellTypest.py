import streamlit as st
import pandas as pd
from celltypist_utils import *


st.title("Bayesian Approach Using CellTypest Interaction Page")

if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = ["All"]
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""



# Define data sources for human and mouse
data_sources_human = {
    "Adult_COVID19_PBMC": "https://celltypist.cog.sanger.ac.uk/models/COVID19_PBMC_Wilk/v1/Adult_COVID19_PBMC.pkl",
    "Adult_Human_PrefrontalCortex": "https://celltypist.cog.sanger.ac.uk/models/Human_PFC_Ma/v1/Adult_Human_PrefrontalCortex.pkl",
    "Cells_Adult_Breast": "https://celltypist.cog.sanger.ac.uk/models/Adult_Breast_Kumar/v1/Cells_Adult_Breast.pkl",
    "Healthy_Adult_Heart":"https://celltypist.cog.sanger.ac.uk/models/Human_Heart_Kanemaru/v1/Healthy_Adult_Heart.pkl",
    "Human_Lung_Atlas":"https://celltypist.cog.sanger.ac.uk/models/Human_Lung_Sikkema/v2/Human_Lung_Atlas.pkl"
}

data_sources_mouse = {
    "Developing_Mouse_Brain":"https://celltypist.cog.sanger.ac.uk/models/Mouse_Devbrain_Manno/v1/Developing_Mouse_Brain.pkl",
    "Mouse_Whole_Brain":"https://celltypist.cog.sanger.ac.uk/models/Adult_MouseBrain_Yao/v1/Mouse_Whole_Brain.pkl"
}

# Select the correct data_sources dictionary based on species
if st.session_state.species == "Homo sapiens":
    data_sources = data_sources_human
elif st.session_state.species == "Mus musculus":
    data_sources = data_sources_mouse
else:
    data_sources = {}

# Only show database selection if there are options
if data_sources:
    database_choice = st.selectbox(
        "Select CellTypist Database",
        list(data_sources.keys()),
        key="celltypist_database"
    )
    st.session_state.background_context = database_choice
else:
    database_choice = None

# Assign marker_genes from session state for local use
marker_genes = st.session_state.marker_genes

st.write(f"**Database Selected:** {database_choice if database_choice else 'None'}")

if marker_genes is not None and marker_genes != "":
    # Here, we add the selectbox for the weighting method
    weighting_method = st.selectbox(
        "Select Weighting Method",
        ["Weighted", "Unweighted"]
    )

    # Add slider for weight decay if weighted method is selected
    weight_decay = 0.9  # default value
    if weighting_method == "Weighted":
        st.markdown(
            "#### Weight Decay Factor\n"
            "Pick a value between 0 and 1. Lower values down-weight later genes more strongly. "
            "A value close to 1 means all genes are weighted similarly; closer to 0 means only the first few genes matter."
        )
        weight_decay = st.slider(
            "Weight Decay (0 = only first gene matters, 1 = all genes equal)",
            min_value=0.0, max_value=1.0, value=0.9, step=0.01
        )

    st.write(f"**Weighting Method Selected:** {weighting_method}")
    # Show marker genes as a list, not a table
    if isinstance(marker_genes, list):
        st.write("**Marker Genes:**")
        st.write(", ".join(marker_genes))
    else:
        st.write(f"**Marker Genes:** {marker_genes}")

    # You can then use `weighting_method` to perform the analysis
    if st.button("Run Analysis"):
        st.success("Analysis complete! Displaying results:")
        if database_choice in data_sources:
            data_path = data_sources[database_choice]
            try:
                st.write(f"Loading data from: {data_path}")
                user_marker_genes = marker_genes if isinstance(marker_genes, list) else [str(marker_genes)]
                # Run analysis based on selected method
                if weighting_method == "Weighted":
                    probs, missing_genes = compute_weighted_probabilities_from_model_url(
                        data_path, user_marker_genes, weight_decay=weight_decay, return_missing=True
                    )
                else:
                    probs, missing_genes = compute_probabilities_from_model_url(
                        data_path, user_marker_genes, return_missing=True
                    )
                # Display missing genes if any
                if missing_genes:
                    st.warning(f"Number of genes not found in model: {len(missing_genes)}")
                    st.write("Missing genes (first 10):", missing_genes[:10])
                # Display results
                st.subheader("Predicted Cell Type")
                top = max(probs, key=probs.get)
                st.write(f"**{top}**")
                st.subheader("Top Probabilities")
                top_n = 10
                prob_df = pd.DataFrame(
                    sorted(probs.items(), key=lambda x: -x[1])[:top_n],
                    columns=["Cell Type", "Probability"]
                )
                st.dataframe(prob_df)
            except FileNotFoundError:
                st.error(f"Data file not found for {database_choice}.")
            except Exception as e:
                st.error(f"Error running analysis: {e}")
        else:
            st.info("Please select a valid database above.")
else:
    st.warning("Please upload genes and select your analysis options on the Home page.")
