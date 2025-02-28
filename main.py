# main.py
import streamlit as st
import os
import pandas as pd
from noLLM_analysis import *


if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = None
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "background_context" not in st.session_state:
    st.session_state.background_context = ""
if "Gene_denominator" not in st.session_state:
    st.session_state.Gene_denominator = ""
if "custom_data" not in st.session_state:
   st.session_state.custom_data = ""
    


st.set_page_config(page_title="Home Page")

st.sidebar.success("Select a Page Above")


st.title("Welcome")
st.write("""
## About This App
This app aims to give a prediction on cell type based on user input.

## Cell Taxonomy
1. **Select an Algorithm**: Choose between "Inverse Weighting", "Exact Match", "Data Base".

    For "Inverse Weighting" or "Exact Match":
    1. **Select a Species**: Choose either "Homo sapiens" or "Mus musculus".
    2. **Select a Tissue Type**: Choose from a drop-down based on the selected species.
    3. **Enter Marker Genes**: Enter the marker genes observed in your cell.

    For "Data Base":
    1. **Select Data Base Options**: Choose from the pre-entered "Mouse Liver" or "Human Breast Cancer".
    2. If "Custom" is chosen, more information must be entered:
        1. **Select a Species**: Choose either "Homo sapiens" or "Mus musculus".
        2. **Select a Tissue Type**: Choose from a drop-down based on the selected species.
        3. **Enter Marker Genes**: Enter the database of genes you want to select from.
        4. **Enter Marker Genes**: Enter the marker genes observed in your cell.

The application uses pre-loaded datasets to match your input against known cell types and provide the best matches based on your selection.

## About the Dataset
This platform is built on a robust resource encompassing a vast array of single-cell data from human and mouse studies. Here's some highlights of Cell Taxonomy:

- **3,143 Cell Types**: Comprehensive classification of cell diversity, providing insights into distinct cellular roles and states.  
- **26,613 Cell Markers**: A curated database of molecular markers critical for identifying specific cell types.   
- **387 Tissues**: Coverage spans nearly all major tissue types, enabling tissue-specific analysis of cell types.  
- **257 Conditions**: Includes a wide range of physiological and pathological conditions for deeper biological understanding.  
- **146 Single-Cell RNA-seq Studies**: Powered by the latest advancements in scRNA-seq technology, ensuring high-resolution cellular profiling.

More information can be found at: https://ngdc.cncb.ac.cn/celltaxonomy/

## Methods
This website employs a straightforward yet flexible approach to cell type prediction and classification, leveraging matching algorithms enhanced with optional adjustments for under-researched areas. The methodology is as follows:

1. **Pure Matching**:  
   - Marker genes provided by the user are matched directly with known cell markers in our curated database.  
   - Matches are based on exact or partial overlap.  

2. **Inverse Log Scale Adjustment**:  
   - To account for under-researched areas, we apply an **inverse log scale** weighting.  
   - This approach reduces the dominance of highly represented entries in the dataset (e.g., commonly studied cell types or tissues) and boosts the significance of rarer matches.  
   - The goal is to ensure that results are not solely influenced by the popularity or frequency of entries in the database, enabling a more balanced and exploratory analysis.

3. **Google AI Matching**:  
   - When provided with a list of potential cell type options, our platform leverages **Google AI** to infer the most likely match.  
   - Using the entered marker genes and contextual tissue information, the AI analyzes the input to predict the cell type that best aligns with the given data.  
   - This method is especially useful when users have predefined options and need additional computational insights to refine their predictions.
""")

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
st.write("**To access pages 4 and 5, Hybrid and LLM OpenAI you will ned to enter an OpenAI API key**")
st.session_state.api_key = st.text_input("Enter your OpenAI API key", type="password", placeholder="sk-...")

if st.button("Save Selection"):
   st.success(f"Selected option: {st.session_state.background_context}, {st.session_state.species}, {st.session_state.tissue}, {st.session_state.marker_genes}")
   if (st.session_state.background_context == "Upload TSV File") and (st.session_state.Gene_denominator.size > 0):
      st.success("File uploaded successfully!")
       
