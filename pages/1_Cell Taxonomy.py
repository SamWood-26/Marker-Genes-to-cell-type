import streamlit as st
import os
import pandas as pd
from noLLM_analysis import *
#show more how it reaches the decision with stats and other stuff
#be able to upload file to run multiple
#have user upload API key
st.title("No LLM")


if "species" not in st.session_state:
    st.session_state.species = None
if "tissue" not in st.session_state:
    st.session_state.tissue = ["All"]
if "marker_genes" not in st.session_state:
    st.session_state.marker_genes = ""
if "background_context" not in st.session_state:
    st.session_state.background_context = ""
if "Gene_denominator" not in st.session_state:
    st.session_state.Gene_denominator = ""
if "custom_data" not in st.session_state:
   st.session_state.custom_data = ""

#Selecting type
# fit_option = st.radio(
#     "Select Option",
#     ("Inverse Weighting", "Exact Match", "Data Base"),
# )
fit_option = st.session_state.background_context

#Preset variables in some cases not all used
tissue_type = None
species = None
dataset = None
custom_genes = None

#Pre subsets the data
@st.cache_data
def get_data():
    df = load_data()
    df_human = df[df['Species'] == 'Homo sapiens']
    df_mouse = df[df['Species'] == 'Mus musculus']

    total_cells = df['Cell_standard'].nunique()
    human_cells = df_human['Cell_standard'].nunique()
    mouse_cells = df_mouse['Cell_standard'].nunique()
    return df, df_human, df_mouse, total_cells, human_cells, mouse_cells

df, df_human, df_mouse, total_cells, human_cells, mouse_cells = get_data()

#if loops that seperate inputs required base on different selections
if fit_option == "Base":
    #Species selection
    species = st.session_state.species

    if species == 'Homo sapiens':
        df_selected = df_human
        total_species_cells = human_cells
    else:
        df_selected = df_mouse
        total_species_cells = mouse_cells
    
    st.write(f"Total unique cell types before filtering: {total_cells}")
    st.write(f"After selecting {species}: {total_species_cells} unique cell types remain.")

    #Tissue type selection via searchable box
    tissue_options = get_all_tissues(df_selected, species)
    selected_tissues = st.session_state.tissue 
    tissue_type = selected_tissues if selected_tissues else None  #Assigns selected tissues
    # Filter dataset based on selected tissues
    if "All" not in selected_tissues:
        df_selected = df_selected[df_selected['Tissue_standard'].isin(selected_tissues)]
    
    # Count remaining unique cell types after filtering by tissue
    remaining_cells = df_selected['Cell_standard'].nunique()
    st.write(f"After selecting tissue type(s) {', '.join(selected_tissues)}: {remaining_cells} unique cell types remain.")


else:
    species = st.session_state.species
    tissue_type = st.session_state.tissue
    dataset = st.radio(
        "Select Dataset",
        ("Mouse Liver", "Human Breast Cancer", "Custom")
    )
    if fit_option == "Mouse Liver":
        file_path = get_file_path('feature.clean.MouseLiver1Slice1.tsv')
        #species = 'Mouse'
        #tissue_type = ['Liver']
    elif fit_option == "Human Breast Cancer":
        file_path = get_file_path('Xenium_FFPE_Human_Breast_Cancer_Rep1_panel.tsv')
        #species = 'Human'
        #tissue_type = ['Breast']

    if fit_option == "Custom Input":
        dataset = "Cutstom"
        species = st.session_state.species

        if species == 'Homo sapiens':
            tissue_options = get_all_tissues(df_human, 'Homo sapiens')
        else:
            tissue_options = get_all_tissues(df_mouse, 'Mus musculus')
        
        selected_tissues = st.session_state.tissue
        tissue_type = selected_tissues if selected_tissues else None  #Assigns selected tissues for custom cases

        custom_genes = st.session_state.marker_genes
    if fit_option == "Upload TSV File":
        species = st.session_state.species
        selected_tissues = st.session_state.tissue
        tissue_type = selected_tissues if selected_tissues else None
        custom_genes = st.session_state.marker_genes

#Marker genes input area
marker_genes_input = st.session_state.marker_genes

#Another if/else statements based on selections
if st.button("Submit"):
    marker_genes = marker_genes_input
    if fit_option == "Base":
        if species == 'Homo sapiens':
            df_selected = df_human
        else:
            df_selected = df_mouse

        if tissue_type == "All":
            tissue_type = None

        # Run both Inverse Weighting and Exact Match
        inverse_weighting_result = infer_top_cell_standards_weighted(df_selected, tissue_type, marker_genes)
        exact_match_result = infer_top_cell_standards(df_selected, tissue_type, marker_genes)

        # Display both results
        st.write(f"### Results for {species} with Tissue Type: {tissue_type} and Marker Genes: {marker_genes}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Inverse Weighting")
            st.write(inverse_weighting_result)

        with col2:
            st.subheader("Exact Match")
            st.write(exact_match_result)
    else:
        if dataset == "Custom":
            custom_genes_array = string_to_gene_array(st.session_state.custom_data)
            try:
                result = predict_cell_type_with_custom_genes(species, tissue_type, custom_genes_array, marker_genes)
                st.write(f"Gene Markers Considered: {', '.join(marker_genes)}")
                st.write(f"Species: {species}, Tissue Type: {', '.join(tissue_type)}")
                st.write("Predicted Cell Types:")
                st.write(result)
            except Exception as e:
                st.write(f"Error processing custom data: {e}")
        elif dataset == "TSV input":
            custom_genes_array = st.session_state.Gene_denominator
            try:
                result = predict_cell_type_with_custom_genes(species, tissue_type, custom_genes_array, marker_genes)
                st.write(f"Gene Markers Considered: {', '.join(marker_genes)}")
                st.write(f"Species: {species}, Tissue Type: {', '.join(tissue_type)}")
                st.write("Predicted Cell Types:")
                st.write(result)
            except Exception as e:
                st.write(f"Error processing custom data: {e}")
        else:
            gene_markers = load_gene_markers(file_path)

            matched_genes = set(marker_genes) & set(gene_markers)
            dropped_genes = set(marker_genes) - matched_genes

            result = predict_cell_type(species, tissue_type, list(matched_genes))

            st.write(f"Results based off marker genes considered: {', '.join(matched_genes)}")
            if dropped_genes:
                st.write(f"Marker genes dropped out: {', '.join(dropped_genes)}")
            st.write(f"Species: {species}, Tissue Type: {tissue_type}")
            st.write("Predicted Cell Types:")
            st.write(result)
